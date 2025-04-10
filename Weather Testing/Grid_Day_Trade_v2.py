# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from freqtrade.strategy.informative_decorator import informative
# --------------------------------

import array as arr
import pandas as pd
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
import warnings
import time


class Grid_Day_Trade_v2(IStrategy):
    """
    Grid Day-Trade V1 with a simple RSI filter check before triggering a new grid buy level.
    This is just an example demonstration of adding a small filter to enhance profit potential.
    """

    INTERFACE_VERSION = 2

    # Index reference for custom_info
    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # Grid Settings
    grid_up_spacing_pct = 1.4
    grid_down_spacing_pct = 3.5
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75

    debug = True

    # DCA config
    position_adjustment_enable = True 
    max_dca_orders = 5
    dca_scale = 1.25
    max_dca_multiplier = (1 - pow(dca_scale, (max_dca_orders + 1))) / (1 - dca_scale)

    # ROI
    minimal_roi = {
        "0": 0.05, 
        "20": 0.03,
        "40": 0.02,
        "60": 0
    }

    # Stoploss
    stoploss = -0.25

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Timeframe
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 50  # need a bit more for RSI

    plot_config = {
        'main_plot': {
            "grid_up": {
                'grid_up': {'color': 'green'}            
            },
            "grid_down": {
                'grid_down': {'color': 'red'}            
            },
        },
        'subplots': {
            "bot_state": {
                'bot_state': {'color': 'yellow'}
            },
            "rsi": {
                'rsi': {'color': 'blue'}
            }
        }
    }

    custom_info = {}
    
    sell_params = {
        "grid_up_spacing_pct": 1.4,
        "grid_down_spacing_pct": 3.5,
    }

    grid_down_spacing_pct = DecimalParameter(
        3.0, 5.0, default=3.5, decimals=1, load=True, space='sell', optimize=True
    )
    grid_up_spacing_pct = DecimalParameter(
        1.2, 3.0, default=1.4, decimals=1, load=True, space='sell', optimize=True
    )

    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Core grid logic with an RSI filter before buy triggers.
        """
        start_time = time.time()
        pair = metadata['pair']

        if pair not in self.custom_info:
            self.custom_info[pair] = ['', 0.0, 0, '', self.live_candles, 0, 0]

        last_row = len(dataframe) - 1
        init_count = self.custom_info[pair][self.INIT_COUNT]

        # Initialize
        if (self.dp.runmode.value in ('live', 'dry_run')):
            if self.custom_info[pair][self.LIVE_DATE] == '':
                init_count = 0
                row = last_row
                bot_state = 0
                grid = dataframe['close'].iloc[row]
                avg_price = grid
                units = 0
            else:
                live_date_candle = dataframe.loc[dataframe['date'] == self.custom_info[pair][self.LIVE_DATE]]
                if len(live_date_candle) > 0:
                    row = live_date_candle.index[0]
                    bot_state = self.custom_info[pair][self.BOT_STATE]
                    grid = self.custom_info[pair][self.GRID]
                    avg_price = self.custom_info[pair][self.AVG_PRICE]
                    units = self.custom_info[pair][self.UNITS]
                else:
                    init_count = 0
                    row = last_row
                    bot_state = 0
                    grid = dataframe['close'].iloc[row]
                    avg_price = grid
                    units = 0
        else:
            row = 0
            bot_state = 0
            grid = dataframe['close'].iloc[0]
            avg_price = grid
            units = 0

        grid_up_shift = grid * ((bot_state * self.grid_shift_pct) / 100)
        grid_up = avg_price * (1 + (self.grid_up_spacing_pct.value / 100)) + grid_up_shift
        grid_down = grid * (1 - (self.grid_down_spacing_pct.value / 100))
        grid_trigger_up = grid * (1 + (self.grid_trigger_pct / 100))
        grid_trigger_down = grid * (1 - (self.grid_trigger_pct / 100))

        Buy_1 = np.zeros((last_row + 1), dtype=int)
        Buy_2 = np.zeros((last_row + 1), dtype=int)
        Sell_1 = np.zeros((last_row + 1), dtype=int)
        Grid_up = np.zeros((last_row + 1))
        Grid_up[:] = np.NaN
        Grid_down = np.zeros((last_row + 1))
        Grid_down[:] = np.NaN
        Bot_state = np.zeros((last_row + 1))
        Bot_state[:] = np.NaN

        if self.debug:
            Grid = np.zeros((last_row + 1))
            Avg_price = np.zeros((last_row + 1))
            Grid[:] = np.NaN
            Avg_price[:] = np.NaN

        Close = dataframe.loc[:, 'close'].values
        Rsi = dataframe.loc[:, 'rsi'].values  # We'll check RSI from the populated indicators

        while row <= last_row:
            close = Close[row]
            rsi_val = Rsi[row] if not np.isnan(Rsi[row]) else 50  # fallback if no RSI

            new_bot_state = bot_state
            new_grid = grid
            new_units = units
            new_avg_price = avg_price

            if (self.dp.runmode.value in ('live', 'dry_run')):
                if row == (last_row - init_count):
                    self.custom_info[pair][self.BOT_STATE] = bot_state
                    self.custom_info[pair][self.GRID] = grid
                    self.custom_info[pair][self.LIVE_DATE] = dataframe['date'].iloc[row]
                    self.custom_info[pair][self.AVG_PRICE] = avg_price
                    self.custom_info[pair][self.UNITS] = units

            # RSI filter applied where we only buy if RSI < 60
            rsi_filter_pass = (rsi_val < 60)

            if bot_state == 0:
                if close > grid_trigger_up:
                    new_grid = close
                    new_units = 0
                    new_avg_price = close
                if close <= grid_trigger_down and rsi_filter_pass:
                    new_bot_state = 1
                    Buy_1[row] = 1
                    new_units = 1 / close
                    new_avg_price = 1 / new_units
                    new_grid = close

            if (bot_state >= 1) and (bot_state <= self.max_dca_orders):
                if close > grid_up:
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0
                    new_avg_price = close
                    new_grid = close

                if close <= grid_down and rsi_filter_pass:
                    new_bot_state = bot_state + 1
                    Buy_2[row] = 1
                    new_amount = pow(self.dca_scale, bot_state)
                    new_total_amount = (1 - pow(self.dca_scale, (bot_state + 1))) / (1 - self.dca_scale)
                    new_units = units + (new_amount / close)
                    new_avg_price = new_total_amount / new_units
                    new_grid = close

            if bot_state == (self.max_dca_orders + 1):
                if (close > grid_up) or (close <= grid_down):
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0
                    new_avg_price = close
                    new_grid = close

            bot_state = new_bot_state
            grid = new_grid
            units = new_units
            avg_price = new_avg_price

            grid_up_shift = grid * ((bot_state * self.grid_shift_pct) / 100)
            grid_up = avg_price * (1 + (self.grid_up_spacing_pct.value / 100)) + grid_up_shift
            grid_down = grid * (1 - (self.grid_down_spacing_pct.value / 100))
            grid_trigger_up = grid * (1 + (self.grid_trigger_pct / 100))
            grid_trigger_down = grid * (1 - (self.grid_trigger_pct / 100))

            if bot_state == 0:
                Grid_up[row] = grid_trigger_up
                Grid_down[row] = grid_trigger_down
            else:
                Grid_up[row] = grid_up
                Grid_down[row] = grid_down

            Bot_state[row] = bot_state

            if self.debug:
                Avg_price[row] = avg_price
                Grid[row] = grid

            row += 1

        if init_count < self.live_candles:
            init_count += 1
        self.custom_info[pair][self.INIT_COUNT] = init_count

        df_buy_1 = pd.DataFrame(Buy_1, columns=['buy_1'])
        df_buy_2 = pd.DataFrame(Buy_2, columns=['buy_2'])
        df_sell_1 = pd.DataFrame(Sell_1, columns=['sell_1'])
        df_grid_up = pd.DataFrame(Grid_up, columns=['grid_up'])
        df_grid_down = pd.DataFrame(Grid_down, columns=['grid_down'])
        df_bot_state = pd.DataFrame(Bot_state, columns=['bot_state'])

        dataframe = pd.concat([dataframe, df_buy_1, df_buy_2, df_sell_1, df_grid_up, df_grid_down, df_bot_state], axis=1)

        if self.debug:
            df_grid = pd.DataFrame(Grid, columns=['grid'])
            df_avg_price = pd.DataFrame(Avg_price, columns=['avg_price'])
            dataframe = pd.concat([dataframe, df_grid, df_avg_price], axis=1)

        end_time = time.time()
        # print("total time taken", end_time - start_time)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        We add RSI to the dataframe for use in the grid logic.
        """
        # Compute RSI with period 14 or any you prefer
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        if (self.dp.runmode.value in ('live', 'dry_run')):
            dataframe = self.calculate_state(dataframe, metadata)
        return dataframe

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float, max_stake: float, **kwargs
    ) -> float:
        fixed_stake = 200.0
        if self.config['stake_amount'] == 'unlimited':
            return (self.wallets.get_total_stake_amount() / self.max_dca_multiplier) * self.stake_to_wallet_ratio
        else:
            stake_for_trade = min(max_stake, max(fixed_stake, min_stake))
            return stake_for_trade

    def adjust_trade_position(
        self, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, min_stake: float, max_stake: float, **kwargs
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if len(dataframe) < 1:
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        if self.custom_info[trade.pair][self.DATESTAMP] != last_candle['date']:
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']
    
            if last_candle['buy_2'] == 1:
                filled_buys = trade.select_filled_orders('buy')
                count_of_buys = len(filled_buys)
                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        stake_amount = filled_buys[0].cost
                        stake_amount = stake_amount * pow(self.dca_scale, (last_candle['bot_state'] - 1))
                        return stake_amount
                    except Exception as exception:
                        return None
                return None
        return None

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not (self.dp.runmode.value in ('live', 'dry_run')):
            dataframe = self.calculate_state(dataframe, metadata)

        dataframe.loc[:, 'buy'] = 0
        dataframe.loc[
            (dataframe['buy_1'] == 1),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0
        dataframe.loc[
            (dataframe['sell_1'] == 1),
            'sell'
        ] = 1
        return dataframe