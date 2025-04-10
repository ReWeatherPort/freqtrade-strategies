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


class Grid_Day_Trade_v1(IStrategy):
    """
    Enhanced version of the original GridV6_tmp7_wether strategy.
    - We keep its main logic and timeframe = '5m' to preserve the original win rate.
    - We make minor tweaks on grid spacing and enable a trailing stop to lock in profits.
    - Everything else remains mostly the same, including DCA logic, to maintain the original strategy's behavior.
    """

    INTERFACE_VERSION = 2

    # Index references for custom_info
    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # Minor adjustments to grid spacing
    grid_up_spacing_pct = 1.5        # Slightly changed from 1.6 to 1.5
    grid_down_spacing_pct = 3.8      # Slightly changed from 4.0 to 3.8
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75
    
    debug = True

    # DCA config
    position_adjustment_enable = True 

    # defines the number of states as well (= max_dca_orders + 1)
    max_dca_orders = 7      # same as the original
    dca_scale = 1.3         # same as the original
    max_dca_multiplier = (1 - pow(dca_scale,(max_dca_orders + 1)))/(1 - dca_scale)

    # ROI table:
    # Keep the original huge ROI to try to hold for bigger potential
    minimal_roi = {
        "0": 100.0
    }

    # Stoploss:
    stoploss = -0.99  # same as the original

    # Enable trailing stop to lock in some profit
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 0

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
        }
    }

    # storage dict for custom info
    custom_info = { }
    
    sell_params = {
        "grid_up_spacing_pct": 1.5,
        "grid_down_spacing_pct": 3.8,
    }

    # We do not mark these as "optimize" because we mainly want to keep them stable
    grid_down_spacing_pct = DecimalParameter(3.0, 5.0, default=3.8, decimals=1, load=True, space='sell', optimize=False)
    grid_up_spacing_pct = DecimalParameter(1.4, 3.0, default=1.5, decimals=1, load=True, space='sell', optimize=False)

    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Main logic to update the bot state (grid, average price, DCA triggers, etc.) for each candle.
        We keep this mostly identical to the original with the new spacing values.
        """
        start_time = time.time()

        pair = metadata['pair']
        if not pair in self.custom_info:
            # Create empty entry for this pair {DATESTAMP, GRID, BOT_STATE, LIVE_DATE, INIT_COUNT, AVG_PRICE, UNITS}
            self.custom_info[pair] = ['', 0.0, 0, '', self.live_candles, 0, 0] 
    
        last_row = len(dataframe) - 1
        init_count = self.custom_info[pair][self.INIT_COUNT]
        
        # Decide how to initialize values based on run mode
        if (self.dp.runmode.value in ('live', 'dry_run')):
            # live or dry, need to initialize bot state and grid from saved values
            if self.custom_info[pair][self.LIVE_DATE] == '':
                # first candle of live/dry
                init_count = 0
                row = last_row
                bot_state = 0
                grid = dataframe['close'].iloc[row]
                avg_price = grid
                units = 0
            else:
                # subsequent candles
                live_date_candle = dataframe.loc[dataframe['date'] == self.custom_info[pair][self.LIVE_DATE]]
                if len(live_date_candle) > 0:
                    # found live start candle
                    row = live_date_candle.index[0]
                    bot_state = self.custom_info[pair][self.BOT_STATE]
                    grid = self.custom_info[pair][self.GRID]
                    avg_price = self.custom_info[pair][self.AVG_PRICE]
                    units = self.custom_info[pair][self.UNITS]
                else:
                    # no live start candle found, default
                    init_count = 0
                    row = last_row
                    bot_state = 0
                    grid = dataframe['close'].iloc[row]
                    avg_price = grid
                    units = 0
        else:
            # backtesting or hyperopt
            row = 0
            bot_state = 0
            grid = dataframe['close'].iloc[0]
            avg_price = grid
            units = 0

        # calculate grid thresholds
        grid_up_shift = grid * ((bot_state * self.grid_shift_pct) / 100)
        grid_up = avg_price * (1 +(self.grid_up_spacing_pct.value/100)) + grid_up_shift
        grid_down = grid * (1 - (self.grid_down_spacing_pct.value/100))
        grid_trigger_up = grid * (1 + (self.grid_trigger_pct/100))
        grid_trigger_down = grid * (1 - (self.grid_trigger_pct/100))
            
        # define np arrays
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
        
        # iterate through dataframe
        while (row <= last_row):
            close = Close[row]

            new_bot_state = bot_state
            new_grid = grid
            new_units = units
            new_avg_price = avg_price
            
            if (self.dp.runmode.value in ('live', 'dry_run')):
                if (row == (last_row - init_count)):
                    # live or dry, save the bot state and live candle date
                    self.custom_info[pair][self.BOT_STATE] = bot_state
                    self.custom_info[pair][self.GRID] = grid
                    self.custom_info[pair][self.LIVE_DATE] = dataframe['date'].iloc[row]
                    self.custom_info[pair][self.AVG_PRICE] = avg_price
                    self.custom_info[pair][self.UNITS] = units

            if (bot_state == 0):
                if (close > grid_trigger_up):
                    new_grid = close
                    new_units = 0
                    new_avg_price = close
                
                if (close <= grid_trigger_down):
                    new_bot_state = 1
                    Buy_1[row] = 1
                    new_units = 1 / close
                    new_avg_price = 1 / new_units
                    new_grid = close
                    
            if ((bot_state >= 1) and (bot_state <= self.max_dca_orders)):
                if (close > grid_up):
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0
                    new_avg_price = close
                    new_grid = close

                if (close <= grid_down):
                    new_bot_state = bot_state + 1
                    Buy_2[row] = 1
                    new_amount = pow(self.dca_scale, (bot_state))
                    new_total_amount = (1 - pow(self.dca_scale,(bot_state + 1)))/(1 - self.dca_scale)
                    new_units = units + (new_amount / close)
                    new_avg_price = new_total_amount / new_units
                    new_grid = close

            if (bot_state == (self.max_dca_orders + 1)):
                if ((close > grid_up) or (close <= grid_down)):
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
            grid_up = avg_price * (1 +(self.grid_up_spacing_pct.value/100)) + grid_up_shift
            grid_down = grid * (1 - (self.grid_down_spacing_pct.value/100))
            grid_trigger_up = grid * (1 + (self.grid_trigger_pct/100))
            grid_trigger_down = grid * (1 - (self.grid_trigger_pct/100))
   
            if(bot_state == 0):
                Grid_up[row] = grid_trigger_up
                Grid_down[row] = grid_trigger_down
            else:
                Grid_up[row] = grid_up
                Grid_down[row] = grid_down
            
            Bot_state[row] = bot_state

            if(self.debug):
                Avg_price[row] = avg_price
                Grid[row] = grid

            row += 1

        if(init_count < self.live_candles):
            init_count += 1
        self.custom_info[pair][self.INIT_COUNT] = init_count

        df_buy_1 = pd.DataFrame(Buy_1, columns=['buy_1'])
        df_buy_2 = pd.DataFrame(Buy_2, columns=['buy_2'])
        df_sell_1 = pd.DataFrame(Sell_1, columns=['sell_1'])
        df_grid_up = pd.DataFrame(Grid_up, columns=['grid_up'])
        df_grid_down = pd.DataFrame(Grid_down, columns=['grid_down'])
        df_bot_state = pd.DataFrame(Bot_state, columns=['bot_state'])

        dataframe = pd.concat([dataframe, df_buy_1, df_buy_2, df_sell_1, df_grid_up, df_grid_down, df_bot_state], axis=1)

        if(self.debug == True):
            df_grid = pd.DataFrame(Grid, columns=['grid'])
            df_avg_price = pd.DataFrame(Avg_price, columns=['avg_price'])
            dataframe = pd.concat([dataframe, df_grid, df_avg_price], axis=1)
        
        end_time = time.time()
        # print("total time taken this loop: ", end_time - start_time)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators needed by this grid logic.
        For live/dry_run, we call calculate_state once here to update the grid status.
        """
        if (self.dp.runmode.value in ('live', 'dry_run')):
            dataframe = self.calculate_state(dataframe, metadata)
        return dataframe

    
    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float, max_stake: float,
        **kwargs
    ) -> float:
        """
        Keep the original logic that uses wallet ratio for stake if stake_amount is unlimited.
        Otherwise scale the proposed stake by stake_to_wallet_ratio.
        """
        if (self.config['stake_amount'] == 'unlimited'):
            return (self.wallets.get_total_stake_amount() / self.max_dca_multiplier) * self.stake_to_wallet_ratio
        else:
            return (proposed_stake / self.max_dca_multiplier) * self.stake_to_wallet_ratio
        
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        DCA logic:
        If the last candle triggers buy_2, we add more to the position using a Safety Order approach.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if(len(dataframe) < 1):
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        if(self.custom_info[trade.pair][self.DATESTAMP] != last_candle['date']):
            # new candle
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']
    
            if(last_candle['buy_2'] == 1):
                filled_buys = trade.select_filled_orders('buy')
                count_of_buys = len(filled_buys)
                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        # This returns first order stake size
                        stake_amount = filled_buys[0].cost
                        # This then calculates current safety order size
                        stake_amount = stake_amount * pow(self.dca_scale, (last_candle['bot_state'] - 1))
                        return stake_amount
                    except Exception as exception:
                        return None
                return None
        return None


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        For backtest/hyperopt, we recalc state each loop if runmode != live/dry_run.
        Then set buy = 1 if buy_1 is triggered.
        """
        if not(self.dp.runmode.value in ('live', 'dry_run')):
            dataframe = self.calculate_state(dataframe, metadata)
        dataframe.loc[:, 'buy'] = 0
        dataframe.loc[
            (dataframe['buy_1'] == 1),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        If sell_1 is triggered, set sell=1.
        """
        dataframe.loc[:, 'sell'] = 0
        dataframe.loc[
            (dataframe['sell_1'] == 1),
            'sell'
        ] = 1
        return dataframe