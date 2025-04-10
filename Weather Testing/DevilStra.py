import numpy as np
import pandas as pd
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import random
from freqtrade.strategy import CategoricalParameter, IStrategy
from pandas import DataFrame

PAIR_LIST_LENGHT = 1
TREND_CHECK_CANDLES = 4
PAIN_RANGE = 1000
WINDOW_SIZE = 50

SPELLS = {
    "Zi": {
        "buy_params": {
            "buy_crossed_indicator0": "BOP-4",
            "buy_crossed_indicator1": "MACD-0-50",
            "buy_crossed_indicator2": "DEMA-52",
            "buy_indicator0": "MINUS_DI-50",
            "buy_indicator1": "HT_TRENDMODE-50",
            "buy_indicator2": "CORREL-128",
            "buy_operator0": "/>R",
            "buy_operator1": "CA",
            "buy_operator2": "CDT",
            "buy_real_num0": 0.1763,
            "buy_real_num1": 0.6891,
            "buy_real_num2": 0.0509,
        },
        "sell_params": {
            "sell_crossed_indicator0": "WCLPRICE-52",
            "sell_crossed_indicator1": "AROONOSC-15",
            "sell_crossed_indicator2": "CDLRISEFALL3METHODS-52",
            "sell_indicator0": "COS-50",
            "sell_indicator1": "CDLCLOSINGMARUBOZU-30",
            "sell_indicator2": "CDL2CROWS-130",
            "sell_operator0": "DT",
            "sell_operator1": ">R",
            "sell_operator2": "/>R",
            "sell_real_num0": 0.0678,
            "sell_real_num1": 0.8698,
            "sell_real_num2": 0.3917,
        }
    },
}

def spell_finder(index, space):
    return SPELLS[index][space + "_params"]

class DevilStra(IStrategy):
    INTERFACE_VERSION: int = 3

    buy_params = {"buy_spell": "Zi"}
    sell_params = {"sell_spell": "La"}

    minimal_roi = {
        "0": 0.20,   # 即刻20%
        "30": 0.10,  # 30分鐘10%
        "60": 0.05,  # 1小時5%
        "120": 0     # 2小時後無最低
    }
    stoploss = -0.05  # -5%
    trailing_stop = True
    trailing_stop_positive = 0.01  # 跌1%止盈
    trailing_stop_positive_offset = 0.03  # 3%利潤啟動
    timeframe = '1h'

    spell_pot = [
        ",".join(tuple(random.choices(list(SPELLS.keys()), k=PAIR_LIST_LENGHT)))
        for i in range(PAIN_RANGE)
    ]

    buy_spell = CategoricalParameter(spell_pot, default=spell_pot[0], space='buy')
    sell_spell = CategoricalParameter(spell_pot, default=spell_pot[0], space='sell')

    def normalize(self, df, window=WINDOW_SIZE):
        if not isinstance(df, pd.Series):
            df = pd.Series(df, index=self.dataframe.index if hasattr(self, 'dataframe') else None)
        rolling_min = df.rolling(window=window, min_periods=1).min()
        rolling_max = df.rolling(window=window, min_periods=1).max()
        return (df - rolling_min) / (rolling_max - rolling_min + 1e-10)

    def gene_calculator(self, dataframe, indicator):
        if 'CDL' in indicator:
            splited_indicator = indicator.split('-')
            splited_indicator[1] = "0"
            indicator = "-".join(splited_indicator)

        gene = indicator.split("-")
        gene_name = gene[0]
        gene_len = len(gene)

        if indicator in dataframe.columns:
            return dataframe[indicator]

        result = None
        if gene_len == 1:
            result = pd.Series(getattr(ta, gene_name)(dataframe).shift(1), index=dataframe.index)
            return self.normalize(result)
        elif gene_len == 2:
            gene_timeperiod = int(gene[1])
            result = pd.Series(getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).shift(1), index=dataframe.index)
            return self.normalize(result)
        elif gene_len == 3:
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            result = pd.Series(getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index].shift(1), index=dataframe.index)
            return self.normalize(result)
        elif gene_len == 4:
            gene_timeperiod = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).shift(1)
            sma_result = pd.Series(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES), index=dataframe.index)
            return self.normalize(sma_result)
        elif gene_len == 5:
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index].shift(1)
            sma_result = pd.Series(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES), index=dataframe.index)
            return self.normalize(sma_result)

    def condition_generator(self, dataframe, operator, indicator, crossed_indicator, real_num):
        conditions = [(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2)]

        dataframe[indicator] = self.gene_calculator(dataframe, indicator)
        dataframe[crossed_indicator] = self.gene_calculator(dataframe, crossed_indicator)

        indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
        if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
            dataframe[indicator_trend_sma] = self.gene_calculator(dataframe, indicator_trend_sma)

        if operator == ">":
            conditions.append(dataframe[indicator] > dataframe[crossed_indicator])
        elif operator == "=":
            conditions.append(np.isclose(dataframe[indicator], dataframe[crossed_indicator]))
        elif operator == "<":
            conditions.append(dataframe[indicator] < dataframe[crossed_indicator])
        elif operator == "C":
            conditions.append(
                (qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])) |
                (qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator]))
            )
        elif operator == "CA":
            conditions.append(qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator]))
        elif operator == "CB":
            conditions.append(qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator]))
        elif operator == ">R":
            conditions.append(dataframe[indicator] > real_num)
        elif operator == "=R":
            conditions.append(np.isclose(dataframe[indicator], real_num))
        elif operator == "<R":
            conditions.append(dataframe[indicator] < real_num)
        elif operator == "/>R":
            conditions.append(dataframe[indicator].div(dataframe[crossed_indicator]) > real_num)
        elif operator == "/=R":
            conditions.append(np.isclose(dataframe[indicator].div(dataframe[crossed_indicator]), real_num))
        elif operator == "/<R":
            conditions.append(dataframe[indicator].div(dataframe[crossed_indicator]) < real_num)
        elif operator == "UT":
            conditions.append(dataframe[indicator] > dataframe[indicator_trend_sma])
        elif operator == "DT":
            conditions.append(dataframe[indicator] < dataframe[indicator_trend_sma])
        elif operator == "OT":
            conditions.append(np.isclose(dataframe[indicator], dataframe[indicator_trend_sma]))
        elif operator == "CUT":
            conditions.append(
                (qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma])) &
                (dataframe[indicator] > dataframe[indicator_trend_sma])
            )
        elif operator == "CDT":
            conditions.append(
                (qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma])) &
                (dataframe[indicator] < dataframe[indicator_trend_sma])
            )
        elif operator == "COT":
            conditions.append(
                ((qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma])) |
                 (qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma]))) &
                (np.isclose(dataframe[indicator], dataframe[indicator_trend_sma]))
            )

        return reduce(lambda x, y: x & y, conditions), dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1)]
        dataframe['sma_short'] = pd.Series(ta.SMA(dataframe['close'], timeperiod=5), index=dataframe.index).shift(1)
        dataframe['sma_long'] = pd.Series(ta.SMA(dataframe['close'], timeperiod=20), index=dataframe.index).shift(1)
        conditions.append(dataframe['sma_short'] > dataframe['sma_long'])
        
        if conditions:
            combined_condition = reduce(lambda x, y: x & y, conditions)
            print(f"Buy triggers for {metadata['pair']}: {combined_condition.sum()}")
            dataframe.loc[combined_condition, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1)]
        dataframe['sma_short'] = pd.Series(ta.SMA(dataframe['close'], timeperiod=5), index=dataframe.index).shift(1)
        dataframe['sma_long'] = pd.Series(ta.SMA(dataframe['close'], timeperiod=20), index=dataframe.index).shift(1)
        conditions.append(dataframe['sma_short'] < dataframe['sma_long'])
        
        if conditions:
            combined_condition = reduce(lambda x, y: x & y, conditions)
            print(f"Sell triggers for {metadata['pair']}: {combined_condition.sum()}")
            dataframe.loc[combined_condition, 'exit_long'] = 1
        return dataframe