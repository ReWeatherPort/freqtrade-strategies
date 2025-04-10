from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta

class SUI_MeanReversion_Optimized_15m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    minimal_roi = {"0": 0.03}  # 3%止盈
    stoploss = -0.05  # 固定止損-5%
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%利潤啟動追蹤
    trailing_stop_positive_offset = 0.01  # 1%啟動
    trailing_only_offset_is_reached = True
    startup_candle_count: int = 21
    max_open_trades = 2

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]
        wallet = self.wallets.get_available_stake_amount()
        stake = (wallet * 0.01) / (atr / current_rate)  # 1%風險，約5 USDT
        max_stake = 50  # 每筆上限50 USDT
        return min(stake, max_stake, proposed_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA快慢線
        dataframe["ema_fast"] = ta.ema(dataframe["close"], length=9)
        dataframe["ema_slow"] = ta.ema(dataframe["close"], length=21)
        
        # RSI（7期）
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=7)
        
        # ATR（動態止損）
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"]) &  # 快EMA上穿慢EMA
                (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1)) &  # 確認交叉
                (dataframe["rsi"] > 50) &  # RSI動量上升
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])  # 快EMA跌破慢EMA
            ),
            "exit_long"] = 1
        return dataframe

    def custom_stop_loss(self, pair: str, trade, current_time, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]
        return -atr / current_rate  # 動態止損：-1倍ATR