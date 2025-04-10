# --- Required Imports ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

class Testing(IStrategy):
    """
    Testing strategy that buys and sells on every candle.
    """

    # Minimal ROI (Return on Investment) table
    minimal_roi = {
        "0": 0  # 1% profit target
    }

    # Stoploss
    stoploss = -0.99  # Effectively disabled

    # Optimal timeframe for the strategy
    timeframe = "5m"  # You can adjust this if needed

    # Disable trailing stop
    trailing_stop = False

    # No startup candle count required
    startup_candle_count = 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        This strategy does not use any indicators.
        """
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generates a buy signal for every candle.
        """
        dataframe["buy"] = 1  # Always buy
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generates a sell signal for every candle.
        """
        dataframe["sell"] = 1  # Always sell
        return dataframe