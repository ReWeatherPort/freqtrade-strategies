import os
import importlib.util
import logging
# ★ Revert to absolute import to avoid "attempted relative import" error
from signals.Signal import Signal

from pandas import DataFrame
import pandas as pd
from freqtrade.strategy import IStrategy, informative
import Indicators as ind


class NostalgiaForSimplicity(IStrategy):
    """
    Strategy managing multiple signals with priority
    """

    INTERFACE_VERSION = 3
    minimal_roi = {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04}
    stoploss = -0.05
    timeframe = "5m"
    startup_candle_count = 100

    def __init__(self, config: dict) -> None:
        self.log = logging.getLogger(__name__)
        super().__init__(config)
        self.signals = self.load_signals()


    def load_signals(self):
        """
        Dynamically load and sort signals by priority.
        """
        signals = []
        signal_dir = os.path.join(os.path.dirname(__file__), "signals")

        for file in os.listdir(signal_dir):
            if file.endswith(".py") and file != "__init__.py":
                module_name = file[:-3]
                file_path = os.path.join(signal_dir, file)

                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Search for subclasses of Signal
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Signal) and attr is not Signal:
                        signals.append(attr())

        # Sort signals by priority
        return sorted(signals, key=lambda signal: signal.get_priority())


    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'EMA_12': {'color': 'red'},
            'EMA_26': {'color': 'blue'},
            'EMA_50': {'color': 'green'},
            'EMA_200': {'color': 'yellow'},
        }
        plot_config['subplots'] = {
            "downtrend": {
                'is_downtrend': {'color': 'red'},
            },
            "downtrend_signals": {
                'downtrend_signals': { 'color': 'blue'},
            },
            "ADX": {
                'ADX': { 'color': 'green'},
            }
        }

        return plot_config

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = ind.add_indicators(df)
        return df
    

    @informative('15m')
    def populate_indicators_15m(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = ind.calculate_aroon(df, length=14)
        return df


    @informative('1h')
    def populate_indicators_1h(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = ind.calculate_willr(df, length=84)
        df = ind.calculate_stochrsi(df)
        df = ind.calculate_bbands(df)
        return df


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on plugin logic, with custom tags for each signal,
        while avoiding entries during a downtrend.
        """
        if dataframe.empty or dataframe.iloc[-1].isna().any():
            self.log.warning(f"Skipping populate_entry_trend for {metadata.get('pair')} due to missing candle data.")
            return dataframe

        if "enter_long" not in dataframe.columns:
            dataframe["enter_long"] = 0
        if "enter_tag" not in dataframe.columns:
            dataframe["enter_tag"] = None

        pair = metadata.get("pair", "Unknown")

        for signal in self.signals:
            if not signal.enabled:
                continue

            self.log.debug(f"Checking entry signals from plugin {signal.get_signal_tag()}.")

            def lazy_evaluation():
                entry_signal = signal.entry_signal(dataframe, metadata)
                if not isinstance(entry_signal, pd.Series) or entry_signal.dtype != bool:
                    raise TypeError(f"Signal {signal.get_signal_tag()} returned an invalid entry signal type.")
                return entry_signal

            new_signals = (dataframe["enter_long"] == 0)

            if new_signals.any():
                entry_signal = lazy_evaluation()
                final_signals = new_signals & entry_signal

                dataframe.loc[final_signals, "enter_long"] = 1
                dataframe.loc[final_signals, "enter_tag"] = f"enter_{signal.get_signal_tag()}"

                signal_count = final_signals.sum()
                if signal_count > 0:
                    self.log.info(f"Signal {signal.get_signal_tag()} generated {signal_count} entry signal(s) for pair {pair}.")

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on plugin logic, with custom tags for each signal.
        """
        if dataframe.empty or dataframe.iloc[-1].isna().any():
            self.log.warning(f"Skipping populate_exit_trend for {metadata.get('pair')} due to missing candle data.")
            return dataframe

        if "exit_long" not in dataframe.columns:
            dataframe["exit_long"] = 0