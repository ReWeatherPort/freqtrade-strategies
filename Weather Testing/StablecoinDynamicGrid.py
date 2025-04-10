from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np

class StablecoinDynamicGrid(IStrategy):
    # 基本參數
    timeframe = "1m"  # 1分鐘時間框架
    minimal_roi = {"0": 0.0005}  # 0.05% 止盈
    stoploss = -0.05  # 0.5% 止損
    max_open_trades = 20  # 最多20個網格單

    # 網格參數
    grid_levels = 5  # 每對5層網格（減少層數，增加觸發機會）
    grid_spacing = 0.0001  # 固定網格間距 0.01%
    base_risk_per_trade = 0.01  # 每筆交易風險1%

    # 交易對
    pair_whitelist = ["FDUSD/USDT", "USDC/USDT", "FDUSD/USDC"]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 無需複雜指標，直接用價格
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 逐行檢查價格是否觸及買入網格
        dataframe['enter_long'] = 0
        current_price = dataframe['close'].iloc[-1]

        for i in range(self.grid_levels):
            buy_price = current_price - (i + 1) * self.grid_spacing
            # 逐行判斷
            dataframe.loc[dataframe['close'] <= buy_price, 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 逐行檢查價格是否觸及賣出網格
        dataframe['exit_long'] = 0
        current_price = dataframe['close'].iloc[-1]

        for i in range(self.grid_levels):
            sell_price = current_price + (i + 1) * self.grid_spacing
            dataframe.loc[dataframe['close'] >= sell_price, 'exit_long'] = 1

        return dataframe

    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_leverage, **kwargs):
        account_balance = self.wallets.get_available_stake_amount()
        stake = account_balance * self.base_risk_per_trade
        return min(stake, account_balance / (self.grid_levels * len(self.pair_whitelist)))

    def leverage(self, pair: str, current_time, current_rate, proposed_leverage, max_leverage, side, **kwargs):
        return 1.0  # 無槓桿