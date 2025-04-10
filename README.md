# 🧠 Crypto Trading Strategies for Freqtrade

Welcome to this repository of custom crypto trading strategies developed for the [Freqtrade](https://www.freqtrade.io/) open-source trading bot framework. This collection includes a variety of strategies such as **Grid Trading**, **Mean Reversion**, **Trend Following**, and **custom experimental models**.

本倉庫收錄咗一系列針對 [Freqtrade](https://www.freqtrade.io/) 自動交易機械人框架設計嘅策略，包括 **網格交易**、**均值回歸**、**趨勢跟隨** 同 **實驗性策略**，適合用於加密貨幣自動化交易。

---

## 📂 Strategy Categories | 策略分類

### 🔷 Grid-Based Strategies | 網格交易策略
- `GridV6_tmp7_wether`
- `GridV6_tmp7_wether_fixed`
- `Grid_Day_Trade_v1` ~ `v4`
- `StablecoinDynamicGrid`

### 🔁 Mean Reversion | 均值回歸策略
- `SUI_MeanReversion_Optimized_15m`
- `Notank_unbiased_no_freqai`
- `NostalgiaForSimplicity`

### 📈 Trend-Following / Momentum | 趨勢跟隨 / 動量策略
- `TrendFollow_MA_Adaptive`
- `VolBreak_RSI_Adaptive`

### 🧪 Experimental / AI / Custom | 實驗性 / AI / 自訂策略
- `GodStra_v2`, `v4`, `GodStraNew`
- `DevilStra`, `DS_Short`
- `NOTankAi_15`
- `checking`, `Testing`

---

## 🚀 Getting Started | 快速開始

1. Install [Freqtrade](https://www.freqtrade.io/en/latest/installation/)
2. Copy any of the strategy files into your `user_data/strategies/` folder
3. Run backtesting or live trading:

```bash
freqtrade backtesting -s GridV6_tmp7_wether_fixed
freqtrade trade --strategy GridV6_tmp7_wether_fixed
