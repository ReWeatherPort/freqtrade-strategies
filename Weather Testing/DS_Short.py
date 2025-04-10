# DS_Short Strategy
# 𝔇𝔖 𝔖𝔥𝔬𝔯𝔱 𝔉𝔲𝔫𝔠𝔱𝔦𝔬𝔫𝔞𝔩 𝔖𝔱𝔯𝔞𝔱𝔢𝔤𝔶
# 𝔇𝔦𝔰𝔱𝔦𝔫𝔠𝔱𝔩𝔶 𝔦𝔫𝔱𝔢𝔤𝔯𝔞𝔱𝔦𝔬𝔫 𝔬𝔣 𝔏𝔬𝔫𝔤 𝔞𝔫𝔡 𝔖𝔥𝔬𝔯𝔱 𝔗𝔯𝔞𝔡𝔢𝔰
# 𝔄𝔦𝔯𝔦𝔫𝔤 𝔱𝔥𝔢 𝔥𝔞𝔯𝔡 𝔦𝔫 𝔒𝔠𝔥𝔩𝔠𝔞𝔵𝔩𝔱 𝔡𝔢𝔱𝔢𝔯𝔪𝔦𝔫𝔞𝔱𝔦𝔬𝔫𝔰.
# Author: @Mablue (Masoud Azizi)
# GitHub: https://github.com/mablue/
# * IMPORTANT: You Need A "STATIC" Pairlist In Your Config.json!
# * IMPORTANT: First set PAIR_LIST_LENGTH={pair_whitelist size}
# * Then re-hyperopt the Strategy and paste the results in the exact
# * place (lines corresponding to spell results)

# Instructions for Hyperoptimization:
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell short cover -s DS_Short

# --- Essential Libraries ---
import numpy as np
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import random
from freqtrade.strategy import CategoricalParameter, IStrategy

from pandas import DataFrame

# ########################## SETTINGS ##############################
# Length of the pair list (use exact count of pairs you used in whitelist size)
PAIR_LIST_LENGTH = 269

# Number of candles to check for trend determination
TREND_CHECK_CANDLES = 4

# Pain range to determine the number of spell combinations
PAIN_RANGE = 1000

# Dictionary of spells containing buy and sell parameters
# Each spell corresponds to a unique set of trading conditions
# Replace GodStraNew results with unique phonemes like 'Zi', 'Gu', or 'Lu'
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
    "Gu": {
        "buy_params": {
            "buy_crossed_indicator0": "SMA-20",
            "buy_crossed_indicator1": "CDLLADDERBOTTOM-20",
            "buy_crossed_indicator2": "OBV-50",
            "buy_indicator0": "MAMA-1-50",
            "buy_indicator1": "SUM-40",
            "buy_indicator2": "VAR-30",
            "buy_operator0": "<R",
            "buy_operator1": "D",
            "buy_operator2": "D",
            "buy_real_num0": 0.2644,
            "buy_real_num1": 0.0736,
            "buy_real_num2": 0.8954,
        },
        "sell_params": {
            "sell_crossed_indicator0": "CDLLADDERBOTTOM-50",
            "sell_crossed_indicator1": "CDLHARAMICROSS-50",
            "sell_crossed_indicator2": "CDLDARKCLOUDCOVER-30",
            "sell_indicator0": "CDLLADDERBOTTOM-10",
            "sell_indicator1": "MAMA-1-40",
            "sell_indicator2": "OBV-30",
            "sell_operator0": "UT",
            "sell_operator1": ">R",
            "sell_operator2": "CUT",
            "sell_real_num0": 0.2707,
            "sell_real_num1": 0.7987,
            "sell_real_num2": 0.6891,
        }
    },
    "Lu": {
        "buy_params": {
            "buy_crossed_indicator0": "HT_SINE-0-28",
            "buy_crossed_indicator1": "ADD-130",
            "buy_crossed_indicator2": "ADD-12",
            "buy_indicator0": "ADD-28",
            "buy_indicator1": "AVGPRICE-15",
            "buy_indicator2": "AVGPRICE-12",
            "buy_operator0": "DT",
            "buy_operator1": "D",
            "buy_operator2": "C",
            "buy_real_num0": 0.3676,
            "buy_real_num1": 0.4284,
            "buy_real_num2": 0.372,
        },
        "sell_params": {
            "sell_crossed_indicator0": "HT_SINE-0-5",
            "sell_crossed_indicator1": "HT_SINE-0-4",
            "sell_crossed_indicator2": "HT_SINE-0-28",
            "sell_indicator0": "ADD-30",
            "sell_indicator1": "AVGPRICE-28",
            "sell_indicator2": "ADD-50",
            "sell_operator0": "CUT",
            "sell_operator1": "DT",
            "sell_operator2": "=R",
            "sell_real_num0": 0.3205,
            "sell_real_num1": 0.2055,
            "sell_real_num2": 0.8467,
        }
    },
    "La": {
        "buy_params": {
            "buy_crossed_indicator0": "WMA-14",
            "buy_crossed_indicator1": "MAMA-1-14",
            "buy_crossed_indicator2": "CDLHIKKAKE-14",
            "buy_indicator0": "T3-14",
            "buy_indicator1": "BETA-14",
            "buy_indicator2": "HT_PHASOR-1-14",
            "buy_operator0": "/>R",
            "buy_operator1": ">",
            "buy_operator2": ">R",
            "buy_real_num0": 0.0551,
            "buy_real_num1": 0.3469,
            "buy_real_num2": 0.3871,
        },
        "sell_params": {
            "sell_crossed_indicator0": "HT_TRENDLINE-14",
            "sell_crossed_indicator1": "LINEARREG-14",
            "sell_crossed_indicator2": "STOCHRSI-1-14",
            "sell_indicator0": "CDLDARKCLOUDCOVER-14",
            "sell_indicator1": "AD-14",
            "sell_indicator2": "CDLSTALLEDPATTERN-14",
            "sell_operator0": "/=R",
            "sell_operator1": "COT",
            "sell_operator2": "OT",
            "sell_real_num0": 0.3992,
            "sell_real_num1": 0.7747,
            "sell_real_num2": 0.7415,
        }
    },
    "Si": {
        "buy_params": {
            "buy_crossed_indicator0": "MACDEXT-2-14",
            "buy_crossed_indicator1": "CORREL-14",
            "buy_crossed_indicator2": "CMO-14",
            "buy_indicator0": "MA-14",
            "buy_indicator1": "ADXR-14",
            "buy_indicator2": "CDLMARUBOZU-14",
            "buy_operator0": "<",
            "buy_operator1": "/<R",
            "buy_operator2": "<R",
            "buy_real_num0": 0.7883,
            "buy_real_num1": 0.8286,
            "buy_real_num2": 0.6512,
        },
        "sell_params": {
            "sell_crossed_indicator0": "AROON-1-14",
            "sell_crossed_indicator1": "STOCHRSI-0-14",
            "sell_crossed_indicator2": "SMA-14",
            "sell_indicator0": "T3-14",
            "sell_indicator1": "AROONOSC-14",
            "sell_indicator2": "MIDPOINT-14",
            "sell_operator0": "C",
            "sell_operator1": "CA",
            "sell_operator2": "CB",
            "sell_real_num0": 0.372,
            "sell_real_num1": 0.5948,
            "sell_real_num2": 0.9872,
        }
    },
    "Pa": {
        "buy_params": {
            "buy_crossed_indicator0": "AROON-0-60",
            "buy_crossed_indicator1": "APO-60",
            "buy_crossed_indicator2": "BBANDS-0-60",
            "buy_indicator0": "WILLR-12",
            "buy_indicator1": "AD-15",
            "buy_indicator2": "MINUS_DI-12",
            "buy_operator0": "D",
            "buy_operator1": ">",
            "buy_operator2": "CA",
            "buy_real_num0": 0.2208,
            "buy_real_num1": 0.1371,
            "buy_real_num2": 0.6389,
        },
        "sell_params": {
            "sell_crossed_indicator0": "MACDEXT-0-15",
            "sell_crossed_indicator1": "BBANDS-2-15",
            "sell_crossed_indicator2": "DEMA-15",
            "sell_indicator0": "ULTOSC-15",
            "sell_indicator1": "MIDPOINT-12",
            "sell_indicator2": "PLUS_DI-12",
            "sell_operator0": "<",
            "sell_operator1": "DT",
            "sell_operator2": "COT",
            "sell_real_num0": 0.278,
            "sell_real_num1": 0.0643,
            "sell_real_num2": 0.7065,
        }
    },
    "De": {
        "buy_params": {
            "buy_crossed_indicator0": "HT_DCPERIOD-12",
            "buy_crossed_indicator1": "HT_PHASOR-0-12",
            "buy_crossed_indicator2": "MACDFIX-1-15",
            "buy_indicator0": "CMO-12",
            "buy_indicator1": "TRIMA-12",
            "buy_indicator2": "MACDEXT-0-15",
            "buy_operator0": "<",
            "buy_operator1": "D",
            "buy_operator2": "<",
            "buy_real_num0": 0.3924,
            "buy_real_num1": 0.5546,
            "buy_real_num2": 0.7648,
        },
        "sell_params": {
            "sell_crossed_indicator0": "MACDFIX-1-15",
            "sell_crossed_indicator1": "MACD-1-15",
            "sell_crossed_indicator2": "WMA-15",
            "sell_indicator0": "ROC-15",
            "sell_indicator1": "MACD-2-15",
            "sell_indicator2": "CCI-60",
            "sell_operator0": "CA",
            "sell_operator1": "<R",
            "sell_operator2": "/<R",
            "sell_real_num0": 0.4989,
            "sell_real_num1": 0.4131,
            "sell_real_num2": 0.8904,
        }
    },
    "Ra": {
        "buy_params": {
            "buy_crossed_indicator0": "EMA-110",
            "buy_crossed_indicator1": "SMA-5",
            "buy_crossed_indicator2": "SMA-6",
            "buy_indicator0": "SMA-6",
            "buy_indicator1": "EMA-12",
            "buy_indicator2": "EMA-5",
            "buy_operator0": "D",
            "buy_operator1": "<",
            "buy_operator2": "/<R",
            "buy_real_num0": 0.9814,
            "buy_real_num1": 0.5528,
            "buy_real_num2": 0.0541,
        },
        "sell_params": {
            "sell_crossed_indicator0": "SMA-50",
            "sell_crossed_indicator1": "EMA-12",
            "sell_crossed_indicator2": "SMA-100",
            "sell_indicator0": "EMA-110",
            "sell_indicator1": "EMA-50",
            "sell_indicator2": "EMA-15",
            "sell_operator0": "<",
            "sell_operator1": "COT",
            "sell_operator2": "/=R",
            "sell_real_num0": 0.3506,
            "sell_real_num1": 0.8767,
            "sell_real_num2": 0.0614,
        }
    },
    "Cu": {
        "buy_params": {
            "buy_crossed_indicator0": "SMA-110",
            "buy_crossed_indicator1": "SMA-110",
            "buy_crossed_indicator2": "SMA-5",
            "buy_indicator0": "SMA-110",
            "buy_indicator1": "SMA-55",
            "buy_indicator2": "SMA-15",
            "buy_operator0": "<R",
            "buy_operator1": "<",
            "buy_operator2": "CA",
            "buy_real_num0": 0.5,
            "buy_real_num1": 0.7,
            "buy_real_num2": 0.9,
        },
        "sell_params": {
            "sell_crossed_indicator0": "SMA-55",
            "sell_crossed_indicator1": "SMA-50",
            "sell_crossed_indicator2": "SMA-100",
            "sell_indicator0": "SMA-5",
            "sell_indicator1": "SMA-50",
            "sell_indicator2": "SMA-50",
            "sell_operator0": "/=R",
            "sell_operator1": "CUT",
            "sell_operator2": "DT",
            "sell_real_num0": 0.4,
            "sell_real_num1": 0.2,
            "sell_real_num2": 0.7,
        }
    }
}

# Dictionary of short spells containing short and cover parameters
SHORT_SPELLS = {
    "ShortZi": {
        "short_params": {
            "short_crossed_indicator0": "RSI-14",
            "short_crossed_indicator1": "SMA-50",
            "short_crossed_indicator2": "MACD-0-50",
            "short_indicator0": "RSI-14",
            "short_indicator1": "SMA-50",
            "short_indicator2": "MACD-0-50",
            "short_operator0": ">",
            "short_operator1": "<",
            "short_operator2": ">R",
            "short_real_num0": 70,
            "short_real_num1": 0,
            "short_real_num2": 0.5,
        },
        "cover_params": {
            "cover_crossed_indicator0": "RSI-14",
            "cover_crossed_indicator1": "SMA-50",
            "cover_crossed_indicator2": "MACD-0-50",
            "cover_indicator0": "RSI-14",
            "cover_indicator1": "SMA-50",
            "cover_indicator2": "MACD-0-50",
            "cover_operator0": "<",
            "cover_operator1": ">",
            "cover_operator2": "<R",
            "cover_real_num0": 30,
            "cover_real_num1": 0,
            "cover_real_num2": 0.5,
        }
    },
    "ShortGu": {
        "short_params": {
            "short_crossed_indicator0": "STOCHRSI-14",
            "short_crossed_indicator1": "BBANDS-20-2",
            "short_crossed_indicator2": "CCI-100",
            "short_indicator0": "STOCHRSI-14",
            "short_indicator1": "BBANDS_upperband-20-2",
            "short_indicator2": "CCI-100",
            "short_operator0": ">R",
            "short_operator1": ">R",
            "short_operator2": ">",
            "short_real_num0": 0.8,
            "short_real_num1": 1.0,
            "short_real_num2": 100,
        },
        "cover_params": {
            "cover_crossed_indicator0": "STOCHRSI-14",
            "cover_crossed_indicator1": "BBANDS-20-2",
            "cover_crossed_indicator2": "CCI-100",
            "cover_indicator0": "STOCHRSI-14",
            "cover_indicator1": "BBANDS_lowerband-20-2",
            "cover_indicator2": "CCI-100",
            "cover_operator0": "<R",
            "cover_operator1": "<R",
            "cover_operator2": "<",
            "cover_real_num0": 0.2,
            "cover_real_num1": -1.0,
            "cover_real_num2": -100,
        }
    },
    # Continue adding more short spells to ensure diversity
    # Example:
    "ShortLu": {
        "short_params": {
            "short_crossed_indicator0": "EMA-50",
            "short_crossed_indicator1": "RSI-30",
            "short_crossed_indicator2": "MACD-12-26-9",
            "short_indicator0": "EMA-50",
            "short_indicator1": "RSI-30",
            "short_indicator2": "MACD-12-26-9",
            "short_operator0": ">R",
            "short_operator1": ">R",
            "short_operator2": ">",
            "short_real_num0": 1.2,
            "short_real_num1": 40,
            "short_real_num2": 0.5,
        },
        "cover_params": {
            "cover_crossed_indicator0": "EMA-50",
            "cover_crossed_indicator1": "RSI-30",
            "cover_crossed_indicator2": "MACD-12-26-9",
            "cover_indicator0": "EMA-50",
            "cover_indicator1": "RSI-30",
            "cover_indicator2": "MACD-12-26-9",
            "cover_operator0": "<R",
            "cover_operator1": "<R",
            "cover_operator2": "<",
            "cover_real_num0": 0.8,
            "cover_real_num1": 20,
            "cover_real_num2": -0.5,
        }
    },
    # Add as many short spells as needed for diversity
}

# ######################## END SETTINGS ############################

def spell_finder(index, space):
    """
    Retrieves the spell parameters based on the spell index and space type.

    Args:
        index (str): Key of the SPELLS or SHORT_SPELLS dictionary.
        space (str): Either 'buy', 'sell', 'short', or 'cover'.

    Returns:
        dict: Corresponding parameters for the spell.
    """
    if space in ["buy", "sell"]:
        return SPELLS[index][f"{space}_params"]
    elif space in ["short", "cover"]:
        return SHORT_SPELLS[index][f"{space}_params"]
    else:
        raise ValueError("Invalid space type. Choose from 'buy', 'sell', 'short', or 'cover'.")

def normalize(series):
    """
    Normalizes a pandas Series to a 0-1 range.

    Args:
        series (pd.Series): Data to normalize.

    Returns:
        pd.Series: Normalized data.
    """
    return (series - series.min()) / (series.max() - series.min())

def gene_calculator(dataframe, indicator):
    """
    Calculates the specified technical indicator and normalizes it.

    Args:
        dataframe (pd.DataFrame): The OHLCV dataframe.
        indicator (str): Indicator string in the format 'INDICATOR-TIME' or 'INDICATOR-INDEX-TIME'.

    Returns:
        pd.Series: Normalized indicator values.
    """
    # Handle CDL pattern indicators which do not depend on time periods
    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = "0"  # Reset time period
        new_indicator = "-".join(splited_indicator)
        indicator = new_indicator

    gene = indicator.split("-")
    gene_name = gene[0]
    gene_len = len(gene)

    # If the indicator is already calculated, return it
    if indicator in dataframe.columns:
        return dataframe[indicator]
    else:
        result = None
        # Calculate the indicator based on the number of parameters
        if gene_len == 1:
            # Indicators without time period
            result = getattr(ta, gene_name)(dataframe)
            return normalize(result)
        elif gene_len == 2:
            # Indicators with a single time period
            gene_timeperiod = int(gene[1])
            result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
            return normalize(result)
        elif gene_len == 3:
            # Indicators with an index (e.g., MACDEXT-2-14)
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
            return normalize(result)
        elif gene_len == 4:
            # Trend operators with additional parameters
            gene_timeperiod = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
        elif gene_len == 5:
            # More complex trend operators (e.g., STOCH-0-4-SMA-4)
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))

def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):
    """
    Generates trading conditions based on the operator and indicators.

    Args:
        dataframe (pd.DataFrame): The OHLCV dataframe.
        operator (str): The operator defining the condition.
        indicator (str): The primary indicator.
        crossed_indicator (str): The indicator to compare with.
        real_num (float): Numerical value for comparison.

    Returns:
        tuple: A tuple containing the condition (pd.Series) and the updated dataframe.
    """
    # Initial condition based on trading volume
    condition = (dataframe['volume'] > 10)

    # Calculate and normalize the primary and crossed indicators
    dataframe[indicator] = gene_calculator(dataframe, indicator)
    dataframe[crossed_indicator] = gene_calculator(dataframe, crossed_indicator)

    # Handle trend-based operators by calculating the trend SMA
    indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
    if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
        dataframe[indicator_trend_sma] = gene_calculator(dataframe, indicator_trend_sma)

    # Define conditions based on the operator type
    if operator == ">":
        condition &= (dataframe[indicator] > dataframe[crossed_indicator])
    elif operator == "=":
        condition &= (np.isclose(dataframe[indicator], dataframe[crossed_indicator]))
    elif operator == "<":
        condition &= (dataframe[indicator] < dataframe[crossed_indicator])
    elif operator == "C":
        condition &= (
            qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator]) |
            qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "CA":
        condition &= qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == "CB":
        condition &= qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == ">R":
        condition &= (dataframe[indicator] > real_num)
    elif operator == "=R":
        condition &= (np.isclose(dataframe[indicator], real_num))
    elif operator == "<R":
        condition &= (dataframe[indicator] < real_num)
    elif operator == "/>R":
        condition &= (dataframe[indicator].div(dataframe[crossed_indicator]) > real_num)
    elif operator == "/=R":
        condition &= (np.isclose(dataframe[indicator].div(dataframe[crossed_indicator]), real_num))
    elif operator == "/<R":
        condition &= (dataframe[indicator].div(dataframe[crossed_indicator]) < real_num)
    elif operator == "UT":
        condition &= (dataframe[indicator] > dataframe[indicator_trend_sma])
    elif operator == "DT":
        condition &= (dataframe[indicator] < dataframe[indicator_trend_sma])
    elif operator == "OT":
        condition &= (np.isclose(dataframe[indicator], dataframe[indicator_trend_sma]))
    elif operator == "CUT":
        condition &= (
            qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma]) &
            (dataframe[indicator] > dataframe[indicator_trend_sma])
        )
    elif operator == "CDT":
        condition &= (
            qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) &
            (dataframe[indicator] < dataframe[indicator_trend_sma])
        )
    elif operator == "COT":
        condition &= (
            (
                qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) |
                qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma])
            ) &
            (np.isclose(dataframe[indicator], dataframe[indicator_trend_sma]))
        )

    return condition, dataframe

class DS_Short(IStrategy):
    """
    DS_Short is a customized trading strategy for the Freqtrade platform.
    It utilizes a combination of technical indicators and "spells" to determine buy, sell, short, and cover signals.
    The strategy is designed to operate on a 15-minute timeframe.
    """

    # Interface version for Freqtrade compatibility
    INTERFACE_VERSION: int = 3

    # Buy hyperspace parameters (Resulted from previous hyperoptimization)
    buy_params = {
        "buy_spell": "Zi,Lu,Ra,Ra,La,Si,Pa,Si,Cu,La,De,Lu,De,La,Zi,Zi,Zi,Zi,Zi,Lu,Lu,Lu,Si,La,Ra,Pa,La,Zi,Zi,Gu,Ra,De,Gu,Zi,Ra,Ra,Ra,Cu,Pa,De,De,La,Lu,Lu,Lu,La,Zi,Cu,Ra,Gu,Pa,La,Zi,Zi,Si,Lu,Ra,Cu,Cu,Pa,Si,Gu,De,De,Lu,Gu,Zi,Pa,Lu,Pa,Ra,Gu,Cu,La,Pa,Lu,Zi,La,Zi,Gu,Zi,De,Cu,Ra,Lu,Ra,Gu,Si,Ra,La,La,Lu,Gu,Zi,Si,La,Pa,Pa,Cu,Cu,Zi,Gu,Pa,Zi,Pa,Cu,Lu,Pa,Si,De,Gu,Lu,Lu,Cu,Ra,Si,Pa,Gu,Si,Cu,Pa,Zi,Pa,Zi,Gu,Lu,Ra,Pa,Ra,De,Ra,Pa,Zi,La,Pa,De,Pa,Cu,Gu,De,Lu,La,Ra,Zi,Si,Zi,Zi,Cu,Cu,De,Pa,Pa,Zi,De,Ra,La,Lu,De,Lu,Gu,Cu,Cu,La,De,Gu,Lu,Ra,Pa,Lu,Cu,Pa,Pa,De,Si,Zi,Cu,De,De,De,Lu,Si,Zi,Gu,Si,Si,Ra,Pa,Si,La,La,Lu,Lu,De,Gu,Gu,Zi,Ra,La,Lu,Lu,La,Si,Zi,Si,Zi,Si,Lu,Cu,Zi,Lu,De,La,Ra,Ra,Lu,De,Pa,Zi,Gu,Cu,Zi,Pa,De,Si,Lu,De,Cu,De,Zi,Ra,Gu,De,Si,Lu,Lu,Ra,De,Gu,Cu,Gu,La,De,Lu,Lu,Si,Cu,Lu,Zi,Lu,Cu,Gu,Lu,Lu,Ra,Si,Ra,Pa,Lu,De,Ra,Zi,Gu,Gu,Zi,Lu,Cu,Cu,Cu,Lu",
    }

    # Sell hyperspace parameters (Resulted from previous hyperoptimization)
    sell_params = {
        "sell_spell": "La,Pa,De,De,La,Si,Si,La,La,La,Si,Pa,Pa,Lu,De,Cu,Cu,Gu,Lu,Ra,Lu,Si,Ra,De,La,Cu,La,La,Gu,La,De,Ra,Ra,Ra,Gu,Lu,Si,Si,Zi,Zi,La,Pa,Pa,Zi,Cu,Gu,Gu,Pa,Gu,Cu,Si,Ra,Ra,La,Gu,De,Si,La,Ra,Pa,Si,Lu,Pa,De,Zi,De,Lu,Si,Gu,De,Lu,De,Ra,Ra,Zi,De,Cu,Zi,Gu,Pa,Ra,De,Pa,De,Pa,Ra,Si,Si,Zi,Cu,Lu,Zi,Ra,De,Ra,Zi,Zi,Pa,Lu,Zi,Cu,Pa,Gu,Pa,Cu,De,Zi,De,De,Pa,Pa,Zi,Lu,Ra,Pa,Ra,Lu,Zi,Gu,Zi,Si,Lu,Ra,Ra,Zi,Lu,Pa,Lu,Si,Pa,Pa,Pa,Si,Zi,La,La,Lu,De,Zi,Gu,Ra,Ra,Ra,Zi,Pa,Zi,Cu,Lu,Gu,Cu,De,Lu,Gu,Lu,Gu,Si,Pa,Pa,Si,La,Gu,Ra,Pa,Si,Si,Si,Cu,Cu,Cu,Si,De,Lu,Gu,Gu,Lu,De,Ra,Gu,Gu,Gu,Cu,La,De,Cu,Zi,Pa,Si,De,Pa,Pa,Pa,La,De,Gu,Zi,La,De,Cu,La,Pa,Ra,Si,Si,Zi,Cu,Ra,Pa,Gu,Pa,Ra,Zi,De,Zi,Gu,Gu,Pa,Cu,Lu,Gu,De,Si,Pa,La,Cu,Zi,Gu,De,Gu,La,Cu,Gu,De,Cu,Cu,Gu,Ra,Lu,Zi,De,La,Ra,Pa,Pa,Si,La,Lu,La,De,De,Ra,De,La,La,Pa,Cu,Lu,Pa,Ra,Pa,Pa,Cu,Zi,Gu,Cu,Gu,La,Si,Ra,Pa",
    }

    # Short hyperspace parameters (Customize as needed)
    short_params = {
        "short_spell": "ShortZi,ShortGu,ShortLu",
        "cover_spell": "ShortZi,ShortGu,ShortLu",
    }

    # ROI table defining the return on investment targets for long trades
    minimal_roi = {
        "0": 0.25,     # 25% ROI from the start
        "60": 0.15,    # 15% ROI after 60 minutes
        "180": 0.07,   # 7% ROI after 180 minutes
        "360": 0        # No ROI after 360 minutes
    }

    # ROI table for short trades with higher profit targets and shorter durations
    minimal_roi_short = {
        "0": 0.30,     # 30% ROI from the start
        "60": 0.20,    # 20% ROI after 60 minutes
        "180": 0.10,   # 10% ROI after 180 minutes
        "360": 0        # No ROI after 360 minutes
    }

    # Stoploss setting to limit losses for both long and short trades
    stoploss = -0.25  # -25%

    # Trading timeframe set to 15 minutes
    timeframe = '5m'

    # Generate a pool of spell combinations based on the pain range and pair list length
    spell_pot = [
        ",".join(
            random.choices(
                list(SPELLS.keys()),
                k=PAIR_LIST_LENGTH  # Number of spells in each combination
            )
        ) for _ in range(PAIN_RANGE)  # Total number of spell combinations
    ]

    # Generate a pool of short spell combinations
    short_spell_pot = [
        ",".join(
            random.choices(
                list(SHORT_SPELLS.keys()),
                k=PAIR_LIST_LENGTH  # Number of spells in each combination
            )
        ) for _ in range(PAIN_RANGE)  # Total number of short spell combinations
    ]

    # Define buy and sell spells as categorical parameters for hyperoptimization
    buy_spell = CategoricalParameter(
        spell_pot, default=spell_pot[0], space='buy')
    sell_spell = CategoricalParameter(
        spell_pot, default=spell_pot[0], space='sell')

    # Define short and cover spells as categorical parameters for hyperoptimization
    short_spell = CategoricalParameter(
        short_spell_pot, default=short_spell_pot[0], space='short')
    cover_spell = CategoricalParameter(
        short_spell_pot, default=short_spell_pot[0], space='cover')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the dataframe with all necessary indicators.
        In this strategy, indicators are generated on-the-fly in the condition generator,
        so this function remains empty.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata.

        Returns:
            pd.DataFrame: The dataframe with indicators populated.
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines the conditions under which the strategy will enter a long position.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata, including the current trading pair.

        Returns:
            pd.DataFrame: The dataframe with 'enter_long' signals populated.
        """
        # Retrieve the current whitelist of trading pairs
        pairs = self.dp.current_whitelist()
        pairs_len = len(pairs)
        try:
            pair_index = pairs.index(metadata['pair'])  # Get index of the current pair
        except ValueError:
            # Pair not found in whitelist
            return dataframe

        # Retrieve the list of buy spells
        buy_spells = self.buy_spell.value.split(",")
        buy_spells_len = len(buy_spells)

        # Ensure the PAIR_LIST_LENGTH matches the whitelist size
        if pairs_len > buy_spells_len:
            print(
                f"First set PAIR_LIST_LENGTH={pairs_len} and re-hyperopt the"
            )
            print("Buy strategy and paste the result in the exact place (lines corresponding to spell results)")
            print("IMPORTANT: You Need A 'STATIC' Pairlist In Your Config.json !!!")
            exit()

        # Get the spell parameters for the current pair
        buy_params_index = buy_spells[pair_index]
        params = spell_finder(buy_params_index, 'buy')

        conditions = []  # List to hold all buy conditions

        # Generate conditions based on the buy parameters
        for i in range(3):  # Assuming three buy indicators
            buy_indicator = params[f'buy_indicator{i}']
            buy_crossed_indicator = params[f'buy_crossed_indicator{i}']
            buy_operator = params[f'buy_operator{i}']
            buy_real_num = params[f'buy_real_num{i}']

            condition, dataframe = condition_generator(
                dataframe,
                buy_operator,
                buy_indicator,
                buy_crossed_indicator,
                buy_real_num
            )
            conditions.append(condition)

        # Combine all conditions using logical AND
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1  # Signal to enter a long position

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines the conditions under which the strategy will exit a long position.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata, including the current trading pair.

        Returns:
            pd.DataFrame: The dataframe with 'exit_long' signals populated.
        """
        # Retrieve the current whitelist of trading pairs
        pairs = self.dp.current_whitelist()
        pairs_len = len(pairs)
        try:
            pair_index = pairs.index(metadata['pair'])  # Get index of the current pair
        except ValueError:
            # Pair not found in whitelist
            return dataframe

        # Retrieve the list of sell spells
        sell_spells = self.sell_spell.value.split(",")
        sell_spells_len = len(sell_spells)

        # Ensure the PAIR_LIST_LENGTH matches the whitelist size
        if pairs_len > sell_spells_len:
            print(
                f"First set PAIR_LIST_LENGTH={pairs_len} and re-hyperopt the"
            )
            print("Sell strategy and paste the result in the exact place (lines corresponding to spell results)")
            print("IMPORTANT: You Need A 'STATIC' Pairlist In Your Config.json !!!")
            exit()

        # Get the spell parameters for the current pair
        sell_params_index = sell_spells[pair_index]
        params = spell_finder(sell_params_index, 'sell')

        conditions = []  # List to hold all sell conditions

        # Generate conditions based on the sell parameters
        for i in range(3):  # Assuming three sell indicators
            sell_indicator = params[f'sell_indicator{i}']
            sell_crossed_indicator = params[f'sell_crossed_indicator{i}']
            sell_operator = params[f'sell_operator{i}']
            sell_real_num = params[f'sell_real_num{i}']

            condition, dataframe = condition_generator(
                dataframe,
                sell_operator,
                sell_indicator,
                sell_crossed_indicator,
                sell_real_num
            )
            conditions.append(condition)

        # Combine all conditions using logical AND
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1  # Signal to exit a long position

        return dataframe

    def populate_short_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines the conditions under which the strategy will enter a short position.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata, including the current trading pair.

        Returns:
            pd.DataFrame: The dataframe with 'enter_short' signals populated.
        """
        # Retrieve the current whitelist of trading pairs
        pairs = self.dp.current_whitelist()
        pairs_len = len(pairs)
        try:
            pair_index = pairs.index(metadata['pair'])  # Get index of the current pair
        except ValueError:
            # Pair not found in whitelist
            return dataframe

        # Retrieve the list of short spells
        short_spells = self.short_spell.value.split(",")
        short_spells_len = len(short_spells)

        # Ensure the PAIR_LIST_LENGTH matches the whitelist size
        if pairs_len > short_spells_len:
            print(
                f"First set PAIR_LIST_LENGTH={pairs_len} and re-hyperopt the"
            )
            print("Short strategy and paste the result in the exact place (lines corresponding to spell results)")
            print("IMPORTANT: You Need A 'STATIC' Pairlist In Your Config.json !!!")
            exit()

        # Get the spell parameters for the current pair
        short_params_index = short_spells[pair_index]
        params = spell_finder(short_params_index, 'short')

        conditions = []  # List to hold all short entry conditions

        # Generate conditions based on the short parameters
        for i in range(3):  # Assuming three short indicators
            short_indicator = params[f'short_indicator{i}']
            short_crossed_indicator = params[f'short_crossed_indicator{i}']
            short_operator = params[f'short_operator{i}']
            short_real_num = params[f'short_real_num{i}']

            condition, dataframe = condition_generator(
                dataframe,
                short_operator,
                short_indicator,
                short_crossed_indicator,
                short_real_num
            )
            conditions.append(condition)

        # Combine all conditions using logical AND
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1  # Signal to enter a short position

        return dataframe

    def populate_cover_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines the conditions under which the strategy will cover a short position.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata, including the current trading pair.

        Returns:
            pd.DataFrame: The dataframe with 'exit_short' signals populated.
        """
        # Retrieve the current whitelist of trading pairs
        pairs = self.dp.current_whitelist()
        pairs_len = len(pairs)
        try:
            pair_index = pairs.index(metadata['pair'])  # Get index of the current pair
        except ValueError:
            # Pair not found in whitelist
            return dataframe

        # Retrieve the list of cover spells
        cover_spells = self.cover_spell.value.split(",")
        cover_spells_len = len(cover_spells)

        # Ensure the PAIR_LIST_LENGTH matches the whitelist size
        if pairs_len > cover_spells_len:
            print(
                f"First set PAIR_LIST_LENGTH={pairs_len} and re-hyperopt the"
            )
            print("Cover strategy and paste the result in the exact place (lines corresponding to spell results)")
            print("IMPORTANT: You Need A 'STATIC' Pairlist In Your Config.json !!!")
            exit()

        # Get the spell parameters for the current pair
        cover_params_index = cover_spells[pair_index]
        params = spell_finder(cover_params_index, 'cover')

        conditions = []  # List to hold all cover conditions

        # Generate conditions based on the cover parameters
        for i in range(3):  # Assuming three cover indicators
            cover_indicator = params[f'cover_indicator{i}']
            cover_crossed_indicator = params[f'cover_crossed_indicator{i}']
            cover_operator = params[f'cover_operator{i}']
            cover_real_num = params[f'cover_real_num{i}']

            condition, dataframe = condition_generator(
                dataframe,
                cover_operator,
                cover_indicator,
                cover_crossed_indicator,
                cover_real_num
            )
            conditions.append(condition)

        # Combine all conditions using logical AND
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_short'] = 1  # Signal to exit a short position

        return dataframe

    def populate_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Overriding the default populate_trend to include short trades.

        Args:
            dataframe (pd.DataFrame): The OHLCV dataframe.
            metadata (dict): Additional metadata.

        Returns:
            pd.DataFrame: The dataframe with all trend signals.
        """
        dataframe = self.populate_entry_trend(dataframe, metadata)
        dataframe = self.populate_exit_trend(dataframe, metadata)
        dataframe = self.populate_short_trend(dataframe, metadata)
        dataframe = self.populate_cover_trend(dataframe, metadata)
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Custom stoploss to handle both long and short positions.

        Args:
            pair (str): The trading pair.
            trade: The trade object.
            current_time: The current time.
            current_rate: The current rate.
            current_profit: The current profit.

        Returns:
            float: New stoploss.
        """
        if trade.is_short:
            return self.stoploss  # Apply the same stoploss for short trades
        else:
            return self.stoploss  # Apply the same stoploss for long trades

# ######################## END STRATEGY ############################