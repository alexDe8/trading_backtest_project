import pathlib
import sys

import pandas as pd
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from trading_backtest.optimize import (
    optimize_with_optuna,
    PARAM_SPACES,
    prune_sma,
    prune_rsi,
    prune_breakout,
    prune_bollinger,
    prune_momentum,
    prune_vol_expansion,
)
from trading_backtest.strategy.sma import SMACrossoverStrategy
from trading_backtest.strategy.rsi import RSIStrategy
from trading_backtest.strategy.breakout import BreakoutStrategy
from trading_backtest.strategy.bollinger import BollingerBandStrategy
from trading_backtest.strategy.momentum import (
    MomentumImpulseStrategy,
    VolatilityExpansionStrategy,
)
from trading_backtest.config import (
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
)


def _dummy_df() -> pd.DataFrame:
    n = 150
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="T"),
            "open": np.linspace(1, n, n),
            "high": np.linspace(1, n, n) + 0.5,
            "low": np.linspace(1, n, n) - 0.5,
            "close": np.linspace(1, n, n),
        }
    )

    # SMA columns
    for i in list(range(5, 55, 5)) + list(range(100, 255, 5)) + [300, 400]:
        df[f"sma_{i}"] = df["close"]

    # RSI & ATR columns
    for i in range(7, 22):
        df[f"rsi_{i}"] = 50
        df[f"atr_{i}"] = 1.0

    # HMAX columns for breakout
    for i in range(20, 105, 5):
        df[f"hmax_{i}"] = df["close"].rolling(i).max()

    # Bollinger columns
    for i in range(10, 32, 2):
        df[f"bbm_{i}"] = df["close"]
        df[f"bbs_{i}"] = 1.0

    # Momentum/volatility columns
    for i in range(5, 21):
        df[f"impulse_{i}"] = 0.1
    for i in range(20, 105, 5):
        df[f"vol_{i}"] = 0.5

    return df.fillna(method="bfill").fillna(method="ffill")


def test_optimize_instantiates_strategies():
    df = _dummy_df()
    configs = [
        (SMACrossoverStrategy, SMAConfig, PARAM_SPACES["sma"], prune_sma),
        (RSIStrategy, RSIConfig, PARAM_SPACES["rsi"], prune_rsi),
        (BreakoutStrategy, BreakoutConfig, PARAM_SPACES["breakout"], prune_breakout),
        (
            BollingerBandStrategy,
            BollingerConfig,
            PARAM_SPACES["bollinger"],
            prune_bollinger,
        ),
        (
            MomentumImpulseStrategy,
            MomentumConfig,
            PARAM_SPACES["momentum"],
            prune_momentum,
        ),
        (
            VolatilityExpansionStrategy,
            VolExpansionConfig,
            PARAM_SPACES["vol_expansion"],
            prune_vol_expansion,
        ),
    ]
    for cls, cfg_cls, space, prune in configs:
        optimize_with_optuna(df, cls, cfg_cls, space, prune_logic=prune, n_trials=1)
