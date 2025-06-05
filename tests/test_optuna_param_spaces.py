import pathlib
import sys
import optuna
import pandas as pd
import numpy as np
import pytest

from trading_backtest.optimize import (
    optimize_with_optuna,
    prune_sma,
    prune_rsi,
    prune_breakout,
    prune_bollinger,
    prune_momentum,
    prune_vol_expansion,
    SMAParamSpace,
    RSIParamSpace,
    BreakoutParamSpace,
    BollingerParamSpace,
    MomentumParamSpace,
    VolExpansionParamSpace,
    check_sl_tp,
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


def test_generate_trades_runs():
    # Variante: puoi anche parametrizzare con pytest.mark.parametrize come nel test di Codex!
    configs = [
        (
            SMACrossoverStrategy,
            SMAConfig(
                sma_fast=5,
                sma_slow=10,
                sma_trend=20,
                sl_pct=1,
                tp_pct=2,
                position_size=1,
                trailing_stop_pct=1,
            ),
        ),
        (
            RSIStrategy,
            RSIConfig(period=14, oversold=30, sl_pct=1, tp_pct=2),
        ),
        (
            BreakoutStrategy,
            BreakoutConfig(
                lookback=20, atr_period=14, atr_mult=1.0, sl_pct=1, tp_pct=2
            ),
        ),
        (
            BollingerBandStrategy,
            BollingerConfig(period=20, nstd=2.0, sl_pct=1, tp_pct=2),
        ),
        (
            MomentumImpulseStrategy,
            MomentumConfig(window=10, threshold=0, sl_pct=1, tp_pct=2),
        ),
        (
            VolatilityExpansionStrategy,
            VolExpansionConfig(
                vol_window=20, vol_threshold=0.4, sl_pct=1, tp_pct=2
            ),
        ),
    ]
    df = _dummy_df()
    for strategy_cls, cfg in configs:
        strat = strategy_cls(cfg)
        trades = strat.generate_trades(df)
        assert isinstance(trades, pd.DataFrame)

def test_optimize_instantiates_strategies():
    df = _dummy_df()
    configs = [
        (SMACrossoverStrategy, SMAConfig, SMAParamSpace(), prune_sma),
        (RSIStrategy, RSIConfig, RSIParamSpace(), prune_rsi),
        (BreakoutStrategy, BreakoutConfig, BreakoutParamSpace(), prune_breakout),
        (
            BollingerBandStrategy,
            BollingerConfig,
            BollingerParamSpace(),
            prune_bollinger,
        ),
        (
            MomentumImpulseStrategy,
            MomentumConfig,
            MomentumParamSpace(),
            prune_momentum,
        ),
        (
            VolatilityExpansionStrategy,
            VolExpansionConfig,
            VolExpansionParamSpace(),
            prune_vol_expansion,
        ),
    ]
    for cls, cfg_cls, space, prune in configs:
        optimize_with_optuna(df, cls, cfg_cls, space, prune_logic=prune, n_trials=1)

def test_check_sl_tp_pruning():
    check_sl_tp({"sl_pct": 5, "tp_pct": 10})
    with pytest.raises(optuna.TrialPruned):
        check_sl_tp({"sl_pct": 10, "tp_pct": 5})

