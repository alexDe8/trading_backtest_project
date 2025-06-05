import pandas as pd
import numpy as np
import pytest

from trading_backtest.strategy import get_strategy

from trading_backtest.config import (
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
    MACDConfig,
    StochasticConfig,
)


def _dummy_df() -> pd.DataFrame:
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="T"),
            "open": np.linspace(1, n, n),
            "high": np.linspace(1, n, n) + 0.5,
            "low": np.linspace(1, n, n) - 0.5,
            "close": np.linspace(1, n, n),
        }
    )
    df["sma_5"] = df["close"]
    df["sma_10"] = df["close"]
    df["sma_20"] = df["close"]
    df["rsi_14"] = 50
    df["atr_14"] = 1.0
    df["hmax_20"] = df["close"].rolling(20).max()
    df["bbm_20"] = df["close"]
    df["bbs_20"] = 1.0
    df["impulse_10"] = 0.1
    df["vol_20"] = 0.5
    return df.bfill().ffill()


@pytest.mark.parametrize(
    "name, cfg",
    [
        (
            "sma",
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
        ("rsi", RSIConfig(period=14, oversold=30, sl_pct=1, tp_pct=2)),
        (
            "breakout",
            BreakoutConfig(
                lookback=20, atr_period=14, atr_mult=1.0, sl_pct=1, tp_pct=2
            ),
        ),
        (
            "bollinger",
            BollingerConfig(period=20, nstd=2.0, sl_pct=1, tp_pct=2),
        ),
        (
            "momentum",
            MomentumConfig(window=10, threshold=0, sl_pct=1, tp_pct=2),
        ),
        (
            "vol_expansion",
            VolExpansionConfig(vol_window=20, vol_threshold=0.4, sl_pct=1, tp_pct=2),
        ),
        ("macd", MACDConfig(fast=12, slow=26, signal=9, sl_pct=1, tp_pct=2)),
        (
            "stochastic",
            StochasticConfig(k_period=14, d_period=3, oversold=20, sl_pct=1, tp_pct=2),
        ),
    ],
)
def test_generate_trades_runs(name, cfg):
    df = _dummy_df()
    strategy_cls, _ = get_strategy(name)
    strat = strategy_cls(cfg)
    trades = strat.generate_trades(df)
    assert isinstance(trades, pd.DataFrame)
