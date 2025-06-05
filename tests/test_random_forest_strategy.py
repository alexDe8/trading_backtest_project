import pandas as pd
import numpy as np

from trading_backtest.strategy.random_forest import RandomForestStrategy
from trading_backtest.config import RandomForestConfig


def _dummy_df():
    n = 30
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="T"),
        "open": np.linspace(1, n, n),
        "high": np.linspace(1, n, n) + 0.5,
        "low": np.linspace(1, n, n) - 0.5,
        "close": np.linspace(1, n, n),
    })
    for w in [5, 10]:
        df[f"sma_{w}"] = df["close"]
    df["rsi_14"] = 50
    df["atr_14"] = 1.0
    df["vol_20"] = 0.5
    df["impulse_5"] = 0.1
    return df


def test_random_forest_generate_trades_runs():
    df = _dummy_df()
    cfg = RandomForestConfig(
        entry_threshold=0.5, exit_threshold=0.5, sl_pct=1, tp_pct=2, n_estimators=10
    )
    strat = RandomForestStrategy(cfg)
    trades = strat.generate_trades(df)
    assert isinstance(trades, pd.DataFrame)
