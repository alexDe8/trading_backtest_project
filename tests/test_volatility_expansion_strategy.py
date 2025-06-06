import pandas as pd
import numpy as np

from trading_backtest.strategy.momentum import VolatilityExpansionStrategy
from trading_backtest.config import VolExpansionConfig


def test_prepare_indicators_backfills_nan():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="T"),
            "open": np.arange(5),
            "high": np.arange(5),
            "low": np.arange(5),
            "close": np.arange(5),
            "vol_20": [np.nan, np.nan, 0.6, 0.4, 0.7],
        }
    )
    cfg = VolExpansionConfig(vol_window=20, vol_threshold=0.01, sl_pct=1, tp_pct=2)
    strat = VolatilityExpansionStrategy(cfg)
    processed = strat.prepare_indicators(df.copy())
    assert not processed["v"].isna().any()
    expected_v = [0.6, 0.6, 0.6, 0.4, 0.7]
    assert processed["v"].tolist() == expected_v

    entries = strat.entry_signal(processed)
    exits = strat.exit_signal(processed)
    assert entries.dtype == bool
    assert exits.dtype == bool
    assert not entries.isna().any()
    assert not exits.isna().any()
    pd.testing.assert_series_equal(entries, processed["v"] > cfg.vol_threshold)
    pd.testing.assert_series_equal(exits, processed["v"] < cfg.vol_threshold)
