import pandas as pd
from trading_backtest.optimize import ensure_indicator_cache


def _dummy_df(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="T"),
            "open": range(rows),
            "high": range(1, rows + 1),
            "low": range(rows),
            "close": range(rows),
        }
    )


def test_grid_search_indicators_cached():
    df = _dummy_df(80)
    combos = [
        {"sma_fast": 5, "sma_slow": 10, "sma_trend": None},
        {"sma_fast": 7, "sma_slow": 15, "sma_trend": 30},
    ]
    periods = ensure_indicator_cache(df, combos)
    for p in [5, 7, 10, 15, 30]:
        assert f"sma_{p}" in df
    assert periods["sma"] == [5, 7, 10, 15, 30]
