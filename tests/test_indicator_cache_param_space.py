import pandas as pd
from trading_backtest.data import add_indicator_cache
from trading_backtest.optimize import gather_indicator_periods, PARAM_SPACES


def _dummy_df(rows: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="T"),
            "open": range(rows),
            "high": range(1, rows + 1),
            "low": range(rows),
            "close": range(rows),
        }
    )


def test_all_param_space_indicators_exist():
    df = _dummy_df(60)
    for name in ["sma", "rsi", "breakout", "bollinger", "momentum", "vol_expansion"]:
        periods = gather_indicator_periods(name)
        copy = df.copy()
        add_indicator_cache(
            copy,
            sma=periods.get("sma", []),
            rsi=periods.get("rsi", []),
            atr=periods.get("atr", []),
            vol=periods.get("vol", []),
            imp=periods.get("imp", []),
            hmax=periods.get("hmax", []),
            bb=periods.get("bb", []),
        )
        for p in periods.get("sma", []):
            assert f"sma_{p}" in copy
        for p in periods.get("rsi", []):
            assert f"rsi_{p}" in copy
        for p in periods.get("atr", []):
            assert f"atr_{p}" in copy
        for p in periods.get("vol", []):
            assert f"vol_{p}" in copy
        for p in periods.get("imp", []):
            assert f"impulse_{p}" in copy
        for p in periods.get("hmax", []):
            assert f"hmax_{p}" in copy
        for p in periods.get("bb", []):
            assert f"bbm_{p}" in copy and f"bbs_{p}" in copy
