import pandas as pd
import numpy as np
import pandas.testing as pdt

from trading_backtest.data import add_indicator_cache


def test_add_indicator_cache_small_df():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1, 2, 3, 4, 5],
        }
    )

    expected = df.copy()
    w = 2

    # SMA
    expected[f"sma_{w}"] = expected["close"].rolling(w).mean().shift(1)

    # RSI
    delta = expected["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.rolling(w, min_periods=1).mean()
    loss = down.rolling(w, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_vals = 100 - 100 / (1 + rs)
    rsi_vals = rsi_vals.bfill()
    expected[f"rsi_{w}"] = rsi_vals.shift(1)

    # ATR and true range
    tr = np.maximum(
        expected["high"] - expected["low"],
        np.maximum(
            abs(expected["high"] - expected["close"].shift()),
            abs(expected["low"] - expected["close"].shift()),
        ),
    )
    expected["tr"] = tr
    expected[f"atr_{w}"] = tr.rolling(w).mean().shift(1)

    # Historical volatility
    expected[f"vol_{w}"] = expected["close"].pct_change().rolling(w).std().shift(1)

    # Impulse
    expected[f"impulse_{w}"] = expected["close"].pct_change(w).shift(1)

    # Breakout high max
    expected[f"hmax_{w}"] = expected["close"].shift(1).rolling(w).max()

    # Bollinger bands
    expected[f"bbm_{w}"] = expected["close"].rolling(w).mean().shift(1)
    expected[f"bbs_{w}"] = expected["close"].rolling(w).std().shift(1)

    add_indicator_cache(
        df, sma=[2], rsi=[2], atr=[2], vol=[2], imp=[2], hmax=[2], bb=[2]
    )

    for col in [
        "sma_2",
        "rsi_2",
        "tr",
        "atr_2",
        "vol_2",
        "impulse_2",
        "hmax_2",
        "bbm_2",
        "bbs_2",
    ]:
        assert col in df.columns

    pdt.assert_series_equal(df["sma_2"], expected["sma_2"])
    pdt.assert_series_equal(df["rsi_2"], expected["rsi_2"])
    pdt.assert_series_equal(df["tr"], expected["tr"])
    pdt.assert_series_equal(df["atr_2"], expected["atr_2"])
    pdt.assert_series_equal(df["vol_2"], expected["vol_2"])
    pdt.assert_series_equal(df["impulse_2"], expected["impulse_2"])
    pdt.assert_series_equal(df["hmax_2"], expected["hmax_2"])
    pdt.assert_series_equal(df["bbm_2"], expected["bbm_2"])
    pdt.assert_series_equal(df["bbs_2"], expected["bbs_2"])
