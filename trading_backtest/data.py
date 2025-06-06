# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA_FILE, log
from .utils.io_utils import load_csv


class DataFormatError(Exception):
    """Raised when CSV content does not match expected format."""


def load_price_data(data_file: Path = DATA_FILE) -> pd.DataFrame:
    """Load OHLCV data from ``data_file``.

    Logs and re-raises ``FileNotFoundError`` if the file is missing and
    ``DataFormatError`` if columns are not parseable.
    """

    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        log.error("Data file not found: %s", data_file)
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        log.error("Invalid CSV format for %s: %s", data_file, e)
        raise DataFormatError(str(e)) from e

    try:
        df["timestamp"] = pd.to_datetime(df["Open time"], errors="coerce")
        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        for old, new in rename.items():
            df[new] = pd.to_numeric(df[old], errors="coerce")
        df = (
            df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    except KeyError as e:
        log.error("Missing expected columns in %s: %s", data_file, e)
        raise DataFormatError(str(e)) from e
    return df


def add_indicator_cache(
    df: pd.DataFrame,
    sma: list[int] | None = None,
    rsi: list[int] | None = None,
    atr: list[int] | None = None,
    vol: list[int] | None = None,
    imp: list[int] | None = None,
    hmax: list[int] | None = None,
    bb: list[int] | None = None,
) -> None:
    """Compute and store commonly used indicators in ``df``.

    Parameters mirror the window lengths for simple moving averages, RSI,
    average true range, historical volatility, impulse, breakout highs and
    Bollinger bands. Columns are added in bulk to minimise fragmentation.
    """

    log.info("Caching indicatori …")

    sma = sorted(set(sma or []))
    rsi = sorted(set(rsi or []))
    atr = sorted(set(atr or []))
    vol = sorted(set(vol or []))
    imp = sorted(set(imp or []))
    hmax = sorted(set(hmax or []))
    bb = sorted(set(bb or []))

    cols = {}

    # SMA
    for w in sma:
        cols[f"sma_{w}"] = df["close"].rolling(w).mean().shift(1)

    # RSI - start rolling early to reduce initial NaNs
    if rsi:
        delta = df["close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        for p in rsi:
            gain = up.rolling(p, min_periods=1).mean()
            loss = down.rolling(p, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_vals = 100 - 100 / (1 + rs)
            rsi_vals = rsi_vals.bfill()  # fill leading NaNs
            cols[f"rsi_{p}"] = rsi_vals.shift(1)

    # ATR and true range
    if atr:
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ),
        )
        cols["tr"] = tr
        for p in atr:
            cols[f"atr_{p}"] = tr.rolling(p).mean().shift(1)

    # Historical volatility
    if vol:
        pct = df["close"].pct_change()
        for w in vol:
            cols[f"vol_{w}"] = pct.rolling(w).std().shift(1)

    # Impulse
    for w in imp:
        cols[f"impulse_{w}"] = df["close"].pct_change(w).shift(1)

    # Breakout highs (HMAX)
    for w in hmax:
        cols[f"hmax_{w}"] = df["close"].shift(1).rolling(w).max()

    # Bollinger bands
    for w in bb:
        cols[f"bbm_{w}"] = df["close"].rolling(w).mean().shift(1)
        cols[f"bbs_{w}"] = df["close"].rolling(w).std().shift(1)

    if cols:
        df[list(cols.keys())] = pd.DataFrame(cols, index=df.index)

    log.info("Indicatori pronti.")
