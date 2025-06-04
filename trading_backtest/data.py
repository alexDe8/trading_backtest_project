# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA_FILE, log


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
    sma: list[int],
    rsi: list[int],
    atr: list[int],
    vol: list[int],
    imp: list[int],
) -> None:
    log.info("Caching indicatori …")

    # SMA
    for w in sma:
        df[f"sma_{w}"] = df["close"].rolling(w).mean().shift(1)

    # RSI
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    for p in rsi:
        rs = up.rolling(p).mean() / down.replace(0, np.nan).rolling(p).mean()
        df[f"rsi_{p}"] = (100 - 100 / (1 + rs)).shift(1)

    # ATR
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift()),
        ),
    )
    df["tr"] = tr
    for p in atr:
        df[f"atr_{p}"] = tr.rolling(p).mean().shift(1)

    # Volatilità storica
    for w in vol:
        df[f"vol_{w}"] = df["close"].pct_change().rolling(w).std().shift(1)

    # Impulso
    for w in imp:
        df[f"impulse_{w}"] = df["close"].pct_change(w).shift(1)

    log.info("Indicatori pronti.")
