from __future__ import annotations
import pandas as pd

from ..config import log


def validate_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Ensure column exists and log NaN stats."""
    if col not in df.columns:
        raise KeyError(f"Colonna {col} mancante")
    series = df[col]
    if series.isna().all():
        log.debug(f"Colonna {col} tutta NaN!")
    else:
        log.debug(f"Colonna {col} OK. Stats:\n{series.describe()}")
    return series


__all__ = ["validate_column"]
