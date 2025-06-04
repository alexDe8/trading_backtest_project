from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(Path(path))


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV without index."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
