from __future__ import annotations
import pandas as pd
from .base import BaseStrategy
from ..config import SMAConfig, log


class SMACrossoverStrategy(BaseStrategy):
    """Classic fast/slow moving average crossover strategy."""

    def __init__(self, config: SMAConfig) -> None:
        super().__init__(config)
        self.position_size = config.position_size
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast_col = f"sma_{self.config.sma_fast}"
        slow_col = f"sma_{self.config.sma_slow}"
        for col in [fast_col, slow_col]:
            if col not in df.columns:
                raise KeyError(f"Colonna {col} mancante")
            if df[col].isna().all():
                log.debug(f"Colonna {col} tutta NaN!")
            else:
                log.debug(f"Colonna {col} OK. Stats:\n{df[col].describe()}")
        df["f"] = df[fast_col]
        df["s"] = df[slow_col]
        if self.config.sma_trend:
            trend_col = f"sma_{self.config.sma_trend}"
            if trend_col not in df.columns:
                raise KeyError(f"Colonna {trend_col} mancante")
            if df[trend_col].isna().all():
                log.debug(f"Colonna {trend_col} tutta NaN!")
            else:
                log.debug(f"Colonna {trend_col} OK. Stats:\n{df[trend_col].describe()}")
            df["t"] = df[trend_col]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        cross_up = (df["f"] > df["s"]) & (df["f"].shift(1) <= df["s"].shift(1))
        return cross_up & (df["close"] > df["t"]) if self.config.sma_trend else cross_up

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["f"] < df["s"]
