from __future__ import annotations
import pandas as pd
from .base import BaseStrategy
from ..config import SMAConfig


class SMACrossoverStrategy(BaseStrategy):
    """Classic fast/slow moving average crossover strategy."""

    def __init__(self, config: SMAConfig) -> None:
        super().__init__(config)
        self.position_size = config.position_size
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["f"] = df[f"sma_{self.config.sma_fast}"]
        df["s"] = df[f"sma_{self.config.sma_slow}"]
        if self.config.sma_trend:
            df["t"] = df[f"sma_{self.config.sma_trend}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        cross_up = (df["f"] > df["s"]) & (df["f"].shift(1) <= df["s"].shift(1))
        return cross_up & (df["close"] > df["t"]) if self.config.sma_trend else cross_up

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["f"] < df["s"]
