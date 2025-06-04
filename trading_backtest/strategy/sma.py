from __future__ import annotations
import pandas as pd
from .base import BaseStrategy
from ..config import SMAConfig


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, config: SMAConfig) -> None:
        super().__init__(config.sl_pct, config.tp_pct, config.trailing_stop_pct)
        self.f, self.s, self.tr = config.sma_fast, config.sma_slow, config.sma_trend
        self.position_size = config.position_size
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["f"] = df[f"sma_{self.f}"]
        df["s"] = df[f"sma_{self.s}"]
        if self.tr:
            df["t"] = df[f"sma_{self.tr}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        cross_up = (df["f"] > df["s"]) & (df["f"].shift(1) <= df["s"].shift(1))
        return cross_up & (df["close"] > df["t"]) if self.tr else cross_up

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["f"] < df["s"]
