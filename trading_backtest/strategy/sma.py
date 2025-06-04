from __future__ import annotations
import pandas as pd
from .base import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    def __init__(
        self,
        sma_fast: int,
        sma_slow: int,
        sma_trend: int | None,
        sl_pct: float,
        tp_pct: float,
        position_size: int,
        trailing_stop_pct: float,
    ) -> None:
        super().__init__(sl_pct, tp_pct, trailing_stop_pct)
        self.f, self.s, self.tr = sma_fast, sma_slow, sma_trend
        self.position_size = position_size

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

    # Nuova logica per il trailing stop
    def trailing_stop(self, entry_price: float, current_price: float) -> float:
        trailing_stop_price = entry_price * (1 - self.trailing_stop_pct / 100)
        return trailing_stop_price if current_price > entry_price else entry_price
