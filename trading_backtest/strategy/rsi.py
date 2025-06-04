import pandas as pd
from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(
        self,
        period: int,
        oversold: int,
        sl_pct: float,
        tp_pct: float,
        position_size: float = 1.0,
    ):
        super().__init__(sl_pct, tp_pct, position_size=position_size)
        self.p, self.ov = period, oversold

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["r"] = df[f"rsi_{self.p}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= self.ov) & (r > self.ov)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= 50) & (r > 50)
