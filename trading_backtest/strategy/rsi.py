import pandas as pd
from .base import BaseStrategy
from ..config import RSIConfig


class RSIStrategy(BaseStrategy):
    def __init__(self, config: RSIConfig):
        super().__init__(config.sl_pct, config.tp_pct)
        self.p, self.ov = config.period, config.oversold
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["r"] = df[f"rsi_{self.p}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= self.ov) & (r > self.ov)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= 50) & (r > 50)
