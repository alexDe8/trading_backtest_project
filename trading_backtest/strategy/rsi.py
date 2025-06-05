import pandas as pd
from .base import BaseStrategy
from ..config import RSIConfig


class RSIStrategy(BaseStrategy):
    """Enter on RSI oversold crosses back above the threshold."""

    def __init__(self, config: RSIConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["r"] = df[f"rsi_{self.config.period}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= self.config.oversold) & (r > self.config.oversold)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        r = df["r"]
        return (r.shift(1) <= 50) & (r > 50)
