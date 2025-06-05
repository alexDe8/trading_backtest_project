import pandas as pd
from .base import BaseStrategy
from ..config import StochasticConfig


class StochasticStrategy(BaseStrategy):
    """Stochastic oscillator strategy using K/D crosses."""

    def __init__(self, config: StochasticConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        low_n = df["low"].rolling(self.config.k_period).min().shift(1)
        high_n = df["high"].rolling(self.config.k_period).max().shift(1)
        df["k"] = (df["close"] - low_n) / (high_n - low_n) * 100
        df["d"] = df["k"].rolling(self.config.d_period).mean().shift(1)
        print(
            f"[DEBUG] Stochastic k_period={self.config.k_period}, d_period={self.config.d_period}"
        )
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        k = df["k"]
        return (k.shift(1) < self.config.oversold) & (k > self.config.oversold)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        k = df["k"]
        return (k.shift(1) >= 50) & (k < 50)
