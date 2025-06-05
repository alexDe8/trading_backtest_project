import pandas as pd
from .base import BaseStrategy
from ..config import MACDConfig


class MACDStrategy(BaseStrategy):
    """Simple MACD crossover strategy."""

    def __init__(self, config: MACDConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = df["close"].ewm(span=self.config.fast, adjust=False).mean().shift(1)
        slow = df["close"].ewm(span=self.config.slow, adjust=False).mean().shift(1)
        df["macd"] = fast - slow
        df["signal"] = (
            df["macd"].ewm(span=self.config.signal, adjust=False).mean().shift(1)
        )
        print(
            f"[DEBUG] MACD fast={self.config.fast}, slow={self.config.slow}, signal={self.config.signal}"
        )
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        macd = df["macd"]
        signal = df["signal"]
        return (macd > signal) & (macd.shift(1) <= signal.shift(1))

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        macd = df["macd"]
        signal = df["signal"]
        return (macd < signal) & (macd.shift(1) >= signal.shift(1))
