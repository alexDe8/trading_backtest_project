import pandas as pd
from .base import BaseStrategy
from ..config import BreakoutConfig


class BreakoutStrategy(BaseStrategy):
    """Breakout del massimo recente + filtro ATR."""

    def __init__(self, config: BreakoutConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        h_col = f"hmax_{self.config.lookback}"
        if h_col not in df:
            df[h_col] = df["close"].shift(1).rolling(self.config.lookback).max()
        df["h"] = df[h_col]
        df["atr"] = df[f"atr_{self.config.atr_period}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df["h"] + self.config.atr_mult * df["atr"]

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df["h"]
