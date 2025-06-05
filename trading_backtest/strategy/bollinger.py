import pandas as pd
from .base import BaseStrategy
from ..config import BollingerConfig


class BollingerBandStrategy(BaseStrategy):
    """Mean-reversion strategy based on Bollinger Bands."""

    def __init__(self, config: BollingerConfig):
        super().__init__(config.sl_pct, config.tp_pct)
        self.p, self.n = config.period, config.nstd
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ma = f"bbm_{self.p}"
        sd = f"bbs_{self.p}"
        if ma not in df:
            df[ma] = df["close"].rolling(self.p).mean().shift(1)
            df[sd] = df["close"].rolling(self.p).std().shift(1)
        df["ma"] = df[ma]
        df["lb"] = df[ma] - self.n * df[sd]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df["lb"]

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df["ma"]
