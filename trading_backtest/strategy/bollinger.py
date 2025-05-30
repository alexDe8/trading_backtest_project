import pandas as pd
from .base import BaseStrategy

class BollingerBandStrategy(BaseStrategy):
    def __init__(self, period:int, nstd:float, sl_pct:float, tp_pct:float):
        super().__init__(sl_pct, tp_pct)
        self.p, self.n = period, nstd

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
