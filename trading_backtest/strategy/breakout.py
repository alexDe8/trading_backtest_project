import pandas as pd
from .base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Breakout del massimo recente + filtro ATR."""

    def __init__(
        self,
        lookback: int,
        atr_period: int,
        atr_mult: float,
        sl_pct: float,
        tp_pct: float,
    ):
        super().__init__(sl_pct, tp_pct)
        self.lb, self.ap, self.m = lookback, atr_period, atr_mult

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        h_col = f"hmax_{self.lb}"
        if h_col not in df:
            df[h_col] = df["close"].shift(1).rolling(self.lb).max()
        df["h"] = df[h_col]
        df["atr"] = df[f"atr_{self.ap}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df["h"] + self.m * df["atr"]

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df["h"]
