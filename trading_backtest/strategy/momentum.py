import pandas as pd
from .base import BaseStrategy


class VolatilityExpansionStrategy(BaseStrategy):
    def __init__(
        self,
        vol_window: int,
        vol_threshold: float,
        sl_pct: float,
        tp_pct: float,
        position_size: float = 1.0,
    ):
        super().__init__(sl_pct, tp_pct, position_size=position_size)
        self.w, self.th = vol_window, vol_threshold

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["v"] = df[f"vol_{self.w}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] > self.th

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] < self.th


class MomentumImpulseStrategy(BaseStrategy):
    def __init__(
        self,
        window: int,
        threshold: float,
        sl_pct: float,
        tp_pct: float,
        position_size: float = 1.0,
    ):
        super().__init__(sl_pct, tp_pct, position_size=position_size)
        self.w, self.t = window, threshold

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["imp"] = df[f"impulse_{self.w}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] > self.t

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] < 0
