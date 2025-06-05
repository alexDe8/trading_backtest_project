import pandas as pd
from .base import BaseStrategy
from ..config import MomentumConfig, VolExpansionConfig


class VolatilityExpansionStrategy(BaseStrategy):
    """Trade when volatility expands beyond a threshold."""

    def __init__(self, config: VolExpansionConfig):
        super().__init__(config.sl_pct, config.tp_pct)
        self.w, self.th = config.vol_window, config.vol_threshold
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["v"] = df[f"vol_{self.w}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] > self.th

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] < self.th


class MomentumImpulseStrategy(BaseStrategy):
    """Follow short-term price momentum using impulse."""

    def __init__(self, config: MomentumConfig):
        super().__init__(config.sl_pct, config.tp_pct)
        self.w, self.t = config.window, config.threshold
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["imp"] = df[f"impulse_{self.w}"]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] > self.t

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] < 0
