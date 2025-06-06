import pandas as pd
from .base import BaseStrategy
from ..config import MomentumConfig, VolExpansionConfig
from ..utils import validate_column


class VolatilityExpansionStrategy(BaseStrategy):
    """Trade when volatility expands beyond a threshold."""

    def __init__(self, config: VolExpansionConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"vol_{self.config.vol_window}"
        df["v"] = validate_column(df, col)
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] > self.config.vol_threshold

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["v"] < self.config.vol_threshold


class MomentumImpulseStrategy(BaseStrategy):
    """Follow short-term price momentum using impulse."""

    def __init__(self, config: MomentumConfig):
        super().__init__(config)
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"impulse_{self.config.window}"
        df["imp"] = validate_column(df, col)
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] > self.config.threshold

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["imp"] < 0
