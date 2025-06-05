from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import BaseStrategy
from ..config import RandomForestConfig


class RandomForestStrategy(BaseStrategy):
    """Simple ML-based strategy using a random forest classifier."""

    def __init__(self, config: RandomForestConfig) -> None:
        super().__init__(config)
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=42,
        )
        self.config = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # basic one-period returns
        df = df.copy()
        df["ret_1"] = df["close"].pct_change().fillna(0)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        # training on 70% of the data
        split = int(len(df) * 0.7)
        X = df[["ret_1"]]
        y = df["target"]
        if split > 0:
            self.model.fit(X.iloc[:split], y.iloc[:split])
            df["pred"] = self.model.predict(X)
        else:
            df["pred"] = 0
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["pred"] > 0

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["pred"] == 0
