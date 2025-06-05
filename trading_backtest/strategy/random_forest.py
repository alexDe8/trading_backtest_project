from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import BaseStrategy
from ..config import RandomForestConfig


class RandomForestStrategy(BaseStrategy):
    """Machine learning strategy using RandomForest predictions."""

    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators, random_state=42
        )

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(("sma_", "rsi_", "atr_", "vol_", "impulse_"))
        ]
        if not feature_cols:
            raise ValueError("No indicator features found for training")

        X = df[feature_cols].fillna(method="bfill").fillna(method="ffill").fillna(0)
        y = (df["close"].shift(-1) > df["close"]).astype(int)

        mask = y.notna()
        self.model.fit(X[mask][:-1], y[mask][:-1])

        df["rf_prob"] = 0.0
        probs = self.model.predict_proba(X[mask])
        prob_col = probs[:, 0]
        if probs.shape[1] > 1:
            prob_col = probs[:, 1]
        df.loc[mask, "rf_prob"] = prob_col
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["rf_prob"] > self.config.entry_threshold

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["rf_prob"] < self.config.exit_threshold
