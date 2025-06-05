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
            n_estimators=config.n_estimators,
            max_depth=getattr(config, "max_depth", None),
            random_state=42,
        )

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Usa tutte le feature tecniche se ci sono, altrimenti fallback su ret_1
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(("sma_", "rsi_", "atr_", "vol_", "impulse_"))
        ]
        if not feature_cols:
            # fallback: usa il return a 1 periodo
            df["ret_1"] = df["close"].pct_change().fillna(0)
            feature_cols = ["ret_1"]

        X = df[feature_cols].bfill().ffill().fillna(0)
        print(f"[DEBUG] RandomForest feature cols: {feature_cols}")
        y = (df["close"].shift(-1) > df["close"]).astype(int)

        split = int(len(df) * 0.7)
        if split > 0:
            self.model.fit(X.iloc[:split], y.iloc[:split])
            probs = self.model.predict_proba(X)
            # Usa la probabilità della classe "1" (up)
            prob_col = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            df["rf_prob"] = prob_col
        else:
            df["rf_prob"] = 0.0
        print(
            f"[DEBUG] RF params n_estimators={self.config.n_estimators}, max_depth={self.config.max_depth}"
        )
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["rf_prob"] > getattr(self.config, "entry_threshold", 0.55)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["rf_prob"] < getattr(self.config, "exit_threshold", 0.45)
