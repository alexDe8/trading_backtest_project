import pandas as pd
from .base import BaseStrategy
from ..config import BreakoutConfig, log


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
        atr_col = f"atr_{self.config.atr_period}"
        if atr_col not in df.columns:
            raise KeyError(f"Colonna {atr_col} mancante")
        if df[atr_col].isna().all():
            log.debug(f"Colonna {atr_col} tutta NaN!")
        else:
            log.debug(f"Colonna {atr_col} OK. Stats:\n{df[atr_col].describe()}")
        df["atr"] = df[atr_col]
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df["h"] + self.config.atr_mult * df["atr"]

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df["h"]
