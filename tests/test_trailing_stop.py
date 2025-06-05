import os
import sys
import pandas as pd
import pytest

from dataclasses import dataclass
from trading_backtest.strategy.base import BaseStrategy


@dataclass
class DummyConfig:
    sl_pct: float
    tp_pct: float
    trailing_stop_pct: float


class DummyStrategy(BaseStrategy):
    def __init__(self, config: DummyConfig):
        super().__init__(config)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def entry_signal(self, df: pd.DataFrame) -> pd.Series:
        sig = [True] + [False] * (len(df) - 1)
        return pd.Series(sig, index=df.index)

    def exit_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


def test_trailing_stop_closes_trade():
    df = pd.DataFrame(
        [
            {"timestamp": 1, "close": 100, "high": 100, "low": 100},
            {"timestamp": 2, "close": 110, "high": 110, "low": 101},
            {"timestamp": 3, "close": 106, "high": 111, "low": 104},
        ]
    )
    cfg = DummyConfig(sl_pct=0, tp_pct=100, trailing_stop_pct=5)
    strat = DummyStrategy(cfg)
    trades = strat.generate_trades(df)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["qty"] == 1
    assert trade["exit"] == pytest.approx(110 * (1 - 0.05))
    assert trade["entry_time"] == 1
    assert trade["exit_time"] == 3
    assert trade["entry"] == 100


def test_trailing_stop_pct_must_be_positive():
    with pytest.raises(ValueError):
        DummyStrategy(DummyConfig(sl_pct=1, tp_pct=2, trailing_stop_pct=0))
