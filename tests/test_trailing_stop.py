import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trading_backtest.strategy.base import BaseStrategy


class DummyStrategy(BaseStrategy):
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
    strat = DummyStrategy(sl_pct=0, tp_pct=100, trailing_stop_pct=5)
    trades = strat.generate_trades(df)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["qty"] == 1
    assert trade["exit"] == pytest.approx(110 * (1 - 0.05))

def test_trailing_stop_pct_must_be_positive():
    with pytest.raises(ValueError):
        DummyStrategy(sl_pct=1, tp_pct=2, trailing_stop_pct=0)

