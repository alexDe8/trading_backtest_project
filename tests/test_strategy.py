import pandas as pd
import pytest

from trading_backtest.strategy import get_strategy
from trading_backtest.config import RSIConfig


def test_rsi_strategy_generate_single_trade():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open": [10, 11, 12, 13, 14],
            "high": [10, 11, 12, 13, 14],
            "low": [10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 14],
            "rsi_14": [25, 28, 35, 45, 55],
        }
    )
    cfg = RSIConfig(period=14, oversold=30, sl_pct=40, tp_pct=50)
    strategy_cls, _ = get_strategy("rsi")
    strat = strategy_cls(cfg)
    trades = strat.generate_trades(df)
    assert len(trades) == 1

    trade = trades.iloc[0]
    assert trade["qty"] == 1
    assert trade["entry_time"] == df.loc[2, "timestamp"]
    assert trade["exit_time"] == df.loc[4, "timestamp"]
    assert trade["entry"] == df.loc[2, "close"]
    assert trade["exit"] == df.loc[4, "close"]
    expected_pct = (df.loc[4, "close"] / df.loc[2, "close"] - 1) * 100
    assert trade["pct_change"] == expected_pct


def test_rsi_strategy_invalid_sl_tp():
    cfg = RSIConfig(period=14, oversold=30, sl_pct=10, tp_pct=5)
    strategy_cls, _ = get_strategy("rsi")
    with pytest.raises(ValueError):
        strategy_cls(cfg)
