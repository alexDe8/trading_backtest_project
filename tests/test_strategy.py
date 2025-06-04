import pandas as pd
import pytest

from trading_backtest.strategy.rsi import RSIStrategy
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
    cfg = RSIConfig(period=14, oversold=30, sl_pct=50, tp_pct=50)
    strat = RSIStrategy(cfg)
    trades = strat.generate_trades(df)
    assert len(trades) == 1

    trade = trades.iloc[0]
    assert trade["entry_time"] == df.loc[2, "timestamp"]
    assert trade["exit_time"] == df.loc[4, "timestamp"]
    expected_pct = ((df.loc[2, "close"] * 1.1) / df.loc[2, "close"] - 1) * 100
    assert trade["pct_change"] == expected_pct


def test_rsi_strategy_invalid_sl_tp():
    cfg = RSIConfig(period=14, oversold=30, sl_pct=10, tp_pct=5)
    with pytest.raises(ValueError):
        RSIStrategy(cfg)

