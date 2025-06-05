import pandas as pd
import pytest
from trading_backtest.performance import PerformanceAnalyzer


def test_total_return_with_commission_and_slippage():
    trades = pd.DataFrame({"pct_change": [10.0, -5.0, 2.5]})
    pa = PerformanceAnalyzer(trades, commission=0.1, slippage=0.2)
    expected = (10 - 0.1 - 0.2) + (-5 - 0.1 - 0.2) + (2.5 - 0.1 - 0.2)
    assert pa.total_return() == expected


def test_sharpe_ratio():
    trades = pd.DataFrame({"pct_change": [10.0, -5.0, 15.0]})
    pa = PerformanceAnalyzer(trades)
    r = trades["pct_change"]
    expected = (r.mean() / r.std()) * (len(r) ** 0.5)
    assert pa.sharpe_ratio() == pytest.approx(expected)


def test_max_drawdown_and_win_rate():
    trades = pd.DataFrame({"pct_change": [10, -5, 15, -20, 5]})
    pa = PerformanceAnalyzer(trades)
    equity = (1 + trades["pct_change"] / 100).cumprod()
    dd = equity.div(equity.cummax()).sub(1) * 100
    assert pa.max_drawdown() == dd.min()
    assert pa.win_rate() == (3 / 5) * 100
