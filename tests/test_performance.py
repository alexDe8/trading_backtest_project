import pandas as pd
from trading_backtest.performance import PerformanceAnalyzer


def test_total_return_with_commission_and_slippage():
    trades = pd.DataFrame({"pct_change": [10.0, -5.0, 2.5]})
    pa = PerformanceAnalyzer(trades, commission=0.1, slippage=0.2)
    expected = (10 - 0.1 - 0.2) + (-5 - 0.1 - 0.2) + (2.5 - 0.1 - 0.2)
    assert pa.total_return() == expected


def test_position_size_weights_returns():
    trades = pd.DataFrame({"pct_change": [10.0, -5.0]})
    pa = PerformanceAnalyzer(trades, position_size=0.5)
    expected = (10 * 0.5) + (-5 * 0.5)
    assert pa.total_return() == expected
