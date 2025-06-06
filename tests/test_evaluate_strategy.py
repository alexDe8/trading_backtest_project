import pytest
from trading_backtest.optimize import evaluate_strategy
from trading_backtest.strategy import get_strategy
from trading_backtest.config import SMAConfig
from trading_backtest.performance import PerformanceAnalyzer
from tests.test_optuna_param_spaces import _dummy_df


def test_evaluate_strategy_with_sharpe():
    df = _dummy_df()
    strategy_cls, _ = get_strategy("sma")
    cfg = SMAConfig(
        sma_fast=5,
        sma_slow=10,
        sma_trend=None,
        sl_pct=1,
        tp_pct=2,
        position_size=1,
        trailing_stop_pct=1,
    )
    score = evaluate_strategy(df, lambda: strategy_cls(cfg), with_sharpe=True)

    trades = strategy_cls(cfg).generate_trades(df)
    pa = PerformanceAnalyzer(trades, commission=0.1, slippage=0.05)
    expected = pa.total_return() + pa.sharpe_ratio()
    assert score == pytest.approx(expected)
