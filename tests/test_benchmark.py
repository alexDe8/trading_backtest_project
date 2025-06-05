from trading_backtest.benchmark import benchmark_strategies
from tests.test_optuna_param_spaces import _dummy_df


def test_benchmark_sorted():
    df = _dummy_df()
    result = benchmark_strategies(df, n_trials=1)
    assert list(result.columns) == ["strategy", "total_return"]
    values = result["total_return"].tolist()
    assert values == sorted(values, reverse=True)
