import os
import sys
import subprocess
import pandas as pd

from trading_backtest.benchmark import benchmark_strategies
from tests.test_optuna_param_spaces import _dummy_df


def test_benchmark_sorted():
    df = _dummy_df()
    result = benchmark_strategies(df, n_trials=1)
    assert list(result.columns) == ["strategy", "total_return"]
    values = result["total_return"].tolist()
    assert values == sorted(values, reverse=True)


def test_benchmark_without_ml():
    df = _dummy_df()
    result = benchmark_strategies(df, n_trials=1, with_ml=False)
    assert "RandomForest" not in result["strategy"].values


def test_cli_benchmark_runs(tmp_path):
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2020-01-01", periods=50, freq="T"),
            "Open": range(50),
            "High": range(1, 51),
            "Low": range(50),
            "Close": range(50),
            "Volume": range(50),
        }
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    env = os.environ.copy()
    env.update({"DATA_FILE": str(csv), "RUN_ML": "0"})

    res = subprocess.run(
        [sys.executable, "-m", "trading_backtest", "--benchmark", "--trials", "1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert res.returncode == 0
    assert "KeyError" not in res.stderr
