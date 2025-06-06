import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from trading_backtest.optimize import grid_search
from tests.test_optuna_param_spaces import _dummy_df


def test_grid_search_rsi_dataframe():
    df = _dummy_df()
    combos = [
        {"period": 14, "oversold": 30, "sl_pct": 5, "tp_pct": 10},
        {"period": 15, "oversold": 30, "sl_pct": 5, "tp_pct": 10},
    ]
    result = grid_search(df, combos, "rsi")
    assert not result.empty
    assert "total_return" in result.columns
    assert len(result) == 2


def test_cli_grid_search_rsi_output(tmp_path):
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
    env.update(
        {
            "DATA_FILE": str(csv),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }
    )

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "trading_backtest",
            "--strategy",
            "rsi",
            "--trials",
            "1",
        ],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert res.returncode == 0
    assert "KeyError" not in res.stderr
    assert (tmp_path / "results_live.csv").is_file()
