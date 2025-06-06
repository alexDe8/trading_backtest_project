import os
import sys
import subprocess
from pathlib import Path

import pandas as pd


def test_cli_best_params_saved(tmp_path):
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
    env["DATA_FILE"] = str(csv)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "trading_backtest",
            "--strategy",
            "sma",
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
    best_file = tmp_path / "best_params.csv"
    assert best_file.exists()
    saved = pd.read_csv(best_file)
    assert list(saved["strategy"]) == ["sma"]
