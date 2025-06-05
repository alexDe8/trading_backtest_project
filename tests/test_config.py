import importlib
from pathlib import Path

import trading_backtest.config as cfg


def test_data_file_default(monkeypatch):
    monkeypatch.delenv("DATA_FILE", raising=False)
    importlib.reload(cfg)
    assert cfg.DATA_FILE == Path("data/btc_15m_data_2018_to_2025.csv")
