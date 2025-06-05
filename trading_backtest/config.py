import os
from pathlib import Path
import logging

# Usa variabile d'ambiente se disponibile, altrimenti path relativo al progetto
DATA_FILE = Path(os.environ.get("DATA_FILE", "data/btc_15m_data_2018_to_2025.csv"))
RESULTS_FILE = Path("results_live.csv")
SUMMARY_FILE = Path("summary_live.csv")

level_name = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s | %(message)s",
    force=True,
)
log = logging.getLogger("trading_backtest")

from dataclasses import dataclass
from typing import Optional


@dataclass
class SMAConfig:
    sma_fast: int
    sma_slow: int
    sma_trend: Optional[int]
    sl_pct: float
    tp_pct: float
    position_size: float
    trailing_stop_pct: float


@dataclass
class RSIConfig:
    period: int
    oversold: int
    sl_pct: float
    tp_pct: float


@dataclass
class BreakoutConfig:
    lookback: int
    atr_period: int
    atr_mult: float
    sl_pct: float
    tp_pct: float


@dataclass
class BollingerConfig:
    period: int
    nstd: float
    sl_pct: float
    tp_pct: float


@dataclass
class MomentumConfig:
    window: int
    threshold: float
    sl_pct: float
    tp_pct: float


@dataclass
class VolExpansionConfig:
    vol_window: int
    vol_threshold: float
    sl_pct: float
    tp_pct: float


@dataclass
class RandomForestConfig:
    entry_threshold: float
    exit_threshold: float
    sl_pct: float
    tp_pct: float
    n_estimators: int = 100
