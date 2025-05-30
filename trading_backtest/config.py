# -*- coding: utf-8 -*-
from pathlib import Path
import logging

DATA_FILE    = Path("/content/btc_15m_data_2018_to_2025.csv")
RESULTS_FILE = Path("results_live.csv")
SUMMARY_FILE = Path("summary_live.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    force=True,
)
log = logging.getLogger("trading_backtest")
