from __future__ import annotations
import pandas as pd

from .config import (
    RESULTS_FILE,
    SUMMARY_FILE,
    DATA_FILE,
    log,
    SMAConfig,
)
from .data import load_price_data, add_indicator_cache
from .utils.io_utils import save_csv
from .optimize import (
    optimize_with_optuna,
    PARAM_SPACES,
    prune_sma,
    refined_sma_grid,
    grid_search,
)
from .benchmark import benchmark_strategies
from .strategy.sma import SMACrossoverStrategy


def main() -> None:
    # 1) Dati + indicatori -------------------------------------------------
    df = load_price_data(DATA_FILE)
    add_indicator_cache(
        df,
        sma=list(range(5, 251)) + [300, 400],
        rsi=[14],
        atr=[14, 21],
        vol=[20, 50],
        imp=[5, 10],
    )
    # 2) Optuna (modulare!) ------------------------------------------------
    best_trial = optimize_with_optuna(
        df,
        SMACrossoverStrategy,
        SMAConfig,
        PARAM_SPACES["sma"],
        prune_logic=prune_sma,
        n_trials=300,
    )
    sma_grid = refined_sma_grid(best_trial.params)
    grid_df = grid_search(df, sma_grid)
    save_csv(grid_df, RESULTS_FILE)
    log.info("Grid SMA salvato in %s", RESULTS_FILE)

    # 3) Benchmark completo -----------------------------------------------
    summary = benchmark_strategies(df, n_trials=50)
    log.info("Riepilogo strategie salvato in %s", SUMMARY_FILE)
    log.info("=== PERFORMANCE ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
