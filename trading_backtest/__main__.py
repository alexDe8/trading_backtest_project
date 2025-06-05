from __future__ import annotations
import os
import argparse
import pandas as pd

from .config import (
    RESULTS_FILE,
    SUMMARY_FILE,
    DATA_FILE,
    log,
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
    RandomForestConfig,
)
from .data import load_price_data, add_indicator_cache
from .utils.io_utils import save_csv
from .optimize import (
    optimize_with_optuna,
    PARAM_SPACES,
    gather_indicator_periods,
    prune_sma,
    prune_rsi,
    prune_breakout,
    prune_bollinger,
    prune_momentum,
    prune_vol_expansion,
    refined_sma_grid,
    grid_search,
)
from .performance import PerformanceAnalyzer
from .benchmark import benchmark_strategies

from .strategy.sma import SMACrossoverStrategy
from .strategy.rsi import RSIStrategy
from .strategy.breakout import BreakoutStrategy
from .strategy.bollinger import BollingerBandStrategy
from .strategy.momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy
from .strategy.random_forest import RandomForestStrategy

# Registry dinamico per CLI
STRATEGY_REGISTRY = {
    "sma": (
        SMACrossoverStrategy,
        SMAConfig,
        PARAM_SPACES["sma"],
        prune_sma,
    ),
    "rsi": (
        RSIStrategy,
        RSIConfig,
        PARAM_SPACES["rsi"],
        prune_rsi,
    ),
    "breakout": (
        BreakoutStrategy,
        BreakoutConfig,
        PARAM_SPACES["breakout"],
        prune_breakout,
    ),
    "bollinger": (
        BollingerBandStrategy,
        BollingerConfig,
        PARAM_SPACES["bollinger"],
        prune_bollinger,
    ),
    "momentum": (
        MomentumImpulseStrategy,
        MomentumConfig,
        PARAM_SPACES["momentum"],
        prune_momentum,
    ),
    "vol_expansion": (
        VolatilityExpansionStrategy,
        VolExpansionConfig,
        PARAM_SPACES["vol_expansion"],
        prune_vol_expansion,
    ),
    "random_forest": (
        RandomForestStrategy,
        RandomForestConfig,
        PARAM_SPACES.get("random_forest", {}),
        lambda params: None,  # Prune function dummy per ML, aggiorna se ne hai una
    ),
}


def main(with_ml: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy name to optimize (overrides STRATEGY env var)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("TRIALS", 50)),
        help="Number of Optuna trials (env TRIALS)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=os.getenv("BENCHMARK", "0") == "1",
        help="Run benchmark for all strategies (env BENCHMARK=1)",
    )
    args = parser.parse_args()

    n_trials = args.trials
    strategy_name = args.strategy or os.getenv("STRATEGY", "sma")

    if strategy_name not in STRATEGY_REGISTRY:
        raise SystemExit(f"Unknown strategy '{strategy_name}'")

    strategy_cls, config_cls, param_space, prune_func = STRATEGY_REGISTRY[strategy_name]

    # 1) Dati + indicatori -------------------------------------------------
    df = load_price_data(DATA_FILE)
    periods = gather_indicator_periods(strategy_name)
    add_indicator_cache(
        df,
        sma=periods.get("sma", []),
        rsi=periods.get("rsi", []),
        atr=periods.get("atr", []),
        vol=periods.get("vol", []),
        imp=periods.get("imp", []),
        hmax=periods.get("hmax", []),
        bb=periods.get("bb", []),
    )

    # 2) Ottimizzazione singola o benchmark -------------------------------
    if not args.benchmark:
        best_trial = optimize_with_optuna(
            df,
            strategy_cls,
            config_cls,
            param_space,
            prune_logic=prune_func,
            n_trials=n_trials,
        )

        if strategy_name == "sma":
            sma_grid = refined_sma_grid(best_trial.params)
            grid_df = grid_search(df, sma_grid)
            save_csv(grid_df, RESULTS_FILE)
            log.info("Grid SMA salvato in %s", RESULTS_FILE)

    # 3) Benchmark completo: classiche + ML -------------------------------
    if args.benchmark:
        summary = benchmark_strategies(df, n_trials=n_trials, with_ml=with_ml)
        log.info("Riepilogo strategie salvato in %s", SUMMARY_FILE)
        log.info("=== PERFORMANCE ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    run_ml = os.getenv("RUN_ML", "0") == "1"
    main(with_ml=run_ml)

