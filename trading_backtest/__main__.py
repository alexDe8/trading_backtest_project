from __future__ import annotations
import argparse
from pathlib import Path
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
)
from .data import load_price_data, add_indicator_cache
from .utils.io_utils import save_csv
from .optimize import (
    optimize_with_optuna,
    PARAM_SPACES,
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

from .strategy.sma import SMACrossoverStrategy
from .strategy.rsi import RSIStrategy
from .strategy.breakout import BreakoutStrategy
from .strategy.bollinger import BollingerBandStrategy
from .strategy.momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy


STRATEGY_MAP = {
    "sma": (SMACrossoverStrategy, SMAConfig, prune_sma),
    "rsi": (RSIStrategy, RSIConfig, prune_rsi),
    "breakout": (BreakoutStrategy, BreakoutConfig, prune_breakout),
    "bollinger": (BollingerBandStrategy, BollingerConfig, prune_bollinger),
    "momentum": (MomentumImpulseStrategy, MomentumConfig, prune_momentum),
    "vol_expansion": (
        VolatilityExpansionStrategy,
        VolExpansionConfig,
        prune_vol_expansion,
    ),
}


def run_reference_strategy(df: pd.DataFrame, strategy_instance) -> float:
    """Return total return for a given strategy instance."""
    trades = strategy_instance.generate_trades(df)
    return PerformanceAnalyzer(trades).total_return()


def main(
    data_file: Path = DATA_FILE,
    strategy: str = "sma",
    n_trials: int = 300,
) -> None:
    """Run the optimisation for the chosen strategy."""

    # 1) Dati + indicatori -------------------------------------------------
    df = load_price_data(data_file)
    add_indicator_cache(
        df,
        sma=list(range(5, 251)) + [300, 400],
        rsi=[14],
        atr=[14, 21],
        vol=[20, 50],
        imp=[5, 10],
    )
    # 2) Optuna -------------------------------------------------------------
    strat_cls, cfg_cls, prune_fn = STRATEGY_MAP[strategy]
    best_trial = optimize_with_optuna(
        df,
        strat_cls,
        cfg_cls,
        PARAM_SPACES[strategy],
        prune_logic=prune_fn,
        n_trials=n_trials,
    )

    if strategy == "sma":
        sma_grid = refined_sma_grid(best_trial.params)
        grid_df = grid_search(df, sma_grid)
        save_csv(grid_df, RESULTS_FILE)
        log.info("Grid SMA salvato in %s", RESULTS_FILE)

    log.info("Best parameters for %s: %s", strategy, best_trial.params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Backtest CLI")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Path to CSV with price data",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_MAP.keys(),
        default="sma",
        help="Which strategy to optimise",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Number of Optuna trials",
    )
    args = parser.parse_args()
    main(data_file=args.data, strategy=args.strategy, n_trials=args.trials)
