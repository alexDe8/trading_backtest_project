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
}


def create_reference_strategies(df: pd.DataFrame):
    """Return list of (name, strategy_instance) tuples."""
    vol_thr = df["vol_50"].quantile(0.80)
    return [
        (
            "RSI",
            RSIStrategy(RSIConfig(period=14, oversold=30, sl_pct=7, tp_pct=20)),
        ),
        (
            "Breakout",
            BreakoutStrategy(
                BreakoutConfig(
                    lookback=55,
                    atr_period=14,
                    atr_mult=1.0,
                    sl_pct=7,
                    tp_pct=20,
                )
            ),
        ),
        (
            "VolExpansion",
            VolatilityExpansionStrategy(
                VolExpansionConfig(
                    vol_window=50,
                    vol_threshold=vol_thr,
                    sl_pct=7,
                    tp_pct=20,
                )
            ),
        ),
        (
            "Bollinger",
            BollingerBandStrategy(
                BollingerConfig(period=20, nstd=2.0, sl_pct=7, tp_pct=15)
            ),
        ),
        (
            "Momentum",
            MomentumImpulseStrategy(
                MomentumConfig(window=10, threshold=0.02, sl_pct=7, tp_pct=20)
            ),
        ),
    ]


def run_reference_strategy(df: pd.DataFrame, strategy_instance) -> float:
    """Return total return for a given strategy instance."""
    trades = strategy_instance.generate_trades(df)
    return PerformanceAnalyzer(trades).total_return()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy name to optimize (overrides STRATEGY env var)",
    )
    args = parser.parse_args()

    strategy_name = args.strategy or os.getenv("STRATEGY", "sma")
    if strategy_name not in STRATEGY_REGISTRY:
        raise SystemExit(f"Unknown strategy '{strategy_name}'")

    strategy_cls, config_cls, param_space, prune_func = STRATEGY_REGISTRY[strategy_name]

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
        strategy_cls,
        config_cls,
        param_space,
        prune_logic=prune_func,
        n_trials=300,
    )
    if strategy_name == "sma":
        sma_grid = refined_sma_grid(best_trial.params)
        grid_df = grid_search(df, sma_grid)
        save_csv(grid_df, RESULTS_FILE)
        log.info("Grid SMA salvato in %s", RESULTS_FILE)

    # 3) Strategie di riferimento ------------------------------------------
    ref_strategies = create_reference_strategies(df)
    other = [
        {
            "strategy": name,
            "total_return": run_reference_strategy(df, strat),
        }
        for name, strat in ref_strategies
    ]

    summary = pd.DataFrame(other).sort_values("total_return", ascending=False)
    save_csv(summary, SUMMARY_FILE)
    log.info("Riepilogo strategie salvato in %s", SUMMARY_FILE)
    log.info("=== PERFORMANCE ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
