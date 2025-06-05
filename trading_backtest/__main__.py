from __future__ import annotations
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
    refined_sma_grid,
    grid_search,
)
from .performance import PerformanceAnalyzer

from .strategy.sma import SMACrossoverStrategy
from .strategy.rsi import RSIStrategy
from .strategy.breakout import BreakoutStrategy
from .strategy.bollinger import BollingerBandStrategy
from .strategy.momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy


def run_reference_strategy(df: pd.DataFrame, strategy_instance) -> float:
    """Return total return for a given strategy instance."""
    trades = strategy_instance.generate_trades(df)
    return PerformanceAnalyzer(trades).total_return()


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

    # 3) Strategie di riferimento ------------------------------------------
    other = []
    other.append(
        {
            "strategy": "RSI",
            "total_return": run_reference_strategy(df, RSIStrategy(14, 30, 7, 20)),
        }
    )
    other.append(
        {
            "strategy": "Breakout",
            "total_return": run_reference_strategy(
                df, BreakoutStrategy(55, 14, 1.0, 7, 20)
            ),
        }
    )
    vol_thr = df["vol_50"].quantile(0.80)
    other.append(
        {
            "strategy": "VolExpansion",
            "total_return": run_reference_strategy(
                df, VolatilityExpansionStrategy(50, vol_thr, 7, 20)
            ),
        }
    )
    other.append(
        {
            "strategy": "Bollinger",
            "total_return": run_reference_strategy(
                df, BollingerBandStrategy(20, 2.0, 7, 15)
            ),
        }
    )
    other.append(
        {
            "strategy": "Momentum",
            "total_return": run_reference_strategy(
                df, MomentumImpulseStrategy(10, 0.02, 7, 20)
            ),
        }
    )

    summary = pd.DataFrame(other).sort_values("total_return", ascending=False)
    save_csv(summary, SUMMARY_FILE)
    log.info("Riepilogo strategie salvato in %s", SUMMARY_FILE)
    log.info("=== PERFORMANCE ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
