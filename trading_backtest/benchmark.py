from __future__ import annotations

import pandas as pd

from .utils.io_utils import save_csv
from .config import (
    SUMMARY_FILE,
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
    RandomForestConfig,
)
from .strategy import (
    SMACrossoverStrategy,
    RSIStrategy,
    BreakoutStrategy,
    BollingerBandStrategy,
    MomentumImpulseStrategy,
    VolatilityExpansionStrategy,
    RandomForestStrategy,
)
from .optimize import (
    optimize_with_optuna,
    PARAM_SPACES,
    prune_sma,
    prune_rsi,
    prune_breakout,
    prune_bollinger,
    prune_momentum,
    prune_vol_expansion,
)
from .performance import PerformanceAnalyzer
from .optimize import evaluate_strategy


def benchmark_strategies(
    df: pd.DataFrame, n_trials: int = 300, with_ml: bool = True
) -> pd.DataFrame:
    """Optimize each classical strategy then evaluate on ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Price data used for generating trades.
    n_trials : int, default 50
        Number of Optuna trials used for each strategy.  The caller should
        explicitly pass this value so all optimizations share the same trial
        count.
    with_ml : bool, default True
        Whether to include the machine learning strategy in the benchmark.

    Returns
    -------
    pandas.DataFrame
        Summary of strategy performance sorted by ``total_return``.
    """

    configs = [
        ("SMA", SMACrossoverStrategy, SMAConfig, PARAM_SPACES["sma"], prune_sma),
        ("RSI", RSIStrategy, RSIConfig, PARAM_SPACES["rsi"], prune_rsi),
        (
            "Breakout",
            BreakoutStrategy,
            BreakoutConfig,
            PARAM_SPACES["breakout"],
            prune_breakout,
        ),
        (
            "Bollinger",
            BollingerBandStrategy,
            BollingerConfig,
            PARAM_SPACES["bollinger"],
            prune_bollinger,
        ),
        (
            "Momentum",
            MomentumImpulseStrategy,
            MomentumConfig,
            PARAM_SPACES["momentum"],
            prune_momentum,
        ),
        (
            "VolExpansion",
            VolatilityExpansionStrategy,
            VolExpansionConfig,
            PARAM_SPACES["vol_expansion"],
            prune_vol_expansion,
        ),
    ]

    results = []
    for name, cls, cfg_cls, space, prune in configs:
        trial = optimize_with_optuna(
            df, cls, cfg_cls, space, prune_logic=prune, n_trials=n_trials
        )
        try:
            cfg = cfg_cls(**trial.params)
            ret = evaluate_strategy(df, lambda cfg=cfg: cls(cfg), with_sharpe=True)
        except ValueError:
            ret = 0.0
        results.append({"strategy": name, "score": ret})

    # Machine learning strategy is not optimized here
    if with_ml:
        rf_cfg = RandomForestConfig(
            n_estimators=50, max_depth=None, sl_pct=5, tp_pct=10
        )
        rf = RandomForestStrategy(rf_cfg)
        results.append(
            {
                "strategy": "RandomForest",
                "score": evaluate_strategy(df, lambda: rf, with_sharpe=True),
            }
        )

    summary = pd.DataFrame(results).sort_values("score", ascending=False)
    save_csv(summary, SUMMARY_FILE)
    return summary
