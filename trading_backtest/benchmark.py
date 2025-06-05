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


def benchmark_strategies(
    df: pd.DataFrame, n_trials: int = 50, with_ml: bool = True
) -> pd.DataFrame:
    """Optimize each classical strategy then evaluate on ``df``.

    The resulting DataFrame is sorted by ``total_return``.
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
        cfg = cfg_cls(**trial.params)
        trades = cls(cfg).generate_trades(df)
        ret = PerformanceAnalyzer(trades).total_return()
        results.append({"strategy": name, "total_return": ret})

    # Machine learning strategy is not optimized here
    if with_ml:
        rf_cfg = RandomForestConfig(
            n_estimators=50, max_depth=None, sl_pct=5, tp_pct=10
        )
        rf = RandomForestStrategy(rf_cfg)
        trades = rf.generate_trades(df)
        results.append(
            {
                "strategy": "RandomForest",
                "total_return": PerformanceAnalyzer(trades).total_return(),
            }
        )

    summary = pd.DataFrame(results).sort_values("total_return", ascending=False)
    save_csv(summary, SUMMARY_FILE)
    return summary
