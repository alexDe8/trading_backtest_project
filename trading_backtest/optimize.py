# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable
import pandas as pd
import optuna
from tqdm import tqdm
from .performance import PerformanceAnalyzer
from .config import (
    log,
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
)

# ---------------------- PARAMETRI STRATEGIE --------------------------
PARAM_SPACES = {
    "sma": {
        "sma_fast": ("int", 5, 50, 5),
        "sma_slow": ("int", 100, 250, 5),
        "sma_trend": ("cat", [None, 200, 300, 400]),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 15, 25, 5),
        "position_size": ("float", 0.01, 0.2),
        "trailing_stop_pct": ("float", 0.5, 10.0),
    },
    "rsi": {
        "period": ("int", 7, 21, 1),
        "oversold": ("int", 20, 40, 5),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 10, 25, 5),
    },
    "breakout": {
        "lookback": ("int", 20, 100, 5),
        "atr_period": ("int", 7, 21, 1),
        "atr_mult": ("float", 0.5, 2.0),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 10, 25, 5),
    },
    "bollinger": {
        "period": ("int", 10, 30, 2),
        "nstd": ("float", 1.5, 3.0, 0.1),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 10, 25, 5),
    },
    "momentum": {
        "window": ("int", 5, 20, 1),
        "threshold": ("float", 0.01, 0.05, 0.01),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 10, 25, 5),
    },
    "vol_expansion": {
        "vol_window": ("int", 20, 100, 5),
        "vol_threshold": ("float", 0.6, 1.0, 0.05),
        "sl_pct": ("int", 5, 10),
        "tp_pct": ("int", 10, 25, 5),
    },
}


# ---------------------- SUGGEST UNIVERSALE ---------------------------
def suggest(trial, param_info, name=None):
    t, *args = param_info
    if name is None:
        raise ValueError("Parametro 'name' mancante in suggest()!")
    log.debug("[DEBUG SUGGEST] %s: tipo=%s, args=%s", name, t, args)
    if t == "int":
        low, high, *rest = args
        if low > high:
            raise ValueError(
                f"[ERROR] Parametro INT range invertito: {name} low={low}, high={high}"
            )
        if rest:
            return trial.suggest_int(name=name, low=low, high=high, step=rest[0])
        else:
            return trial.suggest_int(name=name, low=low, high=high)
    elif t == "float":
        low, high, *rest = args
        if low > high:
            raise ValueError(
                f"[ERROR] Parametro FLOAT range invertito: {name} low={low}, high={high}"
            )
        return trial.suggest_float(name=name, low=low, high=high)
    elif t == "cat":
        categories = args[0]
        return trial.suggest_categorical(name=name, choices=categories)
    else:
        raise ValueError(f"[ERROR] Tipo di parametro non gestito: {t}")


# ---------------------- PRUNE -----------------------------
def prune_sma(params, trial):
    if params["sma_fast"] >= params["sma_slow"]:
        raise optuna.TrialPruned()
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_rsi(params, trial):
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_breakout(params, trial):
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_bollinger(params, trial):
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_momentum(params, trial):
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_vol_expansion(params, trial):
    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


# ---------------------- STRATEGY EVALUATION -----------------------------
def evaluate_strategy(df: pd.DataFrame, make_strategy: Callable[[], Any]) -> float:
    strat = make_strategy()
    trades = strat.generate_trades(df)
    return PerformanceAnalyzer(trades, commission=0.1, slippage=0.05).total_return()


# ---------------------- OBJECTIVE GENERICO ---------------------------
def make_objective(
    df: pd.DataFrame,
    strategy_cls,
    config_cls,
    param_space,
    prune_logic=None,
):
    def objective(trial):
        params = {
            name: suggest(trial, info, name=name) for name, info in param_space.items()
        }
        if prune_logic is not None:
            prune_logic(params, trial)
        config = config_cls(**params)
        return evaluate_strategy(df, lambda: strategy_cls(config))

    return objective


# ---------------------- OPTIMIZZA GENERICO ---------------------------
def optimize_with_optuna(
    df: pd.DataFrame,
    strategy_cls,
    config_cls,
    param_space,
    prune_logic=None,
    n_trials: int = 300,
) -> optuna.FrozenTrial:
    study = optuna.create_study(direction="maximize")
    objective = make_objective(df, strategy_cls, config_cls, param_space, prune_logic)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    try:
        trial = study.best_trial
        log.info("🏆 Best params: %s (%.2f%%)", study.best_params, study.best_value)
        return trial
    except ValueError:
        # No completed trials, likely because the only trial was pruned.
        log.warning("No completed trials found; returning last trial")
        return study.trials[-1]


# ---------------------- RETROCOMPATIBILITA' SMA ----------------------
from .strategy.sma import SMACrossoverStrategy


def optimize_sma(df: pd.DataFrame, n_trials: int = 300):
    return optimize_with_optuna(
        df,
        SMACrossoverStrategy,
        SMAConfig,
        PARAM_SPACES["sma"],
        prune_logic=prune_sma,
        n_trials=n_trials,
    )


# ---------------------- GRIGLIA RAFFINATA ---------------------------
from itertools import product


def _around(val: int, step: int, n: int = 2) -> list[int]:
    return [val + i * step for i in range(-n, n + 1) if val + i * step > 0]


def refined_sma_grid(best: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for f, s, sl, tp in product(
        _around(best["sma_fast"], 2),
        _around(best["sma_slow"], 5),
        _around(best["sl_pct"], 1, 1),
        _around(best["tp_pct"], 5, 1),
    ):
        if f >= s or sl >= tp:
            continue
        grid.append(
            {
                "sma_fast": f,
                "sma_slow": s,
                "sma_trend": best["sma_trend"],
                "sl_pct": sl,
                "tp_pct": tp,
                "position_size": best.get("position_size", 0.1),
                "trailing_stop_pct": best.get("trailing_stop_pct", 1.0),
            }
        )
    return grid


# ---------------------- GRID SEARCH ---------------------------
def grid_search(df: pd.DataFrame, combos: list[dict[str, Any]]) -> pd.DataFrame:
    log.info("Grid SMA – %d combo", len(combos))
    results = []
    for p in tqdm(combos, desc="SMA"):
        cfg = SMAConfig(**p)
        ret = evaluate_strategy(df, lambda cfg=cfg: SMACrossoverStrategy(cfg))
        results.append({**p, "total_return": ret})
    return pd.DataFrame(results).sort_values("total_return", ascending=False)
