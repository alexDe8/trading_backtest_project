# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable
from dataclasses import dataclass, fields
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
    MACDConfig,
    StochasticConfig,
)


@dataclass
class ParamSpace:
    """Base class for parameter ranges."""

    def suggest(self, trial) -> dict[str, Any]:
        params = {}
        for f in fields(self):
            info = getattr(self, f.name)
            params[f.name] = suggest(trial, info, name=f.name)
        return params


@dataclass
class SMAParamSpace(ParamSpace):
    sma_fast: tuple = ("int", 5, 50, 5)
    sma_slow: tuple = ("int", 100, 250, 5)
    sma_trend: tuple = ("cat", [None, 200, 300, 400])
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 15, 25, 5)
    position_size: tuple = ("float", 0.01, 0.2)
    trailing_stop_pct: tuple = ("float", 0.5, 10.0)


@dataclass
class RSIParamSpace(ParamSpace):
    period: tuple = ("int", 7, 21, 1)
    oversold: tuple = ("int", 20, 40, 5)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class BreakoutParamSpace(ParamSpace):
    lookback: tuple = ("int", 20, 100, 5)
    atr_period: tuple = ("int", 7, 21, 1)
    atr_mult: tuple = ("float", 0.5, 2.0)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class BollingerParamSpace(ParamSpace):
    period: tuple = ("int", 10, 30, 2)
    nstd: tuple = ("float", 1.5, 3.0, 0.1)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class MomentumParamSpace(ParamSpace):
    window: tuple = ("int", 5, 20, 1)
    threshold: tuple = ("float", 0.01, 0.05, 0.01)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class VolExpansionParamSpace(ParamSpace):
    vol_window: tuple = ("int", 20, 100, 5)
    vol_threshold: tuple = ("float", 0.6, 1.0, 0.05)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class MACDParamSpace(ParamSpace):
    fast: tuple = ("int", 5, 20, 1)
    slow: tuple = ("int", 21, 50, 1)
    signal: tuple = ("int", 5, 20, 1)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


@dataclass
class StochasticParamSpace(ParamSpace):
    k_period: tuple = ("int", 5, 30, 1)
    d_period: tuple = ("int", 3, 10, 1)
    oversold: tuple = ("int", 20, 40, 5)
    sl_pct: tuple = ("int", 5, 10)
    tp_pct: tuple = ("int", 10, 25, 5)


# ---------------------- PARAMETRI STRATEGIE --------------------------
PARAM_SPACES = {
    "sma": SMAParamSpace(),
    "rsi": RSIParamSpace(),
    "breakout": BreakoutParamSpace(),
    "bollinger": BollingerParamSpace(),
    "momentum": MomentumParamSpace(),
    "vol_expansion": VolExpansionParamSpace(),
    "macd": MACDParamSpace(),
    "stochastic": StochasticParamSpace(),
}


def _int_values(param: tuple) -> list[int]:
    """Return all integer values represented by ``param``."""

    t, low, high, *rest = param
    if t != "int":
        return []
    step = rest[0] if rest else 1
    return list(range(low, high + 1, step))


def gather_indicator_periods(strategy_name: str) -> dict[str, list[int]]:
    """Return indicator windows required by the strategy's parameter space."""

    ps = PARAM_SPACES[strategy_name]
    res: dict[str, set[int]] = {
        "sma": set(),
        "rsi": set(),
        "atr": set(),
        "vol": set(),
        "imp": set(),
        "hmax": set(),
        "bb": set(),
    }

    if strategy_name == "sma":
        res["sma"].update(_int_values(ps.sma_fast))
        res["sma"].update(_int_values(ps.sma_slow))
        res["sma"].update(v for v in ps.sma_trend[1] if isinstance(v, int))
    elif strategy_name == "rsi":
        res["rsi"].update(_int_values(ps.period))
    elif strategy_name == "breakout":
        res["hmax"].update(_int_values(ps.lookback))
        res["atr"].update(_int_values(ps.atr_period))
    elif strategy_name == "bollinger":
        res["bb"].update(_int_values(ps.period))
    elif strategy_name == "momentum":
        res["imp"].update(_int_values(ps.window))
    elif strategy_name == "vol_expansion":
        res["vol"].update(_int_values(ps.vol_window))

    return {k: sorted(v) for k, v in res.items() if v}


# ---------------------- SUGGEST UNIVERSALE ---------------------------
def suggest(trial, param_info, name=None):
    """Wrapper around Optuna suggest functions with basic validation."""

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
def check_sl_tp(params: dict[str, Any]) -> None:
    """Raise :class:`optuna.TrialPruned` when stop loss is not less than take profit."""

    if params["sl_pct"] >= params["tp_pct"]:
        raise optuna.TrialPruned()


def prune_sma(params, trial):
    """Prune trials where SMA parameters are inconsistent."""
    check_sl_tp(params)
    if params["sma_fast"] >= params["sma_slow"]:
        raise optuna.TrialPruned()


def prune_rsi(params, trial):
    """Prune RSI trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_breakout(params, trial):
    """Prune breakout trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_bollinger(params, trial):
    """Prune Bollinger trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_momentum(params, trial):
    """Prune momentum trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_vol_expansion(params, trial):
    """Prune volatility expansion trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_macd(params, trial):
    """Prune MACD trials with invalid stop or EMA order."""
    check_sl_tp(params)
    if params["fast"] >= params["slow"]:
        raise optuna.TrialPruned()


def prune_stochastic(params, trial):
    """Prune stochastic trials with invalid stop or period setup."""
    check_sl_tp(params)
    if params["d_period"] > params["k_period"]:
        raise optuna.TrialPruned()


# ---------------------- STRATEGY EVALUATION -----------------------------
def evaluate_strategy(df: pd.DataFrame, make_strategy: Callable[[], Any]) -> float:
    """Return the total strategy return for the given dataframe."""

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
    """Create an Optuna objective for the provided strategy class."""

    def objective(trial):
        if hasattr(param_space, "suggest"):
            params = param_space.suggest(trial)
        else:
            params = {
                name: suggest(trial, info, name=name)
                for name, info in param_space.items()
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
    """Run Optuna optimization and return the best trial."""
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
from .strategy import get_strategy


def optimize_sma(df: pd.DataFrame, n_trials: int = 300):
    """Backward-compatible wrapper to optimize the SMA strategy."""
    strategy_cls, config_cls = get_strategy("sma")
    return optimize_with_optuna(
        df,
        strategy_cls,
        config_cls,
        PARAM_SPACES["sma"],
        prune_logic=prune_sma,
        n_trials=n_trials,
    )


# ---------------------- GRIGLIA RAFFINATA ---------------------------
from itertools import product


def _around(val: int, step: int, n: int = 2) -> list[int]:
    """Helper to build a symmetric range around ``val``."""

    return [val + i * step for i in range(-n, n + 1) if val + i * step > 0]


def refined_sma_grid(best: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a small grid of parameters around ``best`` for fine search."""
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
    """Evaluate all SMA parameter combinations and rank the results."""

    log.info("Grid SMA – %d combo", len(combos))
    results = []
    strategy_cls, _ = get_strategy("sma")
    for p in tqdm(combos, desc="SMA"):
        cfg = SMAConfig(**p)
        ret = evaluate_strategy(df, lambda cfg=cfg: strategy_cls(cfg))
        results.append({**p, "total_return": ret})
    return pd.DataFrame(results).sort_values("total_return", ascending=False)
