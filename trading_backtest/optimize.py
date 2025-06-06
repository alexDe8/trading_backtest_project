# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable, Mapping
from dataclasses import dataclass, fields
import dataclasses
import pandas as pd
import optuna
from tqdm import tqdm
from .performance import PerformanceAnalyzer
from .data import add_indicator_cache
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


def _value_list(value: Any) -> list[int]:
    """Return all integer values encoded by ``value``.

    This expands parameter space tuples like ``("int", 5, 10, 2)`` or
    ``("cat", [10, 20, None])`` as well as plain integers or iterables of
    integers.  ``None`` values are ignored.
    """

    if value is None:
        return []
    if isinstance(value, tuple):
        kind, *rest = value
        if kind == "int":
            low, high, *opt = rest
            step = opt[0] if opt else 1
            return list(range(low, high + 1, step))
        if kind == "cat":
            return [v for v in rest[0] if isinstance(v, int)]
        return []
    if isinstance(value, (list, set, tuple)):
        return [v for v in value if isinstance(v, int)]
    if isinstance(value, int):
        return [value]
    return []


def gather_all_indicator_periods(params: Any) -> dict[str, list[int]]:
    """Return indicator windows required by a parameter space or grid list."""

    res: dict[str, set[int]] = {
        "sma": set(),
        "rsi": set(),
        "atr": set(),
        "vol": set(),
        "imp": set(),
        "hmax": set(),
        "bb": set(),
    }

    def process(p: Mapping[str, Any]) -> None:
        # Determine strategy to disambiguate ``period``
        strat = None
        if any(k.startswith("sma_") for k in p):
            strat = "sma"
        elif "nstd" in p:
            strat = "bollinger"
        elif "vol_threshold" in p:
            strat = "vol_expansion"
        elif "threshold" in p and "window" in p:
            strat = "momentum"
        elif "lookback" in p or "atr_period" in p:
            strat = "breakout"
        elif (
            "oversold" in p
            and "period" in p
            and not {"k_period", "d_period"} & p.keys()
        ):
            strat = "rsi"

        for name, val in p.items():
            vals = _value_list(val)
            if name in {"sma_fast", "sma_slow", "sma_trend"}:
                res["sma"].update(vals)
            elif name == "period":
                if strat == "rsi":
                    res["rsi"].update(vals)
                elif strat == "bollinger":
                    res["bb"].update(vals)
            elif name == "lookback":
                res["hmax"].update(vals)
            elif name == "atr_period":
                res["atr"].update(vals)
            elif name == "window":
                res["imp"].update(vals)
            elif name == "vol_window":
                res["vol"].update(vals)

    if isinstance(params, list):
        for d in params:
            process(d)
    elif dataclasses.is_dataclass(params):
        process({f.name: getattr(params, f.name) for f in dataclasses.fields(params)})
    elif isinstance(params, Mapping):
        process(params)
    else:
        raise TypeError("Unsupported parameter container")

    for v in res.values():
        v.discard(None)
    return {k: sorted(v) for k, v in res.items() if v}


def ensure_indicator_cache(df: pd.DataFrame, params: Any) -> dict[str, list[int]]:
    """Populate indicator columns based on ``params`` and validate their presence."""

    periods = gather_all_indicator_periods(params)
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

    missing = []
    for p in periods.get("sma", []):
        if f"sma_{p}" not in df:
            missing.append(f"sma_{p}")
    for p in periods.get("rsi", []):
        if f"rsi_{p}" not in df:
            missing.append(f"rsi_{p}")
    for p in periods.get("atr", []):
        if f"atr_{p}" not in df:
            missing.append(f"atr_{p}")
    for p in periods.get("vol", []):
        if f"vol_{p}" not in df:
            missing.append(f"vol_{p}")
    for p in periods.get("imp", []):
        if f"impulse_{p}" not in df:
            missing.append(f"impulse_{p}")
    for p in periods.get("hmax", []):
        if f"hmax_{p}" not in df:
            missing.append(f"hmax_{p}")
    for p in periods.get("bb", []):
        if f"bbm_{p}" not in df or f"bbs_{p}" not in df:
            missing.append(f"bb_{p}")

    if missing:
        raise KeyError(f"Colonne mancanti: {', '.join(missing)}")

    return periods


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
    """Prune RSI trials with invalid parameters.

    Besides stop/take-profit checks, ensure ``oversold`` stays within
    a sensible range (0–50) and the period is positive.
    """
    check_sl_tp(params)
    if not (0 <= params["oversold"] <= 50):
        raise optuna.TrialPruned()
    if params["period"] <= 0:
        raise optuna.TrialPruned()


def prune_breakout(params, trial):
    """Prune breakout trials with invalid stop or take-profit."""
    check_sl_tp(params)


def prune_bollinger(params, trial):
    """Prune Bollinger trials with invalid parameters."""
    check_sl_tp(params)
    if params["nstd"] <= 0:
        raise optuna.TrialPruned()
    if params["period"] <= 0:
        raise optuna.TrialPruned()


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
