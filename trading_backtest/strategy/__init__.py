from __future__ import annotations

from ..config import (
    SMAConfig,
    RSIConfig,
    BreakoutConfig,
    BollingerConfig,
    MomentumConfig,
    VolExpansionConfig,
    MACDConfig,
    StochasticConfig,
)

from .sma import SMACrossoverStrategy
from .rsi import RSIStrategy
from .breakout import BreakoutStrategy
from .bollinger import BollingerBandStrategy
from .momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy
from .macd import MACDStrategy
from .stochastic import StochasticStrategy

STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {}

def register_strategy(name: str, cls: type, config_cls: type) -> None:
    """Register a strategy class and its config."""
    STRATEGY_REGISTRY[name] = (cls, config_cls)

def get_strategy(name: str) -> tuple[type, type]:
    """Return (strategy_cls, config_cls) for name."""
    return STRATEGY_REGISTRY[name]

register_strategy("sma", SMACrossoverStrategy, SMAConfig)
register_strategy("rsi", RSIStrategy, RSIConfig)
register_strategy("breakout", BreakoutStrategy, BreakoutConfig)
register_strategy("bollinger", BollingerBandStrategy, BollingerConfig)
register_strategy("momentum", MomentumImpulseStrategy, MomentumConfig)
register_strategy("vol_expansion", VolatilityExpansionStrategy, VolExpansionConfig)
register_strategy("macd", MACDStrategy, MACDConfig)
register_strategy("stochastic", StochasticStrategy, StochasticConfig)

