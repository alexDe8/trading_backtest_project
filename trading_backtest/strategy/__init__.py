from .sma import SMACrossoverStrategy
from .rsi import RSIStrategy
from .breakout import BreakoutStrategy
from .bollinger import BollingerBandStrategy
from .momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy
from .random_forest import RandomForestStrategy

STRATEGY_REGISTRY = {
    "sma": SMACrossoverStrategy,
    "rsi": RSIStrategy,
    "breakout": BreakoutStrategy,
    "bollinger": BollingerBandStrategy,
    "momentum": MomentumImpulseStrategy,
    "vol_expansion": VolatilityExpansionStrategy,
    "random_forest": RandomForestStrategy,
}

__all__ = [
    "SMACrossoverStrategy",
    "RSIStrategy",
    "BreakoutStrategy",
    "BollingerBandStrategy",
    "MomentumImpulseStrategy",
    "VolatilityExpansionStrategy",
    "RandomForestStrategy",
    "STRATEGY_REGISTRY",
]
