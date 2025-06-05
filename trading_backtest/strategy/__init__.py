from .sma import SMACrossoverStrategy
from .rsi import RSIStrategy
from .breakout import BreakoutStrategy
from .bollinger import BollingerBandStrategy
from .momentum import MomentumImpulseStrategy, VolatilityExpansionStrategy
from .macd import MACDStrategy
from .stochastic import StochasticStrategy

STRATEGY_REGISTRY = {
    "sma": SMACrossoverStrategy,
    "rsi": RSIStrategy,
    "breakout": BreakoutStrategy,
    "bollinger": BollingerBandStrategy,
    "momentum": MomentumImpulseStrategy,
    "vol_expansion": VolatilityExpansionStrategy,
    "macd": MACDStrategy,
    "stochastic": StochasticStrategy,
}
