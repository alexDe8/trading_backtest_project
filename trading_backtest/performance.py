import pandas as pd


class PerformanceAnalyzer:
    """Report basilare di performance single-asset."""

    def __init__(
        self,
        trades: pd.DataFrame,
        commission: float = 0.0,
        slippage: float = 0.0,
        position_size: float = 1.0,
    ):
        self.trades = trades.copy()
        self.position_size = position_size
        if not self.trades.empty:
            self.trades["net_pct"] = (
                self.trades["pct_change"] - commission - slippage
            ) * self.position_size

    def total_return(self) -> float:
        return self.trades["net_pct"].sum() if not self.trades.empty else 0.0

    def trade_count(self) -> int:
        return len(self.trades)

    def avg_trade(self) -> float:
        return self.trades["net_pct"].mean() if not self.trades.empty else 0.0
