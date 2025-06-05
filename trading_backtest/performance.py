import pandas as pd


class PerformanceAnalyzer:
    """Report basilare di performance single-asset."""

    def __init__(
        self, trades: pd.DataFrame, commission: float = 0.0, slippage: float = 0.0
    ):
        self.trades = trades.copy()
        if not self.trades.empty:
            self.trades["net_pct"] = self.trades["pct_change"] - commission - slippage

    def total_return(self) -> float:
        return self.trades["net_pct"].sum() if not self.trades.empty else 0.0

    def trade_count(self) -> int:
        return len(self.trades)

    def avg_trade(self) -> float:
        return self.trades["net_pct"].mean() if not self.trades.empty else 0.0

    def sharpe_ratio(self) -> float:
        """Return the simple Sharpe ratio based on ``net_pct`` returns."""
        if self.trades.empty:
            return 0.0
        r = self.trades["net_pct"]
        std = r.std()
        if std == 0 or len(r) == 0:
            return 0.0
        return (r.mean() / std) * (len(r) ** 0.5)

    def max_drawdown(self) -> float:
        """Return the maximum drawdown in percent."""
        if self.trades.empty:
            return 0.0
        equity = (1 + self.trades["net_pct"] / 100).cumprod()
        drawdown = equity.div(equity.cummax()).sub(1) * 100
        return drawdown.min()

    def win_rate(self) -> float:
        """Return the percentage of profitable trades."""
        if self.trades.empty:
            return 0.0
        return (self.trades["net_pct"] > 0).mean() * 100
