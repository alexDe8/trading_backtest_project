# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseStrategy(ABC):
    """Scheletro comune per strategie long-only."""

    def __init__(
        self, sl_pct: float, tp_pct: float, trailing_stop_pct: float | None = None
    ) -> None:
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.trailing_stop_pct = trailing_stop_pct

    # ---------------- hooks da implementare --------------
    @abstractmethod
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def entry_signal(self, df: pd.DataFrame) -> pd.Series: ...
    @abstractmethod
    def exit_signal(self, df: pd.DataFrame) -> pd.Series: ...

    # ---------------- motore trades ----------------------
    def generate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_indicators(df.copy())
        entries = self.entry_signal(df).fillna(False)
        exits = self.exit_signal(df).fillna(False)

        in_pos = False
        trailing_sl = None
        trades: list[dict[str, Any]] = []
        for i, row in df.iterrows():
            if (not in_pos) and entries.at[i]:
                in_pos = True
                e_price = row["close"]
                sl_price = e_price * (1 - self.sl_pct / 100)
                tp_price = e_price * (1 + self.tp_pct / 100)
                if self.trailing_stop_pct:
                    trailing_sl = e_price * (1 - self.trailing_stop_pct / 100)
                    sl_price = max(sl_price, trailing_sl)
                e_time = row["timestamp"]
                continue

            if in_pos:
                hit_sl = row["low"] <= sl_price
                hit_tp = row["high"] >= tp_price
                force_exit = exits.at[i]

                if hit_sl or hit_tp or force_exit:
                    x_price = (
                        sl_price if hit_sl else tp_price if hit_tp else row["close"]
                    )
                    trades.append(
                        {
                            "entry_time": e_time,
                            "exit_time": row["timestamp"],
                            "entry": e_price,
                            "exit": x_price,
                            "pct_change": (x_price / e_price - 1) * 100,
                        }
                    )
                    in_pos = False
                    trailing_sl = None
                else:
                    if self.trailing_stop_pct:
                        new_trail = row["high"] * (1 - self.trailing_stop_pct / 100)
                        if trailing_sl is None or new_trail > trailing_sl:
                            trailing_sl = new_trail
                        if trailing_sl > sl_price:
                            sl_price = trailing_sl

        # chiusura forzata a fine serie
        if in_pos:
            trades.append(
                {
                    "entry_time": e_time,
                    "exit_time": df.iloc[-1]["timestamp"],
                    "entry": e_price,
                    "exit": df.iloc[-1]["close"],
                    "pct_change": (df.iloc[-1]["close"] / e_price - 1) * 100,
                }
            )
        return pd.DataFrame(trades)
