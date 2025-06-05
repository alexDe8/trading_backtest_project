# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class Trade:
    """Simple representation of a single trade."""

    entry_time: Any
    exit_time: Any
    entry: float
    exit: float
    qty: float = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry": self.entry,
            "exit": self.exit,
            "pct_change": (self.exit / self.entry - 1) * 100,
            "qty": self.qty,
        }


class BaseStrategy(ABC):
    """Scheletro comune per strategie long-only."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.sl_pct = config.sl_pct
        self.tp_pct = config.tp_pct
        self.trailing_stop_pct = getattr(config, "trailing_stop_pct", None)
        if self.sl_pct >= self.tp_pct:
            raise ValueError("sl_pct must be less than tp_pct")
        if self.trailing_stop_pct is not None and self.trailing_stop_pct <= 0:
            raise ValueError("trailing_stop_pct must be positive")
        # opzionale: puoi permettere di impostare la size
        # self.position_size = 1

    # ---------------- hooks da implementare --------------
    @abstractmethod
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def entry_signal(self, df: pd.DataFrame) -> pd.Series: ...
    @abstractmethod
    def exit_signal(self, df: pd.DataFrame) -> pd.Series: ...

    # ---------------- motore trades ----------------------
    def generate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[DEBUG] Config: {self.config}")
        df = self.prepare_indicators(df.copy())
        entries = self.entry_signal(df).fillna(False)
        exits = self.exit_signal(df).fillna(False)

        in_pos = False
        trailing_sl = None
        trades: list[Trade] = []
        for row in df.itertuples():
            idx = row.Index
            if (not in_pos) and entries.at[idx]:
                in_pos = True
                e_price, sl_price, tp_price, trailing_sl, e_time, qty = (
                    self._open_trade(row)
                )
                continue

            if in_pos:
                hit_sl = row.low <= sl_price
                hit_tp = row.high >= tp_price
                force_exit = exits.at[idx]

                if hit_sl or hit_tp or force_exit:
                    trade = self._close_trade(
                        row, e_time, e_price, sl_price, tp_price, hit_sl, hit_tp, qty
                    )
                    trades.append(trade)
                    in_pos = False
                    trailing_sl = None
                else:
                    sl_price, trailing_sl = self._update_trailing_stop(
                        row, sl_price, trailing_sl
                    )

        if in_pos:
            trades.append(
                Trade(
                    entry_time=e_time,
                    exit_time=df.iloc[-1]["timestamp"],
                    entry=e_price,
                    exit=df.iloc[-1]["close"],
                    qty=qty,
                )
            )
        trades_df = pd.DataFrame([t.as_dict() for t in trades])
        print(f"[DEBUG] Numero trade generati: {len(trades_df)}")
        if not trades_df.empty:
            print(trades_df.head(3))
        return trades_df

    # ---------------- metodi interni ----------------------
    def _open_trade(
        self, row: Any
    ) -> tuple[float, float, float, float | None, Any, float]:
        e_price = row.close
        sl_price = e_price * (1 - self.sl_pct / 100)
        tp_price = e_price * (1 + self.tp_pct / 100)
        trailing_sl = None
        if self.trailing_stop_pct:
            trailing_sl = e_price * (1 - self.trailing_stop_pct / 100)
            sl_price = max(sl_price, trailing_sl)
        qty = getattr(self, "position_size", 1)
        return e_price, sl_price, tp_price, trailing_sl, row.timestamp, qty

    def _update_trailing_stop(
        self, row: Any, sl_price: float, trailing_sl: float | None
    ) -> tuple[float, float | None]:
        if not self.trailing_stop_pct:
            return sl_price, trailing_sl

        new_trail = row.high * (1 - self.trailing_stop_pct / 100)
        if trailing_sl is None or new_trail > trailing_sl:
            trailing_sl = new_trail
        if trailing_sl is not None and trailing_sl > sl_price:
            sl_price = trailing_sl
        return sl_price, trailing_sl

    def _close_trade(
        self,
        row: Any,
        e_time: Any,
        e_price: float,
        sl_price: float,
        tp_price: float,
        hit_sl: bool,
        hit_tp: bool,
        qty: float,
    ) -> Trade:
        x_price = sl_price if hit_sl else tp_price if hit_tp else row.close
        return Trade(
            entry_time=e_time,
            exit_time=row.timestamp,
            entry=e_price,
            exit=x_price,
            qty=qty,
        )
