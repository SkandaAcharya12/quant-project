"""Deterministic, long/short paper-trading ledger.

This engine never connects to a broker. Orders fill at a caller-supplied close
with configurable adverse slippage and commission. That fill assumption is
optimistic versus intraday execution and should be disclosed in reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite


class OrderError(ValueError):
    """Raised when a simulated order violates account or market constraints."""


class TradeAction(str, Enum):
    """Supported manual paper-order actions."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


@dataclass(frozen=True)
class Position:
    """Net shares and average entry mark for one instrument."""

    symbol: str
    quantity: float
    average_entry_price: float
    market_price: float
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        direction = 1.0 if self.quantity >= 0 else -1.0
        return abs(self.quantity) * (self.market_price - self.average_entry_price) * direction


@dataclass(frozen=True)
class Fill:
    """Immutable audit record for an order simulated at the supplied close."""

    sequence: int
    symbol: str
    action: TradeAction
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    timestamp: datetime


@dataclass(frozen=True)
class PaperAccount:
    """A point-in-time account snapshot."""

    starting_cash: float
    cash: float
    equity: float
    positions: tuple[Position, ...]
    fills: tuple[Fill, ...]


@dataclass
class PaperSimulator:
    """Owns a paper ledger and enforces cash, position, and cost accounting.

    BUY/SELL manage long inventory; SHORT/COVER manage short inventory. This
    explicit split prevents an accidental sell from opening a short position.
    Cash changes by signed fill notional; commissions are charged on both sides.
    Short proceeds are credited but are not treated as withdrawable buying power.
    """

    starting_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    _cash: float = field(init=False, repr=False)
    _positions: dict[str, Position] = field(default_factory=dict, init=False, repr=False)
    _fills: list[Fill] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isfinite(self.starting_cash) or self.starting_cash <= 0:
            raise OrderError("starting_cash must be finite and positive")
        if any(not isfinite(v) or v < 0 for v in (self.commission_bps, self.slippage_bps)):
            raise OrderError("commission and slippage must be finite and non-negative")
        self._cash = float(self.starting_cash)

    def mark(self, prices: dict[str, float]) -> None:
        """Update marks; missing held symbols keep their previous mark."""
        for symbol, price in prices.items():
            key = self._validate_symbol(symbol)
            mark = self._validate_price(price)
            if key in self._positions:
                old = self._positions[key]
                self._positions[key] = Position(
                    key, old.quantity, old.average_entry_price, mark, old.realized_pnl
                )

    def submit_order(
        self,
        symbol: str,
        action: TradeAction,
        quantity: float,
        reference_price: float,
        timestamp: datetime | None = None,
    ) -> Fill:
        """Fill one manually requested paper order and append it to the audit ledger."""
        key = self._validate_symbol(symbol)
        price = self._validate_price(reference_price)
        if not isinstance(action, TradeAction):
            raise OrderError("action must be a TradeAction")
        if not isfinite(quantity) or quantity <= 0:
            raise OrderError("quantity must be finite and positive")

        current = self._positions.get(key)
        held = 0.0 if current is None else current.quantity
        if action == TradeAction.BUY and held < 0:
            raise OrderError("use COVER to reduce a short position")
        if action == TradeAction.SELL and (held <= 0 or quantity > held + 1e-12):
            raise OrderError("SELL quantity must not exceed the long position")
        if action == TradeAction.SHORT and held > 0:
            raise OrderError("close the long position before opening a short position")
        if action == TradeAction.COVER and (held >= 0 or quantity > abs(held) + 1e-12):
            raise OrderError("COVER quantity must not exceed the short position")

        is_buy = action in (TradeAction.BUY, TradeAction.COVER)
        slip = self.slippage_bps / 10_000.0
        fill_price = price * (1.0 + slip if is_buy else 1.0 - slip)
        signed_quantity = quantity if action in (TradeAction.BUY, TradeAction.COVER) else -quantity
        if action in (TradeAction.SELL, TradeAction.SHORT):
            signed_quantity = -quantity
        commission = abs(fill_price * quantity) * self.commission_bps / 10_000.0

        if action == TradeAction.BUY:
            new_quantity = held + quantity
            avg = ((abs(held) * (current.average_entry_price if current else 0.0))
                   + quantity * fill_price) / new_quantity
            realized = 0.0 if current is None else current.realized_pnl
        elif action == TradeAction.SHORT:
            new_quantity = held - quantity
            avg = ((abs(held) * (current.average_entry_price if current else 0.0))
                   + quantity * fill_price) / abs(new_quantity)
            realized = 0.0 if current is None else current.realized_pnl
        elif action == TradeAction.SELL:
            new_quantity = held - quantity
            avg = 0.0 if abs(new_quantity) < 1e-12 else current.average_entry_price  # type: ignore[union-attr]
            realized = current.realized_pnl + quantity * (fill_price - current.average_entry_price)  # type: ignore[union-attr]
        else:  # COVER
            new_quantity = held + quantity
            avg = 0.0 if abs(new_quantity) < 1e-12 else current.average_entry_price  # type: ignore[union-attr]
            realized = current.realized_pnl + quantity * (current.average_entry_price - fill_price)  # type: ignore[union-attr]

        self._cash -= signed_quantity * fill_price
        self._cash -= commission
        if abs(new_quantity) < 1e-12:
            self._positions.pop(key, None)
        else:
            self._positions[key] = Position(key, new_quantity, avg, price, realized)
        fill = Fill(len(self._fills) + 1, key, action, float(quantity), price, fill_price,
                    commission, timestamp or datetime.now().astimezone())
        self._fills.append(fill)
        return fill

    def snapshot(self) -> PaperAccount:
        """Return an immutable snapshot; equity includes marked long and short value."""
        positions = tuple(self._positions[k] for k in sorted(self._positions))
        equity = self._cash + sum(p.market_value for p in positions)
        return PaperAccount(self.starting_cash, self._cash, equity, positions, tuple(self._fills))

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        key = symbol.strip().upper()
        if not key or any(ch.isspace() for ch in key):
            raise OrderError("symbol must be non-empty and contain no whitespace")
        return key

    @staticmethod
    def _validate_price(price: float) -> float:
        if not isfinite(price) or price <= 0:
            raise OrderError("price must be finite and positive")
        return float(price)

