"""Searchable regional security options backed by Yahoo Finance's screener."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import yfinance as yf


@dataclass(frozen=True)
class TickerOption:
    """A market-data symbol and the human-readable name shown in dropdowns."""

    symbol: str
    name: str
    exchange: str = ""

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.name.strip():
            raise ValueError("ticker options require a symbol and display name")

    @property
    def display_label(self) -> str:
        """Show company name first so people need not know a ticker to find it."""
        exchange = f" · {self.exchange}" if self.exchange else ""
        return f"{self.name} ({self.symbol}){exchange}"


class TickerCatalogError(RuntimeError):
    """Raised when the remote market catalog cannot be read or parsed."""


class TickerCatalog:
    """Load up to 250 equity choices for a regional market.

    The remote screener is used instead of a hand-maintained symbol dump, so the
    picker can show names and discover listed securities without requiring users
    to know the exchange's symbol format. Yahoo's custom screen is capped at 250
    rows; each result is still checked by the price data engine before use.
    """

    MAX_RESULTS: ClassVar[int] = 250
    REGIONS_BY_MARKET: ClassVar[dict[str, tuple[str, ...]]] = {
        "us": ("us",),
        "uk": ("gb",),
        "eurozone": ("de", "fr", "nl", "it", "es", "be", "fi", "ie", "pt", "at", "gr"),
        "japan": ("jp",),
        "india": ("in",),
        "canada": ("ca",),
    }

    def fetch(self, market_key: str) -> tuple[TickerOption, ...]:
        """Fetch the largest available regional equities and label each symbol.

        Market capitalization is used only to order the menu, not as an
        investment recommendation or a universe-selection signal for analysis.
        """
        regions = self.REGIONS_BY_MARKET.get(market_key)
        if regions is None:
            raise ValueError(f"Unknown market section: {market_key}")

        try:
            query = yf.EquityQuery("is-in", ["region", *regions])
            response: Any = yf.screen(
                query,
                size=self.MAX_RESULTS,
                sortField="intradaymarketcap",
                sortAsc=False,
            )
            quotes: Any = response.get("quotes", [])
            if not isinstance(quotes, list):
                raise TypeError("Yahoo Finance returned an invalid quotes list")

            options: list[TickerOption] = []
            seen: set[str] = set()
            for quote in quotes:
                if not isinstance(quote, dict):
                    continue
                symbol = str(quote.get("symbol", "")).strip().upper()
                quote_type = str(quote.get("quoteType", "EQUITY")).upper()
                if not symbol or symbol in seen or quote_type not in {"EQUITY", "ETF"}:
                    continue
                name = str(quote.get("longName") or quote.get("shortName") or symbol).strip()
                exchange = str(quote.get("exchangeName") or quote.get("exchange") or "").strip()
                options.append(TickerOption(symbol=symbol, name=name, exchange=exchange))
                seen.add(symbol)

            if not options:
                raise ValueError("Yahoo Finance returned no usable equity symbols")
            return tuple(options)
        except Exception as exc:
            raise TickerCatalogError(
                f"Could not load the expanded ticker list for {market_key}."
            ) from exc

