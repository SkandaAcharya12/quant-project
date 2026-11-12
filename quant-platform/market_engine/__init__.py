"""Market-section configuration for region-scoped research workflows."""

from market_engine.markets import MARKET_SECTIONS, MarketSection, market_by_key
from market_engine.ticker_catalog import TickerCatalog, TickerCatalogError, TickerOption

__all__ = [
    "MARKET_SECTIONS",
    "MarketSection",
    "TickerCatalog",
    "TickerCatalogError",
    "TickerOption",
    "market_by_key",
]

