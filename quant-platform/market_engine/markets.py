"""Built-in exchange universes used by the regional dashboard sections."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSection:
    """A single-currency market with independent stock and pairs universes."""

    key: str
    name: str
    currency_code: str
    currency_symbol: str
    quote_note: str
    portfolio_tickers: tuple[str, ...]
    pairs_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.name or not self.currency_code or not self.currency_symbol:
            raise ValueError("market identity and currency fields must be non-empty")
        if len(self.portfolio_tickers) < 2 or len(self.pairs_tickers) < 2:
            raise ValueError("each market needs at least two portfolio and pairs tickers")
        for tickers in (self.portfolio_tickers, self.pairs_tickers):
            normalized = tuple(ticker.strip().upper() for ticker in tickers)
            if any(not ticker for ticker in normalized) or len(set(normalized)) != len(normalized):
                raise ValueError("market ticker lists must be non-empty and unique")


MARKET_SECTIONS: tuple[MarketSection, ...] = (
    MarketSection(
        key="us",
        name="United States",
        currency_code="USD",
        currency_symbol="$",
        quote_note="USD-denominated U.S. listings",
        portfolio_tickers=("SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "XLE", "XLV"),
        pairs_tickers=("KO", "PEP", "XOM", "CVX"),
    ),
    MarketSection(
        key="uk",
        name="United Kingdom",
        currency_code="GBp",
        currency_symbol="p",
        quote_note="London Stock Exchange prices are commonly quoted in pence (GBp)",
        portfolio_tickers=("VOD.L", "HSBA.L", "BP.L", "SHEL.L", "AZN.L", "GSK.L", "ULVR.L", "DGE.L"),
        pairs_tickers=("BP.L", "SHEL.L", "VOD.L", "GSK.L"),
    ),
    MarketSection(
        key="eurozone",
        name="Eurozone",
        currency_code="EUR",
        currency_symbol="€",
        quote_note="EUR-denominated listings across major Eurozone exchanges",
        portfolio_tickers=("ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "DTE.DE"),
        pairs_tickers=("ASML.AS", "AIR.PA", "SAP.DE", "SIE.DE"),
    ),
    MarketSection(
        key="japan",
        name="Japan",
        currency_code="JPY",
        currency_symbol="¥",
        quote_note="JPY-denominated Tokyo Stock Exchange listings",
        portfolio_tickers=("7203.T", "6758.T", "9984.T", "8306.T", "9432.T", "7974.T", "9983.T", "8035.T"),
        pairs_tickers=("7203.T", "7267.T", "6758.T", "6752.T"),
    ),
    MarketSection(
        key="india",
        name="India",
        currency_code="INR",
        currency_symbol="₹",
        quote_note="INR-denominated National Stock Exchange listings",
        portfolio_tickers=("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS"),
        pairs_tickers=("TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"),
    ),
    MarketSection(
        key="canada",
        name="Canada",
        currency_code="CAD",
        currency_symbol="C$",
        quote_note="CAD-denominated Toronto Stock Exchange listings",
        portfolio_tickers=("RY.TO", "TD.TO", "ENB.TO", "CNQ.TO", "SHOP.TO", "CNR.TO", "BNS.TO", "BAM.TO"),
        pairs_tickers=("RY.TO", "TD.TO", "BNS.TO", "CM.TO"),
    ),
)


def market_by_key(key: str) -> MarketSection:
    """Return a configured market by stable key, raising on unknown sections."""
    for market in MARKET_SECTIONS:
        if market.key == key:
            return market
    raise ValueError(f"Unknown market section: {key}")

