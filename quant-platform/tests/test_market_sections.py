"""Tests for regional market-section definitions."""
from __future__ import annotations

import unittest

from market_engine import MARKET_SECTIONS, MarketSection, market_by_key


class MarketSectionTests(unittest.TestCase):
    def test_market_keys_currencies_and_universes_are_independent(self) -> None:
        keys = {market.key for market in MARKET_SECTIONS}
        self.assertEqual(keys, {"us", "uk", "eurozone", "japan", "india", "canada"})
        self.assertEqual(market_by_key("japan").currency_code, "JPY")
        self.assertEqual(market_by_key("uk").currency_code, "GBp")
        self.assertIn("RELIANCE.NS", market_by_key("india").portfolio_tickers)
        self.assertIn("7203.T", market_by_key("japan").portfolio_tickers)
        self.assertIn("VOD.L", market_by_key("uk").portfolio_tickers)
        self.assertIn("SAP.DE", market_by_key("eurozone").portfolio_tickers)
        self.assertIn("RY.TO", market_by_key("canada").portfolio_tickers)
        for market in MARKET_SECTIONS:
            self.assertGreaterEqual(len(market.portfolio_tickers), 2)
            self.assertGreaterEqual(len(market.pairs_tickers), 2)
            self.assertEqual(len(set(market.portfolio_tickers)), len(market.portfolio_tickers))
            self.assertEqual(len(set(market.pairs_tickers)), len(market.pairs_tickers))

    def test_unknown_market_and_invalid_configuration_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            market_by_key("unknown")
        with self.assertRaises(ValueError):
            MarketSection("bad", "Bad", "USD", "$", "", ("SPY",), ("KO", "PEP"))


if __name__ == "__main__":
    unittest.main()

