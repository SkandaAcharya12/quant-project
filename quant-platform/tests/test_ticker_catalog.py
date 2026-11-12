"""Tests for regional ticker directory parsing and failure behavior."""
from __future__ import annotations

import unittest
from typing import ClassVar
from unittest import mock

import market_engine.ticker_catalog as catalog_module
from market_engine import TickerCatalog, TickerCatalogError


class TickerCatalogTests(unittest.TestCase):
    def test_fetch_builds_company_labels_and_deduplicates_symbols(self) -> None:
        class FakeYahoo:
            query: ClassVar[object | None] = None
            kwargs: ClassVar[dict[str, object]] = {}

            @staticmethod
            def EquityQuery(operator: str, operand: list[str]) -> object:
                return (operator, operand)

            @classmethod
            def screen(cls, query: object, **kwargs: object) -> dict[str, object]:
                cls.query = query
                cls.kwargs = kwargs
                return {
                    "quotes": [
                        {
                            "symbol": "AAPL",
                            "longName": "Apple Inc.",
                            "quoteType": "EQUITY",
                            "exchangeName": "NasdaqGS",
                        },
                        {
                            "symbol": "AAPL",
                            "shortName": "Duplicate Apple",
                            "quoteType": "EQUITY",
                        },
                        {
                            "symbol": "BND",
                            "shortName": "Bond fund",
                            "quoteType": "MUTUALFUND",
                        },
                        {"symbol": "MSFT", "shortName": "Microsoft", "quoteType": "EQUITY"},
                    ]
                }

        with mock.patch.object(catalog_module, "yf", FakeYahoo):
            options = TickerCatalog().fetch("us")

        self.assertEqual([option.symbol for option in options], ["AAPL", "MSFT"])
        self.assertEqual(options[0].display_label, "Apple Inc. (AAPL) · NasdaqGS")
        self.assertEqual(FakeYahoo.query, ("is-in", ["region", "us"]))
        self.assertEqual(FakeYahoo.kwargs["size"], 250)

    def test_unknown_market_is_rejected_before_api_call(self) -> None:
        with self.assertRaises(ValueError):
            TickerCatalog().fetch("unknown")

    def test_api_errors_are_wrapped_for_the_ui(self) -> None:
        class BrokenYahoo:
            @staticmethod
            def EquityQuery(operator: str, operand: list[str]) -> object:
                return (operator, operand)

            @staticmethod
            def screen(*_args: object, **_kwargs: object) -> dict[str, object]:
                raise TimeoutError("temporary network issue")

        with (
            mock.patch.object(catalog_module, "yf", BrokenYahoo),
            self.assertRaisesRegex(TickerCatalogError, "expanded ticker list"),
        ):
            TickerCatalog().fetch("japan")


if __name__ == "__main__":
    unittest.main()

