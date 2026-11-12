"""Offline tests for paper-ledger accounting and leakage-aware alerts."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paper_trading import (
    AgentError,
    AlertAction,
    OrderError,
    PaperSimulator,
    ResearchAgent,
    TradeAction,
)


class PaperSimulatorTests(unittest.TestCase):
    def test_long_fill_costs_mark_to_market_and_realized_pnl(self) -> None:
        account = PaperSimulator(starting_cash=10_000, commission_bps=10, slippage_bps=10)
        fill = account.submit_order("abc", TradeAction.BUY, 10, 100,
                                    pd.Timestamp("2025-01-02", tz="UTC").to_pydatetime())
        self.assertAlmostEqual(fill.fill_price, 100.1)
        self.assertAlmostEqual(fill.commission, 1.001)
        account.mark({"ABC": 110})
        self.assertAlmostEqual(account.snapshot().equity, 10_097.999)

        sale = account.submit_order("ABC", TradeAction.SELL, 4, 110)
        self.assertAlmostEqual(sale.fill_price, 109.89)
        position = account.snapshot().positions[0]
        self.assertAlmostEqual(position.quantity, 6)
        self.assertAlmostEqual(position.realized_pnl, 4 * (109.89 - 100.1))

    def test_short_cover_accounting_and_validation(self) -> None:
        account = PaperSimulator(starting_cash=10_000, commission_bps=0, slippage_bps=0)
        account.submit_order("XYZ", TradeAction.SHORT, 5, 40)
        account.mark({"XYZ": 35})
        self.assertAlmostEqual(account.snapshot().equity, 10_025)
        account.submit_order("XYZ", TradeAction.COVER, 2, 35)
        position = account.snapshot().positions[0]
        self.assertEqual(position.quantity, -3)
        self.assertAlmostEqual(position.realized_pnl, 10)
        with self.assertRaises(OrderError):
            account.submit_order("XYZ", TradeAction.COVER, 4, 35)
        with self.assertRaises(OrderError):
            account.submit_order("XYZ", TradeAction.SELL, 1, 35)


class ResearchAgentTests(unittest.TestCase):
    @staticmethod
    def prices(periods: int = 500) -> pd.Series:
        rng = np.random.default_rng(17)
        shocks = rng.normal(0.0003, 0.012, periods)
        return pd.Series(
            100 * np.exp(np.cumsum(shocks)),
            index=pd.bdate_range("2022-01-03", periods=periods),
            name="TEST",
        )

    def test_alert_is_reproducible_and_reports_known_training_sample(self) -> None:
        prices = self.prices()
        cutoff = prices.index[420]
        agent = ResearchAgent(min_training_observations=100)
        early = agent.evaluate(prices, "TEST", cutoff)
        extended = agent.evaluate(prices, "TEST", prices.index[-1])
        same_cutoff = agent.evaluate(prices.loc[:cutoff], "TEST")

        self.assertEqual(early, same_cutoff)
        self.assertLess(early.as_of, prices.index[421])
        self.assertGreaterEqual(early.training_observations, 100)
        self.assertAlmostEqual(early.probability_up + early.probability_down, 1.0)
        self.assertIn(early.action, tuple(AlertAction))
        self.assertNotEqual(early.as_of, extended.as_of)
        if early.confidence < 0.90:
            self.assertEqual(early.action, AlertAction.HOLD)
        else:
            self.assertNotEqual(early.action, AlertAction.HOLD)
        self.assertEqual(agent.confidence_threshold, 0.90)

    def test_insufficient_or_malformed_data_fails_clearly(self) -> None:
        agent = ResearchAgent(min_training_observations=100)
        with self.assertRaises(AgentError):
            agent.evaluate(self.prices(100), "TEST")
        bad = self.prices()
        bad.iloc[-1] = -1
        with self.assertRaises(AgentError):
            agent.evaluate(bad, "TEST")


if __name__ == "__main__":
    unittest.main()

