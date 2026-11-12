"""Tests for historical CVaR portfolio optimization."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from optimizer import CVaROptimizer, OptimizationError, OptimizerConfig


def make_returns(observations: int = 700) -> pd.DataFrame:
    """Build correlated assets with distinct volatility and downside tails."""
    rng = np.random.default_rng(23)
    market = rng.normal(0.0003, 0.009, observations)
    defensive = 0.00025 + 0.45 * market + rng.normal(0.0, 0.004, observations)
    growth = 0.00055 + 1.15 * market + rng.normal(0.0, 0.010, observations)
    tail = 0.0006 + 0.80 * market + rng.normal(0.0, 0.008, observations)
    tail[::71] -= 0.055
    index = pd.bdate_range("2022-01-03", periods=observations)
    return pd.DataFrame(
        {"Market": market, "Defensive": defensive, "Growth": growth, "Tail": tail},
        index=index,
    )


class CVaROptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = make_returns()
        cls.optimizer = CVaROptimizer(OptimizerConfig(max_weight=0.65, risk_free_rate=0.02))

    def test_solution_respects_bounds_and_ru_expected_shortfall(self) -> None:
        result = self.optimizer.optimize_returns(self.returns, confidence_level=0.95)
        weights = result.portfolio.weights
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights >= -1e-8).all())
        self.assertTrue((weights <= 0.65 + 1e-8).all())

        losses = -(self.returns.to_numpy() @ weights.to_numpy())
        independent_cvar = result.threshold_loss + np.maximum(
            losses - result.threshold_loss, 0.0
        ).mean() / (1.0 - result.confidence_level)
        self.assertAlmostEqual(result.expected_shortfall, float(independent_cvar), places=10)
        self.assertAlmostEqual(
            result.portfolio.expected_return,
            float((self.returns.to_numpy() @ weights.to_numpy()).mean() * 252),
            places=10,
        )
        self.assertEqual(result.observations, len(self.returns))

    def test_frontier_targets_are_feasible_and_tightening_does_not_reduce_cvar(self) -> None:
        frontier = self.optimizer.efficient_frontier(
            self.returns, confidence_level=0.95, points=7
        )
        self.assertEqual(len(frontier.points), 7)
        targets = frontier.points["target_return"].to_numpy()
        cvar = frontier.points["expected_shortfall"].to_numpy()
        self.assertTrue((np.diff(targets) > 0).all())
        self.assertTrue((np.diff(cvar) >= -1e-8).all())
        self.assertTrue(
            (frontier.points["expected_return"].to_numpy() >= targets - 1e-7).all()
        )
        np.testing.assert_allclose(frontier.weights.sum(axis=1), 1.0, atol=1e-7)
        self.assertTrue((frontier.weights.to_numpy() >= -1e-7).all())
        self.assertTrue((frontier.weights.to_numpy() <= 0.65 + 1e-7).all())

    def test_invalid_confidence_target_data_and_frontier_size_are_rejected(self) -> None:
        with self.assertRaises(OptimizationError):
            self.optimizer.optimize_returns(self.returns, confidence_level=1.0)
        with self.assertRaises(OptimizationError):
            self.optimizer.optimize_returns(self.returns, minimum_expected_return=10.0)
        with self.assertRaises(OptimizationError):
            self.optimizer.efficient_frontier(self.returns, points=52)
        with self.assertRaises(OptimizationError):
            self.optimizer.optimize_returns(self.returns.iloc[:20])
        duplicate_columns = self.returns.copy()
        duplicate_columns.columns = ["A", "A", "C", "D"]
        with self.assertRaises(OptimizationError):
            self.optimizer.optimize_returns(duplicate_columns)


if __name__ == "__main__":
    unittest.main()

