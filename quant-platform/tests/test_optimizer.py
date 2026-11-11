"""Offline tests for the portfolio optimizer (synthetic data with known structure).

The closed-form solutions from the module's proofs serve as ground truth for the numerical
solver. Run with ``python -m pytest`` or ``python -m unittest``.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from optimizer import (
    OptimizationError,
    OptimizerConfig,
    PortfolioOptimizer,
    analytic_frontier_variance,
    analytic_gmv,
    analytic_tangency,
    bayes_stein_means,
    ledoit_wolf_covariance,
)

try:
    from sklearn.covariance import ledoit_wolf

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

FREE = {"min_weight": float("-inf"), "max_weight": float("inf")}  # textbook problem


def make_returns(n_assets: int = 6, t: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Correlated daily returns with a 2-factor covariance and spread-out means."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 0.006, (n_assets, 2))
    cov = loadings @ loadings.T + np.diag(rng.uniform(0.006, 0.012, n_assets) ** 2)
    mean = np.linspace(0.0002, 0.0007, n_assets)
    data = rng.multivariate_normal(mean, cov, size=t)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols, index=pd.bdate_range("2018-01-01", periods=t))


def to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    prices = 100.0 * (1.0 + returns).cumprod()
    start = pd.DataFrame([np.full(returns.shape[1], 100.0)], columns=returns.columns,
                         index=[returns.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, prices])


class EstimatorTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
    def test_ledoit_wolf_matches_sklearn(self) -> None:
        for t, n, seed in ((250, 8, 1), (60, 40, 2), (2000, 5, 3)):
            x = make_returns(n, t, seed).to_numpy()
            ours, delta = ledoit_wolf_covariance(x)
            ref, ref_delta = ledoit_wolf(x)
            self.assertAlmostEqual(delta, ref_delta, places=10)
            np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-14)

    def test_ledoit_wolf_properties(self) -> None:
        x = make_returns(40, 60, seed=4).to_numpy()  # T barely above N: sample cov is awful
        shrunk, delta = ledoit_wolf_covariance(x)
        sample = np.cov(x, rowvar=False, ddof=1)
        self.assertTrue(0.0 < delta < 1.0)
        self.assertGreater(np.linalg.eigvalsh(shrunk).min(), 0.0)  # positive definite
        self.assertLess(np.linalg.cond(shrunk), np.linalg.cond(sample))
        # More data => the sample covariance is trustworthy => less shrinkage.
        d_small = ledoit_wolf_covariance(make_returns(10, 60, 5).to_numpy())[1]
        d_large = ledoit_wolf_covariance(make_returns(10, 5000, 5).to_numpy())[1]
        self.assertLess(d_large, d_small)
        # Works even when there are fewer observations than assets (sample cov is singular).
        wide, _ = ledoit_wolf_covariance(make_returns(30, 100, 6).to_numpy()[:20])
        self.assertGreater(np.linalg.eigvalsh(wide).min(), 0.0)

    def test_bayes_stein(self) -> None:
        returns = make_returns(6, 500, 7).to_numpy()
        mean, cov = returns.mean(0), np.cov(returns, rowvar=False)
        shrunk, w = bayes_stein_means(mean, cov, 500)
        self.assertTrue(0.0 < w < 1.0)
        self.assertLess(shrunk.std(), mean.std())  # pulled toward a common mean
        # Manual formula
        ones = np.ones(6)
        m0 = ones @ np.linalg.solve(cov, mean) / (ones @ np.linalg.solve(cov, ones))
        d = mean - m0
        w_manual = 8.0 / (8.0 + 500 * d @ np.linalg.solve(cov, d))
        self.assertAlmostEqual(w, w_manual, places=12)
        np.testing.assert_allclose(shrunk, (1 - w_manual) * mean + w_manual * m0)
        # More data => estimates are more reliable => weight on the common mean falls.
        self.assertLess(bayes_stein_means(mean, cov, 50_000)[1], w)


class SolverGroundTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = make_returns(6, 1500, seed=10)
        cfg = OptimizerConfig(covariance_method="sample", risk_free_rate=0.0, **FREE)
        cls.free = PortfolioOptimizer(cfg).optimize_returns(cls.returns)
        cls.mu = cls.free.mu.to_numpy()
        cls.cov = cls.free.cov.to_numpy()

    def test_gmv_matches_closed_form(self) -> None:
        w = analytic_gmv(self.cov)
        np.testing.assert_allclose(self.free.gmv.weights.to_numpy(), w, atol=1e-6)
        c = np.ones(6) @ np.linalg.solve(self.cov, np.ones(6))
        self.assertAlmostEqual(self.free.gmv.volatility**2, 1.0 / c, places=9)

    def test_tangency_matches_closed_form(self) -> None:
        w = analytic_tangency(self.mu, self.cov, 0.0)
        np.testing.assert_allclose(self.free.max_sharpe.weights.to_numpy(), w, atol=1e-5)
        excess = self.mu
        sharpe_sq = excess @ np.linalg.solve(self.cov, excess)
        self.assertAlmostEqual(self.free.max_sharpe.sharpe, np.sqrt(sharpe_sq), places=7)

    def test_frontier_matches_closed_form_parabola(self) -> None:
        fr = self.free.frontier
        expected = analytic_frontier_variance(self.mu, self.cov, fr["return"].to_numpy())
        np.testing.assert_allclose(fr["volatility"].to_numpy() ** 2, expected, rtol=1e-6)
        self.assertTrue((np.diff(fr["volatility"]) > 0).all())  # efficient branch: risk rises
        self.assertAlmostEqual(fr["volatility"].iloc[0], self.free.gmv.volatility, places=9)

    def test_constrained_frontier_cannot_beat_unconstrained(self) -> None:
        res = PortfolioOptimizer(OptimizerConfig(covariance_method="sample")).optimize_returns(self.returns)
        floor = analytic_frontier_variance(self.mu, self.cov, res.frontier["return"].to_numpy())
        self.assertTrue((res.frontier["volatility"].to_numpy() ** 2 >= floor - 1e-9).all())
        self.assertLessEqual(res.max_sharpe.sharpe, self.free.max_sharpe.sharpe + 1e-9)

    def test_max_sharpe_beats_every_alternative(self) -> None:
        r = self.free
        self.assertGreaterEqual(r.max_sharpe.sharpe, r.gmv.sharpe)
        self.assertGreaterEqual(r.max_sharpe.sharpe, r.equal_weight.sharpe)
        self.assertGreaterEqual(r.max_sharpe.sharpe, r.monte_carlo["sharpe"].max())


class ConstraintAndMonteCarloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = make_returns(6, 1500, seed=11)
        cls.cfg = OptimizerConfig(min_weight=0.0, max_weight=0.35, risk_free_rate=0.02)
        cls.opt = PortfolioOptimizer(cls.cfg)
        cls.res = cls.opt.optimize_returns(cls.returns)

    def test_weights_respect_bounds_everywhere(self) -> None:
        r = self.res
        for p in (r.gmv, r.max_sharpe, r.equal_weight):
            self.assertAlmostEqual(p.weights.sum(), 1.0, places=7)
            self.assertTrue((p.weights >= -1e-7).all() and (p.weights <= 0.35 + 1e-7).all())
        wcols = [c for c in r.frontier.columns if c.startswith("w_")]
        fw = r.frontier[wcols].to_numpy()
        np.testing.assert_allclose(fw.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue((fw >= -1e-7).all() and (fw <= 0.35 + 1e-7).all())
        mw = r.mc_weights
        np.testing.assert_allclose(mw.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue((mw >= 0).all() and (mw <= 0.35).all())

    def test_ten_thousand_simulations_seeded(self) -> None:
        self.assertEqual(len(self.res.monte_carlo), 10_000)
        again = PortfolioOptimizer(self.cfg).optimize_returns(self.returns)
        np.testing.assert_array_equal(self.res.mc_weights, again.mc_weights)
        other = PortfolioOptimizer(
            OptimizerConfig(max_weight=0.35, risk_free_rate=0.02, seed=7)
        ).optimize_returns(self.returns)
        self.assertFalse(np.array_equal(self.res.mc_weights, other.mc_weights))

    def test_no_random_portfolio_dominates_the_frontier(self) -> None:
        r, opt = self.res, self.opt
        mu, cov = r.mu.to_numpy(), r.cov.to_numpy()
        idx = np.random.default_rng(0).choice(len(r.monte_carlo), 25, replace=False)
        for i in idx:
            m = float(r.monte_carlo["return"].iloc[i])
            best = opt._min_variance(cov, mu, m)
            self.assertGreaterEqual(r.monte_carlo["volatility"].iloc[i] ** 2, best @ cov @ best - 1e-9)
        self.assertLessEqual(r.monte_carlo["sharpe"].max(), r.max_sharpe.sharpe + 1e-9)

    def test_random_search_degrades_as_assets_increase(self) -> None:
        """Monte Carlo is a picture, not an optimizer: the gap to the exact optimum widens."""
        gaps = []
        for n in (3, 15):
            res = PortfolioOptimizer(OptimizerConfig()).optimize_returns(make_returns(n, 1500, 21))
            gaps.append(res.max_sharpe.sharpe - res.best_random_portfolio().sharpe)
        self.assertGreater(gaps[1], gaps[0])

    def test_fallback_when_nothing_beats_risk_free(self) -> None:
        res = PortfolioOptimizer(OptimizerConfig(risk_free_rate=1.0)).optimize_returns(self.returns)
        self.assertIn("risk-free", res.max_sharpe.note)
        np.testing.assert_allclose(res.max_sharpe.weights, res.gmv.weights)

    def test_shrinkage_options_run_and_change_results(self) -> None:
        base = PortfolioOptimizer(OptimizerConfig(covariance_method="sample")).optimize_returns(self.returns)
        shrunk = PortfolioOptimizer(
            OptimizerConfig(covariance_method="ledoit_wolf", mean_estimator="bayes_stein")
        ).optimize_returns(self.returns)
        self.assertEqual(base.shrinkage, 0.0)
        self.assertGreater(shrunk.shrinkage, 0.0)
        self.assertGreater(shrunk.mean_shrinkage, 0.0)
        self.assertLess(shrunk.mu.std(), base.mu.std())

    def test_summary_and_best_random(self) -> None:
        text = self.res.summary()
        self.assertIn("Max Sharpe", text)
        self.assertIn("Exact optimum beats best of 10,000", text)
        self.assertEqual(len(self.res.best_random_portfolio().weights), 6)

    def test_validation(self) -> None:
        opt = PortfolioOptimizer()
        with self.assertRaises(OptimizationError):
            opt.optimize_returns(self.returns[["A0"]])
        bad = self.returns.copy()
        bad.iloc[5, 2] = np.nan
        with self.assertRaises(OptimizationError):
            opt.optimize_returns(bad)
        with self.assertRaises(OptimizationError):
            opt.optimize_returns(self.returns.iloc[:10])
        with self.assertRaises(OptimizationError):
            PortfolioOptimizer(OptimizerConfig(max_weight=0.1)).optimize_returns(self.returns)
        bad_prices = to_prices(self.returns)
        bad_prices.iloc[3, 0] = -1.0
        with self.assertRaises(OptimizationError):
            opt.optimize(bad_prices)
        with self.assertRaises(OptimizationError):
            OptimizerConfig(min_weight=0.5, max_weight=0.2)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_matches_manual(self) -> None:
        returns = make_returns(4, 600, 30)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=returns.columns)
        stats = PortfolioOptimizer(OptimizerConfig(risk_free_rate=0.0)).evaluate(w, returns)
        r = returns.to_numpy() @ w.to_numpy()
        self.assertAlmostEqual(stats["ann_volatility"], r.std(ddof=1) * np.sqrt(252), places=12)
        self.assertAlmostEqual(stats["sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(252), places=12)
        with self.assertRaises(OptimizationError):
            PortfolioOptimizer().evaluate(w.iloc[:3], returns)


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = make_returns(5, 900, seed=40)
        cls.prices = to_prices(cls.returns)
        cls.opt = PortfolioOptimizer(OptimizerConfig(risk_free_rate=0.0, max_weight=0.6))
        cls.wf = cls.opt.walk_forward(cls.prices, lookback=252, rebalance_every=21, cost_bps=5.0)

    def test_shapes_and_weights(self) -> None:
        wf = self.wf
        self.assertEqual(len(wf.returns), 900 - 252)
        for name, table in wf.weights.items():
            np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-6, err_msg=name)
        self.assertEqual(set(wf.returns.columns), {"max_sharpe", "min_variance", "equal_weight"})
        m = wf.metrics()
        self.assertIn("sharpe", m.columns)
        self.assertEqual(len(m), 3)

    def test_equal_weight_daily_rebalance_identity(self) -> None:
        """With rebalance_every=1 and zero cost, the 1/N strategy must equal the cross-sectional mean."""
        wf = self.opt.walk_forward(self.prices, 252, rebalance_every=1, cost_bps=0.0,
                                   strategies=("equal_weight",))
        expected = self.returns.mean(axis=1).iloc[252:]
        np.testing.assert_allclose(wf.returns["equal_weight"].to_numpy(), expected.to_numpy(), atol=1e-14)

    def test_buy_and_hold_drift_accounting(self) -> None:
        wf = self.opt.walk_forward(self.prices, 252, rebalance_every=21, cost_bps=0.0,
                                   strategies=("equal_weight",))
        window = self.returns.iloc[252:273].to_numpy()
        compounded = np.prod(1.0 + wf.returns["equal_weight"].iloc[:21]) - 1.0
        self.assertAlmostEqual(compounded, (np.prod(1.0 + window, axis=0) / 5).sum() - 1.0, places=12)

    def test_no_lookahead(self) -> None:
        cut = 700
        short = self.opt.walk_forward(self.prices.iloc[:cut], 252, 21, 5.0)
        common = short.returns.index
        pd.testing.assert_frame_equal(self.wf.returns.loc[common], short.returns, rtol=1e-9, atol=1e-12)

    def test_costs_reduce_returns_and_turnover_ordering(self) -> None:
        free = self.opt.walk_forward(self.prices, 252, 21, cost_bps=0.0)
        self.assertTrue((free.returns.sum() > self.wf.returns.sum()).all())
        steady = self.wf.turnover["equal_weight"].iloc[1:].mean()  # only drift to correct
        active = self.wf.turnover["max_sharpe"].iloc[1:].mean()
        self.assertLess(steady, active)

    def test_walk_forward_validation(self) -> None:
        with self.assertRaises(OptimizationError):
            self.opt.walk_forward(self.prices, lookback=10)
        with self.assertRaises(OptimizationError):
            self.opt.walk_forward(self.prices, strategies=("nonsense",))
        with self.assertRaises(OptimizationError):
            PortfolioOptimizer(OptimizerConfig(min_weight=-0.5)).walk_forward(self.prices)


class FigureTests(unittest.TestCase):
    def test_figure_builds_when_plotly_present(self) -> None:
        try:
            import plotly.graph_objects  # noqa: F401
        except ImportError:
            self.skipTest("plotly not installed")
        res = PortfolioOptimizer().optimize_returns(make_returns(5, 400, 50))
        self.assertGreaterEqual(len(res.figure().data), 6)


if __name__ == "__main__":
    unittest.main()
