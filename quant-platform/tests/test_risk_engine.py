"""Offline tests for the risk engine (closed forms, hand-computed paths, simulated data)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad

from risk_engine import (
    DEFAULT_MACRO_SCENARIOS,
    FactorScenario,
    RiskConfig,
    RiskEngine,
    RiskError,
    christoffersen_independence,
    cornish_fisher_quantile,
    drawdown_analysis,
    es_historical,
    es_parametric,
    ewma_variance,
    fhs_var_es,
    fit_factor_model,
    kupiec_pof,
    macro_factor_frame,
    risk_contributions,
    var_cornish_fisher,
    var_historical,
    var_parametric,
)


def idx(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def garch_returns(t: int, seed: int, omega: float = 2e-6, a: float = 0.08, b: float = 0.90) -> pd.Series:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(t)
    r, h = np.empty(t), omega / (1.0 - a - b)
    for i in range(t):  # test-data generator only
        r[i] = np.sqrt(h) * z[i]
        h = omega + a * r[i] ** 2 + b * h
    return pd.Series(r, index=idx(t, "2005-01-03"))


class PointEstimatorTests(unittest.TestCase):
    def test_historical_known_values(self) -> None:
        r = np.linspace(-0.05, 0.05, 101)  # 0.001 steps
        self.assertAlmostEqual(var_historical(r, 0.95), 0.045, places=12)
        self.assertAlmostEqual(es_historical(r, 0.95), np.mean([0.05, 0.049, 0.048, 0.047, 0.046, 0.045]), places=12)

    def test_normal_es_matches_numerical_integration(self) -> None:
        r = np.random.default_rng(1).normal(0.0005, 0.012, 3000)
        m, s = r.mean(), r.std(ddof=1)
        for a in (0.95, 0.99):
            var = var_parametric(r, a)
            tail_mean, _ = quad(lambda x: x * stats.norm.pdf(x, m, s), -np.inf, -var)
            self.assertAlmostEqual(es_parametric(r, a), -tail_mean / (1 - a), places=8)
            self.assertAlmostEqual(var, -(m + stats.norm.ppf(1 - a) * s), places=12)

    def test_methods_agree_on_normal_data_and_ordering_holds(self) -> None:
        r = np.random.default_rng(2).normal(0.0003, 0.01, 200_000)
        for a in (0.95, 0.99):
            self.assertAlmostEqual(var_historical(r, a) / var_parametric(r, a), 1.0, delta=0.03)
            self.assertGreater(es_historical(r, a), var_historical(r, a))
            self.assertGreater(es_parametric(r, a), var_parametric(r, a))
        self.assertGreater(var_historical(r, 0.99), var_historical(r, 0.95))

    def test_cornish_fisher_accuracy_and_its_documented_limit(self) -> None:
        self.assertEqual(cornish_fisher_quantile(-2.326, 0.0, 0.0), -2.326)
        r = stats.t.rvs(df=6, size=1_000_000, random_state=3) * 0.01  # fat tails (finite kurtosis)
        emp95, emp99 = var_historical(r, 0.95), var_historical(r, 0.99)
        # Near 95% Cornish-Fisher is far closer to the truth than the normal quantile ...
        self.assertLess(abs(var_cornish_fisher(r, 0.95) - emp95), 0.2 * abs(var_parametric(r, 0.95) - emp95))
        # ... but at 99% with fat tails it overshoots by MORE than the normal undershoots (see docstring).
        self.assertGreater(var_cornish_fisher(r, 0.99), emp99)
        self.assertGreater(abs(var_cornish_fisher(r, 0.99) - emp99), abs(var_parametric(r, 0.99) - emp99))

    def test_ewma_matches_loop(self) -> None:
        r = np.random.default_rng(4).normal(0, 0.01, 500)
        ref = np.empty(500)
        ref[0] = r[0] ** 2
        for t in range(1, 500):
            ref[t] = 0.94 * ref[t - 1] + 0.06 * r[t] ** 2
        np.testing.assert_allclose(ewma_variance(r, 0.94), ref, rtol=1e-12)

    def test_fhs_reacts_to_a_volatility_spike(self) -> None:
        rng = np.random.default_rng(5)
        r = np.concatenate([rng.normal(0, 0.005, 1000), rng.normal(0, 0.02, 30)])
        fhs_var, fhs_es = fhs_var_es(r, 0.99)
        self.assertGreater(fhs_var, 2.0 * var_historical(r, 0.99))
        self.assertGreater(fhs_es, fhs_var)

    def test_multi_day_horizon_scaling(self) -> None:
        r = np.random.default_rng(6).normal(0, 0.01, 4000)
        zero_mean = r - r.mean()  # with m = 0 the horizon scaling is exactly sqrt(h)
        self.assertAlmostEqual(var_parametric(zero_mean, 0.99, 10) / var_parametric(zero_mean, 0.99, 1),
                               np.sqrt(10), places=10)
        table = RiskEngine(RiskConfig(horizon_days=10)).var_table(pd.Series(r, index=idx(4000)))
        self.assertGreater(table.loc[("Historical", 0.99), "var"], 0.05)


class BacktestStatTests(unittest.TestCase):
    def test_kupiec_matches_manual_formula(self) -> None:
        t, x, p = 250, 5, 0.01
        pi = x / t
        lr = -2 * ((t - x) * np.log(1 - p) + x * np.log(p) - (t - x) * np.log(1 - pi) - x * np.log(pi))
        got_lr, got_p = kupiec_pof(t, x, p)
        self.assertAlmostEqual(got_lr, lr, places=10)
        self.assertAlmostEqual(got_p, stats.chi2.sf(lr, 1), places=10)
        lr0, _ = kupiec_pof(250, 0, 0.01)  # zero exceptions: 0*log(0) must be 0, not NaN
        self.assertAlmostEqual(lr0, -2 * 250 * np.log(0.99), places=10)
        self.assertAlmostEqual(kupiec_pof(1000, 10, 0.01)[0], 0.0, places=10)  # exactly as expected
        with self.assertRaises(RiskError):
            kupiec_pof(100, 101, 0.01)

    def test_christoffersen_manual_and_clustering(self) -> None:
        e = np.zeros(60, dtype=bool)
        e[[5, 6, 20, 40, 41, 42]] = True
        prev, cur = e[:-1], e[1:]
        n00, n01 = np.sum(~prev & ~cur), np.sum(~prev & cur)
        n10, n11 = np.sum(prev & ~cur), np.sum(prev & cur)
        pi, pi01, pi11 = (n01 + n11) / 59, n01 / (n00 + n01), n11 / (n10 + n11)
        ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        ll1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        lr, p = christoffersen_independence(e)
        self.assertAlmostEqual(lr, -2 * (ll0 - ll1), places=10)
        clustered = np.zeros(1000, dtype=bool)
        clustered[100:110] = clustered[500:510] = True
        self.assertLess(christoffersen_independence(clustered)[1], 0.001)
        independent = np.random.default_rng(7).random(1000) < 0.05
        self.assertGreater(christoffersen_independence(independent)[1], 0.05)
        self.assertEqual(christoffersen_independence(np.zeros(100, dtype=bool)), (0.0, 1.0))


class BacktestEngineTests(unittest.TestCase):
    def test_no_lookahead_in_any_method(self) -> None:
        r = pd.Series(np.random.default_rng(8).normal(0, 0.01, 1200), index=idx(1200))
        eng = RiskEngine(RiskConfig(backtest_window=250))
        for method in ("historical", "parametric", "ewma", "fhs"):
            full = eng.var_forecasts(r, 0.99, method)
            cut = eng.var_forecasts(r.iloc[:800], 0.99, method)
            pd.testing.assert_series_equal(full.iloc[:800], cut, check_names=False, rtol=1e-10)
            self.assertTrue(full.iloc[:251].isna().all())  # window + 1, identical for every method
            self.assertTrue(full.iloc[251:].notna().all())

    def test_iid_normal_data_is_calibrated(self) -> None:
        r = pd.Series(np.random.default_rng(9).normal(0.0003, 0.01, 4000), index=idx(4000))
        table = RiskEngine(RiskConfig(backtest_window=500)).backtest_table(r, methods=("historical", "parametric"))
        self.assertTrue((table["rate"].xs(0.95, level="confidence").between(0.035, 0.065)).all())
        self.assertTrue((table["rate"].xs(0.99, level="confidence").between(0.005, 0.017)).all())
        self.assertTrue((table["kupiec_p"] > 0.01).all())
        self.assertTrue((table.xs(0.95, level="confidence")["basel_zone_last250"] == "").all())
        self.assertIn(table.xs(0.99, level="confidence")["basel_zone_last250"].iloc[0], {"green", "yellow", "red"})

    def test_adaptive_methods_beat_rolling_windows_under_garch_clustering(self) -> None:
        eng = RiskEngine(RiskConfig(backtest_window=500))
        reject = {m: {0.95: 0, 0.99: 0} for m in ("historical", "parametric", "ewma", "fhs")}
        seeds = range(100, 112)
        for seed in seeds:
            table = eng.backtest_table(garch_returns(5000, seed))
            for (method, conf), row in table.iterrows():
                reject[method][conf] += int(row["cond_coverage_p"] < 0.05)
        for conf in (0.95, 0.99):
            self.assertLess(reject["fhs"][conf], reject["historical"][conf])
            self.assertLess(reject["fhs"][conf], reject["parametric"][conf])
            self.assertLessEqual(reject["fhs"][conf], 2)  # measured: 0 of 12


class DrawdownTests(unittest.TestCase):
    def test_hand_computed_path(self) -> None:
        r = pd.Series([0.10, -0.20, -0.10, 0.30, 0.05, -0.05, 0.10], index=idx(7))
        dd = drawdown_analysis(r)
        equity = np.cumprod(1 + r.to_numpy())
        np.testing.assert_allclose(dd.series["equity"], equity)
        self.assertAlmostEqual(dd.max_drawdown, 0.792 / 1.1 - 1, places=12)
        self.assertEqual(dd.longest_underwater_bars, 5)
        self.assertEqual(len(dd.episodes), 1)
        ep = dd.episodes.iloc[0]
        self.assertEqual(ep["peak_date"], r.index[0])
        self.assertEqual(ep["trough_date"], r.index[2])
        self.assertEqual(ep["recovery_date"], r.index[6])
        self.assertTrue(bool(ep["recovered"]))
        self.assertEqual(dd.current_drawdown, 0.0)

    def test_unrecovered_drawdown_from_start(self) -> None:
        dd = drawdown_analysis(pd.Series([-0.10, -0.10], index=idx(2)))
        self.assertAlmostEqual(dd.max_drawdown, -0.19, places=12)
        self.assertFalse(bool(dd.episodes.iloc[0]["recovered"]))
        self.assertTrue(pd.isna(dd.episodes.iloc[0]["recovery_date"]))
        self.assertAlmostEqual(dd.current_drawdown, -0.19, places=12)

    def test_top_n_sorted_by_depth(self) -> None:
        r = garch_returns(3000, 1)
        eps = drawdown_analysis(r, top_n=4).episodes
        self.assertLessEqual(len(eps), 4)
        self.assertTrue(eps["depth"].is_monotonic_increasing)


class DecompositionAndStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20)
        n, t = 4, 2000
        f = pd.DataFrame({"equity": rng.normal(0, 0.01, t), "rates_bp": rng.normal(0, 5.0, t),
                          "usd": rng.normal(0, 0.005, t), "oil": rng.normal(0, 0.02, t)}, index=idx(t))
        cls.true_beta = np.array([[1.0, 0.3, 0.8, 0.1], [-0.002, 0.004, -0.001, 0.003],
                                  [-0.2, 0.1, 0.0, 0.3], [0.05, 0.0, 0.15, -0.02]])
        y = f.to_numpy() @ cls.true_beta + rng.normal(0, 0.003, (t, n))
        cls.prices = pd.DataFrame(100 * np.exp(np.cumsum(y, axis=0)), index=f.index, columns=list("ABCD"))
        cls.factors = f
        cls.weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("ABCD"))

    def test_euler_contributions_sum_to_volatility(self) -> None:
        cov = self.prices.pct_change().dropna().cov() * 252
        rc = risk_contributions(self.weights, cov)
        vol = np.sqrt(self.weights.to_numpy() @ cov.to_numpy() @ self.weights.to_numpy())
        self.assertAlmostEqual(rc["contribution"].sum(), vol, places=12)
        self.assertAlmostEqual(rc["pct_of_risk"].sum(), 1.0, places=12)
        w, c, eps = self.weights.to_numpy(), cov.to_numpy(), 1e-6  # marginal risk = d(vol)/d(w_i)
        for i in range(4):
            up, dn = w.copy(), w.copy()
            up[i] += eps
            dn[i] -= eps
            fd = (np.sqrt(up @ c @ up) - np.sqrt(dn @ c @ dn)) / (2 * eps)
            self.assertAlmostEqual(rc["marginal_risk"].iloc[i], fd, places=6)

    def test_historical_stress_known_windows(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=50)
        prices = pd.DataFrame({"A": np.linspace(100, 200, 50), "B": 100.0}, index=dates)
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        scen = {"covered": ("2020-01-01", str(dates[-1].date())), "outside": ("2008-09-12", "2008-11-20")}
        table = RiskEngine(RiskConfig(portfolio_value=1_000_000)).historical_stress(prices, w, scen)
        self.assertEqual(list(table.index), ["covered"])  # uncovered scenario is skipped
        self.assertAlmostEqual(table.loc["covered", "portfolio_return"], 0.5 * (prices["A"].iloc[-1] / 100 - 1))
        self.assertAlmostEqual(table.loc["covered", "pnl_usd"], table.loc["covered", "portfolio_return"] * 1e6)
        self.assertEqual(table.loc["covered", "worst_asset"], "B")
        empty = RiskEngine().historical_stress(prices, w, {"outside": ("2008-09-12", "2008-11-20")})
        self.assertTrue(empty.empty)

    def test_factor_model_recovers_betas_and_stress_is_exact(self) -> None:
        model = fit_factor_model(self.prices, self.factors)
        # OLS standard error of each beta is noise_sd / (factor_sd * sqrt(T)); allow 4 of them.
        se = 0.003 / (self.factors.std().to_numpy() * np.sqrt(len(self.factors)))
        err = np.abs(model.betas.to_numpy() - self.true_beta)
        self.assertTrue((err < 4 * se[:, None]).all(), err / se[:, None])
        self.assertTrue((model.r2 > 0.5).all())
        sc = FactorScenario("test", {"equity": -0.20, "rates_bp": 100.0, "oil": 0.30})
        stress = RiskEngine().factor_stress(model, self.weights, (sc,))
        shock = np.array([np.log1p(-0.20), 100.0, 0.0, np.log1p(0.30)])  # factor order: equity, rates, usd, oil
        expected_assets = np.expm1(shock @ model.betas.to_numpy())
        np.testing.assert_allclose(stress.filter(like="asset_").iloc[0].to_numpy(), expected_assets)
        self.assertAlmostEqual(stress.loc["test", "portfolio_return"], float(self.weights.to_numpy() @ expected_assets))

    def test_reverse_stress_hits_the_target(self) -> None:
        model = fit_factor_model(self.prices, self.factors)
        sc = FactorScenario("crash", {"equity": -0.10, "rates_bp": 20.0, "oil": -0.05})
        eng = RiskEngine()
        k = eng.reverse_stress(model, self.weights, sc, target_loss=0.15)
        self.assertGreater(k, 1.0)
        from risk_engine.risk_engine import _asset_shocks  # internal helper, used to verify the root
        self.assertAlmostEqual(float(self.weights.to_numpy() @ _asset_shocks(model, sc, k)), -0.15, places=8)
        gain = FactorScenario("up", {"equity": 0.05})  # a gain can never produce the requested loss
        self.assertTrue(np.isnan(eng.reverse_stress(model, self.weights, gain, 0.2)))

    def test_default_scenarios_skip_when_factor_missing(self) -> None:
        model = fit_factor_model(self.prices, self.factors.drop(columns=["usd"]))
        out = RiskEngine().factor_stress(model, self.weights)  # all defaults use usd -> all skipped
        self.assertTrue(out.empty)
        full = RiskEngine().factor_stress(fit_factor_model(self.prices, self.factors), self.weights)
        self.assertEqual(len(full), len(DEFAULT_MACRO_SCENARIOS))

    def test_scenario_validation(self) -> None:
        model = fit_factor_model(self.prices, self.factors)
        with self.assertRaises(RiskError):
            RiskEngine().factor_stress(model, self.weights, (FactorScenario("x", {"nonsense": 0.1}),))
        with self.assertRaises(RiskError):
            RiskEngine().factor_stress(model, self.weights, (FactorScenario("x", {"equity": -1.2}),))

    def test_macro_factor_frame(self) -> None:
        cols = {"sp500_logret": 0.001, "d_ust_10y_bp": 1.0, "usd_index_logret": 0.0, "wti_crude_logret": 0.0}
        macro = pd.DataFrame(cols, index=idx(5))
        self.assertEqual(list(macro_factor_frame(macro).columns), ["equity", "rates_bp", "usd", "oil"])
        with self.assertRaises(RiskError):
            macro_factor_frame(macro.drop(columns=["d_ust_10y_bp"]))
        self.assertGreaterEqual(len(DEFAULT_MACRO_SCENARIOS), 5)


class IntegrationAndValidationTests(unittest.TestCase):
    def test_analyze_produces_every_table(self) -> None:
        stress = DecompositionAndStressTests
        stress.setUpClass()
        eng = RiskEngine(RiskConfig(portfolio_value=500_000, backtest_window=250))
        report = eng.analyze(stress.prices, stress.weights, factors=stress.factors,
                             historical_scenarios={"all": ("2015-01-05", "2022-01-01")})
        self.assertEqual(len(report.var_table), 8)  # 4 methods x 2 confidence levels
        self.assertAlmostEqual(report.var_table.loc[("Historical", 0.99), "var_usd"],
                               report.var_table.loc[("Historical", 0.99), "var"] * 500_000)
        self.assertEqual(len(report.backtest), 8)
        self.assertEqual(len(report.contributions), 4)
        self.assertIsNotNone(report.factor_stress)
        text = report.summary()
        for needle in ("VaR / ES", "Max drawdown", "VaR backtest", "Macro factor stress"):
            self.assertIn(needle, text)

    def test_validation(self) -> None:
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=idx(300))
        with self.assertRaises(RiskError):
            var_historical(r.to_numpy(), 1.2)
        with self.assertRaises(RiskError):
            RiskEngine().var_table(r.iloc[:10])
        bad = r.copy()
        bad.iloc[5] = np.nan
        with self.assertRaises(RiskError):
            RiskEngine().var_table(bad)
        with self.assertRaises(RiskError):
            RiskEngine().var_forecasts(r, 0.99, "nonsense", 100)
        with self.assertRaises(RiskError):
            RiskConfig(confidence_levels=(1.5,))
        with self.assertRaises(RiskError):
            RiskConfig(backtest_window=10)
        with self.assertRaises(RiskError):
            RiskEngine.portfolio_returns(pd.DataFrame({"A": [0.1] * 5}), pd.Series({"B": 1.0}))


if __name__ == "__main__":
    unittest.main()
