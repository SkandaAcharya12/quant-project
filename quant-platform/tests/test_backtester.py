"""Offline tests for the pairs backtester (synthetic data with known ground truth).

Run with ``python -m pytest`` or ``python -m unittest``. The statsmodels cross-check
runs only where statsmodels is installed (it is, via requirements.txt).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import backtester.backtester as bt
from backtester import (
    BacktestError,
    PairsBacktestConfig,
    PairsBacktester,
    build_signal,
    engle_granger_test,
    rolling_engle_granger,
    screen_pairs,
)

try:
    from statsmodels.tsa.stattools import coint

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False


def make_pair(
    n: int = 1500, beta: float = 1.5, half_life: float = 5.0, sigma: float = 0.02, seed: int = 0
) -> pd.DataFrame:
    """Y = 30 * X**beta * exp(OU noise): cointegrated with known beta and half-life.

    Default half-life is 5 days: for an AR(1) spread the expected ADF t-statistic over
    n = 252 is (phi - 1) / sqrt((1 - phi^2) / n), about -4.2 at half-life 5 but only
    about -2.95 at half-life 10 (below the -3.34 critical value, i.e. the test lacks
    power). ``test_adf_power_matches_theory`` pins that down.
    """
    rng = np.random.default_rng(seed)
    lx = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    phi = 0.5 ** (1.0 / half_life)
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), n)
    e = np.empty(n)
    e[0] = rng.normal(0.0, sigma)
    for t in range(1, n):  # test-data generator only
        e[t] = phi * e[t - 1] + innov[t]
    ly = np.log(30.0) + beta * lx + e
    return pd.DataFrame(
        {"Y": np.exp(ly), "X": np.exp(lx)}, index=pd.bdate_range("2018-01-01", periods=n)
    )


def make_broken_pair(n: int = 1800, brk: int = 900, seed: int = 0) -> pd.DataFrame:
    """Cointegrated until bar ``brk``; afterwards the spread is a random walk."""
    rng = np.random.default_rng(seed)
    lx = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    phi = 0.5 ** (1.0 / 5.0)
    step = 0.02 * np.sqrt(1.0 - phi**2)
    e = np.empty(n)
    e[0] = rng.normal(0.0, 0.02)
    for t in range(1, n):
        e[t] = (phi if t < brk else 1.0) * e[t - 1] + rng.normal(0.0, step * (1.0 if t < brk else 2.5))
    return pd.DataFrame(
        {"Y": np.exp(np.log(30.0) + 1.5 * lx + e), "X": np.exp(lx)},
        index=pd.bdate_range("2018-01-01", periods=n),
    )


def make_random_walks(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, (n, 2))
    return pd.DataFrame(
        50.0 * np.exp(np.cumsum(steps, axis=0)),
        columns=["Y", "X"],
        index=pd.bdate_range("2018-01-01", periods=n),
    )


def adf_reference(y: np.ndarray, x: np.ndarray, p: int) -> tuple[float, float, float]:
    """Independent loop implementation: OLS, residuals, ADF t-stat via lstsq."""
    design = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    e = y - a - b * x
    de = np.diff(e)
    dep = de[p:]
    cols = [e[p:-1]] + [de[p - k : len(de) - k] for k in range(1, p + 1)]
    xmat = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(xmat, dep, rcond=None)
    resid = dep - xmat @ coef
    s2 = resid @ resid / (len(dep) - xmat.shape[1])
    cov = s2 * np.linalg.inv(xmat.T @ xmat)
    return coef[0] / np.sqrt(cov[0, 0]), b, a


class CointegrationTests(unittest.TestCase):
    def test_adf_statistic_matches_independent_ols(self) -> None:
        for seed, maker in ((1, make_pair), (2, make_random_walks)):
            df = maker(n=400, seed=seed)
            ly, lx = np.log(df["Y"].to_numpy()), np.log(df["X"].to_numpy())
            for p in (0, 1, 2):
                res = rolling_engle_granger(ly, lx, window=120, adf_lags=p, chunk_size=64)
                for end in (119, 200, 333, 399):  # spans chunk boundaries
                    stat, beta, alpha = adf_reference(ly[end - 119 : end + 1], lx[end - 119 : end + 1], p)
                    self.assertAlmostEqual(res["adf_stat"].iloc[end], stat, places=8)
                    self.assertAlmostEqual(res["beta"].iloc[end], beta, places=10)
                    self.assertAlmostEqual(res["alpha"].iloc[end], alpha, places=10)

    @unittest.skipUnless(HAS_STATSMODELS, "statsmodels not installed")
    def test_matches_statsmodels_coint(self) -> None:
        for seed, maker in ((3, make_pair), (4, make_random_walks)):
            df = maker(n=300, seed=seed)
            ly, lx = np.log(df["Y"].to_numpy()), np.log(df["X"].to_numpy())
            ours = engle_granger_test(ly, lx, adf_lags=1)
            stat, pval, _ = coint(ly, lx, trend="c", maxlag=1, autolag=None)
            self.assertAlmostEqual(ours["adf_stat"], stat, places=6)
            self.assertAlmostEqual(ours["pvalue"], pval, places=6)

    def test_adf_power_matches_theory(self) -> None:
        """Observed median ADF t-stat should match (phi-1)/sqrt((1-phi^2)/W) within sampling error."""
        for hl in (5.0, 10.0):
            df = make_pair(n=2500, half_life=hl, seed=9)
            res = rolling_engle_granger(np.log(df["Y"]).to_numpy(), np.log(df["X"]).to_numpy(), 252)
            phi = 0.5 ** (1.0 / hl)
            theory = (phi - 1.0) / np.sqrt((1.0 - phi**2) / 252)
            self.assertAlmostEqual(res["adf_stat"].median(), theory, delta=0.6)

    def test_recovers_beta_and_half_life(self) -> None:
        df = make_pair(n=1500, beta=1.5, half_life=5.0, seed=5)
        res = rolling_engle_granger(np.log(df["Y"]).to_numpy(), np.log(df["X"]).to_numpy(), 252)
        tail = res.iloc[252:]
        self.assertAlmostEqual(tail["beta"].median(), 1.5, delta=0.05)
        self.assertAlmostEqual(tail["half_life"].median(), 5.0, delta=2.0)
        self.assertLess(tail["adf_stat"].median(), -3.34)  # rejects "no cointegration"

    def test_independent_random_walks_rarely_reject(self) -> None:
        shares = []
        for seed in range(8):
            df = make_random_walks(n=800, seed=100 + seed)
            res = rolling_engle_granger(np.log(df["Y"]).to_numpy(), np.log(df["X"]).to_numpy(), 252)
            shares.append((res["adf_stat"].dropna() < -3.34).mean())
        self.assertLess(float(np.mean(shares)), 0.20)  # nominal size is 5%; windows overlap

    def test_screen_pairs_ranks_true_pair_first(self) -> None:
        pair = make_pair(n=600, seed=6)
        pair["Z"] = make_random_walks(n=600, seed=7)["Y"].to_numpy()
        table = screen_pairs(pair)
        self.assertEqual(set(table.iloc[0][["y", "x"]]), {"Y", "X"})
        self.assertIn("pvalue_bonferroni", table.columns)
        self.assertTrue((table["pvalue_bonferroni"] >= table["pvalue"]).all())


class SignalTests(unittest.TestCase):
    def test_state_machine_equals_sequential_loop(self) -> None:
        rng = np.random.default_rng(11)
        n = 5000
        flat = rng.random(n) < 0.15
        long_e = (rng.random(n) < 0.10) & ~flat
        short_e = (rng.random(n) < 0.10) & ~flat & ~long_e
        vec = build_signal(flat, long_e, short_e).to_numpy()
        state, ref = 0.0, np.empty(n)
        for t in range(n):
            if flat[t]:
                state = 0.0
            elif long_e[t]:
                state = 1.0
            elif short_e[t]:
                state = -1.0
            ref[t] = state
        np.testing.assert_array_equal(vec, ref)


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = make_pair(n=1500, seed=21)
        cls.cfg = PairsBacktestConfig()
        cls.result = PairsBacktester(cls.cfg).run(cls.prices, "Y", "X")

    def test_trades_happen_and_are_profitable_on_cointegrated_data(self) -> None:
        r = self.result
        self.assertGreater(len(r.trades), 10)
        self.assertGreater(r.metrics["total_return"], 0.0)
        self.assertGreater(r.trades["gross_return"].mean(), 0.0)
        self.assertLessEqual(r.trades["net_return"].mean(), r.trades["gross_return"].mean())

    def test_gross_exposure_is_one_when_in_market(self) -> None:
        f = self.result.frame
        held = f["position"] != 0
        gross = (f["w_y"].abs() + f["w_x"].abs())[held]
        np.testing.assert_allclose(gross, 1.0, atol=1e-12)

    def test_pnl_matches_literal_reference_loop(self) -> None:
        f, cfg = self.result.frame, self.cfg
        k = cfg.fill_delay_bars + 1
        sig, beta = f["signal"].to_numpy(), f["beta"].to_numpy()
        py, px = f["price_Y"].to_numpy(), f["price_X"].to_numpy()
        n = len(f)
        frozen = np.full(n, np.nan)
        cur = np.nan
        for t in range(n):
            prev = sig[t - 1] if t > 0 else 0.0
            if sig[t] != 0 and sig[t] != prev:
                cur = beta[t]
            frozen[t] = cur
        wy, wx = np.zeros(n), np.zeros(n)
        for s in range(k, n):
            pos = sig[s - k]
            if pos != 0:
                b = frozen[s - k]
                wy[s], wx[s] = pos / (1 + abs(b)), -pos * b / (1 + abs(b))
        net = np.zeros(n)
        for s in range(1, n):
            gross = wy[s] * (py[s] / py[s - 1] - 1) + wx[s] * (px[s] / px[s - 1] - 1)
            turn = abs(wy[s] - wy[s - 1]) + abs(wx[s] - wx[s - 1])
            short = max(-wy[s], 0) + max(-wx[s], 0)
            net[s] = (
                gross
                - turn * (cfg.commission_bps + cfg.slippage_bps) / 1e4
                - short * cfg.borrow_bps_annual / 1e4 / cfg.trading_days
            )
        np.testing.assert_allclose(f["net_return"].to_numpy(), net, atol=1e-12)

    def test_trade_log_reconciles_with_equity_curve(self) -> None:
        r = self.result
        total_log = np.log1p(r.frame["net_return"]).sum()
        trade_log = np.log1p(r.trades["net_return"]).sum()
        self.assertAlmostEqual(total_log, trade_log, places=10)
        self.assertTrue(set(r.trades["exit_reason"]) <= {"target", "stop_loss", "coint_break", "reversal", "data", "open"})

    def test_no_lookahead_truncating_the_future_changes_nothing(self) -> None:
        cut = 1100
        short = PairsBacktester(self.cfg).run(self.prices.iloc[:cut], "Y", "X").frame
        cols = ["beta", "alpha", "spread", "zscore", "adf_stat", "half_life", "signal",
                "position", "w_y", "w_x", "net_return"]
        pd.testing.assert_frame_equal(
            self.result.frame.iloc[:cut][cols], short[cols], rtol=1e-9, atol=1e-12
        )

    def test_costs_and_delay_reduce_performance(self) -> None:
        free = PairsBacktester(
            PairsBacktestConfig(commission_bps=0, slippage_bps=0, borrow_bps_annual=0)
        ).run(self.prices, "Y", "X")
        self.assertGreater(free.metrics["total_return"], self.result.metrics["total_return"])
        self.assertAlmostEqual(free.metrics["total_costs"], 0.0, places=12)

    def test_fill_delay_shifts_positions_exactly(self) -> None:
        """Delay changes only WHEN positions start; the signals themselves are identical."""
        fast = PairsBacktester(PairsBacktestConfig(fill_delay_bars=0)).run(self.prices, "Y", "X").frame
        slow = PairsBacktester(PairsBacktestConfig(fill_delay_bars=3)).run(self.prices, "Y", "X").frame
        pd.testing.assert_series_equal(fast["signal"], slow["signal"])
        np.testing.assert_array_equal(slow["position"].to_numpy()[3:], fast["position"].to_numpy()[:-3])

    def test_both_zscore_modes_run(self) -> None:
        for mode in ("formation", "rolling"):
            res = PairsBacktester(PairsBacktestConfig(zscore_mode=mode)).run(self.prices, "Y", "X")
            self.assertGreater(len(res.trades), 5, mode)
        with self.assertRaises(BacktestError):
            PairsBacktestConfig(zscore_mode="nonsense")  # type: ignore[arg-type]

    def test_cointegration_gate_limits_tail_risk_when_relationship_breaks(self) -> None:
        """The gate is insurance, not alpha: over 40 seeds it gave a shallower drawdown in all of
        them but a higher return in only about half. Assert the robust property (drawdown)."""
        for seed in range(4):
            df = make_broken_pair(seed=400 + seed)
            when = df.index[900]
            post = {}
            for gate in (True, False):
                cfg = PairsBacktestConfig(
                    entry_pvalue=0.05 if gate else 1.0, break_pvalue=0.15 if gate else 1.0
                )
                post[gate] = PairsBacktester(cfg).run(df, "Y", "X").performance(start=when)
            self.assertGreater(post[True]["max_drawdown"], post[False]["max_drawdown"], seed)
            self.assertGreater(post[True]["max_drawdown"], -0.20, seed)

    def test_input_validation(self) -> None:
        bt_ = PairsBacktester(self.cfg)
        with self.assertRaises(BacktestError):
            bt_.run(self.prices, "Y", "Y")
        with self.assertRaises(BacktestError):
            bt_.run(self.prices, "Y", "NOPE")
        with self.assertRaises(BacktestError):
            bt_.run(self.prices.iloc[:100], "Y", "X")
        bad = self.prices.copy()
        bad.iloc[10, 0] = np.nan
        with self.assertRaises(BacktestError):
            bt_.run(bad, "Y", "X")
        with self.assertRaises(BacktestError):
            PairsBacktestConfig(entry_z=1.0, exit_z=2.0)
        with self.assertRaises(BacktestError):
            PairsBacktestConfig(entry_pvalue=0.5, break_pvalue=0.1)

    def test_export(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self.result.to_csv(tmp)
            self.assertTrue((Path(tmp) / "trades.csv").exists())
            data = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertIn("sharpe", data)


if __name__ == "__main__":
    unittest.main()
