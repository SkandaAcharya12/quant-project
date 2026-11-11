"""Offline tests for the end-to-end study functions in main.py (synthetic prices, no network)."""
from __future__ import annotations

import contextlib
import io
import unittest

import pandas as pd

import numpy as np

import main
from tests.test_backtester import make_pair
from tests.test_optimizer import make_returns, to_prices


def _quiet(func, *args):  # type: ignore[no-untyped-def]
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


class MainStudyTests(unittest.TestCase):
    def test_pairs_study_selects_on_formation_and_reports_oos(self) -> None:
        pair = make_pair(n=2200, seed=61)
        pair.index = pd.bdate_range("2018-01-02", periods=2200)
        prices = pair.rename(columns={"Y": "XOM", "X": "CVX"})
        prices["KO"] = make_pair(n=2200, seed=62)["X"].to_numpy()
        result = _quiet(main.run_pairs_backtest, prices, "2024-12-31")
        self.assertEqual(set(result.pair), {"XOM", "CVX"})
        oos = result.performance(start="2025-01-01")
        self.assertIn("sharpe", oos)
        self.assertLess(len(result.frame.loc["2025-01-01":]), len(result.frame))

    def test_optimizer_study_runs_and_separates_periods(self) -> None:
        returns = make_returns(8, 2200, seed=3)
        returns.index = pd.bdate_range("2018-01-02", periods=2200)
        prices = to_prices(returns).iloc[1:]
        prices.columns = main.OPTIMIZER_UNIVERSE
        study = _quiet(main.run_optimizer_study, prices, "2024-12-31", "2025-01-01", 0.03)
        table = study["in_vs_out"]
        self.assertEqual(len(table), 3)
        self.assertTrue({"Sharpe in-sample", "Sharpe out-of-sample"} <= set(table.columns))
        self.assertGreaterEqual(study["wf_metrics"].index.size, 3)
        # Walk-forward metrics must cover only the requested out-of-sample window.
        self.assertGreaterEqual(study["walk_forward"].returns.loc["2025-01-01":].index[0],
                                pd.Timestamp("2025-01-01"))


if __name__ == "__main__":
    unittest.main()


class MainRiskStudyTests(unittest.TestCase):
    def test_risk_study_runs_and_reports_oos(self) -> None:
        returns = make_returns(8, 2200, seed=8)
        returns.index = pd.bdate_range("2018-01-02", periods=2200)
        prices = to_prices(returns).iloc[1:]
        prices.columns = main.OPTIMIZER_UNIVERSE
        rng = np.random.default_rng(1)
        macro = pd.DataFrame({"sp500_logret": rng.normal(0, 0.01, 2200), "d_ust_10y_bp": rng.normal(0, 5, 2200),
                              "usd_index_logret": rng.normal(0, 0.004, 2200),
                              "wti_crude_logret": rng.normal(0, 0.02, 2200)}, index=prices.index)
        weights = pd.Series(1 / 8, index=prices.columns)
        report = _quiet(main.run_risk_study, prices, weights, macro, "2025-01-01")
        self.assertEqual(len(report.var_table), 8)
        self.assertIsNotNone(report.factor_stress)
        self.assertFalse(report.factor_stress.empty)
        # also works without macro data (factor stress simply skipped)
        report2 = _quiet(main.run_risk_study, prices, weights, None, "2025-01-01")
        self.assertIsNone(report2.factor_stress)
