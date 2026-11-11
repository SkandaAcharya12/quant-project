"""Offline tests for the Data Engine: yfinance is replaced by deterministic synthetic data.

Runs with either ``pytest`` or ``python -m unittest``. No network access needed.
"""
from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import data_engine.data_engine as de
from data_engine import DataEngine, DataEngineConfig, DataFetchError, TechnicalIndicators as ti

DATES = pd.bdate_range("2022-01-03", periods=400, tz="America/New_York")


def make_ohlcv(seed: int, start_offset: int = 0, gap: tuple[int, int] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES) - start_offset
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    op = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum.reduce([close * (1 + np.abs(rng.normal(0, 0.004, n))), op, close])
    low = np.minimum.reduce([close * (1 - np.abs(rng.normal(0, 0.004, n))), op, close])
    df = pd.DataFrame(
        {"Open": op, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(100_000, 1_000_000, n).astype(float)},
        index=DATES[start_offset:],
    )
    return df.drop(df.index[gap[0]:gap[1]]) if gap else df


class FakeYahoo:
    """Stand-in for ``yfinance`` that also counts downloads per symbol."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self._lock = threading.Lock()
        self.Ticker = self._ticker

    def _ticker(self, symbol: str) -> SimpleNamespace:
        def history(**_: object) -> pd.DataFrame:
            with self._lock:
                self.calls[symbol] = self.calls.get(symbol, 0) + 1
                attempt = self.calls[symbol]
            time.sleep(0.02)
            if symbol == "FLAKY" and attempt < 3:
                raise ConnectionError("transient")
            if symbol == "DEAD":
                return pd.DataFrame()
            if symbol == "LATE":
                return make_ohlcv(3, start_offset=3)
            if symbol == "VERYLATE":
                return make_ohlcv(4, start_offset=200)
            if symbol == "GAPPY":
                return make_ohlcv(5, gap=(100, 103))
            return make_ohlcv(sum(map(ord, symbol)))

        return SimpleNamespace(history=history)


class DataEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.yahoo = FakeYahoo()
        patcher = mock.patch.object(de, "yf", self.yahoo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = DataEngineConfig(start="2022-01-01", retry_base_delay=0.01)
        self.engine = DataEngine(self.cfg)

    # ---- ingestion and cleaning -------------------------------------------
    def test_panel_cleaning_policy(self) -> None:
        panel = self.engine.get_price_panel(
            ["aaa", "BBB", "FLAKY", "DEAD", "LATE", "GAPPY", "VERYLATE", "aaa"]
        )
        rep = self.engine.last_report
        assert rep is not None
        self.assertIn("DEAD", rep.failed)                      # failed download reported
        self.assertIn("VERYLATE", rep.dropped_sparse)          # too sparse, dropped
        self.assertEqual(self.yahoo.calls["FLAKY"], 3)         # retried until success
        self.assertEqual(self.yahoo.calls["AAA"], 1)           # duplicates collapsed
        self.assertEqual(rep.ffill_counts["GAPPY"], 3)         # short gap forward-filled
        self.assertEqual(rep.bfill_counts["LATE"], 3)          # leading edge back-filled
        self.assertFalse(panel.isna().any().any())
        self.assertIsNone(panel.index.tz)
        self.assertTrue(np.isclose(panel["LATE"].iloc[0], make_ohlcv(3, 3)["Close"].iloc[0]))

    def test_all_failures_raise(self) -> None:
        with self.assertRaises(DataFetchError):
            self.engine.fetch_ohlcv(["DEAD"])

    def test_concurrent_callers_share_one_download(self) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            try:
                self.engine.get_price_panel(["ZZZ", "YYY", "XXX"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertTrue(all(self.yahoo.calls[s] == 1 for s in ("ZZZ", "YYY", "XXX")))

    # ---- indicators vs independent reference implementations ---------------
    def test_indicators_match_loop_references(self) -> None:
        df = make_ohlcv(11)
        c, h, lo = df["Close"], df["High"], df["Low"]

        a = 2 / 21
        ref = np.empty(len(c)); ref[0] = c.iloc[0]
        for i in range(1, len(c)):
            ref[i] = a * c.iloc[i] + (1 - a) * ref[i - 1]
        ref[:19] = np.nan
        np.testing.assert_allclose(ti.ema(c, 20).values, ref, equal_nan=True)

        w = c.values[-20:]
        z = (c.values[-1] - w.mean()) / w.std(ddof=1)
        self.assertAlmostEqual(ti.rolling_zscore(c, 20).iloc[-1], z)

        bb = ti.bollinger_bands(c, 20, 2.0)
        self.assertAlmostEqual(bb["bb_upper"].iloc[-1], w.mean() + 2 * w.std(ddof=0))

        rsi = ti.rsi(c, 14).dropna()
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())
        flat = pd.Series(50.0, index=c.index)
        self.assertTrue((ti.rsi(flat, 14).dropna() == 50).all())

        tr0 = max(h.iloc[1] - lo.iloc[1], abs(h.iloc[1] - c.iloc[0]), abs(lo.iloc[1] - c.iloc[0]))
        self.assertGreater(tr0, 0)
        self.assertEqual(ti.atr(h, lo, c, 14).dropna().index[0], c.index[13])

        np.testing.assert_allclose(ti.log_returns(c).dropna(), np.diff(np.log(c.values)))

    def test_features_are_causal(self) -> None:
        """Deleting the future must not change any past feature value."""
        df = make_ohlcv(11)
        df.index = df.index.tz_localize(None)
        full = self.engine.build_technical_features(df)
        cut = self.engine.build_technical_features(df.iloc[:300])
        pd.testing.assert_frame_equal(full.iloc[:300], cut, rtol=1e-12, atol=1e-12)

    # ---- macro -------------------------------------------------------------
    def test_macro_features(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=400)
        macro = pd.DataFrame(
            {"ust_10y": np.linspace(1.5, 4.5, 400), "ust_3m": np.linspace(0.0, 5.0, 400),
             "wti_crude": np.linspace(60, -5, 400)}, index=idx)
        feats = self.engine.build_macro_features(macro)
        self.assertEqual(feats["curve_inverted"].iloc[0], 0.0)
        self.assertEqual(feats["curve_inverted"].iloc[-1], 1.0)
        self.assertTrue(np.isnan(feats["wti_crude_logret"].iloc[-1]))   # negative price -> NaN

    def test_fred_requires_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(DataFetchError):
                self.engine.fetch_fred({"cpi": "CPIAUCSL"})


if __name__ == "__main__":
    unittest.main()


class PatchRegressionTests(unittest.TestCase):
    def test_macro_drops_weekend_rows_and_skips_jump_warnings(self) -> None:
        idx = pd.bdate_range("2022-01-03", periods=60)
        sunday = pd.DatetimeIndex([pd.Timestamp("2022-03-27")])  # a Sunday inside the range
        full = idx.union(sunday)

        def history(**_: object) -> pd.DataFrame:
            close = np.linspace(0.01, 4.0, len(full)) ** 2      # huge log moves, like ^IRX
            return pd.DataFrame({c: close for c in de.OHLCV_COLUMNS}, index=full)

        fake = SimpleNamespace(Ticker=lambda s: SimpleNamespace(history=history))
        cfg = DataEngineConfig(start="2022-01-01", macro_tickers={"vix": "^VIX", "ust_3m": "^IRX"})
        with mock.patch.object(de, "yf", fake):
            engine = DataEngine(cfg)
            with self.assertNoLogs(de.logger, level="WARNING"):
                macro = engine.fetch_macro()
        self.assertTrue((macro.index.dayofweek < 5).all())

    def test_tz_cache_configured_once_when_supported(self) -> None:
        setter = mock.Mock()
        fake = SimpleNamespace(set_tz_cache_location=setter, Ticker=lambda s: None)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(de, "yf", fake):
            DataEngine(DataEngineConfig(tz_cache_dir=tmp + "/tz"))
            setter.assert_called_once()
