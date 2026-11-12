"""Tests for the dashboard.

Part 1 tests the computation layer directly. Part 2 executes the whole ``main()`` UI script against
FAKE ``streamlit`` and ``plotly`` modules and synthetic market data: it proves the wiring (every
tab renders, every chart function runs, no error banners), but not how the real libraries draw it.
Open the app with ``streamlit run app.py`` for the visual check.
"""
from __future__ import annotations

import datetime as dt
import importlib
import sys
import types
import unittest
import zlib
from unittest import mock

import numpy as np
import pandas as pd
from typing_extensions import Self

import data_engine.data_engine as de
from tests.test_optimizer import make_returns, to_prices

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
BASES = {"^VIX": 20.0, "^TNX": 4.0, "^FVX": 3.8, "^IRX": 3.5, "DX-Y.NYB": 100.0,
         "GC=F": 1800.0, "CL=F": 70.0, "^GSPC": 3000.0}


class FakeYahoo:
    """Deterministic synthetic OHLCV for any symbol, 2017-12 to 2026-11."""

    def __init__(self) -> None:
        self.Ticker = lambda symbol: types.SimpleNamespace(history=lambda **kw: self._history(symbol, **kw))

    @staticmethod
    def _history(symbol: str, start: str = "2017-12-01", **_: object) -> pd.DataFrame:
        idx = pd.bdate_range("2017-12-01", "2026-11-06")
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        close = BASES.get(symbol, 100.0) * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(idx))))
        df = pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
                           "Volume": 1e6}, index=idx.tz_localize("America/New_York"))
        return df.loc[df.index >= pd.Timestamp(start).tz_localize("America/New_York")]


class _Ctx:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeStreamlit(types.ModuleType):
    """Minimal Streamlit: widgets return their defaults (or per-label overrides); calls are recorded."""

    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        super().__init__("streamlit")
        self.overrides = overrides or {}
        self.calls: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.session_state: dict[str, object] = {}
        self.tab_labels: list[str] = []
        self.sidebar = _Widgets(self)

    # decorators / containers
    def cache_data(self, **_: object):  # type: ignore[no-untyped-def]
        return lambda f: f

    def columns(self, spec):  # type: ignore[no-untyped-def]
        return [_Widgets(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):  # type: ignore[no-untyped-def]
        self.tab_labels = list(labels)
        return [_Widgets(self) for _ in labels]

    def expander(self, *_a: object, **_k: object) -> _Ctx:
        return _Ctx()

    def form(self, *_a: object, **_k: object) -> _Ctx:
        return _Ctx()

    def spinner(self, *_a: object, **_k: object) -> _Ctx:
        return _Ctx()

    def stop(self) -> None:
        raise RuntimeError("st.stop() called")

    def error(self, msg: object, *_a: object, **_k: object) -> None:
        self.errors.append(str(msg))

    def warning(self, msg: object, *_a: object, **_k: object) -> None:
        self.warnings.append(str(msg))

    # widgets with return values
    def _value(self, label: str, default: object) -> object:
        self.calls.append(f"widget:{label}")
        return self.overrides.get(label, default)

    def text_area(self, label, value="", **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value)

    text_input = text_area

    def date_input(self, label, value=None, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value)

    def selectbox(self, label, options, index=0, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, list(options)[index])

    def radio(self, label, options, index=0, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, list(options)[index])

    def multiselect(self, label, options, default=None, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, list(default or []))

    def checkbox(self, label, value=False, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value)

    def button(self, label, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, False)

    def form_submit_button(self, label, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, False)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*_a: object, **_k: object) -> None:  # title, caption, markdown, text, dataframe, ...
            self.calls.append(name)

        return record


class _Widgets(_Ctx):
    """Columns, tabs and the sidebar expose the same widget methods as the module."""

    def __init__(self, parent: FakeStreamlit) -> None:
        self._parent = parent

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._parent, name)


class _Trace:
    def __init__(self, *_a: object, **kw: object) -> None:
        self.kw = kw


class FakeFigure:
    def __init__(self, *_a: object, **_k: object) -> None:
        self.ops: list[str] = []

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.ops.append(name)


class FakeGO(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("plotly.graph_objects")

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name.startswith("__"):
            raise AttributeError(name)
        return FakeFigure if name == "Figure" else _Trace


def fake_modules(st: FakeStreamlit) -> dict[str, types.ModuleType]:
    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda *a, **k: FakeFigure()  # type: ignore[attr-defined]
    plotly = types.ModuleType("plotly")
    go = FakeGO()
    plotly.graph_objects, plotly.subplots = go, subplots  # type: ignore[attr-defined]
    return {"streamlit": st, "plotly": plotly, "plotly.graph_objects": go, "plotly.subplots": subplots}


def import_app(st: FakeStreamlit):  # type: ignore[no-untyped-def]
    sys.modules.pop("app", None)
    return importlib.import_module("app")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class ComputationLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = import_app(FakeStreamlit())  # helpers do not depend on the UI libraries
        returns = make_returns(8, 2200, seed=12)
        returns.index = pd.bdate_range("2018-01-02", periods=2200)
        cls.prices = to_prices(returns).iloc[1:]
        cls.prices.columns = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_parse_tickers(self) -> None:
        self.assertEqual(self.app.parse_tickers(" spy, qqq ;xom\nspy ,, "), ("SPY", "QQQ", "XOM"))
        self.assertEqual(self.app.parse_tickers(""), ())

    def test_in_vs_out_of_sample_table(self) -> None:
        res = self.app.compute_optimizer(self.prices, "2024-12-31", 0.03, 0.4, "bayes_stein")
        table = self.app.in_vs_out_of_sample(res, self.prices, "2024-12-31", "2025-01-01", 0.03, 0.4, "bayes_stein")
        self.assertEqual(len(table), 3)
        self.assertTrue(table.notna().all().all())

    def test_compute_risk_and_short_sample_error(self) -> None:
        w = tuple([1 / 8] * 8)
        macro = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (2200, 4)), index=self.prices.index,
                             columns=["sp500_logret", "d_ust_10y_bp", "usd_index_logret", "wti_crude_logret"])
        report = self.app.compute_risk(self.prices, w, macro, 500_000.0, 1, (0.95, 0.99), "2018-01-01")
        self.assertEqual(len(report.var_table), 8)
        with self.assertRaises(ValueError):
            self.app.compute_risk(self.prices, w, macro, 500_000.0, 1, (0.95, 0.99), "2026-06-01")


class UiWiringTests(unittest.TestCase):
    def _run(self, overrides: dict[str, object] | None = None) -> FakeStreamlit:
        st = FakeStreamlit(overrides)
        with mock.patch.dict(sys.modules, fake_modules(st)), mock.patch.object(de, "yf", FakeYahoo()):
            app = import_app(st)
            app.main()
        return st

    def test_full_dashboard_renders_without_errors(self) -> None:
        st = self._run()
        self.assertEqual(st.errors, [], st.errors)
        self.assertEqual(st.warnings, [], st.warnings)
        # market data 2, pairs 1, optimizer 3, risk 5 -- a dropped chart or table is a regression
        self.assertEqual(st.calls.count("plotly_chart"), 11)
        self.assertEqual(st.calls.count("dataframe"), 10)
        self.assertIn("title", st.calls)
        self.assertIn("Paper trading", st.tab_labels)

    def test_paper_order_requires_and_records_explicit_user_submission(self) -> None:
        st = FakeStreamlit({"Start paper account": True, "Place paper order": True})
        prices = pd.DataFrame(
            {"TEST": 100 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0002, 0.01, 260)))},
            index=pd.bdate_range("2024-01-02", periods=260),
        )
        with mock.patch.dict(sys.modules, fake_modules(st)):
            app = import_app(st)
            app.render_paper_trading(prices)
        account = st.session_state["paper_trading_simulator"].snapshot()
        self.assertEqual(len(account.fills), 1)
        self.assertEqual(account.fills[0].action.value, "BUY")

    def test_invalid_dates_stop_with_message(self) -> None:
        with self.assertRaises(RuntimeError):
            st = FakeStreamlit({"Out-of-sample starts": dt.date(2017, 1, 1)})
            with mock.patch.dict(sys.modules, fake_modules(st)):
                import_app(st).main()
        self.assertTrue(any("Dates must satisfy" in m for m in st.errors))

    def test_bad_universe_surfaces_error_instead_of_crashing(self) -> None:
        st = FakeStreamlit({"Portfolio universe (tickers)": "SPY"})  # one asset: optimizer cannot run
        with mock.patch.dict(sys.modules, fake_modules(st)), mock.patch.object(de, "yf", FakeYahoo()):
            import_app(st).main()
        self.assertTrue(any("Optimization failed" in m for m in st.errors), st.errors)
        self.assertTrue(any("needs a successful portfolio optimization" in m for m in st.warnings))


if __name__ == "__main__":
    unittest.main()

