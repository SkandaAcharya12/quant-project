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
from market_engine import TickerOption
from paper_trading import ResearchAgent
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

    def select_slider(self, label, options, value=None, **_k):  # type: ignore[no-untyped-def]
        return self._value(label, value if value is not None else next(iter(options)))

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
    def __init__(self, *_a: object, **kwargs: object) -> None:
        self.ops: list[str] = []
        self.traces: list[_Trace] = []
        self.layout_updates: list[dict[str, object]] = []
        self.subplot_options = kwargs
        self.vline_options: list[dict[str, object]] = []

    def add_trace(self, trace: _Trace, *_a: object, **_k: object) -> None:
        self.ops.append("add_trace")
        self.traces.append(trace)

    def update_layout(self, **kwargs: object) -> None:
        self.ops.append("update_layout")
        self.layout_updates.append(kwargs)

    def add_vline(self, *_a: object, **kwargs: object) -> None:
        self.ops.append("add_vline")
        self.vline_options.append(kwargs)

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
    subplots.make_subplots = lambda *a, **k: FakeFigure(**k)  # type: ignore[attr-defined]
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

    def test_cvar_weights_do_not_depend_on_prices_after_formation(self) -> None:
        returns = make_returns(3, 600, seed=31)
        returns.index = pd.bdate_range("2022-01-03", periods=len(returns))
        prices = to_prices(returns)
        prices.columns = ["A", "B", "C"]
        formation_end = str(prices.index[399].date())
        changed_future = prices.copy()
        changed_future.loc[changed_future.index > pd.Timestamp(formation_end)] *= 3.0

        baseline = self.app.compute_cvar_frontier(
            prices, formation_end, 0.95, 0.8, points=5
        )
        perturbed = self.app.compute_cvar_frontier(
            changed_future, formation_end, 0.95, 0.8, points=5
        )
        np.testing.assert_allclose(baseline.weights, perturbed.weights, atol=1e-10)

    def test_compute_risk_and_short_sample_error(self) -> None:
        w = tuple([1 / 8] * 8)
        macro = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (2200, 4)), index=self.prices.index,
                             columns=["sp500_logret", "d_ust_10y_bp", "usd_index_logret", "wti_crude_logret"])
        report = self.app.compute_risk(self.prices, w, macro, 500_000.0, 1, (0.95, 0.99), "2018-01-01")
        self.assertEqual(len(report.var_table), 8)
        with self.assertRaises(ValueError):
            self.app.compute_risk(self.prices, w, macro, 500_000.0, 1, (0.95, 0.99), "2026-06-01")

    def test_paper_signal_chart_uses_real_candles_and_causal_agent_features(self) -> None:
        fake_streamlit = FakeStreamlit()
        with mock.patch.dict(sys.modules, fake_modules(fake_streamlit)):
            app_module = import_app(fake_streamlit)
        close = self.prices["A"].iloc[:600]
        open_prices = close.shift(1).fillna(close.iloc[0])
        ohlcv = pd.DataFrame(
            {
                "Open": open_prices,
                "High": pd.concat([open_prices, close], axis=1).max(axis=1) * 1.01,
                "Low": pd.concat([open_prices, close], axis=1).min(axis=1) * 0.99,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=close.index,
        )
        cutoff = close.index[-2]
        alert = ResearchAgent().evaluate(close, "A", as_of=cutoff)
        figure = app_module.paper_signal_figure(
            ohlcv, "A", cutoff, alert, (), currency_code="USD", currency_symbol="$"
        )

        self.assertIsNotNone(figure)
        self.assertEqual(figure.layout_updates[-1]["hovermode"], "closest")
        self.assertFalse(figure.layout_updates[-1]["showlegend"])
        self.assertNotIn("subplot_titles", figure.subplot_options)
        self.assertTrue(all("annotation_text" not in line for line in figure.vline_options))
        candles = [trace for trace in figure.traces if "open" in trace.kw]
        self.assertEqual(len(candles), 1)
        self.assertEqual(list(candles[0].kw["close"]), list(ohlcv["Close"].tail(180)))
        feature_traces = [
            trace for trace in figure.traces
            if trace.kw.get("name") in {
                "5-session momentum",
                "21-session momentum",
                "63-session momentum",
                "20/60-session moving-average spread",
                "21-session realized volatility",
            }
        ]
        self.assertEqual(len(feature_traces), 5)
        for trace in feature_traces:
            self.assertLessEqual(max(pd.DatetimeIndex(trace.kw["x"])), cutoff)
        feature_readout = app_module.paper_feature_readout(close, cutoff, alert)
        expected = dict(alert.evidence)
        for key, label, _ in app_module._PAPER_FEATURES:
            row = feature_readout.loc[feature_readout["feature"] == label].iloc[0]
            self.assertAlmostEqual(float(row["reading"]), expected[key])


class UiWiringTests(unittest.TestCase):
    def _run(self, overrides: dict[str, object] | None = None) -> FakeStreamlit:
        st = FakeStreamlit(overrides)
        with mock.patch.dict(sys.modules, fake_modules(st)), mock.patch.object(de, "yf", FakeYahoo()):
            app = import_app(st)
            with mock.patch.object(app, "load_market_ticker_catalog", return_value=()):
                app.main()
        return st

    def test_full_dashboard_renders_without_errors(self) -> None:
        st = self._run()
        self.assertEqual(st.errors, [], st.errors)
        self.assertEqual(st.warnings, [], st.warnings)
        # Eleven charts should render; an empty pair-trade log is intentionally not a table.
        self.assertEqual(st.calls.count("plotly_chart"), 11)
        self.assertEqual(st.calls.count("dataframe"), 9)
        self.assertIn("title", st.calls)
        self.assertIn("Paper trading", st.tab_labels)
        self.assertIn("widget:Choose a lesson", st.calls)
        self.assertNotIn("popover", st.calls)

    def test_beginner_tutorial_renders_each_lesson_and_quiz_feedback(self) -> None:
        for lesson in (
            "1 · Start with the basics",
            "2 · Read a candlestick",
            "3 · Understand trade actions",
            "4 · Learn the risk tools",
            "5 · Read the AI research alert",
            "6 · Follow a practice routine",
        ):
            st = FakeStreamlit({
                "Choose a lesson": lesson,
                "Quick check: which candle value is the last price in that interval?": "Close",
            })
            with mock.patch.dict(sys.modules, fake_modules(st)):
                app = import_app(st)
                app.render_trading_tutorial()
            self.assertIn("widget:Choose a lesson", st.calls)
            self.assertIn("widget:Quick check: which candle value is the last price in that interval?", st.calls)
            self.assertIn("success", st.calls)

    def test_cvar_lab_renders_when_enabled(self) -> None:
        st = self._run({"Run CVaR research lab": True})
        self.assertEqual(st.errors, [], st.errors)
        self.assertEqual(st.warnings, [], st.warnings)
        self.assertEqual(st.calls.count("plotly_chart"), 12)
        self.assertEqual(st.calls.count("dataframe"), 11)

    def test_paper_account_uses_selected_tickers_and_can_add_more(self) -> None:
        st = FakeStreamlit({
            "Start paper account": True,
            "Place paper order": True,
            "Stocks available to trade": ["AAPL", "MSFT"],
        })
        ticker_catalog = (
            TickerOption("AAPL", "Apple Inc."),
            TickerOption("MSFT", "Microsoft Corporation"),
            TickerOption("NVDA", "NVIDIA Corporation"),
        )
        prices = pd.DataFrame(
            {"TEST": 100 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0002, 0.01, 260)))},
            index=pd.bdate_range("2024-01-02", periods=260),
        )
        selected = pd.DataFrame(
            {
                ticker: 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0002, 0.01, 260)))
                for ticker, seed in (("AAPL", 20), ("MSFT", 21), ("NVDA", 22))
            },
            index=prices.index,
        )

        def raw_history(symbol: str, _start: str) -> pd.DataFrame:
            close = selected[symbol]
            open_prices = close.shift(1).fillna(close.iloc[0])
            return pd.DataFrame(
                {
                    "Open": open_prices,
                    "High": pd.concat([open_prices, close], axis=1).max(axis=1) * 1.01,
                    "Low": pd.concat([open_prices, close], axis=1).min(axis=1) * 0.99,
                    "Close": close,
                    "Volume": 1_000_000.0,
                },
                index=close.index,
            )

        with mock.patch.dict(sys.modules, fake_modules(st)):
            app = import_app(st)
            with (
                mock.patch.object(app, "load_prices", return_value=selected),
                mock.patch.object(app, "load_ohlcv_history", side_effect=raw_history),
            ):
                app.render_paper_trading(
                    prices, "2024-01-02", ticker_catalog=ticker_catalog
                )
        account = st.session_state["paper_default_simulator"].snapshot()
        self.assertEqual(len(account.fills), 1)
        self.assertEqual(account.fills[0].action.value, "BUY")
        self.assertEqual(account.fills[0].symbol, "AAPL")
        self.assertEqual(st.session_state["paper_default_tradable_tickers"], ["AAPL", "MSFT"])
        self.assertGreaterEqual(st.calls.count("plotly_chart"), 2)

        st.overrides.update({"Place paper order": False, "Add stocks to account": True,
                             "Stocks to add": ["NVDA"]})
        with (
            mock.patch.dict(sys.modules, fake_modules(st)),
            mock.patch.object(app, "load_prices", return_value=selected) as fetch_prices,
            mock.patch.object(app, "load_ohlcv_history", side_effect=raw_history),
        ):
            app.render_paper_trading(
                prices, "2024-01-02", ticker_catalog=ticker_catalog
            )
        fetch_prices.assert_called_once_with(("AAPL", "MSFT", "NVDA"), "2024-01-02")
        self.assertEqual(
            st.session_state["paper_default_tradable_tickers"], ["AAPL", "MSFT", "NVDA"]
        )

    def test_each_market_has_an_independent_paper_account(self) -> None:
        st = FakeStreamlit({"Start paper account": True})
        prices = pd.DataFrame(
            {"TEST": np.linspace(100.0, 150.0, 260)},
            index=pd.bdate_range("2024-01-02", periods=260),
        )
        close = prices["TEST"]
        open_prices = close.shift(1).fillna(close.iloc[0])
        ohlcv = pd.DataFrame(
            {
                "Open": open_prices,
                "High": pd.concat([open_prices, close], axis=1).max(axis=1) * 1.01,
                "Low": pd.concat([open_prices, close], axis=1).min(axis=1) * 0.99,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=prices.index,
        )
        with mock.patch.dict(sys.modules, fake_modules(st)):
            app = import_app(st)
            with mock.patch.object(app, "load_ohlcv_history", return_value=ohlcv):
                app.render_paper_trading(prices, "2024-01-02", "us", "USD", "$")
                app.render_paper_trading(prices, "2024-01-02", "japan", "JPY", "¥")
        self.assertIn("paper_us_simulator", st.session_state)
        self.assertIn("paper_japan_simulator", st.session_state)
        self.assertIsNot(
            st.session_state["paper_us_simulator"], st.session_state["paper_japan_simulator"]
        )

    def test_invalid_dates_stop_with_message(self) -> None:
        st = FakeStreamlit({"Out-of-sample starts": dt.date(2017, 1, 1)})
        with mock.patch.dict(sys.modules, fake_modules(st)):
            app = import_app(st)
            with (
                mock.patch.object(app, "load_market_ticker_catalog", return_value=()),
                self.assertRaises(RuntimeError),
            ):
                app.main()
        self.assertTrue(any("Dates must satisfy" in m for m in st.errors))

    def test_bad_universe_surfaces_error_instead_of_crashing(self) -> None:
        st = FakeStreamlit({"Portfolio stocks": ["SPY"]})
        with mock.patch.dict(sys.modules, fake_modules(st)), mock.patch.object(de, "yf", FakeYahoo()):
            app = import_app(st)
            with (
                mock.patch.object(app, "load_market_ticker_catalog", return_value=()),
                self.assertRaises(RuntimeError),
            ):
                app.main()
        self.assertTrue(any("Choose at least two stocks" in m for m in st.errors), st.errors)


if __name__ == "__main__":
    unittest.main()

