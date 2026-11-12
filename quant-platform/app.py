"""
app.py -- Interactive dashboard for the Algorithmic Trading & Portfolio Optimization Platform.

Run with:   streamlit run app.py

Five tabs sit on top of the research engines and paper simulator:
    Market data       DataEngine: prices, macro regime indicators
    Pairs trading     PairsBacktester: pair screen, signals, equity curve, trade log
    Portfolio         PortfolioOptimizer: efficient frontier, weights, walk-forward test
    Risk & stress     RiskEngine: VaR / ES, drawdowns, VaR backtests, stress scenarios
    Paper trading     PaperSimulator: user-confirmed practice orders and machine-learning alerts

Everything is estimated or selected on a FORMATION period (default: through 2024) and judged on a
later OUT-OF-SAMPLE period (default: 2025 onward). All heavy computation lives in plain functions
(cached by Streamlit) so the logic can be tested without a browser.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

try:  # the helpers stay importable (e.g. for tests) without the UI libraries
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None

from backtester import BacktestResult, PairsBacktestConfig, PairsBacktester, screen_pairs
from data_engine import DataEngine, DataEngineConfig
from optimizer import OptimizationResult, OptimizerConfig, PortfolioOptimizer, WalkForwardResult
from paper_trading import AgentError, OrderError, PaperSimulator, ResearchAgent, TradeAction
from risk_engine import (
    RiskConfig,
    RiskEngine,
    RiskReport,
    macro_factor_frame,
)

logger = logging.getLogger(__name__)

PAIRS_DEFAULT = "KO, PEP, XOM, CVX"
UNIVERSE_DEFAULT = "SPY, QQQ, IWM, EFA, TLT, GLD, XLE, XLV"
TEMPLATE = "plotly_white"


def _cache(func: Any) -> Any:
    """Streamlit's data cache when available, a no-op otherwise."""
    return st.cache_data(show_spinner=False)(func) if st is not None else func


# --------------------------------------------------------------------------- #
# Computation layer (pure functions; no UI calls)
# --------------------------------------------------------------------------- #
def parse_tickers(text: str) -> tuple[str, ...]:
    """'ko, pep ; xom' -> ('KO', 'PEP', 'XOM'); duplicates removed, order kept."""
    cleaned = text.replace(";", ",").replace("\n", ",").split(",")
    return tuple(dict.fromkeys(t.strip().upper() for t in cleaned if t.strip()))


@_cache
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    """Cleaned, gap-free close-price panel from the Data Engine."""
    return DataEngine(DataEngineConfig(start=start)).get_price_panel(list(tickers))


@_cache
def load_macro(start: str) -> pd.DataFrame:
    """Macro levels plus derived regime features (yield curve, VIX Z-score, log changes)."""
    engine = DataEngine(DataEngineConfig(start=start))
    return engine.build_macro_features(engine.fetch_macro())


@_cache
def compute_screen(prices: pd.DataFrame, formation_end: str) -> pd.DataFrame:
    """Rank pairs by Engle-Granger p-value using ONLY the formation period."""
    return screen_pairs(prices.loc[:formation_end])


@_cache
def compute_pairs(
    prices: pd.DataFrame, y: str, x: str, zscore_mode: str, entry_z: float, exit_z: float,
    stop_z: float, fill_delay: int,
) -> BacktestResult:
    cfg = PairsBacktestConfig(zscore_mode=zscore_mode, entry_z=entry_z, exit_z=exit_z,  # type: ignore[arg-type]
                              stop_z=stop_z, fill_delay_bars=fill_delay)
    return PairsBacktester(cfg).run(prices, y, x)


def _optimizer(rf: float, max_weight: float, estimator: str) -> PortfolioOptimizer:
    return PortfolioOptimizer(OptimizerConfig(
        risk_free_rate=rf, max_weight=max_weight, mean_estimator=estimator))  # type: ignore[arg-type]


@_cache
def compute_optimizer(
    prices: pd.DataFrame, formation_end: str, rf: float, max_weight: float, estimator: str
) -> OptimizationResult:
    return _optimizer(rf, max_weight, estimator).optimize(prices.loc[:formation_end])


@_cache
def compute_walk_forward(
    prices: pd.DataFrame, rf: float, max_weight: float, estimator: str, lookback: int
) -> WalkForwardResult:
    return _optimizer(rf, max_weight, estimator).walk_forward(prices, lookback=lookback)


def in_vs_out_of_sample(
    result: OptimizationResult, prices: pd.DataFrame, formation_end: str, oos_start: str,
    rf: float, max_weight: float, estimator: str,
) -> pd.DataFrame:
    """Sharpe of fixed formation-period weights on the formation sample vs. the later sample."""
    opt = _optimizer(rf, max_weight, estimator)
    returns = opt.simple_returns(prices)
    rows = {}
    for p in (result.max_sharpe, result.gmv, result.equal_weight):
        ins = opt.evaluate(p.weights, returns.loc[:formation_end])
        oos = opt.evaluate(p.weights, returns.loc[oos_start:])
        rows[p.label] = {"Sharpe in-sample": ins["sharpe"], "Sharpe out-of-sample": oos["sharpe"],
                         "OOS return": oos["ann_return"], "OOS volatility": oos["ann_volatility"],
                         "OOS max drawdown": oos["max_drawdown"]}
    return pd.DataFrame(rows).T


@_cache
def compute_risk(
    prices: pd.DataFrame, weights: tuple[float, ...], macro: pd.DataFrame, value: float,
    horizon: int, confidences: tuple[float, ...], risk_start: str,
) -> RiskReport:
    """Full risk analysis of a fixed-weight portfolio from ``risk_start`` onward."""
    sub = prices.loc[risk_start:]
    n_ret = len(sub) - 1
    if n_ret < 200:
        raise ValueError(f"Only {n_ret} daily returns from {risk_start}; need at least 200 for risk statistics")
    window = int(min(500, max(60, n_ret // 3)))
    cfg = RiskConfig(confidence_levels=confidences, horizon_days=horizon, portfolio_value=value,
                     backtest_window=window)
    w = pd.Series(weights, index=list(prices.columns))
    return RiskEngine(cfg).analyze(sub, w, factors=macro_factor_frame(macro) if not macro.empty else None)


# --------------------------------------------------------------------------- #
# Charts (plotly)
# --------------------------------------------------------------------------- #
def _need_plotly() -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("plotly is required: pip install -r requirements.txt")


def _oos_marker(fig: Any, oos_start: str | None) -> None:
    if oos_start:
        fig.add_shape(type="line", x0=oos_start, x1=oos_start, y0=0, y1=1, yref="paper",
                      line={"dash": "dash", "color": "darkorange"})


def fig_prices(prices: pd.DataFrame, oos_start: str | None = None) -> Any:
    _need_plotly()
    rebased = prices / prices.iloc[0] * 100.0
    fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[c], name=c, mode="lines") for c in rebased])
    _oos_marker(fig, oos_start)
    fig.update_layout(title="Price history (rebased to 100; dashed line = out-of-sample start)",
                      template=TEMPLATE, height=420, yaxis_title="Rebased price")
    return fig


def fig_macro(macro: pd.DataFrame) -> Any:
    _need_plotly()
    panels = [(c, t) for c, t in (("vix", "VIX"), ("yield_curve_10y3m", "10Y minus 3M yield (pp)"),
                                  ("gold", "Gold"), ("wti_crude", "WTI crude")) if c in macro.columns]
    fig = make_subplots(rows=max(len(panels), 1), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[t for _, t in panels])
    for i, (col, _) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=macro.index, y=macro[col], name=col, mode="lines"), row=i, col=1)
    if "yield_curve_10y3m" in macro.columns:
        row = [c for c, _ in panels].index("yield_curve_10y3m") + 1
        fig.add_hline(y=0.0, line_dash="dot", line_color="crimson", row=row, col=1)  # inversion line
    fig.update_layout(template=TEMPLATE, height=240 * max(len(panels), 1), showlegend=False,
                      title="Macro regime indicators")
    return fig


def fig_pairs(result: BacktestResult, cfg_entry: float, cfg_exit: float, oos_start: str | None) -> Any:
    _need_plotly()
    f = result.frame
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.4, 0.35, 0.25],
                        subplot_titles=("Spread Z-score", "Equity", "Drawdown"))
    fig.add_trace(go.Scatter(x=f.index, y=f["zscore"], name="Z-score", mode="lines"), row=1, col=1)
    for level in (cfg_entry, -cfg_entry):
        fig.add_hline(y=level, line_dash="dash", line_color="crimson", row=1, col=1)
    for level in (cfg_exit, -cfg_exit):
        fig.add_hline(y=level, line_dash="dot", line_color="gray", row=1, col=1)
    held = f[f["position"] != 0]
    fig.add_trace(go.Scatter(x=held.index, y=held["zscore"], mode="markers", name="In position",
                             marker={"size": 4, "color": "royalblue"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=f.index, y=f["equity"], name="Equity", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=f.index, y=f["drawdown"], name="Drawdown", fill="tozeroy"), row=3, col=1)
    _oos_marker(fig, oos_start)
    fig.update_layout(template=TEMPLATE, height=760, showlegend=False,
                      title=f"Pairs trade: long/short {result.pair[0]} vs {result.pair[1]}")
    return fig


def fig_weights(result: OptimizationResult) -> Any:
    _need_plotly()
    fig = go.Figure()
    for p in (result.max_sharpe, result.gmv, result.equal_weight):
        fig.add_trace(go.Bar(x=list(p.weights.index), y=p.weights.to_numpy(), name=p.label))
    fig.update_layout(template=TEMPLATE, barmode="group", title="Portfolio weights", yaxis_tickformat=".0%")
    return fig


def fig_walk_forward(wf: WalkForwardResult, oos_start: str) -> Any:
    _need_plotly()
    growth = (1.0 + wf.returns.loc[oos_start:]).cumprod()
    fig = go.Figure([go.Scatter(x=growth.index, y=growth[c], name=c, mode="lines") for c in growth])
    fig.update_layout(template=TEMPLATE, title="Walk-forward growth of $1 (out-of-sample, after costs)")
    return fig


def fig_return_distribution(report: RiskReport, alpha: float) -> Any:
    _need_plotly()
    r = report.returns.to_numpy()
    fig = go.Figure(go.Histogram(x=r, nbinsx=80, name="Daily returns", marker_color="lightsteelblue"))
    table = report.var_table
    for method, color in (("Historical", "crimson"), ("Parametric (normal)", "darkgreen"),
                          ("Filtered historical (EWMA)", "purple")):
        if (method, alpha) in table.index:
            fig.add_vline(x=-float(table.loc[(method, alpha), "var"]), line_color=color, line_dash="dash",
                          annotation_text=f"{method} VaR", annotation_position="top")
    fig.update_layout(template=TEMPLATE, title=f"Return distribution with {alpha:.0%} VaR (1-day horizon scale)",
                      xaxis_tickformat=".1%")
    return fig


def fig_drawdown(report: RiskReport) -> Any:
    _need_plotly()
    dd = report.drawdown.series["drawdown"]
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown", line={"color": "crimson"}))
    fig.update_layout(template=TEMPLATE, title="Underwater curve", yaxis_tickformat=".0%")
    return fig


def fig_var_backtest(bt: pd.DataFrame, title: str) -> Any:
    _need_plotly()
    valid = bt.dropna(subset=["var"])
    hits = valid[valid["exception"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid.index, y=valid["return"], mode="lines", name="Return",
                             line={"color": "lightgray"}))
    fig.add_trace(go.Scatter(x=valid.index, y=-valid["var"], mode="lines", name="-VaR forecast",
                             line={"color": "royalblue"}))
    fig.add_trace(go.Scatter(x=hits.index, y=hits["return"], mode="markers", name="Exception",
                             marker={"color": "crimson", "size": 7}))
    fig.update_layout(template=TEMPLATE, title=title, yaxis_tickformat=".1%")
    return fig


def fig_contributions(report: RiskReport) -> Any:
    _need_plotly()
    c = report.contributions
    fig = go.Figure([go.Bar(x=c.index, y=c["pct_of_risk"], name="Share of risk"),
                     go.Bar(x=c.index, y=c["weight"], name="Weight")])
    fig.update_layout(template=TEMPLATE, barmode="group", title="Weight vs. share of portfolio risk",
                      yaxis_tickformat=".0%")
    return fig


def fig_stress(report: RiskReport) -> Any:
    _need_plotly()
    parts = [report.historical_stress["portfolio_return"]]
    if report.factor_stress is not None and not report.factor_stress.empty:
        parts.append(report.factor_stress["portfolio_return"])
    s = pd.concat(parts).sort_values()
    fig = go.Figure(go.Bar(x=s.to_numpy(), y=list(s.index), orientation="h",
                           marker_color=["crimson" if v < 0 else "seagreen" for v in s]))
    fig.update_layout(template=TEMPLATE, title="Portfolio return under stress scenarios",
                      xaxis_tickformat=".0%", height=max(300, 60 * len(s)))
    return fig


# --------------------------------------------------------------------------- #
# Streamlit layout
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x:.2%}" if np.isfinite(x) else "n/a"


def render_overview(prices: pd.DataFrame, macro: pd.DataFrame | None, oos_start: str) -> None:
    st.plotly_chart(fig_prices(prices, oos_start))
    summary = pd.DataFrame({"first date": prices.index[0].date(), "last date": prices.index[-1].date(),
                            "trading days": len(prices), "total return": prices.iloc[-1] / prices.iloc[0] - 1},
                           index=prices.columns)
    st.dataframe(summary.assign(**{"total return": summary["total return"].map(_fmt_pct)}))
    if macro is not None and not macro.empty:
        st.plotly_chart(fig_macro(macro))
    else:
        st.info("Macro data unavailable, so macro charts and factor stress tests are skipped.")


def render_pairs(pair_prices: pd.DataFrame, formation_end: str, oos_start: str) -> None:
    st.markdown("**Pair selection uses only formation-period data**, so the out-of-sample result is not "
                "contaminated by hindsight. The Bonferroni column corrects for testing many pairs.")
    screen = compute_screen(pair_prices, formation_end)
    st.dataframe(screen.round(4))
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Z-score mode", ["formation", "rolling"], index=0)
    entry = c2.slider("Entry |Z|", 1.0, 3.5, 2.0, 0.1)
    exit_ = c3.slider("Exit |Z|", 0.0, 1.5, 0.5, 0.1)
    delay = c4.selectbox("Fill delay (bars)", [0, 1, 2], index=1)
    auto = st.checkbox("Use the top-ranked pair from the formation screen", value=True)
    cols = list(pair_prices.columns)
    if auto:
        y, x = str(screen.iloc[0]["y"]), str(screen.iloc[0]["x"])
    else:
        y = st.selectbox("Long/short leg Y", cols, index=0)
        x = st.selectbox("Hedge leg X", cols, index=min(1, len(cols) - 1))
    if y == x:
        st.warning("Choose two different instruments.")
        return
    result = compute_pairs(pair_prices, y, x, mode, entry, exit_, 4.0, delay)
    full, oos = result.metrics, result.performance(start=oos_start)
    m = st.columns(4)
    m[0].metric("Sharpe (full)", f"{full['sharpe']:.2f}")
    m[1].metric("Sharpe (out-of-sample)", f"{oos['sharpe']:.2f}", help="Standard error: "
                f"{oos['sharpe_stderr']:.2f}. Short samples leave wide error bars.")
    m[2].metric("Max drawdown (OOS)", _fmt_pct(oos["max_drawdown"]))
    m[3].metric("Trades (OOS)", f"{oos['n_trades']:.0f}")
    st.plotly_chart(fig_pairs(result, entry, exit_, oos_start))
    with st.expander("Full performance report"):
        st.text(result.report())
        st.text("Out-of-sample:\n" + result.report(oos))
    st.dataframe(result.trades.tail(25))


def render_optimizer(
    prices: pd.DataFrame, formation_end: str, oos_start: str, rf: float, max_weight: float, estimator: str,
) -> OptimizationResult:
    result = compute_optimizer(prices, formation_end, rf, max_weight, estimator)
    st.text(result.summary())
    st.plotly_chart(result.figure())
    st.plotly_chart(fig_weights(result))
    st.markdown("**Weights chosen on the formation period, evaluated afterwards** "
                "(the drop in Sharpe is estimation error, not bad luck):")
    st.dataframe(in_vs_out_of_sample(result, prices, formation_end, oos_start, rf, max_weight, estimator).round(3))
    lookback = st.slider("Walk-forward lookback (trading days)", 126, 504, 252, 21)
    wf = compute_walk_forward(prices, rf, max_weight, estimator, lookback)
    st.markdown(f"**Walk-forward re-optimization from {oos_start}** (monthly, 5 bps trading cost):")
    st.dataframe(wf.metrics(start=oos_start).round(3))
    st.plotly_chart(fig_walk_forward(wf, oos_start))
    return result


def render_risk(
    prices: pd.DataFrame, macro: pd.DataFrame, optimized: OptimizationResult, value: float,
    horizon: int, confidences: tuple[float, ...], oos_start: str, start_date: str,
) -> None:
    choice = st.radio("Portfolio to analyze", ["Max Sharpe", "Min variance", "Equal weight (1/N)"], horizontal=True)
    port = {"Max Sharpe": optimized.max_sharpe, "Min variance": optimized.gmv,
            "Equal weight (1/N)": optimized.equal_weight}[choice]
    scope = st.radio("Risk sample", ["Full history", "Out-of-sample only"], horizontal=True)
    risk_start = start_date if scope == "Full history" else oos_start
    try:
        report = compute_risk(prices, tuple(float(v) for v in port.weights.reindex(prices.columns)),
                              macro, value, horizon, confidences, risk_start)
    except ValueError as exc:
        st.warning(str(exc))
        return
    alpha = max(confidences)
    table = report.var_table
    hist = table.loc[("Historical", alpha)]
    m = st.columns(4)
    m[0].metric(f"{alpha:.0%} VaR (historical)", f"${hist['var_usd']:,.0f}", _fmt_pct(-hist["var"]))
    m[1].metric(f"{alpha:.0%} Expected shortfall", f"${hist['es_usd']:,.0f}", _fmt_pct(-hist["es"]))
    m[2].metric("Max drawdown", _fmt_pct(report.drawdown.max_drawdown))
    m[3].metric("Current drawdown", _fmt_pct(report.drawdown.current_drawdown))
    st.dataframe(table.round(4))
    st.plotly_chart(fig_return_distribution(report, alpha))
    st.plotly_chart(fig_drawdown(report))
    st.dataframe(report.drawdown.episodes)

    st.subheader("Is the VaR model any good? (backtest)")
    st.caption("A forecast is made before each day using only past data. Kupiec tests the exception COUNT; "
               "Christoffersen tests whether exceptions CLUSTER. 'rejected' means the model is not trustworthy.")
    st.dataframe(report.backtest.round(4))
    method = st.selectbox("Show exceptions for", ["historical", "parametric", "ewma", "fhs"], index=3)
    engine = RiskEngine(report.config)
    st.plotly_chart(fig_var_backtest(engine.backtest_var(report.returns, alpha, method),
                                     f"{method} {alpha:.0%} VaR vs. realized returns"))

    st.subheader("Where does the risk come from?")
    st.plotly_chart(fig_contributions(report))
    st.subheader("Stress scenarios")
    st.caption("Historical scenarios replay a past crisis on today's weights. Macro scenarios push hypothetical "
               "factor shocks through estimated sensitivities: they assume linear betas and ignore the "
               "correlation spike seen in real crises, so treat them as illustrations, not forecasts.")
    st.plotly_chart(fig_stress(report))
    st.dataframe(report.historical_stress)
    if report.factor_stress is not None and not report.factor_stress.empty:
        st.dataframe(report.factor_stress[["portfolio_return", "pnl_usd"]].round(4))


def render_paper_trading(prices: pd.DataFrame) -> None:
    """Render a user-confirmed paper ledger and an advisory ML alert."""
    st.subheader("Paper trading simulator")
    st.caption("Orders are simulated at the latest available close with configurable slippage and fees. "
               "The research agent only creates alerts; it never places orders.")
    if prices.empty or prices.dropna(how="all").empty:
        st.warning("Price history is unavailable for the simulator.")
        return

    state_key = "paper_trading_simulator"
    if state_key not in st.session_state:
        initial_cash = st.number_input("Starting paper cash ($)", 1_000.0, 10_000_000.0,
                                       100_000.0, 5_000.0, key="paper_initial_cash")
        if not st.button("Start paper account"):
            st.info("Start an account to practice with the currently loaded market data.")
            return
        st.session_state[state_key] = PaperSimulator(
            starting_cash=float(initial_cash), commission_bps=1.0, slippage_bps=2.0
        )
    simulator = st.session_state[state_key]
    history = prices.dropna(how="all").sort_index()
    latest = history.dropna(how="all").iloc[-1].dropna()
    simulator.mark({str(k): float(v) for k, v in latest.items()})
    account = simulator.snapshot()

    ticker = st.selectbox("Instrument", list(latest.index), key="paper_ticker")
    ticker_history = history[ticker].dropna()
    if len(ticker_history) < 2:
        st.warning("At least two closes are required to create an alert and simulate a later fill.")
        return
    signal_time = ticker_history.index[-2]
    fill_time = ticker_history.index[-1]
    fill_price = float(ticker_history.iloc[-1])
    current_position = next((p for p in account.positions if p.symbol == ticker), None)
    quantity_held = 0.0 if current_position is None else current_position.quantity
    try:
        alert = ResearchAgent().evaluate(
            ticker_history, ticker, as_of=signal_time, current_quantity=quantity_held
        )
        st.info(f"**Research agent: {alert.action.value}** · signal through {alert.as_of.date()} · "
                f"confidence {alert.confidence:.1%}\n\n{alert.explanation}")
        st.caption(f"Model trained on {alert.training_observations:,} historical feature rows. "
                   f"The simulated fill uses the next available close ({fill_time.date()}) to avoid "
                   "same-bar look-ahead. Predictions are uncertain and are not personalized advice.")
    except AgentError as exc:
        st.warning(f"No alert available: {exc}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Equity", f"${account.equity:,.2f}")
    metric_cols[1].metric("Cash", f"${account.cash:,.2f}")
    metric_cols[2].metric("Positions", str(len(account.positions)))
    metric_cols[3].metric("Latest close", f"${fill_price:,.2f}",
                          help=f"Market data timestamp: {fill_time}")

    with st.form("paper_order_form"):
        action_name = st.selectbox("Paper order action", [a.value for a in TradeAction])
        quantity = st.number_input("Quantity (shares)", min_value=0.001, value=1.0, step=1.0)
        submitted = st.form_submit_button("Place paper order")
    if submitted:
        try:
            fill = simulator.submit_order(
                ticker, TradeAction(action_name), float(quantity), fill_price,
                timestamp=fill_time.to_pydatetime(),
            )
            st.success(f"Simulated {fill.action.value} {fill.quantity:g} {fill.symbol} at "
                       f"${fill.fill_price:,.2f}; commission ${fill.commission:,.2f}.")
            account = simulator.snapshot()
        except OrderError as exc:
            st.error(str(exc))

    if account.positions:
        st.markdown("**Open paper positions**")
        st.dataframe(pd.DataFrame([{
            "Symbol": p.symbol, "Shares": p.quantity, "Average entry": p.average_entry_price,
            "Latest mark": p.market_price, "Market value": p.market_value,
            "Unrealized P&L": p.unrealized_pnl, "Realized P&L": p.realized_pnl,
        } for p in account.positions]).set_index("Symbol"))
    if account.fills:
        st.markdown("**Order audit log**")
        st.dataframe(pd.DataFrame([{
            "Time": f.timestamp, "Symbol": f.symbol, "Action": f.action.value,
            "Shares": f.quantity, "Reference close": f.reference_price,
            "Fill price": f.fill_price, "Commission": f.commission,
        } for f in account.fills]).set_index("Time").sort_index(ascending=False))
    if st.button("Reset paper account", key="reset_paper_account"):
        del st.session_state[state_key]
        st.rerun()


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is not installed: pip install -r requirements.txt")
    st.set_page_config(page_title="Quant Platform", layout="wide")
    st.title("Algorithmic Trading & Portfolio Optimization Platform")
    st.caption("Estimated on a formation period, judged out-of-sample. Educational research tool, not investment advice.")

    with st.sidebar:
        st.header("Settings")
        universe_text = st.text_area("Portfolio universe (tickers)", UNIVERSE_DEFAULT)
        pairs_text = st.text_input("Pairs universe (tickers)", PAIRS_DEFAULT)
        start = st.date_input("History starts", dt.date(2018, 1, 1))
        formation_end = st.date_input("Formation period ends", dt.date(2024, 12, 31))
        oos_start = st.date_input("Out-of-sample starts", dt.date(2025, 1, 1))
        rf = st.number_input("Risk-free rate (annual)", 0.0, 0.15, 0.03, 0.005, format="%.3f")
        max_weight = st.slider("Max weight per asset", 0.15, 1.0, 0.40, 0.05)
        estimator = st.selectbox("Expected-return estimator", ["bayes_stein", "historical"], index=0)
        value = st.number_input("Portfolio value ($)", 10_000.0, 1e9, 1_000_000.0, 50_000.0)
        horizon = st.selectbox("VaR horizon (days)", [1, 5, 10], index=0)
        confidences = tuple(sorted(st.multiselect("Confidence levels", [0.90, 0.95, 0.975, 0.99],
                                                  default=[0.95, 0.99]))) or (0.95, 0.99)

    if not start < formation_end < oos_start:
        st.error("Dates must satisfy: history start < formation end < out-of-sample start.")
        st.stop()
    start_s, formation_s, oos_s = str(start), str(formation_end), str(oos_start)

    try:
        with st.spinner("Downloading and cleaning market data..."):
            prices = load_prices(parse_tickers(universe_text), start_s)
            pair_prices = load_prices(parse_tickers(pairs_text), start_s)
    except Exception as exc:  # noqa: BLE001 - surface any data problem to the user, never crash the page
        st.error(f"Could not load market data: {exc}")
        st.stop()
    try:
        macro: pd.DataFrame | None = load_macro(start_s)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Macro data unavailable ({exc}). Factor stress tests will be skipped.")
        macro = None

    tabs = st.tabs(["Market data", "Pairs trading", "Portfolio optimizer", "Risk & stress",
                    "Paper trading"])
    with tabs[0]:
        render_overview(prices, macro, oos_s)
    with tabs[1]:
        if pair_prices.shape[1] < 2:
            st.warning("Enter at least two tickers for pairs trading.")
        else:
            try:
                render_pairs(pair_prices, formation_s, oos_s)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pairs analysis failed: {exc}")
    optimized: OptimizationResult | None = None
    with tabs[2]:
        try:
            optimized = render_optimizer(prices, formation_s, oos_s, float(rf), float(max_weight), estimator)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimization failed: {exc}")
    with tabs[3]:
        if optimized is None:
            st.warning("The risk analysis needs a successful portfolio optimization first.")
        else:
            try:
                render_risk(prices, macro if macro is not None else pd.DataFrame(), optimized, float(value),
                            int(horizon), confidences, oos_s, start_s)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Risk analysis failed: {exc}")
    with tabs[4]:
        try:
            render_paper_trading(prices)
        except Exception as exc:  # noqa: BLE001 - isolate optional research UI failures
            st.error(f"Paper trading research tool failed: {exc}")


if __name__ == "__main__":
    main()

