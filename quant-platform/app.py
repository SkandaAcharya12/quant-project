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
import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any, cast

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
from data_engine import DataEngine, DataEngineConfig, DataEngineError
from market_engine import (
    MARKET_SECTIONS,
    MarketSection,
    TickerCatalog,
    TickerCatalogError,
    TickerOption,
)
from optimizer.optimizer import (
    OptimizationError,
    OptimizationResult,
    OptimizerConfig,
    PortfolioOptimizer,
    WalkForwardResult,
)
from optimizer.tail_risk import CVaRFrontier, CVaROptimizer
from paper_trading import (
    AgentAlert,
    AgentError,
    Fill,
    OrderError,
    PaperSimulator,
    ResearchAgent,
    TradeAction,
)
from risk_engine import (
    RiskConfig,
    RiskEngine,
    RiskReport,
    macro_factor_frame,
)

logger = logging.getLogger(__name__)

TEMPLATE = "plotly_white"


def _cache(func: Any) -> Any:
    """Streamlit's data cache when available, a no-op otherwise."""
    return st.cache_data(show_spinner=False)(func) if st is not None else func


def _cache_ttl(
    ttl_seconds: int,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache remote reference data for a bounded interval when Streamlit is present."""
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if st is None:
            return func
        return cast(Callable[..., Any], st.cache_data(ttl=ttl_seconds, show_spinner=False)(func))

    return decorate


# --------------------------------------------------------------------------- #
# Computation layer (pure functions; no UI calls)
# --------------------------------------------------------------------------- #
def parse_tickers(text: str) -> tuple[str, ...]:
    """'ko, pep ; xom' -> ('KO', 'PEP', 'XOM'); duplicates removed, order kept."""
    cleaned = text.replace(";", ",").replace("\n", ",").split(",")
    return tuple(dict.fromkeys(t.strip().upper() for t in cleaned if t.strip()))


def _stored_tickers(value: object) -> tuple[str, ...] | None:
    """Read saved multiselect state or migrate an older comma-separated value."""
    if isinstance(value, str):
        return parse_tickers(value)
    if isinstance(value, (list, tuple)):
        return tuple(dict.fromkeys(
            ticker.strip().upper() for ticker in value
            if isinstance(ticker, str) and ticker.strip()
        ))
    return None


def _ticker_menu(
    catalog: Iterable[TickerOption], include: Iterable[str] = ()
) -> tuple[tuple[str, ...], dict[str, TickerOption]]:
    """Combine a remote catalog with saved/default symbols and display metadata."""
    options_by_symbol = {option.symbol: option for option in catalog}
    for ticker in include:
        symbol = ticker.strip().upper()
        if symbol and symbol not in options_by_symbol:
            options_by_symbol[symbol] = TickerOption(symbol=symbol, name=symbol)
    return tuple(options_by_symbol), options_by_symbol


def _ticker_label(symbol: str, options_by_symbol: dict[str, TickerOption]) -> str:
    """Return a name-plus-symbol label for a ticker dropdown option."""
    option = options_by_symbol.get(symbol)
    return option.display_label if option is not None else symbol


@_cache
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    """Cleaned, gap-free close-price panel from the Data Engine."""
    return DataEngine(DataEngineConfig(start=start)).get_price_panel(list(tickers))


@_cache
def load_ohlcv_history(ticker: str, start: str) -> pd.DataFrame:
    """Load one symbol's cleaned OHLCV without cross-market panel filling."""
    symbol = ticker.strip().upper()
    frames = DataEngine(DataEngineConfig(start=start)).fetch_ohlcv([symbol])
    if symbol not in frames:
        raise DataEngineError(f"No OHLCV history returned for {symbol}")
    return frames[symbol]


@_cache_ttl(24 * 60 * 60)
def load_market_ticker_catalog(market_key: str) -> tuple[TickerOption, ...]:
    """Retrieve and cache a region's searchable security list for one day."""
    return TickerCatalog().fetch(market_key)


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
def compute_cvar_frontier(
    prices: pd.DataFrame,
    formation_end: str,
    confidence_level: float,
    max_weight: float,
    points: int = 11,
) -> CVaRFrontier:
    """Fit CVaR allocations only on returns ending at the formation cutoff."""
    formation_prices = prices.loc[:formation_end]
    formation_returns = PortfolioOptimizer.simple_returns(formation_prices)
    optimizer = CVaROptimizer(OptimizerConfig(max_weight=max_weight))
    return optimizer.efficient_frontier(formation_returns, confidence_level, points)


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


def _wide_widget_kwargs(widget: Any) -> dict[str, Any]:
    """Use the installed Streamlit width API while supporting the older app minimum."""
    try:
        supports_width = "width" in inspect.signature(widget).parameters
    except (TypeError, ValueError):
        supports_width = False
    return {"width": "stretch"} if supports_width else {"use_container_width": True}


def _display_chart(figure: Any) -> None:
    """Render Plotly figures consistently at the available content width."""
    st.plotly_chart(figure, theme=None, **_wide_widget_kwargs(st.plotly_chart))


def _pretty_column_name(name: object) -> str:
    """Turn engine-style column names into readable table headings."""
    aliases = {
        "adf": "ADF",
        "es": "ES",
        "ewma": "EWMA",
        "fhs": "FHS",
        "gmv": "GMV",
        "oos": "OOS",
        "pnl": "P&L",
        "pvalue": "p-value",
        "usd": "USD",
        "var": "VaR",
    }
    words = str(name).replace("_", " ").split()
    return " ".join(aliases.get(word.lower(), word.capitalize()) for word in words)


def _display_table(
    frame: pd.DataFrame,
    *,
    index_name: str | None = None,
    percent_columns: tuple[str, ...] = (),
    integer_columns: tuple[str, ...] = (),
    number_formats: dict[str, str] | None = None,
    currency_columns: tuple[str, ...] = (),
    currency_symbol: str = "$",
    column_labels: dict[str, str] | None = None,
    height: int | None = None,
    hide_index: bool | None = None,
) -> None:
    """Show a full-width table with consistent labels and numeric precision.

    Formatting is applied only to the display configuration. The underlying
    numeric values stay numeric, so sorting and CSV export remain useful.
    """
    labels = {column: _pretty_column_name(column) for column in frame.columns}
    if column_labels:
        labels.update(column_labels)
    display = frame.rename(columns=labels).copy()
    if index_name is not None and not isinstance(display.index, pd.MultiIndex):
        display.index.name = index_name
    should_hide_index = (
        isinstance(display.index, pd.RangeIndex) and display.index.name is None
        if hide_index is None
        else hide_index
    )

    column_config: dict[str, Any] = {}
    config_api = getattr(st, "column_config", None)
    number_column = getattr(config_api, "NumberColumn", None)
    date_column = getattr(config_api, "DatetimeColumn", None)
    if callable(number_column):
        for source_name in percent_columns:
            label = labels.get(source_name, source_name)
            column_config[label] = number_column(label, format="percent")
        for source_name in integer_columns:
            label = labels.get(source_name, source_name)
            column_config[label] = number_column(label, format="%,d")
        for source_name in currency_columns:
            label = labels.get(source_name, source_name)
            column_config[label] = number_column(label, format=f"{currency_symbol}%,.2f")
        for source_name, number_format in (number_formats or {}).items():
            label = labels.get(source_name, source_name)
            column_config[label] = number_column(label, format=number_format)
    if callable(date_column):
        for column_name, series in display.items():
            if pd.api.types.is_datetime64_any_dtype(series):
                label = str(column_name)
                column_config[label] = date_column(label, format="YYYY-MM-DD HH:mm")

    kwargs: dict[str, Any] = {
        **_wide_widget_kwargs(st.dataframe),
        "hide_index": should_hide_index,
        "column_config": column_config or None,
    }
    if height is not None:
        kwargs["height"] = height
    st.dataframe(display, **kwargs)


def _polish_figure(
    figure: Any,
    *,
    title: str | None = None,
    height: int | None = None,
    hovermode: str = "x unified",
    show_legend: bool | None = None,
) -> Any:
    """Apply one visual system to charts while keeping each plot's data intact."""
    layout: dict[str, Any] = {
        "template": TEMPLATE,
        "font": {"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": "#243447"},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "colorway": ["#2563EB", "#0F766E", "#7C3AED", "#D97706", "#DC2626", "#0891B2"],
        "margin": {"l": 68, "r": 26, "t": 82, "b": 58},
        "hovermode": hovermode,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.85)",
            "font": {"size": 11},
        },
    }
    if title is not None:
        layout["title"] = {
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": "#172B4D"},
        }
    if height is not None:
        layout["height"] = height
    if show_legend is not None:
        layout["showlegend"] = show_legend
    figure.update_layout(**layout)
    figure.update_xaxes(showline=True, linecolor="#CBD5E1", showgrid=False, zeroline=False,
                        ticks="outside", automargin=True)
    figure.update_yaxes(showline=False, showgrid=True, gridcolor="#E7EDF3", zeroline=False,
                        ticks="outside", automargin=True)
    layout_object = getattr(figure, "layout", None)
    if getattr(layout_object, "annotations", None):
        figure.update_annotations(font={"size": 12, "color": "#334155"})
    return figure


def _oos_marker(fig: Any, oos_start: str | None) -> None:
    if oos_start:
        fig.add_shape(type="line", x0=oos_start, x1=oos_start, y0=0, y1=1, yref="paper",
                      line={"dash": "dash", "color": "#D97706", "width": 1.5})
        fig.add_annotation(
            x=oos_start,
            y=1,
            yref="paper",
            text="OOS start",
            showarrow=False,
            xanchor="left",
            yshift=10,
            font={"size": 10, "color": "#B45309"},
            bgcolor="rgba(255,255,255,0.85)",
        )


def fig_prices(prices: pd.DataFrame, oos_start: str | None = None) -> Any:
    _need_plotly()
    rebased = prices / prices.iloc[0] * 100.0
    fig = go.Figure([
        go.Scatter(
            x=rebased.index,
            y=rebased[c],
            name=str(c),
            mode="lines",
            line={"width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>Index: %{y:.1f}<extra>%{fullData.name}</extra>",
        )
        for c in rebased
    ])
    _oos_marker(fig, oos_start)
    fig.update_yaxes(title_text="Growth of 100", tickformat=".0f")
    fig.update_xaxes(title_text="Date")
    show_legend = len(rebased.columns) <= 12
    title = "Price performance · rebased to 100"
    if not show_legend:
        title += " · hover for ticker names"
    return _polish_figure(
        fig,
        title=title,
        height=430,
        hovermode="x unified" if show_legend else "closest",
        show_legend=show_legend,
    )


def fig_macro(macro: pd.DataFrame) -> Any:
    _need_plotly()
    panels = [
        ("vix", "Volatility index", "Index points"),
        ("yield_curve_10y3m", "10Y–3M Treasury spread", "Percentage points"),
        ("gold", "Gold", "USD per troy ounce"),
        ("wti_crude", "WTI crude oil", "USD per barrel"),
    ]
    panels = [(col, title, unit) for col, title, unit in panels if col in macro.columns]
    if not panels:
        return _polish_figure(go.Figure(), title="Macro regime indicators", height=320)
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[title for _, title, _ in panels],
    )
    for row, (col, _, unit) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(
                x=macro.index,
                y=macro[col],
                name=col,
                mode="lines",
                line={"width": 1.8},
                hovertemplate="%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=row, col=1)
    if "yield_curve_10y3m" in macro.columns:
        row = [col for col, _, _ in panels].index("yield_curve_10y3m") + 1
        fig.add_hline(y=0.0, line_dash="dot", line_color="#DC2626", row=row, col=1)
    fig.update_xaxes(title_text="Date", row=len(panels), col=1)
    return _polish_figure(
        fig,
        title="Macro regime indicators",
        height=max(330, 205 * len(panels) + 90),
        show_legend=False,
    )


def fig_pairs(result: BacktestResult, cfg_entry: float, cfg_exit: float, oos_start: str | None) -> Any:
    _need_plotly()
    f = result.frame
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=("Spread z-score", "Strategy equity", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["zscore"],
            name="Spread z-score",
            mode="lines",
            line={"color": "#2563EB", "width": 1.8},
            hovertemplate="%{x|%b %d, %Y}<br>Z-score: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for level in (cfg_entry, -cfg_entry):
        fig.add_hline(y=level, line_dash="dash", line_color="#DC2626", row=1, col=1)
    for level in (cfg_exit, -cfg_exit):
        fig.add_hline(y=level, line_dash="dot", line_color="#94A3B8", row=1, col=1)
    held = f[f["position"] != 0]
    fig.add_trace(
        go.Scatter(
            x=held.index,
            y=held["zscore"],
            mode="markers",
            name="In position",
            marker={"size": 5, "color": "#0F766E"},
            hovertemplate="%{x|%b %d, %Y}<br>Position z-score: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["equity"],
            name="Equity",
            mode="lines",
            line={"color": "#0F766E", "width": 1.8},
            hovertemplate="%{x|%b %d, %Y}<br>Equity: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["drawdown"],
            name="Drawdown",
            fill="tozeroy",
            line={"color": "#DC2626", "width": 1.2},
            hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _oos_marker(fig, oos_start)
    fig.update_yaxes(title_text="Z-score", row=1, col=1)
    fig.update_yaxes(title_text="Growth of 1", row=2, col=1, tickformat=".2f")
    fig.update_yaxes(title_text="Drawdown", row=3, col=1, tickformat=".0%")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return _polish_figure(
        fig,
        title=f"Pairs strategy · {result.pair[0]} / {result.pair[1]}",
        height=760,
        show_legend=False,
    )


def fig_weights(result: OptimizationResult) -> Any:
    _need_plotly()
    fig = go.Figure()
    portfolios = (result.max_sharpe, result.gmv, result.equal_weight)
    weights = pd.concat(
        [portfolio.weights.rename(portfolio.label) for portfolio in portfolios], axis=1
    ).fillna(0.0)
    ranked_assets = weights.abs().max(axis=1).nlargest(20).index
    omitted_assets = weights.index.difference(ranked_assets)
    for p in portfolios:
        selected_weights = p.weights.reindex(ranked_assets, fill_value=0.0)
        labels = [str(asset) for asset in selected_weights.index]
        values = selected_weights.to_numpy().tolist()
        if len(omitted_assets):
            labels.append("Other")
            values.append(float(p.weights.reindex(omitted_assets, fill_value=0.0).sum()))
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=p.label,
                hovertemplate="%{x}<br>Weight: %{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Portfolio weight", tickformat=".0%")
    fig.update_xaxes(title_text="Asset", tickangle=-35)
    title = "Portfolio weights"
    if len(omitted_assets):
        title = "Portfolio weights · top 20 assets and remainder"
    return _polish_figure(fig, title=title, height=430)


def fig_cvar_frontier(frontier: CVaRFrontier, selected_index: int) -> Any:
    """Plot historical expected shortfall against expected return for feasible targets."""
    _need_plotly()
    points = frontier.points
    selected = points.iloc[selected_index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=points["expected_shortfall"],
        y=points["expected_return"],
        customdata=points[["value_at_risk", "volatility"]].to_numpy(),
        mode="lines+markers",
        name=f"{frontier.confidence_level:.0%} CVaR frontier",
        line={"color": "#7C3AED", "width": 2.5},
        marker={"size": 8},
        hovertemplate=(
            "Expected shortfall: %{x:.2%}<br>Expected return: %{y:.2%}"
            "<br>VaR: %{customdata[0]:.2%}<br>Volatility: %{customdata[1]:.2%}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=[selected["expected_shortfall"]],
        y=[selected["expected_return"]],
        mode="markers",
        name="Selected target",
        marker={"size": 15, "symbol": "star", "color": "#D97706"},
        hovertemplate="Selected CVaR portfolio<extra></extra>",
    ))
    fig.update_xaxes(title_text=f"Historical daily {frontier.confidence_level:.0%} CVaR (loss)",
                     tickformat=".2%")
    fig.update_yaxes(title_text="Annualized arithmetic return", tickformat=".1%")
    return _polish_figure(
        fig,
        title="Historical tail-risk efficient frontier",
        height=460,
        hovermode="closest",
    )


def fig_walk_forward(wf: WalkForwardResult, oos_start: str) -> Any:
    _need_plotly()
    growth = (1.0 + wf.returns.loc[oos_start:]).cumprod()
    fig = go.Figure([
        go.Scatter(
            x=growth.index,
            y=growth[c],
            name=str(c),
            mode="lines",
            line={"width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>Growth: %{y:.3f}<extra>%{fullData.name}</extra>",
        )
        for c in growth
    ])
    fig.update_yaxes(title_text="Value of one starting unit", tickformat=".2f")
    fig.update_xaxes(title_text="Date")
    return _polish_figure(
        fig,
        title="Walk-forward performance · out-of-sample, after costs",
        height=430,
    )


def fig_return_distribution(report: RiskReport, alpha: float) -> Any:
    _need_plotly()
    r = report.returns.to_numpy()
    fig = go.Figure(go.Histogram(
        x=r,
        nbinsx=60,
        name="Daily returns",
        marker={"color": "#93C5FD", "line": {"color": "#FFFFFF", "width": 1}},
        hovertemplate="Daily return: %{x:.2%}<br>Observations: %{y}<extra></extra>",
    ))
    table = report.var_table
    thresholds = 0
    for method, color in (("Historical", "#DC2626"), ("Parametric (normal)", "#0F766E"),
                          ("Filtered historical (EWMA)", "#7C3AED")):
        if (method, alpha) in table.index:
            threshold = -float(table.loc[(method, alpha), "var"])
            fig.add_vline(x=threshold, line_color=color, line_dash="dash", line_width=1.5)
            fig.add_annotation(
                x=threshold,
                y=0.97 - 0.11 * thresholds,
                xref="x",
                yref="paper",
                text=f"{method} VaR",
                showarrow=False,
                xanchor="left",
                font={"size": 10, "color": color},
                bgcolor="rgba(255,255,255,0.85)",
            )
            thresholds += 1
    fig.update_xaxes(title_text="Daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Trading days")
    return _polish_figure(
        fig,
        title=f"Daily return distribution · {alpha:.0%} VaR thresholds",
        height=420,
        hovermode="closest",
        show_legend=False,
    )


def fig_drawdown(report: RiskReport) -> Any:
    _need_plotly()
    dd = report.drawdown.series["drawdown"]
    fig = go.Figure(go.Scatter(
        x=dd.index,
        y=dd,
        fill="tozeroy",
        name="Drawdown",
        line={"color": "#DC2626", "width": 1.8},
        fillcolor="rgba(220,38,38,0.12)",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
    ))
    fig.update_yaxes(title_text="Drawdown from prior peak", tickformat=".0%")
    fig.update_xaxes(title_text="Date")
    return _polish_figure(fig, title="Portfolio drawdown", height=410, show_legend=False)


def fig_var_backtest(bt: pd.DataFrame, title: str) -> Any:
    _need_plotly()
    valid = bt.dropna(subset=["var"])
    hits = valid[valid["exception"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid.index,
        y=valid["return"],
        mode="lines",
        name="Realized return",
        line={"color": "#94A3B8", "width": 1.2},
        hovertemplate="%{x|%b %d, %Y}<br>Realized return: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=valid.index,
        y=-valid["var"],
        mode="lines",
        name="VaR limit",
        line={"color": "#2563EB", "width": 1.8},
        hovertemplate="%{x|%b %d, %Y}<br>VaR limit: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hits.index,
        y=hits["return"],
        mode="markers",
        name="VaR exception",
        marker={"color": "#DC2626", "size": 8, "line": {"color": "#FFFFFF", "width": 1}},
        hovertemplate="%{x|%b %d, %Y}<br>Exception return: %{y:.2%}<extra></extra>",
    ))
    fig.update_yaxes(title_text="Daily return", tickformat=".1%")
    fig.update_xaxes(title_text="Date")
    return _polish_figure(fig, title=title, height=430)


def fig_contributions(report: RiskReport) -> Any:
    _need_plotly()
    c = report.contributions
    ranked_assets = c["pct_of_risk"].abs().nlargest(20).index
    omitted_assets = c.index.difference(ranked_assets)
    display = c.loc[ranked_assets, ["pct_of_risk", "weight"]].copy()
    if len(omitted_assets):
        display.loc["Other"] = c.loc[omitted_assets, ["pct_of_risk", "weight"]].sum()
    fig = go.Figure([
        go.Bar(
            x=display.index,
            y=display["pct_of_risk"],
            name="Share of risk",
            hovertemplate="%{x}<br>Risk contribution: %{y:.1%}<extra></extra>",
        ),
        go.Bar(
            x=display.index,
            y=display["weight"],
            name="Portfolio weight",
            hovertemplate="%{x}<br>Weight: %{y:.1%}<extra></extra>",
        ),
    ])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Share", tickformat=".0%")
    fig.update_xaxes(title_text="Asset", tickangle=-35)
    title = "Portfolio risk contribution vs. weight"
    if len(omitted_assets):
        title = "Portfolio risk contribution · top 20 assets and remainder"
    return _polish_figure(fig, title=title, height=430)


def fig_stress(report: RiskReport) -> Any:
    _need_plotly()
    parts = [report.historical_stress["portfolio_return"]]
    if report.factor_stress is not None and not report.factor_stress.empty:
        parts.append(report.factor_stress["portfolio_return"])
    s = pd.concat(parts).sort_values()
    fig = go.Figure(go.Bar(
        x=s.to_numpy(),
        y=[str(label) for label in s.index],
        orientation="h",
        marker={
            "color": ["#DC2626" if value < 0 else "#0F766E" for value in s],
            "line": {"color": "#FFFFFF", "width": 1},
        },
        hovertemplate="%{y}<br>Portfolio return: %{x:.2%}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Portfolio return", tickformat=".0%", zeroline=True,
                     zerolinecolor="#94A3B8")
    fig.update_yaxes(title_text="Scenario", automargin=True)
    return _polish_figure(
        fig,
        title="Portfolio returns under stress scenarios",
        height=max(360, 48 * len(s) + 120),
        hovermode="closest",
        show_legend=False,
    )


# --------------------------------------------------------------------------- #
# Streamlit layout
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x:.2%}" if np.isfinite(x) else "n/a"


def render_overview(prices: pd.DataFrame, macro: pd.DataFrame | None, oos_start: str) -> None:
    _display_chart(fig_prices(prices, oos_start))
    summary = pd.DataFrame({"first date": prices.index[0].strftime("%b %d, %Y"),
                            "last date": prices.index[-1].strftime("%b %d, %Y"),
                            "trading days": len(prices), "total return": prices.iloc[-1] / prices.iloc[0] - 1},
                           index=prices.columns)
    summary.index = summary.index.map(str)
    _display_table(
        summary,
        index_name="Ticker",
        percent_columns=("total return",),
        integer_columns=("trading days",),
    )
    if macro is not None and not macro.empty:
        _display_chart(fig_macro(macro))
    else:
        st.info("Macro data unavailable, so macro charts and factor stress tests are skipped.")


def render_pairs(pair_prices: pd.DataFrame, formation_end: str, oos_start: str) -> None:
    st.markdown("**Pair selection uses only formation-period data**, so the out-of-sample result is not "
                "contaminated by hindsight. The Bonferroni column corrects for testing many pairs.")
    screen = compute_screen(pair_prices, formation_end)
    screen = screen.copy()
    screen.index = pd.RangeIndex(1, len(screen) + 1, name="Rank")
    _display_table(
        screen,
        number_formats={
            "adf_stat": "%.3f",
            "pvalue": "%.4f",
            "beta": "%.3f",
            "alpha": "%.3f",
            "half_life": "%.1f",
            "pvalue_bonferroni": "%.4f",
        },
        column_labels={
            "y": "Leg Y",
            "x": "Hedge leg X",
            "adf_stat": "ADF statistic",
            "pvalue": "p-value",
            "beta": "Hedge ratio",
            "alpha": "Intercept",
            "half_life": "Half-life (days)",
            "pvalue_bonferroni": "Adjusted p-value",
        },
    )
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
    _display_chart(fig_pairs(result, entry, exit_, oos_start))
    with st.expander("Full performance report"):
        st.text(result.report())
        st.text("Out-of-sample:\n" + result.report(oos))
    trades = result.trades.tail(25)
    if trades.empty:
        st.info("No pair trades were generated for these settings and dates.")
    else:
        _display_table(
            trades,
            percent_columns=("gross_return", "net_return", "frictions"),
            integer_columns=("trade_id", "holding_days"),
            number_formats={"entry_z": "%.2f", "exit_z": "%.2f", "hedge_ratio": "%.3f"},
            column_labels={
                "trade_id": "Trade #",
                "entry_signal_date": "Entry signal date",
                "side": "Position side",
                "entry_z": "Entry z-score",
                "hedge_ratio": "Hedge ratio",
                "exit_signal_date": "Exit signal date",
                "exit_z": "Exit z-score",
                "exit_reason": "Exit reason",
                "holding_days": "Holding days",
                "gross_return": "Gross return",
                "net_return": "Net return",
                "frictions": "Trading costs",
            },
            height=360,
        )


def render_optimizer(
    prices: pd.DataFrame, formation_end: str, oos_start: str, rf: float, max_weight: float, estimator: str,
) -> OptimizationResult:
    result = compute_optimizer(prices, formation_end, rf, max_weight, estimator)
    st.text(result.summary())
    frontier_figure = result.figure()
    frontier_figure.update_xaxes(title_text="Annualized volatility", tickformat=".1%")
    frontier_figure.update_yaxes(title_text="Annualized return", tickformat=".1%")
    if len(result.assets) > 16:
        for trace in frontier_figure.data:
            if trace.name == "Assets":
                trace.mode = "markers"
                trace.hovertemplate = (
                    "%{text}<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
                )
    _display_chart(_polish_figure(frontier_figure, height=500))
    _display_chart(fig_weights(result))
    st.markdown("**Weights chosen on the formation period, evaluated afterwards** "
                "(the drop in Sharpe is estimation error, not bad luck):")
    comparison = in_vs_out_of_sample(
        result, prices, formation_end, oos_start, rf, max_weight, estimator
    )
    _display_table(
        comparison,
        index_name="Portfolio",
        percent_columns=("OOS return", "OOS volatility", "OOS max drawdown"),
        number_formats={"Sharpe in-sample": "%.2f", "Sharpe out-of-sample": "%.2f"},
    )
    lookback = st.slider("Walk-forward lookback (trading days)", 126, 504, 252, 21)
    wf = compute_walk_forward(prices, rf, max_weight, estimator, lookback)
    st.markdown(f"**Walk-forward re-optimization from {oos_start}** (monthly, 5 bps trading cost):")
    wf_metrics = wf.metrics(start=oos_start)
    _display_table(
        wf_metrics,
        index_name="Portfolio",
        percent_columns=("total_return", "ann_return", "ann_volatility", "max_drawdown",
                         "avg_turnover"),
        number_formats={"sharpe": "%.2f"},
    )
    _display_chart(fig_walk_forward(wf, oos_start))
    render_cvar_lab(prices, formation_end, oos_start, result, rf, max_weight)
    return result


def render_cvar_lab(
    prices: pd.DataFrame,
    formation_end: str,
    oos_start: str,
    optimized: OptimizationResult,
    rf: float,
    max_weight: float,
) -> None:
    """Offer an optional historical-CVaR allocation and strict out-of-sample check."""
    with st.expander("Historical tail-risk optimizer (CVaR)"):
        st.markdown(
            "The solver minimizes empirical tail loss using formation-period returns. "
            "It traces the lowest CVaR portfolio at each feasible annual return floor."
        )
        st.latex(
            r"\min_{w,\eta,u}\; \eta + \frac{1}{(1-\alpha)T}\sum_t u_t"
            r"\quad\text{s.t.}\quad u_t\ge -r_t^\top w-\eta,\;u_t\ge0,"
            r"\;\mathbf{1}^\top w=1,\;\hat\mu^\top w\ge r_{\min}"
        )
        if not st.checkbox("Run CVaR research lab", value=False, key="run_cvar_lab"):
            st.caption("Enable the lab to calculate the tail-risk frontier for this portfolio universe.")
            return

        confidence_level = float(st.selectbox(
            "Tail confidence level",
            [0.90, 0.95, 0.975, 0.99],
            index=1,
            key="cvar_confidence",
        ))
        try:
            frontier = compute_cvar_frontier(
                prices,
                formation_end,
                confidence_level,
                max_weight,
                points=11,
            )
        except (OptimizationError, ValueError) as exc:
            st.warning(f"CVaR optimization could not be run: {exc}")
            return

        points = frontier.points
        if len(points) > 1:
            targets = points["target_return"].astype(float).tolist()
            selected_target = float(st.select_slider(
                "Minimum annual return target",
                options=targets,
                value=targets[len(targets) // 2],
                format_func=lambda target: f"{target:.1%}",
                key="cvar_target_index",
            ))
            selected_index = int(np.abs(np.asarray(targets) - selected_target).argmin())
        else:
            selected_index = 0
            st.caption("The configured weight bounds leave only one feasible return target.")

        selected = points.iloc[selected_index]
        weights = frontier.weights.iloc[selected_index].copy()
        expected_return = float(selected["expected_return"])
        historical_var = float(selected["value_at_risk"])
        historical_cvar = float(selected["expected_shortfall"])
        _display_chart(fig_cvar_frontier(frontier, selected_index))

        oos_returns = PortfolioOptimizer.simple_returns(prices).loc[oos_start:]
        oos_stats: dict[str, float] | None = None
        if len(oos_returns) >= 2:
            evaluator = PortfolioOptimizer(
                OptimizerConfig(risk_free_rate=rf, max_weight=max_weight)
            )
            oos_stats = evaluator.evaluate(weights, oos_returns)
        else:
            st.warning("There are fewer than two out-of-sample returns for this target.")

        cards = st.columns(4)
        cards[0].metric("Return floor", f"{float(selected['target_return']):.1%}")
        cards[1].metric("Formation return", f"{expected_return:.1%}")
        cards[2].metric("Historical daily VaR", f"{historical_var:.2%}")
        cards[3].metric(
            "Out-of-sample return",
            f"{oos_stats['ann_return']:.1%}" if oos_stats is not None else "Unavailable",
        )
        st.caption(
            f"Expected shortfall at {confidence_level:.0%}: {historical_cvar:.2%} of daily portfolio value. "
            "VaR and expected shortfall are loss magnitudes estimated on formation data; "
            "the next table evaluates fixed weights after the formation cutoff."
        )

        st.markdown("**Selected formation-period weights**")
        weight_table = weights.rename("Weight").sort_values(ascending=False).to_frame()
        _display_table(weight_table, index_name="Asset", percent_columns=("Weight",))

        if oos_stats is not None:
            comparison_rows = {
                "Minimum CVaR": oos_stats,
                "Max Sharpe": evaluator.evaluate(optimized.max_sharpe.weights, oos_returns),
                "Min variance": evaluator.evaluate(optimized.gmv.weights, oos_returns),
                "Equal weight": evaluator.evaluate(optimized.equal_weight.weights, oos_returns),
            }
            comparison = pd.DataFrame(comparison_rows).T[
                ["ann_return", "ann_volatility", "sharpe", "max_drawdown"]
            ]
            _display_table(
                comparison,
                index_name="Portfolio",
                percent_columns=("ann_return", "ann_volatility", "max_drawdown"),
                number_formats={"sharpe": "%.2f"},
                column_labels={
                    "ann_return": "Annualized return",
                    "ann_volatility": "Annualized volatility",
                    "max_drawdown": "Maximum drawdown",
                    "sharpe": "Sharpe ratio",
                },
            )


def render_risk(
    prices: pd.DataFrame, macro: pd.DataFrame, optimized: OptimizationResult, value: float,
    horizon: int, confidences: tuple[float, ...], oos_start: str, start_date: str,
    currency_code: str = "USD", currency_symbol: str = "$",
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
    m[0].metric(f"{alpha:.0%} VaR (historical)", f"{currency_symbol}{hist['var_usd']:,.0f}",
                _fmt_pct(-hist["var"]))
    m[1].metric(f"{alpha:.0%} Expected shortfall", f"{currency_symbol}{hist['es_usd']:,.0f}",
                _fmt_pct(-hist["es"]))
    m[2].metric("Max drawdown", _fmt_pct(report.drawdown.max_drawdown))
    m[3].metric("Current drawdown", _fmt_pct(report.drawdown.current_drawdown))
    risk_table = table.reset_index()
    _display_table(
        risk_table,
        percent_columns=("confidence", "var", "es"),
        currency_columns=("var_usd", "es_usd"),
        currency_symbol=currency_symbol,
        number_formats={"confidence": "percent", "var": "percent", "es": "percent"},
        column_labels={
            "method": "Method",
            "confidence": "Confidence level",
            "var": "VaR return",
            "es": "Expected shortfall return",
            "var_usd": f"VaR ({currency_code})",
            "es_usd": f"Expected shortfall ({currency_code})",
        },
    )
    _display_chart(fig_return_distribution(report, alpha))
    _display_chart(fig_drawdown(report))
    _display_table(
        report.drawdown.episodes,
        percent_columns=("depth",),
        integer_columns=("bars_to_trough", "bars_underwater"),
        column_labels={
            "peak_date": "Peak date",
            "trough_date": "Trough date",
            "recovery_date": "Recovery date",
            "depth": "Drawdown depth",
            "bars_to_trough": "Days to trough",
            "bars_underwater": "Days underwater",
            "recovered": "Recovered",
        },
    )

    st.subheader("Is the VaR model any good? (backtest)")
    st.caption("A forecast is made before each day using only past data. Kupiec tests the exception COUNT; "
               "Christoffersen tests whether exceptions CLUSTER. 'rejected' means the model is not trustworthy.")
    backtest_table = report.backtest.reset_index()
    _display_table(
        backtest_table,
        percent_columns=("rate",),
        integer_columns=("days", "exceptions"),
        number_formats={
            "expected": "%.1f",
            "kupiec_p": "%.3f",
            "christoffersen_p": "%.3f",
            "cond_coverage_p": "%.3f",
        },
        column_labels={
            "method": "Method",
            "confidence": "Confidence level",
            "days": "Days",
            "exceptions": "Exceptions",
            "expected": "Expected exceptions",
            "rate": "Exception rate",
            "kupiec_p": "Kupiec p-value",
            "christoffersen_p": "Christoffersen p-value",
            "cond_coverage_p": "Conditional coverage p-value",
            "verdict": "Model verdict",
            "basel_zone_last250": "Basel zone (last 250)",
        },
    )
    method = st.selectbox("Show exceptions for", ["historical", "parametric", "ewma", "fhs"], index=3)
    engine = RiskEngine(report.config)
    _display_chart(
        fig_var_backtest(
            engine.backtest_var(report.returns, alpha, method),
            f"{method.upper()} {alpha:.0%} VaR vs. realized returns",
        )
    )

    st.subheader("Where does the risk come from?")
    _display_chart(fig_contributions(report))
    st.subheader("Stress scenarios")
    st.caption("Historical scenarios replay a past crisis on today's weights. Macro scenarios push hypothetical "
               "factor shocks through estimated sensitivities: they assume linear betas and ignore the "
               "correlation spike seen in real crises, so treat them as illustrations, not forecasts.")
    _display_chart(fig_stress(report))
    _display_table(
        report.historical_stress,
        index_name="Scenario",
        percent_columns=("portfolio_return", "worst_asset_return"),
        integer_columns=("n_days",),
        currency_columns=("pnl_usd",),
        currency_symbol=currency_symbol,
        column_labels={
            "start": "Start date",
            "end": "End date",
            "n_days": "Trading days",
            "portfolio_return": "Portfolio return",
            "pnl_usd": f"P&L ({currency_code})",
            "worst_asset": "Worst asset",
            "worst_asset_return": "Worst asset return",
        },
    )
    if report.factor_stress is not None and not report.factor_stress.empty:
        factor_stress = report.factor_stress[["portfolio_return", "pnl_usd"]]
        _display_table(
            factor_stress,
            index_name="Scenario",
            percent_columns=("portfolio_return",),
            currency_columns=("pnl_usd",),
            currency_symbol=currency_symbol,
            column_labels={
                "portfolio_return": "Portfolio return",
                "pnl_usd": f"P&L ({currency_code})",
            },
        )


def paper_trade_figure(
    prices: pd.Series,
    symbol: str,
    fills: tuple[Fill, ...],
    currency_symbol: str = "$",
    currency_code: str = "USD",
) -> Any | None:
    """Plot close history and every simulated fill for one symbol."""
    history = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if go is None or history.empty:
        return None

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=history.index,
        y=history.to_numpy(),
        mode="lines",
        name="Closing price",
        line={"color": "#2563EB", "width": 2},
        hovertemplate=(
            f"%{{x|%b %d, %Y}}<br>Close: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        ),
    ))
    symbol_fills = [fill for fill in fills if fill.symbol == symbol]
    if symbol_fills:
        colors = {"BUY": "#0F766E", "COVER": "#0891B2", "SELL": "#D97706", "SHORT": "#DC2626"}
        markers = {
            "BUY": "triangle-up",
            "COVER": "circle",
            "SELL": "triangle-down",
            "SHORT": "triangle-down",
        }
        for action in ("BUY", "SELL", "SHORT", "COVER"):
            action_fills = [fill for fill in symbol_fills if fill.action.value == action]
            if not action_fills:
                continue
            figure.add_trace(go.Scatter(
                x=[fill.timestamp for fill in action_fills],
                y=[fill.fill_price for fill in action_fills],
                mode="markers",
                name=action.title(),
                marker={
                    "size": 11,
                    "color": colors[action],
                    "symbol": markers[action],
                    "line": {"color": "#FFFFFF", "width": 1.2},
                },
                text=[f"{fill.action.value} {fill.quantity:g} shares" for fill in action_fills],
                customdata=[[fill.reference_price, fill.commission] for fill in action_fills],
                hovertemplate=(
                    f"%{{text}}<br>%{{x|%b %d, %Y}}<br>Fill: {currency_symbol}%{{y:,.2f}}"
                    f"<br>Reference: {currency_symbol}%{{customdata[0]:,.2f}}"
                    f"<br>Commission: {currency_symbol}%{{customdata[1]:,.2f}}<extra></extra>"
                ),
            ))
    figure.update_xaxes(title_text="Date")
    figure.update_yaxes(
        title_text=f"Price ({currency_code})",
        tickprefix=currency_symbol,
        tickformat=",.2f",
    )
    return _polish_figure(
        figure,
        title=f"{symbol} · price and paper trades",
        height=430,
        hovermode="closest",
    )


_PAPER_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("momentum_5", "5-session momentum", "Price return over the prior 5 sessions."),
    ("momentum_21", "21-session momentum", "Price return over the prior 21 sessions."),
    ("momentum_63", "63-session momentum", "Price return over the prior 63 sessions."),
    (
        "volatility_21",
        "21-session realized volatility",
        "Sample standard deviation of daily returns over the trailing 21 sessions.",
    ),
    (
        "ma_spread_20_60",
        "20/60-session moving-average spread",
        "20-session mean divided by the 60-session mean, minus one.",
    ),
)


def paper_feature_readout(
    prices: pd.Series,
    as_of: pd.Timestamp,
    alert: AgentAlert | None = None,
) -> pd.DataFrame:
    """Return the model's named, trailing-only inputs at the displayed cutoff."""
    if alert is not None and alert.as_of == as_of:
        values: dict[str, float] = {
            name: float(value) for name, value in alert.evidence
        }
    else:
        feature_frame = ResearchAgent.feature_frame(prices)
        if as_of not in feature_frame.index:
            return pd.DataFrame(columns=("feature", "reading", "definition"))
        feature_row = feature_frame.loc[[as_of]].iloc[0]
        values = {
            str(name): float(value)
            for name, value in feature_row.items()
        }
    rows = [
        {
            "feature": label,
            "reading": float(values[key]),
            "definition": definition,
        }
        for key, label, definition in _PAPER_FEATURES
        if key in values and np.isfinite(float(values[key]))
    ]
    return pd.DataFrame(rows, columns=("feature", "reading", "definition"))


def paper_signal_figure(
    ohlcv: pd.DataFrame,
    symbol: str,
    decision_time: pd.Timestamp,
    alert: AgentAlert | None,
    fills: tuple[Fill, ...],
    currency_symbol: str = "$",
    currency_code: str = "USD",
    max_candles: int = 180,
) -> Any | None:
    """Show real candles plus the exact trailing features used by ``ResearchAgent``.

    Feature traces stop at ``decision_time``. Candles may include the next available
    close used as a paper-fill reference, making the no-lookahead boundary visible.
    """
    required_columns = ("Open", "High", "Low", "Close")
    if go is None or make_subplots is None or not set(required_columns).issubset(ohlcv.columns):
        return None
    history = ohlcv.loc[:, list(required_columns)].apply(pd.to_numeric, errors="coerce")
    history = history.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    history = history.loc[~history.index.duplicated(keep="last")]
    if history.empty or decision_time not in history.index or max_candles < 2:
        return None
    close = history["Close"]
    features = ResearchAgent.feature_frame(close).loc[:decision_time].dropna(how="all")
    if features.empty:
        return None

    visible = history.tail(max_candles)
    visible_features = features.loc[features.index >= visible.index[0]]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=(0.62, 0.22, 0.16),
    )
    figure.add_trace(
        go.Candlestick(
            x=visible.index,
            open=visible["Open"],
            high=visible["High"],
            low=visible["Low"],
            close=visible["Close"],
            name="OHLC candle",
            increasing_line_color="#0F766E",
            increasing_fillcolor="#34D399",
            decreasing_line_color="#DC2626",
            decreasing_fillcolor="#F87171",
            whiskerwidth=0.35,
            hovertemplate=(
                "%{x|%b %d, %Y}<br>Open: " + currency_symbol + "%{open:,.2f}"
                "<br>High: " + currency_symbol + "%{high:,.2f}"
                "<br>Low: " + currency_symbol + "%{low:,.2f}"
                "<br>Close: " + currency_symbol + "%{close:,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    for window, color in ((20, "#2563EB"), (60, "#7C3AED")):
        average = close.rolling(window, min_periods=window).mean().loc[visible.index]
        figure.add_trace(
            go.Scatter(
                x=visible.index,
                y=average,
                mode="lines",
                name=f"{window}-session SMA",
                line={"color": color, "width": 1.5},
                hovertemplate=(
                    f"%{{x|%b %d, %Y}}<br>{window}-session SMA: "
                    f"{currency_symbol}%{{y:,.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    action = "FEATURE CUTOFF" if alert is None else alert.action.value
    signal_color = "#64748B" if alert is None else {
        "BUY": "#0F766E",
        "COVER": "#0891B2",
        "SELL": "#D97706",
        "SHORT": "#DC2626",
        "HOLD": "#64748B",
    }[alert.action.value]
    confidence_text = "" if alert is None else f" · {alert.confidence:.1%} confidence"
    signal_close = float(close.loc[decision_time])
    figure.add_trace(
        go.Scatter(
            x=[decision_time],
            y=[signal_close],
            mode="markers",
            name=f"AI {action}",
            marker={"symbol": "star", "size": 14, "color": signal_color,
                    "line": {"color": "#FFFFFF", "width": 1}},
            text=[f"{action}{confidence_text}"],
            hovertemplate=(
                "%{text}<br>%{x|%b %d, %Y}<br>Decision close: "
                + currency_symbol + "%{y:,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if history.index[-1] > decision_time:
        fill_reference_time = history.index[-1]
        figure.add_trace(
            go.Scatter(
                x=[fill_reference_time],
                y=[float(close.iloc[-1])],
                mode="markers",
                name="Next-session fill reference",
                marker={"symbol": "diamond", "size": 10, "color": "#0891B2",
                        "line": {"color": "#FFFFFF", "width": 1}},
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>Next available close: "
                    + currency_symbol + "%{y:,.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    feature_colors = {
        "momentum_5": "#2563EB",
        "momentum_21": "#0F766E",
        "momentum_63": "#7C3AED",
        "ma_spread_20_60": "#D97706",
    }
    for key, label, _ in _PAPER_FEATURES:
        if key == "volatility_21" or key not in visible_features:
            continue
        figure.add_trace(
            go.Scatter(
                x=visible_features.index,
                y=visible_features[key],
                mode="lines",
                name=label,
                line={"color": feature_colors[key], "width": 1.7},
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{label}: %{{y:.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=visible_features.index,
            y=visible_features["volatility_21"],
            mode="lines",
            name="21-session realized volatility",
            line={"color": "#DC2626", "width": 1.8},
            hovertemplate=(
                "%{x|%b %d, %Y}<br>21-session realized volatility: "
                "%{y:.2%}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    symbol_fills = [fill for fill in fills if fill.symbol == symbol]
    fill_colors = {"BUY": "#0F766E", "COVER": "#0891B2", "SELL": "#D97706", "SHORT": "#DC2626"}
    fill_markers = {"BUY": "triangle-up", "COVER": "circle", "SELL": "triangle-down", "SHORT": "triangle-down"}
    for fill_action in ("BUY", "SELL", "SHORT", "COVER"):
        action_fills = [fill for fill in symbol_fills if fill.action.value == fill_action]
        if not action_fills:
            continue
        figure.add_trace(
            go.Scatter(
                x=[fill.timestamp for fill in action_fills],
                y=[fill.fill_price for fill in action_fills],
                mode="markers",
                name=f"Paper {fill_action}",
                marker={"size": 10, "color": fill_colors[fill_action],
                        "symbol": fill_markers[fill_action],
                        "line": {"color": "#FFFFFF", "width": 1}},
                hovertemplate=(
                    f"Paper {fill_action}<br>%{{x|%b %d, %Y}}<br>Fill: "
                    f"{currency_symbol}%{{y:,.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    for row in (1, 2, 3):
        figure.add_vline(
            x=decision_time,
            line_dash="dash",
            line_color="#475569",
            line_width=1.2,
            row=row,
            col=1,
        )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#94A3B8", row=2, col=1)
    figure.update_yaxes(title_text=f"Price ({currency_code})", tickprefix=currency_symbol,
                        tickformat=",.2f", row=1, col=1)
    figure.update_yaxes(title_text="Return / spread", tickformat=".1%", row=2, col=1)
    figure.update_yaxes(title_text="Daily volatility", tickformat=".1%", row=3, col=1)
    figure.update_xaxes(title_text="Date", row=3, col=1)
    figure.update_layout(xaxis_rangeslider_visible=False)
    title = f"{symbol} · AI signal and model inputs"
    return _polish_figure(
        figure,
        title=title,
        height=690,
        hovermode="closest",
        show_legend=False,
    )


def render_paper_trading(
    prices: pd.DataFrame,
    start_date: str | None = None,
    market_key: str = "default",
    currency_code: str = "USD",
    currency_symbol: str = "$",
    ticker_catalog: tuple[TickerOption, ...] = (),
) -> None:
    """Render a user-controlled paper ledger, watchlist, and advisory ML alert."""
    st.subheader("Paper trading simulator")
    st.caption(f"Prices, cash, and P&L use the market's local quote currency ({currency_code}); no FX "
               "conversion is applied. Orders simulate the latest close with fees and slippage. "
               "The research agent only creates alerts; it never places orders.")
    if prices.empty or prices.dropna(how="all").empty:
        st.warning("Price history is unavailable for the simulator.")
        return

    key_prefix = f"paper_{market_key}"
    state_key = f"{key_prefix}_simulator"
    tickers_key = f"{key_prefix}_tradable_tickers"
    market_data_key = f"{key_prefix}_market_data"
    # Preserve the original US account through the change to market-scoped keys.
    if market_key == "us" and state_key not in st.session_state:
        legacy = st.session_state.get("paper_trading_simulator")
        if legacy is not None:
            st.session_state[state_key] = legacy
            if "paper_tradable_tickers" in st.session_state:
                st.session_state[tickers_key] = st.session_state["paper_tradable_tickers"]
            if "paper_market_data" in st.session_state:
                st.session_state[market_data_key] = st.session_state["paper_market_data"]
    ticker_selection_key = f"{key_prefix}_initial_ticker_selection"
    saved_selection = _stored_tickers(st.session_state.get(ticker_selection_key))
    if saved_selection is None:
        saved_selection = _stored_tickers(st.session_state.get(f"{key_prefix}_initial_tickers"))
    initial_tickers = saved_selection if saved_selection is not None else tuple(
        str(ticker).upper() for ticker in prices.columns
    )
    picker_tickers, picker_labels = _ticker_menu(
        ticker_catalog, (*prices.columns, *initial_tickers)
    )
    if state_key not in st.session_state:
        selected_tickers = st.multiselect(
            "Stocks available to trade",
            picker_tickers,
            default=list(initial_tickers),
            format_func=lambda symbol: _ticker_label(symbol, picker_labels),
            help="Search by company name or ticker, then select multiple stocks for this paper account.",
            key=ticker_selection_key,
        )
        initial_cash = st.number_input(
            f"Starting paper cash ({currency_code})", 1_000.0, 10_000_000.0,
            100_000.0, 5_000.0, key=f"{key_prefix}_initial_cash",
        )
        if not st.button("Start paper account", key=f"{key_prefix}_start_account"):
            st.info("Choose the stocks for your account, then start it to practice with market data.")
            return
        tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in selected_tickers))
        if not tickers:
            st.error("Choose at least one stock for the paper account.")
            return
        data_start = start_date or str(prices.index[0].date())
        try:
            paper_prices = _load_paper_prices(prices, tickers, data_start)
        except (DataEngineError, ValueError, RuntimeError) as exc:
            st.error(f"Could not load that trading universe: {exc}")
            return
        st.session_state[state_key] = PaperSimulator(
            starting_cash=float(initial_cash), commission_bps=1.0, slippage_bps=2.0
        )
        st.session_state[tickers_key] = list(tickers)
        st.session_state[market_data_key] = paper_prices

    simulator = st.session_state[state_key]
    # Upgrade an account created by the previous UI version without discarding it.
    if market_data_key not in st.session_state:
        st.session_state[market_data_key] = prices
        st.session_state[tickers_key] = list(prices.columns)
    data_start = start_date or str(prices.index[0].date())
    with st.expander(f"Manage {market_key.upper()} tradable stocks"):
        current_tickers = list(st.session_state[tickers_key])
        account_picker_tickers, account_picker_labels = _ticker_menu(
            ticker_catalog, (*picker_tickers, *current_tickers)
        )
        addable_tickers = tuple(
            ticker for ticker in account_picker_tickers if ticker not in current_tickers
        )
        requested_additions = st.multiselect(
            "Stocks to add",
            addable_tickers,
            format_func=lambda symbol: _ticker_label(symbol, account_picker_labels),
            help="Search the regional stock list by company name or ticker.",
            key=f"{key_prefix}_add_ticker_selection",
        )
        if st.button("Add stocks to account", key=f"{key_prefix}_add_button"):
            additions = [
                str(ticker).strip().upper()
                for ticker in requested_additions
                if str(ticker).strip().upper() not in current_tickers
            ]
            if not additions:
                st.info("Choose at least one stock that is not already in this account.")
            else:
                combined = tuple(current_tickers + additions)
                try:
                    updated_prices = _load_paper_prices(prices, combined, data_start)
                    old_history = st.session_state[market_data_key]
                    archived = [
                        ticker for ticker in old_history.columns if str(ticker).upper() not in combined
                    ]
                    if archived:
                        updated_prices = pd.concat(
                            [old_history.loc[:, archived], updated_prices], axis=1
                        ).sort_index()
                    st.session_state[market_data_key] = updated_prices
                    st.session_state[tickers_key] = list(combined)
                except (DataEngineError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not add the requested ticker(s): {exc}")

        current_tickers = list(st.session_state[tickers_key])
        remove_ticker = st.selectbox(
            "Remove a stock",
            current_tickers,
            format_func=lambda symbol: _ticker_label(symbol, account_picker_labels),
            key=f"{key_prefix}_remove_ticker",
        )
        if st.button("Remove ticker", key=f"{key_prefix}_remove_button"):
            held = {position.symbol for position in simulator.snapshot().positions}
            if remove_ticker in held:
                st.error("Close this paper position before removing its ticker.")
            elif len(current_tickers) == 1:
                st.error("Keep at least one ticker in the account.")
            else:
                remaining = [ticker for ticker in current_tickers if ticker != remove_ticker]
                st.session_state[tickers_key] = remaining

    history = st.session_state[market_data_key].dropna(how="all").sort_index()
    if history.empty or history.dropna(how="all").empty:
        st.warning("No usable price history is available for this account's stocks.")
        return
    latest = history.dropna(how="all").iloc[-1].dropna()
    simulator.mark({str(k): float(v) for k, v in latest.items()})

    tradable_tickers = list(st.session_state[tickers_key])
    ticker = st.selectbox(
        "Instrument",
        tradable_tickers,
        format_func=lambda symbol: _ticker_label(symbol, account_picker_labels),
        key=f"{key_prefix}_ticker",
    )
    ticker_history = history[ticker].dropna()
    ohlcv_history: pd.DataFrame | None = None
    try:
        ohlcv_history = load_ohlcv_history(ticker, data_start)
        raw_close = pd.to_numeric(ohlcv_history["Close"], errors="coerce").dropna().sort_index()
        if not raw_close.empty:
            # The model, chart, and next-close fill reference share the same unfilled bars.
            ticker_history = raw_close
    except (DataEngineError, ValueError, RuntimeError) as exc:
        st.warning(
            f"Candlestick history is unavailable for {ticker}; the AI alert will use "
            f"the dashboard's close-price history. Details: {exc}"
        )
    if len(ticker_history) < 2:
        st.warning("At least two closes are required to create an alert and simulate a later fill.")
        return
    signal_time = ticker_history.index[-2]
    fill_time = ticker_history.index[-1]
    fill_price = float(ticker_history.iloc[-1])
    if ohlcv_history is not None:
        simulator.mark({ticker: fill_price})
    account = simulator.snapshot()
    current_position = next((p for p in account.positions if p.symbol == ticker), None)
    quantity_held = 0.0 if current_position is None else current_position.quantity
    alert: AgentAlert | None = None
    research_agent = ResearchAgent()
    try:
        alert = research_agent.evaluate(
            ticker_history, ticker, as_of=signal_time, current_quantity=quantity_held
        )
        st.info(f"**Research agent: {alert.action.value}** · signal through {alert.as_of.date()} · "
                f"confidence {alert.confidence:.1%}\n\n{alert.explanation}")
        st.caption(f"Model trained on {alert.training_observations:,} historical feature rows. "
                   f"Actionable alerts require at least {research_agent.confidence_threshold:.0%} "
                   f"confidence; lower-confidence predictions stay HOLD. The simulated fill uses "
                   f"the next available close ({fill_time.date()}) to avoid "
                   "same-bar look-ahead. Predictions are uncertain and are not personalized advice.")
    except AgentError as exc:
        st.warning(f"No alert available: {exc}")

    if ohlcv_history is not None:
        st.markdown("**What the model measured at the decision close**")
        feature_readout = paper_feature_readout(ticker_history, signal_time, alert)
        if not feature_readout.empty:
            _display_table(
                feature_readout,
                hide_index=True,
                percent_columns=("reading",),
                column_labels={
                    "feature": "Model input",
                    "reading": "Value at decision close",
                    "definition": "Definition",
                },
                height=250,
            )
        st.caption(
            "Candles and moving averages provide price context. The lower panels plot the "
            "model's trailing momentum, moving-average spread, and realized volatility through "
            "the dashed cutoff. Hover a line or marker for its name and value. The next candle "
            "is only a paper-fill reference and is excluded from the model features."
        )
        signal_chart = paper_signal_figure(
            ohlcv_history,
            ticker,
            signal_time,
            alert,
            account.fills,
            currency_symbol,
            currency_code,
        )
        if signal_chart is not None:
            _display_chart(signal_chart)
        else:
            st.warning("The available OHLC history is too short to draw the AI analysis chart.")

    metric_cols = st.columns(4)
    metric_cols[0].metric(f"Equity ({currency_code})", f"{currency_symbol}{account.equity:,.2f}")
    metric_cols[1].metric(f"Cash ({currency_code})", f"{currency_symbol}{account.cash:,.2f}")
    metric_cols[2].metric("Positions", str(len(account.positions)))
    metric_cols[3].metric("Latest close", f"{currency_symbol}{fill_price:,.2f}",
                          help=f"Market data timestamp: {fill_time}")

    with st.form(f"{key_prefix}_order_form"):
        action_name = st.selectbox("Paper order action", [a.value for a in TradeAction],
                                   key=f"{key_prefix}_order_action")
        quantity = st.number_input("Quantity (shares)", min_value=0.001, value=1.0, step=1.0,
                                   key=f"{key_prefix}_order_quantity")
        submitted = st.form_submit_button("Place paper order", key=f"{key_prefix}_submit_order")
    if submitted:
        try:
            fill = simulator.submit_order(
                ticker, TradeAction(action_name), float(quantity), fill_price,
                timestamp=fill_time.to_pydatetime(),
            )
            st.success(f"Simulated {fill.action.value} {fill.quantity:g} {fill.symbol} at "
                       f"{currency_symbol}{fill.fill_price:,.2f}; commission "
                       f"{currency_symbol}{fill.commission:,.2f}.")
            account = simulator.snapshot()
        except OrderError as exc:
            st.error(str(exc))

    if account.positions:
        st.markdown("**Open paper positions**")
        positions = pd.DataFrame([{
            "Symbol": p.symbol, "Shares": p.quantity,
            f"Average entry ({currency_code})": p.average_entry_price,
            f"Latest mark ({currency_code})": p.market_price,
            f"Market value ({currency_code})": p.market_value,
            f"Unrealized P&L ({currency_code})": p.unrealized_pnl,
            f"Realized P&L ({currency_code})": p.realized_pnl,
        } for p in account.positions]).set_index("Symbol")
        _display_table(
            positions,
            index_name="Symbol",
            currency_columns=tuple(positions.columns[1:]),
            currency_symbol=currency_symbol,
            number_formats={"Shares": "%,.3f"},
        )
    if account.fills:
        traded_symbols = sorted({fill.symbol for fill in account.fills})
        st.markdown("**Stock price charts with paper trades**")
        for traded_symbol in traded_symbols:
            st.markdown(f"**{traded_symbol}**")
            if traded_symbol not in history.columns:
                st.info(f"No retained price history is available for {traded_symbol}.")
                continue
            trade_chart = paper_trade_figure(
                history[traded_symbol],
                traded_symbol,
                account.fills,
                currency_symbol,
                currency_code,
            )
            if trade_chart is not None:
                _display_chart(trade_chart)
            else:
                st.line_chart(
                    history[traded_symbol], **_wide_widget_kwargs(st.line_chart)
                )

        st.markdown("**Order audit log**")
        audit_log = pd.DataFrame([{
            "Time": f.timestamp, "Symbol": f.symbol, "Action": f.action.value,
            "Shares": f.quantity, "Reference close": f.reference_price,
            "Fill price": f.fill_price, "Commission": f.commission,
        } for f in account.fills]).sort_values("Time", ascending=False)
        _display_table(
            audit_log,
            currency_columns=("Reference close", "Fill price", "Commission"),
            currency_symbol=currency_symbol,
            number_formats={"Shares": "%,.3f"},
            height=360,
        )
    if st.button("Reset paper account", key=f"{key_prefix}_reset_account"):
        del st.session_state[state_key]
        st.rerun()


def _load_paper_prices(
    existing_prices: pd.DataFrame, tickers: tuple[str, ...], start_date: str
) -> pd.DataFrame:
    """Use the loaded dashboard panel when possible, otherwise fetch requested symbols."""
    existing = existing_prices.dropna(how="all").sort_index()
    if set(tickers).issubset({str(column).upper() for column in existing.columns}):
        columns = {str(column).upper(): column for column in existing.columns}
        return existing.loc[:, [columns[ticker] for ticker in tickers]].rename(
            columns={columns[ticker]: ticker for ticker in tickers}
        )

    loaded = load_prices(tickers, start_date)
    columns = {str(column).upper(): column for column in loaded.columns}
    missing = [ticker for ticker in tickers if ticker not in columns]
    if missing:
        raise ValueError(f"No price data returned for: {', '.join(missing)}")
    return loaded.loc[:, [columns[ticker] for ticker in tickers]].rename(
        columns={columns[ticker]: ticker for ticker in tickers}
    )


_TRADING_101_LESSONS: dict[str, str] = {
    "1 · Start with the basics": """### Shares, tickers, and returns

A **share** is a unit of ownership in a company. A **ticker** is its short market symbol.
For example, this site lets you search by company name or ticker in the market selector.

**Simple price return** = (ending price − starting price) ÷ starting price. If a share moves
from $100 to $105, its price return is 5% before fees. A return can also be negative, and
past price moves do not predict future returns.

Learn more: [Investor.gov stock basics](https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks).""",
    "2 · Read a candlestick": """### What one candle shows

Each candle summarizes a time interval using four prices: **open**, **high**, **low**, and
**close**. A candle body compares the open with the close; its thin lines show the high and
low reached during that interval. This chart uses daily bars. Candle colors are a visual
convention and can differ between charting tools.

In the paper-trading chart, the dashed line marks when the model made its decision. The next
available close is shown as a fill reference, and is excluded from the model inputs.""",
    "3 · Understand trade actions": """### Long and short positions

- **Buy** opens or adds to a long position: you benefit if the price rises.
- **Sell** reduces or closes a long position.
- **Short** opens or adds to a short position: you benefit if the price falls.
- **Cover** reduces or closes a short position.

Short selling can lose money as the price rises. This simulator is simplified: it applies
commission and slippage, but does not model real stock borrowing, margin calls, or a live
order book. It fills your paper order using the displayed close. Real market orders can fill
at a different price, while a limit order may not fill at all. Read [FINRA's overview of
order types](https://syndication.finra.org/content/understanding-order-types-can-save-time-and-money)
and the [SEC's short-sale overview](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-51).""",
    "4 · Learn the risk tools": """### Risk is part of every trade

**Diversification** spreads exposure across investments, but cannot remove all risk.
**Volatility** describes how much returns have varied; it does not say which way a price will
move next. The portfolio optimizer's maximum weight limits concentration in one stock.

**Value at Risk (VaR)** estimates a loss threshold over a chosen time horizon and confidence
level. It is not a maximum-loss estimate: losses can exceed VaR. Stress tests show how a
portfolio would respond to selected historical or hypothetical shocks.

Learn more: [Investor.gov on diversification](https://www.investor.gov/introduction-investing/getting-started/asset-allocation).""",
    "5 · Read the AI research alert": """### What the alert means

The research agent estimates the probability of a positive or negative next-session return
from five trailing inputs: 5-, 21-, and 63-session momentum; 21-session realized volatility;
and the 20/60-session moving-average spread. The chart and table show those inputs at the
decision close.

Actionable alerts require the model's stronger direction to reach its 90% probability
threshold; otherwise the suggestion is **HOLD**. That percentage is a model estimate, not a
90% proven win rate or a promise. The agent only suggests actions and never places orders.""",
    "6 · Follow a practice routine": """### A guided tour of this site

1. Choose a **market** and search for stocks by company name or ticker in the sidebar.
2. Start in **Market data** to review prices and indicators.
3. Use **Pairs trading** to study whether two stocks have moved together historically.
4. Use **Portfolio optimizer** to explore sample-based portfolio weights.
5. Review **Risk & stress** to inspect loss estimates and scenario results.
6. Open **Paper trading**, choose the stocks for that market's separate practice account,
   and start with an amount you can track comfortably.
7. Before placing an order, write down why you chose it. Afterwards, review the fill, costs,
   position size, and result. Treat the AI alert as one research input, not an instruction.

All market data and model outputs can be incomplete or wrong. This site is for education and
simulation; it does not connect to a broker or provide personalized financial advice.""",
}


def render_trading_tutorial() -> None:
    """Render a concise, interactive trading guide before the market tools."""
    with st.expander("Trading 101 · Beginner tutorial", expanded=False):
        st.markdown(
            "A short, practical guide to the charts, risk measures, and paper account below. "
            "Choose a lesson, then try the practice simulator when you are ready."
        )
        lesson = st.selectbox(
            "Choose a lesson",
            tuple(_TRADING_101_LESSONS),
            key="trading_101_lesson",
        )
        st.markdown(_TRADING_101_LESSONS[lesson])
        st.divider()
        answer = st.selectbox(
            "Quick check: which candle value is the last price in that interval?",
            (
                "Choose an answer",
                "Close",
                "Open",
                "High",
                "Low",
            ),
            key="trading_101_candle_quiz",
        )
        if answer == "Close":
            st.success("Correct. The close is the interval's final recorded price.")
        elif answer != "Choose an answer":
            st.info("Try again: the high and low are extremes, while the open is the first price.")


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is not installed: pip install -r requirements.txt")
    st.set_page_config(page_title="Quant Platform", layout="wide")
    st.title("Algorithmic Trading & Portfolio Optimization Platform")
    st.caption(
        "Explore market data, compare research methods, measure risk, and practice with a "
        "simulated account. Educational use only; results are estimates, not investment advice."
    )
    render_trading_tutorial()

    with st.sidebar:
        st.header("Settings")
        st.caption("Choose a market and stock universe here. Each market has its own paper account.")
        market: MarketSection = st.selectbox(
            "Market section", MARKET_SECTIONS, format_func=lambda item: item.name,
            key="market_section",
            help="Select a region to change the ticker list, local quote currency, and paper account.",
        )

    try:
        with st.spinner(f"Loading {market.name} ticker choices..."):
            ticker_catalog = load_market_ticker_catalog(market.key)
    except TickerCatalogError as exc:
        logger.warning("Regional ticker catalog unavailable: %s", exc)
        ticker_catalog = ()
        st.warning(
            "The expanded ticker directory is temporarily unavailable. The built-in market "
            "choices remain available; try again later to browse the full list."
        )

    portfolio_state_key = f"portfolio_ticker_selection_{market.key}"
    pairs_state_key = f"pairs_ticker_selection_{market.key}"
    saved_portfolio = _stored_tickers(st.session_state.get(portfolio_state_key))
    saved_pairs = _stored_tickers(st.session_state.get(pairs_state_key))
    if saved_portfolio is None:
        saved_portfolio = _stored_tickers(
            st.session_state.get(f"portfolio_universe_{market.key}")
        )
    if saved_pairs is None:
        saved_pairs = _stored_tickers(st.session_state.get(f"pairs_universe_{market.key}"))
    portfolio_default = saved_portfolio or market.portfolio_tickers
    pairs_default = saved_pairs or market.pairs_tickers
    portfolio_options, portfolio_labels = _ticker_menu(
        ticker_catalog, (*market.portfolio_tickers, *portfolio_default)
    )
    pairs_options, pairs_labels = _ticker_menu(
        ticker_catalog, (*market.pairs_tickers, *pairs_default)
    )

    with st.sidebar:
        if ticker_catalog:
            st.caption(
                f"Browsing {len(ticker_catalog)} regional listings. Search by company name or ticker."
            )
        portfolio_selection = st.multiselect(
            "Portfolio stocks",
            portfolio_options,
            default=list(portfolio_default),
            format_func=lambda symbol: _ticker_label(symbol, portfolio_labels),
            help="Search by company name or ticker, then choose multiple stocks for analysis.",
            key=portfolio_state_key,
        )
        portfolio_tickers = _stored_tickers(portfolio_selection)
        if portfolio_tickers is None:
            portfolio_tickers = portfolio_default
        pairs_selection = st.multiselect(
            "Pairs trading stocks",
            pairs_options,
            default=list(pairs_default),
            format_func=lambda symbol: _ticker_label(symbol, pairs_labels),
            help="Choose at least two stocks to search for cointegrated pairs.",
            key=pairs_state_key,
        )
        pairs_tickers = _stored_tickers(pairs_selection)
        if pairs_tickers is None:
            pairs_tickers = pairs_default
        start = st.date_input(
            "History starts",
            dt.date(2018, 1, 1),
            help="First date used to load prices. Longer histories may take longer to download.",
        )
        formation_end = st.date_input(
            "Formation period ends",
            dt.date(2024, 12, 31),
            help="End of the historical sample used to estimate strategy and portfolio inputs.",
        )
        oos_start = st.date_input(
            "Out-of-sample starts",
            dt.date(2025, 1, 1),
            help="Start of the held-out period used to review results after estimation.",
        )
        rf = st.number_input(
            "Risk-free rate (annual)",
            0.0,
            0.15,
            0.03,
            0.005,
            format="%.3f",
            help="Annual comparison rate used in risk-adjusted portfolio calculations.",
        )
        max_weight = st.slider(
            "Max weight per asset",
            0.15,
            1.0,
            0.40,
            0.05,
            help="Upper limit on how much of the optimized portfolio can go to one stock.",
        )
        estimator = st.selectbox(
            "Expected-return estimator",
            ["bayes_stein", "historical"],
            index=0,
            help="Choose how expected returns are estimated from the formation period.",
        )
        value = st.number_input(
            f"Portfolio value ({market.currency_code})", 10_000.0, 1e9, 1_000_000.0,
            50_000.0, key=f"portfolio_value_{market.key}",
            help="Practice account value used to scale portfolio risk estimates.",
        )
        horizon = st.selectbox(
            "VaR horizon (days)",
            [1, 5, 10],
            index=0,
            help="Number of trading days in each loss estimate.",
        )
        confidences = tuple(sorted(st.multiselect(
            "Confidence levels",
            [0.90, 0.95, 0.975, 0.99],
            default=[0.95, 0.99],
            help="Higher confidence levels show more severe loss thresholds; losses can exceed VaR.",
        ))) or (0.95, 0.99)

    if not start < formation_end < oos_start:
        st.error("Dates must satisfy: history start < formation end < out-of-sample start.")
        st.stop()
    if len(portfolio_tickers) < 2:
        st.error("Choose at least two stocks for portfolio research.")
        st.stop()
    if len(pairs_tickers) < 2:
        st.error("Choose at least two stocks for pairs trading.")
        st.stop()
    start_s, formation_s, oos_s = str(start), str(formation_end), str(oos_start)
    st.header(f"{market.name} market section")
    st.caption(
        f"{market.quote_note}. Prices, risk values, and paper accounts use local quote currency "
        f"({market.currency_code}) and are not converted to USD. The macro panel uses the platform's "
        "shared U.S. indicators."
    )

    try:
        with st.spinner("Downloading and cleaning market data..."):
            prices = load_prices(tuple(portfolio_tickers), start_s)
            pair_prices = load_prices(tuple(pairs_tickers), start_s)
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
                            int(horizon), confidences, oos_s, start_s,
                            market.currency_code, market.currency_symbol)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Risk analysis failed: {exc}")
    with tabs[4]:
        try:
            render_paper_trading(
                prices,
                start_s,
                market.key,
                market.currency_code,
                market.currency_symbol,
                ticker_catalog,
            )
        except Exception as exc:  # noqa: BLE001 - isolate optional research UI failures
            st.error(f"Paper trading research tool failed: {exc}")


if __name__ == "__main__":
    main()

