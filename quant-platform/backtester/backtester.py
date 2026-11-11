"""
backtester.py -- Statistical Arbitrage (Pairs Trading) Backtester (Phase 3)
===========================================================================

Pipeline (every step uses only information available at that bar's close)
-------------------------------------------------------------------------
1. Rolling Engle-Granger cointegration test on log prices.
2. Out-of-sample spread and dynamic Z-score.
3. Hysteresis signal state machine (entry / target / stop / cointegration break).
4. Delayed order execution, frozen hedge ratio, transaction costs, borrow cost.
5. Equity curve, trade log, performance statistics, logged report.

1. Cointegration (Engle & Granger, 1987)
----------------------------------------
Model log prices:  ln Py_t = a + b * ln Px_t + e_t
Step 1 -- OLS over the trailing ``formation_window`` W:
    b = Sxy / Sxx,   a = mean(ln Py) - b * mean(ln Px),   e_t = ln Py_t - a - b ln Px_t
Step 2 -- Augmented Dickey-Fuller regression on the residuals (no constant,
because OLS residuals already have zero mean), with p lagged differences:
    De_t = g * e_{t-1} + sum_{k=1..p} f_k * De_{t-k} + u_t
    H0: g = 0 (unit root => no cointegration).   Statistic: t = g_hat / se(g_hat)
The p-value must come from MacKinnon's response surfaces for the
*residual-based* test (statsmodels ``mackinnonp(N=2, regression="c")``), NOT the
plain Dickey-Fuller table. Residuals come from a fitted b chosen to make them look
stationary, so the test is biased toward rejection; the correct 5% critical value
is about -3.34, versus about -1.94 for a plain no-constant Dickey-Fuller test.
All W-sized regressions for all dates are solved at once through batched normal
equations (X'X) c = X'y, so there is no per-date Python loop.
Note: Engle-Granger is order-dependent (y on x vs x on y). ``screen_pairs`` tests both.

2. Spread, half-life and Z-score
--------------------------------
Out-of-sample spread (parameters lagged one bar, so the bar being judged was not
in the regression that produced its own residual):
    s_t = ln Py_t - b_{t-1} ln Px_t - a_{t-1}
Two dynamic Z-score definitions (``zscore_mode``):
  "formation" (default): the spread is already centered by the lagged fit, so
      Z_t = s_t / sd_{t-1},  sd = OLS residual std of the formation window.
      It measures the deviation from the long-run equilibrium, which is what
      mean reversion actually pulls toward.
  "rolling": Z_t = (s_t - mean_n(s)) / std_n(s) over ``zscore_window`` n (ddof=1).
      A short rolling mean partly absorbs the very deviation being measured, which
      weakens the signal when reversion is slow relative to n; it is useful when the
      equilibrium itself drifts. Compare both on your own data.
Ornstein-Uhlenbeck half-life from the residual AR(1) coefficient:
    phi = sum(e_t e_{t-1}) / sum(e_{t-1}^2),   half-life = -ln 2 / ln(phi),  0 < phi < 1
A spread that reverts in 2 days is usually noise; one that takes 200 is untradeable.

3. Signal state machine
-----------------------
Entry: Z < -entry_z => long spread (+1); Z > +entry_z => short spread (-1),
only if p < entry_pvalue and half-life is inside [min, max].
Exit: |Z| < exit_z (target).  Stop: |Z| > stop_z.  Break: p > break_pvalue.
Every event is defined from the current bar alone (never from the current
position), so the sequential machine  "set +1 / -1 / 0 on events, otherwise keep
the previous value"  is exactly  events.ffill(). That removes the Python loop
while remaining provably identical (tests compare against a literal loop).
The exit band lies strictly inside the entry band and the stop strictly outside,
so events on one bar can never conflict.

4. Execution, sizing and costs
------------------------------
Signal at close t; the order fills ``fill_delay_bars`` bars later, so the first
return earned is that of bar t + delay + 1:  position_s = signal_{s - delay - 1}.
(delay 0 = fill at the signal bar's own close, which is optimistic.)
One unit of spread = long $1 of Y and short $b of X (b is the log-log hedge
ratio, so the exposures offset in return space). Weights are normalized to 100%
gross:  w_y = pos / (1 + |b|),  w_x = -pos * b / (1 + |b|).
The hedge ratio is frozen at entry and held until exit; weights are treated as
rebalanced daily to that ratio (drift rebalancing costs are ignored).
    gross_t  = w_y,t * Ry_t + w_x,t * Rx_t              (simple returns R)
    turnover = |dw_y| + |dw_x|            cost   = turnover * (commission + slippage)
    borrow_t = (short weight) * borrow_bps / 10^4 / 252
    net_t    = gross_t - cost_t - borrow_t
Costs are booked on the first bar the changed position is in force.

5. Statistics
-------------
Sharpe = mean(r - rf/252) / std(r) * sqrt(252);  its standard error under i.i.d.
returns (Lo, 2002) is sqrt((1 + SR^2/2) / years): with a few years of data the
error bar is large, so a Sharpe of 1 is rarely statistically distinguishable from 0.
Sortino uses downside deviation; Calmar = CAGR / |max drawdown|.

Limitations to state in any write-up
------------------------------------
* Choosing a pair after looking at the whole sample is selection bias. Select
  pairs with ``screen_pairs`` on a formation period, then judge performance only
  after it (``BacktestResult.performance(start=...)``).
* ``screen_pairs`` tests many pairs; its Bonferroni column is the honest p-value.
* Daily closes only; no intraday fills, no market impact, constant borrow cost.
* Weights are not dividend- or margin-interest-adjusted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from statsmodels.tsa.adfvalues import mackinnonp

__all__ = [
    "BacktestError",
    "BacktestResult",
    "PairsBacktestConfig",
    "PairsBacktester",
    "build_signal",
    "engle_granger_test",
    "rolling_engle_granger",
    "screen_pairs",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
_EPS = 1e-12


class BacktestError(Exception):
    """Invalid input data or configuration for the backtester."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairsBacktestConfig:
    """Immutable strategy and cost parameters.

    Cost defaults are deliberately not zero: ``commission_bps`` and
    ``slippage_bps`` are charged per unit of notional traded (each side);
    ``borrow_bps_annual`` is a typical easy-to-borrow large-cap fee.
    """

    formation_window: int = 252
    adf_lags: int = 1
    zscore_mode: Literal["formation", "rolling"] = "formation"
    zscore_window: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    entry_pvalue: float = 0.05
    break_pvalue: float = 0.15
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    fill_delay_bars: int = 1
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 50.0
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.0
    trading_days: int = 252
    chunk_size: int = 1024

    def __post_init__(self) -> None:
        if self.adf_lags < 0:
            raise BacktestError("adf_lags must be >= 0")
        if self.formation_window < self.adf_lags + 30:
            raise BacktestError("formation_window too short for a meaningful ADF regression")
        if self.zscore_mode not in ("formation", "rolling"):
            raise BacktestError("zscore_mode must be 'formation' or 'rolling'")
        if self.zscore_window < 5:
            raise BacktestError("zscore_window must be >= 5")
        if not 0.0 <= self.exit_z < self.entry_z < self.stop_z:
            raise BacktestError("require 0 <= exit_z < entry_z < stop_z")
        if not 0.0 < self.entry_pvalue <= self.break_pvalue <= 1.0:
            raise BacktestError("require 0 < entry_pvalue <= break_pvalue <= 1")
        if not 0.0 < self.min_half_life < self.max_half_life:
            raise BacktestError("require 0 < min_half_life < max_half_life")
        if self.fill_delay_bars < 0:
            raise BacktestError("fill_delay_bars must be >= 0")
        if min(self.commission_bps, self.slippage_bps, self.borrow_bps_annual) < 0:
            raise BacktestError("costs must be non-negative")
        if self.initial_capital <= 0 or self.trading_days <= 0 or self.chunk_size < 1:
            raise BacktestError("initial_capital, trading_days and chunk_size must be positive")


# --------------------------------------------------------------------------- #
# Engle-Granger test, vectorized over many windows at once
# --------------------------------------------------------------------------- #
def _eg_block(yw: FloatArray, xw: FloatArray, adf_lags: int) -> dict[str, FloatArray]:
    """Engle-Granger on a block of windows. ``yw``, ``xw`` have shape (B, W).

    Returns arrays of length B: beta, alpha, adf_stat, half_life, resid_std
    (NaN where undefined). ``resid_std`` is the OLS residual std, sqrt(SSR/(W-2)).
    """
    n_win, width = yw.shape
    y_mean = yw.mean(axis=1)
    x_mean = xw.mean(axis=1)
    xc = xw - x_mean[:, None]
    yc = yw - y_mean[:, None]

    # Step 1: OLS slope and intercept for every window.
    sxx = np.einsum("bw,bw->b", xc, xc)
    sxy = np.einsum("bw,bw->b", xc, yc)
    ok = sxx > _EPS
    beta = np.divide(sxy, sxx, out=np.full(n_win, np.nan), where=ok)
    alpha = y_mean - beta * x_mean
    # alpha absorbs the means, so residual = centered y - beta * centered x.
    e = np.where(ok[:, None], yc - np.nan_to_num(beta)[:, None] * xc, 0.0)

    # Step 2: ADF regression  De_t = g e_{t-1} + sum_k f_k De_{t-k} + u_t.
    p = adf_lags
    de = np.diff(e, axis=1)  # De, shape (B, W-1)
    dep = de[:, p:]  # dependent variable
    columns = [e[:, p : width - 1]] + [de[:, p - k : width - 1 - k] for k in range(1, p + 1)]
    design = np.stack(columns, axis=-1)  # (B, m, p+1)
    m, k_par = design.shape[1], p + 1
    xtx = np.einsum("bmi,bmj->bij", design, design)
    xty = np.einsum("bmi,bm->bi", design, dep)
    inv = np.linalg.pinv(xtx)  # pinv tolerates degenerate windows
    coef = np.einsum("bij,bj->bi", inv, xty)
    resid = dep - np.einsum("bmi,bi->bm", design, coef)
    s2 = np.einsum("bm,bm->b", resid, resid) / max(m - k_par, 1)
    var_g = s2 * inv[:, 0, 0]
    good = ok & (var_g > 0) & (m > k_par)
    stat = np.divide(
        coef[:, 0], np.sqrt(np.where(good, var_g, 1.0)), out=np.full(n_win, np.nan), where=good
    )

    # Half-life from the AR(1) coefficient of the residual.
    num = np.einsum("bw,bw->b", e[:, 1:], e[:, :-1])
    den = np.einsum("bw,bw->b", e[:, :-1], e[:, :-1])
    phi = np.divide(num, den, out=np.full(n_win, np.nan), where=den > _EPS)
    valid = ok & (phi > 0) & (phi < 1)
    half_life = np.where(valid, -np.log(2.0) / np.log(np.where(valid, phi, 0.5)), np.nan)
    resid_std = np.where(ok, np.sqrt(np.einsum("bw,bw->b", e, e) / max(width - 2, 1)), np.nan)
    return {
        "beta": beta,
        "alpha": alpha,
        "adf_stat": stat,
        "half_life": half_life,
        "resid_std": resid_std,
    }


def _eg_pvalues(stat: FloatArray) -> FloatArray:
    """MacKinnon p-values (N=2 variables, constant in the cointegrating regression).

    ``mackinnonp`` evaluates a polynomial response surface for one scalar at a
    time, so this is the one place a Python loop is mathematically unavoidable.
    """
    out = np.full(stat.shape, np.nan)
    for i in np.flatnonzero(np.isfinite(stat)):
        out[i] = mackinnonp(float(stat[i]), regression="c", N=2)
    return out


def rolling_engle_granger(
    log_y: FloatArray, log_x: FloatArray, window: int, adf_lags: int = 1, chunk_size: int = 1024
) -> pd.DataFrame:
    """Engle-Granger test on every trailing window; row t uses data up to t only.

    Rows before the first full window are NaN. Windows are processed in chunks
    to cap memory at about ``chunk_size * window`` floats per temporary array.
    """
    y = np.ascontiguousarray(log_y, dtype=float)
    x = np.ascontiguousarray(log_x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise BacktestError("log_y and log_x must be 1-D arrays of equal length")
    n = len(y)
    if n < window:
        raise BacktestError(f"Need at least {window} observations, got {n}")

    y_view = sliding_window_view(y, window)
    x_view = sliding_window_view(x, window)
    n_win = y_view.shape[0]
    cols = {k: np.full(n, np.nan) for k in ("beta", "alpha", "adf_stat", "half_life", "resid_std")}
    for start in range(0, n_win, chunk_size):
        stop = min(start + chunk_size, n_win)
        block = _eg_block(
            np.ascontiguousarray(y_view[start:stop]), np.ascontiguousarray(x_view[start:stop]), adf_lags
        )
        for key, values in block.items():
            cols[key][window - 1 + start : window - 1 + stop] = values
    out = pd.DataFrame(cols)
    out["pvalue"] = _eg_pvalues(out["adf_stat"].to_numpy())
    return out


def engle_granger_test(y: FloatArray, x: FloatArray, adf_lags: int = 1) -> dict[str, float]:
    """Single full-sample Engle-Granger test on two *log-price* arrays."""
    ya = np.asarray(y, dtype=float)[None, :]
    xa = np.asarray(x, dtype=float)[None, :]
    if ya.shape != xa.shape or ya.shape[1] < adf_lags + 30:
        raise BacktestError("Series must have equal length and at least adf_lags + 30 points")
    res = _eg_block(ya, xa, adf_lags)
    stat = res["adf_stat"]
    return {
        "adf_stat": float(stat[0]),
        "pvalue": float(_eg_pvalues(stat)[0]),
        "beta": float(res["beta"][0]),
        "alpha": float(res["alpha"][0]),
        "half_life": float(res["half_life"][0]),
    }


def screen_pairs(prices: pd.DataFrame, adf_lags: int = 1) -> pd.DataFrame:
    """Rank all pairs by Engle-Granger p-value. Run this on a FORMATION period only.

    Both orderings (A on B, B on A) are tested and the better one kept, because the
    test is not symmetric. With N tickers there are N(N-1)/2 pairs, so a raw
    p < 0.05 appears by luck about once per 20 pairs; ``pvalue_bonferroni``
    multiplies by the number of pairs (conservative) for an honest read.
    """
    log_p = np.log(prices.where(prices > 0))
    if log_p.isna().any().any():
        raise BacktestError("prices must be positive and gap-free (use DataEngine output)")
    pairs = list(combinations(prices.columns, 2))
    if not pairs:
        raise BacktestError("Need at least two columns to form a pair")
    rows = []
    for a, b in pairs:  # pairs are independent regressions over a small list
        fwd = engle_granger_test(log_p[a].to_numpy(), log_p[b].to_numpy(), adf_lags)
        rev = engle_granger_test(log_p[b].to_numpy(), log_p[a].to_numpy(), adf_lags)
        use_fwd = np.nan_to_num(fwd["pvalue"], nan=1.0) <= np.nan_to_num(rev["pvalue"], nan=1.0)
        best = fwd if use_fwd else rev
        rows.append({"y": a if use_fwd else b, "x": b if use_fwd else a, **best})
    out = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    out["pvalue_bonferroni"] = np.minimum(1.0, out["pvalue"] * len(pairs))
    return out


# --------------------------------------------------------------------------- #
# Signal state machine
# --------------------------------------------------------------------------- #
def build_signal(
    force_flat: NDArray[np.bool_], long_entry: NDArray[np.bool_], short_entry: NDArray[np.bool_]
) -> pd.Series:
    """Position signal (+1 / 0 / -1) from per-bar events, without a loop.

    Events are functions of the current bar only, with priority flat > long >
    short. Leaving a bar with no event as NaN and forward-filling reproduces
    "keep the previous position" exactly.
    """
    events = np.select([force_flat, long_entry, short_entry], [0.0, 1.0, -1.0], default=np.nan)
    return pd.Series(events).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
_REPORT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Total return", "total_return", "{:.2%}"),
    ("CAGR", "cagr", "{:.2%}"),
    ("Annualized volatility", "ann_volatility", "{:.2%}"),
    ("Sharpe ratio", "sharpe", "{:.2f}"),
    ("Sharpe std. error (i.i.d.)", "sharpe_stderr", "{:.2f}"),
    ("Sortino ratio", "sortino", "{:.2f}"),
    ("Max drawdown", "max_drawdown", "{:.2%}"),
    ("Longest drawdown (days)", "max_drawdown_days", "{:.0f}"),
    ("Calmar ratio", "calmar", "{:.2f}"),
    ("Return skew / excess kurtosis", "skew", "{:.2f}"),
    ("Days in market", "exposure", "{:.1%}"),
    ("Annual turnover (x capital)", "annual_turnover", "{:.1f}"),
    ("Total costs (return pts)", "total_costs", "{:.2%}"),
    ("Trades", "n_trades", "{:.0f}"),
    ("Win rate", "win_rate", "{:.1%}"),
    ("Average trade return", "avg_trade_return", "{:.2%}"),
    ("Profit factor", "profit_factor", "{:.2f}"),
    ("Average holding (days)", "avg_holding_days", "{:.1f}"),
)


@dataclass(frozen=True)
class BacktestResult:
    """Everything a run produced. ``frame`` holds one row per bar."""

    pair: tuple[str, str]
    config: PairsBacktestConfig
    frame: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]

    def performance(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> dict[str, float]:
        """Metrics restricted to [start, end], e.g. out-of-sample after a formation period."""
        sub = self.frame.loc[start:end]
        if len(sub) < 2:
            raise BacktestError("Selected period contains fewer than 2 bars")
        trades = self.trades
        if not trades.empty:
            when = trades["entry_signal_date"]
            trades = trades[(when >= sub.index[0]) & (when <= sub.index[-1])]
        return _performance_metrics(sub, trades, self.config)

    def report(self, metrics: dict[str, float] | None = None) -> str:
        """Human-readable performance table."""
        m = self.metrics if metrics is None else metrics
        lines = [f"Pairs backtest: long/short {self.pair[0]} vs {self.pair[1]}"]
        lines.append(f"Period: {self.frame.index[0]:%Y-%m-%d} to {self.frame.index[-1]:%Y-%m-%d}")
        for label, key, fmt in _REPORT_FIELDS:
            value = m.get(key, float("nan"))
            text = "inf" if value == float("inf") else (fmt.format(value) if np.isfinite(value) else "n/a")
            if key == "skew":
                kurt = m.get("excess_kurtosis", float("nan"))
                text += " / " + (f"{kurt:.2f}" if np.isfinite(kurt) else "n/a")
            lines.append(f"  {label:<32}{text:>12}")
        return "\n".join(lines)

    def to_csv(self, directory: str | Path) -> None:
        """Write the per-bar frame, the trade log and the metrics (JSON) to ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(out / "backtest_frame.csv")
        self.trades.to_csv(out / "trades.csv", index=False)
        clean = {k: (float(v) if np.isfinite(v) else None) for k, v in self.metrics.items()}
        (out / "metrics.json").write_text(json.dumps(clean, indent=2))


# --------------------------------------------------------------------------- #
# Backtester
# --------------------------------------------------------------------------- #
class PairsBacktester:
    """Vectorized backtest of a cointegration-gated mean-reversion pair.

    Example:
        >>> bt = PairsBacktester(PairsBacktestConfig())
        >>> result = bt.run(prices, y="XOM", x="CVX")
        >>> print(result.report())
        >>> result.performance(start="2022-01-01")   # out-of-sample slice
    """

    def __init__(self, config: PairsBacktestConfig | None = None) -> None:
        self.cfg = config or PairsBacktestConfig()

    # ------------------------------------------------------------------ #
    def run(self, prices: pd.DataFrame, y: str, x: str) -> BacktestResult:
        """Backtest long/short ``y`` vs ``x`` on a gap-free close-price panel."""
        data = self._validate(prices, y, x)
        frame, reasons = self._signals(data, y, x)
        frame = self._simulate(frame, data, y, x)
        trades = self._trade_log(frame, reasons)
        metrics = _performance_metrics(frame, trades, self.cfg)
        result = BacktestResult((y, x), self.cfg, frame, trades, metrics)
        logger.info("\n%s", result.report())
        return result

    # ------------------------------------------------------------------ #
    def _validate(self, prices: pd.DataFrame, y: str, x: str) -> pd.DataFrame:
        cfg = self.cfg
        if y == x:
            raise BacktestError("y and x must be different instruments")
        missing = [c for c in (y, x) if c not in prices.columns]
        if missing:
            raise BacktestError(f"Columns not found in prices: {missing}")
        data = prices[[y, x]].astype(float)
        if not isinstance(data.index, pd.DatetimeIndex):
            raise BacktestError("prices must have a DatetimeIndex")
        if not data.index.is_monotonic_increasing or data.index.has_duplicates:
            raise BacktestError("index must be sorted ascending with no duplicates")
        bad = ~np.isfinite(data.to_numpy()) | (data.to_numpy() <= 0)
        if bad.any():
            raise BacktestError(
                f"prices must be finite and positive; found {int(bad.sum())} bad values "
                "(use DataEngine output, which is already cleaned)"
            )
        needed = cfg.formation_window + cfg.zscore_window + cfg.fill_delay_bars + 10
        if len(data) < needed:
            raise BacktestError(f"Need at least {needed} bars for this configuration, got {len(data)}")
        return data

    def _signals(self, data: pd.DataFrame, y: str, x: str) -> tuple[pd.DataFrame, pd.Series]:
        cfg = self.cfg
        ly, lx = np.log(data[y]), np.log(data[x])
        eg = rolling_engle_granger(
            ly.to_numpy(), lx.to_numpy(), cfg.formation_window, cfg.adf_lags, cfg.chunk_size
        )
        eg.index = data.index

        # Parameters lagged one bar: the bar being scored was not in its own regression.
        spread = ly - eg["beta"].shift(1) * lx - eg["alpha"].shift(1)
        if cfg.zscore_mode == "formation":
            # Mean is 0 by construction of the lagged fit; scale by the lagged residual std.
            sd = eg["resid_std"].shift(1)
            z = spread / sd.where(sd > 0)
        else:
            roll = spread.rolling(cfg.zscore_window, min_periods=cfg.zscore_window)
            sd = roll.std(ddof=1)
            z = (spread - roll.mean()) / sd.where(sd > 0)

        pval, hl = eg["pvalue"], eg["half_life"]
        tradable = (pval < cfg.entry_pvalue) & hl.between(cfg.min_half_life, cfg.max_half_life)
        stop = z.abs() > cfg.stop_z
        broken = pval > cfg.break_pvalue
        target = z.abs() < cfg.exit_z
        no_data = z.isna()
        long_entry = (z < -cfg.entry_z) & (z > -cfg.stop_z) & tradable
        short_entry = (z > cfg.entry_z) & (z < cfg.stop_z) & tradable

        signal = build_signal(
            (stop | broken | target | no_data).to_numpy(),
            long_entry.to_numpy(),
            short_entry.to_numpy(),
        )
        signal.index = data.index
        reason = np.select(
            [no_data.to_numpy(), stop.to_numpy(), broken.to_numpy(), target.to_numpy()],
            ["data", "stop_loss", "coint_break", "target"],
            default="reversal",
        )
        frame = pd.DataFrame(
            {
                "beta": eg["beta"],
                "alpha": eg["alpha"],
                "spread": spread,
                "zscore": z,
                "adf_stat": eg["adf_stat"],
                "pvalue": pval,
                "half_life": hl,
                "signal": signal,
            }
        )
        return frame, pd.Series(reason, index=data.index, name="reason")

    def _simulate(self, frame: pd.DataFrame, data: pd.DataFrame, y: str, x: str) -> pd.DataFrame:
        cfg = self.cfg
        lag = cfg.fill_delay_bars + 1  # signal at t earns the return of bar t + delay + 1
        sig = frame["signal"]
        entry = (sig != 0) & (sig != sig.shift(1, fill_value=0.0))

        # Hedge ratio frozen at the entry bar, carried to the exit, then delayed like the signal.
        beta_held = frame["beta"].where(entry).ffill().shift(lag)
        pos = sig.shift(lag, fill_value=0.0)
        denom = 1.0 + beta_held.abs()
        w_y = (pos / denom).fillna(0.0)
        w_x = (-pos * beta_held / denom).fillna(0.0)

        rets = data / data.shift(1) - 1.0
        gross = (w_y * rets[y] + w_x * rets[x]).fillna(0.0)
        turnover = (w_y.diff().abs() + w_x.diff().abs()).fillna(0.0)
        cost = turnover * (cfg.commission_bps + cfg.slippage_bps) / 1e4
        short_weight = (-w_y).clip(lower=0.0) + (-w_x).clip(lower=0.0)
        borrow = short_weight * cfg.borrow_bps_annual / 1e4 / cfg.trading_days
        net = gross - cost - borrow
        equity = cfg.initial_capital * (1.0 + net).cumprod()

        out = frame.copy()
        out.insert(0, f"price_{x}", data[x])
        out.insert(0, f"price_{y}", data[y])
        out["position"] = pos
        out["w_y"], out["w_x"] = w_y, w_x
        out["gross_return"], out["turnover"] = gross, turnover
        out["cost"], out["borrow"], out["net_return"] = cost, borrow, net
        out["equity"] = equity
        out["drawdown"] = equity / equity.cummax() - 1.0
        return out

    def _trade_log(self, frame: pd.DataFrame, reasons: pd.Series) -> pd.DataFrame:
        cfg = self.cfg
        lag = cfg.fill_delay_bars + 1
        sig, pos = frame["signal"], frame["position"]
        sig_prev = sig.shift(1, fill_value=0.0)
        entry = (sig != 0) & (sig != sig_prev)
        exit_ = (sig_prev != 0) & (sig != sig_prev)
        trade_id = entry.cumsum().where(sig != 0)  # constant within a continuous holding

        # Accounting id for P&L: held bars, plus the bar on which the position is closed
        # (so the exit cost lands in the trade that caused it).
        held_id = trade_id.shift(lag)
        closed_bar = (pos == 0) & (pos.shift(1, fill_value=0.0) != 0)
        acct_id = held_id.where(~closed_bar, held_id.shift(1))

        log_net = np.log1p(frame["net_return"]).groupby(acct_id).sum()
        log_gross = np.log1p(frame["gross_return"]).groupby(acct_id).sum()
        frictions = (frame["cost"] + frame["borrow"]).groupby(acct_id).sum()
        bars = (pos != 0).groupby(acct_id).sum()

        opens = pd.DataFrame(
            {
                "entry_signal_date": frame.index[entry.to_numpy()],
                "side": np.where(sig[entry] > 0, "long_spread", "short_spread"),
                "entry_z": frame.loc[entry, "zscore"].to_numpy(),
                "hedge_ratio": frame.loc[entry, "beta"].to_numpy(),
            },
            index=trade_id[entry].astype(int).to_numpy(),
        )
        closes = pd.DataFrame(
            {
                "exit_signal_date": frame.index[exit_.to_numpy()],
                "exit_z": frame.loc[exit_, "zscore"].to_numpy(),
                "exit_reason": reasons[exit_].to_numpy(),
            },
            index=trade_id.shift(1)[exit_].astype(int).to_numpy(),
        )
        perf = pd.DataFrame(
            {
                "holding_days": bars.astype(int),
                "gross_return": np.expm1(log_gross),
                "net_return": np.expm1(log_net),
                "frictions": frictions,
            }
        )
        perf.index = perf.index.astype(int)
        trades = opens.join(closes).join(perf, how="inner")
        trades["exit_reason"] = trades["exit_reason"].fillna("open")
        trades = trades[trades["holding_days"] > 0]
        trades.index.name = "trade_id"
        return trades.reset_index()


# --------------------------------------------------------------------------- #
# Performance statistics
# --------------------------------------------------------------------------- #
def _performance_metrics(
    frame: pd.DataFrame, trades: pd.DataFrame, cfg: PairsBacktestConfig
) -> dict[str, float]:
    """All summary statistics from per-bar net returns and the trade log."""
    nan = float("nan")
    net = frame["net_return"]
    td = cfg.trading_days
    n = len(net)
    years = n / td
    equity = (1.0 + net).cumprod()
    final = float(equity.iloc[-1])

    std = float(net.std(ddof=1))
    excess = net - cfg.risk_free_rate / td
    sharpe = float(excess.mean() / std * np.sqrt(td)) if std > _EPS else nan
    downside = float(np.sqrt((np.minimum(excess, 0.0) ** 2).mean()))
    sortino = float(excess.mean() / downside * np.sqrt(td)) if downside > _EPS else nan

    dd = equity / equity.cummax() - 1.0
    under = (dd < 0).astype(int)
    streak = under.groupby((under == 0).cumsum()).cumsum()
    max_dd = float(dd.min())
    cagr = final ** (1.0 / years) - 1.0 if final > 0 else -1.0

    m: dict[str, float] = {
        "total_return": final - 1.0,
        "cagr": cagr,
        "ann_volatility": std * float(np.sqrt(td)),
        "sharpe": sharpe,
        "sharpe_stderr": float(np.sqrt((1.0 + 0.5 * sharpe**2) / years)) if np.isfinite(sharpe) else nan,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_days": float(streak.max()),
        "calmar": cagr / abs(max_dd) if max_dd < -_EPS else nan,
        "skew": float(net.skew()) if std > _EPS else nan,
        "excess_kurtosis": float(net.kurt()) if std > _EPS else nan,
        "exposure": float((frame["position"] != 0).mean()),
        "annual_turnover": float(frame["turnover"].sum() / years),
        "total_costs": float((frame["cost"] + frame["borrow"]).sum()),
        "n_trades": float(len(trades)),
    }
    if len(trades):
        r = trades["net_return"]
        gains, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
        m.update(
            win_rate=float((r > 0).mean()),
            avg_trade_return=float(r.mean()),
            profit_factor=gains / losses if losses > 0 else (float("inf") if gains > 0 else nan),
            avg_holding_days=float(trades["holding_days"].mean()),
        )
    else:
        m.update(win_rate=nan, avg_trade_return=nan, profit_factor=nan, avg_holding_days=nan)
    return m
