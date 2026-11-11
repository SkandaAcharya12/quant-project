"""
optimizer.py -- Modern Portfolio Theory Optimizer (Phase 4)
===========================================================

Contents: shrinkage estimators, closed-form solutions (used as ground truth),
numerical constrained optimization (scipy SLSQP with analytic gradients), a
10,000-portfolio Monte Carlo, the efficient frontier, and a walk-forward
(out-of-sample) evaluation against the 1/N benchmark.

Notation: N assets, mu = expected-return vector (annual), S = covariance matrix
(annual, symmetric positive definite), 1 = ones vector, rf = risk-free rate,
w = weights with 1'w = 1.  Portfolio return m = w'mu, variance v = w'Sw.

PROOF 1 -- Global minimum variance (GMV)
    min  (1/2) w'Sw   s.t.  1'w = 1.
    Lagrangian  L = (1/2) w'Sw - l (1'w - 1).     dL/dw = Sw - l 1 = 0  =>  w = l S^-1 1.
    Constraint 1'w = 1  =>  l = 1 / (1'S^-1 1).   Hence  w_gmv = S^-1 1 / (1'S^-1 1)
    and  v_min = w'Sw = l = 1 / (1'S^-1 1).
    The Hessian S is positive definite, so the objective is strictly convex and this
    stationary point is the unique global minimum.  Q.E.D.

PROOF 2 -- Efficient frontier (shorting allowed)
    min (1/2) w'Sw  s.t.  w'mu = m,  1'w = 1.
    Stationarity:  Sw = l1 mu + l2 1  =>  w = S^-1 (l1 mu + l2 1).
    Define  a = mu'S^-1 mu,  b = mu'S^-1 1,  c = 1'S^-1 1  (inner products under the
    positive-definite form <u,v> = u'S^-1 v).  The constraints become
        [a b; b c] [l1; l2] = [m; 1]   =>   [l1; l2] = A^-1 [m; 1].
    Then  v(m) = w'Sw = [m 1] A^-1 [m 1]' = (c m^2 - 2 b m + a) / D,   D = ac - b^2.
    D > 0 by Cauchy-Schwarz (b^2 <= ac, equality only if mu is proportional to 1).
    So variance is a convex parabola in m with minimum at m = b/c (the GMV return)
    where v = 1/c, matching Proof 1.  In (sigma, m) space this is a hyperbola.  Q.E.D.

PROOF 3 -- Maximum Sharpe (tangency) portfolio
    max  S(w) = (w'mu - rf) / sqrt(w'Sw)   s.t. 1'w = 1.
    Gradient:  dS/dw = mu/s - (m - rf) Sw / s^3,   s = sqrt(w'Sw).
    Setting it to zero:  Sw = [s^2 / (m - rf)] (mu - rf 1)  =>  w is proportional to
    S^-1 (mu - rf 1).  Normalizing to 1'w = 1:
        w_tan = S^-1 (mu - rf 1) / (1'S^-1 (mu - rf 1)),   requires b - rf c > 0.
    The maximal Sharpe squared is (mu - rf1)'S^-1 (mu - rf1).  The numerical SLSQP
    solution is checked against this closed form in the tests.  Q.E.D.

Estimation error (why raw MPT disappoints out of sample)
    The optimizer is an "error maximizer" (Michaud): it overweights assets whose
    estimated mean is too high or variance too low.  Means are the worst offender:
    the standard error of an annual mean is about sigma / sqrt(years), so with five
    years of data a 20%-vol asset has a mean known to roughly +/- 9 percentage points,
    larger than most true differences between assets.  Two defenses are built in:
    * Ledoit-Wolf covariance shrinkage (below).
    * Bayes-Stein / James-Stein shrinkage of the means toward the GMV mean (Jorion 1986).
    The walk-forward routine measures honestly how much (or little) the optimizer
    beats the naive 1/N rule (DeMiguel, Garlappi & Uppal, 2009).

Ledoit-Wolf shrinkage (identity target, Ledoit & Wolf 2004)
    X = demeaned T x N returns,  S = X'X / T,  m = tr(S)/N,  F = m I.
    d2 = ||S - F||_F^2 / N                      (distance of S from the target)
    b2bar = (1 / T^2) sum_t ||x_t x_t' - S||_F^2 / N    (sampling noise of S)
    delta = min(b2bar, d2) / d2,   Sigma* = delta F + (1 - delta) S.
    Using  sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t||^4 - T ||S||_F^2  (expand the
    square; sum_t x_t' S x_t = T ||S||_F^2) it vectorizes with no loop over t.
    Shrinking toward a well-conditioned target trades a little bias for a large
    variance reduction, and keeps S invertible even when T < N.

Bayes-Stein means (Jorion 1986), in per-period units
    m0 = 1'S^-1 mu_hat / 1'S^-1 1,   d = mu_hat - m0 1,
    w = (N + 2) / ((N + 2) + T d'S^-1 d),    mu_bs = (1 - w) mu_hat + w m0 1.
    Estimates far from the common mean (relative to their sampling noise) are pulled in
    harder.  w -> 0 as T grows.  (Computed on daily moments, because T counts daily
    observations; the result is annualized afterward.)

Monte Carlo
    Weights ~ Dirichlet(1,...,1), which is exactly uniform on the simplex, mapped affinely
    to the lower bound and rejected if above the cap.  Random search is a *visualization*
    of the feasible set, not an optimizer: with many assets random portfolios cluster
    near 1/N and never approach the frontier.  The gap between the best random Sharpe and
    the exact SLSQP optimum is reported to make that point.

Limitations to state in any write-up
    Returns are assumed i.i.d.; annualization by 252 and sqrt(252) rests on that.
    Moments are estimated from history and assumed stable.  Transaction costs are
    modeled only in ``walk_forward``.  Mean-variance ignores skew, kurtosis and tail risk
    (the Risk Engine in Phase 5 handles those).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

__all__ = [
    "OptimizationError",
    "OptimizationResult",
    "OptimizerConfig",
    "Portfolio",
    "PortfolioOptimizer",
    "WalkForwardResult",
    "analytic_frontier_variance",
    "analytic_gmv",
    "analytic_tangency",
    "bayes_stein_means",
    "ledoit_wolf_covariance",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
_TOL = 1e-8


class OptimizationError(Exception):
    """The optimization problem is infeasible, ill-posed, or the solver failed."""


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def ledoit_wolf_covariance(returns: FloatArray) -> tuple[FloatArray, float]:
    """Ledoit-Wolf shrunk covariance (identity target). Returns (covariance, delta).

    ``returns`` is a T x N matrix of per-period returns. See the module docstring for the
    derivation; the result matches scikit-learn's ``ledoit_wolf`` (verified in the tests).
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise OptimizationError("returns must be a 2-D array with at least 2 rows")
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu_target = np.trace(s) / n
    d2 = float(np.sum((s - mu_target * np.eye(n)) ** 2) / n)
    if d2 <= 1e-30:  # S already equals the target
        return s, 1.0
    norms = np.einsum("ti,ti->t", xc, xc)
    b2_bar = float((np.sum(norms**2) - t * np.sum(s**2)) / (t**2 * n))
    delta = min(max(b2_bar, 0.0), d2) / d2
    return delta * mu_target * np.eye(n) + (1.0 - delta) * s, float(delta)


def bayes_stein_means(mean: FloatArray, cov: FloatArray, n_obs: int) -> tuple[FloatArray, float]:
    """Bayes-Stein shrinkage of expected returns toward the GMV mean. Returns (means, w).

    ``mean`` and ``cov`` must be in the same per-period units as ``n_obs`` observations.
    """
    mu = np.asarray(mean, dtype=float)
    n = len(mu)
    ones = np.ones(n)
    sinv_mu = _solve(cov, mu)
    sinv_1 = _solve(cov, ones)
    m0 = float(ones @ sinv_mu / (ones @ sinv_1))
    d = mu - m0
    quad = float(d @ _solve(cov, d))
    w = (n + 2.0) / ((n + 2.0) + n_obs * quad)
    return (1.0 - w) * mu + w * m0, float(w)


def _solve(a: FloatArray, b: FloatArray) -> FloatArray:
    """Solve a x = b, falling back to the pseudo-inverse for a singular matrix."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


# --------------------------------------------------------------------------- #
# Closed-form solutions (unconstrained: shorting allowed, no caps)
# --------------------------------------------------------------------------- #
def analytic_gmv(cov: FloatArray) -> FloatArray:
    """w = S^-1 1 / (1'S^-1 1)   (Proof 1)."""
    x = _solve(cov, np.ones(len(cov)))
    return x / x.sum()


def analytic_tangency(mu: FloatArray, cov: FloatArray, rf: float) -> FloatArray:
    """w = S^-1 (mu - rf 1) / (1'S^-1 (mu - rf 1))   (Proof 3)."""
    x = _solve(cov, np.asarray(mu) - rf)
    total = x.sum()
    if total <= 0:
        raise OptimizationError("Tangency portfolio undefined: need rf below the GMV return")
    return x / total


def analytic_frontier_variance(mu: FloatArray, cov: FloatArray, m: FloatArray) -> FloatArray:
    """Minimum variance for each target return m: (c m^2 - 2 b m + a) / D   (Proof 2)."""
    ones = np.ones(len(mu))
    s_mu, s_1 = _solve(cov, mu), _solve(cov, ones)
    a, b, c = float(mu @ s_mu), float(mu @ s_1), float(ones @ s_1)
    d = a * c - b * b
    m_arr = np.asarray(m, dtype=float)
    return (c * m_arr**2 - 2.0 * b * m_arr + a) / d


# --------------------------------------------------------------------------- #
# Configuration and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptimizerConfig:
    """Immutable optimizer settings.

    ``min_weight`` / ``max_weight`` bound every weight (use -inf / inf for the unconstrained
    textbook problem; ``min_weight=0`` is long-only). ``covariance_method`` and
    ``mean_estimator`` choose the estimators (shrinkage versions recommended).
    """

    trading_days: int = 252
    risk_free_rate: float = 0.0  # annual
    n_simulations: int = 10_000
    frontier_points: int = 50
    min_weight: float = 0.0
    max_weight: float = 1.0
    covariance_method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf"
    mean_estimator: Literal["historical", "bayes_stein"] = "historical"
    n_starts: int = 6
    seed: int = 42

    def __post_init__(self) -> None:
        if self.trading_days < 1 or self.n_simulations < 1 or self.n_starts < 1:
            raise OptimizationError("trading_days, n_simulations, n_starts must be >= 1")
        if self.frontier_points < 3:
            raise OptimizationError("frontier_points must be >= 3")
        if not self.min_weight < self.max_weight:
            raise OptimizationError("min_weight must be below max_weight")
        if self.covariance_method not in ("ledoit_wolf", "sample"):
            raise OptimizationError("covariance_method must be 'ledoit_wolf' or 'sample'")
        if self.mean_estimator not in ("historical", "bayes_stein"):
            raise OptimizationError("mean_estimator must be 'historical' or 'bayes_stein'")


@dataclass(frozen=True)
class Portfolio:
    """A weight vector with its annualized statistics."""

    label: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    note: str = ""


@dataclass(frozen=True)
class OptimizationResult:
    """Output of one optimization run (all moments annualized)."""

    assets: tuple[str, ...]
    mu: pd.Series
    cov: pd.DataFrame
    shrinkage: float  # Ledoit-Wolf delta (0 when method="sample")
    mean_shrinkage: float  # Bayes-Stein weight (0 when estimator="historical")
    gmv: Portfolio
    max_sharpe: Portfolio
    equal_weight: Portfolio
    frontier: pd.DataFrame
    monte_carlo: pd.DataFrame  # columns: return, volatility, sharpe
    mc_weights: FloatArray
    config: OptimizerConfig

    def best_random_portfolio(self) -> Portfolio:
        """Highest-Sharpe portfolio among the Monte Carlo draws."""
        i = int(self.monte_carlo["sharpe"].to_numpy().argmax())
        row = self.monte_carlo.iloc[i]
        w = pd.Series(self.mc_weights[i], index=list(self.assets))
        return Portfolio("Best Monte Carlo", w, float(row["return"]), float(row["volatility"]),
                         float(row["sharpe"]))

    def summary(self) -> str:
        """Weights and statistics side by side."""
        best_mc = self.best_random_portfolio()
        cols = [self.max_sharpe, self.gmv, self.equal_weight, best_mc]
        table = pd.DataFrame({p.label: p.weights for p in cols})
        stats = pd.DataFrame(
            {p.label: {"Expected return": p.expected_return, "Volatility": p.volatility,
                       "Sharpe": p.sharpe} for p in cols}
        )
        gap = self.max_sharpe.sharpe - best_mc.sharpe
        lines = [
            "Weights:", table.map("{:.1%}".format).to_string(),
            "", "Annualized statistics:",
            stats.T.to_string(float_format=lambda v: f"{v:.4f}"),
            "", f"Covariance shrinkage delta = {self.shrinkage:.3f};"
            f" mean shrinkage weight = {self.mean_shrinkage:.3f}",
            f"Exact optimum beats best of {len(self.monte_carlo):,} random portfolios by"
            f" {gap:.4f} Sharpe",
        ]
        if self.max_sharpe.note:
            lines.append(f"Note: {self.max_sharpe.note}")
        return "\n".join(lines)

    def figure(self) -> Any:
        """Plotly figure: Monte Carlo cloud, frontier, capital market line, assets."""
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise OptimizationError("plotly is required for figure() (pip install plotly)") from exc
        fig = go.Figure()
        mc = self.monte_carlo
        fig.add_trace(go.Scattergl(
            x=mc["volatility"], y=mc["return"], mode="markers", name="Random portfolios",
            marker={"size": 4, "color": mc["sharpe"], "colorscale": "Viridis", "opacity": 0.5,
                    "colorbar": {"title": "Sharpe"}}))
        fr = self.frontier
        fig.add_trace(go.Scatter(x=fr["volatility"], y=fr["return"], mode="lines",
                                 name="Efficient frontier", line={"width": 3, "color": "crimson"}))
        sigma = np.array([0.0, float(max(mc["volatility"].max(), fr["volatility"].max()))])
        rf = self.config.risk_free_rate
        fig.add_trace(go.Scatter(x=sigma, y=rf + self.max_sharpe.sharpe * sigma, mode="lines",
                                 name="Capital market line", line={"dash": "dash", "color": "gray"}))
        vols = np.sqrt(np.diag(self.cov.to_numpy()))
        fig.add_trace(go.Scatter(x=vols, y=self.mu.to_numpy(), mode="markers+text",
                                 text=list(self.assets), textposition="top center",
                                 name="Assets", marker={"size": 9, "color": "black"}))
        for p, symbol, color in ((self.max_sharpe, "star", "gold"), (self.gmv, "diamond", "blue"),
                                 (self.equal_weight, "square", "orange")):
            fig.add_trace(go.Scatter(x=[p.volatility], y=[p.expected_return], mode="markers",
                                     name=p.label, marker={"size": 16, "symbol": symbol,
                                                           "color": color, "line": {"width": 1}}))
        fig.update_layout(title="Efficient frontier and random portfolios",
                          xaxis_title="Annualized volatility", yaxis_title="Annualized return",
                          template="plotly_white")
        return fig


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample portfolio returns from rolling re-optimization."""

    returns: pd.DataFrame  # daily net returns, one column per strategy
    weights: dict[str, pd.DataFrame]  # target weights at each rebalance date
    turnover: pd.DataFrame  # per-rebalance turnover (sum of |trades| / capital)
    config: OptimizerConfig

    def metrics(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Annualized statistics per strategy over [start, end]."""
        sub = self.returns.loc[start:end]
        if len(sub) < 2:
            raise OptimizationError("Selected period contains fewer than 2 bars")
        td, rf = self.config.trading_days, self.config.risk_free_rate
        rows = {}
        for name in sub.columns:
            r = sub[name]
            equity = (1.0 + r).cumprod()
            std = float(r.std(ddof=1))
            to = self.turnover[name].loc[start:end]
            rows[name] = {
                "total_return": float(equity.iloc[-1] - 1.0),
                "ann_return": float(equity.iloc[-1] ** (td / len(r)) - 1.0),
                "ann_volatility": std * float(np.sqrt(td)),
                "sharpe": float((r.mean() - rf / td) / std * np.sqrt(td)) if std > 0 else np.nan,
                "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
                "avg_turnover": float(to.mean()) if len(to) else np.nan,
            }
        return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
class PortfolioOptimizer:
    """Mean-variance optimization with shrinkage estimators and honest evaluation.

    Example:
        >>> opt = PortfolioOptimizer(OptimizerConfig(risk_free_rate=0.04))
        >>> result = opt.optimize(prices.loc[:"2024-12-31"])      # estimate in-sample
        >>> print(result.summary())
        >>> opt.evaluate(result.max_sharpe.weights, oos_returns)  # judge out-of-sample
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.cfg = config or OptimizerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Simple returns R = P_t / P_{t-1} - 1 (weights act linearly on simple returns)."""
        if (prices <= 0).any().any() or prices.isna().any().any():
            raise OptimizationError("prices must be positive and gap-free (use DataEngine output)")
        return (prices / prices.shift(1) - 1.0).iloc[1:]

    def optimize(self, prices: pd.DataFrame) -> OptimizationResult:
        """Estimate moments from a price panel and solve every portfolio problem."""
        return self.optimize_returns(self.simple_returns(prices))

    def optimize_returns(self, returns: pd.DataFrame) -> OptimizationResult:
        """Same as ``optimize`` but starting from a DataFrame of simple returns."""
        cfg = self.cfg
        x = self._validate_returns(returns)
        n = x.shape[1]
        mu, cov, delta, w_bs = self._moments(x)
        self._check_bounds(n)
        rng = np.random.default_rng(cfg.seed)
        names = list(returns.columns)

        gmv_w = self._min_variance(cov)
        m_max = self._max_return(mu)
        sharpe_w, note = self._max_sharpe(mu, cov, gmv_w, m_max, rng)
        equal_w = np.full(n, 1.0 / n)
        frontier = self._frontier(mu, cov, gmv_w, m_max, names)
        mc, mc_w = self._monte_carlo(mu, cov, rng)

        def pack(label: str, w: FloatArray, comment: str = "") -> Portfolio:
            ret, vol, sr = self._stats(w, mu, cov)
            return Portfolio(label, pd.Series(w, index=names), ret, vol, sr, comment)

        return OptimizationResult(
            assets=tuple(names),
            mu=pd.Series(mu, index=names),
            cov=pd.DataFrame(cov, index=names, columns=names),
            shrinkage=delta,
            mean_shrinkage=w_bs,
            gmv=pack("Min variance", gmv_w),
            max_sharpe=pack("Max Sharpe", sharpe_w, note),
            equal_weight=pack("Equal weight (1/N)", equal_w),
            frontier=frontier,
            monte_carlo=mc,
            mc_weights=mc_w,
            config=cfg,
        )

    def evaluate(self, weights: pd.Series, returns: pd.DataFrame) -> dict[str, float]:
        """Annualized statistics of fixed (daily-rebalanced) weights on a returns sample.

        Use it on data the weights were NOT estimated from to see out-of-sample performance.
        """
        w = weights.reindex(returns.columns)
        if w.isna().any():
            raise OptimizationError("weights must cover every column of returns")
        r = returns.to_numpy() @ w.to_numpy()
        td, rf = self.cfg.trading_days, self.cfg.risk_free_rate
        std = float(r.std(ddof=1))
        equity = np.cumprod(1.0 + r)
        return {
            "ann_return": float(equity[-1] ** (td / len(r)) - 1.0),
            "ann_volatility": std * float(np.sqrt(td)),
            "sharpe": float((r.mean() - rf / td) / std * np.sqrt(td)) if std > 0 else float("nan"),
            "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1.0).min()),
            "n_obs": float(len(r)),
        }

    def walk_forward(
        self,
        prices: pd.DataFrame,
        lookback: int = 252,
        rebalance_every: int = 21,
        cost_bps: float = 5.0,
        strategies: tuple[str, ...] = ("max_sharpe", "min_variance", "equal_weight"),
    ) -> WalkForwardResult:
        """Rolling out-of-sample test: estimate on the trailing window, hold, repeat.

        At each rebalance the weights use only the ``lookback`` returns that precede the
        holding period (no look-ahead). Positions then drift (buy and hold) until the next
        rebalance; rebalancing costs ``cost_bps`` per unit of notional traded.
        Re-optimizations are sequential solves, so the loop over rebalance dates is inherent.
        """
        cfg = self.cfg
        unknown = set(strategies) - {"max_sharpe", "min_variance", "equal_weight"}
        if unknown or not strategies:
            raise OptimizationError(f"Unknown or empty strategies: {sorted(unknown)}")
        if cfg.min_weight < 0:
            raise OptimizationError("walk_forward supports long-only (min_weight >= 0) only")
        rets = self.simple_returns(prices)
        values = rets.to_numpy()
        t_total, n = values.shape
        if lookback < max(30, n + 2) or rebalance_every < 1 or t_total <= lookback + 1:
            raise OptimizationError("lookback too short, or not enough history after it")
        self._check_bounds(n)
        rng = np.random.default_rng(cfg.seed)
        names = list(rets.columns)

        out = {s: np.zeros(t_total - lookback) for s in strategies}
        held = {s: np.zeros(n) for s in strategies}  # drifted weights (zero = all cash)
        weight_rows: dict[str, list[FloatArray]] = {s: [] for s in strategies}
        turnover_rows: dict[str, list[float]] = {s: [] for s in strategies}
        dates: list[pd.Timestamp] = []

        for i0 in range(lookback, t_total, rebalance_every):
            i1 = min(i0 + rebalance_every, t_total)
            mu, cov, _, _ = self._moments(values[i0 - lookback : i0])
            gmv_w = self._min_variance(cov)
            targets: dict[str, FloatArray] = {"equal_weight": np.full(n, 1.0 / n)}
            if "min_variance" in strategies:
                targets["min_variance"] = gmv_w
            if "max_sharpe" in strategies:
                targets["max_sharpe"], _ = self._max_sharpe(mu, cov, gmv_w, self._max_return(mu), rng)
            growth = np.cumprod(1.0 + values[i0:i1], axis=0)  # per-asset wealth path
            dates.append(rets.index[i0])
            for s in strategies:
                w = targets[s]
                turnover = float(np.abs(w - held[s]).sum())
                wealth = growth @ w
                prev = np.concatenate(([1.0], wealth[:-1]))
                r = wealth / prev - 1.0
                r[0] -= turnover * cost_bps / 1e4
                out[s][i0 - lookback : i1 - lookback] = r
                held[s] = w * growth[-1] / wealth[-1]
                weight_rows[s].append(w)
                turnover_rows[s].append(turnover)

        idx = rets.index[lookback:]
        reb = pd.DatetimeIndex(dates)
        return WalkForwardResult(
            returns=pd.DataFrame(out, index=idx),
            weights={s: pd.DataFrame(weight_rows[s], index=reb, columns=names) for s in strategies},
            turnover=pd.DataFrame(turnover_rows, index=reb),
            config=cfg,
        )

    # ------------------------------------------------------------------ #
    # Moments
    # ------------------------------------------------------------------ #
    def _validate_returns(self, returns: pd.DataFrame) -> FloatArray:
        if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
            raise OptimizationError("Need a DataFrame with at least two assets")
        x = returns.to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise OptimizationError("returns contain NaN or infinite values")
        t, n = x.shape
        if t < max(30, n + 2):
            raise OptimizationError(f"Need at least max(30, N+2) observations; got {t} for N={n}")
        if t < 10 * n:
            logger.warning("Only %d observations for %d assets (<10x): estimates will be noisy", t, n)
        return x

    def _moments(self, x: FloatArray) -> tuple[FloatArray, FloatArray, float, float]:
        """Annualized (mu, cov) plus the covariance and mean shrinkage intensities."""
        cfg = self.cfg
        t = x.shape[0]
        mean_d = x.mean(axis=0)
        if cfg.covariance_method == "ledoit_wolf":
            cov_d, delta = ledoit_wolf_covariance(x)
        else:
            cov_d, delta = np.cov(x, rowvar=False, ddof=1), 0.0
        cov_d = (cov_d + cov_d.T) / 2.0
        w_bs = 0.0
        if cfg.mean_estimator == "bayes_stein":
            mean_d, w_bs = bayes_stein_means(mean_d, cov_d, t)
        return mean_d * cfg.trading_days, cov_d * cfg.trading_days, delta, w_bs

    # ------------------------------------------------------------------ #
    # Constraints and statistics
    # ------------------------------------------------------------------ #
    def _check_bounds(self, n: int) -> None:
        lo, hi = self.cfg.min_weight, self.cfg.max_weight
        if n * lo > 1.0 + _TOL or n * hi < 1.0 - _TOL:
            raise OptimizationError(
                f"Infeasible bounds: with {n} assets, need {n}*min <= 1 <= {n}*max "
                f"(min={lo}, max={hi})"
            )

    def _bounds(self, n: int) -> list[tuple[float | None, float | None]] | None:
        lo, hi = self.cfg.min_weight, self.cfg.max_weight
        if np.isinf(lo) and np.isinf(hi):
            return None
        lo_b = None if np.isinf(lo) else lo
        hi_b = None if np.isinf(hi) else hi
        return [(lo_b, hi_b)] * n

    def _stats(self, w: FloatArray, mu: FloatArray, cov: FloatArray) -> tuple[float, float, float]:
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        sharpe = (ret - self.cfg.risk_free_rate) / vol if vol > 0 else float("nan")
        return ret, vol, sharpe

    def _feasible(self, w: FloatArray) -> bool:
        lo, hi = self.cfg.min_weight, self.cfg.max_weight
        return bool(abs(w.sum() - 1.0) < 1e-6 and (w >= lo - 1e-7).all() and (w <= hi + 1e-7).all())

    # ------------------------------------------------------------------ #
    # Solvers
    # ------------------------------------------------------------------ #
    def _slsqp(
        self, fun: Any, jac: Any, x0: FloatArray, extra_eq: list[dict[str, Any]] | None = None
    ) -> FloatArray | None:
        n = len(x0)
        cons: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}
        ]
        cons += extra_eq or []
        res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=self._bounds(n), constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-13})
        if res.success and self._feasible(res.x):
            return np.asarray(res.x, dtype=float)
        return None

    def _start_point(self, n: int) -> FloatArray:
        return np.full(n, 1.0 / n)

    def _min_variance(
        self, cov: FloatArray, mu: FloatArray | None = None, target: float | None = None,
        x0: FloatArray | None = None,
    ) -> FloatArray:
        """Minimize w'Sw s.t. 1'w=1 (and w'mu = target if given), with bounds."""
        n = len(cov)
        extra = None
        if target is not None and mu is not None:
            extra = [{"type": "eq", "fun": lambda w: w @ mu - target, "jac": lambda w: mu}]
        start = self._start_point(n) if x0 is None else x0
        w = self._slsqp(lambda w: w @ cov @ w, lambda w: 2.0 * cov @ w, start, extra)
        if w is None and x0 is not None:  # retry from the neutral start
            w = self._slsqp(lambda w: w @ cov @ w, lambda w: 2.0 * cov @ w, self._start_point(n), extra)
        if w is None:
            raise OptimizationError("Minimum-variance solver failed to converge")
        return w

    def _max_return(self, mu: FloatArray) -> float:
        """Largest achievable return under the bounds (a linear program)."""
        bounds = self._bounds(len(mu))
        if bounds is None:
            return float(mu.max())
        res = linprog(-mu, A_eq=np.ones((1, len(mu))), b_eq=[1.0], bounds=bounds, method="highs")
        if not res.success:
            raise OptimizationError("Return-maximization LP failed (infeasible bounds?)")
        return float(-res.fun)

    def _max_sharpe(
        self, mu: FloatArray, cov: FloatArray, gmv_w: FloatArray, m_max: float,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, str]:
        """Maximize the Sharpe ratio with SLSQP from several starts (Proof 3 gradient)."""
        rf = self.cfg.risk_free_rate
        n = len(mu)
        if m_max <= rf:
            return gmv_w, "No portfolio beats the risk-free rate; returned min-variance instead"
        excess = mu - rf

        def neg_sharpe(w: FloatArray) -> float:
            return -float(w @ excess) / float(np.sqrt(max(w @ cov @ w, 1e-18)))

        def neg_sharpe_grad(w: FloatArray) -> FloatArray:
            s = float(np.sqrt(max(w @ cov @ w, 1e-18)))
            return -excess / s + float(w @ excess) * (cov @ w) / s**3

        starts = [self._start_point(n), gmv_w]
        starts += [row for row in self._sample_weights(self.cfg.n_starts, n, rng)]
        best: FloatArray | None = None
        best_val = np.inf
        for x0 in starts:
            w = self._slsqp(neg_sharpe, neg_sharpe_grad, x0)
            if w is not None and neg_sharpe(w) < best_val - 1e-12:
                best, best_val = w, neg_sharpe(w)
        if best is None:
            raise OptimizationError("Max-Sharpe solver failed from every starting point")
        return best, ""

    def _frontier(
        self, mu: FloatArray, cov: FloatArray, gmv_w: FloatArray, m_max: float, names: list[str]
    ) -> pd.DataFrame:
        """Efficient frontier: min variance for each target return from GMV up to the maximum."""
        m_lo = float(gmv_w @ mu)
        targets = np.linspace(m_lo, max(m_max, m_lo), self.cfg.frontier_points)
        bounded = self._bounds(len(mu)) is not None
        rows: list[tuple[float, FloatArray]] = [(m_lo, gmv_w)]
        warm = gmv_w
        # Sequential: each solve warm-starts from the previous point on the curve.
        for m in (targets[1:-1] if bounded else targets[1:]):
            try:
                warm = self._min_variance(cov, mu, float(m), warm)
            except OptimizationError:
                logger.warning("Frontier point at return %.4f failed; skipped", m)
                continue
            rows.append((float(m), warm))
        if bounded:
            # Endpoint = the max-return corner from the LP (a degenerate set for SLSQP).
            corner = self._max_return_weights(mu)
            rows.append((float(corner @ mu), corner))
        records = []
        for target, w in rows:
            ret, vol, sr = self._stats(w, mu, cov)
            records.append({"target_return": target, "return": ret, "volatility": vol, "sharpe": sr,
                            **{f"w_{a}": wi for a, wi in zip(names, w)}})
        return pd.DataFrame(records)

    def _max_return_weights(self, mu: FloatArray) -> FloatArray:
        bounds = self._bounds(len(mu))
        if bounds is None:
            w = np.zeros(len(mu))
            w[int(mu.argmax())] = 1.0
            return w
        res = linprog(-mu, A_eq=np.ones((1, len(mu))), b_eq=[1.0], bounds=bounds, method="highs")
        if not res.success:
            raise OptimizationError("Return-maximization LP failed")
        return np.asarray(res.x, dtype=float)

    # ------------------------------------------------------------------ #
    # Monte Carlo
    # ------------------------------------------------------------------ #
    def _sample_weights(self, count: int, n: int, rng: np.random.Generator) -> FloatArray:
        """Feasible random weights: Dirichlet(1) on the simplex, mapped to [min, .] and capped.

        w = m + (1 - N m) d with d ~ Dirichlet(1,...,1) gives uniform samples on
        {w >= m, 1'w = 1}; draws above ``max_weight`` are rejected.
        """
        lo = self.cfg.min_weight if np.isfinite(self.cfg.min_weight) else -1.0
        hi = self.cfg.max_weight
        accepted: list[FloatArray] = []
        have = 0
        for _ in range(500):
            d = rng.dirichlet(np.ones(n), size=max(count * 2, 256))
            w = lo + (1.0 - n * lo) * d
            if np.isfinite(hi):
                w = w[(w <= hi).all(axis=1)]
            if len(w):
                accepted.append(w)
                have += len(w)
            if have >= count:
                return np.vstack(accepted)[:count]
        raise OptimizationError("Weight caps too tight for random sampling; loosen max_weight")

    def _monte_carlo(
        self, mu: FloatArray, cov: FloatArray, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, FloatArray]:
        """Evaluate ``n_simulations`` random portfolios in a few matrix operations."""
        w = self._sample_weights(self.cfg.n_simulations, len(mu), rng)
        ret = w @ mu
        vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))  # row-wise w' S w
        sharpe = (ret - self.cfg.risk_free_rate) / vol
        return pd.DataFrame({"return": ret, "volatility": vol, "sharpe": sharpe}), w
