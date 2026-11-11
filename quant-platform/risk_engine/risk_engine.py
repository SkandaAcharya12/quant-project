"""
risk_engine.py -- Risk Engine (Phase 5)
=======================================

Value at Risk (VaR) and Expected Shortfall (CVaR) by four methods, drawdown analysis,
formal backtests of the VaR forecasts, risk decomposition, and macro stress testing.

Conventions: returns are simple per-period returns; VaR and ES are reported as POSITIVE
loss fractions of portfolio value (0.02 = lose 2%).  alpha is the confidence level
(0.99), so the tail probability is 1 - alpha.

1. Value at Risk and Expected Shortfall
---------------------------------------
    VaR_a = -Q_{1-a}(R)                    the loss exceeded with probability 1 - a
    ES_a  = -E[R | R <= -VaR_a]            the average loss *given* that VaR is breached
VaR says nothing about how bad the tail is beyond the threshold, and it is not
subadditive (diversifying can raise it).  ES fixes both, which is why Basel's
market-risk framework moved from 99% VaR to 97.5% ES.  Both are reported.

Historical:   empirical quantile / tail mean of the sample.  No distributional
              assumption, but it only knows about events in the sample and reacts
              slowly when volatility changes.
Parametric:   assume R ~ N(m, s^2).  With z = Phi^-1(a):
                  VaR = z s - m
              ES derivation:  E[-R | R <= -VaR] = -m + s * phi(z) / (1 - a), because
              the tail integral of x * phi(x) over (-inf, -z] equals -phi(z)
              (d/dx phi(x) = -x phi(x)), then divide by the tail probability.
              Normal tails are too thin for equities, so this understates extreme risk.
Cornish-Fisher: correct the normal quantile for skewness S and excess kurtosis K:
                  z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
              (a Taylor expansion of the true quantile around the normal one; no closed-form
              ES is offered).  Measured on Student-t data (error in sigma units, 2M draws):
              at 95% it is excellent (t6: 0.002 vs 0.059 for the normal), but at 99% with
              fat tails it OVERSHOOTS (t6: +0.40 vs -0.25 for the normal).  Trust it near
              95%-97.5%; for 99% prefer filtered historical simulation.
Filtered historical simulation (FHS): standardize each return by the EWMA volatility
              that was known when it occurred, z_t = r_t / sigma_{t|t-1}, take the empirical
              quantile of z, then rescale by TODAY's volatility forecast:
                  VaR = -sigma_{T+1|T} * Q_{1-a}(z).
              EWMA (RiskMetrics): s2_t = lam s2_{t-1} + (1 - lam) r_t^2, lam = 0.94.
              This keeps the empirical (fat-tailed) shape while adapting to volatility
              clustering.  Measured in the tests on simulated GARCH(1,1) returns (12 seeds,
              5,000 days): conditional coverage was rejected at 5% in 0% of seeds for FHS,
              25% for EWMA-normal, and 67-83% for the rolling-window historical and
              parametric methods.  No method anticipates an abrupt regime switch: on
              returns whose volatility jumps 4x overnight every method failed.

Multi-day horizons: historical uses overlapping compounded h-day returns; the other
methods scale by sqrt(h) assuming i.i.d. returns (Cornish-Fisher scales S by 1/sqrt(h)
and K by 1/h).  The i.i.d. assumption is a known approximation.

2. Backtesting VaR (Kupiec 1995, Christoffersen 1998)
-----------------------------------------------------
A forecast made before each day is compared with that day's return; a day with
R < -VaR is an "exception" and should occur with probability p = 1 - a.
Kupiec POF (unconditional coverage), x exceptions in T days, pi = x/T:
    LR_pof = -2 ln[ (1-p)^(T-x) p^x / ((1-pi)^(T-x) pi^x) ]  ~  chi2(1)
Christoffersen independence: exceptions should not cluster.  With transition counts n_ij
(from state i yesterday to state j today), pi01 = n01/(n00+n01), pi11 = n11/(n10+n11),
pi = (n01+n11)/n:
    LR_ind = -2 ln[ (1-pi)^(n00+n10) pi^(n01+n11) /
                    ((1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11) ]  ~  chi2(1)
Conditional coverage LR_cc = LR_pof + LR_ind ~ chi2(2).  Basel "traffic light" for 99%
VaR over 250 days: green 0-4 exceptions, yellow 5-9, red 10+.

3. Risk decomposition (Euler)
-----------------------------
Portfolio volatility s(w) = sqrt(w'Sw) is homogeneous of degree 1, so by Euler's theorem
    s = sum_i w_i * ds/dw_i = sum_i w_i (Sw)_i / s.
Each term is asset i's contribution to risk; the contributions sum exactly to s.

4. Stress testing
-----------------
Historical scenarios replay a past crisis window on today's weights.  Factor scenarios
regress asset log returns on macro factors,  r_j = a_j + sum_k B_kj F_k + e_j  (OLS,
B = (X'X)^-1 X'R), then push a hypothetical factor shock through B.  Limits worth
stating: betas are estimated in calm markets and applied to extreme shocks (linearity),
correlations rise in crises, and the scenarios are judgmental illustrations, not forecasts.
Collinear factors (e.g., equities and VIX) make betas unstable, so the default factor
set is small; the condition number of the factor matrix is reported.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq
from scipy.special import xlog1py, xlogy

__all__ = [
    "DEFAULT_HISTORICAL_SCENARIOS",
    "DEFAULT_MACRO_SCENARIOS",
    "DrawdownAnalysis",
    "FactorModel",
    "FactorScenario",
    "RiskConfig",
    "RiskEngine",
    "RiskError",
    "RiskReport",
    "christoffersen_independence",
    "cornish_fisher_quantile",
    "drawdown_analysis",
    "es_historical",
    "es_parametric",
    "ewma_variance",
    "fhs_var_es",
    "fit_factor_model",
    "kupiec_pof",
    "macro_factor_frame",
    "risk_contributions",
    "var_cornish_fisher",
    "var_historical",
    "var_parametric",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
METHODS: tuple[str, ...] = ("historical", "parametric", "ewma", "fhs")

DEFAULT_HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "Global financial crisis (Oct 2007 - Mar 2009)": ("2007-10-09", "2009-03-09"),
    "Lehman shock (Sep 12 - Nov 20, 2008)": ("2008-09-12", "2008-11-20"),
    "2018 Q4 sell-off (Sep 20 - Dec 24, 2018)": ("2018-09-20", "2018-12-24"),
    "COVID crash (Feb 19 - Mar 23, 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan 3 - Oct 12, 2022)": ("2022-01-03", "2022-10-12"),
}


class RiskError(Exception):
    """Invalid input or configuration for the risk engine."""


# --------------------------------------------------------------------------- #
# Point estimators
# --------------------------------------------------------------------------- #
def _clean(returns: pd.Series | FloatArray, minimum: int = 30) -> FloatArray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or len(x) < minimum:
        raise RiskError(f"Need a 1-D return series with at least {minimum} observations")
    if not np.isfinite(x).all():
        raise RiskError("returns contain NaN or infinite values")
    return x


def _check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1.0:
        raise RiskError("confidence level must lie in (0.5, 1)")


def var_historical(r: FloatArray, alpha: float) -> float:
    """VaR = -Q_{1-a} of the empirical distribution (linear interpolation)."""
    _check_alpha(alpha)
    return float(-np.quantile(r, 1.0 - alpha))


def es_historical(r: FloatArray, alpha: float) -> float:
    """ES = minus the mean of all returns at or below the (1-a) quantile."""
    _check_alpha(alpha)
    q = np.quantile(r, 1.0 - alpha)
    return float(-r[r <= q].mean())


def var_parametric(r: FloatArray, alpha: float, horizon: int = 1) -> float:
    """Normal VaR = z s sqrt(h) - m h, with z = Phi^-1(a)."""
    _check_alpha(alpha)
    m, s = float(r.mean()), float(r.std(ddof=1))
    return float(stats.norm.ppf(alpha) * s * np.sqrt(horizon) - m * horizon)


def es_parametric(r: FloatArray, alpha: float, horizon: int = 1) -> float:
    """Normal ES = s sqrt(h) phi(z) / (1 - a) - m h   (derivation in the module docstring)."""
    _check_alpha(alpha)
    m, s = float(r.mean()), float(r.std(ddof=1))
    z = stats.norm.ppf(alpha)
    return float(s * np.sqrt(horizon) * stats.norm.pdf(z) / (1.0 - alpha) - m * horizon)


def cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    """Cornish-Fisher expansion of a standard-normal quantile z."""
    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def var_cornish_fisher(r: FloatArray, alpha: float, horizon: int = 1) -> float:
    """VaR using the Cornish-Fisher adjusted quantile (skew and kurtosis of the sample)."""
    _check_alpha(alpha)
    m, s = float(r.mean()), float(r.std(ddof=1))
    skew = float(stats.skew(r, bias=False)) / np.sqrt(horizon)
    kurt = float(stats.kurtosis(r, bias=False)) / horizon
    z_cf = cornish_fisher_quantile(float(stats.norm.ppf(1.0 - alpha)), skew, kurt)
    return float(-(m * horizon + s * np.sqrt(horizon) * z_cf))


def ewma_variance(r: FloatArray, lam: float = 0.94) -> FloatArray:
    """RiskMetrics EWMA variance: s2_t = lam s2_{t-1} + (1-lam) r_t^2, seeded with r_0^2."""
    if not 0.0 < lam < 1.0:
        raise RiskError("ewma lambda must lie in (0, 1)")
    return pd.Series(np.asarray(r, dtype=float) ** 2).ewm(alpha=1.0 - lam, adjust=False).mean().to_numpy()


def fhs_var_es(r: FloatArray, alpha: float, lam: float = 0.94, horizon: int = 1) -> tuple[float, float]:
    """Filtered historical simulation VaR and ES (see module docstring)."""
    _check_alpha(alpha)
    s2 = ewma_variance(r, lam)
    burn = min(30, len(r) // 5)  # EWMA is seeded with r_0^2, so early values are unreliable
    z = r[1:] / np.sqrt(s2[:-1])  # return divided by the volatility known the day before
    z = z[max(burn - 1, 0):]
    sigma_next = float(np.sqrt(s2[-1])) * np.sqrt(horizon)
    q = np.quantile(z, 1.0 - alpha)
    return float(-sigma_next * q), float(-sigma_next * z[z <= q].mean())


# --------------------------------------------------------------------------- #
# Backtest statistics
# --------------------------------------------------------------------------- #
def kupiec_pof(n_obs: int, n_exceptions: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR statistic and chi2(1) p-value."""
    if n_obs < 1 or not 0 <= n_exceptions <= n_obs or not 0.0 < p < 1.0:
        raise RiskError("invalid Kupiec inputs")
    x, t = n_exceptions, n_obs
    pi = x / t
    ll0 = xlogy(t - x, 1.0 - p) + xlogy(x, p)
    ll1 = xlogy(t - x, 1.0 - pi) + xlogy(x, pi)  # xlogy handles 0*log(0) = 0
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_independence(exceptions: NDArray[np.bool_]) -> tuple[float, float]:
    """Christoffersen LR statistic for independence of exceptions, and its p-value."""
    e = np.asarray(exceptions, dtype=bool)
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum(~prev & ~cur))
    n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur))
    n11 = int(np.sum(prev & cur))
    total = n00 + n01 + n10 + n11
    if total == 0 or (n01 + n11) == 0:
        return 0.0, 1.0  # no exceptions: nothing to test
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll0 = xlog1py(n00 + n10, -pi) + xlogy(n01 + n11, pi)
    ll1 = (xlog1py(n00, -pi01) + xlogy(n01, pi01) + xlog1py(n10, -pi11) + xlogy(n11, pi11))
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DrawdownAnalysis:
    """Underwater curve plus the worst drawdown episodes."""

    series: pd.DataFrame  # columns: equity, peak, drawdown
    episodes: pd.DataFrame
    max_drawdown: float
    longest_underwater_bars: int
    current_drawdown: float
    ulcer_index: float  # sqrt(mean(drawdown^2)): penalizes depth AND duration


def drawdown_analysis(returns: pd.Series, top_n: int = 5) -> DrawdownAnalysis:
    """Equity curve, running peak, drawdown series and the ``top_n`` deepest episodes.

    drawdown_t = equity_t / max_{s<=t} equity_s - 1, starting from an initial equity of 1.
    """
    r = pd.Series(_clean(returns, 2), index=returns.index)
    equity = (1.0 + r).cumprod()
    peak = pd.Series(np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:], index=r.index)
    dd = equity / peak - 1.0
    under = dd < 0
    n = len(dd)
    group = (~under).cumsum()  # a new group id starts at every bar that sets a new peak
    positions = pd.Series(np.arange(n), index=dd.index)
    rows = []
    for _, idx in positions[under].groupby(group[under]):
        p0, p1 = int(idx.iloc[0]), int(idx.iloc[-1])
        seg = dd.iloc[p0 : p1 + 1]
        trough_pos = p0 + int(np.argmin(seg.to_numpy()))
        recovered = p1 + 1 < n
        rows.append(
            {
                "peak_date": dd.index[p0 - 1] if p0 > 0 else dd.index[0],
                "trough_date": dd.index[trough_pos],
                "recovery_date": dd.index[p1 + 1] if recovered else pd.NaT,
                "depth": float(seg.min()),
                "bars_to_trough": trough_pos - p0 + 1,
                "bars_underwater": p1 - p0 + 1,
                "recovered": recovered,
            }
        )
    cols = ["peak_date", "trough_date", "recovery_date", "depth", "bars_to_trough",
            "bars_underwater", "recovered"]
    episodes = pd.DataFrame(rows, columns=cols)
    if not episodes.empty:
        episodes = episodes.sort_values("depth").head(top_n).reset_index(drop=True)
    run = under.astype(int).groupby(group).cumsum()
    return DrawdownAnalysis(
        series=pd.DataFrame({"equity": equity, "peak": peak, "drawdown": dd}),
        episodes=episodes,
        max_drawdown=float(dd.min()),
        longest_underwater_bars=int(run.max()),
        current_drawdown=float(dd.iloc[-1]),
        ulcer_index=float(np.sqrt((dd**2).mean())),
    )


# --------------------------------------------------------------------------- #
# Risk decomposition
# --------------------------------------------------------------------------- #
def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility (contributions sum to the volatility)."""
    names = list(cov.columns)
    w = weights.reindex(names)
    if w.isna().any():
        raise RiskError("weights must cover every asset in the covariance matrix")
    wv, c = w.to_numpy(dtype=float), cov.to_numpy(dtype=float)
    sigma = float(np.sqrt(wv @ c @ wv))
    if sigma <= 0:
        raise RiskError("portfolio variance is zero")
    marginal = c @ wv / sigma  # ds/dw_i
    contribution = wv * marginal
    return pd.DataFrame(
        {"weight": wv, "marginal_risk": marginal, "contribution": contribution,
         "pct_of_risk": contribution / sigma},
        index=names,
    )


# --------------------------------------------------------------------------- #
# Factor model and stress scenarios
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorScenario:
    """A hypothetical macro shock. Price factors in simple % change (-0.30 = -30%);
    factors whose name ends in ``_bp`` (rates) in basis points."""

    name: str
    shocks: Mapping[str, float]
    description: str = ""


DEFAULT_MACRO_SCENARIOS: tuple[FactorScenario, ...] = (
    FactorScenario("Equity crash with flight to quality",
                   {"equity": -0.30, "rates_bp": -75.0, "usd": 0.05, "oil": -0.25},
                   "Risk-off: stocks -30%, 10Y yield -75bp, dollar +5%, oil -25%"),
    FactorScenario("Stagflation / rate shock",
                   {"equity": -0.15, "rates_bp": 150.0, "usd": 0.08, "oil": 0.40},
                   "Inflation surge: 10Y yield +150bp, oil +40%, dollar +8%, stocks -15%"),
    FactorScenario("Oil supply shock",
                   {"equity": -0.10, "rates_bp": 25.0, "usd": 0.03, "oil": 0.60},
                   "Oil +60% with modest equity and rate impact"),
    FactorScenario("Dollar surge",
                   {"equity": -0.12, "rates_bp": 40.0, "usd": 0.12, "oil": -0.15},
                   "Dollar +12% tightening global financial conditions"),
    FactorScenario("Deflationary bust",
                   {"equity": -0.25, "rates_bp": -150.0, "usd": 0.04, "oil": -0.45},
                   "Demand collapse: yields -150bp, oil -45%, stocks -25%"),
)


def macro_factor_frame(macro_features: pd.DataFrame) -> pd.DataFrame:
    """Build the default factor panel from DataEngine macro features.

    Output columns: equity (S&P log return), rates_bp (daily 10Y yield change in bp),
    usd (dollar log return), oil (WTI log return).
    """
    mapping = {"equity": "sp500_logret", "rates_bp": "d_ust_10y_bp",
               "usd": "usd_index_logret", "oil": "wti_crude_logret"}
    missing = [src for src in mapping.values() if src not in macro_features.columns]
    if missing:
        raise RiskError(f"macro features lack columns needed for the factor model: {missing}")
    return macro_features[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()}).dropna()


@dataclass(frozen=True)
class FactorModel:
    """OLS sensitivities of each asset's log return to each macro factor."""

    betas: pd.DataFrame  # index = factors, columns = assets
    intercept: pd.Series
    r2: pd.Series
    n_obs: int
    condition_number: float  # of the standardized factor matrix; > 30 suggests collinearity


def fit_factor_model(prices: pd.DataFrame, factors: pd.DataFrame) -> FactorModel:
    """Regress asset log returns on factor changes (dates are aligned by intersection)."""
    asset_ret = np.log(prices.where(prices > 0)).diff().dropna(how="all")
    data = asset_ret.join(factors, how="inner", rsuffix="_factor").dropna()
    if len(data) < max(60, 5 * (factors.shape[1] + 1)):
        raise RiskError(f"Only {len(data)} overlapping observations for the factor model")
    y = data[list(asset_ret.columns)].to_numpy()
    f = data[list(factors.columns)].to_numpy()
    x = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)  # (X'X)^-1 X'Y for every asset at once
    resid = y - x @ coef
    r2 = 1.0 - resid.var(axis=0) / np.maximum(y.var(axis=0), 1e-30)
    std_f = (f - f.mean(axis=0)) / np.maximum(f.std(axis=0), 1e-30)
    cond = float(np.linalg.cond(std_f))
    if cond > 30:
        logger.warning("Factor matrix condition number %.1f: betas may be unstable (collinear)", cond)
    return FactorModel(
        betas=pd.DataFrame(coef[1:], index=list(factors.columns), columns=list(asset_ret.columns)),
        intercept=pd.Series(coef[0], index=list(asset_ret.columns)),
        r2=pd.Series(r2, index=list(asset_ret.columns)),
        n_obs=len(data),
        condition_number=cond,
    )


def _shock_vector(model: FactorModel, scenario: FactorScenario, scale: float = 1.0) -> FloatArray:
    """Scenario shocks in the factors' regression units (log returns, or bp)."""
    unknown = set(scenario.shocks) - set(model.betas.index)
    if unknown:
        raise RiskError(f"Scenario {scenario.name!r} uses unknown factors: {sorted(unknown)}")
    out = np.zeros(len(model.betas.index))
    for i, name in enumerate(model.betas.index):
        value = float(scenario.shocks.get(name, 0.0)) * scale
        if name.endswith("_bp"):
            out[i] = value
        else:
            if value <= -1.0:
                raise RiskError(f"{name}: a shock of {value:+.0%} would take the price to zero or below")
            out[i] = np.log1p(value)
    return out


def _asset_shocks(model: FactorModel, scenario: FactorScenario, scale: float = 1.0) -> FloatArray:
    log_move = _shock_vector(model, scenario, scale) @ model.betas.to_numpy()
    return np.expm1(log_move)  # back to simple returns


# --------------------------------------------------------------------------- #
# Configuration and report containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskConfig:
    """Immutable risk settings."""

    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    horizon_days: int = 1
    portfolio_value: float = 1_000_000.0
    trading_days: int = 252
    ewma_lambda: float = 0.94
    backtest_window: int = 500

    def __post_init__(self) -> None:
        if not self.confidence_levels or any(not 0.5 < a < 1.0 for a in self.confidence_levels):
            raise RiskError("confidence_levels must be in (0.5, 1)")
        if self.horizon_days < 1 or self.portfolio_value <= 0 or self.trading_days < 1:
            raise RiskError("horizon_days, portfolio_value, trading_days must be positive")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise RiskError("ewma_lambda must lie in (0, 1)")
        if self.backtest_window < 60:
            raise RiskError("backtest_window must be at least 60")


@dataclass(frozen=True)
class RiskReport:
    """Everything ``RiskEngine.analyze`` produces for one portfolio."""

    returns: pd.Series
    var_table: pd.DataFrame
    drawdown: DrawdownAnalysis
    backtest: pd.DataFrame
    contributions: pd.DataFrame
    historical_stress: pd.DataFrame
    factor_model: FactorModel | None
    factor_stress: pd.DataFrame | None
    config: RiskConfig = field(default_factory=RiskConfig)

    def summary(self) -> str:
        fmt = {"var": "{:.2%}".format, "es": "{:.2%}".format, "var_usd": "{:,.0f}".format,
               "es_usd": "{:,.0f}".format}
        lines = [f"VaR / ES ({self.config.horizon_days}-day, portfolio ${self.config.portfolio_value:,.0f}):",
                 self.var_table.to_string(formatters=fmt, na_rep="n/a"), ""]
        d = self.drawdown
        lines += [f"Max drawdown {d.max_drawdown:.2%}; current {d.current_drawdown:.2%}; "
                  f"longest underwater {d.longest_underwater_bars} days; ulcer index {d.ulcer_index:.4f}", ""]
        lines += ["VaR backtest (forecast made before each day):",
                  self.backtest.round(4).to_string(), ""]
        lines += ["Risk contribution by asset:", self.contributions.round(4).to_string(), ""]
        if not self.historical_stress.empty:
            lines += ["Historical stress scenarios:",
                      self.historical_stress[["portfolio_return", "pnl_usd", "n_days"]].round(4).to_string(), ""]
        if self.factor_stress is not None and not self.factor_stress.empty:
            lines += ["Macro factor stress scenarios:",
                      self.factor_stress[["portfolio_return", "pnl_usd"]].round(4).to_string()]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RiskEngine:
    """VaR/ES, drawdown, VaR backtests, risk contributions and stress tests.

    Example:
        >>> engine = RiskEngine(RiskConfig(portfolio_value=250_000))
        >>> report = engine.analyze(prices, weights, factors=macro_factor_frame(macro))
        >>> print(report.summary())
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()

    # ------------------------------------------------------------------ #
    @staticmethod
    def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """Constant-weight (daily rebalanced) portfolio return series."""
        w = weights.reindex(asset_returns.columns)
        if w.isna().any():
            raise RiskError("weights must cover every column of asset_returns")
        return pd.Series(asset_returns.to_numpy() @ w.to_numpy(dtype=float), index=asset_returns.index,
                         name="portfolio")

    @staticmethod
    def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        if (prices <= 0).any().any() or prices.isna().any().any():
            raise RiskError("prices must be positive and gap-free (use DataEngine output)")
        return (prices / prices.shift(1) - 1.0).iloc[1:]

    # ------------------------------------------------------------------ #
    def var_table(self, returns: pd.Series, portfolio_value: float | None = None) -> pd.DataFrame:
        """VaR and ES for every method and confidence level (fractions and dollars)."""
        cfg = self.cfg
        value = cfg.portfolio_value if portfolio_value is None else portfolio_value
        h = cfg.horizon_days
        r = _clean(returns, max(30, h + 30))
        hist = r
        if h > 1:  # overlapping compounded h-day returns
            hist = np.expm1(pd.Series(np.log1p(r)).rolling(h).sum().dropna().to_numpy())
        rows = []
        for a in cfg.confidence_levels:
            fhs_v, fhs_e = fhs_var_es(r, a, cfg.ewma_lambda, h)
            rows += [
                ("Historical", a, var_historical(hist, a), es_historical(hist, a)),
                ("Parametric (normal)", a, var_parametric(r, a, h), es_parametric(r, a, h)),
                ("Cornish-Fisher", a, var_cornish_fisher(r, a, h), float("nan")),
                ("Filtered historical (EWMA)", a, fhs_v, fhs_e),
            ]
        table = pd.DataFrame(rows, columns=["method", "confidence", "var", "es"])
        table["var_usd"], table["es_usd"] = table["var"] * value, table["es"] * value
        return table.set_index(["method", "confidence"])

    def drawdowns(self, returns: pd.Series, top_n: int = 5) -> DrawdownAnalysis:
        return drawdown_analysis(returns, top_n)

    # ------------------------------------------------------------------ #
    def var_forecasts(
        self, returns: pd.Series, alpha: float, method: str, window: int | None = None
    ) -> pd.Series:
        """One-day-ahead VaR forecast for each day, using only data BEFORE that day.

        Fully vectorized: rolling quantiles / moments are shifted one bar so the forecast
        for day t never sees r_t.  The first ``window + 1`` days are NaN for every method.
        """
        _check_alpha(alpha)
        if method not in METHODS:
            raise RiskError(f"method must be one of {METHODS}")
        w = window or self.cfg.backtest_window
        r = pd.Series(_clean(returns, w + 10), index=returns.index)
        z_tail = float(stats.norm.ppf(1.0 - alpha))  # negative
        if method == "historical":
            var = -r.rolling(w).quantile(1.0 - alpha).shift(1)
        elif method == "parametric":
            var = -(r.rolling(w).mean() + z_tail * r.rolling(w).std(ddof=1)).shift(1)
        else:
            s2 = pd.Series(ewma_variance(r.to_numpy(), self.cfg.ewma_lambda), index=r.index)
            sigma_prev = np.sqrt(s2.shift(1))  # volatility known before day t
            if method == "ewma":
                var = -z_tail * sigma_prev
            else:  # fhs: rolling empirical quantile of standardized returns, times known sigma
                z = r / sigma_prev
                q = z.rolling(w).quantile(1.0 - alpha).shift(1)
                var = -sigma_prev * q
        var.iloc[: w + 1] = np.nan  # common evaluation sample (FHS needs one extra bar)
        return var.rename(f"{method}_{alpha:.0%}")

    def backtest_var(
        self, returns: pd.Series, alpha: float, method: str, window: int | None = None
    ) -> pd.DataFrame:
        """Realized return, VaR forecast and exception flag for each day."""
        var = self.var_forecasts(returns, alpha, method, window)
        r = pd.Series(np.asarray(returns, dtype=float), index=returns.index)
        out = pd.DataFrame({"return": r, "var": var})
        out["exception"] = out["return"] < -out["var"]
        out.loc[out["var"].isna(), "exception"] = False
        return out

    def backtest_table(
        self, returns: pd.Series, methods: tuple[str, ...] = METHODS, window: int | None = None
    ) -> pd.DataFrame:
        """Kupiec / Christoffersen tests and Basel zone for every method and confidence level."""
        rows = []
        for a in self.cfg.confidence_levels:
            p = 1.0 - a
            for method in methods:
                bt = self.backtest_var(returns, a, method, window).dropna(subset=["var"])
                t, x = len(bt), int(bt["exception"].sum())
                lr_pof, p_pof = kupiec_pof(t, x, p)
                lr_ind, p_ind = christoffersen_independence(bt["exception"].to_numpy())
                p_cc = float(stats.chi2.sf(lr_pof + lr_ind, 2))
                last250 = int(bt["exception"].iloc[-250:].sum())
                zone = ("green" if last250 <= 4 else "yellow" if last250 <= 9 else "red") if abs(a - 0.99) < 1e-9 else ""
                rows.append({"method": method, "confidence": a, "days": t, "exceptions": x,
                             "expected": t * p, "rate": x / t, "kupiec_p": p_pof,
                             "christoffersen_p": p_ind, "cond_coverage_p": p_cc,
                             "verdict": "adequate" if p_cc >= 0.05 else "rejected (5%)",
                             "basel_zone_last250": zone})
        return pd.DataFrame(rows).set_index(["method", "confidence"])

    # ------------------------------------------------------------------ #
    def risk_contributions(self, asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
        """Euler risk decomposition using the annualized sample covariance."""
        cov = asset_returns.cov() * self.cfg.trading_days
        return risk_contributions(weights, cov)

    def historical_stress(
        self, prices: pd.DataFrame, weights: pd.Series,
        scenarios: Mapping[str, tuple[str, str]] | None = None, portfolio_value: float | None = None,
    ) -> pd.DataFrame:
        """Replay crisis windows on today's weights (buy-and-hold from the window start)."""
        value = self.cfg.portfolio_value if portfolio_value is None else portfolio_value
        w = weights.reindex(prices.columns)
        if w.isna().any():
            raise RiskError("weights must cover every column of prices")
        rows = {}
        for name, (start, end) in (scenarios or DEFAULT_HISTORICAL_SCENARIOS).items():
            lo = int(prices.index.searchsorted(pd.Timestamp(start), side="left"))
            hi = int(prices.index.searchsorted(pd.Timestamp(end), side="right")) - 1
            if lo >= len(prices) or hi <= lo or prices.index[lo] > pd.Timestamp(end):
                logger.info("Scenario %r skipped: price history does not cover it", name)
                continue
            asset = prices.iloc[hi] / prices.iloc[lo] - 1.0
            port = float(w @ asset)
            rows[name] = {"start": prices.index[lo], "end": prices.index[hi], "n_days": hi - lo,
                          "portfolio_return": port, "pnl_usd": port * value,
                          "worst_asset": str(asset.idxmin()), "worst_asset_return": float(asset.min())}
        cols = ["start", "end", "n_days", "portfolio_return", "pnl_usd", "worst_asset", "worst_asset_return"]
        return pd.DataFrame.from_dict(rows, orient="index", columns=cols)

    def factor_stress(
        self, model: FactorModel, weights: pd.Series,
        scenarios: tuple[FactorScenario, ...] | None = None, portfolio_value: float | None = None,
    ) -> pd.DataFrame:
        """Portfolio and per-asset loss under each hypothetical macro shock."""
        value = self.cfg.portfolio_value if portfolio_value is None else portfolio_value
        w = weights.reindex(model.betas.columns)
        if w.isna().any():
            raise RiskError("weights must cover every asset in the factor model")
        rows = {}
        use_defaults = scenarios is None
        for sc in scenarios or DEFAULT_MACRO_SCENARIOS:
            missing = set(sc.shocks) - set(model.betas.index)
            if use_defaults and missing:  # explicit scenarios stay strict (error below)
                logger.warning("Default scenario %r skipped: model lacks factors %s", sc.name, sorted(missing))
                continue
            asset = _asset_shocks(model, sc)
            port = float(w.to_numpy(dtype=float) @ asset)
            rows[sc.name] = {"portfolio_return": port, "pnl_usd": port * value,
                             **{f"asset_{a}": v for a, v in zip(model.betas.columns, asset)}}
        return pd.DataFrame.from_dict(rows, orient="index")

    def reverse_stress(
        self, model: FactorModel, weights: pd.Series, scenario: FactorScenario, target_loss: float
    ) -> float:
        """How many multiples of ``scenario`` produce a portfolio loss of ``target_loss``?

        Solves portfolio_return(k * shocks) = -target_loss for k with a bracketing root find
        (the map is mildly nonlinear because log moves are converted back to simple returns).
        Returns NaN if even 10x the scenario does not reach the target.
        """
        if not 0.0 < target_loss < 1.0:
            raise RiskError("target_loss must be a fraction in (0, 1)")
        w = weights.reindex(model.betas.columns).to_numpy(dtype=float)

        def gap(k: float) -> float:
            return float(w @ _asset_shocks(model, scenario, k)) + target_loss

        if gap(0.0) <= 0:
            return 0.0
        # A price factor cannot fall 100% or more, so k * shock > -1 bounds the search.
        worst_price_drop = max((-v for n, v in scenario.shocks.items() if not n.endswith("_bp") and v < 0),
                               default=0.0)
        k_cap = min(10.0, 0.999 / worst_price_drop) if worst_price_drop > 0 else 10.0
        if gap(k_cap) > 0:
            return float("nan")
        return float(brentq(gap, 0.0, k_cap))

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        prices: pd.DataFrame,
        weights: pd.Series,
        factors: pd.DataFrame | None = None,
        historical_scenarios: Mapping[str, tuple[str, str]] | None = None,
        macro_scenarios: tuple[FactorScenario, ...] | None = None,
    ) -> RiskReport:
        """Run every analysis for a portfolio of ``weights`` held in ``prices``' assets."""
        asset_returns = self.simple_returns(prices)
        port = self.portfolio_returns(asset_returns, weights)
        model = fit_factor_model(prices, factors) if factors is not None else None
        return RiskReport(
            returns=port,
            var_table=self.var_table(port),
            drawdown=self.drawdowns(port),
            backtest=self.backtest_table(port),
            contributions=self.risk_contributions(asset_returns, weights),
            historical_stress=self.historical_stress(prices, weights, historical_scenarios),
            factor_model=model,
            factor_stress=self.factor_stress(model, weights, macro_scenarios) if model else None,
            config=self.cfg,
        )
