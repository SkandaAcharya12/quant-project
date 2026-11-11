"""
data_engine.py -- Quantitative Data Engine (Phase 2)
====================================================

Responsibilities
----------------
1. Thread-safe ingestion of OHLCV history from Yahoo Finance (``yfinance``).
2. Cleaning and gap handling that never silently fabricates history.
3. Macro indicators: market-implied proxies from Yahoo (VIX, Treasury yields,
   dollar, gold, oil) plus optional FRED series with a publication-lag shift.
4. Technical indicators and statistical features written from scratch
   (no TA-Lib), fully vectorized and strictly causal.

Mathematical definitions (used throughout; r_t is the log return)
------------------------------------------------------------------
    r_t        = ln(P_t) - ln(P_{t-1})            additive over time, so a
                                                  k-day return is sum(r) and
                                                  rolling statistics compose.
    SMA_n(t)   = (1/n) * sum_{i=0}^{n-1} P_{t-i}
    EMA(t)     = a*P_t + (1-a)*EMA(t-1),  a = 2/(n+1)
    Z_n(t)     = (x_t - mu_n(t)) / sigma_n(t)     mu, sigma over the trailing
                                                  window INCLUDING t, so no
                                                  future data is used.
    Bollinger  = SMA_n +/- k * sigma_n            (population sigma, ddof=0)
    RSI_n(t)   = 100 * G / (G + L)                G, L = Wilder-smoothed average
                                                  gain / loss. Algebraically equal
                                                  to 100 - 100/(1 + G/L) but
                                                  safe when L = 0.
    MACD       = EMA_12 - EMA_26;  signal = EMA_9(MACD);  hist = MACD - signal
    TR_t       = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|);  ATR = Wilder-smoothed TR
    RealVol    = std(r over n days) * sqrt(252)   (annualized)

Thread-safety model
-------------------
Worker threads share only (a) the per-symbol cache and (b) the lock registry,
both guarded by ``_state_lock``. A per-symbol lock gives "single flight"
semantics: if two threads request the same symbol, one downloads and the
other waits and reads the cache. Workers never mutate shared DataFrames;
they return private copies, and all merging happens in the calling thread.

Known data limitations (be explicit about them in any write-up)
----------------------------------------------------------------
* Yahoo data is free and convenient but is not survivorship-bias-free: tickers
  that were delisted are mostly absent, which flatters historical backtests.
* ``auto_adjust=True`` rescales history for splits and dividends, so absolute
  price levels differ from what was quoted at the time. Returns are correct.
* FRED macro series are revised after release. The rigorous fix is vintage data
  (ALFRED); here we approximate with a publication-lag shift.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

__all__ = [
    "DataEngine",
    "DataEngineConfig",
    "DataEngineError",
    "DataFetchError",
    "DataQualityReport",
    "DataValidationError",
    "IndicatorParams",
    "TechnicalIndicators",
    "retry",
]

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

# Market-implied macro proxies available without an API key.
# Treasury tickers (^TNX, ^FVX, ^IRX) are quoted in percent (4.25 == 4.25%).
DEFAULT_MACRO_TICKERS: dict[str, str] = {
    "vix": "^VIX",
    "ust_10y": "^TNX",
    "ust_5y": "^FVX",
    "ust_3m": "^IRX",
    "usd_index": "DX-Y.NYB",
    "gold": "GC=F",
    "wti_crude": "CL=F",
    "sp500": "^GSPC",
}

# Intervals at which one bar == one calendar day or longer, so timestamps can
# safely be normalized to midnight.
_DAILY_OR_SLOWER: frozenset[str] = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", pd.Series, pd.DataFrame)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DataEngineError(Exception):
    """Base class for all Data Engine failures."""


class DataFetchError(DataEngineError):
    """A remote API call failed or returned nothing usable."""


class DataValidationError(DataEngineError):
    """Data was retrieved but violates a structural or quality requirement."""


# --------------------------------------------------------------------------- #
# Retry with exponential backoff
# --------------------------------------------------------------------------- #
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.25,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a flaky callable with exponential backoff and random jitter.

    The wait before attempt k+1 is ``base_delay * backoff**(k-1) * (1 + U)``,
    ``U ~ Uniform(-jitter, +jitter)``. Exponential growth gives a struggling
    server room to recover; jitter de-synchronizes worker threads so they do
    not all retry at the same instant (the "thundering herd" problem).

    Raises:
        DataFetchError: once all attempts are exhausted (original error chained).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = base_delay * backoff ** (attempt - 1)
                    delay *= 1.0 + random.uniform(-jitter, jitter)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s -- retrying in %.2fs",
                        getattr(func, "__name__", "call"),
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(max(delay, 0.0))
            raise DataFetchError(
                f"{getattr(func, '__name__', 'call')} failed after "
                f"{max_attempts} attempts: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Configuration and reporting containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IndicatorParams:
    """Windows and constants for technical features (defaults are market convention)."""

    sma_window: int = 20
    ema_span: int = 20
    zscore_window: int = 20
    rsi_period: int = 14
    bb_window: int = 20
    bb_num_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    vol_window: int = 20
    trading_days: int = 252


@dataclass(frozen=True)
class DataEngineConfig:
    """Immutable engine configuration (immutability makes sharing across threads safe).

    ``ffill_limit``: max consecutive NaNs forward-filled. Forward fill is causal
        (uses only the past), so it is safe for backtesting. ``None`` = unlimited.
    ``bfill_limit``: max NaNs back-filled at the very START of a series only.
        Back-filling is NOT causal (it pulls a future price into the past), so it
        is restricted to the leading edge and capped. Its legitimate use is
        aligning calendars across markets, not inventing pre-listing history.
    ``max_missing_fraction``: tickers missing more than this share of the panel
        calendar are dropped and reported, rather than being heavily imputed.
    ``drop_incomplete_rows``: keep only dates where every ticker has a price,
        which yields the balanced matrix that covariance estimation requires.
    """

    start: str = "2015-01-01"
    end: str | None = None  # yfinance treats `end` as exclusive
    interval: str = "1d"
    max_workers: int = 8
    max_retries: int = 3
    retry_base_delay: float = 1.0
    ffill_limit: int | None = 5
    bfill_limit: int = 5
    max_missing_fraction: float = 0.20
    drop_incomplete_rows: bool = True
    suspect_jump_threshold: float = 0.5  # |log return| flagged as a possible bad tick
    fred_api_key: str | None = None  # falls back to the FRED_API_KEY env variable
    tz_cache_dir: str | None = None  # None -> <project>/data/cache/yf_tz
    macro_tickers: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_MACRO_TICKERS))
    indicators: IndicatorParams = field(default_factory=IndicatorParams)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if self.ffill_limit is not None and self.ffill_limit < 1:
            raise ValueError("ffill_limit must be >= 1 or None")
        if self.bfill_limit < 0:
            raise ValueError("bfill_limit must be >= 0")
        if not 0.0 <= self.max_missing_fraction < 1.0:
            raise ValueError("max_missing_fraction must lie in [0, 1)")


@dataclass(frozen=True)
class DataQualityReport:
    """Audit trail of every cleaning decision applied to a price panel."""

    requested: tuple[str, ...]
    failed: dict[str, str]  # ticker -> reason the download failed
    dropped_sparse: dict[str, float]  # ticker -> missing fraction that got it dropped
    ffill_counts: dict[str, int]  # ticker -> observations forward-filled
    bfill_counts: dict[str, int]  # ticker -> leading observations back-filled
    first_valid: dict[str, pd.Timestamp]  # ticker -> first real (unfilled) observation
    rows_dropped_incomplete: int
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp


# --------------------------------------------------------------------------- #
# Technical indicators (from scratch, vectorized, causal)
# --------------------------------------------------------------------------- #
class TechnicalIndicators:
    """Stateless library of vectorized indicators.

    Every value at time t depends only on data at times <= t, so features can
    be fed to a backtester without look-ahead bias. Functions typed with ``T``
    accept a Series or a whole DataFrame panel (computed column-wise at once).
    """

    @staticmethod
    def log_returns(prices: T) -> T:
        """r_t = ln(P_t / P_{t-1}). Non-positive prices become NaN, never -inf."""
        return np.log(prices.where(prices > 0)).diff()

    @staticmethod
    def simple_returns(prices: T) -> T:
        """R_t = P_t / P_{t-1} - 1 (the arithmetic return used for portfolio sums)."""
        return prices / prices.shift(1) - 1.0

    @staticmethod
    def sma(x: T, window: int) -> T:
        """Simple moving average; NaN until a full window is available."""
        return x.rolling(window, min_periods=window).mean()

    @staticmethod
    def ema(x: T, span: int) -> T:
        """Exponential moving average with a = 2/(span+1).

        ``adjust=False`` selects the recursive form y_t = a*x_t + (1-a)*y_{t-1}.
        The first ``span - 1`` values are masked because the seed value still
        dominates them (they would be a transient, not a signal).
        """
        return x.ewm(span=span, adjust=False, min_periods=span).mean()

    @staticmethod
    def rolling_zscore(x: T, window: int) -> T:
        """Z_t = (x_t - mean_n) / std_n with the sample std (ddof=1).

        Zero-variance windows return NaN instead of +/-inf. Used as the
        standardized spread signal in statistical arbitrage.
        """
        roll = x.rolling(window, min_periods=window)
        mu = roll.mean()
        sd = roll.std(ddof=1)
        return (x - mu) / sd.where(sd > 0)

    @staticmethod
    def rsi(close: T, period: int = 14) -> T:
        """Relative Strength Index with Wilder smoothing.

        Wilder's smoothing is an EMA with a = 1/period (not 2/(period+1)),
        which is why ``ewm(alpha=1/period)`` is used. Flat prices (G = L = 0)
        are defined as the neutral value 50.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        denom = avg_gain + avg_loss
        rsi = 100.0 * avg_gain / denom.where(denom > 0)
        return rsi.where(denom != 0, 50.0)

    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands, %B and bandwidth.

        %B = (P - lower) / (upper - lower) locates price inside the band
        (0 = lower, 1 = upper). Bandwidth = (upper - lower) / mid is a
        scale-free volatility measure. Population std (ddof=0) follows
        Bollinger's original definition.
        """
        roll = close.rolling(window, min_periods=window)
        mid = roll.mean()
        sd = roll.std(ddof=0)
        upper = mid + num_std * sd
        lower = mid - num_std * sd
        width = upper - lower
        return pd.DataFrame(
            {
                "bb_mid": mid,
                "bb_upper": upper,
                "bb_lower": lower,
                "bb_pct_b": (close - lower) / width.where(width > 0),
                "bb_bandwidth": width / mid.where(mid != 0),
            }
        )

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD line, signal line and histogram (trend-momentum oscillator)."""
        line = TechnicalIndicators.ema(close, fast) - TechnicalIndicators.ema(close, slow)
        sig = TechnicalIndicators.ema(line, signal)
        return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range: Wilder-smoothed TR = max(H-L, |H-C_prev|, |L-C_prev|).

        Including the previous close captures overnight gaps that H-L misses.
        All three candidates are built as columns and reduced with one
        vectorized ``max(axis=1)``.
        """
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def realized_volatility(
        close: pd.Series, window: int = 20, trading_days: int = 252
    ) -> pd.Series:
        """Annualized rolling volatility: std(r) * sqrt(trading_days).

        Scaling by sqrt(T) is exact when returns are i.i.d.; real returns show
        volatility clustering, so treat it as a convention, not a law.
        """
        r = TechnicalIndicators.log_returns(close)
        return r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(trading_days)


# --------------------------------------------------------------------------- #
# Data Engine
# --------------------------------------------------------------------------- #
class DataEngine:
    """Thread-safe market-data pipeline: fetch -> clean -> fill -> features.

    Example:
        >>> engine = DataEngine(DataEngineConfig(start="2018-01-01"))
        >>> prices = engine.get_price_panel(["KO", "PEP", "XOM"])
        >>> returns = engine.get_returns(prices)
        >>> engine.last_report.ffill_counts
    """

    def __init__(self, config: DataEngineConfig | None = None) -> None:
        self.cfg = config or DataEngineConfig()
        self._cache: dict[str, pd.DataFrame] = {}
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._state_lock = threading.Lock()
        self._last_report: DataQualityReport | None = None
        self._configure_tz_cache()
        # Wrap the raw network call with retry/backoff using this engine's settings.
        self._download = retry(
            max_attempts=self.cfg.max_retries, base_delay=self.cfg.retry_base_delay
        )(self._download_raw)

    def _configure_tz_cache(self) -> None:
        """Create yfinance's timezone cache once, before any worker thread exists.

        yfinance lazily creates a small SQLite cache the first time it is used.
        If several threads trigger that creation simultaneously they race
        (WinError 183 on Windows, "database is locked"). Creating the folder
        here, single-threaded, removes the race at its source.
        """
        setter = getattr(yf, "set_tz_cache_location", None)
        if setter is None:  # very old yfinance, or a test double
            return
        target = Path(self.cfg.tz_cache_dir) if self.cfg.tz_cache_dir else (
            Path(__file__).resolve().parent.parent / "data" / "cache" / "yf_tz"
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            setter(str(target))
        except OSError as exc:
            logger.warning("Could not set yfinance tz cache at %s: %s", target, exc)

    # ------------------------------------------------------------------ #
    # Public: ingestion
    # ------------------------------------------------------------------ #
    @property
    def last_report(self) -> DataQualityReport | None:
        """Report of the most recent ``get_price_panel`` call."""
        with self._state_lock:
            return self._last_report

    def clear_cache(self) -> None:
        """Discard every cached download."""
        with self._state_lock:
            self._cache.clear()

    def fetch_ohlcv(
        self, tickers: Sequence[str], refresh: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Download cleaned OHLCV frames concurrently.

        Tickers that fail are logged and skipped (partial results are usable).

        Raises:
            DataFetchError: if *every* ticker fails.
        """
        symbols = self._clean_tickers(tickers)
        frames, failures = self._fetch_many(symbols, refresh)
        if not frames:
            raise DataFetchError(f"All downloads failed: {failures}")
        return frames

    def build_price_panel(
        self,
        tickers: Sequence[str],
        field_name: str = "Close",
        refresh: bool = False,
        require_positive: bool = True,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Return a (dates x tickers) panel plus an audit report.

        Pipeline: concurrent download -> align on the union calendar -> drop
        sparse tickers -> forward fill (causal) -> bounded leading back fill ->
        optionally keep only complete rows.

        Args:
            field_name: OHLCV column to pivot into the panel.
            require_positive: mask non-positive values as NaN. Correct for
                prices (log returns need P > 0); disable for yields or spreads,
                which can legitimately be zero or negative.
        """
        if field_name not in OHLCV_COLUMNS:
            raise ValueError(f"field_name must be one of {OHLCV_COLUMNS}")
        symbols = self._clean_tickers(tickers)
        frames, failures = self._fetch_many(symbols, refresh)
        if not frames:
            raise DataFetchError(f"All downloads failed: {failures}")
        return self._assemble_panel(frames, failures, symbols, field_name, require_positive)

    def get_price_panel(
        self,
        tickers: Sequence[str],
        field_name: str = "Close",
        refresh: bool = False,
        require_positive: bool = True,
    ) -> pd.DataFrame:
        """Convenience wrapper over ``build_price_panel`` that stores ``last_report``."""
        panel, report = self.build_price_panel(tickers, field_name, refresh, require_positive)
        with self._state_lock:
            self._last_report = report
        return panel

    @staticmethod
    def get_returns(
        prices: pd.DataFrame, method: Literal["log", "simple"] = "log"
    ) -> pd.DataFrame:
        """Convert a price panel to returns (first row dropped).

        Use log returns for time aggregation and statistics, simple returns when
        combining assets in a portfolio at a single date (weights act linearly
        on simple returns, not on log returns).
        """
        if method == "log":
            out = TechnicalIndicators.log_returns(prices)
        elif method == "simple":
            out = TechnicalIndicators.simple_returns(prices)
        else:
            raise ValueError("method must be 'log' or 'simple'")
        return out.iloc[1:]

    # ------------------------------------------------------------------ #
    # Public: macro indicators
    # ------------------------------------------------------------------ #
    def fetch_macro(self, refresh: bool = False) -> pd.DataFrame:
        """Macro level series from market proxies, columns named by ``cfg.macro_tickers``."""
        mapping = dict(self.cfg.macro_tickers)
        panel, _ = self.build_price_panel(
            list(mapping.values()), refresh=refresh, require_positive=False
        )
        inverse = {symbol: name for name, symbol in mapping.items()}
        # Futures/FX print Sunday evening; equities, VIX and yields do not trade, so a
        # Sunday row would only be a forward-filled fake zero-change observation.
        weekdays = panel.loc[panel.index.dayofweek < 5]
        return weekdays.rename(columns=inverse)

    def build_macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Derive regime features from macro levels (only where inputs exist).

        * ``yield_curve_10y3m``: 10Y minus 3M yield in percentage points. A
          negative slope (inversion) has historically preceded recessions.
        * ``vix_zscore_252``: VIX standardized against its trailing year, which
          flags stress relative to the current regime rather than in absolute terms.
        * ``*_logret``: log returns for dollar, gold, oil, equities.
        * ``d_ust_10y_bp``: daily change in the 10Y yield, in basis points.
        """
        ti = TechnicalIndicators
        data = macro.copy()
        for col in [c for c in data.columns if c.startswith("ust_")]:
            # Defensive unit check: Yahoo once quoted yields x10 (42.5 == 4.25%).
            if data[col].median() > 25.0:
                logger.warning("%s looks quoted x10; rescaling by 0.1", col)
                data[col] = data[col] / 10.0

        feats: dict[str, pd.Series] = {}
        if {"ust_10y", "ust_3m"} <= set(data.columns):
            slope = data["ust_10y"] - data["ust_3m"]
            feats["yield_curve_10y3m"] = slope
            feats["curve_inverted"] = (slope < 0).astype(float).where(slope.notna())
        if "ust_10y" in data.columns:
            feats["d_ust_10y_bp"] = data["ust_10y"].diff() * 100.0
        if "vix" in data.columns:
            feats["vix_zscore_252"] = ti.rolling_zscore(data["vix"], 252)
        for col in ("usd_index", "gold", "wti_crude", "sp500"):
            if col in data.columns:
                # WTI printed negative in April 2020; log_returns maps that to NaN.
                feats[f"{col}_logret"] = ti.log_returns(data[col])
        return pd.concat([data, pd.DataFrame(feats)], axis=1)

    def fetch_fred(
        self,
        series: Mapping[str, str],
        publication_lag_days: Mapping[str, int] | None = None,
    ) -> pd.DataFrame:
        """Fetch FRED series (requires a free API key) as a daily forward-filled frame.

        Macro releases arrive days or weeks after the period they describe
        (CPI for March appears in mid-April). Using the observation date as if it
        were known on that date is look-ahead bias, so each series is shifted
        forward by ``publication_lag_days`` before use. Revisions are still
        ignored; ALFRED vintages would be needed to remove them.

        Args:
            series: {friendly_name: FRED_series_id}, e.g. {"cpi": "CPIAUCSL"}.
            publication_lag_days: {friendly_name: lag}; defaults to 0.
        """
        api_key = self.cfg.fred_api_key or os.environ.get("FRED_API_KEY")
        if not api_key:
            raise DataFetchError("No FRED API key: set FRED_API_KEY or config.fred_api_key")
        try:
            from fredapi import Fred  # lazy import keeps FRED optional
        except ImportError as exc:
            raise DataFetchError("fredapi is not installed (pip install fredapi)") from exc

        lags = dict(publication_lag_days or {})
        client = Fred(api_key=api_key)
        get_series = retry(
            max_attempts=self.cfg.max_retries, base_delay=self.cfg.retry_base_delay
        )(client.get_series)

        columns: dict[str, pd.Series] = {}
        for name, series_id in series.items():
            raw = get_series(
                series_id, observation_start=self.cfg.start, observation_end=self.cfg.end
            )
            s = pd.Series(raw, dtype=float).dropna()
            s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=int(lags.get(name, 0)))
            columns[name] = s[~s.index.duplicated(keep="last")]
        if not columns:
            raise DataValidationError("No FRED series requested")
        frame = pd.DataFrame(columns).sort_index()
        daily = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
        return daily.ffill()

    @staticmethod
    def align_to(macro: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Reindex a lower-frequency frame onto a trading calendar without look-ahead.

        Taking the union first means the last release *before* the first trading
        day is still carried in, instead of being lost to the reindex.
        """
        union = macro.index.union(index)
        return macro.reindex(union).ffill().reindex(index)

    # ------------------------------------------------------------------ #
    # Public: technical features
    # ------------------------------------------------------------------ #
    def build_technical_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute the full indicator set for one instrument's OHLCV frame."""
        missing = set(OHLCV_COLUMNS) - set(ohlcv.columns)
        if missing:
            raise DataValidationError(f"OHLCV frame missing columns: {sorted(missing)}")
        p = self.cfg.indicators
        ti = TechnicalIndicators
        close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

        base = pd.DataFrame(
            {
                "ret_simple": ti.simple_returns(close),
                "ret_log": ti.log_returns(close),
                f"sma_{p.sma_window}": ti.sma(close, p.sma_window),
                f"ema_{p.ema_span}": ti.ema(close, p.ema_span),
                f"zscore_{p.zscore_window}": ti.rolling_zscore(close, p.zscore_window),
                f"rsi_{p.rsi_period}": ti.rsi(close, p.rsi_period),
                f"atr_{p.atr_period}": ti.atr(high, low, close, p.atr_period),
                f"vol_{p.vol_window}d_ann": ti.realized_volatility(
                    close, p.vol_window, p.trading_days
                ),
            }
        )
        bands = ti.bollinger_bands(close, p.bb_window, p.bb_num_std)
        macd = ti.macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        return pd.concat([base, bands, macd], axis=1)

    def get_features(
        self, tickers: Sequence[str], refresh: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Technical features per ticker, computed on each ticker's own trading days."""
        frames = self.fetch_ohlcv(tickers, refresh=refresh)
        return {sym: self.build_technical_features(df) for sym, df in frames.items()}

    # ------------------------------------------------------------------ #
    # Internals: download and normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_tickers(tickers: Sequence[str]) -> list[str]:
        """Strip, upper-case and de-duplicate while preserving the caller's order."""
        if isinstance(tickers, str):
            raise TypeError("tickers must be a sequence of strings, not a single string")
        cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
        unique = list(dict.fromkeys(cleaned))
        if not unique:
            raise ValueError("No valid tickers supplied")
        return unique

    def _download_raw(self, symbol: str) -> pd.DataFrame:
        """One network call. Wrapped by ``retry`` in ``__init__``."""
        raw = yf.Ticker(symbol).history(
            start=self.cfg.start,
            end=self.cfg.end,
            interval=self.cfg.interval,
            auto_adjust=True,  # prices adjusted for splits and dividends
            actions=False,
        )
        if raw is None or raw.empty:
            raise DataFetchError(f"No data returned for {symbol!r}")
        return raw

    def _normalize_frame(self, symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, strip timezones, de-duplicate and sort the index."""
        missing = set(OHLCV_COLUMNS) - set(raw.columns)
        if missing:
            raise DataValidationError(f"{symbol}: response lacks columns {sorted(missing)}")

        frame = raw.loc[:, list(OHLCV_COLUMNS)].astype(float)
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)  # keep exchange-local wall time
        if self.cfg.interval in _DAILY_OR_SLOWER:
            idx = idx.normalize()
        frame.index = idx.rename("Date")

        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.dropna(subset=["Close"])
        if frame.empty:
            raise DataValidationError(f"{symbol}: no valid Close prices")

        return frame

    def _symbol_lock(self, symbol: str) -> threading.Lock:
        with self._state_lock:
            return self._symbol_locks.setdefault(symbol, threading.Lock())

    def _fetch_single(self, symbol: str) -> pd.DataFrame:
        """Cache-aware, single-flight download of one symbol (runs in a worker thread)."""
        with self._symbol_lock(symbol):
            with self._state_lock:
                cached = self._cache.get(symbol)
            if cached is not None:
                return cached.copy()
            frame = self._normalize_frame(symbol, self._download(symbol))
            with self._state_lock:
                self._cache[symbol] = frame
            return frame.copy()  # callers get a private copy, the cache stays pristine

    def _fetch_many(
        self, symbols: list[str], refresh: bool
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Fan out downloads across a thread pool; merge results in the calling thread."""
        if refresh:
            with self._state_lock:
                for s in symbols:
                    self._cache.pop(s, None)

        frames: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        workers = min(self.cfg.max_workers, len(symbols))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="data-engine") as pool:
            futures = {pool.submit(self._fetch_single, s): s for s in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    frames[symbol] = future.result()
                except DataEngineError as exc:
                    failures[symbol] = str(exc)
                    logger.error("Skipping %s: %s", symbol, exc)
                except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the batch
                    failures[symbol] = f"unexpected {type(exc).__name__}: {exc}"
                    logger.exception("Unexpected error for %s", symbol)
        # as_completed yields in finish order; restore the caller's ordering.
        return {s: frames[s] for s in symbols if s in frames}, failures

    # ------------------------------------------------------------------ #
    # Internals: panel assembly and gap handling
    # ------------------------------------------------------------------ #
    def _assemble_panel(
        self,
        frames: dict[str, pd.DataFrame],
        failures: dict[str, str],
        requested: list[str],
        field_name: str,
        require_positive: bool,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        cfg = self.cfg
        raw = pd.concat({s: f[field_name] for s, f in frames.items()}, axis=1).sort_index()

        if require_positive:
            bad = (raw <= 0).sum()
            for sym in bad[bad > 0].index:
                logger.warning("%s: %d non-positive values masked as NaN", sym, int(bad[sym]))
            raw = raw.where(raw > 0)

            # A one-day |log return| above the threshold is more often a bad tick
            # than reality for prices. Skipped for yields/indices (require_positive
            # False), where near-zero levels and volatility spikes are genuine.
            log_ret = TechnicalIndicators.log_returns(raw)
            jumps = (log_ret.abs() > cfg.suspect_jump_threshold).sum()
            for sym in jumps[jumps > 0].index:
                logger.warning(
                    "%s: %d suspect one-day jumps (>%.0f%% log move)",
                    sym, int(jumps[sym]), 100 * cfg.suspect_jump_threshold,
                )

        # 1) Drop tickers too sparse to impute honestly.
        missing_frac = raw.isna().mean()
        sparse = missing_frac[missing_frac > cfg.max_missing_fraction]
        dropped_sparse = {str(k): float(v) for k, v in sparse.items()}
        raw = raw.drop(columns=sparse.index)
        if raw.shape[1] == 0:
            raise DataValidationError(
                f"Every ticker exceeds max_missing_fraction: {dropped_sparse}"
            )

        # 2) Forward fill: causal (carries the last known price forward).
        ffilled = raw.ffill(limit=cfg.ffill_limit)
        ffill_counts = (ffilled.notna() & raw.isna()).sum()

        # 3) Back fill, restricted to the leading edge (before the first real print)
        #    and capped, so the past is never overwritten with future information.
        leading = raw.notna().cumsum() == 0
        if cfg.bfill_limit > 0:
            filled = ffilled.where(~leading, raw.bfill(limit=cfg.bfill_limit))
        else:
            filled = ffilled
        bfill_counts = (filled.notna() & ffilled.isna()).sum()

        # 4) Balanced panel: keep only dates where every ticker has a price.
        before = len(filled)
        filled = filled.dropna(how="any" if cfg.drop_incomplete_rows else "all")
        if filled.empty:
            raise DataValidationError(
                "No dates remain after cleaning; widen limits or drop tickers"
            )

        report = DataQualityReport(
            requested=tuple(requested),
            failed=dict(failures),
            dropped_sparse=dropped_sparse,
            ffill_counts={str(k): int(v) for k, v in ffill_counts.items()},
            bfill_counts={str(k): int(v) for k, v in bfill_counts.items()},
            first_valid={str(k): pd.Timestamp(v) for k, v in raw.notna().idxmax().items()},
            rows_dropped_incomplete=before - len(filled),
            n_obs=len(filled),
            start=filled.index[0],
            end=filled.index[-1],
        )
        filled.columns.name = None
        return filled, report


# --------------------------------------------------------------------------- #
# Demo (requires network access)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = DataEngine(DataEngineConfig(start="2018-01-01"))

    prices = engine.get_price_panel(["KO", "PEP", "XOM", "CVX"])
    print(prices.tail(), "\n")
    print(engine.last_report, "\n")

    returns = engine.get_returns(prices)
    print("Correlation matrix of daily log returns:\n", returns.corr().round(3), "\n")

    macro = engine.build_macro_features(engine.fetch_macro())
    print(macro.tail(), "\n")

    features = engine.get_features(["KO"])["KO"]
    print(features.dropna().tail())


if __name__ == "__main__":
    _demo()
