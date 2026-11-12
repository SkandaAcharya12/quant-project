"""Leakage-aware machine-learning research alerts for paper-trading practice.

The agent fits a logistic classifier to historical features and next-bar return
direction. For a decision at time t, it trains only on feature rows before t,
whose next-bar labels are therefore observable by t. It produces an advisory
alert; order submission always remains a separate user action.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class AgentError(ValueError):
    """Invalid inputs or insufficient history for a model alert."""


class AlertAction(str, Enum):
    """Advisory action; HOLD is intentionally unavailable as an order type."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"


@dataclass(frozen=True)
class AgentAlert:
    """Machine-readable prediction with model provenance and a plain-language reason."""

    symbol: str
    as_of: pd.Timestamp
    action: AlertAction
    probability_up: float
    probability_down: float
    confidence: float
    training_observations: int
    evidence: tuple[tuple[str, float], ...]
    explanation: str


class ResearchAgent:
    """Train an expanding-history classifier and emit an advisory trade alert.

    Features are close-to-close 5/21/63 bar momentum, 21-bar realized volatility,
    and a 20-vs-60 bar moving-average spread. Scaling is fitted inside each
    decision's training sample. The target is the next bar's return sign, so the
    latest row is never used as a training label. Sigmoid calibration uses
    forward-only folds inside the training sample. The default 90% confidence
    threshold avoids turning weak probability differences into forced trades;
    it can legitimately produce mostly HOLD alerts.
    """

    def __init__(self, min_training_observations: int = 126,
                 confidence_threshold: float = 0.90) -> None:
        if min_training_observations < 30:
            raise AgentError("min_training_observations must be at least 30")
        if not 0.5 < confidence_threshold < 1.0:
            raise AgentError("confidence_threshold must be between 0.5 and 1.0")
        self.min_training_observations = min_training_observations
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        prices: pd.Series,
        symbol: str,
        as_of: str | pd.Timestamp | None = None,
        current_quantity: float = 0.0,
    ) -> AgentAlert:
        """Fit with labels known by ``as_of`` and evaluate the latest eligible close."""
        source_index = prices.index
        if not isinstance(source_index, pd.DatetimeIndex):
            raise AgentError("prices must have a DatetimeIndex")
        series = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
        if series.index.has_duplicates or (series <= 0).any():
            raise AgentError("prices must have unique dates and positive values")
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise AgentError("prices must be finite")
        if not np.isfinite(current_quantity):
            raise AgentError("current_quantity must be finite")
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            if source_index.tz is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(source_index.tz)
            elif source_index.tz is None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
            series = series.loc[:cutoff]
        if len(series) < self.min_training_observations + 65:
            raise AgentError("insufficient history: need at least min_training_observations + 65 closes")

        feature = self.feature_frame(series)
        next_return = series.pct_change().shift(-1)
        decision_time = series.index[-1]
        # Rows before the decision time have labels ending no later than the
        # decision close. The decision row itself has an unknown next-bar label.
        train = feature.join(next_return.rename("next_return")).dropna()
        train = train.loc[train.index < decision_time]
        if len(train) < self.min_training_observations:
            raise AgentError("not enough complete historical feature/label rows")
        y = (train.pop("next_return") > 0.0).astype(int)
        if y.nunique() < 2:
            raise AgentError("training sample contains only one return direction")
        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1_000,
                random_state=0,
                solver="lbfgs",
            ),
        )
        # TimeSeriesSplit avoids using later observations to calibrate earlier
        # predictions. Calibration is still an estimate, not a guarantee.
        model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3),
        )
        try:
            model.fit(train.to_numpy(dtype=float), y.to_numpy(dtype=int))
            probabilities = model.predict_proba(feature.loc[[decision_time]].to_numpy(dtype=float))[0]
        except (ValueError, FloatingPointError) as exc:
            raise AgentError(f"could not fit alert model: {exc}") from exc

        classes = model.classes_
        probability_up = float(probabilities[int(np.flatnonzero(classes == 1)[0])])
        probability_down = 1.0 - probability_up
        confidence = max(probability_up, probability_down)
        if confidence < self.confidence_threshold:
            action = AlertAction.HOLD
        elif probability_up > probability_down:
            action = AlertAction.COVER if current_quantity < 0 else AlertAction.BUY
        else:
            action = AlertAction.SELL if current_quantity > 0 else AlertAction.SHORT

        point = feature.loc[decision_time]
        evidence = tuple((name, float(value)) for name, value in point.items())
        evidence_text = ", ".join(f"{name}={value:.3%}" for name, value in evidence)
        explanation = (
            f"Time-calibrated logistic model estimates {probability_up:.1%} probability of a positive next-bar "
            f"return ({probability_down:.1%} negative). Latest features: {evidence_text}. "
            f"Alert: {action.value}; this is a research signal, not "
            "a forecast guarantee."
        )
        if action == AlertAction.HOLD:
            explanation = (f"Model confidence ({confidence:.1%}) is below the configured "
                           f"{self.confidence_threshold:.1%} threshold; hold and wait for more evidence.")
        return AgentAlert(symbol.strip().upper(), decision_time, action, probability_up,
                          probability_down, confidence, len(train), evidence, explanation)

    @staticmethod
    def feature_frame(prices: pd.Series) -> pd.DataFrame:
        """Expose the exact causal inputs used by the model for chart transparency."""
        return ResearchAgent._features(prices)

    @staticmethod
    def _features(prices: pd.Series) -> pd.DataFrame:
        """Compute trailing-only features; pandas rolling windows are right-aligned."""
        returns = prices.pct_change()
        return pd.DataFrame(
            {
                "momentum_5": prices.pct_change(5),
                "momentum_21": prices.pct_change(21),
                "momentum_63": prices.pct_change(63),
                "volatility_21": returns.rolling(21, min_periods=21).std(ddof=1),
                "ma_spread_20_60": prices.rolling(20, min_periods=20).mean()
                / prices.rolling(60, min_periods=60).mean() - 1.0,
            },
            index=prices.index,
        )

