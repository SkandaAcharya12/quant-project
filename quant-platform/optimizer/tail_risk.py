"""Historical CVaR portfolio optimization using the Rockafellar-Uryasev linear program."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from optimizer.optimizer import OptimizationError, OptimizerConfig, Portfolio

__all__ = ["CVaRFrontier", "CVaROptimizer", "CVaRResult"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CVaRResult:
    """One minimum-CVaR portfolio and its in-sample tail diagnostics.

    VaR and CVaR are positive loss magnitudes, in daily-return units. The
    optimized CVaR is the empirical Rockafellar-Uryasev objective.
    """

    portfolio: Portfolio
    confidence_level: float
    minimum_expected_return: float | None
    value_at_risk: float
    expected_shortfall: float
    threshold_loss: float
    observations: int
    daily_returns: pd.Series
    solver_message: str


@dataclass(frozen=True)
class CVaRFrontier:
    """Minimum historical CVaR solutions over feasible expected-return targets."""

    points: pd.DataFrame
    weights: pd.DataFrame
    confidence_level: float


class CVaROptimizer:
    """Minimize historical expected shortfall subject to portfolio constraints.

    For scenario return vectors ``r_t`` and confidence ``alpha``, define loss
    ``L_t(w) = -r_t' w``. Rockafellar and Uryasev show that CVaR can be written
    as the convex program

        min_(w, eta, u) eta + [1 / ((1-alpha) T)] sum_t u_t
        subject to u_t >= L_t(w) - eta, u_t >= 0.

    We add full-investment, configured weight bounds, and optionally an
    annualized arithmetic-return floor. The LP is solved with HiGHS and sparse
    scenario constraints; no Gaussian tail assumption is made. All scenarios
    passed to ``optimize_returns`` must be from the formation sample.
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def optimize_returns(
        self,
        returns: pd.DataFrame,
        confidence_level: float = 0.95,
        minimum_expected_return: float | None = None,
    ) -> CVaRResult:
        """Find the feasible portfolio with the smallest historical CVaR."""
        values, names, index = self._validate_returns(returns)
        return self._solve(
            values,
            names,
            index,
            confidence_level,
            minimum_expected_return,
        )

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        confidence_level: float = 0.95,
        points: int = 11,
    ) -> CVaRFrontier:
        """Trace minimum CVaR as the feasible annual return floor is raised.

        The linear-programming return extrema define the range. Convexity means
        every intermediate target is feasible, and the CVaR optimum cannot
        improve as the return floor tightens.
        """
        values, names, index = self._validate_returns(returns)
        self._validate_confidence(confidence_level)
        if points < 2 or points > 51:
            raise OptimizationError("frontier points must be between 2 and 51")

        low_return, high_return = self._return_range(values)
        if np.isclose(low_return, high_return, atol=1e-10, rtol=1e-8):
            targets = np.array([low_return], dtype=float)
        else:
            targets = np.linspace(low_return, high_return, points)

        solutions = [
            self._solve(values, names, index, confidence_level, float(target))
            for target in targets
        ]
        point_rows = [
            {
                "target_return": float(target),
                "expected_return": solution.portfolio.expected_return,
                "value_at_risk": solution.value_at_risk,
                "expected_shortfall": solution.expected_shortfall,
                "volatility": solution.portfolio.volatility,
                "sharpe": solution.portfolio.sharpe,
            }
            for target, solution in zip(targets, solutions, strict=True)
        ]
        point_frame = pd.DataFrame(point_rows)
        weight_frame = pd.DataFrame(
            [solution.portfolio.weights.to_numpy() for solution in solutions],
            index=pd.Index(targets, name="target_return"),
            columns=names,
        )
        return CVaRFrontier(point_frame, weight_frame, confidence_level)

    def _solve(
        self,
        values: FloatArray,
        names: tuple[str, ...],
        index: pd.Index,
        confidence_level: float,
        minimum_expected_return: float | None,
    ) -> CVaRResult:
        self._validate_confidence(confidence_level)
        if minimum_expected_return is not None and not np.isfinite(minimum_expected_return):
            raise OptimizationError("minimum_expected_return must be finite")

        observations, asset_count = values.shape
        self._validate_bounds(asset_count)
        annual_mean = values.mean(axis=0) * self.config.trading_days
        variable_count = asset_count + 1 + observations

        # Variables are [asset weights, VaR threshold eta, nonnegative tail slacks].
        objective = np.zeros(variable_count, dtype=float)
        objective[asset_count] = 1.0
        objective[asset_count + 1 :] = 1.0 / (
            (1.0 - confidence_level) * observations
        )
        loss_constraints = hstack(
            (
                csr_matrix(-values),
                csr_matrix(-np.ones((observations, 1), dtype=float)),
                -eye(observations, format="csr"),
            ),
            format="csr",
        )
        upper_bounds = np.zeros(observations, dtype=float)
        if minimum_expected_return is not None:
            # -mu'w <= -target encodes the annual return floor.
            target_constraint = csr_matrix(
                np.concatenate(
                    (-annual_mean, np.zeros(observations + 1, dtype=float))
                )[None, :]
            )
            loss_constraints = vstack(
                (loss_constraints, target_constraint), format="csr"
            )
            upper_bounds = np.append(upper_bounds, -minimum_expected_return)

        full_investment = csr_matrix(
            np.concatenate((np.ones(asset_count), np.zeros(observations + 1)))[None, :]
        )
        weight_bounds = self._weight_bounds()
        bounds = [weight_bounds] * asset_count + [(None, None)] + [(0.0, None)] * observations

        try:
            solution = linprog(
                objective,
                A_ub=loss_constraints,
                b_ub=upper_bounds,
                A_eq=full_investment,
                b_eq=np.array([1.0]),
                bounds=bounds,
                method="highs",
            )
        except (ValueError, RuntimeError) as exc:
            raise OptimizationError(f"CVaR solver failed: {exc}") from exc
        if not solution.success or solution.x is None:
            reason = solution.message or "unknown solver failure"
            raise OptimizationError(f"CVaR optimization is infeasible: {reason}")

        weights = np.asarray(solution.x[:asset_count], dtype=float)
        if not self._is_feasible(weights, annual_mean, minimum_expected_return):
            raise OptimizationError("CVaR solver returned weights outside the requested constraints")

        portfolio_returns = values @ weights
        losses = -portfolio_returns
        threshold = float(solution.x[asset_count])
        cvar = float(
            threshold
            + np.maximum(losses - threshold, 0.0).mean() / (1.0 - confidence_level)
        )
        var = float(np.quantile(losses, confidence_level, method="linear"))
        expected_return = float(portfolio_returns.mean() * self.config.trading_days)
        volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(self.config.trading_days))
        sharpe = (
            (expected_return - self.config.risk_free_rate) / volatility
            if volatility > 0
            else float("nan")
        )
        portfolio = Portfolio(
            label="Minimum CVaR",
            weights=pd.Series(weights, index=list(names), dtype=float),
            expected_return=expected_return,
            volatility=volatility,
            sharpe=float(sharpe),
            note=f"Historical {confidence_level:.1%} CVaR, estimated from {observations:,} returns",
        )
        return CVaRResult(
            portfolio=portfolio,
            confidence_level=confidence_level,
            minimum_expected_return=minimum_expected_return,
            value_at_risk=var,
            expected_shortfall=cvar,
            threshold_loss=threshold,
            observations=observations,
            daily_returns=pd.Series(portfolio_returns, index=index, name="Minimum CVaR"),
            solver_message=str(solution.message),
        )

    def _validate_returns(
        self, returns: pd.DataFrame
    ) -> tuple[FloatArray, tuple[str, ...], pd.Index]:
        if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
            raise OptimizationError("returns must contain at least two asset columns")
        if returns.columns.has_duplicates:
            raise OptimizationError("returns columns must be unique")
        try:
            values = returns.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise OptimizationError("returns must contain only numeric values") from exc
        if values.ndim != 2 or not np.isfinite(values).all():
            raise OptimizationError("returns contain NaN or infinite values")
        observations, asset_count = values.shape
        if observations < max(30, asset_count + 2):
            raise OptimizationError(
                f"need at least max(30, N+2) observations; got {observations} for {asset_count} assets"
            )
        self._validate_bounds(asset_count)
        names = tuple(str(name) for name in returns.columns)
        if len(set(names)) != len(names):
            raise OptimizationError("returns columns must have unique display names")
        return values, names, returns.index

    @staticmethod
    def _validate_confidence(confidence_level: float) -> None:
        if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise OptimizationError("confidence_level must be strictly between 0 and 1")

    def _return_range(self, values: FloatArray) -> tuple[float, float]:
        annual_mean = values.mean(axis=0) * self.config.trading_days
        equality = np.ones((1, len(annual_mean)), dtype=float)
        result_low = linprog(
            annual_mean,
            A_eq=equality,
            b_eq=[1.0],
            bounds=self._weight_bounds(),
            method="highs",
        )
        result_high = linprog(
            -annual_mean,
            A_eq=equality,
            b_eq=[1.0],
            bounds=self._weight_bounds(),
            method="highs",
        )
        if not result_low.success or not result_high.success:
            message = result_low.message if not result_low.success else result_high.message
            raise OptimizationError(f"Cannot determine feasible return range: {message}")
        return float(result_low.fun), float(-result_high.fun)

    def _weight_bounds(self) -> tuple[float | None, float | None]:
        lower = self.config.min_weight
        upper = self.config.max_weight
        return (
            None if np.isneginf(lower) else float(lower),
            None if np.isposinf(upper) else float(upper),
        )

    def _validate_bounds(self, asset_count: int) -> None:
        lower, upper = self.config.min_weight, self.config.max_weight
        if np.isnan(lower) or np.isnan(upper) or asset_count * lower > 1.0 + 1e-8 or asset_count * upper < 1.0 - 1e-8:
            raise OptimizationError(
                f"infeasible weight bounds for {asset_count} assets: "
                f"min={lower}, max={upper}"
            )

    def _is_feasible(
        self,
        weights: FloatArray,
        annual_mean: FloatArray,
        minimum_expected_return: float | None,
    ) -> bool:
        lower, upper = self.config.min_weight, self.config.max_weight
        within_bounds = (weights >= lower - 1e-7).all() and (weights <= upper + 1e-7).all()
        meets_return = (
            minimum_expected_return is None
            or float(weights @ annual_mean) >= minimum_expected_return - 1e-6
        )
        return bool(np.isfinite(weights).all() and abs(weights.sum() - 1.0) < 1e-6
                    and within_bounds and meets_return)

