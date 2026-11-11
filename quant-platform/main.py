"""Command-line entry point: runs the platform's modules end to end on live data.

Everything is estimated / selected on a FORMATION period and judged on a later
OUT-OF-SAMPLE period (default: formation through 2024, judged on 2025-2026).

Examples:
    python main.py                                  # all demos, formation to 2024, test 2025+
    python main.py --only pairs --oos-start 2025-01-01
    python main.py --only optimizer --risk-free 0.04
    python main.py --only risk                      # VaR / ES, drawdowns, backtests, stress tests
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtester import BacktestResult, PairsBacktester, screen_pairs
from data_engine import DataEngine, DataEngineConfig
from optimizer import OptimizerConfig, PortfolioOptimizer
from risk_engine import RiskConfig, RiskEngine, RiskReport, drawdown_analysis, macro_factor_frame

PAIRS_UNIVERSE = ["KO", "PEP", "XOM", "CVX"]
OPTIMIZER_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "XLE", "XLV"]
OUTPUT_DIR = Path("data/backtest_output")


# --------------------------------------------------------------------------- #
# Reusable study functions (also exercised by the tests with synthetic prices)
# --------------------------------------------------------------------------- #
def run_pairs_backtest(prices: pd.DataFrame, formation_end: str) -> BacktestResult:
    """Pick the pair using ONLY data up to ``formation_end``, then backtest the full history.

    Choosing the pair from the whole sample would leak the future into the choice.
    Judge results with ``result.performance(start=...)`` on the period after ``formation_end``.
    """
    table = screen_pairs(prices.loc[:formation_end])
    print("\nPair screen on formation period (lower p = stronger evidence):")
    print(table.round(4).to_string(index=False))
    best = table.iloc[0]
    return PairsBacktester().run(prices, y=str(best["y"]), x=str(best["x"]))


def run_optimizer_study(
    prices: pd.DataFrame, formation_end: str, oos_start: str, risk_free: float
) -> dict[str, object]:
    """Estimate on the formation period; compare in-sample vs out-of-sample; walk forward."""
    opt = PortfolioOptimizer(
        OptimizerConfig(risk_free_rate=risk_free, max_weight=0.40, mean_estimator="bayes_stein")
    )
    result = opt.optimize(prices.loc[:formation_end])
    print("\n" + result.summary())

    returns = opt.simple_returns(prices)
    in_sample, out_sample = returns.loc[:formation_end], returns.loc[oos_start:]
    rows = {}
    for p in (result.max_sharpe, result.gmv, result.equal_weight):
        rows[p.label] = {
            "Sharpe in-sample": opt.evaluate(p.weights, in_sample)["sharpe"],
            "Sharpe out-of-sample": opt.evaluate(p.weights, out_sample)["sharpe"],
            "OOS return": opt.evaluate(p.weights, out_sample)["ann_return"],
            "OOS volatility": opt.evaluate(p.weights, out_sample)["ann_volatility"],
        }
    table = pd.DataFrame(rows).T
    print(f"\nFixed weights chosen on data through {formation_end}, judged from {oos_start}:")
    print(table.round(3).to_string())

    wf = opt.walk_forward(prices, lookback=252, rebalance_every=21, cost_bps=5.0)
    wf_metrics = wf.metrics(start=oos_start)
    print(f"\nWalk-forward re-optimization (1y lookback, monthly, 5 bps costs), from {oos_start}:")
    print(wf_metrics.round(3).to_string())
    return {"result": result, "in_vs_out": table, "walk_forward": wf, "wf_metrics": wf_metrics}


def run_risk_study(
    prices: pd.DataFrame, weights: pd.Series, macro_features: pd.DataFrame | None, oos_start: str,
    portfolio_value: float = 1_000_000.0,
) -> RiskReport:
    """Full-history risk report, then VaR/ES and drawdown for the out-of-sample period alone."""
    engine = RiskEngine(RiskConfig(portfolio_value=portfolio_value))
    factors = macro_factor_frame(macro_features) if macro_features is not None else None
    report = engine.analyze(prices, weights, factors=factors)
    print("\n=== RISK: FULL HISTORY ===\n" + report.summary())
    oos_returns = report.returns.loc[oos_start:]
    if len(oos_returns) >= 60:
        print(f"\n=== RISK: OUT-OF-SAMPLE ({oos_start} onward, {len(oos_returns)} days) ===")
        print(engine.var_table(oos_returns).round(4).to_string())
        dd = drawdown_analysis(oos_returns)
        print(f"Max drawdown {dd.max_drawdown:.2%}; longest underwater {dd.longest_underwater_bars} days")
    return report


def estimate_risk_free(engine: DataEngine, formation_end: str) -> float:
    """Average 3-month T-bill yield over the formation period (percent -> decimal)."""
    try:
        macro = engine.fetch_macro()
        return float(macro["ust_3m"].loc[:formation_end].mean() / 100.0)
    except Exception as exc:  # noqa: BLE001 - macro data is a convenience, not a requirement
        logging.getLogger(__name__).warning("Could not estimate risk-free rate (%s); using 2%%", exc)
        return 0.02


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def demo_data(engine: DataEngine) -> None:
    prices = engine.get_price_panel(PAIRS_UNIVERSE)
    print("\nLatest prices:\n", prices.tail().round(2))
    print("\nData quality report:\n", engine.last_report)
    print("\nCorrelation of daily log returns:\n", engine.get_returns(prices).corr().round(3))
    macro = engine.build_macro_features(engine.fetch_macro())
    print("\nMacro snapshot:\n", macro.tail(3).round(3).T)
    features = engine.get_features(["KO"])["KO"]
    print("\nKO technical features:\n", features.dropna().tail(3).round(3).T)


def demo_pairs(engine: DataEngine, formation_end: str, oos_start: str) -> None:
    prices = engine.get_price_panel(PAIRS_UNIVERSE)
    result = run_pairs_backtest(prices, formation_end)
    print("\n=== PAIRS: FULL SAMPLE ===\n" + result.report())
    print(f"\n=== PAIRS: OUT OF SAMPLE ({oos_start} onward; pair chosen before it) ===")
    print(result.report(result.performance(start=oos_start)))
    result.to_csv(OUTPUT_DIR / "pairs")


def demo_optimizer(engine: DataEngine, formation_end: str, oos_start: str, rf: float | None) -> None:
    prices = engine.get_price_panel(OPTIMIZER_UNIVERSE)
    risk_free = estimate_risk_free(engine, formation_end) if rf is None else rf
    print(f"\nRisk-free rate used: {risk_free:.2%}")
    study = run_optimizer_study(prices, formation_end, oos_start, risk_free)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        study["result"].figure().write_html(str(OUTPUT_DIR / "efficient_frontier.html"))  # type: ignore[attr-defined]
        print(f"\nInteractive frontier chart saved to {OUTPUT_DIR / 'efficient_frontier.html'}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n(Chart skipped: {exc})")


def demo_risk(engine: DataEngine, formation_end: str, oos_start: str, rf: float | None) -> None:
    prices = engine.get_price_panel(OPTIMIZER_UNIVERSE)
    risk_free = estimate_risk_free(engine, formation_end) if rf is None else rf
    opt = PortfolioOptimizer(OptimizerConfig(risk_free_rate=risk_free, max_weight=0.40, mean_estimator="bayes_stein"))
    weights = opt.optimize(prices.loc[:formation_end]).max_sharpe.weights
    print("\nMax-Sharpe weights (chosen on the formation period):\n", weights.round(3).to_string())
    try:
        macro: pd.DataFrame | None = engine.build_macro_features(engine.fetch_macro())
    except Exception as exc:  # noqa: BLE001 - factor stress is optional
        print(f"(Macro data unavailable, factor stress skipped: {exc})")
        macro = None
    run_risk_study(prices, weights, macro, oos_start)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--only", choices=["data", "pairs", "optimizer", "risk"], help="run a single demo")
    parser.add_argument("--start", default="2018-01-01", help="first date of history to download")
    parser.add_argument("--formation-end", default="2024-12-31", help="last date used to choose/estimate")
    parser.add_argument("--oos-start", default="2025-01-01", help="first date of the out-of-sample test")
    parser.add_argument("--risk-free", type=float, default=None, help="annual rate, e.g. 0.04 (default: auto)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = DataEngine(DataEngineConfig(start=args.start))
    if args.only in (None, "data"):
        demo_data(engine)
    if args.only in (None, "pairs"):
        demo_pairs(engine, args.formation_end, args.oos_start)
    if args.only in (None, "optimizer"):
        demo_optimizer(engine, args.formation_end, args.oos_start, args.risk_free)
    if args.only in (None, "risk"):
        demo_risk(engine, args.formation_end, args.oos_start, args.risk_free)


if __name__ == "__main__":
    main()
