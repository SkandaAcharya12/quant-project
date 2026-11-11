"""Module 2: statistical arbitrage (pairs trading) backtester."""
from backtester.backtester import (
    BacktestError,
    BacktestResult,
    PairsBacktestConfig,
    PairsBacktester,
    build_signal,
    engle_granger_test,
    rolling_engle_granger,
    screen_pairs,
)

__all__ = [
    "BacktestError", "BacktestResult", "PairsBacktestConfig", "PairsBacktester",
    "build_signal", "engle_granger_test", "rolling_engle_granger", "screen_pairs",
]
