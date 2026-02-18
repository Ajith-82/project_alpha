from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import structlog
import os

from classes.backtesting.engine import BacktestEngine
from backtesting import Backtest

logger = structlog.get_logger()

@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    start_date: str
    end_date: str
    duration_days: int
    exposure_time_pct: float
    equity_final: float
    equity_peak: float
    return_pct: float
    buy_hold_return_pct: float
    return_ann_pct: float
    volatility_ann_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: int
    trade_count: int
    win_rate_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_trade_pct: float
    max_trade_duration_days: int
    avg_trade_duration_days: int
    profit_factor: float
    expectancy_pct: float
    sqn: float

class BacktestPerformance:
    """
    Utilities for analyzing and reporting backtest results.
    """
    @staticmethod
    def extract_metrics(stats: pd.Series, ticker: str, strategy_name: str) -> BacktestResult:
        """
        Convert backtesting.py stats Series into a structured dataclass.
        """
        # Helper to safely get value or 0.0
        def get(key, default=0.0):
            val = stats.get(key, default)
            return val if pd.notnull(val) else default

        return BacktestResult(
            ticker=ticker,
            strategy=strategy_name,
            start_date=str(get('Start')),
            end_date=str(get('End')),
            duration_days=int(get('Duration').days) if hasattr(get('Duration'), 'days') else 0,
            exposure_time_pct=get('Exposure Time [%]'),
            equity_final=get('Equity Final [$]'),
            equity_peak=get('Equity Peak [$]'),
            return_pct=get('Return [%]'),
            buy_hold_return_pct=get('Buy & Hold Return [%]'),
            return_ann_pct=get('Return (Ann.) [%]'),
            volatility_ann_pct=get('Volatility (Ann.) [%]'),
            sharpe_ratio=get('Sharpe Ratio'),
            sortino_ratio=get('Sortino Ratio'),
            calmar_ratio=get('Calmar Ratio'),
            max_drawdown_pct=get('Max. Drawdown [%]'),
            avg_drawdown_pct=get('Avg. Drawdown [%]'),
            max_drawdown_duration_days=int(get('Max. Drawdown Duration').days) if hasattr(get('Max. Drawdown Duration'), 'days') else 0,
            avg_drawdown_duration_days=int(get('Avg. Drawdown Duration').days) if hasattr(get('Avg. Drawdown Duration'), 'days') else 0,
            trade_count=int(get('# Trades')),
            win_rate_pct=get('Win Rate [%]'),
            best_trade_pct=get('Best Trade [%]'),
            worst_trade_pct=get('Worst Trade [%]'),
            avg_trade_pct=get('Avg. Trade [%]'),
            max_trade_duration_days=int(get('Max. Trade Duration').days) if hasattr(get('Max. Trade Duration'), 'days') else 0,
            avg_trade_duration_days=int(get('Avg. Trade Duration').days) if hasattr(get('Avg. Trade Duration'), 'days') else 0,
            profit_factor=get('Profit Factor'),
            expectancy_pct=get('Expectancy [%]'),
            sqn=get('SQN'),
        )

    @staticmethod
    def generate_report(bt: Backtest, filename: str = "backtest_report.html", open_browser: bool = False):
        """
        Generate interactive HTML report.
        """
        try:
             # Ensure directory exists
             output_dir = os.path.dirname(filename)
             if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
             
             bt.plot(filename=filename, open_browser=open_browser)
             logger.info(f"Backtest report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to generate backtest report: {e}")

