import pandas as pd
import numpy as np
from typing import List, Dict, Type, Generator, Tuple
from datetime import timedelta
from structlog import get_logger

from classes.backtesting.engine import BacktestEngine, ProjectAlphaStrategy
from classes.screeners.base import BaseScreener

logger = get_logger()

class WalkForwardValidator:
    """
    Performs Walk-Forward Validation using Anchored Expanding Windows.
    Wrapper around BacktestEngine to run sequential tests.
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        train_period_days: int = 365, 
        test_period_days: int = 90,
        initial_capital: float = 10000
    ):
        """
        Initialize the validator.

        Args:
            data: DataFrame with OHLCV data.
            train_period_days: Initial size of the training window (days).
            test_period_days: Size of the testing window (days).
            initial_capital: Capital for each backtest run.
        """
        self.data = data
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.initial_capital = initial_capital
        
        # Ensure DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Results storage
        self.results = []


    def generate_windows(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]], None, None]:
        """
        Yields (train_df, test_df, (test_start, test_end)) tuples.
        Strategy: Anchored Expanding Window.
        - Start is fixed at data.index[0]
        - Train End moves forward by test_period
        - Test Start = Train End
        - Test End = Test Start + test_period
        """
        if self.data.empty:
            return

        start_date = self.data.index[0]
        max_date = self.data.index[-1]
        
        # Initial window end
        current_train_end = start_date + self.train_period
        
        window_idx = 1
        
        while current_train_end < max_date:
            test_start = current_train_end
            test_end = test_start + self.test_period
            
            if test_end > max_date:
                test_end = max_date
            
            # Slice data
            train_mask = (self.data.index >= start_date) & (self.data.index < current_train_end)
            test_mask = (self.data.index >= test_start) & (self.data.index <= test_end)
            
            train_df = self.data.loc[train_mask]
            test_df = self.data.loc[test_mask]
            
            # Ensure enough data
            if len(test_df) > 5: # Minimal checks
                yield train_df, test_df, (test_start, test_end)
            
            # Stop if we reached the end
            if test_end == max_date:
                break
                
            # Expand window
            current_train_end = test_end
            window_idx += 1

    def validate(self, screener_cls: Type[BaseScreener]) -> List[Dict]:
        """
        Run the validation loop.
        
        Args:
            screener_cls: The screener class to test.
            
        Returns:
            List of result dictionaries for each window.
        """
        logger.info("Starting Walk-Forward Validation", 
                    screener=screener_cls.__name__,
                    train_days=self.train_period.days,
                    test_days=self.test_period.days)
        
        self.results = []
        
        for i, (train_df, test_df, (test_start, test_end)) in enumerate(self.generate_windows()):
            logger.info(f"Processing Window {i+1}", 
                        test_start=test_start.date(), 
                        test_end=test_end.date(),
                        train_rows=len(train_df),
                        test_rows=len(test_df))
            
            # Run In-Sample (IS) Backtest
            # Note: For strict WFV we would Optimize params here. 
            # Current engine doesn't support optimization yet, so we just run to get IS baseline.
            engine_is = BacktestEngine(train_df, self.initial_capital)
            _, stats_is = engine_is.run(ProjectAlphaStrategy, screener_cls)
            
            # Run Out-of-Sample (OOS) Backtest
            engine_oos = BacktestEngine(test_df, self.initial_capital)
            _, stats_oos = engine_oos.run(ProjectAlphaStrategy, screener_cls)
            
            window_result = {
                "window": i + 1,
                "test_start": test_start,
                "test_end": test_end,
                # Metrics
                "IS_Return": stats_is["Return [%]"],
                "IS_Sharpe": stats_is["Sharpe Ratio"],
                "IS_Drawdown": stats_is["Max. Drawdown [%]"],
                "OOS_Return": stats_oos["Return [%]"],
                "OOS_Sharpe": stats_oos["Sharpe Ratio"],
                "OOS_Drawdown": stats_oos["Max. Drawdown [%]"],
                # Ratios (OOS / IS) - strictly speaking signs matter, so simple ratio might be misleading if negative
                # We'll compute them in summary or reporting
            }
            self.results.append(window_result)
            
        return self.results

    def get_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of the results.
        And checks for overfitting.
        """
        if not self.results:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.results)
        
        # Calculate degradation
        # 1.0 means OOS matched IS. < 0.5 suggests overfitting.
        # Handle division by zero or negative sharpes gracefully?
        # For Sharpe:
        df["Sharpe_Degradation"] = df["OOS_Sharpe"] / df["IS_Sharpe"].replace(0, np.nan)
        
        # Determine performance
        df["Overfit_Warning"] = df["Sharpe_Degradation"] < 0.5
        
        # Overall aggregates
        mean_oos_sharpe = df["OOS_Sharpe"].mean()
        logger.info("Validation Complete", mean_oos_sharpe=mean_oos_sharpe)
        
        return df
