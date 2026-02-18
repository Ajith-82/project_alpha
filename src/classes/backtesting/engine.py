import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import structlog
from typing import Type

from classes.backtesting.adapter import ScreenerSignalAdapter
from classes.risk.risk_manager import RiskManager
from classes.risk.transaction_costs import TransactionCosts
from classes.screeners.base import BaseScreener
from classes.screeners.breakout import BreakoutScreener
from classes.screeners.trendline import TrendlineScreener
from config.settings import settings

logger = structlog.get_logger()

class ProjectAlphaStrategy(Strategy):
    """
    Base strategy for Project Alpha backtesting.
    Integrates point-in-time signals from adapter with RiskManager logic.
    """
    risk_manager = RiskManager()
    transaction_costs = TransactionCosts.us_default() # Default, can be overridden
    adapter_class = ScreenerSignalAdapter
    screener_class = BreakoutScreener # Default
    
    def init(self):
        # Initialize screener adapter
        self.screener = self.screener_class()
        self.adapter = self.adapter_class(self.screener)
        
        # Pre-compute signals using adapter's vectorized methods if possible
        # For MVP, we use the simple vectorized proxies defined in adapter
        # Pre-compute signals using adapter's logic
        # We use a dummy ticker since the strategy runs on a single dataframe context
        # and Screeners often just need the dataframe.
        ticker = "BACKTEST" 
        self.signal = self.I(self.adapter.compute_signals, self.data.df, ticker)

        # Pre-compute ATR for dynamic stop-loss
        # Using simple ATR proxy or pandas-ta if available. 
        # Backtesting.py's self.I requires an array-like result.
        # We can implement a simple ATR helper here.
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        self.atr = self.I(self._compute_atr, high, low, close, settings.atr_period)

    def _compute_atr(self, high, low, close, period=14):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        return self._pandas_ta_atr(high, low, close, period)

    def _pandas_ta_atr(self, high, low, close, length=14):
        # Manual ATR calculation if numpy arrays are passed or pandas-ta not available in scope
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=length).mean().bfill().fillna(0).values

    def next(self):
        price = self.data.Close[-1]
        
        # Entry Logic
        if self.signal[-1] == 1 and not self.position:
            # Calculate Stop Loss
            stop_price = self.risk_manager.calculate_stop_loss(price, self.atr[-1], "long")
            
            # Calculate Position Size
            # Equity is available via self.equity
            # We assume single position for this strategy test context
            size = self.risk_manager.calculate_position_size(self.equity, price, stop_price)
            
            if size > 0:
                # Place Buy Order with Stop Loss
                self.buy(size=size, sl=stop_price)
                
        # Exit Logic
        elif self.signal[-1] == -1 and self.position:
             self.position.close()

class BacktestEngine:
    """
    Engine to run backtests for a specific ticker and strategy.
    """
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000, commission: float = 0.0):
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, strategy_class: Type[Strategy] = ProjectAlphaStrategy, screener_cls: Type[BaseScreener] = BreakoutScreener):
        """
        Run the backtest.
        """
        # Configure strategy class with specific screener
        # Using a subclass to avoid modifying base class state if reused
        # Capture closure variable explicitly
        target_screener = screener_cls
        
        class ConfiguredStrategy(strategy_class):
            screener_class = target_screener
        
        bt = Backtest(
            self.data, 
            ConfiguredStrategy, 
            cash=self.initial_capital, 
            commission=self.commission,
            exclusive_orders=True 
        )
        
        stats = bt.run()
        return bt, stats
