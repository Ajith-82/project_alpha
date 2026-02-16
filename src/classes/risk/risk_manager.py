from dataclasses import dataclass
from typing import Optional, Literal
import math

from config.settings import settings

@dataclass
class OrderValidation:
    valid: bool
    reason: Optional[str] = None

class RiskManager:
    """
    Manages trading risk through position sizing, stop-loss calculation, and exposure limits.
    """
    def __init__(self):
        self.risk_per_trade = settings.risk_per_trade
        self.atr_multiplier = settings.atr_multiplier
        self.max_positions = settings.max_positions
        self.max_portfolio_exposure = settings.max_portfolio_exposure
        self.trailing_stop = settings.trailing_stop

    def calculate_stop_loss(self, entry_price: float, atr: float, direction: Literal["long", "short"] = "long") -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Execution price.
            atr: Average True Range value.
            direction: 'long' or 'short'.
            
        Returns:
            Stop-loss price.
        """
        if direction == "long":
            return entry_price - (atr * self.atr_multiplier)
        else:
            return entry_price + (atr * self.atr_multiplier)

    def calculate_position_size(self, account_size: float, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size (number of shares) based on risk per trade.
        
        Risk Amount = Account Size * Risk Per Trade %
        Risk Per Share = |Entry - Stop Loss|
        Shares = Risk Amount / Risk Per Share
        
        Args:
            account_size: Total account equity.
            entry_price: Execution price.
            stop_loss: Stop protection price.
            
        Returns:
            Number of shares (integer).
        """
        risk_amount = account_size * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        shares = math.floor(risk_amount / risk_per_share)
        return shares

    def validate_order(self, current_positions: int, current_exposure: float, order_value: float, account_size: float) -> OrderValidation:
        """
        Validate if an order can be placed within risk limits.
        
        Args:
            current_positions: Number of currently open positions.
            current_exposure: Total value of current positions.
            order_value: Value of the new order (Price * Quantity).
            account_size: Total account equity.
            
        Returns:
            OrderValidation object.
        """
        if current_positions >= self.max_positions:
            return OrderValidation(False, f"Max positions reached ({self.max_positions})")
        
        projected_exposure = (current_exposure + order_value) / account_size
        if projected_exposure > self.max_portfolio_exposure:
            return OrderValidation(False, f"Max portfolio exposure exceeded ({projected_exposure:.2%} > {self.max_portfolio_exposure:.2%})")
            
        return OrderValidation(True)
