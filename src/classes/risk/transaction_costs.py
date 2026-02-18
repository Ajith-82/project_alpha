from dataclasses import dataclass
from typing import Optional

@dataclass
class TransactionCosts:
    """
    Model for estimating transaction costs including commission, slippage, and spread.
    
    Attributes:
        commission_per_trade (float): Fixed commission fee per trade (buy or sell).
        slippage_bps (float): Estimated slippage in basis points (1 bps = 0.01%).
        spread_bps (float): Average bid-ask spread in basis points.
    """
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0
    spread_bps: float = 3.0

    def calculate_cost(self, price: float, quantity: int) -> float:
        """
        Calculate total transaction cost for a single leg (entry or exit).
        
        Cost = Commission + (Price * Quantity * (Slippage + 0.5 * Spread) / 10000)
        
        Args:
            price: Execution price per share.
            quantity: Number of shares.
            
        Returns:
            Total estimated cost in currency units.
        """
        total_value = price * quantity
        variable_bps = self.slippage_bps + (self.spread_bps / 2)
        variable_cost = total_value * (variable_bps / 10000)
        return self.commission_per_trade + variable_cost

    @classmethod
    def us_default(cls):
        """Default costs for US markets (zero commission, standard liquidity)."""
        return cls(commission_per_trade=0.0, slippage_bps=5.0, spread_bps=3.0)

    @classmethod
    def india_default(cls):
        """Default costs for Indian markets (brokerage fees, lower liquidity)."""
        return cls(commission_per_trade=20.0, slippage_bps=10.0, spread_bps=5.0)
