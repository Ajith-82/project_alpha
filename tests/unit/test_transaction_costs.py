import pytest
from src.classes.risk.transaction_costs import TransactionCosts

def test_calculate_cost_zero_commission():
    # Commission=0, Slippage=5bps, Spread=3bps -> Variable=6.5bps
    tc = TransactionCosts(commission_per_trade=0.0, slippage_bps=5.0, spread_bps=3.0)
    price = 100.0
    quantity = 100
    expected_variable = (100 * 100) * (6.5 / 10000)  # 10000 * 0.00065 = 6.5
    assert tc.calculate_cost(price, quantity) == pytest.approx(6.5)

def test_calculate_cost_with_commission():
    # Commission=20, Slippage=10bps, Spread=5bps -> Variable=12.5bps
    tc = TransactionCosts(commission_per_trade=20.0, slippage_bps=10.0, spread_bps=5.0)
    price = 500.0
    quantity = 10
    total_value = 5000.0
    expected_variable = total_value * (12.5 / 10000)  # 5000 * 0.00125 = 6.25
    assert tc.calculate_cost(price, quantity) == pytest.approx(20.0 + 6.25)

def test_defaults():
    us = TransactionCosts.us_default()
    assert us.commission_per_trade == 0.0
    
    ind = TransactionCosts.india_default()
    assert ind.commission_per_trade == 20.0
