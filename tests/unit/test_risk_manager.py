import pytest
import math
from unittest.mock import patch, MagicMock
from src.classes.risk.risk_manager import RiskManager, OrderValidation

# Mock settings to ensure consistent test environment
@pytest.fixture
def mock_settings():
    with patch("src.classes.risk.risk_manager.settings") as mock:
        mock.risk_per_trade = 0.01
        mock.atr_multiplier = 2.0
        mock.max_positions = 5
        mock.max_portfolio_exposure = 1.0
        mock.trailing_stop = True
        yield mock

@pytest.fixture
def risk_manager(mock_settings):
    # settings are imported in risk_manager, so patching it effectively controls defaults
    # Re-instantiating inside test ensures fresh values
    return RiskManager()

def test_calculate_stop_loss_long(risk_manager):
    """Test stop loss calculation for long positions."""
    entry_price = 100.0
    atr = 2.0
    # Long SL = Entry - (ATR * Multiplier) = 100 - (2 * 2) = 96
    # Note: risk_manager reads settings on init
    sl = risk_manager.calculate_stop_loss(entry_price, atr, "long")
    assert sl == 96.0

def test_calculate_stop_loss_short(risk_manager):
    """Test stop loss calculation for short positions."""
    entry_price = 100.0
    atr = 2.0
    # Short SL = Entry + (ATR * Multiplier) = 100 + (2 * 2) = 104
    sl = risk_manager.calculate_stop_loss(entry_price, atr, "short")
    assert sl == 104.0

def test_calculate_position_size(risk_manager):
    """Test position sizing calculation."""
    account_size = 10000.0
    entry_price = 100.0
    stop_loss = 95.0 # Risk per share = 5.0
    
    # Risk Amount = 10000 * 0.01 = 100.0
    # Shares = floor(100 / 5) = 20
    shares = risk_manager.calculate_position_size(account_size, entry_price, stop_loss)
    assert shares == 20

def test_calculate_position_size_zero_risk(risk_manager):
    """Test position sizing when risk per share is zero (or very close)."""
    account_size = 10000.0
    entry_price = 100.0
    stop_loss = 100.0 # Risk per share = 0
    
    shares = risk_manager.calculate_position_size(account_size, entry_price, stop_loss)
    assert shares == 0

def test_calculate_position_size_small_risk(risk_manager):
    """Test position sizing with very tight stop."""
    account_size = 10000.0
    entry_price = 100.0
    stop_loss = 99.9 # Risk per share = 0.1
    
    # Risk Amount = 100.0
    # Shares = floor(100 / 0.1) = 1000
    shares = risk_manager.calculate_position_size(account_size, entry_price, stop_loss)
    assert shares == 1000

def test_validate_order_success(risk_manager):
    """Test valid order validation."""
    validation = risk_manager.validate_order(
        current_positions=2,
        current_exposure=5000,
        order_value=1000,
        account_size=10000
    )
    assert validation.valid is True
    assert validation.reason is None

def test_validate_order_max_positions(mock_settings):
    """Test validation fails when max positions reached."""
    mock_settings.max_positions = 2
    # Re-init to pick up new setting
    rm = RiskManager() 
    
    # Current 2 (max 2) -> Fail
    validation = rm.validate_order(
        current_positions=2,
        current_exposure=5000,
        order_value=1000,
        account_size=10000
    )
    assert validation.valid is False
    assert "Max positions reached" in validation.reason

def test_validate_order_max_exposure(mock_settings):
    """Test validation fails when max exposure exceeded."""
    mock_settings.max_portfolio_exposure = 0.5 # 50% max
    rm = RiskManager()
    
    # Current 4000 + Order 2000 = 6000 (60%) > 50%
    validation = rm.validate_order(
        current_positions=2,
        current_exposure=4000,
        order_value=2000,
        account_size=10000
    )
    assert validation.valid is False
    assert "Max portfolio exposure exceeded" in validation.reason
