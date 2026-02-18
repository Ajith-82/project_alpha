import pytest
from src.config.settings import Settings

def test_settings_defaults():
    """Verify default values are loaded correctly."""
    settings = Settings()
    assert settings.market == "us"
    assert settings.risk_per_trade == 0.01
    assert settings.atr_multiplier == 2.0
    assert settings.smtp_port == 587

def test_settings_env_override(monkeypatch):
    """Verify environment variables override defaults."""
    monkeypatch.setenv("PA_MARKET", "india")
    monkeypatch.setenv("PA_RISK_PER_TRADE", "0.02")
    monkeypatch.setenv("PA_ATR_MULTIPLIER", "3.0")
    
    settings = Settings()
    assert settings.market == "india"
    assert settings.risk_per_trade == 0.02
    assert settings.atr_multiplier == 3.0

def test_settings_type_coercion(monkeypatch):
    """Verify string env vars are coerced to correct types."""
    monkeypatch.setenv("PA_MAX_POSITIONS", "20")
    monkeypatch.setenv("PA_LEARNING_RATE", "0.05")
    
    settings = Settings()
    assert settings.max_positions == 20
    assert isinstance(settings.max_positions, int)
    assert settings.learning_rate == 0.05
    assert isinstance(settings.learning_rate, float)
