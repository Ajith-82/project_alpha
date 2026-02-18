import pytest
from classes.screeners.consensus import ConsensusEngine, ConsensusResult
from classes.screeners.base import ScreenerResult, Signal

@pytest.fixture
def engine():
    return ConsensusEngine()

def test_consensus_score_calculation(engine):
    """Test basic weighted score calculation."""
    # Breakout (0.4) + Trend (0.3) + Volatility (0.1) + Fund (0.1) + Sent (0.1)
    # Total = 1.0
    
    results = {
        "breakout": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.8),
        "trend": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.6),
    }
    
    # breakout: 0.8 * 0.4 = 0.32
    # trend: 0.6 * 0.3 = 0.18
    # Synergy bonus apply? Yes, both > 0
    # Base = 0.50 + 0.1 bonus = 0.60
    
    consensus = engine.calculate_score("TEST", results)
    
    assert consensus.score == 0.60
    assert consensus.primary_signal == Signal.BUY
    assert consensus.recommendation == "BUY"

def test_consensus_synergy_boost(engine):
    """Test that having both Breakout and Trend boosts the score."""
    results = {
        "breakout": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.5), # 0.2
        "trend": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.5),    # 0.15
    }
    # Base = 0.35 + 0.1 bonus = 0.45
    
    consensus = engine.calculate_score("TEST", results)
    assert consensus.signals["synergy_bonus"] == 0.1
    assert consensus.score == 0.45

def test_consensus_single_signal(engine):
    """Test score with only one signal."""
    results = {
        "breakout": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.9), # 0.36
    }
    # Base = 0.36, No bonus
    
    consensus = engine.calculate_score("TEST", results)
    assert "synergy_bonus" not in consensus.signals
    assert consensus.score == 0.36
    assert consensus.primary_signal == Signal.HOLD # Score < 0.5

def test_consensus_with_filters(engine):
    """Test consensus with fundamental/sentiment filters."""
    results = {
        "breakout": ScreenerResult(ticker="TEST", signal=Signal.BUY, confidence=0.8), # 0.32
    }
    filters = {
        "fundamental": 1.0, # 0.1
        "sentiment": 0.5    # 0.05
    }
    # Base = 0.32 + 0.1 + 0.05 = 0.47
    
    consensus = engine.calculate_score("TEST", results, filters)
    assert consensus.score == 0.47
