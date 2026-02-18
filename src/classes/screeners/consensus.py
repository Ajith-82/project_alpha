from typing import Dict, List, Optional
from dataclasses import dataclass
from classes.screeners.base import ScreenerResult, Signal
from config.settings import settings

@dataclass
class ConsensusResult:
    ticker: str
    score: float  # 0.0 to 1.0
    signals: Dict[str, float]  # contributing signals and their weights/scores
    primary_signal: Signal
    recommendation: str

class ConsensusEngine:
    """
    Aggregates signals from multiple screeners and filters to produce a single confidence score.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or settings.consensus_weights.copy()
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def calculate_score(self, ticker: str, screener_results: Dict[str, ScreenerResult], 
                       filter_results: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """
        Calculate consensus score for a single ticker.
        
        Args:
            ticker: Ticker symbol
            screener_results: Dict mapping screener name to ScreenerResult
            filter_results: Optional dict of filter scores (0.0 to 1.0)
            
        Returns:
            ConsensusResult object
        """
        total_score = 0.0
        details = {}
        
        # Process Screener Signals
        for name, weight in self.weights.items():
            score = 0.0
            
            # Check tech screeners
            if name in screener_results:
                res = screener_results[name]
                if res.signal == Signal.BUY:
                    score = res.confidence
                elif res.signal == Signal.SELL:
                    score = -res.confidence  # Penalize for sell signals? 
                    # For now, let's treat sell as negative score, but we want 0-1 range.
                    # Simple approach: Only add to score if BUY.
                    score = 0.0 
            
            # Check filters
            elif filter_results and name in filter_results:
                score = filter_results[name]
                
            total_score += score * weight
            details[name] = round(score, 2)
            
        # Synergy Bonus: If Breakout AND Trend are both present, boost score
        if "breakout" in details and details["breakout"] > 0 and \
           "trend" in details and details["trend"] > 0:
            synergy_bonus = 0.1
            total_score = min(1.0, total_score + synergy_bonus)
            details["synergy_bonus"] = synergy_bonus
            
        # Determine final signal
        primary_signal = Signal.HOLD
        rec = "HOLD"
        
        if total_score >= 0.7:
            primary_signal = Signal.BUY
            rec = "STRONG BUY"
        elif total_score >= 0.5:
            primary_signal = Signal.BUY
            rec = "BUY"
            
        return ConsensusResult(
            ticker=ticker,
            score=round(total_score, 2),
            signals=details,
            primary_signal=primary_signal,
            recommendation=rec
        )
