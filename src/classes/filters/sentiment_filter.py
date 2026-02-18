import structlog
from typing import List, Dict, Any, Optional
from transformers import pipeline

logger = structlog.get_logger()

class SentimentFilter:
    """
    Filters stocks based on news sentiment using FinBERT.
    """
    
    _classifier = None  # Singleton model instance
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        
    def _load_model(self):
        """Lazy load the model if not already loaded."""
        if SentimentFilter._classifier is None:
            try:
                logger.info(f"Loading sentiment model: {self.model_name}")
                SentimentFilter._classifier = pipeline("sentiment-analysis", model=self.model_name)
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                SentimentFilter._classifier = None

    @property
    def classifier(self):
        if SentimentFilter._classifier is None:
            self._load_model()
        return SentimentFilter._classifier

    def analyze_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of a list of headlines.
        
        Returns:
             Dict with 'score' (-1 to +1) and 'label' (positive/negative/neutral)
        """
        if not self.classifier or not headlines:
            return {"score": 0.0, "label": "neutral", "details": []}
            
        try:
            results = self.classifier(headlines)
            
            # Calculate aggregate score
            # positive=1, negative=-1, neutral=0
            total_score = 0.0
            details = []
            
            for headline, res in zip(headlines, results):
                label = res['label']
                score = res['score']
                
                val = 0.0
                if label == 'positive':
                    val = score
                elif label == 'negative':
                    val = -score
                # neutral is 0.0
                
                total_score += val
                details.append({"headline": headline, "label": label, "score": round(score, 4)})
                
            avg_score = total_score / len(headlines) if headlines else 0.0
            
            final_label = "neutral"
            if avg_score > 0.15:  # Lowered threshold slightly
                final_label = "positive"
            elif avg_score < -0.15:
                final_label = "negative"
                
            return {
                "score": round(avg_score, 2),
                "label": final_label,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"score": 0.0, "label": "error", "details": str(e)}
