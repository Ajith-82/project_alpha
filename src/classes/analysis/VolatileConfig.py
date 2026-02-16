"""
Volatile Configuration Module

Provides configuration dataclasses for the Volatile analysis layer.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


from config.settings import settings

@dataclass
class RatingThresholds:
    """Thresholds for stock rating based on z-scores."""
    highly_below_trend: float = 3.0
    below_trend: float = 2.0
    along_trend: float = -2.0
    above_trend: float = -3.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to legacy format for backward compatibility."""
        return {
            "HIGHLY BELOW TREND": self.highly_below_trend,
            "BELOW TREND": self.below_trend,
            "ALONG TREND": self.along_trend,
            "ABOVE TREND": self.above_trend,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = settings.learning_rate
    correlation_steps: int = settings.correlation_steps  # Steps for correlation model
    trend_steps: int = settings.trend_steps  # Steps for trend model
    
    # Polynomial orders
    order_correlation: int = settings.correlation_order  # High-frequency patterns
    order_trend: int = settings.model_order  # Quadratic trend fitting


@dataclass
class VolatileConfig:
    """
    Main configuration for Volatile analysis.
    
    Attributes:
        horizon: Days ahead for prediction (default: 5)
        training: Model training configuration
        thresholds: Rating thresholds
        plot_losses: Show training loss plots
        verbose: Enable verbose output
    """
    horizon: int = 5
    training: TrainingConfig = field(default_factory=TrainingConfig)
    thresholds: RatingThresholds = field(default_factory=RatingThresholds)
    
    # Options
    plot_losses: bool = False
    verbose: bool = True
    save_model: Optional[str] = None
    load_model: Optional[str] = None
    
    @classmethod
    def from_args(cls, args) -> "VolatileConfig":
        """
        Create config from CLI arguments.
        
        Args:
            args: Parsed CLI arguments (argparse namespace or click context)
            
        Returns:
            VolatileConfig instance
        """
        config = cls()
        
        if hasattr(args, "plot_losses"):
            config.plot_losses = args.plot_losses
        if hasattr(args, "save_model"):
            config.save_model = args.save_model
        if hasattr(args, "load_model"):
            config.load_model = args.load_model
        if hasattr(args, "verbose"):
            config.verbose = args.verbose
            
        return config
