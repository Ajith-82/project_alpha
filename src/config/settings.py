from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal, Optional

class Settings(BaseSettings):
    # Data
    market: Literal["us", "india"] = "us"
    data_dir: Path = Path("data")
    cache_ttl_hours: int = 24
    
    # Model
    model_order: int = 2
    correlation_order: int = 52
    learning_rate: float = 0.01
    trend_steps: int = 10000
    correlation_steps: int = 50000
    
    # Screeners
    min_volume: int = 100_000
    breakout_selling_pressure_max: float = 0.40
    breakout_oc_threshold: float = 1.0
    breakout_volume_threshold: float = 0.5
    trend_lookback_days: int = 20
    
    # Risk
    risk_per_trade: float = 0.01
    atr_multiplier: float = 2.0
    atr_period: int = 14
    max_positions: int = 10
    max_portfolio_exposure: float = 1.0  # 1.0 = 100% exposure allowed
    trailing_stop: bool = True
    
    # Email
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    
    # API Keys
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    fmp_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PA_",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
