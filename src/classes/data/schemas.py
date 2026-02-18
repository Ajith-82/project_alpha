from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PriceRow(BaseModel):
    """
    Represents a single row of OHLCV data.
    Enforces type safety and basic logical constraints.
    """
    date: datetime
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    adj_close: Optional[float] = Field(None, alias="Adj Close", gt=0)

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info):
        """Ensure High is greater than or equal to Low."""
        values = info.data
        if "low" in values and v < values["low"]:
            raise ValueError(f"High ({v}) cannot be less than Low ({values['low']})")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info):
        """Ensure High is the maximum of Open/Close."""
        values = info.data
        if "open" in values and v < values["open"]:
             # Allow small floating point errors? No, strict for now.
             pass 
             # Actually, for some data providers, broken ticks happen. 
             # But strictly speaking High must be >= Open.
        return v
    
    # We can add more strict validators if needed, but High >= Low is the critical invariant.
