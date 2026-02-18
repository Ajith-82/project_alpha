import pandas as pd
import numpy as np
import structlog
from typing import List, Tuple
from exceptions import DataValidationError
from classes.data.schemas import PriceRow

logger = structlog.get_logger()

def validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure DataFrame has all required OHLCV columns.
    
    Args:
        df: DataFrame to check
        
    Raises:
        DataValidationError: If required columns are missing
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise DataValidationError(f"Missing required columns: {missing}")

def validate_dtypes(df: pd.DataFrame) -> None:
    """
    Ensure DataFrame columns are numeric.
    """
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
             raise DataValidationError(f"Column {col} must be numeric")

def validate_price_sanity(df: pd.DataFrame) -> List[str]:
    """
    Check for sanity rules:
    - Prices > 0
    - Volume >= 0
    - High >= Low
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check negative prices
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        bad_rows = df[(df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)]
        errors.append(f"Negative or zero prices detected in {len(bad_rows)} rows")
        
    # Check negative volume
    if (df["Volume"] < 0).any():
        bad_rows = df[df["Volume"] < 0]
        errors.append(f"Negative volume detected in {len(bad_rows)} rows")
        
    # Check High >= Low constraint
    # Allow small epsilon for floating point issues? Usually not needed for OHLC values.
    # Using a small epsilon 1e-9 just in case of weird aggregations, but raw data should be clean.
    if (df["High"] < df["Low"]).any():
        bad_rows = df[df["High"] < df["Low"]]
        errors.append(f"High < Low detected in {len(bad_rows)} rows")
        
    return errors

def validate_data_quality(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Comprehensive validation of OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol for logging
        
    Returns:
        Validated DataFrame (potentially cleaned if we add repair logic later)
        
    Raises:
        DataValidationError: If validation fails critically
    """
    try:
        if df.empty:
            raise DataValidationError(f"Empty DataFrame for {ticker}")
            
        validate_columns(df)
        validate_dtypes(df)
        
        # Minimum data length
        min_rows = 30
        if len(df) < min_rows:
            raise DataValidationError(f"Insufficient data for {ticker}: {len(df)} rows (min {min_rows})")
            
        # Sanity checks
        sanity_errors = validate_price_sanity(df)
        if sanity_errors:
            error_msg = "; ".join(sanity_errors)
            logger.warning("Data validation failed", ticker=ticker, errors=error_msg)
            # Checking if we should raise strict error or just warn. 
            # For now, let's raise strict error for Phase 2 validation goal.
            raise DataValidationError(f"Data sanity check failed for {ticker}: {error_msg}")
            
        # Check for missing values
        if df.isnull().any().any():
            # Report which columns have NaNs
            nan_counts = df.isnull().sum()
            has_nans = nan_counts[nan_counts > 0]
            logger.warning("Missing values detected", ticker=ticker, details=has_nans.to_dict())
            # We fail on NaNs? Usually we fill them in Download.py
            # But validator should enforce cleanliness.
            # If we decide validation runs AFTER cleaning, then this is an error.
            if df[["Open", "High", "Low", "Close"]].isnull().any().any():
                 raise DataValidationError(f"NaNs in price columns for {ticker}")

        logger.debug("Data validation passed", ticker=ticker, rows=len(df))
        return df
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Validation error for {ticker}: {str(e)}") from e

def repair_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Attempt to repair common data issues:
    - Interpolate small gaps of NaNs
    - Set negative volume to 0
    - Drop rows with NaN if interpolation fails
    """
    df_clean = df.copy()
    
    # 1. Fix negative volume
    if "Volume" in df_clean.columns:
        neg_vol = df_clean["Volume"] < 0
        if neg_vol.any():
            logger.warning(f"Repairing negative volume for {ticker} in {neg_vol.sum()} rows")
            df_clean.loc[neg_vol, "Volume"] = 0
            
    # 2. Interpolate missing prices
    price_cols = ["Open", "High", "Low", "Close"]
    if df_clean[price_cols].isnull().any().any():
        logger.info(f"Interpolating missing prices for {ticker}")
        df_clean[price_cols] = df_clean[price_cols].interpolate(method='time', limit=3)
        
    # 3. Drop remaining NaNs
    if df_clean.isnull().any().any():
        before_len = len(df_clean)
        df_clean.dropna(inplace=True)
        dropped = before_len - len(df_clean)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} unrepairable rows for {ticker}")
            
    return df_clean
