"""
Data Transformer Module

Provides data transformation utilities for stock price data,
including technical indicators, volatile data extraction, and data cleaning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Technical analysis imports
try:
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for data transformations."""
    nan_threshold: float = 0.33  # Max NaN ratio before column removal
    add_indicators: bool = True
    normalize_prices: bool = False
    fill_method: str = "ffill"  # 'ffill', 'bfill', 'interpolate'


class DataTransformer:
    """
    Transforms stock price data for analysis.
    
    Features:
    - Technical indicator calculation (SMA, MACD, RSI)
    - Data cleaning and NaN handling
    - Volatile data extraction for TensorFlow model
    - Price normalization
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        """
        Initialize the DataTransformer.
        
        Args:
            config: Transformation configuration
        """
        self.config = config or TransformConfig()
    
    # ==================== Technical Indicators ====================
    
    def add_sma(self, df: pd.DataFrame, windows: List[int] = [10, 30, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Average indicators.
        
        Args:
            df: Price DataFrame with 'Adj Close' column
            windows: List of SMA window sizes
            
        Returns:
            DataFrame with SMA columns added
        """
        if not TA_AVAILABLE:
            logger.warning("Technical analysis library not available")
            return df
        
        df = df.copy()
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        
        for window in windows:
            col_name = f"SMA_{window}"
            if len(df) >= window:
                df[col_name] = SMAIndicator(close=df[close_col], window=window).sma_indicator()
            else:
                df[col_name] = np.nan
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD indicator.
        
        Args:
            df: Price DataFrame with 'Adj Close' column
            
        Returns:
            DataFrame with MACD columns added
        """
        if not TA_AVAILABLE:
            logger.warning("Technical analysis library not available")
            return df
        
        df = df.copy()
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        
        if len(df) >= 26:  # Minimum data needed for MACD
            macd = MACD(close=df[close_col], window_slow=26, window_fast=12, window_sign=9)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"] = macd.macd_diff()
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index indicator.
        
        Args:
            df: Price DataFrame with 'Close' column
            window: RSI window (default 14)
            
        Returns:
            DataFrame with RSI column added
        """
        if not TA_AVAILABLE:
            logger.warning("Technical analysis library not available")
            return df
        
        df = df.copy()
        
        if len(df) >= window:
            rsi = RSIIndicator(close=df["Close"], window=window)
            df["RSI"] = rsi.rsi()
        
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with all indicators added
        """
        df = self.add_sma(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        return df
    
    # ==================== Data Cleaning ====================
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price DataFrame by handling NaN values and duplicates.
        
        Args:
            df: Price DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicate dates
        if hasattr(df.index, 'duplicated'):
            df = df.loc[~df.index.duplicated(keep="first")]
        
        # Fill NaN values based on config
        if self.config.fill_method == "ffill":
            df = df.ffill().bfill()
        elif self.config.fill_method == "bfill":
            df = df.bfill().ffill()
        elif self.config.fill_method == "interpolate":
            df = df.interpolate(method="linear").ffill().bfill()
        
        # Remove remaining duplicates
        df = df.drop_duplicates()
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        return df
    
    def remove_high_nan_columns(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove columns with NaN ratio exceeding threshold.
        
        Args:
            df: DataFrame
            threshold: NaN ratio threshold (default from config)
            
        Returns:
            Tuple of (cleaned DataFrame, list of removed columns)
        """
        threshold = threshold or self.config.nan_threshold
        
        nan_ratios = df.isnull().sum() / len(df)
        high_nan_cols = nan_ratios[nan_ratios > threshold].index.tolist()
        
        if high_nan_cols:
            logger.info(f"Removing {len(high_nan_cols)} columns with >{threshold:.0%} NaN values")
            df = df.drop(columns=high_nan_cols)
        
        return df, high_nan_cols
    
    # ==================== Volatile Data Transformation ====================
    
    def extract_volatile_columns(
        self, 
        price_data: Dict[str, pd.DataFrame],
        columns: List[str] = ["Adj Close", "Volume"],
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract specific columns from price data for volatile analysis.
        
        Args:
            price_data: Dictionary of ticker -> DataFrame
            columns: Columns to extract
            
        Returns:
            Dictionary of ticker -> extracted DataFrame
        """
        result = {}
        
        for ticker, df in price_data.items():
            try:
                available_cols = [c for c in columns if c in df.columns]
                if available_cols:
                    result[ticker] = df[available_cols].copy()
                else:
                    logger.warning(f"{ticker}: None of {columns} available")
            except Exception as e:
                logger.warning(f"Failed to extract columns for {ticker}: {e}")
        
        return result
    
    def combine_price_data(
        self,
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine multiple ticker DataFrames into a single MultiIndex DataFrame.
        
        Args:
            price_data: Dictionary of ticker -> DataFrame
            
        Returns:
            Combined DataFrame with (ticker, column) MultiIndex columns
        """
        if not price_data:
            raise ValueError("No price data to combine")
        
        combined = pd.concat(
            price_data.values(),
            keys=price_data.keys(),
            axis=1,
            sort=True,
        )
        
        return combined
    
    def prepare_volatile_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        company_info: Optional[Dict[str, Dict]] = None,
        market: str = "us",
    ) -> Dict:
        """
        Prepare data for volatility model.
        
        Extracts Adj Close and Volume, combines into multi-index DataFrame,
        applies cleaning, and returns structured data.
        
        Args:
            price_data: Dictionary of ticker -> price DataFrame
            company_info: Dictionary of ticker -> company info
            market: Market identifier for currency
            
        Returns:
            Dictionary with structured volatile data
        """
        # Extract volatile columns
        volatile_data = self.extract_volatile_columns(price_data)
        
        if not volatile_data:
            raise ValueError("No valid volatile data extracted")
        
        # Combine into multi-index DataFrame
        combined = self.combine_price_data(volatile_data)
        
        # Remove high-NaN columns
        combined, removed = self.remove_high_nan_columns(combined)
        
        # Clean data
        combined = combined.ffill().bfill().drop_duplicates()
        
        # Get valid tickers (those still in combined after NaN removal)
        valid_tickers = combined.columns.get_level_values(0).unique().tolist()
        
        if removed:
            logger.info(f"Removed {len(removed)} tickers due to incomplete data")
        
        # Extract price and volume arrays
        try:
            price_array = combined.xs("Adj Close", level=1, axis=1).to_numpy().T
            volume_array = combined.xs("Volume", level=1, axis=1).to_numpy().T
        except KeyError:
            logger.warning("Could not extract price/volume arrays")
            price_array = None
            volume_array = None
        
        # Build sector/industry maps
        sectors = {}
        industries = {}
        currencies = []
        
        default_currency = "INR" if market == "india" else "USD"
        
        for ticker in valid_tickers:
            info = (company_info or {}).get(ticker, {})
            sectors[ticker] = info.get("sector", f"NA_{ticker}")
            industries[ticker] = info.get("industry", f"NA_{ticker}")
            currencies.append(default_currency)
        
        return {
            "tickers": valid_tickers,
            "dates": pd.to_datetime(combined.index),
            "price": price_array,
            "volume": volume_array,
            "combined_df": combined,
            "currencies": currencies,
            "default_currency": default_currency,
            "sectors": sectors,
            "industries": industries,
        }
    
    # ==================== Normalization ====================
    
    def normalize_prices(
        self, df: pd.DataFrame, method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize price data.
        
        Args:
            df: Price DataFrame
            method: 'minmax' or 'zscore'
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "minmax":
            for col in numeric_cols:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == "zscore":
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
        
        return df
    
    def calculate_returns(
        self, df: pd.DataFrame, periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate percentage returns.
        
        Args:
            df: Price DataFrame
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with returns
        """
        df = df.copy()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        
        df["Returns"] = df[price_col].pct_change(periods=periods)
        df["Log_Returns"] = np.log(df[price_col] / df[price_col].shift(periods))
        
        return df
