"""
Data Validator Module

Provides data quality validation for stock price data and company information.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, any] = field(default_factory=dict)


class DataValidator:
    """
    Validates stock data for quality and completeness.
    
    Validation Rules:
    - Price data must not be empty
    - Required columns must be present
    - NaN ratio must be below threshold
    - Prices must be positive
    - Dates must be valid and ordered
    """
    
    REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    OPTIONAL_PRICE_COLUMNS = ["Adj Close", "Dividends", "Stock Splits"]
    
    DEFAULT_NAN_THRESHOLD = 0.33  # Max 33% NaN values
    DEFAULT_MIN_ROWS = 5  # Minimum data points
    
    def __init__(
        self,
        nan_threshold: float = DEFAULT_NAN_THRESHOLD,
        min_rows: int = DEFAULT_MIN_ROWS,
        strict: bool = False,
    ):
        """
        Initialize the DataValidator.
        
        Args:
            nan_threshold: Maximum allowed NaN ratio (0-1)
            min_rows: Minimum required data rows
            strict: If True, warnings become errors
        """
        self.nan_threshold = nan_threshold
        self.min_rows = min_rows
        self.strict = strict
    
    def validate_price_data(
        self, df: pd.DataFrame, ticker: str = ""
    ) -> ValidationResult:
        """
        Validate price data DataFrame.
        
        Args:
            df: Price data DataFrame
            ticker: Ticker symbol for error messages
            
        Returns:
            ValidationResult with errors, warnings, and stats
        """
        result = ValidationResult(valid=True)
        prefix = f"[{ticker}] " if ticker else ""
        
        # Check for None or non-DataFrame
        if df is None:
            result.valid = False
            result.errors.append(f"{prefix}Price data is None")
            return result
        
        if not isinstance(df, pd.DataFrame):
            result.valid = False
            result.errors.append(f"{prefix}Price data is not a DataFrame")
            return result
        
        # Check for empty DataFrame
        if df.empty:
            result.valid = False
            result.errors.append(f"{prefix}Price data is empty")
            return result
        
        # Check minimum rows
        if len(df) < self.min_rows:
            result.valid = False
            result.errors.append(
                f"{prefix}Insufficient data: {len(df)} rows (minimum {self.min_rows})"
            )
            return result
        
        # Check required columns
        missing_cols = [c for c in self.REQUIRED_PRICE_COLUMNS if c not in df.columns]
        if missing_cols:
            result.valid = False
            result.errors.append(f"{prefix}Missing required columns: {missing_cols}")
            return result
        
        # Check for optional columns
        missing_optional = [c for c in self.OPTIONAL_PRICE_COLUMNS if c not in df.columns]
        if missing_optional:
            result.warnings.append(f"{prefix}Missing optional columns: {missing_optional}")
        
        # Check NaN ratio
        nan_ratio = df[self.REQUIRED_PRICE_COLUMNS].isna().sum().sum() / (
            len(df) * len(self.REQUIRED_PRICE_COLUMNS)
        )
        result.stats["nan_ratio"] = nan_ratio
        
        if nan_ratio > self.nan_threshold:
            msg = f"{prefix}Excessive NaN values: {nan_ratio:.1%} (threshold {self.nan_threshold:.1%})"
            if self.strict:
                result.valid = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)
        
        # Check for negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    result.valid = False
                    result.errors.append(f"{prefix}Negative prices in {col}: {neg_count} rows")
        
        # Check for zero volume (warning only)
        if "Volume" in df.columns:
            zero_vol_ratio = (df["Volume"] == 0).sum() / len(df)
            if zero_vol_ratio > 0.5:
                result.warnings.append(
                    f"{prefix}High ratio of zero volume: {zero_vol_ratio:.1%}"
                )
        
        # Check date ordering
        if hasattr(df.index, 'is_monotonic_increasing'):
            if not df.index.is_monotonic_increasing:
                result.warnings.append(f"{prefix}Dates are not in ascending order")
        
        # Check for duplicate dates
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            result.warnings.append(f"{prefix}Duplicate dates: {dup_count}")
        
        # Collect stats
        result.stats["rows"] = len(df)
        result.stats["columns"] = len(df.columns)
        result.stats["date_range"] = (
            str(df.index.min()) if len(df) > 0 else "N/A",
            str(df.index.max()) if len(df) > 0 else "N/A",
        )
        
        # If strict mode, convert warnings to errors
        if self.strict and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.valid = False
        
        return result
    
    def validate_company_info(
        self, info: Dict, ticker: str = ""
    ) -> ValidationResult:
        """
        Validate company information dictionary.
        
        Args:
            info: Company info dictionary
            ticker: Ticker symbol for error messages
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)
        prefix = f"[{ticker}] " if ticker else ""
        
        if info is None:
            result.valid = False
            result.errors.append(f"{prefix}Company info is None")
            return result
        
        if not isinstance(info, dict):
            result.valid = False
            result.errors.append(f"{prefix}Company info is not a dictionary")
            return result
        
        if not info:
            result.valid = False
            result.errors.append(f"{prefix}Company info is empty")
            return result
        
        # Check for important fields
        important_fields = ["sector", "industry"]
        missing = [f for f in important_fields if f not in info or not info.get(f)]
        if missing:
            result.warnings.append(f"{prefix}Missing important fields: {missing}")
        
        # Collect stats
        result.stats["field_count"] = len(info)
        result.stats["has_sector"] = "sector" in info and bool(info.get("sector"))
        result.stats["has_industry"] = "industry" in info and bool(info.get("industry"))
        
        return result
    
    def validate_batch(
        self,
        price_data: Dict[str, pd.DataFrame],
        company_info: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict[str, ValidationResult], Dict[str, ValidationResult]]:
        """
        Validate a batch of stock data.
        
        Args:
            price_data: Dictionary mapping tickers to price DataFrames
            company_info: Dictionary mapping tickers to company info
            
        Returns:
            Tuple of (price_results, company_results)
        """
        price_results = {}
        company_results = {}
        
        for ticker, df in price_data.items():
            price_results[ticker] = self.validate_price_data(df, ticker)
        
        if company_info:
            for ticker, info in company_info.items():
                company_results[ticker] = self.validate_company_info(info, ticker)
        
        return price_results, company_results
    
    def filter_valid(
        self,
        price_data: Dict[str, pd.DataFrame],
        company_info: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], List[str]]:
        """
        Filter out invalid data and return only valid entries.
        
        Args:
            price_data: Dictionary mapping tickers to price DataFrames
            company_info: Dictionary mapping tickers to company info
            
        Returns:
            Tuple of (valid_price_data, valid_company_info, removed_tickers)
        """
        price_results, _ = self.validate_batch(price_data, company_info)
        
        valid_tickers = [t for t, r in price_results.items() if r.valid]
        removed_tickers = [t for t, r in price_results.items() if not r.valid]
        
        valid_price_data = {t: price_data[t] for t in valid_tickers}
        valid_company_info = {}
        
        if company_info:
            valid_company_info = {t: company_info[t] for t in valid_tickers if t in company_info}
        
        if removed_tickers:
            logger.warning(f"Removed {len(removed_tickers)} invalid tickers: {removed_tickers[:10]}...")
        
        return valid_price_data, valid_company_info, removed_tickers
    
    def get_summary(
        self, results: Dict[str, ValidationResult]
    ) -> Dict[str, any]:
        """
        Generate a summary of validation results.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Summary statistics
        """
        total = len(results)
        valid = sum(1 for r in results.values() if r.valid)
        invalid = total - valid
        
        all_errors = []
        all_warnings = []
        for r in results.values():
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_ratio": valid / total if total > 0 else 0,
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
        }
