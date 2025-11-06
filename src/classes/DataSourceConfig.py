#!/usr/bin/env python
"""
Configuration manager for data source API keys and settings.
"""
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DataSourceConfig:
    """Manages configuration for multiple data sources."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # API Keys
        self.twelve_data_api_key: Optional[str] = os.getenv("TWELVE_DATA_API_KEY")
        self.alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")

        # Source priority
        priority_str = os.getenv("DATA_SOURCE_PRIORITY", "yfinance,twelvedata,alphavantage")
        self.source_priority: List[str] = [s.strip() for s in priority_str.split(",")]

        # Enable/disable flags
        self.enable_yfinance: bool = os.getenv("ENABLE_YFINANCE", "true").lower() == "true"
        self.enable_twelve_data: bool = os.getenv("ENABLE_TWELVE_DATA", "true").lower() == "true"
        self.enable_alpha_vantage: bool = os.getenv("ENABLE_ALPHA_VANTAGE", "true").lower() == "true"

    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources based on configuration.

        Returns:
            List of enabled source names with valid configuration.
        """
        available = []

        for source in self.source_priority:
            if source == "yfinance" and self.enable_yfinance:
                available.append("yfinance")
            elif source == "twelvedata" and self.enable_twelve_data and self.twelve_data_api_key:
                available.append("twelvedata")
            elif source == "alphavantage" and self.enable_alpha_vantage and self.alpha_vantage_api_key:
                available.append("alphavantage")

        # Always include yfinance as fallback if enabled
        if "yfinance" not in available and self.enable_yfinance:
            available.append("yfinance")

        return available

    def is_source_available(self, source: str) -> bool:
        """
        Check if a specific data source is available.

        Args:
            source: Name of the data source to check.

        Returns:
            True if source is configured and enabled.
        """
        return source in self.get_available_sources()

    def get_api_key(self, source: str) -> Optional[str]:
        """
        Get API key for a specific data source.

        Args:
            source: Name of the data source.

        Returns:
            API key if available, None otherwise.
        """
        if source == "twelvedata":
            return self.twelve_data_api_key
        elif source == "alphavantage":
            return self.alpha_vantage_api_key
        return None

    def validate_configuration(self) -> Dict[str, str]:
        """
        Validate configuration and return status for each source.

        Returns:
            Dictionary with source names as keys and status messages as values.
        """
        status = {}

        if self.enable_yfinance:
            status["yfinance"] = "Available (no API key required)"
        else:
            status["yfinance"] = "Disabled in configuration"

        if self.enable_twelve_data:
            if self.twelve_data_api_key:
                status["twelvedata"] = "Available (API key configured)"
            else:
                status["twelvedata"] = "Disabled (missing API key)"
        else:
            status["twelvedata"] = "Disabled in configuration"

        if self.enable_alpha_vantage:
            if self.alpha_vantage_api_key:
                status["alphavantage"] = "Available (API key configured)"
            else:
                status["alphavantage"] = "Disabled (missing API key)"
        else:
            status["alphavantage"] = "Disabled in configuration"

        return status

    def __repr__(self) -> str:
        """String representation of configuration."""
        available = self.get_available_sources()
        return f"DataSourceConfig(available_sources={available})"


# Global configuration instance
_config_instance: Optional[DataSourceConfig] = None


def get_config() -> DataSourceConfig:
    """
    Get global configuration instance (singleton pattern).

    Returns:
        DataSourceConfig instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DataSourceConfig()
    return _config_instance
