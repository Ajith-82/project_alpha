class ProjectAlphaError(Exception):
    """Base exception for all application errors."""

class DataFetchError(ProjectAlphaError):
    """Failed to download market data."""

class ScreenerError(ProjectAlphaError):
    """Screener execution failed."""

class ModelTrainingError(ProjectAlphaError):
    """Model failed to converge."""

class ConfigurationError(ProjectAlphaError):
    """Invalid configuration."""

class DataValidationError(ProjectAlphaError):
    """Input data failed validation."""
