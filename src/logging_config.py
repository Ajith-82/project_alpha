import structlog
import logging
import sys

def configure_logging(level: str = "INFO", json_output: bool = False):
    """
    Configure structured logging for the application.
    
    Args:
        level: transform string level (INFO, DEBUG) to logging constant
        json_output: if True, output JSON, otherwise colorful console output
    """
    
    # Map string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )
