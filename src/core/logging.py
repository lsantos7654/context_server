"""
Unified logging system for Context Server.

Consolidates logging approaches from multiple modules following CLAUDE.md principles:
- Structured logging with contextual information
- Consistent format across all modules
- Configurable log levels and outputs
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from .utils import FileUtils

# No typing imports needed for Python 3.12+


class ContextServerFormatter(logging.Formatter):
    """
    Custom formatter for Context Server logs with emoji support and structured output.
    """

    # Emoji mapping for different log levels
    LEVEL_EMOJIS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    # Color codes for terminal output
    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[36m",  # Cyan
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with emoji, color, and structured data."""
        # Add emoji and timestamp
        emoji = self.LEVEL_EMOJIS.get(record.levelname, "📝")
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Base message
        message = f"[{timestamp}] {emoji}  {record.getMessage()}"

        # Add structured data if present
        if hasattr(record, "extra_data") and record.extra_data:
            extra_parts = []
            for key, value in record.extra_data.items():
                extra_parts.append(f"{key}={value}")
            if extra_parts:
                message += f" ({', '.join(extra_parts)})"

        # Add color for terminal output
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            message = f"{color}{message}{reset}"

        return message


class StructuredLogger:
    """
    Structured logger that provides consistent logging interface across the application.

    Replaces the custom logging implementations in smart_extract.py and standardizes
    logging throughout the codebase.
    """

    def __init__(self, name: str, level: str = "INFO"):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler with custom formatter
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(ContextServerFormatter())
        self.logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional structured data."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional structured data."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional structured data."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with optional structured data."""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, extra_data: dict[str, any]) -> None:
        """Internal method to log with structured data."""
        # Create log record with extra data
        record = self.logger.makeRecord(
            self.logger.name, level, __file__, 0, message, (), None
        )
        record.extra_data = extra_data
        self.logger.handle(record)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Setup application-wide logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ContextServerFormatter())
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        FileUtils.ensure_directory(log_file.parent)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
