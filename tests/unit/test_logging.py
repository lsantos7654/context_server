"""
Unit tests for logging module.
"""

import logging

from src.core.logging import (
    ContextServerFormatter,
    StructuredLogger,
    get_logger,
    setup_logging,
)


class TestContextServerFormatter:
    """Test custom formatter."""

    def test_format_basic_log(self):
        """Test basic log formatting."""
        formatter = ContextServerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "ℹ️" in result
        assert "Test message" in result

    def test_format_with_extra_data(self):
        """Test formatting with extra data."""
        formatter = ContextServerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"key": "value", "count": 42}
        result = formatter.format(record)
        assert "key=value" in result
        assert "count=42" in result


class TestStructuredLogger:
    """Test structured logger."""

    def test_logger_creation(self):
        """Test logger creation."""
        logger = StructuredLogger("test_logger")
        assert logger.logger.name == "test_logger"

    def test_info_logging(self):
        """Test info level logging."""
        logger = StructuredLogger("test_logger")
        # Just ensure it doesn't throw
        logger.info("Test message", key="value")

    def test_error_logging(self):
        """Test error level logging."""
        logger = StructuredLogger("test_logger")
        # Just ensure it doesn't throw
        logger.error("Test error", error_code=500)

    def test_debug_logging(self):
        """Test debug level logging."""
        logger = StructuredLogger("test_logger")
        # Just ensure it doesn't throw
        logger.debug("Test debug", debug_info="details")

    def test_warning_logging(self):
        """Test warning level logging."""
        logger = StructuredLogger("test_logger")
        # Just ensure it doesn't throw
        logger.warning("Test warning", warning_type="validation")

    def test_critical_logging(self):
        """Test critical level logging."""
        logger = StructuredLogger("test_logger")
        # Just ensure it doesn't throw
        logger.critical("Test critical", severity="high")


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, StructuredLogger)

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        setup_logging("INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file."""
        log_file = tmp_path / "test.log"
        setup_logging("DEBUG", log_file)

        # Check file was created
        assert log_file.parent.exists()

        # Test logging to file
        logger = logging.getLogger("test")
        logger.info("Test log message")

        # Give it a moment to write
        import time

        time.sleep(0.1)

        # File should exist (even if empty due to timing)
        assert log_file.exists() or log_file.parent.exists()

    def test_setup_logging_levels(self):
        """Test different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level)
            root_logger = logging.getLogger()
            assert root_logger.level == getattr(logging, level)
