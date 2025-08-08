"""Configuration for the MCP server."""

import logging

# Default configuration
DEFAULT_CONFIG = {
    "context_server_url": "http://localhost:8000",
    "mcp_server_name": "context-server",
    "mcp_server_version": "0.1.0",
    "log_level": logging.INFO,
    "request_timeout": 30.0,
}


class Config:
    """MCP server configuration."""

    def __init__(self, **overrides: str | int | float):
        """Initialize configuration with optional overrides."""
        self.context_server_url = overrides.get(
            "context_server_url", DEFAULT_CONFIG["context_server_url"]
        )
        self.mcp_server_name = overrides.get(
            "mcp_server_name", DEFAULT_CONFIG["mcp_server_name"]
        )
        self.mcp_server_version = overrides.get(
            "mcp_server_version", DEFAULT_CONFIG["mcp_server_version"]
        )
        self.log_level = overrides.get("log_level", DEFAULT_CONFIG["log_level"])
        self.request_timeout = overrides.get(
            "request_timeout", DEFAULT_CONFIG["request_timeout"]
        )

    def __repr__(self) -> str:
        return f"Config(context_server_url='{self.context_server_url}')"
