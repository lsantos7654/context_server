"""Configuration management for Context Server CLI."""

import os
from pathlib import Path
from typing import Any

import yaml

from context_server.models.config import CLIConfig, ServerConfig

# Global configuration instance
_config: CLIConfig | None = None


def get_config() -> CLIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CLIConfig.load_from_file()
    return _config


def set_config(config: CLIConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def get_api_base_url() -> str:
    """Get the API base URL."""
    config = get_config()
    return f"http://{config.server.host}:{config.server.port}"


def get_api_url(path: str) -> str:
    """Get full API URL for a given path."""
    base_url = get_api_base_url()
    path = path.lstrip("/")
    return f"{base_url}/api/{path}"


def get_compose_file_path() -> Path:
    """Get docker-compose.yml path with intelligent fallbacks."""
    config = get_config()

    # 1st: User-configured project path
    if config.project_path and (config.project_path / "docker-compose.yml").exists():
        return config.project_path / "docker-compose.yml"

    # 2nd: Development mode - editable install (current directory has docker-compose.yml)
    current_dir = Path.cwd()
    if (current_dir / "docker-compose.yml").exists():
        # Auto-configure for convenience
        config.project_path = current_dir
        config.save_to_file()
        return current_dir / "docker-compose.yml"

    # 3rd: Development mode - package location (for editable installs)
    try:
        # Get the package root directory (parent of context_server)
        package_root = Path(__file__).parent.parent.parent
        compose_path = package_root / "docker-compose.yml"
        if compose_path.exists():
            # Auto-configure for convenience
            config.project_path = package_root
            config.save_to_file()
            return compose_path
    except Exception:
        pass

    # 4th: Bundled resources (future enhancement for proper package installs)
    # This would be where we'd check for bundled docker-compose.yml in package data

    raise FileNotFoundError(
        "Could not locate docker-compose.yml. Please run from the Context Server project directory "
        "or configure the project path with: ctx setup --project-path /path/to/context-server"
    )


def ensure_project_setup() -> None:
    """Ensure project is properly configured, auto-detecting if possible."""
    try:
        get_compose_file_path()  # This will auto-configure if possible
    except FileNotFoundError:
        # Could not auto-detect, but that's okay - error will be shown when needed
        pass
