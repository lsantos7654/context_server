"""Configuration management for Context Server CLI."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "localhost"
    port: int = 8000
    database_url: str = (
        "postgresql://context_user:context_password@localhost:5432/context_server"
    )
    openai_api_key: Optional[str] = None


class CLIConfig(BaseSettings):
    """Main CLI configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CONTEXT_SERVER_",
        env_file=".env",
        extra="ignore",
    )
    
    # Direct environment variables (without prefix)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    voyage_api_key: Optional[str] = Field(default=None, alias="VOYAGE_API_KEY")

    # Server settings
    server: ServerConfig = Field(default_factory=ServerConfig)

    # CLI settings
    verbose: bool = False
    color: bool = True
    config_dir: Path = Field(default_factory=lambda: Path.home() / ".context-server")

    # Docker settings
    compose_file: str = "docker-compose.yml"
    project_path: Optional[Path] = None

    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> "CLIConfig":
        """Load configuration from file."""
        if config_path is None:
            config_path = Path.home() / ".context-server" / "config.yaml"

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Handle nested server config
            if "server" in config_data:
                config_data["server"] = ServerConfig(**config_data["server"])
            
            # Handle Path objects
            if "config_dir" in config_data and isinstance(config_data["config_dir"], str):
                config_data["config_dir"] = Path(config_data["config_dir"])
            if "project_path" in config_data and isinstance(config_data["project_path"], str):
                config_data["project_path"] = Path(config_data["project_path"])

            return cls(**config_data)
        except Exception as e:
            # If config file is invalid, return default config
            return cls()

    def save_to_file(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = self.config_dir / "config.yaml"

        # Create config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for YAML serialization
        config_dict = self.model_dump()

        # Handle Path objects
        if isinstance(config_dict.get("config_dir"), Path):
            config_dict["config_dir"] = str(config_dict["config_dir"])
        if isinstance(config_dict.get("project_path"), Path):
            config_dict["project_path"] = str(config_dict["project_path"])

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)


# Global configuration instance
_config: Optional[CLIConfig] = None


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
