"""CLI configuration models."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from context_server.models.config.server import ServerConfig


class CLIConfig(BaseSettings):
    """Main CLI configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CONTEXT_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    # Direct environment variables (without prefix)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")

    # Server settings
    server: ServerConfig = Field(default_factory=ServerConfig)

    # CLI settings
    verbose: bool = False
    color: bool = True
    config_dir: Path = Field(default_factory=lambda: Path.home() / ".context-server")

    # Docker settings
    compose_file: str = "docker-compose.yml"
    project_path: Path | None = None

    @classmethod
    def load_from_file(cls, config_path: Path | None = None) -> "CLIConfig":
        """Load configuration from file."""
        import yaml

        if config_path is None:
            config_path = Path.home() / ".context-server" / "config.yaml"

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Handle nested server config (now Pydantic)
            if "server" in config_data:
                config_data["server"] = ServerConfig.parse_obj(config_data["server"])

            # Handle Path objects
            if "config_dir" in config_data and isinstance(
                config_data["config_dir"], str
            ):
                config_data["config_dir"] = Path(config_data["config_dir"])
            if "project_path" in config_data and isinstance(
                config_data["project_path"], str
            ):
                config_data["project_path"] = Path(config_data["project_path"])

            return cls(**config_data)
        except Exception:
            # If config file is invalid, return default config
            return cls()

    def save_to_file(self, config_path: Path | None = None) -> None:
        """Save configuration to file."""
        import yaml

        if config_path is None:
            config_path = self.config_dir / "config.yaml"

        # Create config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for YAML serialization
        config_dict = self.model_dump()

        # Handle nested Pydantic models
        if "server" in config_dict and hasattr(config_dict["server"], "model_dump"):
            config_dict["server"] = config_dict["server"].model_dump()

        # Handle Path objects
        if isinstance(config_dict.get("config_dir"), Path):
            config_dict["config_dir"] = str(config_dict["config_dir"])
        if isinstance(config_dict.get("project_path"), Path):
            config_dict["project_path"] = str(config_dict["project_path"])

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)


__all__ = ["CLIConfig"]
