"""Server management commands for Context Server CLI."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from context_server.cli.config import (
    ensure_project_setup,
    get_api_base_url,
    get_api_url,
    get_compose_file_path,
)
from context_server.cli.help_formatter import rich_help_option
from context_server.cli.utils import (
    check_api_health,
    check_docker_running,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    run_command,
)

console = Console()

# Container names (hardcoded for Context Server)
API_CONTAINER = "context_server-api-1"
POSTGRES_CONTAINER = "context_server-postgres-1"


@click.group()
@rich_help_option("-h", "--help")
def server():
    """Manage Context Server Docker services and database.

    Commands for controlling the lifecycle of Context Server services
    including the PostgreSQL database and FastAPI application.

    Examples:
        ctx server up                       # Start all services
        ctx server down                     # Stop all services
        ctx server status                   # Check service health
        ctx server logs --follow api        # Follow API logs
    """
    pass


@server.command()
@rich_help_option("-h", "--help")
def up():
    """Start the Context Server services.

    Starts PostgreSQL database and FastAPI server containers with fresh builds.
    Once running, Claude can connect via MCP if configured.
    """
    if not check_docker_running():
        echo_error("Docker is not running. Please start Docker first.")
        return

    echo_info("Starting Context Server services...")

    try:
        # Auto-detect and ensure project is configured
        ensure_project_setup()

        # Get the docker-compose.yml file path
        compose_file = get_compose_file_path()
        echo_info(f"Using docker-compose file: {compose_file}")

        # Use docker-compose with explicit file path to start services with build
        echo_info("Building and starting services with docker-compose...")
        run_command(
            ["docker-compose", "--file", str(compose_file), "up", "-d", "--build"]
        )

        echo_success("Context Server started!")
        echo_info(f"API available at: {get_api_base_url()}")
        echo_info(f"API docs at: {get_api_base_url()}/docs")
        echo_info("Database at: localhost:5432")

        # Wait for services to be ready
        echo_info("Waiting for services to be ready...")
        if wait_for_services():
            echo_success("All services are ready!")
            echo_info("Configure Claude integration with: ctx init")
        else:
            echo_warning("Services may not be fully ready yet")

    except FileNotFoundError as e:
        echo_error(str(e))
        echo_info(
            "Make sure you're in the Context Server project directory or configure the path."
        )
    except Exception as e:
        echo_error(f"Failed to start services: {e}")


@server.command()
@rich_help_option("-h", "--help")
def down():
    """Stop the Context Server services.

    Stops all running Docker containers for the Context Server.
    """
    echo_info("Stopping Context Server services...")

    try:
        # Get the docker-compose.yml file path
        compose_file = get_compose_file_path()

        # Use docker-compose with explicit file path to stop services
        echo_info("Stopping services with docker-compose...")
        run_command(["docker-compose", "--file", str(compose_file), "down"])

        echo_success("Context Server stopped!")
    except FileNotFoundError as e:
        echo_warning(str(e))
        echo_info("Falling back to direct container commands...")
        try:
            # Fallback to direct container commands
            echo_info(f"Stopping {API_CONTAINER}...")
            run_command(["docker", "stop", API_CONTAINER])

            echo_info(f"Stopping {POSTGRES_CONTAINER}...")
            run_command(["docker", "stop", POSTGRES_CONTAINER])

            echo_success("Context Server stopped!")
        except Exception as fallback_e:
            echo_error(f"Failed to stop services: {fallback_e}")
    except Exception as e:
        echo_error(f"Failed to stop services: {e}")


@server.command()
@rich_help_option("-h", "--help")
def restart():
    """Restart the Context Server services.

    Stops and then starts all services with fresh builds.
    """
    echo_info("Restarting Context Server...")

    try:
        # Auto-detect and ensure project is configured
        ensure_project_setup()

        # Get the docker-compose.yml file path
        compose_file = get_compose_file_path()
        echo_info(f"Using docker-compose file: {compose_file}")

        # Stop services first
        echo_info("Stopping services...")
        run_command(["docker-compose", "--file", str(compose_file), "down"])

        # Start services with build
        echo_info("Building and starting services...")
        run_command(
            ["docker-compose", "--file", str(compose_file), "up", "-d", "--build"]
        )

        echo_success("Context Server restarted!")

        # Wait for services to be ready
        echo_info("Waiting for services to be ready...")
        if wait_for_services():
            echo_success("All services are ready!")
        else:
            echo_warning("Services may not be fully ready yet")

    except FileNotFoundError as e:
        echo_error(str(e))
        echo_info(
            "Make sure you're in the Context Server project directory or configure the path."
        )
    except Exception as e:
        echo_error(f"Failed to restart services: {e}")


@server.command()
@click.option(
    "--service", default="api", help="Service to show logs for (api, postgres)"
)
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option(
    "--tail", default=100, help="Number of lines to show from the end of logs"
)
@rich_help_option("-h", "--help")
def logs(service, follow, tail):
    """Show service logs.

    Args:
        service: Service name (api, postgres)
        follow: Follow log output in real-time
        tail: Number of lines to show from end
    """
    # Map service names to container names
    container_map = {
        "api": API_CONTAINER,
        "postgres": POSTGRES_CONTAINER,
    }

    if service not in container_map:
        echo_error(f"Unknown service: {service}. Use 'api' or 'postgres'")
        return

    container_name = container_map[service]

    # Check if container is running
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            echo_error(f"Container {container_name} is not running")
            return
    except Exception as e:
        echo_error(f"Failed to check container status: {e}")
        return

    cmd = ["docker", "logs"]

    if follow:
        cmd.append("-f")

    cmd.extend(["--tail", str(tail)])
    cmd.append(container_name)

    try:
        run_command(cmd)
    except KeyboardInterrupt:
        echo_info("Log following stopped")
    except Exception as e:
        echo_error(f"Failed to show logs: {e}")


@server.command()
@click.option("--wait", is_flag=True, help="Wait for services to be ready")
@rich_help_option("-h", "--help")
def status(wait):
    """Check server status and health.

    Shows status of Context Server, Docker services, and MCP server.

    Args:
        wait: Wait for services to be ready
    """
    echo_info("Checking server status...")

    # Check Docker
    if not check_docker_running():
        echo_error("Docker is not running")
        return

    echo_success("Docker is running")

    # Check container status
    containers_running = _check_containers_running()
    if not containers_running:
        echo_error("Context Server containers are not running")
        echo_info("Run: ctx server up")
        return

    echo_success("Context Server containers are running")

    # Check API health
    if wait:
        echo_info("Waiting for API to be ready...")
        if wait_for_services():
            echo_success("API is ready!")
        else:
            echo_error("API failed to become ready")
    else:

        async def check_health():
            healthy, error = await check_api_health()
            if healthy:
                echo_success("API is healthy")
            else:
                echo_error(f"API is not healthy: {error}")

        asyncio.run(check_health())

    # Check MCP configuration
    echo_info("Checking Claude MCP configuration...")
    claude_config_dir = _detect_claude_config_dir()
    if claude_config_dir:
        claude_config_file = Path(claude_config_dir) / "config.json"
        if claude_config_file.exists():
            try:
                with open(claude_config_file, "r") as f:
                    config = json.load(f)
                if "mcpServers" in config and "context-server" in config["mcpServers"]:
                    echo_success("Claude MCP integration is configured")
                else:
                    echo_warning("Claude MCP integration not configured")
                    echo_info("Run: ctx init")
            except Exception:
                echo_warning("Could not read Claude configuration")
                echo_info("Run: ctx init")
        else:
            echo_warning("Claude MCP integration not configured")
            echo_info("Run: ctx init")
    else:
        echo_warning("Claude configuration directory not found")
        echo_info("Run: ctx init")


@server.command()
@click.option("--database", default="context_server", help="Database name")
@click.option("--user", default="context_user", help="Database user")
def shell(database, user):
    """Connect to the PostgreSQL database shell.

    Args:
        database: Database name to connect to
        user: Database user
    """
    # Check if postgres container is running
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={POSTGRES_CONTAINER}"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            echo_error(f"PostgreSQL container ({POSTGRES_CONTAINER}) is not running")
            echo_info("Run: ctx server up")
            return
    except Exception as e:
        echo_error(f"Failed to check container status: {e}")
        return

    echo_info(f"Connecting to database: {database}")

    cmd = [
        "docker",
        "exec",
        "-it",
        POSTGRES_CONTAINER,
        "psql",
        "-U",
        user,
        "-d",
        database,
    ]

    try:
        run_command(cmd)
    except Exception as e:
        echo_error(f"Failed to connect to database: {e}")


def _check_containers_running() -> bool:
    """Check if Context Server containers are running."""
    try:
        containers = [API_CONTAINER, POSTGRES_CONTAINER]
        for container in containers:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container}"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                return False
        return True
    except Exception:
        return False


def wait_for_services(timeout: int = 60, check_interval: float = 2.0) -> bool:
    """Wait for services to be ready.

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if services are ready, False if timeout
    """
    start_time = time.time()

    with Live(Spinner("dots", "Waiting for services..."), console=console):
        while time.time() - start_time < timeout:
            try:
                # Check if API is healthy
                async def check():
                    return await check_api_health()

                healthy, _ = asyncio.run(check())
                if healthy:
                    return True

                time.sleep(check_interval)
            except Exception:
                time.sleep(check_interval)

    return False


def _detect_claude_config_dir() -> str | None:
    """Detect Claude configuration directory."""
    possible_paths = [
        # macOS
        Path.home() / "Library" / "Application Support" / "Claude",
        # Linux
        Path.home() / ".config" / "claude",
        # Windows
        Path.home() / "AppData" / "Roaming" / "Claude",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None
