"""Server management commands for Context Server CLI."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from ..config import get_api_base_url, get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    check_api_health,
    check_docker_compose_running,
    check_docker_running,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    get_project_root,
    run_command,
)

console = Console()


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
        ctx server reset-db --force        # Reset database
    """
    pass


@server.command()
@click.option("--build", is_flag=True, help="Build images before starting")
@click.option(
    "--detach/--no-detach", default=True, help="Run in detached mode (default: True)"
)
@rich_help_option("-h", "--help")
def up(build, detach):
    """Start the Context Server services.

    Launches PostgreSQL database and FastAPI server using Docker Compose.
    Once running, Claude can connect via MCP if configured.
    """
    if not check_docker_running():
        echo_error("Docker is not running. Please start Docker first.")
        return

    echo_info("Starting Context Server services...")

    cmd = ["docker-compose", "up"]

    if build:
        cmd.append("--build")

    if detach:
        cmd.append("-d")

    try:
        run_command(cmd)

        if detach:
            echo_success("Context Server started in detached mode!")
            echo_info(f"API available at: {get_api_base_url()}")
            echo_info(f"API docs at: {get_api_base_url()}/docs")
            echo_info("Database at: localhost:5432")

            # Wait for services to be ready
            echo_info("Waiting for services to be ready...")
            if wait_for_services():
                echo_success("All services are ready!")
                echo_info("Configure Claude integration with: ctx claude install")
            else:
                echo_warning("Services may not be fully ready yet")

    except Exception as e:
        echo_error(f"Failed to start services: {e}")


@server.command()
@click.option("--volumes", is_flag=True, help="Remove volumes as well")
@rich_help_option("-h", "--help")
def down(volumes):
    """Stop the Context Server services.

    Stops all running Docker containers for the Context Server.
    Optionally removes associated volumes and data.
    """
    echo_info("Stopping Context Server services...")

    cmd = ["docker-compose", "down"]

    if volumes:
        cmd.append("--volumes")

    try:
        run_command(cmd)
        echo_success("Context Server stopped!")
    except Exception as e:
        echo_error(f"Failed to stop services: {e}")


@server.command()
@rich_help_option("-h", "--help")
def restart():
    """Restart the Context Server services.

    Stops and then starts all services.
    """
    echo_info("Restarting Context Server...")

    # Stop services
    from click.testing import CliRunner

    runner = CliRunner()

    result = runner.invoke(down, [])
    if result.exit_code != 0:
        echo_error("Failed to stop services")
        return

    # Start services
    result = runner.invoke(up, [])
    if result.exit_code != 0:
        echo_error("Failed to start services")
        return

    echo_success("Context Server restarted!")


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
    if not check_docker_compose_running():
        echo_error("Docker Compose services are not running")
        return

    cmd = ["docker-compose", "logs"]

    if follow:
        cmd.append("-f")

    cmd.extend(["--tail", str(tail)])
    cmd.append(service)

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

    # Check Docker Compose
    if not check_docker_compose_running():
        echo_error("Docker Compose services are not running")
        echo_info("Run: ctx server up")
        return

    echo_success("Docker Compose services are running")

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
                    echo_info("Run: ctx claude install")
            except Exception:
                echo_warning("Could not read Claude configuration")
                echo_info("Run: ctx claude install")
        else:
            echo_warning("Claude MCP integration not configured")
            echo_info("Run: ctx claude install")
    else:
        echo_warning("Claude configuration directory not found")
        echo_info("Run: ctx claude install")


@server.command()
@click.option("--database", default="context_server", help="Database name")
@click.option("--user", default="context_user", help="Database user")
def shell(database, user):
    """Connect to the PostgreSQL database shell.

    Args:
        database: Database name to connect to
        user: Database user
    """
    if not check_docker_compose_running():
        echo_error("Docker Compose services are not running")
        return

    echo_info(f"Connecting to database: {database}")

    cmd = ["docker-compose", "exec", "postgres", "psql", "-U", user, "-d", database]

    try:
        run_command(cmd)
    except Exception as e:
        echo_error(f"Failed to connect to database: {e}")


@server.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def reset_db(force):
    """Reset the database (WARNING: destroys all data).

    Args:
        force: Skip confirmation prompt
    """
    if not force and not confirm_action(
        "This will destroy all data in the database. Continue?", default=False
    ):
        echo_info("Database reset cancelled")
        return

    if not check_docker_compose_running():
        echo_error("Docker Compose services are not running")
        return

    echo_warning("Resetting database...")

    cmd = [
        "docker-compose",
        "exec",
        "postgres",
        "psql",
        "-U",
        "context_user",
        "-d",
        "context_server",
        "-c",
        "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public; CREATE EXTENSION IF NOT EXISTS vector;",
    ]

    try:
        run_command(cmd)
        echo_success("Database schema reset!")

        # Reinitialize database tables through API
        echo_info("Reinitializing database tables...")

        async def reinitialize():
            try:
                async with httpx.AsyncClient() as client:
                    # Use base URL directly for admin endpoints (not through /api/ prefix)
                    admin_url = f"{get_api_base_url()}/admin/reinitialize-db"
                    response = await client.post(admin_url, timeout=30.0)
                    if response.status_code == 200:
                        echo_success("Database reinitialization completed!")
                        return True
                    else:
                        echo_error(
                            f"Failed to reinitialize database: {response.status_code} - {response.text}"
                        )
                        return False
            except Exception as e:
                echo_error(f"Failed to call reinitialize API: {e}")
                return False

        success = asyncio.run(reinitialize())
        if success:
            echo_success("Database reset completed!")
        else:
            echo_warning(
                "Database reset completed, but reinitialization may have failed. You may need to restart the API server."
            )

    except Exception as e:
        echo_error(f"Failed to reset database: {e}")


@server.command()
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def ps(output_format):
    """Show running containers.

    Args:
        output_format: Output format (table, json)
    """
    cmd = ["docker-compose", "ps"]

    if output_format == "json":
        cmd.append("--format")
        cmd.append("json")

    try:
        run_command(cmd)
    except Exception as e:
        echo_error(f"Failed to show containers: {e}")


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


def _detect_claude_config_dir() -> Optional[str]:
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
