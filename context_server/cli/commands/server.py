"""Server management commands for Context Server CLI."""

import asyncio
import time
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from ..config import get_api_base_url, get_api_url
from ..utils import (
    check_api_health,
    check_docker_compose_running,
    check_docker_running,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    run_command,
)

console = Console()


@click.group()
@click.help_option("-h", "--help")
def server():
    """Server management commands.

    Commands for starting, stopping, and managing the Context Server
    Docker containers and services.
    """
    pass


@server.command()
@click.option("--build", is_flag=True, help="Build images before starting")
@click.option(
    "--detach/--no-detach", default=True, help="Run in detached mode (default: True)"
)
@click.help_option("-h", "--help")
def up(build, detach):
    """Start the Context Server services.

    Starts PostgreSQL database and FastAPI server using Docker Compose.
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
            else:
                echo_warning("Services may not be fully ready yet")

    except Exception as e:
        echo_error(f"Failed to start services: {e}")


@server.command()
@click.option("--volumes", is_flag=True, help="Remove volumes as well")
@click.help_option("-h", "--help")
def down(volumes):
    """Stop the Context Server services.

    Stops all running Docker containers for the Context Server.
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
@click.help_option("-h", "--help")
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
@click.help_option("-h", "--help")
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
@click.help_option("-h", "--help")
def status(wait):
    """Check server status and health.

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
        echo_info("Run: context-server server up")
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
        echo_success("Database reset completed!")
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
