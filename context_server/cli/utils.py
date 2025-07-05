"""Utility functions for Context Server CLI."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import httpx
from rich.console import Console
from rich.table import Table

from .config import get_api_url, get_config

console = Console()


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command with better error handling."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        if not capture_output:
            console.print(f"[red]Command failed: {' '.join(command)}[/red]")
            if e.stderr:
                console.print(f"[red]Error: {e.stderr}[/red]")
        raise


def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = run_command(
            ["docker", "info"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_docker_compose_running() -> bool:
    """Check if Docker Compose services are running."""
    try:
        result = run_command(
            ["docker-compose", "ps", "-q"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except FileNotFoundError:
        return False


async def check_api_health() -> Tuple[bool, Optional[str]]:
    """Check if the API is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                get_api_url("health"),
                timeout=5.0,
            )
            if response.status_code == 200:
                return True, None
            else:
                return False, f"API returned status {response.status_code}"
    except httpx.RequestError as e:
        return False, f"Connection error: {e}"


def ensure_venv() -> bool:
    """Ensure we're running in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path.cwd()


def print_table(
    data: List[Dict], title: str = "", headers: Optional[List[str]] = None
) -> None:
    """Print data as a rich table."""
    if not data:
        console.print(f"[yellow]No {title.lower()} found.[/yellow]")
        return

    table = Table(title=title)

    # Add columns
    if headers:
        for header in headers:
            table.add_column(header)
    else:
        # Use keys from first row
        for key in data[0].keys():
            table.add_column(key.replace("_", " ").title())

    # Add rows
    for row in data:
        if headers:
            table.add_row(*[str(row.get(header, "")) for header in headers])
        else:
            table.add_row(*[str(value) for value in row.values()])

    console.print(table)


def format_size(size_bytes: int) -> str:
    """Format byte size as human readable string."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    return click.confirm(message, default=default)


def echo_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓ {message}[/green]")


def echo_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗ {message}[/red]")


def echo_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠ {message}[/yellow]")


def echo_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ {message}[/blue]")


async def get_context_names() -> List[str]:
    """Get list of available context names from the server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                get_api_url("contexts"),
                timeout=5.0,
            )
            if response.status_code == 200:
                contexts = response.json()
                return [ctx["name"] for ctx in contexts]
            else:
                return []
    except httpx.RequestError:
        return []


def get_context_names_sync() -> List[str]:
    """Synchronous wrapper for getting context names."""
    try:
        import asyncio

        return asyncio.run(get_context_names())
    except Exception:
        return []
