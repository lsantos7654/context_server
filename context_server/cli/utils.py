"""Utility functions for Context Server CLI."""

import subprocess
import sys
from pathlib import Path

import click
import httpx
from pydantic import BaseModel, get_config
from rich.console import Console
from rich.table import Table

from context_server.cli.config import get_api_url
from context_server.models.api.contexts import ContextListResponse, ContextResponse
from context_server.models.api.documents import (
    DocumentContentResponse,
    DocumentsResponse,
)
from context_server.models.api.search import (
    CodeSearchResponse,
    CompactCodeSearchResponse,
    CompactSearchResponse,
    SearchResponse,
)

console = Console()


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
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


async def check_api_health() -> tuple[bool, str | None]:
    """Check if the API is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            # Health endpoint is at root level, not under /api/
            from .config import get_api_base_url

            health_url = f"{get_api_base_url()}/health"
            response = await client.get(
                health_url,
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
    data: list[dict], title: str = "", headers: list[str] | None = None
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


class APIClient:
    """Shared HTTP client for Context Server API requests."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def get(
        self, endpoint: str, params: dict = None
    ) -> tuple[bool, dict | list | str]:
        """Make GET request to API endpoint.

        Returns:
            tuple: (success: bool, response: dict|list|str)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(get_api_url(endpoint), params=params)

                if response.status_code == 200:
                    return True, response.json()
                else:
                    return False, f"HTTP {response.status_code}: {response.text}"

        except httpx.RequestError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Request failed: {e}"

    async def post(
        self, endpoint: str, data: dict = None
    ) -> tuple[bool, dict | list | str]:
        """Make POST request to API endpoint.

        Returns:
            tuple: (success: bool, response: dict|list|str)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(get_api_url(endpoint), json=data)

                if response.status_code in [200, 201]:
                    return True, response.json()
                else:
                    return False, f"HTTP {response.status_code}: {response.text}"

        except httpx.RequestError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Request failed: {e}"

    def parse_response(
        self, endpoint: str, method: str, raw_response: dict | list
    ) -> BaseModel | dict | list:
        """Parse raw HTTP response into appropriate Pydantic model based on endpoint."""
        try:
            # Handle list responses (mainly for contexts)
            if isinstance(raw_response, list):
                if "contexts" in endpoint:
                    contexts = [ContextResponse(**ctx) for ctx in raw_response]
                    return ContextListResponse(contexts=contexts, total=len(contexts))
                return raw_response  # Fallback for other lists

            # Handle dict responses based on endpoint patterns
            if not isinstance(raw_response, dict):
                return raw_response

            # Context endpoints
            if "contexts" in endpoint:
                if method == "POST":
                    return ContextResponse(**raw_response)
                elif method == "GET" and not endpoint.endswith("contexts"):
                    return ContextResponse(**raw_response)
                elif method == "DELETE":
                    # Return the raw response for delete operations (typically success messages)
                    return raw_response

            # Document endpoints
            elif "documents" in endpoint:
                if method == "GET":
                    if "documents" in endpoint and endpoint.endswith("documents"):
                        return DocumentsResponse(**raw_response)
                    else:
                        return DocumentContentResponse(**raw_response)

            # Search endpoints
            elif "search" in endpoint:
                if "/code" in endpoint:
                    # Check if compact format was requested
                    if "format=compact" in str(endpoint):
                        return CompactCodeSearchResponse(**raw_response)
                    return CodeSearchResponse(**raw_response)
                else:
                    # Check if compact format was requested
                    if "format=compact" in str(endpoint):
                        return CompactSearchResponse(**raw_response)
                    return SearchResponse(**raw_response)

            # Fallback to raw response for unknown endpoints
            return raw_response

        except Exception as e:
            # Return raw response as fallback if parsing fails
            return raw_response

    async def get_typed(
        self, endpoint: str, params: dict = None
    ) -> tuple[bool, BaseModel | dict | list | str]:
        """Make GET request and return typed Pydantic model when possible."""
        success, response = await self.get(endpoint, params)
        if success and isinstance(response, (dict, list)):
            parsed = self.parse_response(endpoint, "GET", response)
            return success, parsed
        return success, response

    async def post_typed(
        self, endpoint: str, data: dict = None
    ) -> tuple[bool, BaseModel | dict | list | str]:
        """Make POST request and return typed Pydantic model when possible."""
        success, response = await self.post(endpoint, data)
        if success and isinstance(response, (dict, list)):
            parsed = self.parse_response(endpoint, "POST", response)
            return success, parsed
        return success, response

    async def delete(
        self, endpoint: str, data: dict = None
    ) -> tuple[bool, dict | list | str]:
        """Make DELETE request to API endpoint.

        Returns:
            tuple: (success: bool, response: dict|list|str)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if data:
                    response = await client.delete(get_api_url(endpoint), json=data)
                else:
                    response = await client.delete(get_api_url(endpoint))

                if response.status_code in [200, 204]:
                    if response.status_code == 204:
                        return True, {"success": True}
                    return True, response.json()
                else:
                    return False, f"HTTP {response.status_code}: {response.text}"

        except httpx.RequestError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Request failed: {e}"

    async def delete_typed(
        self, endpoint: str, data: dict = None
    ) -> tuple[bool, BaseModel | dict | list | str]:
        """Make DELETE request and return typed Pydantic model when possible."""
        success, response = await self.delete(endpoint, data)
        if success and isinstance(response, (dict, list)):
            parsed = self.parse_response(endpoint, "DELETE", response)
            return success, parsed
        return success, response


async def get_context_names() -> list[str]:
    """Get list of available context names from the server."""
    client = APIClient(timeout=5.0)
    success, response = await client.get("contexts")

    if success and isinstance(response, list):
        return [ctx["name"] for ctx in response]
    return []


def get_context_names_sync() -> list[str]:
    """Synchronous wrapper for getting context names."""
    try:
        import asyncio

        return asyncio.run(get_context_names())
    except Exception:
        return []


def complete_context_name(ctx, param, incomplete) -> list[str]:
    """Complete context names by fetching from server."""
    context_names = get_context_names_sync()
    return [name for name in context_names if name.startswith(incomplete)]
