"""Context management commands for Context Server CLI."""

import asyncio
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.table import Table

from ..config import get_api_url
from ..utils import (
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    get_context_names_sync,
    print_table,
)

console = Console()


def complete_context_name(ctx, param, incomplete):
    """Complete context names by fetching from server."""
    context_names = get_context_names_sync()
    return [name for name in context_names if name.startswith(incomplete)]


@click.group()
@click.help_option("-h", "--help")
def context():
    """Context management commands.

    Commands for creating, listing, and managing contexts.
    Contexts are isolated namespaces for organizing your documentation.
    """
    pass


@context.command()
@click.argument("name")
@click.option("--description", "-d", default="", help="Context description")
@click.option(
    "--embedding-model", default="text-embedding-3-small", help="Embedding model to use"
)
@click.help_option("-h", "--help")
def create(name, description, embedding_model):
    """Create a new context.

    Args:
        name: Context name (must be unique)
        description: Optional description
        embedding_model: Embedding model to use
    """

    async def create_context():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_api_url("contexts"),
                    json={
                        "name": name,
                        "description": description,
                        "embedding_model": embedding_model,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    context_data = response.json()
                    echo_success(f"Context '{name}' created successfully!")

                    # Show context details
                    table = Table(title=f"Context: {name}")
                    table.add_column("Property")
                    table.add_column("Value")

                    table.add_row("ID", context_data["id"])
                    table.add_row("Name", context_data["name"])
                    table.add_row("Description", context_data["description"] or "None")
                    table.add_row("Embedding Model", context_data["embedding_model"])
                    table.add_row("Created", str(context_data["created_at"]))

                    console.print(table)

                elif response.status_code == 409:
                    echo_error(f"Context '{name}' already exists")
                else:
                    echo_error(
                        f"Failed to create context: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(create_context())


@context.command()
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@click.help_option("-h", "--help")
def list(output_format):
    """List all contexts.

    Args:
        output_format: Output format (table or json)
    """

    async def list_contexts():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    get_api_url("contexts"),
                    timeout=30.0,
                )

                if response.status_code == 200:
                    contexts = response.json()

                    if output_format == "json":
                        console.print(contexts)
                    else:
                        if not contexts:
                            echo_info("No contexts found")
                            echo_info(
                                "Create one with: context-server context create <name>"
                            )
                            return

                        table = Table(title="Contexts")
                        table.add_column("Name")
                        table.add_column("Description")
                        table.add_column("Documents")
                        table.add_column("Model")
                        table.add_column("Created")

                        for ctx in contexts:
                            table.add_row(
                                ctx["name"],
                                ctx["description"] or "-",
                                str(ctx["document_count"]),
                                ctx["embedding_model"],
                                str(ctx["created_at"])[:19],  # Truncate timestamp
                            )

                        console.print(table)
                else:
                    echo_error(
                        f"Failed to list contexts: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(list_contexts())


@context.command()
@click.argument("name", shell_complete=complete_context_name)
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def info(name, output_format):
    """Show detailed information about a context.

    Args:
        name: Context name
        output_format: Output format (table or json)
    """

    async def get_context_info():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    get_api_url(f"contexts/{name}"),
                    timeout=30.0,
                )

                if response.status_code == 200:
                    context_data = response.json()

                    if output_format == "json":
                        console.print(context_data)
                    else:
                        table = Table(title=f"Context: {name}")
                        table.add_column("Property")
                        table.add_column("Value")

                        table.add_row("ID", context_data["id"])
                        table.add_row("Name", context_data["name"])
                        table.add_row(
                            "Description", context_data["description"] or "None"
                        )
                        table.add_row(
                            "Embedding Model", context_data["embedding_model"]
                        )
                        table.add_row(
                            "Document Count", str(context_data["document_count"])
                        )
                        table.add_row("Size", f"{context_data['size_mb']:.1f} MB")
                        table.add_row("Created", str(context_data["created_at"]))
                        table.add_row("Last Updated", str(context_data["last_updated"]))

                        console.print(table)

                elif response.status_code == 404:
                    echo_error(f"Context '{name}' not found")
                else:
                    echo_error(
                        f"Failed to get context info: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(get_context_info())


@context.command()
@click.argument("name", shell_complete=complete_context_name)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.help_option("-h", "--help")
def delete(name, force):
    """Delete a context and all its data.

    Args:
        name: Context name
        force: Skip confirmation prompt
    """
    if not force and not confirm_action(
        f"This will permanently delete context '{name}' and all its documents. Continue?",
        default=False,
    ):
        echo_info("Context deletion cancelled")
        return

    async def delete_context():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    get_api_url(f"contexts/{name}"),
                    timeout=30.0,
                )

                if response.status_code == 200:
                    echo_success(f"Context '{name}' deleted successfully!")
                elif response.status_code == 404:
                    echo_error(f"Context '{name}' not found")
                else:
                    echo_error(
                        f"Failed to delete context: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(delete_context())


@context.command()
@click.argument("old_name", shell_complete=complete_context_name)
@click.argument("new_name")
def rename(old_name, new_name):
    """Rename a context.

    Args:
        old_name: Current context name
        new_name: New context name
    """
    # This would require an API endpoint for renaming
    echo_error("Context renaming is not yet implemented")
    echo_info("As a workaround, you can create a new context and migrate documents")


@context.command()
@click.argument("name", shell_complete=complete_context_name)
@click.option("--new-description", help="New description for the context")
@click.option(
    "--new-embedding-model",
    help="New embedding model (requires re-embedding all documents)",
)
def update(name, new_description, new_embedding_model):
    """Update context properties.

    Args:
        name: Context name
        new_description: New description
        new_embedding_model: New embedding model
    """
    if not new_description and not new_embedding_model:
        echo_error("At least one property must be specified for update")
        return

    if new_embedding_model:
        echo_warning("Changing embedding model will require re-embedding all documents")
        if not confirm_action("Continue?", default=False):
            echo_info("Update cancelled")
            return

    # This would require an API endpoint for updating contexts
    echo_error("Context updating is not yet implemented")
    echo_info("Context updates will be available in a future version")
