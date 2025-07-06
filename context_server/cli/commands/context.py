"""Context management commands for Context Server CLI."""

import asyncio
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table

from ..help_formatter import rich_help_option
from ..utils import (
    APIClient,
    complete_context_name,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
)

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def context():
    """Manage documentation contexts for organizing documents.

    Contexts are containers that group related documents together.
    Each context maintains its own search index and can contain
    documents from URLs, files, and other sources.

    Examples:
        ctx context create my-docs                # Create new context
        ctx context list                          # List all contexts
        ctx context info my-docs                  # Show context details
        ctx context delete my-docs --force       # Delete context
    """
    pass


@context.command()
@click.argument("name")
@click.option("--description", "-d", default="", help="Context description")
@click.option(
    "--embedding-model", default="text-embedding-3-small", help="Embedding model to use"
)
@rich_help_option("-h", "--help")
def create(name, description, embedding_model):
    """Create a new documentation context.

    Creates a new context container for organizing and indexing
    related documents. Context names must be unique and follow
    standard naming conventions.

    Args:
        name: Unique name for the new context
        description: Optional description for the context
        embedding_model: Embedding model to use for vector search
    """

    async def create_context():
        client = APIClient()
        success, response = await client.post(
            "contexts",
            {
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
            },
        )

        if success:
            echo_success(f"Context '{name}' created successfully!")

            # Show context details
            table = Table(title=f"Context: {name}")
            table.add_column("Property")
            table.add_column("Value")

            table.add_row("ID", response["id"])
            table.add_row("Name", response["name"])
            table.add_row("Description", response["description"] or "None")
            table.add_row("Embedding Model", response["embedding_model"])
            table.add_row("Created", str(response["created_at"]))

            console.print(table)
        else:
            if "409" in str(response):
                echo_error(f"Context '{name}' already exists")
            else:
                echo_error(f"Failed to create context: {response}")
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
@rich_help_option("-h", "--help")
def list(output_format):
    """List all available contexts.

    Displays all contexts with their document counts, descriptions,
    and creation dates in table or JSON format.

    Args:
        output_format: Output format (table or json)
    """

    async def list_contexts():
        client = APIClient()
        success, response = await client.get("contexts")

        if success:
            contexts = response

            if output_format == "json":
                console.print(contexts)
            else:
                if not contexts:
                    echo_info("No contexts found")
                    echo_info("Create one with: context-server context create <name>")
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
            echo_error(f"Failed to list contexts: {response}")
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

    Displays comprehensive context metadata including document count,
    storage size, embedding model, creation date, and recent activity.

    Args:
        name: Name of context to inspect
        output_format: Output format (table or json)
    """

    async def get_context_info():
        client = APIClient()
        success, response = await client.get(f"contexts/{name}")

        if success:
            context_data = response

            if output_format == "json":
                console.print(context_data)
            else:
                table = Table(title=f"Context: {name}")
                table.add_column("Property")
                table.add_column("Value")

                table.add_row("ID", context_data["id"])
                table.add_row("Name", context_data["name"])
                table.add_row("Description", context_data["description"] or "None")
                table.add_row("Embedding Model", context_data["embedding_model"])
                table.add_row("Document Count", str(context_data["document_count"]))
                table.add_row("Size", f"{context_data['size_mb']:.1f} MB")
                table.add_row("Created", str(context_data["created_at"]))
                table.add_row("Last Updated", str(context_data["last_updated"]))

                console.print(table)
        else:
            if "404" in str(response):
                echo_error(f"Context '{name}' not found")
            else:
                echo_error(f"Failed to get context info: {response}")
                echo_info("Make sure the server is running: context-server server up")

    asyncio.run(get_context_info())


@context.command()
@click.argument("name", shell_complete=complete_context_name)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@rich_help_option("-h", "--help")
def delete(name, force):
    """Delete a context and all its documents.

    Permanently removes the context and all associated documents,
    chunks, and embeddings. This action cannot be undone.

    Args:
        name: Name of context to delete
        force: Skip confirmation prompt
    """
    if not force and not confirm_action(
        f"This will permanently delete context '{name}' and all its documents. Continue?",
        default=False,
    ):
        echo_info("Context deletion cancelled")
        return

    async def delete_context():
        client = APIClient()
        success, response = await client.delete(f"contexts/{name}")

        if success:
            echo_success(f"Context '{name}' deleted successfully!")
        else:
            if "404" in str(response):
                echo_error(f"Context '{name}' not found")
            else:
                echo_error(f"Failed to delete context: {response}")
                echo_info("Make sure the server is running: context-server server up")

    asyncio.run(delete_context())


@context.command()
@click.argument("old_name", shell_complete=complete_context_name)
@click.argument("new_name")
def rename(old_name, new_name):
    """Rename a context.

    Changes the name of an existing context while preserving
    all documents and metadata.

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

    Modifies context metadata such as description or embedding model.
    Changing the embedding model requires re-processing all documents.

    Args:
        name: Name of context to update
        new_description: Updated description
        new_embedding_model: New embedding model (triggers re-embedding)
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


@context.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--output-file", "-o", type=click.Path(), help="Output file path")
def export(context_name, output_file):
    """Export context data to a file.

    Args:
        context_name: Name of the context to export
        output_file: Output file path (optional)
    """
    import asyncio
    import json
    from pathlib import Path

    async def export_context():
        client = APIClient()
        success, response = await client.get(f"contexts/{context_name}/export")

        if success:
            # Generate output filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{context_name}_export_{timestamp}.json"
            else:
                filename = output_file

            # Write to file
            with open(filename, "w") as f:
                json.dump(response, f, indent=2)

            echo_success(f"Context '{context_name}' exported to {filename}")
        else:
            echo_error(f"Failed to export context: {response}")

    asyncio.run(export_context())


@context.command()
@click.argument("import_file", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing context")
def import_context(import_file, force):
    """Import context data from a file.

    Args:
        import_file: Path to the import file
        force: Overwrite existing context if it exists
    """
    import asyncio
    import json

    async def import_context_data():
        # Read import file
        try:
            with open(import_file, "r") as f:
                import_data = json.load(f)
        except Exception as e:
            echo_error(f"Failed to read import file: {e}")
            return

        # Add force flag to import data
        import_data["force"] = force

        client = APIClient()
        success, response = await client.post("contexts/import", import_data)

        if success:
            echo_success(
                f"Context imported successfully: {response.get('name', 'Unknown')}"
            )
        else:
            echo_error(f"Failed to import context: {response}")

    asyncio.run(import_context_data())


@context.command()
@click.argument("source_context", shell_complete=complete_context_name)
@click.argument("target_context", shell_complete=complete_context_name)
@click.option(
    "--strategy",
    type=click.Choice(["merge", "replace"]),
    default="merge",
    help="Merge strategy",
)
def merge(source_context, target_context, strategy):
    """Merge one context into another.

    Args:
        source_context: Source context to merge from
        target_context: Target context to merge into
        strategy: Merge strategy (merge or replace)
    """
    import asyncio

    async def merge_contexts():
        client = APIClient()
        merge_data = {
            "source_context_name": source_context,
            "target_context_name": target_context,
            "strategy": strategy,
        }

        success, response = await client.post("contexts/merge", merge_data)

        if success:
            echo_success(
                f"Successfully merged '{source_context}' into '{target_context}'"
            )
        else:
            echo_error(f"Failed to merge contexts: {response}")

    asyncio.run(merge_contexts())
