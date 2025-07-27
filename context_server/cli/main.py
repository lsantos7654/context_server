"""Main CLI entry point for Context Server."""

import click
from rich.console import Console

from .config import get_config, set_config
from .help_formatter import rich_help_option
from .utils import check_api_health, echo_error, echo_info

console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "-v", "--version", 
    is_flag=True, 
    help="Show version information"
)
@rich_help_option("-h", "--help")
@click.pass_context
def cli(ctx, version):
    """Context Server - Modern CLI for documentation RAG system.

    A powerful command-line interface for managing your local documentation
    extraction and search system. Extract docs from URLs, manage contexts,
    and perform semantic search across your documentation.

    Examples:
        ctx -v                                    # Show version
        ctx setup init                            # Set up Claude integration
        ctx server up                             # Start the server
        ctx context create my-docs                # Create a new context
        ctx extract https://... my-docs           # Extract documentation
        ctx search query "async patterns" docs   # Search documentation
        ctx get chunk my-docs chunk-id-123        # Get individual chunk
    """
    # Handle version flag
    if version:
        from . import __version__
        console.print(f"Context Server CLI v{__version__}")
        ctx.exit()

    # If no subcommand is provided and no version flag, show help
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())
        ctx.exit()

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    config = get_config()

    # Store in context
    ctx.obj["config"] = config
    set_config(config)

    # Set up console for color
    if not config.color:
        console.no_color = True


# Setup commands (init, config, completion) moved to 'ctx setup' command group
# Version command removed - use 'ctx -v' instead


# Import and register command groups
def register_commands():
    """Register all command groups."""
    try:
        from .commands.server import server

        cli.add_command(server)
    except ImportError as e:
        echo_error(f"Failed to load server commands: {e}")

    try:
        from .commands.context import context

        cli.add_command(context)
    except ImportError as e:
        echo_error(f"Failed to load context commands: {e}")

    try:
        from .commands.extract import extract

        cli.add_command(extract)
    except ImportError as e:
        echo_error(f"Failed to load extract command: {e}")

    try:
        from .commands.search import search

        cli.add_command(search)
    except ImportError as e:
        echo_error(f"Failed to load search commands: {e}")

    try:
        from .commands.setup import setup

        cli.add_command(setup)
    except ImportError as e:
        echo_error(f"Failed to load setup commands: {e}")

    # Code commands moved to 'ctx search code' and 'ctx get code'
    # Remove redundant 'ctx code' command group

    try:
        from .commands.job import job

        cli.add_command(job)
    except ImportError as e:
        echo_error(f"Failed to load job commands: {e}")

    try:
        from .commands.get import get

        cli.add_command(get)
    except ImportError as e:
        echo_error(f"Failed to load get commands: {e}")



# Register commands when module is imported
register_commands()


if __name__ == "__main__":
    cli()
