"""Main CLI entry point for Context Server."""

import click
from rich.console import Console

from .config import get_config, set_config
from .help_formatter import rich_help_option
from .utils import echo_error, echo_info

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.option(
    "--config-file", type=click.Path(exists=True), help="Path to configuration file"
)
@rich_help_option("-h", "--help")
@click.pass_context
def cli(ctx, verbose, no_color, config_file):
    """Context Server - Modern CLI for documentation RAG system.

    A powerful command-line interface for managing your local documentation
    extraction and search system. Extract docs from URLs, manage contexts,
    and perform semantic search across your documentation.

    Examples:
        ctx server up                             # Start the server
        ctx claude install                        # Set up Claude integration
        ctx context create my-docs                # Create a new context
        ctx docs extract https://... my-docs      # Extract documentation
        ctx search query "async patterns" docs   # Search documentation
        ctx docs list my-docs                     # List documents
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    config = get_config()

    # Update config from CLI options
    if verbose:
        config.verbose = verbose
    if no_color:
        config.color = False

    # Store in context
    ctx.obj["config"] = config
    set_config(config)

    # Set up console for color
    if not config.color:
        console.no_color = True


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from . import __version__

    console.print(f"Context Server CLI v{__version__}")


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    config = ctx.obj["config"]

    console.print("[bold]Context Server Configuration[/bold]")
    console.print(f"Server: {config.server.host}:{config.server.port}")
    console.print(f"Database: {config.server.database_url}")
    console.print(f"Config Directory: {config.config_dir}")
    console.print(f"Verbose: {config.verbose}")
    console.print(f"Color: {config.color}")

    if config.server.openai_api_key:
        console.print(
            f"OpenAI API Key: {'*' * 20}...{config.server.openai_api_key[-8:]}"
        )
    else:
        console.print("OpenAI API Key: Not set")


# Import and register command groups
def register_commands():
    """Register all command groups."""
    try:
        from .commands.dev import dev

        cli.add_command(dev)
    except ImportError as e:
        echo_error(f"Failed to load dev commands: {e}")

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
        from .commands.docs import docs

        cli.add_command(docs)
    except ImportError as e:
        echo_error(f"Failed to load docs commands: {e}")

    try:
        from .commands.search import search

        cli.add_command(search)
    except ImportError as e:
        echo_error(f"Failed to load search commands: {e}")

    try:
        from .commands.examples import examples

        cli.add_command(examples)
    except ImportError as e:
        echo_error(f"Failed to load examples commands: {e}")

    try:
        from .commands.completion import completion

        cli.add_command(completion)
    except ImportError as e:
        echo_error(f"Failed to load completion commands: {e}")

    try:
        from .commands.claude import claude

        cli.add_command(claude)
    except ImportError as e:
        echo_error(f"Failed to load claude commands: {e}")


# Register commands when module is imported
register_commands()


if __name__ == "__main__":
    cli()
