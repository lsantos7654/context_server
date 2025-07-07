"""Main CLI entry point for Context Server."""

import click
from rich.console import Console

from .config import get_config, set_config
from .help_formatter import rich_help_option
from .utils import check_api_health, echo_error, echo_info

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
        ctx init                                  # Set up Claude integration
        ctx server up                             # Start the server
        ctx context create my-docs                # Create a new context
        ctx docs extract https://... my-docs      # Extract documentation
        ctx search query "async patterns" docs   # Search documentation
        ctx tui explorer my-docs                  # Launch interactive TUI
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
@click.option(
    "--overwrite",
    is_flag=True,
    help="Remove existing MCP configuration before adding",
)
@rich_help_option("-h", "--help")
def init(overwrite):
    """Initialize Context Server MCP integration for current project.

    Sets up Claude Code MCP tools in the current directory, allowing
    Claude to extract documentation, manage contexts, and search content.

    Examples:
        ctx init                    # Set up MCP in current directory
        ctx init --overwrite        # Update existing MCP configuration
    """
    import subprocess
    from pathlib import Path

    from .utils import echo_error, echo_info, echo_success, echo_warning

    echo_info("Initializing Context Server MCP integration...")

    # Check if we're in a valid directory
    current_dir = Path.cwd()
    echo_info(f"Setting up MCP integration in: {current_dir}")

    # Get MCP server executable path
    def _get_mcp_executable_path():
        uv_tool_path = Path.home() / ".local" / "bin" / "context-server-mcp"
        if uv_tool_path.exists() and uv_tool_path.is_file():
            return uv_tool_path
        return None

    mcp_executable_path = _get_mcp_executable_path()
    if not mcp_executable_path:
        echo_error(
            "Context Server MCP executable not found at ~/.local/bin/context-server-mcp"
        )
        echo_info("Install with: uv tool install -e . (from Context Server directory)")
        echo_info("Then verify with: uv tool list")
        return

    # Remove existing server if requested
    if overwrite:
        echo_info("Removing existing context-server MCP configuration...")
        try:
            result = subprocess.run(
                ["claude", "mcp", "remove", "context-server", "-s", "project"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                echo_success("Existing configuration removed")
            else:
                echo_info("No existing configuration found")
        except Exception as e:
            echo_warning(f"Could not remove existing configuration: {e}")

    # Add MCP server using Claude Code CLI
    echo_info("Adding context-server MCP integration...")
    try:
        cmd = [
            "claude",
            "mcp",
            "add",
            "context-server",
            str(mcp_executable_path),
            "--scope",
            "project",
            "-e",
            "CONTEXT_SERVER_URL=http://localhost:8000",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            echo_success("Context Server MCP integration initialized!")
            echo_info("MCP tools are now available in Claude Code for this project")
        else:
            echo_error(f"Failed to initialize MCP integration: {result.stderr.strip()}")
            echo_info("Make sure Claude Code CLI is installed and in your PATH")
            return
    except Exception as e:
        echo_error(f"Failed to run claude mcp command: {e}")
        echo_info("Make sure Claude Code CLI is installed and in your PATH")
        return

    # Show next steps
    echo_success("Setup complete!")
    echo_info("Next steps:")
    echo_info("  1. Start Context Server: ctx server up")
    echo_info("  2. Restart Claude Code (if running)")
    echo_info("  3. Test: Ask Claude to 'list all available contexts'")
    echo_info("")
    echo_info("Available MCP tools:")
    echo_info("  • create_context - Create documentation contexts")
    echo_info("  • extract_url - Extract docs from websites")
    echo_info("  • search_context - Search with vector/fulltext")
    echo_info("  • get_document - Retrieve full document content")
    echo_info("  • And 8 more tools for context management...")


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    import asyncio

    from rich.panel import Panel
    from rich.table import Table

    config = ctx.obj["config"]

    # Create main configuration table
    config_table = Table(title="Context Server Configuration", show_header=False)
    config_table.add_column("Setting", style="bold cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("Server Host", f"{config.server.host}:{config.server.port}")
    config_table.add_row("Database URL", config.server.database_url)
    config_table.add_row("Config Directory", str(config.config_dir))
    config_table.add_row("Verbose Mode", "✓" if config.verbose else "✗")
    config_table.add_row("Color Output", "✓" if config.color else "✗")

    if config.server.openai_api_key:
        masked_key = f"{'*' * 20}...{config.server.openai_api_key[-8:]}"
        config_table.add_row("OpenAI API Key", f"[green]{masked_key}[/green]")
    else:
        config_table.add_row("OpenAI API Key", "[red]Not configured[/red]")

    # Check server status
    async def get_status():
        try:
            healthy, error = await check_api_health()
            return (
                "[green]Running[/green]" if healthy else f"[red]Offline[/red] ({error})"
            )
        except Exception as e:
            return f"[red]Error[/red] ({str(e)})"

    status = asyncio.run(get_status())
    config_table.add_row("Server Status", status)

    # Display in a panel
    panel = Panel(
        config_table,
        title="[bold blue]Context Server[/bold blue]",
        subtitle="Use 'ctx server up' to start services",
        border_style="blue",
    )

    console.print(panel)


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
        from .commands.completion import completion

        cli.add_command(completion)
    except ImportError as e:
        echo_error(f"Failed to load completion commands: {e}")

    try:
        from .commands.tui import tui

        cli.add_command(tui)
    except ImportError as e:
        echo_error(f"Failed to load TUI commands: {e}")



# Register commands when module is imported
register_commands()


if __name__ == "__main__":
    cli()
