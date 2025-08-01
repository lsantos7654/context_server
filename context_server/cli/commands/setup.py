"""Setup commands for Context Server CLI."""

import asyncio
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..claude_md_template import (
    append_to_claude_md,
    create_claude_md,
    get_claude_md_template,
    should_create_claude_md,
)
from ..config import get_config
from ..help_formatter import rich_help_option
from ..utils import check_api_health, echo_error, echo_info, echo_success, echo_warning

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def setup():
    """Setup and configuration commands for Context Server.

    Commands for initial setup, configuration management, and shell completion.

    Examples:
        ctx setup init                    # Initialize MCP integration
        ctx setup config                  # Show current configuration
        ctx setup completion bash         # Set up bash completion
    """
    pass


@setup.command()
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
        ctx setup init                    # Set up MCP in current directory
        ctx setup init --overwrite        # Update existing MCP configuration
    """
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

    # Create or enhance CLAUDE.md with intelligent documentation strategy
    echo_info("Setting up intelligent documentation strategy...")
    try:
        if should_create_claude_md(str(current_dir)):
            # No CLAUDE.md exists, run claude /init first
            echo_info("No CLAUDE.md found, running claude /init first...")
            try:
                init_result = subprocess.run(
                    ["claude", "/init", "-p", "--dangerously-skip-permissions"],
                    capture_output=True,
                    text=True,
                    cwd=current_dir,
                )
                if init_result.returncode == 0:
                    echo_success("Claude /init completed successfully")
                    # Now append our documentation strategy template
                    if append_to_claude_md(str(current_dir), get_claude_md_template()):
                        echo_success("Enhanced CLAUDE.md with documentation strategy")
                    else:
                        echo_warning(
                            "Failed to enhance CLAUDE.md with documentation strategy"
                        )
                else:
                    echo_warning(
                        f"Claude /init returned non-zero exit code: {init_result.stderr}"
                    )
                    # Continue anyway and create our own CLAUDE.md
                    if not create_claude_md(str(current_dir)):
                        echo_warning("Failed to create CLAUDE.md file")
                    else:
                        echo_success("Created CLAUDE.md with documentation strategy")
            except Exception as e:
                echo_warning(f"Failed to run claude /init: {e}")
                # Fallback: create our own CLAUDE.md
                if create_claude_md(str(current_dir)):
                    echo_success("Created CLAUDE.md with documentation strategy")
                else:
                    echo_warning("Failed to create CLAUDE.md file")
        else:
            # CLAUDE.md exists, append our strategy
            if append_to_claude_md(str(current_dir), get_claude_md_template()):
                echo_success("Enhanced existing CLAUDE.md with documentation strategy")
            else:
                echo_warning("Failed to enhance CLAUDE.md file")
    except Exception as e:
        echo_warning(f"Error setting up CLAUDE.md: {e}")

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
    echo_info("  • get_chunk - Retrieve specific chunks")
    echo_info("  • get_code_snippet - Get individual code examples")
    echo_info("  • search_code - Search code with specialized embeddings")
    echo_info("  • And more tools for job management...")
    echo_info("")
    echo_success("CLAUDE.md created/enhanced with intelligent documentation strategy!")


@setup.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    config = get_config()

    # Create main configuration table
    config_table = Table(title="Context Server Configuration", show_header=False)
    config_table.add_column("Setting", style="bold cyan", width=20)
    config_table.add_column("Value", style="white")

    if config.openai_api_key:
        masked_key = f"{'*' * 20}...{config.openai_api_key[-8:]}"
        config_table.add_row("OpenAI API Key", f"[green]{masked_key}[/green]")
    else:
        config_table.add_row("OpenAI API Key", "[red]Not configured[/red]")

    if config.voyage_api_key:
        masked_key = f"{'*' * 20}...{config.voyage_api_key[-8:]}"
        config_table.add_row("Voyage API Key", f"[green]{masked_key}[/green]")
    else:
        config_table.add_row("Voyage API Key", "[red]Not configured[/red]")

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


@setup.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@rich_help_option("-h", "--help")
def completion(shell):
    """Shell completion setup commands.

    Set up tab completion for the specified shell.

    Args:
        shell: Shell type (bash, zsh, or fish)

    Examples:
        ctx setup completion bash         # Set up bash completion
        ctx setup completion zsh          # Set up zsh completion
    """
    if shell == "bash":
        completion_script = """
# Context Server completion for bash
eval "$(_CTX_COMPLETE=bash_source ctx)"
"""
        echo_info("Add this to your ~/.bashrc:")
        console.print(Panel(completion_script.strip(), border_style="green"))

    elif shell == "zsh":
        completion_script = """
# Context Server completion for zsh
eval "$(_CTX_COMPLETE=zsh_source ctx)"
"""
        echo_info("Add this to your ~/.zshrc:")
        console.print(Panel(completion_script.strip(), border_style="green"))

    elif shell == "fish":
        completion_script = """
# Context Server completion for fish
eval (env _CTX_COMPLETE=fish_source ctx)
"""
        echo_info("Add this to your ~/.config/fish/config.fish:")
        console.print(Panel(completion_script.strip(), border_style="green"))

    echo_info(f"\nAfter adding to your shell config, restart your terminal or run:")
    echo_info(
        f"source ~/.{shell}rc"
        if shell != "fish"
        else "source ~/.config/fish/config.fish"
    )
