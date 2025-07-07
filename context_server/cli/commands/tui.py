"""TUI commands for Context Server CLI."""

import asyncio
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

from ..help_formatter import rich_help_option
from ..utils import complete_context_name, echo_error, echo_info, echo_success, echo_warning

console = Console()


def complete_search_query(ctx, param, incomplete):
    """Complete common search queries."""
    common_queries = [
        "widget",
        "ratatui",
        "async",
        "function",
        "example",
        "tutorial",
        "configuration",
        "layout",
        "event",
        "handler",
        "terminal",
        "interface",
        "component",
        "custom",
        "implementation",
        "pattern",
        "best practices",
        "performance",
        "error handling",
        "debugging"
    ]
    return [query for query in common_queries if query.startswith(incomplete.lower())]


@click.group()
@rich_help_option("-h", "--help")
def tui():
    """Launch Terminal User Interface applications.
    
    Interactive TUI applications for exploring and managing Context Server data.
    
    Examples:
        ctx tui explorer my-docs                 # Launch code snippet explorer
        ctx tui explorer test --query "widget"   # Start with pre-filled search
        
    Tab Completion:
        Use TAB to complete context names and common search queries
    """
    pass


@tui.command()
@click.argument("context_name", required=False, shell_complete=complete_context_name)
@click.option("--query", "-q", shell_complete=complete_search_query, help="Pre-fill search query")
@click.option("--server-url", default="http://localhost:8000", help="Context Server URL")
@rich_help_option("-h", "--help")
def explorer(context_name, query, server_url):
    """Launch the interactive code snippet explorer TUI.
    
    A split-pane terminal interface for searching documentation and viewing
    code snippets with syntax highlighting. Navigate with vim-style keys.
    
    Controls:
        s           - Start search mode
        Enter       - Execute search
        ↑↓ (j/k)    - Navigate search results  
        ←→ (h/l)    - Navigate code snippets
        q           - Quit application
        Esc         - Exit search mode
    
    Args:
        context_name: Name of context to search (optional, can choose in TUI)
        
    Examples:
        ctx tui explorer                        # Launch with context selection
        ctx tui explorer test                   # Launch directly in 'test' context
        ctx tui explorer docs -q "async"        # Start with search for "async"
        
    Tab Completion:
        ctx tui explorer <TAB>                  # Complete context names
        ctx tui explorer test -q <TAB>          # Complete common queries
    """
    
    echo_info("Launching Context Server TUI Explorer...")
    
    # Check if server is running
    try:
        import httpx
        with httpx.Client() as client:
            response = client.get(f"{server_url}/health", timeout=2.0)
            if response.status_code != 200:
                echo_error(f"Context Server not responding at {server_url}")
                echo_info("Start the server with: ctx server up")
                return
    except Exception as e:
        echo_error(f"Cannot connect to Context Server at {server_url}")
        echo_info("Start the server with: ctx server up")
        echo_info(f"Error: {e}")
        return
    
    # Get path to TUI executable  
    tui_executable = Path(__file__).parent.parent.parent.parent / "target" / "debug" / "context_tui_explorer"
    
    if not tui_executable.exists():
        echo_warning("TUI explorer not found, building...")
        
        # Build the Rust TUI
        build_dir = Path(__file__).parent.parent.parent.parent
        echo_info(f"Building TUI in: {build_dir}")
        
        try:
            result = subprocess.run(
                ["cargo", "build"],
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout for build
            )
            
            if result.returncode != 0:
                echo_error("Failed to build TUI explorer:")
                echo_error(result.stderr)
                return
            
            echo_success("TUI explorer built successfully!")
            
        except subprocess.TimeoutExpired:
            echo_error("Build timed out after 60 seconds")
            return
        except Exception as e:
            echo_error(f"Build failed: {e}")
            return
    
    # Prepare environment
    env = {
        "CONTEXT_SERVER_URL": server_url,
        "RUST_LOG": "warn",  # Reduce log noise
    }
    
    if context_name:
        env["DEFAULT_CONTEXT"] = context_name
    if query:
        env["DEFAULT_QUERY"] = query
    
    # Launch TUI
    echo_success("Starting TUI Explorer...")
    echo_info("Press 's' to search, 'q' to quit")
    
    try:
        # Run the TUI application
        result = subprocess.run(
            [str(tui_executable)],
            env={**dict(os.environ), **env} if 'os' in globals() else env,
            cwd=build_dir
        )
        
        if result.returncode != 0:
            echo_warning(f"TUI exited with code {result.returncode}")
        else:
            echo_info("TUI explorer closed")
            
    except KeyboardInterrupt:
        echo_info("\nTUI explorer interrupted")
    except Exception as e:
        echo_error(f"Failed to launch TUI: {e}")


@tui.command()
@click.option("--server-url", default="http://localhost:8000", help="Context Server URL")
@rich_help_option("-h", "--help") 
def status(server_url):
    """Show TUI application status and requirements.
    
    Displays information about available TUI applications, build status,
    and Context Server connectivity.
    """
    
    from rich.panel import Panel
    from rich.table import Table
    
    echo_info("Checking TUI application status...")
    
    # Create status table
    status_table = Table(title="TUI Applications Status", show_header=True)
    status_table.add_column("Component", style="bold cyan", width=20)
    status_table.add_column("Status", style="white", width=15)
    status_table.add_column("Details", style="dim")
    
    # Check Context Server
    try:
        import httpx
        with httpx.Client() as client:
            response = client.get(f"{server_url}/health", timeout=2.0)
            if response.status_code == 200:
                status_table.add_row("Context Server", "[green]✓ Running[/green]", f"{server_url}")
            else:
                status_table.add_row("Context Server", "[red]✗ Error[/red]", f"HTTP {response.status_code}")
    except Exception as e:
        status_table.add_row("Context Server", "[red]✗ Offline[/red]", str(e))
    
    # Check Rust/Cargo
    try:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            status_table.add_row("Rust/Cargo", "[green]✓ Available[/green]", version)
        else:
            status_table.add_row("Rust/Cargo", "[red]✗ Error[/red]", "Command failed")
    except Exception as e:
        status_table.add_row("Rust/Cargo", "[red]✗ Missing[/red]", "Install from rustup.rs")
    
    # Check TUI executable
    tui_executable = Path(__file__).parent.parent.parent.parent / "target" / "debug" / "context_tui_explorer"
    if tui_executable.exists():
        size_mb = tui_executable.stat().st_size / (1024 * 1024)
        status_table.add_row("TUI Explorer", "[green]✓ Built[/green]", f"{size_mb:.1f} MB")
    else:
        status_table.add_row("TUI Explorer", "[yellow]- Not Built[/yellow]", "Run 'ctx tui explorer' to build")
    
    # Check dependencies
    try:
        import httpx
        status_table.add_row("Python httpx", "[green]✓ Available[/green]", f"v{httpx.__version__}")
    except ImportError:
        status_table.add_row("Python httpx", "[red]✗ Missing[/red]", "pip install httpx")
    
    # Display panel
    panel = Panel(
        status_table,
        title="[bold blue]TUI Status Check[/bold blue]",
        subtitle="Use 'ctx tui explorer' to launch the interactive explorer",
        border_style="blue",
    )
    
    console.print(panel)
    
    # Show usage examples
    echo_info("\nAvailable TUI commands:")
    echo_info("  ctx tui explorer              # Launch code snippet explorer")
    echo_info("  ctx tui explorer test         # Start in 'test' context")  
    echo_info("  ctx tui explorer docs -q rust # Search for 'rust' in 'docs'")


# Add missing import
import os