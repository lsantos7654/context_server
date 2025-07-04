"""Example workflow commands for Context Server CLI."""

import asyncio

import click
from rich.console import Console

from ..utils import echo_error, echo_info, echo_success, echo_warning

console = Console()


@click.group()
@click.help_option("-h", "--help")
def examples():
    """Example workflows and setups.

    Pre-configured examples for common documentation sources.
    These commands will create contexts and extract documentation
    from popular sources to get you started quickly.
    """
    pass


@examples.command()
@click.option("--max-pages", default=20, help="Maximum pages to extract")
@click.help_option("-h", "--help")
def rust(max_pages):
    """Set up Rust standard library documentation.

    Creates a 'rust-std' context and extracts Rust documentation.

    Args:
        max_pages: Maximum pages to extract
    """
    context_name = "rust-std"
    url = "https://doc.rust-lang.org/std/"
    description = "Rust standard library documentation"

    echo_info("Setting up Rust documentation example...")

    # Create context
    echo_info(f"Creating context '{context_name}'...")
    from click.testing import CliRunner

    from .context import create as create_context

    runner = CliRunner()
    result = runner.invoke(create_context, [context_name, "--description", description])

    if result.exit_code != 0:
        echo_warning(f"Context '{context_name}' may already exist")

    # Extract documentation
    echo_info(f"Extracting documentation from {url}...")
    from .docs import extract as extract_docs

    result = runner.invoke(
        extract_docs, [url, context_name, "--max-pages", str(max_pages), "--wait"]
    )

    if result.exit_code == 0:
        echo_success("Rust documentation setup complete!")
        echo_info(f"Try: context-server search query 'HashMap' {context_name}")
        echo_info(f"Try: context-server search query 'async traits' {context_name}")
    else:
        echo_error("Failed to set up Rust documentation")


@examples.command()
@click.option("--max-pages", default=30, help="Maximum pages to extract")
@click.help_option("-h", "--help")
def fastapi(max_pages):
    """Set up FastAPI documentation.

    Creates a 'fastapi' context and extracts FastAPI documentation.

    Args:
        max_pages: Maximum pages to extract
    """
    context_name = "fastapi"
    url = "https://fastapi.tiangolo.com/"
    description = "FastAPI web framework documentation"

    echo_info("Setting up FastAPI documentation example...")

    # Create context
    echo_info(f"Creating context '{context_name}'...")
    from click.testing import CliRunner

    from .context import create as create_context

    runner = CliRunner()
    result = runner.invoke(create_context, [context_name, "--description", description])

    if result.exit_code != 0:
        echo_warning(f"Context '{context_name}' may already exist")

    # Extract documentation
    echo_info(f"Extracting documentation from {url}...")
    from .docs import extract as extract_docs

    result = runner.invoke(
        extract_docs, [url, context_name, "--max-pages", str(max_pages), "--wait"]
    )

    if result.exit_code == 0:
        echo_success("FastAPI documentation setup complete!")
        echo_info(
            f"Try: context-server search query 'dependency injection' {context_name}"
        )
        echo_info(f"Try: context-server search query 'async endpoints' {context_name}")
    else:
        echo_error("Failed to set up FastAPI documentation")


@examples.command()
@click.option("--max-pages", default=25, help="Maximum pages to extract")
def textual(max_pages):
    """Set up Textual TUI framework documentation.

    Creates a 'textual' context and extracts Textual documentation.

    Args:
        max_pages: Maximum pages to extract
    """
    context_name = "textual"
    url = "https://textual.textualize.io/"
    description = "Textual TUI framework documentation"

    echo_info("Setting up Textual documentation example...")

    # Create context
    echo_info(f"Creating context '{context_name}'...")
    from click.testing import CliRunner

    from .context import create as create_context

    runner = CliRunner()
    result = runner.invoke(create_context, [context_name, "--description", description])

    if result.exit_code != 0:
        echo_warning(f"Context '{context_name}' may already exist")

    # Extract documentation
    echo_info(f"Extracting documentation from {url}...")
    from .docs import extract as extract_docs

    result = runner.invoke(
        extract_docs, [url, context_name, "--max-pages", str(max_pages), "--wait"]
    )

    if result.exit_code == 0:
        echo_success("Textual documentation setup complete!")
        echo_info(f"Try: context-server search query 'widgets' {context_name}")
        echo_info(f"Try: context-server search query 'CSS styling' {context_name}")
    else:
        echo_error("Failed to set up Textual documentation")


@examples.command()
@click.option("--max-pages", default=40, help="Maximum pages to extract")
def python(max_pages):
    """Set up Python standard library documentation.

    Creates a 'python-std' context and extracts Python documentation.

    Args:
        max_pages: Maximum pages to extract
    """
    context_name = "python-std"
    url = "https://docs.python.org/3/"
    description = "Python standard library documentation"

    echo_info("Setting up Python documentation example...")

    # Create context
    echo_info(f"Creating context '{context_name}'...")
    from click.testing import CliRunner

    from .context import create as create_context

    runner = CliRunner()
    result = runner.invoke(create_context, [context_name, "--description", description])

    if result.exit_code != 0:
        echo_warning(f"Context '{context_name}' may already exist")

    # Extract documentation
    echo_info(f"Extracting documentation from {url}...")
    from .docs import extract as extract_docs

    result = runner.invoke(
        extract_docs, [url, context_name, "--max-pages", str(max_pages), "--wait"]
    )

    if result.exit_code == 0:
        echo_success("Python documentation setup complete!")
        echo_info(f"Try: context-server search query 'asyncio' {context_name}")
        echo_info(f"Try: context-server search query 'dataclasses' {context_name}")
    else:
        echo_error("Failed to set up Python documentation")


@examples.command()
@click.option("--max-pages", default=35, help="Maximum pages to extract")
def django(max_pages):
    """Set up Django web framework documentation.

    Creates a 'django' context and extracts Django documentation.

    Args:
        max_pages: Maximum pages to extract
    """
    context_name = "django"
    url = "https://docs.djangoproject.com/"
    description = "Django web framework documentation"

    echo_info("Setting up Django documentation example...")

    # Create context
    echo_info(f"Creating context '{context_name}'...")
    from click.testing import CliRunner

    from .context import create as create_context

    runner = CliRunner()
    result = runner.invoke(create_context, [context_name, "--description", description])

    if result.exit_code != 0:
        echo_warning(f"Context '{context_name}' may already exist")

    # Extract documentation
    echo_info(f"Extracting documentation from {url}...")
    from .docs import extract as extract_docs

    result = runner.invoke(
        extract_docs, [url, context_name, "--max-pages", str(max_pages), "--wait"]
    )

    if result.exit_code == 0:
        echo_success("Django documentation setup complete!")
        echo_info(f"Try: context-server search query 'models' {context_name}")
        echo_info(f"Try: context-server search query 'middleware' {context_name}")
    else:
        echo_error("Failed to set up Django documentation")


@examples.command()
@click.help_option("-h", "--help")
def list_all():
    """List all available example setups."""
    echo_info("Available example setups:")
    console.print()

    examples_list = [
        ("rust", "Rust standard library documentation"),
        ("fastapi", "FastAPI web framework documentation"),
        ("textual", "Textual TUI framework documentation"),
        ("python", "Python standard library documentation"),
        ("django", "Django web framework documentation"),
    ]

    from rich.table import Table

    table = Table(title="Example Workflows")
    table.add_column("Command", style="bold blue")
    table.add_column("Description", style="dim")

    for cmd, desc in examples_list:
        table.add_row(f"context-server examples {cmd}", desc)

    console.print(table)
    console.print()
    echo_info("Run any example with: context-server examples <name>")
    echo_info("All examples support --max-pages option to limit extraction")


@examples.command()
def custom():
    """Create a custom example setup.

    Interactive command to create a custom documentation extraction setup.
    """
    echo_info("Creating custom example setup...")

    try:
        # Get user input
        context_name = console.input("[bold blue]Context name:[/bold blue] ")
        description = console.input("[bold blue]Description:[/bold blue] ")
        url = console.input("[bold blue]Documentation URL:[/bold blue] ")
        max_pages = (
            console.input("[bold blue]Max pages (default 25):[/bold blue] ") or "25"
        )

        if not context_name or not url:
            echo_error("Context name and URL are required")
            return

        # Create context
        echo_info(f"Creating context '{context_name}'...")
        from click.testing import CliRunner

        from .context import create as create_context

        runner = CliRunner()
        result = runner.invoke(
            create_context, [context_name, "--description", description]
        )

        if result.exit_code != 0:
            echo_warning(f"Context '{context_name}' may already exist")

        # Extract documentation
        echo_info(f"Extracting documentation from {url}...")
        from .docs import extract as extract_docs

        result = runner.invoke(
            extract_docs, [url, context_name, "--max-pages", max_pages, "--wait"]
        )

        if result.exit_code == 0:
            echo_success(f"Custom documentation setup complete!")
            echo_info(f"Try: context-server search query '<your-query>' {context_name}")
        else:
            echo_error("Failed to set up custom documentation")

    except KeyboardInterrupt:
        echo_info("\nCustom setup cancelled")
    except EOFError:
        echo_info("\nCustom setup cancelled")
