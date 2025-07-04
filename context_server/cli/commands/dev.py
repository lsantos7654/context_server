"""Development commands for Context Server CLI."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from ..utils import (
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    ensure_venv,
    get_project_root,
    run_command,
)

console = Console()


@click.group()
@click.help_option("-h", "--help")
def dev():
    """Development commands for Context Server.

    Commands for setting up the development environment, running tests,
    formatting code, and performing quality checks.
    """
    pass


@dev.command()
@click.option("--force", is_flag=True, help="Force reinstallation even if venv exists")
@click.help_option("-h", "--help")
def init(force):
    """Initialize the development environment.

    Creates a virtual environment, installs dependencies, and sets up
    pre-commit hooks.
    """
    project_root = get_project_root()
    venv_path = project_root / ".venv"

    if venv_path.exists() and not force:
        echo_info("Virtual environment already exists")
        echo_info("Updating dependencies...")

        # Update dependencies
        run_command([str(venv_path / "bin" / "python"), "-m", "pip", "install", "uv"])
        run_command(
            [str(venv_path / "bin" / "uv"), "pip", "install", "-e", ".[dev,test]"]
        )
    else:
        if venv_path.exists():
            echo_info("Removing existing virtual environment...")
            import shutil

            shutil.rmtree(venv_path)

        echo_info("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])

        echo_info("Installing uv...")
        run_command([str(venv_path / "bin" / "python"), "-m", "pip", "install", "uv"])

        echo_info("Installing dependencies...")
        run_command(
            [str(venv_path / "bin" / "uv"), "pip", "install", "-e", ".[dev,test]"]
        )

        echo_info("Setting up pre-commit hooks...")
        try:
            run_command([str(venv_path / "bin" / "pre-commit"), "install"])
        except Exception as e:
            echo_warning(f"Failed to install pre-commit hooks: {e}")

    echo_success("Development environment initialized!")
    echo_info(f"Activate with: source {venv_path}/bin/activate")


@dev.command()
@click.option(
    "--coverage/--no-coverage", default=True, help="Run tests with coverage reporting"
)
@click.option("--watch", is_flag=True, help="Run tests in watch mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose test output")
@click.argument("path", required=False)
@click.help_option("-h", "--help")
def test(coverage, watch, verbose, path):
    """Run tests with optional coverage and watch mode.

    Args:
        path: Optional path to specific test file or directory
    """
    if not ensure_venv():
        echo_error("Please run from within the virtual environment")
        echo_info("Run: source .venv/bin/activate")
        return

    cmd = ["pytest"]

    if path:
        cmd.append(path)
    else:
        cmd.append("tests/")

    if coverage:
        cmd.extend(
            [
                "--cov=context_server",
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    if verbose:
        cmd.append("-v")

    if watch:
        # Use pytest-watch for watch mode
        cmd = ["pytest-watch"] + cmd[1:]  # Replace pytest with pytest-watch
        cmd.append("--")
        if verbose:
            cmd.append("-v")

    echo_info(f"Running: {' '.join(cmd)}")
    run_command(cmd)


@dev.command()
@click.option("--check", is_flag=True, help="Only check formatting, don't modify files")
@click.help_option("-h", "--help")
def format(check):
    """Format code with black and isort.

    Args:
        check: Only check formatting without making changes
    """
    if not ensure_venv():
        echo_error("Please run from within the virtual environment")
        return

    echo_info("Formatting with black...")
    cmd = ["black"]
    if check:
        cmd.append("--check")
    cmd.append(".")

    try:
        run_command(cmd)
        echo_success("Black formatting completed")
    except Exception as e:
        echo_error(f"Black formatting failed: {e}")

    echo_info("Sorting imports with isort...")
    cmd = ["isort"]
    if check:
        cmd.append("--check-only")
    cmd.append(".")

    try:
        run_command(cmd)
        echo_success("Import sorting completed")
    except Exception as e:
        echo_error(f"Import sorting failed: {e}")


@dev.command()
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
def lint(fix):
    """Run linting with flake8, mypy, and bandit.

    Args:
        fix: Automatically fix issues where possible
    """
    if not ensure_venv():
        echo_error("Please run from within the virtual environment")
        return

    success = True

    echo_info("Running flake8...")
    try:
        run_command(["flake8", "."])
        echo_success("Flake8 checks passed")
    except Exception as e:
        echo_error(f"Flake8 found issues: {e}")
        success = False

    echo_info("Running mypy...")
    try:
        run_command(["mypy", "context_server/", "src/"])
        echo_success("MyPy checks passed")
    except Exception as e:
        echo_error(f"MyPy found issues: {e}")
        success = False

    echo_info("Running bandit security checks...")
    try:
        run_command(["bandit", "-r", "context_server/", "src/", "--skip", "B101"])
        echo_success("Bandit security checks passed")
    except Exception as e:
        echo_error(f"Bandit found security issues: {e}")
        success = False

    if success:
        echo_success("All linting checks passed!")
    else:
        echo_error("Some linting checks failed")


@dev.command()
def quality():
    """Run full quality check: format, lint, and test.

    This command runs formatting, linting, and tests in sequence.
    """
    echo_info("Running quality checks...")

    # Run formatting
    echo_info("Step 1/3: Formatting...")
    try:
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(format, [])
        if result.exit_code != 0:
            echo_error("Formatting failed")
            return
    except Exception as e:
        echo_error(f"Formatting step failed: {e}")
        return

    # Run linting
    echo_info("Step 2/3: Linting...")
    try:
        result = runner.invoke(lint, [])
        if result.exit_code != 0:
            echo_error("Linting failed")
            return
    except Exception as e:
        echo_error(f"Linting step failed: {e}")
        return

    # Run tests
    echo_info("Step 3/3: Testing...")
    try:
        result = runner.invoke(test, [])
        if result.exit_code != 0:
            echo_error("Tests failed")
            return
    except Exception as e:
        echo_error(f"Testing step failed: {e}")
        return

    echo_success("All quality checks passed!")


@dev.command()
@click.option("--all", is_flag=True, help="Clean all caches and build artifacts")
def clean(all):
    """Clean up development artifacts.

    Args:
        all: Clean all caches and build artifacts
    """
    project_root = get_project_root()

    patterns = [
        "**/*.pyc",
        "**/__pycache__",
        "**/.pytest_cache",
        "**/.mypy_cache",
    ]

    if all:
        patterns.extend(
            [
                "build/",
                "dist/",
                "*.egg-info/",
                ".coverage",
                "htmlcov/",
            ]
        )

    echo_info("Cleaning development artifacts...")

    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            for path in project_root.glob(pattern):
                if path.is_dir():
                    echo_info(f"Removing directory: {path}")
                    import shutil

                    shutil.rmtree(path)
        else:
            # File pattern
            for path in project_root.glob(pattern):
                if path.is_file():
                    echo_info(f"Removing file: {path}")
                    path.unlink()

    echo_success("Cleanup completed!")


@dev.command()
def reset():
    """Reset the development environment.

    Removes the virtual environment and recreates it.
    """
    if not confirm_action("This will remove the virtual environment. Continue?"):
        echo_info("Reset cancelled")
        return

    project_root = get_project_root()
    venv_path = project_root / ".venv"

    if venv_path.exists():
        echo_info("Removing virtual environment...")
        import shutil

        shutil.rmtree(venv_path)

    # Re-run init
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(init, [])

    if result.exit_code == 0:
        echo_success("Development environment reset successfully!")
    else:
        echo_error("Failed to reset development environment")
