"""Shell completion commands for Context Server CLI."""

import os
from pathlib import Path

import click
from rich.console import Console

from context_server.cli.help_formatter import rich_help_option
from context_server.cli.utils import echo_error, echo_info, echo_success, echo_warning

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def completion() -> None:
    """Shell completion setup commands.

    Commands for installing shell completion for bash, zsh, and fish.

    Examples:
        ctx completion install        # Auto-install completion
        ctx completion show bash     # Show bash completion script
        ctx completion status       # Check installation status
    """
    pass


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell to install completion for (auto-detected if not specified)",
)
@click.option("--force", is_flag=True, help="Overwrite existing completion files")
@rich_help_option("-h", "--help")
def install(shell, force) -> None:
    """Install shell completion for context-server and ctx commands.

    Args:
        shell: Target shell (bash, zsh, fish)
        force: Overwrite existing completion files
    """
    # Auto-detect shell if not specified
    if not shell:
        shell = detect_shell()
        if not shell:
            echo_error("Could not auto-detect shell. Please specify --shell")
            return

    echo_info(f"Installing {shell} completion...")

    try:
        if shell == "bash":
            install_bash_completion(force)
        elif shell == "zsh":
            install_zsh_completion(force)
        elif shell == "fish":
            install_fish_completion(force)

        echo_success(f"{shell.capitalize()} completion installed!")
        echo_info(f"Restart your shell or run: source ~/.{shell}rc")

    except Exception as e:
        echo_error(f"Failed to install completion: {e}")


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell to uninstall completion for",
)
def uninstall(shell) -> None:
    """Uninstall shell completion.

    Args:
        shell: Target shell (bash, zsh, fish)
    """
    if not shell:
        shell = detect_shell()
        if not shell:
            echo_error("Could not auto-detect shell. Please specify --shell")
            return

    echo_info(f"Uninstalling {shell} completion...")

    try:
        if shell == "bash":
            uninstall_bash_completion()
        elif shell == "zsh":
            uninstall_zsh_completion()
        elif shell == "fish":
            uninstall_fish_completion()

        echo_success(f"{shell.capitalize()} completion uninstalled!")
        echo_info(f"Restart your shell or run: source ~/.{shell}rc")

    except Exception as e:
        echo_error(f"Failed to uninstall completion: {e}")


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell to show completion for",
)
def show(shell) -> None:
    """Show completion script for manual installation.

    Args:
        shell: Target shell (bash, zsh, fish)
    """
    if not shell:
        shell = detect_shell()
        if not shell:
            echo_error("Could not auto-detect shell. Please specify --shell")
            return

    echo_info(f"Generating {shell} completion script...")

    try:
        if shell == "bash":
            script = generate_bash_completion()
        elif shell == "zsh":
            script = generate_zsh_completion()
        elif shell == "fish":
            script = generate_fish_completion()

        console.print(f"\n[bold]Add this to your ~/.{shell}rc:[/bold]\n")
        console.print(script)
        console.print()

    except Exception as e:
        echo_error(f"Failed to generate completion: {e}")


def detect_shell() -> str:
    """Auto-detect the current shell."""
    shell = os.environ.get("SHELL", "")

    if "bash" in shell:
        return "bash"
    elif "zsh" in shell:
        return "zsh"
    elif "fish" in shell:
        return "fish"

    return ""


def install_bash_completion(force: bool = False) -> None:
    """Install bash completion."""
    home = Path.home()
    bashrc = home / ".bashrc"

    # Generate completion script
    completion_script = generate_bash_completion()

    # Check if already installed
    if bashrc.exists():
        content = bashrc.read_text()
        if "context-server" in content and not force:
            echo_warning("Bash completion already installed. Use --force to overwrite.")
            return

    # Add to .bashrc
    with open(bashrc, "a") as f:
        f.write(f"\n# Context Server CLI completion\n")
        f.write(completion_script)
        f.write("\n")


def install_zsh_completion(force: bool = False) -> None:
    """Install zsh completion."""
    home = Path.home()
    zshrc = home / ".zshrc"

    # Generate completion script
    completion_script = generate_zsh_completion()

    # Check if already installed
    if zshrc.exists():
        content = zshrc.read_text()
        if "context-server" in content and not force:
            echo_warning("Zsh completion already installed. Use --force to overwrite.")
            return

    # Add to .zshrc
    with open(zshrc, "a") as f:
        f.write(f"\n# Context Server CLI completion\n")
        f.write(completion_script)
        f.write("\n")


def install_fish_completion(force: bool = False) -> None:
    """Install fish completion."""
    home = Path.home()
    fish_config_dir = home / ".config" / "fish"
    completions_dir = fish_config_dir / "completions"

    # Create completions directory if it doesn't exist
    completions_dir.mkdir(parents=True, exist_ok=True)

    # Generate completion files
    context_server_completion = completions_dir / "context-server.fish"
    ctx_completion = completions_dir / "ctx.fish"

    # Check if already installed
    if context_server_completion.exists() and not force:
        echo_warning("Fish completion already installed. Use --force to overwrite.")
        return

    # Write completion files
    fish_script = generate_fish_completion()
    context_server_completion.write_text(fish_script)
    ctx_completion.write_text(fish_script.replace("context-server", "ctx"))


def uninstall_bash_completion() -> None:
    """Uninstall bash completion."""
    home = Path.home()
    bashrc = home / ".bashrc"

    if not bashrc.exists():
        echo_info("No .bashrc file found")
        return

    # Read and filter content
    lines = bashrc.read_text().splitlines()
    filtered_lines = []
    skip_section = False

    for line in lines:
        if "Context Server CLI completion" in line:
            skip_section = True
            continue
        elif skip_section and line.strip() == "":
            skip_section = False
            continue
        elif not skip_section:
            filtered_lines.append(line)

    # Write back filtered content
    bashrc.write_text("\n".join(filtered_lines))


def uninstall_zsh_completion() -> None:
    """Uninstall zsh completion."""
    home = Path.home()
    zshrc = home / ".zshrc"

    if not zshrc.exists():
        echo_info("No .zshrc file found")
        return

    # Read and filter content
    lines = zshrc.read_text().splitlines()
    filtered_lines = []
    skip_section = False

    for line in lines:
        if "Context Server CLI completion" in line:
            skip_section = True
            continue
        elif skip_section and line.strip() == "":
            skip_section = False
            continue
        elif not skip_section:
            filtered_lines.append(line)

    # Write back filtered content
    zshrc.write_text("\n".join(filtered_lines))


def uninstall_fish_completion() -> None:
    """Uninstall fish completion."""
    home = Path.home()
    completions_dir = home / ".config" / "fish" / "completions"

    context_server_completion = completions_dir / "context-server.fish"
    ctx_completion = completions_dir / "ctx.fish"

    if context_server_completion.exists():
        context_server_completion.unlink()

    if ctx_completion.exists():
        ctx_completion.unlink()


def generate_bash_completion() -> str:
    """Generate bash completion script."""
    return """# Enable bash completion for context-server and ctx
if command -v context-server &> /dev/null; then
    eval "$(_CONTEXT_SERVER_COMPLETE=bash_source context-server)"
fi
if command -v ctx &> /dev/null; then
    eval "$(_CTX_COMPLETE=bash_source ctx)"
fi"""


def generate_zsh_completion() -> str:
    """Generate zsh completion script."""
    return """# Enable zsh completion for context-server and ctx
if command -v context-server &> /dev/null; then
    eval "$(_CONTEXT_SERVER_COMPLETE=zsh_source context-server)"
fi
if command -v ctx &> /dev/null; then
    eval "$(_CTX_COMPLETE=zsh_source ctx)"
fi"""


def generate_fish_completion() -> str:
    """Generate fish completion script."""
    return """# Fish completion for context-server
if command -v context-server > /dev/null
    complete -c context-server -f
    eval (env _CONTEXT_SERVER_COMPLETE=fish_source context-server)
end
if command -v ctx > /dev/null
    complete -c ctx -f
    eval (env _CTX_COMPLETE=fish_source ctx)
end"""


@completion.command()
@rich_help_option("-h", "--help")
def status() -> None:
    """Show completion installation status."""
    echo_info("Checking completion installation status...")

    shells = ["bash", "zsh", "fish"]

    for shell in shells:
        if shell == "bash":
            installed = check_bash_completion()
        elif shell == "zsh":
            installed = check_zsh_completion()
        elif shell == "fish":
            installed = check_fish_completion()

        status_icon = "✓" if installed else "✗"
        status_color = "green" if installed else "red"
        console.print(
            f"[{status_color}]{status_icon}[/{status_color}] {shell.capitalize()}: {'Installed' if installed else 'Not installed'}"
        )


def check_bash_completion() -> bool:
    """Check if bash completion is installed."""
    home = Path.home()
    bashrc = home / ".bashrc"

    if not bashrc.exists():
        return False

    content = bashrc.read_text()
    return "context-server" in content and "_CONTEXT_SERVER_COMPLETE" in content


def check_zsh_completion() -> bool:
    """Check if zsh completion is installed."""
    home = Path.home()
    zshrc = home / ".zshrc"

    if not zshrc.exists():
        return False

    content = zshrc.read_text()
    return "context-server" in content and "_CONTEXT_SERVER_COMPLETE" in content


def check_fish_completion() -> bool:
    """Check if fish completion is installed."""
    home = Path.home()
    completions_dir = home / ".config" / "fish" / "completions"
    context_server_completion = completions_dir / "context-server.fish"

    return context_server_completion.exists()
