"""Rich-enabled help formatter for Click commands."""

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichHelpFormatter:
    """Custom Click help formatter using Rich for colorful output."""

    def __init__(self, console: Console = None):
        self.console = console or Console()

    def format_help(self, ctx, formatter):
        """Format the help page with Rich styling."""
        # Get the command and its info
        command = ctx.command

        # Create help content
        help_text = []

        # Title/Command name
        title = f"[bold cyan]{ctx.info_name}[/bold cyan]"
        if hasattr(command, "short_help") and command.short_help:
            title += f" - [dim]{command.short_help}[/dim]"

        help_text.append(title)
        help_text.append("")

        # Usage
        usage_text = self._format_usage(ctx)
        if usage_text:
            help_text.append("[bold yellow]Usage:[/bold yellow]")
            help_text.append(f"  {usage_text}")
            help_text.append("")

        # Description
        if command.help:
            help_text.append("[bold yellow]Description:[/bold yellow]")
            # Split help into main description and examples
            help_parts = command.help.split("\n\n")
            main_desc = help_parts[0]
            help_text.append(f"  {main_desc}")

            # Handle examples section
            if len(help_parts) > 1:
                for part in help_parts[1:]:
                    if "Examples:" in part or "Args:" in part:
                        help_text.append("")
                        help_text.append(self._format_examples_or_args(part))
                    else:
                        help_text.append(f"  {part}")
            help_text.append("")

        # Options
        options_help = self._format_options(ctx)
        if options_help:
            help_text.append(options_help)

        # Commands (for groups)
        if hasattr(command, "commands") and command.commands:
            commands_help = self._format_commands(command.commands)
            if commands_help:
                help_text.append(commands_help)

        # Create panel with content
        content = "\n".join(help_text)
        panel = Panel(
            content,
            title=f"[bold blue]Help: {ctx.info_name}[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)

    def _format_usage(self, ctx):
        """Format the usage line."""
        pieces = []
        pieces.append(f"[bold green]{ctx.find_root().info_name}[/bold green]")

        # Add parent command names
        parent_ctx = ctx.parent
        parent_pieces = []
        while parent_ctx and parent_ctx.info_name != ctx.find_root().info_name:
            parent_pieces.append(f"[cyan]{parent_ctx.info_name}[/cyan]")
            parent_ctx = parent_ctx.parent
        pieces.extend(reversed(parent_pieces))

        # Add current command
        if ctx.info_name != ctx.find_root().info_name:
            pieces.append(f"[cyan]{ctx.info_name}[/cyan]")

        # Add options placeholder
        pieces.append("[dim]\\[OPTIONS][/dim]")

        # Add arguments
        for param in ctx.command.params:
            if isinstance(param, click.Argument):
                if param.required:
                    pieces.append(f"[yellow]{param.name.upper()}[/yellow]")
                else:
                    pieces.append(f"[dim]\\[{param.name.upper()}][/dim]")

        return " ".join(pieces)

    def _format_options(self, ctx):
        """Format the options section."""
        options = []
        for param in ctx.command.params:
            if isinstance(param, click.Option):
                option_help = self._format_option(param)
                if option_help:
                    options.append(option_help)

        if not options:
            return ""

        # Format options as simple list instead of table
        option_lines = []
        for opt_name, opt_desc in options:
            # Pad option name to consistent width
            padded_name = f"{opt_name:<35}"
            option_lines.append(f"    {padded_name} {opt_desc}")

        options_text = "\n".join(option_lines)
        return f"[bold yellow]Options:[/bold yellow]\n{options_text}"

    def _format_option(self, param):
        """Format a single option."""
        option_names = []

        # Collect all option names
        for opt_name in param.opts:
            if opt_name.startswith("--"):
                option_names.append(f"[cyan]{opt_name}[/cyan]")
            else:
                option_names.append(f"[cyan]{opt_name}[/cyan]")

        # Add secondary opts
        for opt_name in param.secondary_opts:
            option_names.append(f"[dim cyan]{opt_name}[/dim cyan]")

        # Format the option display
        opt_display = ", ".join(option_names)

        # Add type info
        if param.type and param.type.name != "flag":
            if hasattr(param.type, "choices") and param.type.choices:
                choices = "|".join(param.type.choices)
                opt_display += f" [dim]\\[{choices}][/dim]"
            elif param.type.name != "text":
                opt_display += f" [dim]{param.type.name.upper()}[/dim]"

        # Get help text
        help_text = param.help or ""
        if param.default is not None and param.default != () and not param.is_flag:
            if param.default != "":
                help_text += f" [dim](default: {param.default})[/dim]"

        return (opt_display, help_text)

    def _format_commands(self, commands):
        """Format the commands section for command groups."""
        if not commands:
            return ""

        # Format commands as simple list
        command_lines = []
        for name, command in sorted(commands.items()):
            desc = command.short_help or command.help or ""
            if desc:
                desc = desc.split("\n")[0]  # First line only

            padded_name = f"[green]{name}[/green]"
            padded_name_plain = f"{name:<20}"  # For spacing calculation
            spacing = " " * (20 - len(name))
            command_lines.append(f"    [green]{name}[/green]{spacing} {desc}")

        commands_text = "\n".join(command_lines)
        return f"[bold yellow]Commands:[/bold yellow]\n{commands_text}"

    def _format_examples_or_args(self, text):
        """Format examples or args sections with syntax highlighting."""
        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            if line.strip().startswith("context-server") or line.strip().startswith(
                "ctx"
            ):
                # Highlight command examples
                formatted_lines.append(f"    [dim green]{line.strip()}[/dim green]")
            elif line.strip() and not line.startswith(" "):
                # Section headers
                formatted_lines.append(f"[bold yellow]{line}[/bold yellow]")
            else:
                # Regular text
                formatted_lines.append(f"  {line}")

        return "\n".join(formatted_lines)


def rich_help_option(*param_decls, **kwargs):
    """Enhanced help option that uses Rich formatting."""

    def decorator(f):
        def callback(ctx, param, value):
            if not value or ctx.resilient_parsing:
                return

            formatter = RichHelpFormatter()
            formatter.format_help(ctx, None)
            ctx.exit()

        kwargs.setdefault("is_flag", True)
        kwargs.setdefault("expose_value", False)
        kwargs.setdefault("is_eager", True)
        kwargs.setdefault("help", "Show this message and exit.")
        kwargs["callback"] = callback

        return click.option(*param_decls, **kwargs)(f)

    return decorator
