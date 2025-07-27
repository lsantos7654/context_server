"""Get commands for retrieving individual items from Context Server CLI."""

import asyncio
import json
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..config import get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    APIClient,
    complete_context_name,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
)

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def get():
    """Retrieve individual items (chunks, code snippets, documents) from contexts.

    Commands for accessing specific items by their IDs.
    Provides detailed information with full content and metadata.

    Examples:
        ctx get chunk my-docs chunk-id-123           # Get individual chunk
        ctx get code my-docs snippet-id-456          # Get code snippet
        ctx get document my-docs doc-id-789          # Get full document
    """
    pass


@get.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("chunk_id")
@click.option(
    "--format",
    "output_format",
    default="card",
    type=click.Choice(["card", "json", "raw"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def chunk(context_name, chunk_id, output_format):
    """Get a specific chunk by ID with full content and metadata.

    Retrieves a single chunk with its complete content, summary,
    and context information including related code snippets.

    Args:
        context_name: Context name
        chunk_id: Chunk ID (UUID from search results)
        output_format: Output format (card, json, or raw)
    """

    async def get_chunk():
        try:
            client = APIClient()
            success, response = await client.get(
                f"contexts/{context_name}/chunks/{chunk_id}"
            )

            if success:
                if output_format == "json":
                    console.print(json.dumps(response, indent=2))
                elif output_format == "raw":
                    console.print(response.get("content", ""))
                else:  # card format
                    _display_chunk_card(response)
            else:
                if "404" in str(response):
                    echo_error(
                        f"Chunk '{chunk_id}' not found in context '{context_name}'"
                    )
                else:
                    echo_error(f"Failed to get chunk: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_chunk())


@get.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("snippet_id")
@click.option(
    "--format",
    "output_format",
    default="card",
    type=click.Choice(["card", "json", "raw"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def code(context_name, snippet_id, output_format):
    """Get a specific code snippet by ID.

    Retrieves a single code snippet with its full content, metadata,
    and context information.

    Args:
        context_name: Context name
        snippet_id: Code snippet ID
        output_format: Output format (card, json, or raw)
    """

    async def get_code_snippet():
        try:
            client = APIClient()
            success, response = await client.get(
                f"contexts/{context_name}/code-snippets/{snippet_id}"
            )

            if success:
                if output_format == "json":
                    console.print(json.dumps(response, indent=2))
                elif output_format == "raw":
                    console.print(response.get("content", ""))
                else:  # card format
                    _display_code_snippet_card(response)
            else:
                if "404" in str(response):
                    echo_error(
                        f"Code snippet '{snippet_id}' not found in context '{context_name}'"
                    )
                else:
                    echo_error(f"Failed to get code snippet: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_code_snippet())


@get.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_id")
@click.option(
    "--format",
    "output_format",
    default="card",
    type=click.Choice(["card", "json", "raw"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def document(context_name, document_id, output_format):
    """Get a specific document by ID.

    Retrieves a full document with its content and metadata.

    Args:
        context_name: Context name
        document_id: Document ID
        output_format: Output format (card, json, or raw)
    """

    async def get_document():
        try:
            client = APIClient()
            success, response = await client.get(
                f"contexts/{context_name}/documents/{document_id}/raw"
            )

            if success:
                if output_format == "json":
                    console.print(json.dumps(response, indent=2))
                elif output_format == "raw":
                    console.print(response.get("content", ""))
                else:  # card format
                    _display_document_card(response)
            else:
                if "404" in str(response):
                    echo_error(
                        f"Document '{document_id}' not found in context '{context_name}'"
                    )
                else:
                    echo_error(f"Failed to get document: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_document())


def _display_chunk_card(chunk):
    """Display a single chunk as a rich card."""
    # Extract metadata
    chunk_id = chunk.get("id", "N/A")
    document_id = chunk.get("document_id", "N/A")
    chunk_index = chunk.get("chunk_index", "N/A")
    tokens = chunk.get("tokens", 0)
    content = chunk.get("content", "")
    summary = chunk.get("summary", "")
    summary_model = chunk.get("summary_model", "")

    # Get document context
    title = chunk.get("title", "")
    url = chunk.get("url", "")

    # Get line information
    start_line = chunk.get("start_line", 0)
    end_line = chunk.get("end_line", 0)
    char_start = chunk.get("char_start", 0)
    char_end = chunk.get("char_end", 0)

    # Get metadata
    metadata = chunk.get("metadata", {})
    code_snippets = metadata.get("code_snippets", [])

    # Create info section
    info_lines = []
    info_lines.append(f"[bold cyan]Chunk ID:[/bold cyan] {chunk_id}")
    info_lines.append(f"[bold cyan]Document ID:[/bold cyan] {document_id}")
    info_lines.append(f"[bold cyan]Chunk Index:[/bold cyan] {chunk_index}")
    info_lines.append(f"[bold cyan]Tokens:[/bold cyan] {tokens}")
    info_lines.append(
        f"[bold cyan]Location:[/bold cyan] Lines {start_line}-{end_line} ({char_start}-{char_end})"
    )

    if title:
        info_lines.append(f"[bold cyan]Document:[/bold cyan] {title}")
    if url:
        info_lines.append(f"[bold cyan]URL:[/bold cyan] {url}")

    # Summary
    if summary:
        summary_header = f"[bold cyan]Summary"
        if summary_model:
            summary_header += f" ({summary_model})"
        summary_header += ":[/bold cyan]"
        info_lines.append(f"{summary_header} {summary}")

    # Code snippets
    if code_snippets:
        info_lines.append(
            f"[bold cyan]Code Snippets:[/bold cyan] {len(code_snippets)} found"
        )
        for i, snippet in enumerate(code_snippets[:3], 1):  # Show first 3
            snippet_id = snippet.get("id", "N/A")
            snippet_type = snippet.get("type", "unknown")
            info_lines.append(f"  {i}. [green]{snippet_id}[/green] ({snippet_type})")
        if len(code_snippets) > 3:
            remaining = len(code_snippets) - 3
            info_lines.append(f"  ... and {remaining} more")

    info_section = "\n".join(info_lines)

    # Content section
    content_section = ""
    if content:
        content_section = f"\n\n[bold white]Content:[/bold white]\n{content}"

    # Combine sections
    panel_content = info_section + content_section

    # Create panel
    header = f"Chunk • {len(content)} chars • {tokens} tokens"
    panel = Panel(
        panel_content,
        title=f"[bold blue]{header}[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()  # Empty line


def _display_code_snippet_card(snippet):
    """Display a single code snippet as a rich card."""
    # Extract metadata
    snippet_id = snippet.get("id", "N/A")
    language = snippet.get("language", "text")
    start_line = snippet.get("start_line", 0)
    end_line = snippet.get("end_line", 0)
    char_start = snippet.get("char_start", 0)
    char_end = snippet.get("char_end", 0)
    content = snippet.get("content", "")
    snippet_type = snippet.get("type", "code_block")

    # Get document context
    doc_title = snippet.get("document_title", "")
    doc_url = snippet.get("document_url", "")

    # Calculate line count
    line_count = (
        max(1, end_line - start_line + 1)
        if start_line and end_line
        else len(content.split("\n"))
    )

    # Create info section
    info_lines = []
    info_lines.append(f"[bold cyan]ID:[/bold cyan] {snippet_id}")
    info_lines.append(f"[bold cyan]Type:[/bold cyan] {snippet_type}")
    info_lines.append(
        f"[bold cyan]Location:[/bold cyan] Lines {start_line}-{end_line} ({char_start}-{char_end})"
    )

    if doc_title:
        info_lines.append(f"[bold cyan]Document:[/bold cyan] {doc_title}")
    if doc_url:
        info_lines.append(f"[bold cyan]URL:[/bold cyan] {doc_url}")

    info_section = "\n".join(info_lines)

    # Code content with syntax highlighting
    code_content = None
    if content:
        try:
            # Use syntax highlighting for known languages
            if language == "text":
                # Try to detect language from content
                if "import " in content and ("def " in content or "class " in content):
                    language = "python"
                elif "function " in content or "const " in content or "let " in content:
                    language = "javascript"
                elif "#!/bin/bash" in content or "#!/bin/sh" in content:
                    language = "bash"

            code_content = Syntax(content, language, theme="monokai", line_numbers=True)
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            code_content = content

    # Combine info and code
    from rich.console import Group

    panel_content = (
        Group(info_section, "", code_content) if code_content else info_section
    )

    # Create panel
    header = f"Code Snippet • {line_count} lines"
    panel = Panel(
        panel_content,
        title=f"[bold green]{header}[/bold green]",
        border_style="green",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()  # Empty line


def _display_document_card(document):
    """Display a document as a rich card."""
    # Extract metadata
    doc_id = document.get("id", "N/A")
    title = document.get("title", "")
    url = document.get("url", "")
    content = document.get("content", "")
    created_at = document.get("created_at", "")

    # Create info section
    info_lines = []
    info_lines.append(f"[bold cyan]Document ID:[/bold cyan] {doc_id}")
    if title:
        info_lines.append(f"[bold cyan]Title:[/bold cyan] {title}")
    if url:
        info_lines.append(f"[bold cyan]URL:[/bold cyan] {url}")
    if created_at:
        info_lines.append(f"[bold cyan]Created:[/bold cyan] {created_at}")

    info_lines.append(f"[bold cyan]Size:[/bold cyan] {len(content):,} characters")

    info_section = "\n".join(info_lines)

    # Content preview (first 1000 chars)
    content_preview = content[:1000] + ("..." if len(content) > 1000 else "")
    content_section = (
        f"\n\n[bold white]Content Preview:[/bold white]\n{content_preview}"
    )

    # Combine sections
    panel_content = info_section + content_section

    # Create panel
    header = f"Document • {len(content):,} chars"
    panel = Panel(
        panel_content,
        title=f"[bold blue]{header}[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()  # Empty line
