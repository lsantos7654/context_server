"""Code snippet management commands for Context Server CLI."""

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
def code():
    """Manage and search code snippets within contexts.

    Commands for accessing individual code snippets, searching through code,
    and managing code-specific operations.

    Examples:
        ctx code get my-docs snippet-id-123           # Get individual code snippet
        ctx code search "function definition" my-docs  # Search for code
        ctx code list my-docs doc-id-456              # List all snippets in document
    """
    pass


@code.command()
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
def get(context_name, snippet_id, output_format):
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
                    echo_error(f"Code snippet '{snippet_id}' not found in context '{context_name}'")
                else:
                    echo_error(f"Failed to get code snippet: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_code_snippet())


@code.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_id")
@click.option(
    "--format",
    "output_format",
    default="cards",
    type=click.Choice(["cards", "json"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def list(context_name, document_id, output_format):
    """List all code snippets from a document.

    Retrieves all code snippets from a specific document with their
    summaries and metadata.

    Args:
        context_name: Context name
        document_id: Document ID
        output_format: Output format (cards or json)
    """

    async def list_code_snippets():
        try:
            client = APIClient()
            success, response = await client.get(
                f"contexts/{context_name}/documents/{document_id}/code-snippets"
            )

            if success:
                snippets = response.get("snippets", [])

                if output_format == "json":
                    console.print(json.dumps(response, indent=2))
                else:
                    if not snippets:
                        echo_info("No code snippets found in this document")
                        return

                    # Display each snippet as a card
                    for i, snippet in enumerate(snippets, 1):
                        _display_code_snippet_card(snippet, i, len(snippets))

                    echo_info(f"Found {len(snippets)} code snippets")
            else:
                if "404" in str(response):
                    echo_error(f"Document '{document_id}' not found in context '{context_name}'")
                else:
                    echo_error(f"Failed to get code snippets: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(list_code_snippets())


@code.command()
@click.argument("query")
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--language", help="Filter by programming language (e.g., python, javascript)")
@click.option("--limit", default=10, help="Maximum number of results")
@click.option(
    "--format",
    "output_format",
    default="cards",
    type=click.Choice(["cards", "json"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def search(query, context_name, language, limit, output_format):
    """Search for code snippets using code-optimized embeddings.

    Uses voyage-code-3 embeddings specifically designed for code search.
    Results show code snippets with syntax highlighting and metadata.

    Args:
        query: Search query focused on code (e.g., 'function definition', 'error handling')
        context_name: Name of context to search within
        language: Optional programming language filter
        limit: Maximum number of results to return
        output_format: Display format (cards or json)
    """

    async def search_code():
        try:
            echo_info(f"Searching code in '{context_name}' for: {query}")
            if language:
                echo_info(f"Language filter: {language}")

            client = APIClient(timeout=60.0)
            success, response = await client.post(
                f"contexts/{context_name}/search/code",
                {
                    "query": query,
                    "mode": "hybrid",
                    "limit": limit,
                },
            )

            if success:
                results = response["results"]
                total = response["total"]
                execution_time = response["execution_time_ms"]

                # Apply language filter if specified
                if language:
                    results = [r for r in results if r.get("language", "").lower() == language.lower()]
                    total = len(results)

                if output_format == "json":
                    filtered_response = {
                        **response,
                        "results": results,
                        "total": total,
                        "language_filter": language,
                    }
                    console.print(json.dumps(filtered_response, indent=2))
                    return

                if not results:
                    echo_info("No code snippets found")
                    if language:
                        echo_info(f"Try removing the language filter ({language}) or using a different query")
                    else:
                        echo_info("Try a different query or check if code snippets exist in this context")
                    return

                echo_success(f"Found {total} code snippet(s) in {execution_time}ms")
                console.print()

                # Display code results
                _display_code_results_cards(results)

            else:
                if "404" in str(response):
                    echo_error(f"Context '{context_name}' not found")
                    echo_info("Create it with: ctx context create <name>")
                else:
                    echo_error(f"Code search failed: {response}")
                    echo_info("Make sure the server is running: ctx server up")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(search_code())


def _display_code_snippet_card(snippet, index=None, total=None):
    """Display a single code snippet as a rich card."""
    # Extract metadata
    snippet_id = snippet.get("id", "N/A")
    language = snippet.get("language", "text")
    start_line = snippet.get("start_line", 0)
    end_line = snippet.get("end_line", 0)
    char_start = snippet.get("char_start", 0)
    char_end = snippet.get("char_end", 0)
    content = snippet.get("content", "")
    
    # Get summary - check different possible fields
    summary = snippet.get("summary", "") or snippet.get("preview", "")
    
    # Get document context
    doc_title = snippet.get("document_title", "")
    doc_url = snippet.get("document_url", "")
    
    # Calculate line count
    line_count = max(1, end_line - start_line + 1) if start_line and end_line else len(content.split('\n'))
    
    # Create info section
    info_lines = []
    info_lines.append(f"[bold cyan]ID:[/bold cyan] {snippet_id}")
    info_lines.append(f"[bold cyan]Language:[/bold cyan] {language}")
    info_lines.append(f"[bold cyan]Location:[/bold cyan] Lines {start_line}-{end_line} ({char_start}-{char_end})")
    
    if doc_title:
        info_lines.append(f"[bold cyan]Document:[/bold cyan] {doc_title}")
    if doc_url:
        info_lines.append(f"[bold cyan]URL:[/bold cyan] {doc_url}")
    
    # Summary
    if summary:
        info_lines.append(f"[bold cyan]Summary:[/bold cyan] {summary}")
    
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
    panel_content = Group(info_section, "", code_content) if code_content else info_section
    
    # Create panel
    if index and total:
        header = f"Code Snippet {index}/{total} • {language} • {line_count} lines"
    else:
        header = f"Code Snippet • {language} • {line_count} lines"
    
    panel = Panel(
        panel_content,
        title=f"[bold blue]{header}[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    
    console.print(panel)
    console.print()  # Empty line between cards


def _display_code_results_cards(results):
    """Display code search results in card format with syntax highlighting."""
    for i, result in enumerate(results, 1):
        # Create card header
        score = result['score']
        language = result.get("language", "text")
        snippet_type = result.get("snippet_type", "code_block")
        
        # Extract metadata
        snippet_id = result.get("id", "N/A")
        start_line = result.get("start_line", "")
        end_line = result.get("end_line", "")
        char_start = result.get("char_start", "")
        char_end = result.get("char_end", "")
        content = result.get("content", "")
        
        # Get document context
        doc_title = result.get("document_title", result.get("title", ""))
        doc_url = result.get("document_url", result.get("url", ""))
        
        # Calculate line count
        line_count = max(1, end_line - start_line + 1) if start_line and end_line else len(content.split('\n'))
        
        # Create info section
        info_lines = []
        info_lines.append(f"[bold cyan]Score:[/bold cyan] {score:.3f}")
        info_lines.append(f"[bold cyan]ID:[/bold cyan] {snippet_id}")
        info_lines.append(f"[bold cyan]Language:[/bold cyan] {language}")
        info_lines.append(f"[bold cyan]Type:[/bold cyan] {snippet_type}")
        info_lines.append(f"[bold cyan]Location:[/bold cyan] Lines {start_line}-{end_line} ({char_start}-{char_end})")
        
        if doc_title:
            info_lines.append(f"[bold cyan]Document:[/bold cyan] {doc_title}")
        if doc_url:
            info_lines.append(f"[bold cyan]URL:[/bold cyan] {doc_url}")
        
        info_section = "\n".join(info_lines)
        
        # Code content with syntax highlighting
        code_content = None
        if content:
            try:
                # Truncate very long code snippets for card view
                display_content = content
                if len(content) > 800:
                    display_content = content[:800] + "\n... (truncated)"
                
                # Use syntax highlighting for the code
                code_content = Syntax(display_content, language, theme="monokai", line_numbers=True, word_wrap=True)
            except Exception:
                # Fallback to plain text if syntax highlighting fails
                code_content = f"[dim]{display_content}[/dim]"
        
        # Combine info and code
        from rich.console import Group
        panel_content = Group(info_section, "", code_content) if code_content else info_section
        
        # Create panel
        header = f"Code Result {i} • {language} • {line_count} lines"
        panel = Panel(
            panel_content,
            title=f"[bold green]{header}[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Empty line between cards