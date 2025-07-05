"""Search commands for Context Server CLI."""

import asyncio
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..config import get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    get_context_names_sync,
)

console = Console()


def complete_context_name(ctx, param, incomplete):
    """Complete context names by fetching from server."""
    context_names = get_context_names_sync()
    return [name for name in context_names if name.startswith(incomplete)]


@click.group()
@rich_help_option("-h", "--help")
def search():
    """🔍 Search commands for Context Server.

    Commands for searching documents within contexts using different
    search modes: vector similarity, full-text, and hybrid search.

    Examples:
        ctx search query "async patterns" my-docs          # Basic search
        ctx search query "rendering" docs --mode vector    # Vector search only
        ctx search query "widgets" docs --expand-context 10 # Expand surrounding lines
        ctx search query "concepts" docs --expand-context 50 # Get lots of context
    """
    pass


@search.command()
@click.argument("query")
@click.argument("context_name", shell_complete=complete_context_name)
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["vector", "fulltext", "hybrid"]),
    help="Search mode",
)
@click.option("--limit", default=5, help="Maximum number of results")
@click.option(
    "--format",
    "output_format",
    default="rich",
    type=click.Choice(["rich", "table", "json"]),
    help="Output format",
)
@click.option(
    "--show-content/--no-show-content",
    default=True,
    help="Show result content snippets",
)
@click.option(
    "--expand-context",
    default=0,
    type=click.IntRange(0, 300),
    help="Number of surrounding lines to include (0-300)",
)
@rich_help_option("-h", "--help")
def query(
    query,
    context_name,
    mode,
    limit,
    output_format,
    show_content,
    expand_context,
):
    """🎯 Search for documents in a context.

    Args:
        query: Search query
        context_name: Context to search in
        mode: Search mode (vector, fulltext, hybrid)
        limit: Maximum number of results
        output_format: Output format (rich, table, json)
        show_content: Show content snippets
    """

    async def search_documents():
        try:
            echo_info(f"Searching '{context_name}' for: {query}")
            echo_info(f"Search mode: {mode}")

            # Warn about large context expansion
            if expand_context > 100:
                echo_warning(
                    f"Large context expansion ({expand_context} lines) may take longer and use more memory"
                )
            elif expand_context > 50:
                echo_info(f"Expanding context by {expand_context} lines around matches")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_api_url(f"contexts/{context_name}/search"),
                    json={
                        "query": query,
                        "mode": mode,
                        "limit": limit,
                        "expand_context": expand_context,
                    },
                    timeout=60.0,  # Search can take a while
                )

                if response.status_code == 200:
                    result = response.json()
                    results = result["results"]
                    total = result["total"]
                    execution_time = result["execution_time_ms"]

                    if output_format == "json":
                        console.print(result)
                        return

                    if not results:
                        echo_info("No results found")
                        echo_info("Try a different query or search mode")
                        return

                    echo_success(f"Found {total} result(s) in {execution_time}ms")
                    console.print()  # Empty line

                    if output_format == "table":
                        display_results_table(results, show_content)
                    else:  # rich format
                        display_results_rich(results, query, show_content)

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                    echo_info("Create it with: context-server context create <name>")
                else:
                    echo_error(
                        f"Search failed: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(search_documents())


@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--limit", default=10, help="Maximum number of suggestions")
def suggest(context_name, limit):
    """💡 Get search query suggestions for a context.

    Args:
        context_name: Context name
        limit: Maximum number of suggestions
    """
    # This would require an API endpoint for search suggestions
    echo_error("Search suggestions are not yet implemented")
    echo_info("Search suggestions will be available in a future version")


@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--interactive", "-i", is_flag=True, help="Interactive search mode")
@rich_help_option("-h", "--help")
def interactive(context_name, interactive):
    """💬 Start an interactive search session.

    Args:
        context_name: Context to search in
        interactive: Enable interactive mode
    """
    if not interactive:
        echo_info("Starting interactive search session...")
        echo_info("Type 'quit' or 'exit' to stop")
        echo_info(f"Context: {context_name}")
        console.print()

    while True:
        try:
            query = console.input("[bold blue]Search query:[/bold blue] ")

            if query.lower() in ["quit", "exit", "q"]:
                echo_info("Goodbye!")
                break

            if not query.strip():
                continue

            # Perform search directly using the async function
            async def search_documents():
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            get_api_url(f"contexts/{context_name}/search"),
                            json={
                                "query": query,
                                "mode": "hybrid",
                                "limit": 5,
                                "expand_context": 0,
                            },
                            timeout=60.0,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            results = result["results"]
                            total = result["total"]
                            execution_time = result["execution_time_ms"]

                            if not results:
                                echo_info("No results found")
                                return

                            echo_success(
                                f"Found {total} result(s) in {execution_time}ms"
                            )
                            console.print()
                            display_results_rich(results, query, True)

                        elif response.status_code == 404:
                            echo_error(f"Context '{context_name}' not found")
                        else:
                            echo_error(f"Search failed: {response.status_code}")

                except httpx.RequestError as e:
                    echo_error(f"Connection error: {e}")

            asyncio.run(search_documents())
            console.print()  # Empty line between searches

        except KeyboardInterrupt:
            echo_info("\nGoodbye!")
            break
        except EOFError:
            echo_info("\nGoodbye!")
            break


def display_results_table(results: list, show_content: bool = True):
    """Display search results in table format."""
    table = Table(title="Search Results")
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Doc ID", style="cyan", width=12)
    table.add_column("Title", style="bold")
    table.add_column("URL", style="blue")

    if show_content:
        table.add_column("Content", style="dim", width=50)

    for result in results:
        score = f"{result['score']:.3f}"
        doc_id = result.get("document_id", "N/A")
        # Truncate document ID if too long
        if len(str(doc_id)) > 10:
            doc_id = str(doc_id)[:8] + ".."

        title = result["title"][:50] + ("..." if len(result["title"]) > 50 else "")

        # Truncate URL
        url = result.get("url", "")
        if len(url) > 30:
            url = url[:27] + "..."

        row = [score, str(doc_id), title, url]

        if show_content:
            content = result["content"][:200] + (
                "..." if len(result["content"]) > 200 else ""
            )
            row.append(content)

        table.add_row(*row)

    console.print(table)


def display_results_rich(results: list, query: str, show_content: bool = True):
    """Display search results in rich format with highlighting."""
    for i, result in enumerate(results, 1):
        score = result["score"]
        title = result["title"]
        url = result.get("url", "")
        content = result["content"]
        doc_id = result.get("document_id", "N/A")
        content_type = result.get("content_type", "chunk")

        # Create header with score, document ID, and content type
        header = (
            f"Result {i} (Score: {score:.3f}, Doc ID: {doc_id}, Type: {content_type})"
        )

        # Create title section
        title_text = f"[bold blue]{title}[/bold blue]"
        if url:
            title_text += f"\n[dim blue]{url}[/dim blue]"

        # Add document ID as a separate line
        title_text += f"\n[dim cyan]Document ID: {doc_id}[/dim cyan]"

        # Add content type indicator
        if content_type == "expanded_chunk":
            # Show actual line count for expanded context
            line_count = len(content.split("\n"))
            title_text += (
                f"\n[bold yellow]🔍 Expanded Context ({line_count} lines)[/bold yellow]"
            )

        if show_content:
            # Highlight query terms in content (simple highlighting)
            highlighted_content = highlight_query_terms(content, query)

            # Intelligent truncation based on content type
            max_display_length = 2000
            if content_type == "expanded_chunk":
                max_display_length = 5000  # Allow more for expanded context

            if len(highlighted_content) > max_display_length:
                highlighted_content = (
                    highlighted_content[:max_display_length]
                    + f"\n\n[dim]... (content truncated, showing first {max_display_length:,} characters)[/dim]"
                )

            # Create panel with content
            panel_content = f"{title_text}\n\n{highlighted_content}"
        else:
            panel_content = title_text

        # Display panel
        panel = Panel(
            panel_content,
            title=header,
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Empty line between results


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text."""
    # Simple highlighting - in a real implementation, this would be more sophisticated
    query_terms = query.lower().split()
    highlighted = text

    for term in query_terms:
        if len(term) > 2:  # Only highlight terms longer than 2 characters
            # Simple case-insensitive replacement
            import re

            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"[bold yellow]{term}[/bold yellow]", highlighted)

    return highlighted


# Alias for the query command to avoid naming conflicts
query_cmd = query
