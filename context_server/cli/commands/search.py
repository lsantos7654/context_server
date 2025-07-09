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
    APIClient,
    complete_context_name,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    get_context_names_sync,
)

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def search():
    """Search documents in contexts using vector, full-text, or hybrid modes.

    Search your documentation contexts using different algorithms:
    • Vector search: Semantic similarity using embeddings
    • Full-text search: Traditional keyword matching
    • Hybrid search: Combined approach for best results

    Examples:
        ctx search query "async patterns" my-docs             # Basic hybrid search
        ctx search query "rendering" docs --mode vector       # Vector search only
        ctx search query "widgets" docs --limit 10             # More results
        ctx search code "function definition" my-docs         # Code search with voyage-code-3
        ctx search code "error handling" docs --language python # Code search with language filter
        ctx search interactive my-docs                        # Interactive mode
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
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@click.option(
    "--show-content/--no-show-content",
    default=True,
    help="Show result content snippets",
)
@rich_help_option("-h", "--help")
def query(
    query,
    context_name,
    mode,
    limit,
    output_format,
    show_content,
):
    """Search for documents in a context.

    Performs semantic or text-based search within a specific context.
    Results are ranked by relevance and optionally expanded with
    surrounding lines for better context understanding.

    Args:
        query: Search query text
        context_name: Name of context to search within
        mode: Search algorithm (vector, fulltext, hybrid)
        limit: Maximum number of results to return
        output_format: Display format (rich, table, json)
        show_content: Whether to display content snippets
    """

    async def search_documents():
        echo_info(f"Searching '{context_name}' for: {query}")
        echo_info(f"Search mode: {mode}")

        client = APIClient(timeout=60.0)  # Search can take a while
        success, response = await client.post(
            f"contexts/{context_name}/search",
            {
                "query": query,
                "mode": mode,
                "limit": limit,
            },
        )

        if success:
            results = response["results"]
            total = response["total"]
            execution_time = response["execution_time_ms"]

            if output_format == "json":
                console.print(response)
                return

            if not results:
                echo_info("No results found")
                echo_info("Try a different query or search mode")
                return

            echo_success(f"Found {total} result(s) in {execution_time}ms")
            console.print()  # Empty line

            if output_format == "table":
                display_results_table(results, show_content)
            else:  # rich format - using table format as default
                display_results_table(results, show_content)

        else:
            if "404" in str(response):
                echo_error(f"Context '{context_name}' not found")
                echo_info("Create it with: context-server context create <name>")
            else:
                echo_error(f"Search failed: {response}")
                echo_info("Make sure the server is running: context-server server up")

    asyncio.run(search_documents())


@search.command()
@click.argument("query")
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--language", help="Filter by programming language (e.g., python, javascript)")
@click.option("--limit", default=10, help="Maximum number of results")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def code(query, context_name, language, limit, output_format):
    """Search for code snippets in a context using code-optimized embeddings.

    Uses voyage-code-3 embeddings specifically designed for code search.
    Results show code snippets with syntax highlighting and metadata.

    Args:
        query: Search query focused on code (e.g., 'function definition', 'error handling')
        context_name: Name of context to search within
        language: Optional programming language filter
        limit: Maximum number of results to return
        output_format: Display format (table or json)
    """

    async def search_code():
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
                console.print(filtered_response)
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

            # Display code results in table format
            display_code_results_table(results)

        else:
            if "404" in str(response):
                echo_error(f"Context '{context_name}' not found")
                echo_info("Create it with: ctx context create <name>")
            else:
                echo_error(f"Code search failed: {response}")
                echo_info("Make sure the server is running: ctx server up")

    asyncio.run(search_code())




@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@rich_help_option("-h", "--help")
def interactive(context_name):
    """Start an interactive search session.

    Provides a continuous search interface where you can enter
    multiple queries without restarting the command. Type 'quit'
    or press Ctrl+C to exit.

    Args:
        context_name: Name of context to search within
    """
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

            # Perform search using shared client
            async def search_documents():
                client = APIClient(timeout=60.0)
                success, response = await client.post(
                    f"contexts/{context_name}/search",
                    {
                        "query": query,
                        "mode": "hybrid",
                        "limit": 5,
                    },
                )

                if success:
                    results = response["results"]
                    total = response["total"]
                    execution_time = response["execution_time_ms"]

                    if not results:
                        echo_info("No results found")
                        return

                    echo_success(f"Found {total} result(s) in {execution_time}ms")
                    console.print()
                    display_results_table(results, True)  # show_content=True for interactive

                else:
                    if "404" in str(response):
                        echo_error(f"Context '{context_name}' not found")
                    else:
                        echo_error(f"Search failed: {response}")

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
        table.add_column("Summary", style="yellow", width=30)
        table.add_column("Content", style="dim", width=40)

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
            # Show summary if available
            summary = result.get("summary", "")
            summary_display = summary[:120] + ("..." if len(summary) > 120 else "") if summary else ""
            
            # Show truncated content
            content = result["content"][:150] + (
                "..." if len(result["content"]) > 150 else ""
            )
            row.extend([summary_display, content])

        table.add_row(*row)

    console.print(table)


def display_results_rich(
    results: list, query: str, show_content: bool = True, verbose: bool = False
):
    """Display search results in rich format with highlighting."""
    for i, result in enumerate(results, 1):
        score = result["score"]
        title = result["title"]
        url = result.get("url", "")
        page_url = result.get("page_url", url)  # Use page URL if available
        content = result["content"]
        doc_id = result.get("document_id", "N/A")
        content_type = result.get("content_type", "chunk")
        base_url = result.get("base_url", "")
        is_individual_page = result.get("is_individual_page", False)

        # Create header with score, document ID, and content type
        header = (
            f"Result {i} (Score: {score:.3f}, Doc ID: {doc_id}, Type: {content_type})"
        )

        # Create title section
        title_text = f"[bold blue]{title}[/bold blue]"

        # Show page URL if different from base URL
        if page_url and page_url != base_url:
            title_text += f"\n[dim blue]{page_url}[/dim blue]"
        elif url:
            title_text += f"\n[dim blue]{url}[/dim blue]"

        # Show base URL if this is an individual page
        if is_individual_page and base_url and base_url != page_url:
            title_text += f"\n[dim cyan]From: {base_url}[/dim cyan]"

        # Get common metadata for both modes
        metadata = result.get("metadata", {})
        document_metadata = metadata.get("document", {})
        chunk_metadata = metadata.get("chunk", {})
        chunk_index = result.get("chunk_index", "N/A")

        # Add basic metadata for both modes
        title_text += f"\n[dim cyan]Document ID: {doc_id}[/dim cyan]"
        title_text += f"\n[dim cyan]Chunk ID: {result.get('id', 'N/A')}[/dim cyan]"
        title_text += (
            f"\n[dim white]Chunk {chunk_index} • Score: {score:.4f}[/dim white]"
        )
        
        # Show summary if available, or create one from content (always shown, not just in verbose mode)
        summary = result.get("summary", "")
        summary_model = result.get("summary_model", "")
        
        # If no summary exists, create a brief summary from content
        if not summary:
            content = result.get("content", "")
            summary = content[:150] + "..." if len(content) > 150 else content
            summary_type = "Content Preview"
        else:
            summary_type = "Summary"
            if summary_model:
                summary_type += f" ({summary_model})"
        
        if summary.strip():
            title_text += f"\n[bold yellow]{summary_type}:[/bold yellow]\n[italic cyan]{summary}[/italic cyan]"

        # Add document size if available
        doc_size = document_metadata.get("size", 0)
        if doc_size:
            title_text += f"\n[dim green]Document Size: {doc_size:,} chars[/dim green]"

        # Add code snippets for both modes
        code_snippets = metadata.get("code_snippets", [])
        if code_snippets:
            title_text += f"\n\n[bold yellow]Code Snippets ({len(code_snippets)} found):[/bold yellow]"

            # In non-verbose mode, show first 3 snippets; in verbose mode, show all
            snippets_to_show = code_snippets if verbose else code_snippets[:3]

            for snippet in snippets_to_show:
                snippet_id = snippet.get("id", "")
                snippet_type = snippet.get("type", "unknown")
                start_line = snippet.get("start_line", "")
                end_line = snippet.get("end_line", "")
                preview = snippet.get("preview", "")[:60]
                if len(snippet.get("preview", "")) > 60:
                    preview += "..."

                line_info = (
                    f"L{start_line}-{end_line}" if start_line and end_line else ""
                )
                title_text += f"\n[cyan]  {snippet_id}[/cyan] [yellow]{snippet_type}[/yellow] {line_info}"
                if preview:
                    title_text += f"\n[dim]    {preview}[/dim]"

            # Show count of remaining snippets in non-verbose mode
            if not verbose and len(code_snippets) > 3:
                remaining = len(code_snippets) - 3
                title_text += f"\n[dim]  ... and {remaining} more (use --verbose to see all)[/dim]"

        # Add content type indicator
        if content_type == "expanded_chunk":
            # Show actual line count for expanded context
            line_count = len(content.split("\n"))
            expansion_info = result.get("expansion_info", {})
            if expansion_info:
                original_lines = expansion_info.get("original_lines", "Unknown")
                expanded_lines = expansion_info.get("expanded_lines", "Unknown")
                title_text += (
                    f"\n[bold yellow]Expanded Context ({line_count} lines)[/bold yellow]"
                    f"\n[dim yellow]Original: {original_lines} → Expanded: {expanded_lines}[/dim yellow]"
                )
            else:
                title_text += f"\n[bold yellow]Expanded Context ({line_count} lines)[/bold yellow]"

        # Add additional metadata for verbose mode only
        if verbose:
            title_text += f"\n\n[bold cyan]Additional Metadata:[/bold cyan]"
            title_text += (
                f"\n[dim white]Content Length: {len(content):,} chars[/dim white]"
            )

            # Document metadata from JSON
            total_chunks = document_metadata.get("total_chunks", 0)
            if total_chunks:
                title_text += (
                    f"\n[dim white]Total Document Chunks: {total_chunks}[/dim white]"
                )

            total_links = document_metadata.get("total_links", 0)
            if total_links:
                title_text += (
                    f"\n[dim white]Total Document Links: {total_links}[/dim white]"
                )

            # Chunk metadata from JSON
            links_count = chunk_metadata.get("links_count", 0)
            if links_count:
                title_text += (
                    f"\n[dim white]Links in This Chunk: {links_count}[/dim white]"
                )

                # Show actual links if available
                links = chunk_metadata.get("links", {})
                if links:
                    title_text += f"\n[dim cyan]  Chunk Links:[/dim cyan]"
                    for href, link_data in list(links.items())[
                        :3
                    ]:  # Show first 3 links
                        link_text = link_data.get("text", "")
                        title_text += f"\n[dim]    {link_text}: {href}[/dim]"
                    if len(links) > 3:
                        title_text += (
                            f"\n[dim]    ... and {len(links) - 3} more links[/dim]"
                        )

        if show_content:
            # Show summary if available
            summary = result.get("summary", "")
            summary_model = result.get("summary_model", "")
            content_section = ""
            
            if summary:
                summary_header = "[bold yellow]Summary"
                if summary_model:
                    summary_header += f" ({summary_model})"
                summary_header += ":[/bold yellow]"
                content_section += f"{summary_header}\n[italic yellow]{summary}[/italic yellow]\n\n"
            
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

            # Add content section
            content_section += f"[bold white]Content:[/bold white]\n{highlighted_content}"

            # Create panel with summary and content
            panel_content = f"{title_text}\n\n{content_section}"
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


def display_code_results_table(results: list):
    """Display code search results in table format with syntax highlighting."""
    from rich.syntax import Syntax
    
    table = Table(title="Code Search Results")
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Language", style="cyan", width=12)
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Title", style="bold", width=30)
    table.add_column("Lines", style="blue", width=10)
    table.add_column("Preview", style="dim", width=50)

    for result in results:
        score = f"{result['score']:.3f}"
        language = result.get("language", "text")
        snippet_type = result.get("snippet_type", "code_block")
        
        # Extract title (truncate if too long)
        title = result.get("title", "")[:28] + ("..." if len(result.get("title", "")) > 28 else "")
        
        # Format line numbers
        start_line = result.get("start_line", "")
        end_line = result.get("end_line", "")
        line_info = f"{start_line}-{end_line}" if start_line and end_line else "N/A"
        
        # Create code preview (first 100 chars)
        content = result.get("content", "")
        preview = content[:100] + ("..." if len(content) > 100 else "")
        
        table.add_row(score, language, snippet_type, title, line_info, preview)

    console.print(table)
    console.print()
    
    # Show first result with syntax highlighting
    if results:
        first_result = results[0]
        content = first_result.get("content", "")
        language = first_result.get("language", "text")
        
        console.print(f"[bold]Top Result Preview:[/bold]")
        console.print(f"[dim]Language: {language} | Score: {first_result['score']:.3f}[/dim]")
        
        # Truncate very long code snippets
        if len(content) > 1000:
            content = content[:1000] + "\n... (truncated)"
        
        try:
            syntax = Syntax(content, language, theme="monokai", line_numbers=True)
            console.print(syntax)
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            console.print(f"[dim]{content}[/dim]")
        
        console.print()


# Alias for the query command to avoid naming conflicts
query_cmd = query


@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("query_text")
@click.option("--limit", "-l", default=5, help="Number of suggestions")
@rich_help_option("-h", "--help")
def suggest(context_name, query_text, limit):
    """Get search query suggestions.

    Args:
        context_name: Context to search in
        query_text: Query text to get suggestions for
        limit: Maximum number of suggestions
    """
    import asyncio

    from rich.table import Table

    async def get_suggestions():
        client = APIClient()
        success, response = await client.get(
            f"contexts/{context_name}/search/suggestions",
            {"query": query_text, "limit": limit},
        )

        if success:
            suggestions = response.get("suggestions", [])

            if not suggestions:
                echo_info("No suggestions available")
                return

            table = Table(title=f"Search Suggestions for '{query_text}'")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Suggestion", style="white")
            table.add_column("Score", style="yellow", width=8)

            for i, suggestion in enumerate(suggestions, 1):
                score = suggestion.get("score", 0.0)
                text = suggestion.get("text", "")
                table.add_row(str(i), text, f"{score:.3f}")

            console.print(table)
            echo_info(f"Found {len(suggestions)} suggestions")
        else:
            echo_error(f"Failed to get suggestions: {response}")

    asyncio.run(get_suggestions())
