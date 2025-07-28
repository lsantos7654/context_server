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
        ctx search query "widget traits" docs --format mcp_json      # Compact JSON format like MCP server
        ctx search query "async patterns" docs --format cards      # Detailed card layout (default)
        ctx search code "function definition" my-docs         # Code search with voyage-code-3
        ctx search code "error handling" docs                 # Code search without filters
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
    default="cards",
    type=click.Choice(["cards", "json", "mcp_json"]),
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
        output_format: Display format (cards, json, mcp_json)
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

            if not results:
                echo_info("No results found")
                echo_info("Try a different query or search mode")
                return

            if output_format == "json":
                console.print(response)
                return

            echo_success(f"Found {total} result(s) in {execution_time}ms")
            console.print()  # Empty line

            if output_format == "mcp_json":
                # Use the shared transformation method from DatabaseManager
                from ...core.storage import DatabaseManager
                db_manager = DatabaseManager()
                compact_response = await db_manager._transform_to_compact_format(
                    results,
                    query=query,
                    mode=mode,
                    execution_time_ms=execution_time
                )
                console.print(compact_response)
            elif output_format == "cards":
                display_results_cards(results, show_content, query)
            else:  # default to cards format
                display_results_cards(results, show_content, query)

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
@click.option("--limit", default=10, help="Maximum number of results")
@click.option(
    "--format",
    "output_format",
    default="cards",
    type=click.Choice(["cards", "json", "mcp_json"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def code(query, context_name, limit, output_format):
    """Search for code snippets in a context using code-optimized embeddings.

    Uses voyage-code-3 embeddings specifically designed for code search.
    Results show code snippets with syntax highlighting and metadata.

    Args:
        query: Search query focused on code (e.g., 'function definition', 'error handling')
        context_name: Name of context to search within
        limit: Maximum number of results to return
        output_format: Display format (cards, json, mcp_json)
    """

    async def search_code():
        echo_info(f"Searching code in '{context_name}' for: {query}")

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

            if output_format == "json":
                console.print(response)
                return

            if not results:
                echo_info("No code snippets found")
                echo_info("Try a different query or check if code snippets exist in this context")
                return

            echo_success(f"Found {total} code snippet(s) in {execution_time}ms")
            console.print()

            # Display code results
            if output_format == "mcp_json":
                # Use the shared transformation method from DatabaseManager
                from ...core.storage import DatabaseManager
                db_manager = DatabaseManager()
                compact_response = db_manager._transform_code_to_compact_format(
                    results,
                    query=query,
                    execution_time_ms=execution_time
                )
                console.print(compact_response)
            else:  # cards format
                display_code_results_cards(results)

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
                    display_results_cards(results, True, query)  # show_content=True for interactive

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


def display_results_cards(results: list, show_content: bool = True, query: str = ""):
    """Display search results in card format with all metadata from MCP format."""
    for i, result in enumerate(results, 1):
        # Create simple card header
        score = result['score']
        
        header = f"Result {i} • Score: {score:.3f}"
        
        # Create card content
        card_content = []
        
        # Title and URL
        title = result["title"]
        url = result.get("url", "")
        card_content.append(f"[bold blue]{title}[/bold blue]")
        if url:
            card_content.append(f"[dim blue]{url}[/dim blue]")
        
        # Metadata information
        result_id = result.get("id", "N/A")
        doc_id = result.get("document_id", "N/A")
        content_type = result.get("content_type", "chunk")
        has_summary = bool(result.get("summary"))
        
        card_content.append("")  # Empty line
        card_content.append(f"[dim cyan]ID: {result_id}[/dim cyan]")
        card_content.append(f"[dim cyan]Doc: {doc_id}[/dim cyan]")
        card_content.append(f"[dim cyan]Type: {content_type} • Summary: {has_summary}[/dim cyan]")
        
        # Summary (if available and show_content is True)
        if show_content:
            summary = result.get("summary", "")
            if summary:
                card_content.append("")  # Empty line
                card_content.append("[bold yellow]Summary:[/bold yellow]")
                # Highlight query terms in summary
                highlighted_summary = highlight_query_terms(summary, query) if query else summary
                card_content.append(f"[italic]{highlighted_summary}[/italic]")
            
            # Content preview
            content = result.get("content", "")
            if content and len(content.strip()) > 0:
                if not summary:  # Only show content if no summary
                    card_content.append("")  # Empty line
                    card_content.append("[bold white]Content:[/bold white]")
                    # Truncate content for card view
                    preview = content[:300] + "..." if len(content) > 300 else content
                    card_content.append(f"[dim]{preview}[/dim]")
        
        # Enhanced metadata (code snippets, chunk info) - same as MCP format
        metadata = result.get("metadata", {})
        code_snippets = metadata.get("code_snippets", [])
        code_snippets_count = len(code_snippets)
        chunk_index = result.get("chunk_index", "N/A")
        
        # Generate detailed code snippet info like MCP format
        code_snippet_details = []
        if code_snippets:
            # Import the transformation logic
            from ...core.storage import DatabaseManager
            db_manager = DatabaseManager()
            
            for snippet in code_snippets:
                if isinstance(snippet, dict) and "id" in snippet:
                    snippet_detail = {
                        "id": snippet["id"],
                        "size": len(snippet.get("preview", snippet.get("content", ""))),
                        "summary": db_manager._generate_code_summary(snippet)
                    }
                    code_snippet_details.append(snippet_detail)
        
        if code_snippets_count > 0 or chunk_index != "N/A":
            card_content.append("")  # Empty line
            metadata_line = []
            if chunk_index != "N/A":
                metadata_line.append(f"Chunk: {chunk_index}")
            if code_snippets_count > 0:
                metadata_line.append(f"Code snippets: {code_snippets_count}")
            card_content.append(f"[dim cyan]{' • '.join(metadata_line)}[/dim cyan]")
            
            # Show detailed code snippet information
            if code_snippet_details:
                card_content.append("")
                card_content.append("[bold cyan]Code Snippets:[/bold cyan]")
                for snippet in code_snippet_details[:3]:  # Show first 3
                    snippet_id_full = str(snippet["id"])
                    card_content.append(f"[cyan]• {snippet_id_full}[/cyan] ({snippet['size']} chars)")
                    if snippet["summary"]:
                        # Show full code summary (no truncation for enhanced 3-4 sentence summaries)
                        summary = snippet["summary"]
                        card_content.append(f"  [dim]{summary}[/dim]")
                
                if len(code_snippet_details) > 3:
                    remaining = len(code_snippet_details) - 3
                    card_content.append(f"[dim]  ... and {remaining} more[/dim]")
        
        # Create panel
        panel = Panel(
            "\n".join(card_content),
            title=header,
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Empty line between cards


def display_results_table(results: list, show_content: bool = True):
    """Display search results in table format (legacy)."""
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

                # Show preview length instead of line numbers
                preview_length = len(snippet.get("preview", ""))
                size_info = f"({preview_length} chars)" if preview_length else ""
                title_text += f"\n[cyan]  {snippet_id}[/cyan] [yellow]{snippet_type}[/yellow] {size_info}"
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
            highlighted = pattern.sub(lambda m: f"[bold yellow]{m.group(0)}[/bold yellow]", highlighted)

    return highlighted


def display_code_results_cards(results: list):
    """Display code search results in card format with syntax highlighting."""
    for i, result in enumerate(results, 1):
        # Create card header
        score = result['score']
        snippet_type = result.get("snippet_type", "code_block")
        snippet_id = result.get("id", "N/A")
        
        header = f"Code Result {i} • Score: {score:.3f} • Type: {snippet_type}"
        
        # Create card content
        card_content = []
        
        # Snippet ID - prominently displayed
        card_content.append(f"[bold green]ID: {snippet_id}[/bold green]")
        
        # Title and URL
        title = result.get("title", "")
        if title:
            card_content.append(f"[bold blue]{title}[/bold blue]")
        
        url = result.get("url", "")
        if url:
            card_content.append(f"[dim blue]{url}[/dim blue]")
        
        # Code content with syntax highlighting
        content = result.get("content", "")
        if content:
            # Calculate and show line count (clean calculation)
            line_count = len(content.split('\n')) if content else 0
            card_content.append(f"[dim cyan]Lines: {line_count}[/dim cyan]")
            card_content.append("")  # Empty line
            card_content.append("[bold white]Code:[/bold white]")
            
            # Truncate very long code snippets for card view
            if len(content) > 800:
                content = content[:800] + "\n... (truncated)"
            
            try:
                # Try to detect language from content for syntax highlighting
                language = "text"
                if "import " in content and ("def " in content or "class " in content):
                    language = "python"
                elif "function " in content or "const " in content or "let " in content:
                    language = "javascript"
                elif "#!/bin/bash" in content or "#!/bin/sh" in content:
                    language = "bash"
                
                # Create syntax-highlighted content
                syntax = Syntax(content, language, theme="monokai", line_numbers=True, word_wrap=True)
                
                # Add the syntax object to the panel content
                card_content.append("")  # Empty line before code
            except Exception:
                # Fallback to plain text if syntax highlighting fails
                card_content.append(f"[dim]{content}[/dim]")
                syntax = None
        else:
            syntax = None
        
        # Create panel with the text content
        panel_text = "\n".join(card_content)
        
        # If we have syntax highlighting, create a panel that includes both text and syntax
        if syntax is not None:
            # Create a group with text content and syntax
            from rich.console import Group
            panel_content = Group(panel_text, syntax)
        else:
            panel_content = panel_text
        
        panel = Panel(
            panel_content,
            title=header,
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Empty line between cards


def display_code_results_table(results: list):
    """Display code search results in table format (legacy)."""
    from rich.syntax import Syntax
    
    table = Table(title="Code Search Results")
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Title", style="bold", width=30)
    table.add_column("Lines", style="blue", width=10)
    table.add_column("Preview", style="dim", width=50)

    for result in results:
        score = f"{result['score']:.3f}"
        snippet_type = result.get("snippet_type", "code_block")
        
        # Extract title (truncate if too long)
        title = result.get("title", "")[:28] + ("..." if len(result.get("title", "")) > 28 else "")
        
        # Format line count
        content = result.get("content", "")
        line_count = result.get("line_count", len(content.split('\n')) if content else 0)
        line_info = f"{line_count}" if line_count else "N/A"
        
        # Create code preview (first 100 chars)
        preview = content[:100] + ("..." if len(content) > 100 else "")
        
        table.add_row(score, snippet_type, title, line_info, preview)

    console.print(table)
    console.print()
    
    # Show first result with syntax highlighting
    if results:
        first_result = results[0]
        content = first_result.get("content", "")
        
        console.print(f"[bold]Top Result Preview:[/bold]")
        console.print(f"[dim]Score: {first_result['score']:.3f}[/dim]")
        
        # Truncate very long code snippets
        if len(content) > 1000:
            content = content[:1000] + "\n... (truncated)"
        
        try:
            # Try to detect language from content for syntax highlighting
            language = "text"
            if "import " in content and ("def " in content or "class " in content):
                language = "python"
            elif "function " in content or "const " in content or "let " in content:
                language = "javascript"
            elif "#!/bin/bash" in content or "#!/bin/sh" in content:
                language = "bash"
            
            syntax = Syntax(content, language, theme="monokai", line_numbers=True)
            console.print(syntax)
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            console.print(f"[dim]{content}[/dim]")
        
        console.print()


# Alias for the query command to avoid naming conflicts
query_cmd = query


