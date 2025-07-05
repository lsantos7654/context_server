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
    """Search documents in contexts using vector, full-text, or hybrid modes.

    Search your documentation contexts using different algorithms:
    • Vector search: Semantic similarity using embeddings
    • Full-text search: Traditional keyword matching
    • Hybrid search: Combined approach for best results
    • Enhanced search: Multi-modal search with recommendations and insights

    Examples:
        ctx search query "async patterns" my-docs             # Basic hybrid search
        ctx search enhanced "patterns" docs                   # Enhanced search with recommendations
        ctx search query "rendering" docs --mode vector       # Vector search only
        ctx search query "widgets" docs --expand-context 10   # With context expansion
        ctx search query "data" docs --verbose               # Show all metadata
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show all available metadata fields",
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
    verbose,
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
                        display_results_rich(results, query, show_content, verbose)

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
@click.argument("query")
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--limit", default=5, help="Maximum number of results")
@click.option(
    "--format",
    "output_format",
    default="rich",
    type=click.Choice(["rich", "table", "json"]),
    help="Output format",
)
@click.option(
    "--include-recommendations/--no-recommendations",
    default=True,
    help="Include content recommendations",
)
@click.option(
    "--include-clusters/--no-clusters",
    default=True,
    help="Include related topic clusters",
)
@click.option(
    "--include-insights/--no-insights",
    default=True,
    help="Include knowledge graph insights",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed metadata")
@rich_help_option("-h", "--help")
def enhanced(
    query,
    context_name,
    limit,
    output_format,
    include_recommendations,
    include_clusters,
    include_insights,
    verbose,
):
    """Enhanced search with recommendations and insights.

    Uses the v2 API with multi-modal search, content recommendations,
    topic clustering, and knowledge graph insights.

    Args:
        query: Search query text
        context_name: Name of context to search within
        limit: Maximum number of results to return
        output_format: Display format (rich, table, json)
        include_recommendations: Include content recommendations
        include_clusters: Include related topic clusters
        include_insights: Include knowledge graph insights
        verbose: Show detailed metadata
    """

    async def enhanced_search():
        try:
            echo_info(f"Enhanced search in '{context_name}' for: {query}")

            async with httpx.AsyncClient() as client:
                # First, get the context ID from the name
                contexts_response = await client.get(
                    get_api_url(f"contexts/{context_name}")
                )
                if contexts_response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                    return
                elif contexts_response.status_code != 200:
                    echo_error(f"Failed to get context: {contexts_response.text}")
                    return

                context = contexts_response.json()
                context_id = context["id"]

                response = await client.post(
                    get_api_url("v2/search/enhanced"),
                    json={
                        "query": query,
                        "context_id": context_id,
                        "limit": limit,
                        "include_recommendations": include_recommendations,
                        "include_clusters": include_clusters,
                        "include_graph_insights": include_insights,
                        "enable_caching": True,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()

                    if output_format == "json":
                        console.print(result)
                        return

                    # Display search results
                    search_response = result["search_response"]
                    results = search_response["results"]

                    if not results:
                        echo_info("No results found")
                        return

                    echo_success(
                        f"Found {search_response['total_results']} result(s) in {search_response['search_time_ms']}ms"
                    )
                    console.print()

                    # Display results
                    if output_format == "table":
                        display_enhanced_results_table(results)
                    else:
                        display_enhanced_results_rich(results, query, verbose)

                    # Display recommendations if included
                    if include_recommendations and result.get(
                        "context_recommendations"
                    ):
                        display_recommendations(result["context_recommendations"])

                    # Display clusters if included
                    if include_clusters and result.get("related_clusters"):
                        display_clusters(result["related_clusters"])

                    # Display insights if included
                    if include_insights and result.get("knowledge_graph_insights"):
                        display_insights(result["knowledge_graph_insights"])

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                elif response.status_code == 501:
                    echo_warning(
                        "Enhanced search not yet available - falling back to standard search"
                    )
                    # Fall back to standard search
                    from .search import query as standard_query

                    ctx = click.get_current_context()
                    ctx.invoke(
                        standard_query,
                        query=query,
                        context_name=context_name,
                        mode="hybrid",
                        limit=limit,
                        output_format=output_format,
                        show_content=True,
                        expand_context=0,
                        verbose=verbose,
                    )
                else:
                    echo_error(
                        f"Enhanced search failed: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(enhanced_search())


@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--limit", default=10, help="Maximum number of suggestions")
def suggest(context_name, limit):
    """Get search query suggestions for a context.

    Analyzes context content to suggest relevant search queries
    based on common terms and document titles.

    Args:
        context_name: Name of context to analyze
        limit: Maximum number of suggestions to return
    """
    # This would require an API endpoint for search suggestions
    echo_error("Search suggestions are not yet implemented")
    echo_info("Search suggestions will be available in a future version")


@search.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--interactive", "-i", is_flag=True, help="Interactive search mode")
@rich_help_option("-h", "--help")
def interactive(context_name, interactive):
    """Start an interactive search session.

    Provides a continuous search interface where you can enter
    multiple queries without restarting the command. Type 'quit'
    or press Ctrl+C to exit.

    Args:
        context_name: Name of context to search within
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
                            display_results_rich(
                                results, query, True, False
                            )  # verbose=False for interactive

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

        # Add document ID as a separate line
        title_text += f"\n[dim cyan]Document ID: {doc_id}[/dim cyan]"

        # Add metadata information
        metadata = result.get("metadata", {})
        source_type = result.get("source_type", metadata.get("source_type", "Unknown"))
        chunk_index = result.get("chunk_index", "N/A")
        extraction_time = metadata.get("extraction_time", "")

        # Add source type and chunk info
        title_text += (
            f"\n[dim white]Source: {source_type} | Chunk: {chunk_index}[/dim white]"
        )

        # Add extraction timestamp if available
        if extraction_time:
            # Format the timestamp nicely
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(extraction_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                title_text += f"\n[dim white]Extracted: {formatted_time}[/dim white]"
            except:
                title_text += (
                    f"\n[dim white]Extracted: {extraction_time[:16]}[/dim white]"
                )

        # Add extraction statistics only for non-individual pages (batch-level info)
        # Individual pages should show their own extraction success status
        if source_type == "crawl4ai":
            if is_individual_page:
                # Show page-specific extraction status
                extraction_success = metadata.get("extraction_success", True)
                status_text = "successful" if extraction_success else "failed"
                title_text += f"\n[dim green]Page extraction: {status_text}[/dim green]"

                # Show content length for individual pages
                content_length = metadata.get("content_length")
                if content_length:
                    title_text += f"\n[dim white]Content length: {content_length:,} chars[/dim white]"

                # Show enhanced content analysis
                content_analysis_type = metadata.get("content_type")
                primary_language = metadata.get("primary_language")
                code_percentage = metadata.get("code_percentage", 0)

                if content_analysis_type and content_analysis_type != "general":
                    title_text += (
                        f"\n[dim cyan]Content type: {content_analysis_type}[/dim cyan]"
                    )
                if primary_language:
                    title_text += f"\n[dim cyan]Language: {primary_language}[/dim cyan]"
                if code_percentage > 5:
                    title_text += (
                        f"\n[dim cyan]Code content: {code_percentage:.1f}%[/dim cyan]"
                    )
            else:
                # Show batch statistics only for non-individual pages
                total_links = metadata.get("total_links_found")
                successful = metadata.get("successful_extractions")
                if total_links and successful:
                    title_text += f"\n[dim green]Batch extraction: {successful}/{total_links} pages successful[/dim green]"

        # Add content type indicator
        if content_type == "expanded_chunk":
            # Show actual line count for expanded context
            line_count = len(content.split("\n"))
            expansion_info = result.get("expansion_info", {})
            if expansion_info:
                original_lines = expansion_info.get("original_lines", "Unknown")
                expanded_lines = expansion_info.get("expanded_lines", "Unknown")
                title_text += (
                    f"\n[bold yellow]🔍 Expanded Context ({line_count} lines)[/bold yellow]"
                    f"\n[dim yellow]Original: {original_lines} → Expanded: {expanded_lines}[/dim yellow]"
                )
            else:
                title_text += f"\n[bold yellow]🔍 Expanded Context ({line_count} lines)[/bold yellow]"

        # Add comprehensive metadata if verbose mode is enabled
        if verbose:
            title_text += "\n\n[bold cyan]📋 Complete Metadata:[/bold cyan]"

            # Core identifiers
            title_text += f"\n[dim white]• Document ID: {doc_id}[/dim white]"
            title_text += (
                f"\n[dim white]• Chunk ID: {result.get('id', 'N/A')}[/dim white]"
            )
            title_text += f"\n[dim white]• Chunk Index: {chunk_index}[/dim white]"

            # URLs and sources
            if page_url:
                title_text += f"\n[dim white]• Page URL: {page_url}[/dim white]"
            if base_url and base_url != page_url:
                title_text += f"\n[dim white]• Base URL: {base_url}[/dim white]"
            if url and url != page_url:
                title_text += f"\n[dim white]• Document URL: {url}[/dim white]"

            # Content information
            title_text += f"\n[dim white]• Content Type: {content_type}[/dim white]"
            title_text += f"\n[dim white]• Score: {score:.6f}[/dim white]"
            content_length = len(content)
            title_text += (
                f"\n[dim white]• Content Length: {content_length:,} chars[/dim white]"
            )

            # Line tracking information
            start_line = result.get("start_line")
            end_line = result.get("end_line")
            char_start = result.get("char_start")
            char_end = result.get("char_end")

            if start_line is not None and end_line is not None:
                title_text += (
                    f"\n[dim white]• Line Range: {start_line}-{end_line}[/dim white]"
                )
            if char_start is not None and char_end is not None:
                title_text += f"\n[dim white]• Character Range: {char_start:,}-{char_end:,}[/dim white]"

            # Token information
            tokens = result.get("tokens") or metadata.get("tokens")
            if tokens:
                title_text += f"\n[dim white]• Estimated Tokens: {tokens}[/dim white]"

            # Extraction metadata
            if source_type:
                title_text += f"\n[dim white]• Source Type: {source_type}[/dim white]"
            if extraction_time:
                title_text += (
                    f"\n[dim white]• Extraction Time: {extraction_time}[/dim white]"
                )

            # File metadata for file sources
            file_path = metadata.get("file_path")
            file_type = metadata.get("file_type")
            file_size = metadata.get("file_size")
            if file_path:
                title_text += f"\n[dim white]• File Path: {file_path}[/dim white]"
            if file_type:
                title_text += f"\n[dim white]• File Type: {file_type}[/dim white]"
            if file_size:
                title_text += (
                    f"\n[dim white]• File Size: {file_size:,} bytes[/dim white]"
                )

            # Crawl4ai specific metadata
            if source_type == "crawl4ai":
                if is_individual_page:
                    # Individual page metadata
                    title_text += f"\n[dim white]• Individual Page: Yes[/dim white]"
                    extraction_success = metadata.get("extraction_success", True)
                    title_text += f"\n[dim white]• Page Extraction Success: {extraction_success}[/dim white]"
                    content_length = metadata.get("content_length")
                    if content_length:
                        title_text += f"\n[dim white]• Page Content Length: {content_length:,} chars[/dim white]"

                    # Enhanced content analysis information
                    content_analysis_type = metadata.get("content_type")
                    primary_language = metadata.get("primary_language")
                    code_percentage = metadata.get("code_percentage")
                    summary = metadata.get("summary")

                    if content_analysis_type:
                        title_text += f"\n[dim white]• Content Type: {content_analysis_type}[/dim white]"
                    if primary_language:
                        title_text += f"\n[dim white]• Primary Language: {primary_language}[/dim white]"
                    if code_percentage is not None:
                        title_text += f"\n[dim white]• Code Percentage: {code_percentage:.1f}%[/dim white]"
                    if summary and len(summary) > 10:
                        title_text += f"\n[dim white]• Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}[/dim white]"

                    # Code analysis details
                    code_blocks_count = metadata.get("code_blocks_count", 0)
                    if code_blocks_count > 0:
                        title_text += f"\n[dim white]• Code Blocks: {code_blocks_count}[/dim white]"

                    code_analysis = metadata.get("code_analysis", {})
                    if code_analysis:
                        functions = code_analysis.get("functions", [])
                        classes = code_analysis.get("classes", [])
                        if functions:
                            title_text += f"\n[dim white]• Functions: {', '.join(functions[:5])}{'...' if len(functions) > 5 else ''}[/dim white]"
                        if classes:
                            title_text += f"\n[dim white]• Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}[/dim white]"

                    # Key concepts and patterns
                    key_concepts = metadata.get("key_concepts", [])
                    if key_concepts:
                        title_text += f"\n[dim white]• Key Concepts: {', '.join(key_concepts)}[/dim white]"

                    detected_patterns = metadata.get("detected_patterns", {})
                    if detected_patterns:
                        pattern_summary = []
                        for pattern_type, patterns in detected_patterns.items():
                            if patterns:
                                pattern_summary.append(
                                    f"{pattern_type}: {len(patterns)}"
                                )
                        if pattern_summary:
                            title_text += f"\n[dim white]• Detected Patterns: {', '.join(pattern_summary)}[/dim white]"
                else:
                    # Batch-level metadata (only show for non-individual pages)
                    filtered_links = metadata.get("filtered_links")
                    total_links = metadata.get("total_links_found")
                    successful = metadata.get("successful_extractions")

                    if total_links:
                        title_text += f"\n[dim white]• Total Links Found: {total_links}[/dim white]"
                    if filtered_links:
                        title_text += f"\n[dim white]• Filtered Links: {filtered_links}[/dim white]"
                    if successful:
                        title_text += f"\n[dim white]• Successful Extractions: {successful}[/dim white]"

            # Search-specific metadata
            vector_score = result.get("vector_score")
            fulltext_score = result.get("fulltext_score")
            hybrid_score = result.get("hybrid_score")

            if vector_score is not None:
                title_text += (
                    f"\n[dim white]• Vector Score: {vector_score:.6f}[/dim white]"
                )
            if fulltext_score is not None:
                title_text += (
                    f"\n[dim white]• Full-text Score: {fulltext_score:.6f}[/dim white]"
                )
            if hybrid_score is not None:
                title_text += (
                    f"\n[dim white]• Hybrid Score: {hybrid_score:.6f}[/dim white]"
                )

            # Expansion metadata
            expansion_info = result.get("expansion_info", {})
            if expansion_info:
                title_text += f"\n[dim white]• Expansion Method: {expansion_info.get('method', 'N/A')}[/dim white]"
                title_text += f"\n[dim white]• Original Lines: {expansion_info.get('original_lines', 'N/A')}[/dim white]"
                title_text += f"\n[dim white]• Expanded Lines: {expansion_info.get('expanded_lines', 'N/A')}[/dim white]"
                lines_added = expansion_info.get("lines_added")
                if lines_added:
                    title_text += (
                        f"\n[dim white]• Lines Added: {lines_added}[/dim white]"
                    )

            # Additional metadata from the metadata object
            additional_metadata = {}
            for key, value in metadata.items():
                if key not in [
                    "extraction_time",
                    "source_type",
                    "file_path",
                    "file_type",
                    "file_size",
                    "total_links_found",
                    "filtered_links",
                    "successful_extractions",
                    "page_url",
                    "base_url",
                    "is_individual_page",
                    "extracted_pages",
                    "tokens",
                ]:
                    additional_metadata[key] = value

            if additional_metadata:
                title_text += f"\n[dim white]• Additional Metadata:[/dim white]"
                for key, value in additional_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        title_text += f"\n[dim gray]  • {key}: {value}[/dim gray]"
                    elif isinstance(value, list) and len(value) <= 3:
                        title_text += f"\n[dim gray]  • {key}: {value}[/dim gray]"
                    else:
                        title_text += (
                            f"\n[dim gray]  • {key}: <complex object>[/dim gray]"
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


def display_enhanced_results_table(results: list):
    """Display enhanced search results in table format."""
    table = Table(title="Enhanced Search Results")
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Title", style="bold")
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Language", style="magenta", width=10)
    table.add_column("Quality", style="yellow", width=8)

    for result in results:
        score = f"{result.get('score', 0):.3f}"
        title = result.get("title", "")[:40] + (
            "..." if len(result.get("title", "")) > 40 else ""
        )
        content_type = result.get("content_type", "")[:10]
        language = result.get("programming_language", "")[:8] or "N/A"
        quality = f"{result.get('quality_score', 0):.2f}"

        table.add_row(score, title, content_type, language, quality)

    console.print(table)


def display_enhanced_results_rich(results: list, query: str, verbose: bool = False):
    """Display enhanced search results in rich format."""
    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        title = result.get("title", "")
        content = result.get("content", "")
        content_type = result.get("content_type", "")
        language = result.get("programming_language", "")
        quality_score = result.get("quality_score", 0)
        matched_keywords = result.get("matched_keywords", [])

        # Create header with enhanced metadata
        header = f"Result {i} (Score: {score:.3f}, Quality: {quality_score:.2f})"
        if language:
            header += f", Language: {language}"
        if content_type:
            header += f", Type: {content_type}"

        # Create title section with enhanced info
        title_text = f"[bold blue]{title}[/bold blue]"

        if matched_keywords:
            title_text += f"\n[dim yellow]Keywords: {', '.join(matched_keywords[:5])}[/dim yellow]"

        if verbose:
            # Add comprehensive metadata
            url = result.get("url", "")
            if url:
                title_text += f"\n[dim blue]{url}[/dim blue]"

            code_elements = result.get("code_elements", [])
            api_references = result.get("api_references", [])

            if code_elements:
                title_text += (
                    f"\n[dim green]Code: {', '.join(code_elements[:3])}[/dim green]"
                )
            if api_references:
                title_text += f"\n[dim magenta]APIs: {', '.join(api_references[:3])}[/dim magenta]"

        # Highlight content
        highlighted_content = highlight_query_terms(content, query)

        # Truncate if too long
        max_length = 1500
        if len(highlighted_content) > max_length:
            highlighted_content = (
                highlighted_content[:max_length] + "\n\n[dim]... (truncated)[/dim]"
            )

        panel_content = f"{title_text}\n\n{highlighted_content}"

        panel = Panel(
            panel_content,
            title=header,
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()


def display_recommendations(recommendations: dict):
    """Display content recommendations."""
    if not recommendations:
        return

    console.print("\n[bold cyan]📋 Content Recommendations[/bold cyan]")

    primary_recs = recommendations.get("primary_recommendations", [])
    if primary_recs:
        table = Table(title="Recommended Content")
        table.add_column("Title", style="bold")
        table.add_column("Type", style="cyan")
        table.add_column("Relevance", style="green")
        table.add_column("Why Recommended", style="dim")

        for rec in primary_recs[:5]:  # Show top 5
            title = rec.get("title", "")[:40] + (
                "..." if len(rec.get("title", "")) > 40 else ""
            )
            rec_type = rec.get("recommendation_type", "")
            relevance = f"{rec.get('relevance_score', 0):.2f}"
            why = rec.get("why_recommended", "")[:50] + (
                "..." if len(rec.get("why_recommended", "")) > 50 else ""
            )

            table.add_row(title, rec_type, relevance, why)

        console.print(table)

    learning_path = recommendations.get("learning_path", [])
    if learning_path:
        console.print(
            f"\n[bold yellow]🎯 Learning Path:[/bold yellow] {' → '.join(learning_path[:3])}"
        )

    knowledge_gaps = recommendations.get("knowledge_gaps", [])
    if knowledge_gaps:
        console.print(
            f"\n[bold red]🔍 Knowledge Gaps:[/bold red] {', '.join(knowledge_gaps[:3])}"
        )


def display_clusters(clusters: list):
    """Display related topic clusters."""
    if not clusters:
        return

    console.print("\n[bold magenta]🏷️  Related Topic Clusters[/bold magenta]")

    table = Table()
    table.add_column("Cluster", style="bold")
    table.add_column("Topics", style="dim")
    table.add_column("Languages", style="cyan")
    table.add_column("Quality", style="green")

    for cluster in clusters[:5]:  # Show top 5
        name = cluster.get("name", "")[:30]
        topics = ", ".join(cluster.get("topic_keywords", [])[:3])
        languages = ", ".join(cluster.get("programming_languages", [])[:2]) or "General"
        quality = f"{cluster.get('quality_score', 0):.2f}"

        table.add_row(name, topics, languages, quality)

    console.print(table)


def display_insights(insights: dict):
    """Display knowledge graph insights."""
    if not insights:
        return

    console.print("\n[bold green]🧠 Knowledge Graph Insights[/bold green]")

    # Display key insights as bullet points
    for key, value in insights.items():
        if isinstance(value, (int, float)):
            console.print(f"• {key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, str):
            console.print(f"• {key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, list) and value:
            console.print(
                f"• {key.replace('_', ' ').title()}: {', '.join(map(str, value[:3]))}"
            )


# Alias for the query command to avoid naming conflicts
query_cmd = query
