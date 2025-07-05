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
    """Search documents in contexts using enhanced multi-modal search.

    Enhanced search provides intelligent multi-modal search with:
    • Semantic similarity using embeddings
    • Content recommendations and insights
    • Related topic clusters
    • Knowledge graph insights
    • Multi-strategy search optimization

    Examples:
        ctx search query "async patterns" my-docs                    # Enhanced search
        ctx search query "patterns" docs --no-recommendations        # Without recommendations
        ctx search query "widgets" docs --no-clusters               # Without topic clusters
        ctx search query "data" docs --verbose                      # Show detailed metadata
        ctx search interactive my-docs                               # Interactive mode
    """
    pass


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
@click.option(
    "--expand-context",
    type=int,
    default=0,
    help="Number of lines to expand around each result for fuller context",
)
@rich_help_option("-h", "--help")
def query(
    query,
    context_name,
    limit,
    output_format,
    include_recommendations,
    include_clusters,
    include_insights,
    verbose,
    expand_context,
):
    """Search for documents in a context using enhanced multi-modal search.

    Performs intelligent search with content recommendations, topic clustering,
    and knowledge graph insights for comprehensive results. Can expand results
    with surrounding context using Redis-cached line-based expansion.

    Args:
        query: Search query text
        context_name: Name of context to search within
        limit: Maximum number of results to return
        output_format: Display format (rich, table, json)
        include_recommendations: Include content recommendations
        include_clusters: Include related topic clusters
        include_insights: Include knowledge graph insights
        verbose: Show detailed metadata
        expand_context: Number of lines to expand around each result
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

                    # Apply context expansion if requested
                    if expand_context > 0:
                        echo_info(f"Expanding context by {expand_context} lines...")
                        results = await expand_search_results(
                            results, expand_context, context_id
                        )

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
                else:
                    echo_error(
                        f"Enhanced search failed: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(enhanced_search())


# The enhanced command has been replaced by the query command above


# Search suggestions command removed - will be implemented with v2 API later


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

            # Perform enhanced search using v2 API
            async def enhanced_search():
                try:
                    async with httpx.AsyncClient() as client:
                        # First, get the context ID from the name
                        contexts_response = await client.get(
                            get_api_url(f"contexts/{context_name}")
                        )
                        if contexts_response.status_code == 404:
                            echo_error(f"Context '{context_name}' not found")
                            return
                        elif contexts_response.status_code != 200:
                            echo_error(
                                f"Failed to get context: {contexts_response.text}"
                            )
                            return

                        context = contexts_response.json()
                        context_id = context["id"]

                        response = await client.post(
                            get_api_url("v2/search/enhanced"),
                            json={
                                "query": query,
                                "context_id": context_id,
                                "limit": 5,
                                "include_recommendations": False,  # Simplified for interactive
                                "include_clusters": False,
                                "include_graph_insights": False,
                                "enable_caching": True,
                            },
                            timeout=60.0,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            search_response = result["search_response"]
                            results = search_response["results"]

                            if not results:
                                echo_info("No results found")
                                return

                            echo_success(
                                f"Found {search_response['total_results']} result(s) in {search_response['search_time_ms']}ms"
                            )
                            console.print()
                            display_enhanced_results_rich(
                                results, query, False  # verbose=False for interactive
                            )

                        elif response.status_code == 404:
                            echo_error(f"Context '{context_name}' not found")
                        else:
                            echo_error(
                                f"Enhanced search failed: {response.status_code}"
                            )

                except httpx.RequestError as e:
                    echo_error(f"Connection error: {e}")

            asyncio.run(enhanced_search())
            console.print()  # Empty line between searches

        except KeyboardInterrupt:
            echo_info("\nGoodbye!")
            break
        except EOFError:
            echo_info("\nGoodbye!")
            break


# Old v1 display_results_table function removed - using enhanced versions only


# Old v1 display_results_rich function removed - using enhanced versions only
# This was a very long function that handled v1 API response format


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


async def expand_search_results(
    results: list, expand_lines: int, context_id: str
) -> list:
    """
    Expand search results with surrounding context using the ContextExpansionService.

    Args:
        results: List of search result dictionaries
        expand_lines: Number of lines to expand above and below each result
        context_id: Context ID for database operations

    Returns:
        List of expanded search results
    """
    try:
        # Import the required services
        from ...core.cache import DocumentCacheService
        from ...core.enhanced_storage import EnhancedDatabaseManager
        from ...core.expansion import ContextExpansionService
        from ..config import get_api_url

        # Initialize services
        db_manager = EnhancedDatabaseManager()
        await db_manager.initialize()

        cache_service = DocumentCacheService()
        try:
            await cache_service.initialize()
        except Exception as e:
            echo_warning(f"Redis cache not available: {e}")
            echo_info("Context expansion will use database fallback")

        # Create a wrapper for compatibility with ContextExpansionService
        class EnhancedDatabaseWrapper:
            def __init__(self, enhanced_db, context_id):
                self.enhanced_db = enhanced_db
                self.context_id = context_id
                self.pool = enhanced_db.pool

            async def get_document_content_by_id(self, document_id):
                doc = await self.enhanced_db.get_document_by_id(
                    self.context_id, document_id
                )
                return doc if doc else None

        db_wrapper = EnhancedDatabaseWrapper(db_manager, context_id)
        expansion_service = ContextExpansionService(db_wrapper, cache_service)

        # Expand each result
        expanded_results = []
        for result in results:
            try:
                # Map v2 API result format to expansion service format
                expansion_result = {
                    "id": result.get("id"),
                    "document_id": result.get("document_id"),
                    "content": result.get("content"),
                    "start_line": result.get("start_line"),
                    "end_line": result.get("end_line"),
                }

                expanded = await expansion_service.expand_search_result(
                    expansion_result, expand_lines, prefer_boundaries=True
                )

                # Update the original result with expanded content
                if expanded.get("content") != result.get("content"):
                    result["content"] = expanded["content"]
                    result["expansion_info"] = expanded.get("expansion_info", {})
                    result["content_type"] = "expanded_chunk"

                expanded_results.append(result)

            except Exception as e:
                logger.error(
                    f"Failed to expand result {result.get('id', 'unknown')}: {e}"
                )
                # Keep original result if expansion fails
                expanded_results.append(result)

        # Clean up connections
        await db_manager.close()
        await cache_service.close()

        return expanded_results

    except Exception as e:
        echo_error(f"Context expansion failed: {e}")
        echo_info("Returning original results without expansion")
        return results


# Alias for the query command to avoid naming conflicts
query_cmd = query
