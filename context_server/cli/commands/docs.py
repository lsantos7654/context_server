"""Document management commands for Context Server CLI."""

import asyncio
import time
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..config import get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    APIClient,
    complete_context_name,
    complete_job_id,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    get_context_names_sync,
)

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def docs():
    """Extract and manage documents within contexts.

    Commands for processing documents from various sources including
    URLs, files, and local directories.

    Examples:
        ctx docs extract https://docs.rust-lang.org rust-docs    # Extract from URL
        ctx docs extract ./my-project my-code --source-type local # Extract local files
        ctx docs list my-docs                                     # List documents
        ctx docs show my-docs doc-id-123                          # Show document content
    """
    pass


@docs.command()
@click.argument("source")
@click.argument("context_name", shell_complete=complete_context_name)
@click.option(
    "--source-type",
    type=click.Choice(["url", "file", "local"]),
    help="Source type (auto-detected if not specified)",
)
@click.option("--max-pages", default=50, help="Maximum pages to extract for URLs")
@click.option("--wait/--no-wait", default=True, help="Wait for extraction to complete")
@click.option(
    "--output-path",
    type=click.Path(),
    help="Local directory to extract files to (for local extraction)",
)
@click.option(
    "--include-patterns",
    multiple=True,
    help="File patterns to include (e.g., '*.md', '*.py')",
)
@click.option(
    "--exclude-patterns",
    multiple=True,
    help="File patterns to exclude (e.g., '*.pyc', '__pycache__')",
)
@rich_help_option("-h", "--help")
def extract(
    source,
    context_name,
    source_type,
    max_pages,
    wait,
    output_path,
    include_patterns,
    exclude_patterns,
):
    """Extract documents from a source into a context.

    Processes documents from URLs, files, or local directories
    and adds them to the specified context for search and retrieval.

    Args:
        source: URL, file path, or local directory
        context_name: Name of target context
        source_type: Source type (url, file, local)
        max_pages: Maximum pages to extract for URLs
        wait: Wait for extraction to complete
        output_path: Local directory to save extracted files (for local mode)
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
    """
    # Auto-detect source type if not specified
    if not source_type:
        if source.startswith(("http://", "https://")):
            source_type = "url"
        elif Path(source).exists():
            if Path(source).is_dir():
                source_type = "local"
            else:
                source_type = "file"
        else:
            echo_error(
                "Could not auto-detect source type. Please specify --source-type"
            )
            return

    # Handle local extraction
    if source_type == "local":
        asyncio.run(
            handle_local_extraction(
                source, context_name, output_path, include_patterns, exclude_patterns
            )
        )
        return

    async def extract_document():
        try:
            echo_info(f"Extracting {source_type}: {source}")
            echo_info(f"Target context: {context_name}")

            # Prepare request data
            request_data = {"source_type": source_type, "source": source, "options": {}}

            if source_type == "url":
                request_data["options"]["max_pages"] = max_pages

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_api_url(f"contexts/{context_name}/documents"),
                    json=request_data,
                    timeout=300.0,  # 5 minute timeout for extraction
                )

                if response.status_code == 202:
                    result = response.json()
                    job_id = result["job_id"]
                    echo_success(f"Extraction started! Job ID: {job_id}")

                    if wait:
                        echo_info("Waiting for extraction to complete...")
                        await wait_for_extraction(job_id)
                    else:
                        echo_info("Extraction running in background")
                        echo_info(f"Check status with: ctx job status {job_id}")

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                    echo_info("Create it with: ctx context create <name>")
                else:
                    echo_error(
                        f"Failed to start extraction: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(extract_document())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option("--offset", default=0, help="Number of documents to skip")
@click.option("--limit", default=50, help="Maximum number of documents to show")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@rich_help_option("-h", "--help")
def list(context_name, offset, limit, output_format):
    """List documents in a context.

    Args:
        context_name: Context name
        offset: Number of documents to skip
        limit: Maximum documents to show
        output_format: Output format (table or json)
    """

    async def list_documents():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    get_api_url(f"contexts/{context_name}/documents"),
                    params={"offset": offset, "limit": limit},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    documents = result["documents"]
                    total = result["total"]

                    if output_format == "json":
                        console.print(result)
                    else:
                        if not documents:
                            echo_info(f"No documents found in context '{context_name}'")
                            echo_info(
                                "Extract some with: ctx docs extract <source> <context>"
                            )
                            return

                        table = Table(
                            title=f"Documents in '{context_name}' ({total} total)"
                        )
                        table.add_column("Title")
                        table.add_column("URL")
                        table.add_column("Chunks")
                        table.add_column("Indexed")

                        for doc in documents:
                            # Truncate long URLs
                            url = doc["url"]
                            if len(url) > 50:
                                url = url[:47] + "..."

                            table.add_row(
                                doc["title"][:50]
                                + ("..." if len(doc["title"]) > 50 else ""),
                                url,
                                str(doc["chunks"]),
                                str(doc["indexed_at"])[:19],  # Truncate timestamp
                            )

                        console.print(table)

                        if total > offset + limit:
                            echo_info(
                                f"Showing {offset + 1}-{min(offset + limit, total)} of {total} documents"
                            )
                            echo_info(f"Use --offset {offset + limit} to see more")

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                else:
                    echo_error(
                        f"Failed to list documents: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(list_documents())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def delete(context_name, document_ids, force):
    """Delete documents from a context.

    Args:
        context_name: Context name
        document_ids: Document IDs to delete
        force: Skip confirmation prompt
    """
    if not force and not confirm_action(
        f"This will permanently delete {len(document_ids)} document(s). Continue?",
        default=False,
    ):
        echo_info("Document deletion cancelled")
        return

    async def delete_documents():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    get_api_url(f"contexts/{context_name}/documents"),
                    json={"document_ids": list(document_ids)},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    deleted_count = result["deleted_count"]
                    echo_success(f"Deleted {deleted_count} document(s) successfully!")
                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                else:
                    echo_error(
                        f"Failed to delete documents: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(delete_documents())




@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_id")
@click.option("--output-file", help="Save content to file instead of displaying")
@click.option("--page", default=1, help="Page number to display (default: 1)")
@click.option("--page-size", default=25000, help="Characters per page (default: 25000)")
@rich_help_option("-h", "--help")
def show(context_name, document_id, output_file, page, page_size):
    """Show raw document content with pagination support.

    Large documents are automatically paginated for better display.
    Use --page and --page-size to navigate through content.

    Args:
        context_name: Context name
        document_id: Document ID
        output_file: Output file path (saves full content, ignores pagination)
        page: Page number to display (1-based)
        page_size: Characters per page (default optimized for Claude's limit)
    """

    async def show_document():
        try:
            async with httpx.AsyncClient() as client:
                # Add pagination parameters
                params = {}
                if not output_file:  # Only paginate for display, not for file output
                    params["page_number"] = page
                    params["page_size"] = page_size
                
                response = await client.get(
                    get_api_url(f"contexts/{context_name}/documents/{document_id}/raw"),
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    document = response.json()

                    if output_file:
                        # Save to file
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(document["content"])
                        echo_success(f"Document content saved to {output_file}")
                    else:
                        # Display in terminal
                        from rich.panel import Panel
                        from rich.syntax import Syntax

                        # Create header with document info
                        header = f"Document: {document['title']}"
                        if document.get("url"):
                            header += f"\nURL: {document['url']}"
                        header += f"\nID: {document['id']}"
                        header += f"\nSource: {document.get('source_type', 'unknown')}"
                        if document.get("created_at"):
                            header += f"\nCreated: {document['created_at'][:19]}"
                        
                        # Add pagination info if available
                        pagination = document.get("pagination", {})
                        if pagination:
                            current_page = pagination.get("page_number", 1)
                            total_pages = pagination.get("total_pages", 1)
                            current_length = pagination.get("current_page_length", 0)
                            full_length = document.get("full_content_length", current_length)
                            
                            header += f"\nPage: {current_page}/{total_pages}"
                            header += f"\nShowing: {current_length:,} chars of {full_length:,} total"
                            
                            if pagination.get("has_next_page"):
                                header += f"\nNext page: --page {current_page + 1}"
                            if pagination.get("has_previous_page"):
                                header += f"\nPrevious page: --page {current_page - 1}"

                        # Try to detect if content is markdown/code for syntax highlighting
                        content = document["content"]
                        url = document.get("url", "")

                        if url.endswith(".md") or "markdown" in document.get(
                            "metadata", {}
                        ):
                            syntax = Syntax(
                                content, "markdown", theme="monokai", line_numbers=True
                            )
                        elif url.endswith(".py"):
                            syntax = Syntax(
                                content, "python", theme="monokai", line_numbers=True
                            )
                        elif url.endswith(".js"):
                            syntax = Syntax(
                                content,
                                "javascript",
                                theme="monokai",
                                line_numbers=True,
                            )
                        elif url.endswith(".html"):
                            syntax = Syntax(
                                content, "html", theme="monokai", line_numbers=True
                            )
                        else:
                            syntax = content

                        # Create subtitle with pagination info
                        subtitle = f"[dim]{len(content):,} characters"
                        if pagination:
                            total_pages = pagination.get("total_pages", 1)
                            current_page = pagination.get("page_number", 1)
                            subtitle += f" (Page {current_page}/{total_pages})"
                        subtitle += "[/dim]"
                        
                        # Display in panel
                        panel = Panel(
                            syntax,
                            title=f"[bold blue]{document['title']}[/bold blue]",
                            subtitle=subtitle,
                            border_style="blue",
                            padding=(1, 2),
                        )

                        # Print header info first
                        console.print(f"[bold cyan]{header}[/bold cyan]")
                        console.print()
                        console.print(panel)

                elif response.status_code == 404:
                    echo_error(
                        f"Document '{document_id}' not found in context '{context_name}'"
                    )
                else:
                    echo_error(
                        f"Failed to get document: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(show_document())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option(
    "--source-type",
    type=click.Choice(["url", "file"]),
    help="Filter by source type",
)
@click.option("--since", help="Show documents indexed since date (YYYY-MM-DD)")
def count(context_name, source_type, since):
    """Count documents in a context.

    Args:
        context_name: Context name
        source_type: Filter by source type
        since: Show documents since date
    """

    async def count_documents():
        try:
            async with httpx.AsyncClient() as client:
                params = {}
                if source_type:
                    params["source_type"] = source_type
                if since:
                    params["since"] = since

                response = await client.get(
                    get_api_url(f"contexts/{context_name}/documents"),
                    params={**params, "limit": 1},  # We just need the total
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    total = result["total"]

                    filters = []
                    if source_type:
                        filters.append(f"source_type={source_type}")
                    if since:
                        filters.append(f"since={since}")

                    filter_text = f" ({', '.join(filters)})" if filters else ""
                    echo_info(
                        f"Context '{context_name}' contains {total} document(s){filter_text}"
                    )

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                else:
                    echo_error(
                        f"Failed to count documents: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(count_documents())


async def wait_for_extraction(
    job_id: str, check_interval: float = 2.0, timeout: int = 600
):
    """Wait for extraction job to complete by polling job status API.

    Args:
        job_id: Job ID to monitor
        check_interval: Time between checks in seconds
        timeout: Maximum time to wait in seconds
    """
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting extraction...", total=None)
        last_status = None

        while time.time() - start_time < timeout:
            try:
                # Poll job status from API
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        get_api_url(f"jobs/{job_id}/status"),
                        timeout=10.0,
                    )

                    if response.status_code == 200:
                        job_data = response.json()
                        status = job_data["status"]
                        progress_pct = job_data["progress"]

                        # Update progress display based on job metadata
                        if "metadata" in job_data and job_data["metadata"]:
                            metadata = job_data["metadata"]
                            phase = metadata.get("phase", "processing")

                            if phase == "crawling":
                                description = f"Crawling: {metadata.get('url', '...')}"
                            elif phase == "content_extracted":
                                pages = metadata.get("pages_found", 0)
                                description = (
                                    f"Found {pages} pages, processing content..."
                                )
                            elif phase == "chunking_and_embedding":
                                processed = metadata.get("processed_pages", 0)
                                total = metadata.get("total_pages", 1)
                                description = f"Chunking and embedding ({processed}/{total} pages)"
                            elif phase == "storing_documents":
                                stored = metadata.get("stored_docs", 0)
                                total = metadata.get("total_docs", 1)
                                description = f"Storing documents ({stored}/{total})"
                            else:
                                description = (
                                    f"Processing... ({int(progress_pct * 100)}%)"
                                )
                        else:
                            description = f"Processing... ({int(progress_pct * 100)}%)"

                        progress.update(task, description=description)

                        if status == "completed":
                            echo_success("Document extraction completed!")
                            result_data = job_data.get("result_data", {})
                            if result_data:
                                docs_count = result_data.get("documents_processed", 0)
                                chunks_count = result_data.get("total_chunks", 0)
                                echo_info(
                                    f"Processed {docs_count} documents with {chunks_count} chunks"
                                )
                            return
                        elif status == "failed":
                            error_msg = job_data.get("error_message", "Unknown error")
                            echo_error(f"Document extraction failed: {error_msg}")
                            return

                        last_status = status

                    elif response.status_code == 404:
                        echo_error(f"Job {job_id} not found")
                        return
                    else:
                        echo_warning(
                            f"Failed to get job status: {response.status_code}"
                        )

            except httpx.RequestError as e:
                echo_warning(f"Connection error while checking job status: {e}")

            await asyncio.sleep(check_interval)

        echo_warning("Extraction timed out, but may still be running in the background")
        echo_info(f"Check status with: ctx job status {job_id}")


async def handle_local_extraction(
    source_dir: str,
    context_name: str,
    output_path: str = None,
    include_patterns: tuple = None,
    exclude_patterns: tuple = None,
):
    """Handle local directory extraction.

    Args:
        source_dir: Source directory to extract from
        context_name: Target context name
        output_path: Optional output directory to save files
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
    """
    import fnmatch
    import shutil

    source_path = Path(source_dir)
    if not source_path.exists():
        echo_error(f"Source directory does not exist: {source_dir}")
        return

    if not source_path.is_dir():
        echo_error(f"Source is not a directory: {source_dir}")
        return

    # Set default patterns if none provided
    if not include_patterns:
        include_patterns = [
            "*.md",
            "*.txt",
            "*.rst",
            "*.py",
            "*.js",
            "*.ts",
            "*.html",
            "*.json",
        ]

    if not exclude_patterns:
        exclude_patterns = [
            "*.pyc",
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            "*.log",
            ".DS_Store",
            "*.tmp",
            "dist",
            "build",
        ]

    echo_info(f"Scanning directory: {source_path}")
    echo_info(f"Include patterns: {', '.join(include_patterns)}")
    echo_info(f"Exclude patterns: {', '.join(exclude_patterns)}")

    # Collect files to process
    files_to_process = []

    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            # Check exclude patterns first
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(
                    str(file_path.relative_to(source_path)), pattern
                ):
                    excluded = True
                    break

            if excluded:
                continue

            # Check include patterns
            included = False
            for pattern in include_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    included = True
                    break

            if included:
                files_to_process.append(file_path)

    if not files_to_process:
        echo_warning("No files found matching the criteria")
        return

    echo_info(f"Found {len(files_to_process)} files to process")

    # Copy files to output directory if specified
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        echo_info(f"Copying files to: {output_dir}")

        for file_path in files_to_process:
            relative_path = file_path.relative_to(source_path)
            dest_path = output_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                echo_warning(f"Failed to copy {file_path}: {e}")

        echo_success(f"Files copied to {output_dir}")

    # Process files through the API
    echo_info("Processing files through Context Server...")

    total_processed = 0
    total_errors = 0

    for file_path in files_to_process:
        try:
            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try with different encoding for binary-like files
                try:
                    content = file_path.read_text(encoding="latin-1")
                except:
                    echo_warning(f"Skipping binary file: {file_path}")
                    continue

            # Send to API for processing
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_api_url(f"contexts/{context_name}/documents"),
                    json={
                        "source_type": "file",
                        "source": str(file_path),
                        "content": content,
                        "options": {
                            "filename": file_path.name,
                            "relative_path": str(file_path.relative_to(source_path)),
                        },
                    },
                    timeout=300.0,
                )

                if response.status_code == 202:
                    total_processed += 1
                    if total_processed % 10 == 0:
                        echo_info(
                            f"Processed {total_processed}/{len(files_to_process)} files..."
                        )
                else:
                    total_errors += 1
                    echo_warning(
                        f"Failed to process {file_path}: {response.status_code}"
                    )

        except Exception as e:
            total_errors += 1
            echo_warning(f"Error processing {file_path}: {e}")

    echo_success(f"Local extraction completed!")
    echo_info(f"Processed: {total_processed} files")
    if total_errors > 0:
        echo_warning(f"Errors: {total_errors} files")

    echo_info(f"Files added to context '{context_name}'")
    echo_info(f"Try: ctx search query '<your-query>' {context_name}")




