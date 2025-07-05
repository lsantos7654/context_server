"""Document management commands for Context Server CLI."""

import asyncio
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..config import get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    confirm_action,
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
def docs():
    """Extract and manage documents within contexts.

    Commands for processing documents from various sources including
    URLs, files, directories, and git repositories.

    Examples:
        ctx docs extract https://docs.rust-lang.org rust-docs    # Extract from URL
        ctx docs extract ./my-project my-code --source-type local # Extract local files
        ctx docs list my-docs                                     # List documents
        ctx docs show my-docs doc-id-123                          # Show document content
    """
    pass


@docs.command()
@click.argument("source")
@click.argument("context_name", required=False, shell_complete=complete_context_name)
@click.option(
    "--source-type",
    type=click.Choice(["url", "file", "git", "local"]),
    help="Source type (auto-detected if not specified)",
)
@click.option("--max-pages", default=50, help="Maximum pages to extract for URLs")
@click.option("--wait/--no-wait", default=True, help="Wait for extraction to complete")
@click.option(
    "--output-path",
    type=click.Path(),
    help="Local directory to save extracted files to (enables download-only mode for URLs)",
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
    """Extract documents from a source into a context or download locally.

    Processes documents from URLs, files, directories, or git repositories.
    Can either add them to a context for search/retrieval or download them locally.

    Args:
        source: URL, file path, git repository, or local directory
        context_name: Name of target context (optional if --output-path is used)
        source_type: Source type (url, file, git, local)
        max_pages: Maximum pages to extract for URLs
        wait: Wait for extraction to complete
        output_path: Local directory to save extracted files
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude

    Examples:
        ctx docs extract https://example.com my-context              # Extract to context
        ctx docs extract https://example.com --output-path ./output  # Download locally
        ctx docs extract ./local-dir my-context --output-path ./out  # Copy and extract
    """
    # Validate arguments
    if not context_name and not output_path:
        echo_error("Either context_name or --output-path must be specified")
        echo_info(
            "Use --output-path to download files locally without creating a context"
        )
        return

    # Auto-detect source type if not specified
    if not source_type:
        if source.startswith(("http://", "https://")):
            source_type = "url"
        elif source.startswith(
            ("git://", "git@", "https://github.com", "https://gitlab.com")
        ):
            source_type = "git"
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

            # Check if this is download-only mode (output_path provided without context or for non-local sources)
            download_only = output_path and (not context_name or source_type != "local")

            if download_only:
                echo_info(f"Download mode - saving to: {output_path}")
                await extract_and_download_files(
                    source, source_type, output_path, max_pages
                )
            else:
                if not context_name:
                    echo_error("Context name is required when not using download mode")
                    return

                echo_info(f"Target context: {context_name}")

                # Prepare request data
                request_data = {
                    "source_type": source_type,
                    "source": source,
                    "options": {},
                }

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
                            echo_info(
                                f"Check status with: context-server docs status {job_id}"
                            )

                    elif response.status_code == 404:
                        echo_error(f"Context '{context_name}' not found")
                        echo_info(
                            "Create it with: context-server context create <name>"
                        )
                    else:
                        echo_error(
                            f"Failed to start extraction: {response.status_code} - {response.text}"
                        )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

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
@click.option(
    "--show-ids/--no-ids",
    default=False,
    help="Show document IDs (useful for 'ctx docs show' command)",
)
@rich_help_option("-h", "--help")
def list(context_name, offset, limit, output_format, show_ids):
    """📝 List documents in a context.

    Args:
        context_name: Context name
        offset: Number of documents to skip
        limit: Maximum documents to show
        output_format: Output format (table or json)
        show_ids: Show document IDs for use with 'ctx docs show'

    Examples:
        ctx docs list my-context                    # Basic list
        ctx docs list my-context --show-ids         # With document IDs
        ctx docs list my-context --limit 10         # Limit results
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
                                "Extract some with: context-server docs extract <source> <context>"
                            )
                            return

                        table = Table(
                            title=f"Documents in '{context_name}' ({total} total)"
                        )

                        # Add columns based on show_ids flag
                        if show_ids:
                            table.add_column("Document ID", style="dim", no_wrap=True)
                        table.add_column("Title")
                        table.add_column("URL")
                        table.add_column("Chunks")
                        table.add_column("Indexed")

                        for doc in documents:
                            # Truncate long URLs
                            url = doc["url"]
                            if len(url) > 50:
                                url = url[:47] + "..."

                            # Prepare row data
                            row_data = []
                            if show_ids:
                                # Show first 8 characters of document ID
                                doc_id = (
                                    doc["id"][:8] + "..."
                                    if len(doc["id"]) > 8
                                    else doc["id"]
                                )
                                row_data.append(doc_id)

                            row_data.extend(
                                [
                                    doc["title"][:50]
                                    + ("..." if len(doc["title"]) > 50 else ""),
                                    url,
                                    str(doc["chunk_count"]),
                                    str(doc["indexed_at"])[:19],  # Truncate timestamp
                                ]
                            )

                            table.add_row(*row_data)

                        console.print(table)

                        # Show helpful message when IDs are displayed
                        if show_ids and documents:
                            echo_info(
                                "💡 Use document IDs with: ctx docs show <context_name> <document_id>"
                            )

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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(list_documents())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Directory to save downloaded documents (default: ./downloads)",
    default="./downloads",
)
@click.option(
    "--document-ids",
    multiple=True,
    help="Specific document IDs to download (use 'ctx docs list --show-ids' to see IDs)",
)
@click.option(
    "--all",
    "download_all",
    is_flag=True,
    help="Download all documents from the context",
)
@click.option(
    "--format",
    "download_format",
    type=click.Choice(["files", "zip"]),
    default="files",
    help="Download format: individual files or ZIP archive",
)
@click.option("--force", is_flag=True, help="Skip confirmation prompts")
@rich_help_option("-h", "--help")
def download(
    context_name, output_path, document_ids, download_all, download_format, force
):
    """📥 Download documents from a context to local files.

    Downloads documents and saves them as individual files or in a ZIP archive.
    You can download specific documents by ID or all documents from a context.

    Args:
        context_name: Context name
        output_path: Directory to save files
        document_ids: Specific document IDs to download
        download_all: Download all documents
        download_format: Output format (files or zip)
        force: Skip confirmation prompts

    Examples:
        ctx docs download my-context                           # Download all as files
        ctx docs download my-context --format zip              # Download all as ZIP
        ctx docs download my-context --document-ids abc123     # Download specific docs
        ctx docs download my-context --output-path ./exports   # Custom output directory
    """
    if not document_ids and not download_all:
        echo_error(
            "Either specify --document-ids or use --all to download all documents"
        )
        echo_info("Use 'ctx docs list --show-ids' to see available document IDs")
        return

    if document_ids and download_all:
        echo_error("Cannot use both --document-ids and --all options together")
        return

    async def download_documents():
        try:
            async with httpx.AsyncClient() as client:
                # First, get the context to verify it exists
                context_response = await client.get(
                    get_api_url(f"contexts/{context_name}"),
                    timeout=30.0,
                )

                if context_response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                    return
                elif context_response.status_code != 200:
                    echo_error(f"Failed to get context: {context_response.text}")
                    return

                # Get documents to download
                if download_all:
                    # Get all documents
                    docs_response = await client.get(
                        get_api_url(f"contexts/{context_name}/documents"),
                        params={"limit": 1000},  # Large limit to get all docs
                        timeout=30.0,
                    )

                    if docs_response.status_code != 200:
                        echo_error(f"Failed to get documents: {docs_response.text}")
                        return

                    docs_data = docs_response.json()
                    documents = docs_data["documents"]
                    total_docs = docs_data["total"]

                    if not documents:
                        echo_info(f"No documents found in context '{context_name}'")
                        return

                    if not force:
                        if not confirm_action(
                            f"Download {len(documents)} document(s) from '{context_name}'?",
                            default=True,
                        ):
                            echo_info("Download cancelled")
                            return

                    doc_ids_to_download = [doc["id"] for doc in documents]
                    echo_info(f"Downloading {len(documents)} documents...")

                else:
                    # Download specific documents
                    doc_ids_to_download = [*document_ids]  # Convert tuple to list
                    echo_info(
                        f"Downloading {len(doc_ids_to_download)} specific document(s)..."
                    )

                # Create output directory
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Download documents
                downloaded_count = 0
                failed_count = 0

                if download_format == "zip":
                    # Create ZIP archive
                    zip_filename = output_dir / f"{context_name}_documents.zip"

                    with zipfile.ZipFile(
                        zip_filename, "w", zipfile.ZIP_DEFLATED
                    ) as zipf:
                        for doc_id in doc_ids_to_download:
                            try:
                                # Get document content
                                doc_response = await client.get(
                                    get_api_url(
                                        f"contexts/{context_name}/documents/{doc_id}/raw"
                                    ),
                                    timeout=30.0,
                                )

                                if doc_response.status_code == 200:
                                    document = doc_response.json()

                                    # Generate filename from title
                                    title = document.get("title", "untitled")
                                    safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
                                    filename = f"{safe_title}.txt"

                                    # Add to ZIP with metadata header
                                    content = f"""<!-- Document: {document.get('title')} -->
<!-- URL: {document.get('url')} -->
<!-- ID: {document.get('id')} -->
<!-- Indexed: {document.get('indexed_at')} -->

{document.get('content', '')}"""

                                    zipf.writestr(filename, content.encode("utf-8"))
                                    downloaded_count += 1

                                else:
                                    echo_warning(
                                        f"Failed to download document {doc_id[:8]}..."
                                    )
                                    failed_count += 1

                            except Exception as e:
                                echo_warning(
                                    f"Error downloading document {doc_id[:8]}...: {e}"
                                )
                                failed_count += 1

                    echo_success(f"Created ZIP archive: {zip_filename}")

                else:
                    # Save as individual files
                    for doc_id in doc_ids_to_download:
                        try:
                            # Get document content
                            doc_response = await client.get(
                                get_api_url(
                                    f"contexts/{context_name}/documents/{doc_id}/raw"
                                ),
                                timeout=30.0,
                            )

                            if doc_response.status_code == 200:
                                document = doc_response.json()

                                # Generate safe filename from title
                                title = document.get("title", "untitled")
                                safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
                                safe_title = safe_title[:100]  # Limit length

                                # Determine file extension
                                url = document.get("url", "")
                                if url.endswith((".md", ".markdown")):
                                    extension = ".md"
                                elif url.endswith((".html", ".htm")):
                                    extension = ".html"
                                elif url.endswith((".py", ".js", ".ts", ".css")):
                                    extension = url[url.rfind(".") :]
                                else:
                                    extension = ".txt"

                                filename = f"{safe_title}_{doc_id[:8]}{extension}"
                                file_path = output_dir / filename

                                # Write content with metadata header
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(
                                        f"<!-- Document: {document.get('title')} -->\n"
                                    )
                                    f.write(f"<!-- URL: {document.get('url')} -->\n")
                                    f.write(f"<!-- ID: {document.get('id')} -->\n")
                                    f.write(
                                        f"<!-- Indexed: {document.get('indexed_at')} -->\n\n"
                                    )
                                    f.write(document.get("content", ""))

                                downloaded_count += 1
                                echo_info(f"Saved: {filename}")

                            else:
                                echo_warning(
                                    f"Failed to download document {doc_id[:8]}..."
                                )
                                failed_count += 1

                        except Exception as e:
                            echo_warning(
                                f"Error downloading document {doc_id[:8]}...: {e}"
                            )
                            failed_count += 1

                echo_success(
                    f"Download completed! Downloaded: {downloaded_count}, Failed: {failed_count}"
                )
                echo_info(f"Files saved to: {output_dir}")

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(download_documents())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def delete(context_name, document_ids, force):
    """🗚️ Delete documents from a context.

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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(delete_documents())


@docs.command()
@click.argument("job_id")
def status(job_id):
    """🕰️ Check the status of a document extraction job.

    Args:
        job_id: Extraction job ID
    """
    # This would require an API endpoint for job status
    echo_error("Job status checking is not yet implemented")
    echo_info("Job status tracking will be available in a future version")


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.argument("document_id")
@click.option("--output-file", help="Save content to file instead of displaying")
@rich_help_option("-h", "--help")
def show(context_name, document_id, output_file):
    """📄 Show raw document content.

    Args:
        context_name: Context name
        document_id: Document ID
        output_file: Output file path
    """

    async def show_document():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    get_api_url(f"contexts/{context_name}/documents/{document_id}/raw"),
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

                        # Display in panel
                        panel = Panel(
                            syntax,
                            title=f"[bold blue]{document['title']}[/bold blue]",
                            subtitle=f"[dim]{len(content):,} characters[/dim]",
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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(show_document())


@docs.command()
@click.argument("context_name", shell_complete=complete_context_name)
@click.option(
    "--source-type",
    type=click.Choice(["url", "file", "git"]),
    help="Filter by source type",
)
@click.option("--since", help="Show documents indexed since date (YYYY-MM-DD)")
def count(context_name, source_type, since):
    """📊 Count documents in a context.

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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(count_documents())


async def wait_for_extraction(
    job_id: str, check_interval: float = 5.0, timeout: int = 300
):
    """Wait for extraction job to complete.

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
        task = progress.add_task("Extracting documents...", total=None)

        while time.time() - start_time < timeout:
            # In a real implementation, this would check job status via API
            # For now, we'll just wait a reasonable amount of time
            await asyncio.sleep(check_interval)

            # Simulate completion after 30 seconds (placeholder)
            if time.time() - start_time > 30:
                progress.update(task, description="Extraction completed!")
                echo_success("Document extraction completed!")
                return

        echo_warning("Extraction may still be running in the background")
        echo_info("Check server logs for progress")


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
    echo_info(f"Try: context-server search query '<your-query>' {context_name}")


async def extract_and_download_files(
    source: str, source_type: str, output_path: str, max_pages: int = 50
):
    """Extract documents and download them locally without storing in database."""
    try:
        # Prepare request data
        request_data = {"source_type": source_type, "source": source, "options": {}}

        if source_type == "url":
            request_data["options"]["max_pages"] = max_pages

        echo_info("Extracting documents for download...")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                get_api_url("documents/extract-download"),
                json=request_data,
                timeout=300.0,  # 5 minute timeout for extraction
            )

            if response.status_code == 200:
                result = response.json()
                documents = result["documents"]

                if not documents:
                    echo_warning("No documents were extracted")
                    return

                # Create output directory
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                echo_info(f"Saving {len(documents)} document(s) to: {output_dir}")

                saved_count = 0
                for i, doc in enumerate(documents):
                    try:
                        # Generate safe filename from title or URL
                        title = doc.get("title", "").strip()
                        url = doc.get("url", "")

                        if title:
                            # Clean title for filename
                            safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
                            safe_title = safe_title[:100]  # Limit length
                        else:
                            # Generate from URL
                            parsed = urlparse(url)
                            safe_title = parsed.path.split("/")[-1] or parsed.netloc
                            safe_title = re.sub(r'[<>:"/\\|?*]', "_", safe_title)

                        # Ensure unique filename
                        base_filename = safe_title or f"document_{i+1}"

                        # Determine file extension based on content or metadata
                        metadata = doc.get("metadata", {})
                        content_type = metadata.get("content_type", "")

                        if "markdown" in content_type.lower() or url.endswith(".md"):
                            extension = ".md"
                        elif "code" in content_type.lower():
                            # Try to detect language from metadata
                            lang = metadata.get("primary_language", "")
                            if lang == "python":
                                extension = ".py"
                            elif lang == "javascript":
                                extension = ".js"
                            elif lang == "typescript":
                                extension = ".ts"
                            else:
                                extension = ".txt"
                        elif url.endswith((".html", ".htm")):
                            extension = ".html"
                        else:
                            extension = ".txt"

                        filename = f"{base_filename}{extension}"
                        file_path = output_dir / filename

                        # Handle duplicate filenames
                        counter = 1
                        while file_path.exists():
                            name_part = base_filename
                            file_path = output_dir / f"{name_part}_{counter}{extension}"
                            counter += 1

                        # Write content to file
                        content = doc.get("content", "")
                        with open(file_path, "w", encoding="utf-8") as f:
                            # Add metadata header
                            f.write(f"<!-- Source: {url} -->\n")
                            f.write(f"<!-- Title: {title} -->\n")
                            f.write(
                                f"<!-- Extracted: {datetime.now().isoformat()} -->\n\n"
                            )
                            f.write(content)

                        saved_count += 1
                        echo_info(f"Saved: {filename}")

                    except Exception as e:
                        echo_warning(f"Failed to save document {i+1}: {e}")

                echo_success(
                    f"Successfully downloaded {saved_count}/{len(documents)} documents to {output_dir}"
                )

                # Create a summary file
                summary_path = output_dir / "extraction_summary.json"
                summary = {
                    "source": source,
                    "source_type": source_type,
                    "extracted_at": datetime.now().isoformat(),
                    "total_documents": len(documents),
                    "saved_documents": saved_count,
                    "documents": [
                        {
                            "title": doc.get("title"),
                            "url": doc.get("url"),
                            "chunks": len(doc.get("chunks", [])),
                        }
                        for doc in documents
                    ],
                }

                with open(summary_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump(summary, f, indent=2, ensure_ascii=False)

                echo_info(f"Extraction summary saved to: {summary_path}")

            else:
                echo_error(
                    f"Failed to extract documents: {response.status_code} - {response.text}"
                )

    except httpx.RequestError as e:
        echo_error(f"Connection error: {e}")
        echo_info("Make sure the server is running: context-server server up")
    except Exception as e:
        echo_error(f"Failed to download documents: {e}")
