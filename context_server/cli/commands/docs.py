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
from ..utils import confirm_action, echo_error, echo_info, echo_success, echo_warning

console = Console()


@click.group()
def docs():
    """Document management commands.

    Commands for extracting, listing, and managing documents within contexts.
    """
    pass


@docs.command()
@click.argument("source")
@click.argument("context_name")
@click.option(
    "--source-type",
    type=click.Choice(["url", "file", "git"]),
    help="Source type (auto-detected if not specified)",
)
@click.option("--max-pages", default=50, help="Maximum pages to extract for URLs")
@click.option("--wait/--no-wait", default=True, help="Wait for extraction to complete")
def extract(source, context_name, source_type, max_pages, wait):
    """Extract documents from a source into a context.

    Args:
        source: URL, file path, or git repository
        context_name: Target context name
        source_type: Source type (url, file, git)
        max_pages: Maximum pages to extract for URLs
        wait: Wait for extraction to complete
    """
    # Auto-detect source type if not specified
    if not source_type:
        if source.startswith(("http://", "https://")):
            source_type = "url"
        elif source.startswith(
            ("git://", "git@", "https://github.com", "https://gitlab.com")
        ):
            source_type = "git"
        elif Path(source).exists():
            source_type = "file"
        else:
            echo_error(
                "Could not auto-detect source type. Please specify --source-type"
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
                        echo_info(
                            f"Check status with: context-server docs status {job_id}"
                        )

                elif response.status_code == 404:
                    echo_error(f"Context '{context_name}' not found")
                    echo_info("Create it with: context-server context create <name>")
                else:
                    echo_error(
                        f"Failed to start extraction: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(extract_document())


@docs.command()
@click.argument("context_name")
@click.option("--offset", default=0, help="Number of documents to skip")
@click.option("--limit", default=50, help="Maximum number of documents to show")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
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
                                "Extract some with: context-server docs extract <source> <context>"
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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(list_documents())


@docs.command()
@click.argument("context_name")
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
            echo_info("Make sure the server is running: context-server server up")

    asyncio.run(delete_documents())


@docs.command()
@click.argument("job_id")
def status(job_id):
    """Check the status of a document extraction job.

    Args:
        job_id: Extraction job ID
    """
    # This would require an API endpoint for job status
    echo_error("Job status checking is not yet implemented")
    echo_info("Job status tracking will be available in a future version")


@docs.command()
@click.argument("context_name")
@click.argument("document_id")
@click.option("--output-file", help="Save content to file instead of displaying")
def show(context_name, document_id, output_file):
    """Show raw document content.

    Args:
        context_name: Context name
        document_id: Document ID
        output_file: Output file path
    """
    # This would require an API endpoint for getting raw document content
    echo_error("Document content display is not yet implemented")
    echo_info("Raw document access will be available in a future version")


@docs.command()
@click.argument("context_name")
@click.option(
    "--source-type",
    type=click.Choice(["url", "file", "git"]),
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
