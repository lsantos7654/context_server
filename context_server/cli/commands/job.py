"""Job management commands for Context Server CLI."""

import asyncio
from datetime import datetime

import click
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from context_server.cli.config import get_api_url
from context_server.cli.help_formatter import rich_help_option
from context_server.cli.utils import echo_error, echo_info, echo_success, echo_warning

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def job() -> None:
    """Manage extraction and processing jobs.

    Monitor ongoing extractions, check job status, and clean up completed jobs.

    Examples:
        ctx job list                    # Show active jobs
        ctx job status <job-id>         # Check specific job status
        ctx job cancel <job-id>         # Cancel running job
        ctx job cleanup                 # Clean up old completed jobs
    """
    pass


@job.command()
@click.option("--context", help="Filter jobs by context name")
@rich_help_option("-h", "--help")
def list(context) -> None:
    """List all active jobs."""

    async def list_jobs() -> None:
        try:
            params = {}
            if context:
                # First get context ID
                async with httpx.AsyncClient() as client:
                    response = await client.get(get_api_url(f"contexts/{context}"))
                    if response.status_code == 200:
                        context_data = response.json()
                        params["context_id"] = context_data["id"]
                    elif response.status_code == 404:
                        echo_error(f"Context '{context}' not found")
                        return
                    else:
                        echo_error(f"Failed to get context: {response.status_code}")
                        return

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    get_api_url("jobs/active"), params=params, timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    active_jobs = data.get("active_jobs", [])

                    if not active_jobs:
                        echo_info("No active jobs found")
                        return

                    # Create a table to display jobs
                    table = Table(
                        title="Active Jobs",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Job ID", style="cyan", no_wrap=True)
                    table.add_column("Type", style="green")
                    table.add_column("Status", style="yellow")
                    table.add_column("Progress", style="blue")
                    table.add_column("Started", style="dim")
                    table.add_column("Phase", style="magenta")

                    for job in active_jobs:
                        progress_pct = f"{job['progress'] * 100:.1f}%"
                        started = datetime.fromisoformat(
                            job["started_at"].replace("Z", "+00:00")
                        )
                        started_str = started.strftime("%H:%M:%S")

                        metadata = job.get("metadata", {})
                        phase = metadata.get("phase", "unknown")

                        table.add_row(
                            job["id"],
                            job["type"],
                            job["status"],
                            progress_pct,
                            started_str,
                            phase,
                        )

                    console.print(table)
                    echo_info(f"Total: {len(active_jobs)} active jobs")

                else:
                    echo_error(
                        f"Failed to list jobs: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(list_jobs())


@job.command()
@click.argument("job_id")
@click.option("--watch", "-w", is_flag=True, help="Watch job progress continuously")
@rich_help_option("-h", "--help")
def status(job_id, watch) -> None:
    """Get detailed status of a specific job."""

    async def get_job_status() -> None:
        try:
            if watch:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Monitoring job...", total=None)

                    while True:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(
                                get_api_url(f"jobs/{job_id}/status"), timeout=10.0
                            )

                            if response.status_code == 200:
                                job_data = response.json()
                                status = job_data["status"]
                                progress_pct = job_data["progress"]

                                # Update progress description
                                metadata = job_data.get("metadata", {})
                                phase = metadata.get("phase", "processing")

                                if phase == "extracting":
                                    url = metadata.get("url", "content")
                                    description = f"Extracting content from {url}"
                                elif phase == "crawling":
                                    description = (
                                        f"Crawling: {metadata.get('url', '...')}"
                                    )
                                elif phase == "content_extracted":
                                    pages = metadata.get("pages_found", 0)
                                    description = (
                                        f"Found {pages} pages, processing content..."
                                    )
                                elif phase == "code_extraction":
                                    status = metadata.get(
                                        "status", "analyzing code blocks"
                                    )
                                    content_size = metadata.get("content_size", 0)
                                    description = f"Code extraction: {status} ({content_size:,} chars)"
                                elif phase == "code_embedding":
                                    snippets_found = metadata.get("snippets_found", 0)
                                    model = metadata.get("model", "voyage-code-3")
                                    description = f"Found {snippets_found} code snippets, generating embeddings ({model})"
                                elif phase == "text_chunking":
                                    content_size = metadata.get("content_size", 0)
                                    code_snippets = metadata.get(
                                        "code_snippets_processed", 0
                                    )
                                    description = f"Creating text chunks ({content_size:,} chars, {code_snippets} code snippets)"
                                elif phase == "text_embedding":
                                    chunks_created = metadata.get("chunks_created", 0)
                                    embedding_model = metadata.get(
                                        "embedding_model", "text-embedding-3-large"
                                    )
                                    summary_model = metadata.get(
                                        "summary_model", "gpt-4o-mini"
                                    )
                                    description = f"Processing {chunks_created} chunks (embeddings: {embedding_model}, summaries: {summary_model})"
                                elif phase == "chunking_and_embedding":
                                    processed = metadata.get("processed_pages", 0)
                                    total = metadata.get("total_pages", 1)
                                    description = f"Chunking and embedding ({processed}/{total} pages)"
                                elif phase == "storing_documents":
                                    stored = metadata.get("stored_docs", 0)
                                    total = metadata.get("total_docs", 1)
                                    description = (
                                        f"Storing documents ({stored}/{total})"
                                    )
                                else:
                                    description = (
                                        f"Processing... ({int(progress_pct * 100)}%)"
                                    )

                                progress.update(task, description=description)

                                if status in ["completed", "failed"]:
                                    break

                            elif response.status_code == 404:
                                echo_error(f"Job {job_id} not found")
                                return
                            else:
                                echo_warning(
                                    f"Failed to get job status: {response.status_code}"
                                )

                            await asyncio.sleep(2)

                    # Final status
                    if status == "completed":
                        echo_success("Job completed successfully!")
                    elif status == "failed":
                        error_msg = job_data.get("error_message", "Unknown error")
                        echo_error(f"Job failed: {error_msg}")
            else:
                # Single status check
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        get_api_url(f"jobs/{job_id}/status"), timeout=10.0
                    )

                    if response.status_code == 200:
                        job_data = response.json()

                        # Create status table
                        table = Table(title=f"Job Status: {job_id}", show_header=False)
                        table.add_column("Property", style="cyan", no_wrap=True)
                        table.add_column("Value", style="white")

                        table.add_row("ID", job_data["id"])
                        table.add_row("Type", job_data["type"])
                        table.add_row("Status", job_data["status"])
                        table.add_row("Progress", f"{job_data['progress'] * 100:.1f}%")

                        started = datetime.fromisoformat(
                            job_data["started_at"].replace("Z", "+00:00")
                        )
                        table.add_row("Started", started.strftime("%Y-%m-%d %H:%M:%S"))

                        if job_data.get("completed_at"):
                            completed = datetime.fromisoformat(
                                job_data["completed_at"].replace("Z", "+00:00")
                            )
                            table.add_row(
                                "Completed", completed.strftime("%Y-%m-%d %H:%M:%S")
                            )

                        if job_data.get("error_message"):
                            table.add_row("Error", job_data["error_message"])

                        # Add metadata details
                        metadata = job_data.get("metadata", {})
                        if metadata:
                            table.add_row("", "")  # Separator
                            table.add_row("Phase", metadata.get("phase", "unknown"))

                            if "pages_found" in metadata:
                                table.add_row(
                                    "Pages Found", str(metadata["pages_found"])
                                )
                            if "processed_pages" in metadata:
                                table.add_row(
                                    "Processed Pages", str(metadata["processed_pages"])
                                )
                            if "total_pages" in metadata:
                                table.add_row(
                                    "Total Pages", str(metadata["total_pages"])
                                )

                        console.print(table)

                    elif response.status_code == 404:
                        echo_error(f"Job {job_id} not found")
                    else:
                        echo_error(
                            f"Failed to get job status: {response.status_code} - {response.text}"
                        )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_job_status())


@job.command()
@click.argument("job_id")
@click.option(
    "--force", "-f", is_flag=True, help="Force cancellation without confirmation"
)
@rich_help_option("-h", "--help")
def cancel(job_id, force) -> None:
    """Cancel a running job."""

    async def cancel_job() -> None:
        try:
            if not force:
                if not click.confirm(f"Are you sure you want to cancel job {job_id}?"):
                    echo_info("Cancellation aborted")
                    return

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    get_api_url(f"jobs/{job_id}"), timeout=10.0
                )

                if response.status_code == 200:
                    echo_success(f"Job {job_id} cancelled successfully")
                elif response.status_code == 404:
                    echo_error(f"Job {job_id} not found")
                elif response.status_code == 400:
                    result = response.json()
                    echo_error(
                        f"Cannot cancel job: {result.get('detail', 'Unknown reason')}"
                    )
                else:
                    echo_error(
                        f"Failed to cancel job: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(cancel_job())


@job.command()
@click.option(
    "--days", default=7, help="Remove jobs older than this many days (default: 7)"
)
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@rich_help_option("-h", "--help")
def cleanup(days, force) -> None:
    """Clean up old completed and failed jobs."""

    async def cleanup_jobs() -> None:
        try:
            if not force:
                if not click.confirm(
                    f"Remove all completed/failed jobs older than {days} days?"
                ):
                    echo_info("Cleanup aborted")
                    return

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_api_url("admin/jobs/cleanup"), json={"days": days}, timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    deleted_count = result.get("deleted_count", 0)
                    echo_success(f"Cleaned up {deleted_count} old jobs")
                else:
                    echo_error(
                        f"Failed to cleanup jobs: {response.status_code} - {response.text}"
                    )

        except httpx.RequestError as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(cleanup_jobs())
