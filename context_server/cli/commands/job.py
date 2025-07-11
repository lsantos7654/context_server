"""Job management commands for Context Server CLI."""

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import get_api_url
from ..help_formatter import rich_help_option
from ..utils import (
    APIClient,
    complete_job_id,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
)

console = Console()


@click.group()
@rich_help_option("-h", "--help")
def job():
    """Manage document processing jobs.

    Commands for monitoring, canceling, and managing long-running
    document extraction and processing jobs.

    Examples:
        ctx job status job-id-123    # Check job status
        ctx job list                 # List all jobs
        ctx job cancel job-id-123    # Cancel running job
        ctx job cleanup              # Clean up completed jobs
    """
    pass


@job.command()
@click.argument("job_id", shell_complete=complete_job_id)
@rich_help_option("-h", "--help")
def status(job_id):
    """Check the status of a document processing job.

    Shows detailed information about a job including progress,
    current phase, and any error messages.

    Args:
        job_id: Job ID to check
    """

    async def get_job_status():
        try:
            client = APIClient()
            success, response = await client.get(f"jobs/{job_id}/status")

            if success:
                job_data = response

                # Create a status table
                table = Table(title=f"Job Status: {job_id}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Status", job_data["status"])
                table.add_row("Progress", f"{int(job_data['progress'] * 100)}%")
                table.add_row("Type", job_data["type"])
                table.add_row(
                    "Started",
                    job_data["started_at"][:19]
                    if job_data["started_at"]
                    else "N/A",
                )
                table.add_row(
                    "Updated",
                    job_data["updated_at"][:19]
                    if job_data["updated_at"]
                    else "N/A",
                )

                if job_data["completed_at"]:
                    table.add_row("Completed", job_data["completed_at"][:19])

                if job_data["error_message"]:
                    table.add_row("Error", job_data["error_message"])

                console.print(table)

                # Show metadata if available
                if job_data["metadata"]:
                    metadata = job_data["metadata"]
                    if "phase" in metadata:
                        echo_info(f"Current phase: {metadata['phase']}")
                    if "source" in metadata:
                        echo_info(f"Source: {metadata['source']}")

                # Show result data if completed
                if job_data["status"] == "completed" and job_data["result_data"]:
                    result_data = job_data["result_data"]
                    echo_success("Job Results:")
                    echo_info(
                        f"  Documents processed: {result_data.get('documents_processed', 0)}"
                    )
                    echo_info(
                        f"  Total chunks: {result_data.get('total_chunks', 0)}"
                    )
                    echo_info(
                        f"  Code snippets: {result_data.get('total_code_snippets', 0)}"
                    )

            else:
                if "404" in str(response):
                    echo_error(f"Job '{job_id}' not found")
                else:
                    echo_error(f"Failed to get job status: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(get_job_status())


@job.command()
@click.option("--status", help="Filter by job status (pending, running, completed, failed)")
@click.option("--type", help="Filter by job type (document_extraction)")
@click.option("--limit", default=20, help="Maximum number of jobs to show")
@rich_help_option("-h", "--help")
def list(status, type, limit):
    """List recent jobs with their status.

    Shows a table of recent jobs with their current status,
    progress, and basic information.

    Args:
        status: Filter by job status
        type: Filter by job type
        limit: Maximum number of jobs to show
    """

    async def list_jobs():
        try:
            client = APIClient()
            params = {"limit": limit}
            if status:
                params["status"] = status
            if type:
                params["type"] = type

            success, response = await client.get("jobs", params=params)

            if success:
                jobs = response.get("jobs", [])

                if not jobs:
                    echo_info("No jobs found")
                    return

                # Create jobs table
                table = Table(title=f"Recent Jobs ({len(jobs)} found)")
                table.add_column("Job ID", style="cyan", width=30)
                table.add_column("Status", style="white", width=12)
                table.add_column("Progress", style="yellow", width=10)
                table.add_column("Type", style="green", width=15)
                table.add_column("Started", style="blue", width=20)

                for job in jobs:
                    # Truncate long job IDs
                    job_id = job["id"]
                    if len(job_id) > 25:
                        job_id = job_id[:22] + "..."

                    status_display = job["status"]
                    progress = f"{int(job['progress'] * 100)}%"
                    job_type = job.get("type", "unknown")
                    started_at = job.get("started_at", "")
                    if started_at:
                        started_at = started_at[:19]

                    table.add_row(job_id, status_display, progress, job_type, started_at)

                console.print(table)

            else:
                echo_error(f"Failed to list jobs: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(list_jobs())


@job.command()
@click.argument("job_id", shell_complete=complete_job_id)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@rich_help_option("-h", "--help")
def cancel(job_id, force):
    """Cancel a running job.

    Cancels a job that is currently running or pending.
    Completed jobs cannot be canceled.

    Args:
        job_id: Job ID to cancel
        force: Skip confirmation prompt
    """

    async def cancel_job():
        try:
            # Get job status first
            client = APIClient()
            success, job_data = await client.get(f"jobs/{job_id}/status")

            if not success:
                if "404" in str(job_data):
                    echo_error(f"Job '{job_id}' not found")
                else:
                    echo_error(f"Failed to get job status: {job_data}")
                return

            # Check if job can be canceled
            job_status = job_data["status"]
            if job_status in ["completed", "failed", "cancelled"]:
                echo_warning(f"Job is already {job_status} and cannot be canceled")
                return

            # Confirm cancellation
            if not force:
                from ..utils import confirm_action
                if not confirm_action(
                    f"Cancel job '{job_id}' (status: {job_status})?",
                    default=False,
                ):
                    echo_info("Job cancellation cancelled")
                    return

            # Cancel the job
            success, response = await client.post(f"jobs/{job_id}/cancel")

            if success:
                echo_success(f"Job '{job_id}' has been canceled")
            else:
                echo_error(f"Failed to cancel job: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(cancel_job())


@job.command()
@click.option("--days", default=7, help="Remove jobs older than this many days")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@rich_help_option("-h", "--help")
def cleanup(days, force):
    """Clean up old completed and failed jobs.

    Removes completed and failed jobs older than the specified
    number of days to free up database space.

    Args:
        days: Remove jobs older than this many days
        force: Skip confirmation prompt
    """

    async def cleanup_jobs():
        try:
            # Confirm cleanup
            if not force:
                from ..utils import confirm_action
                if not confirm_action(
                    f"Remove all completed/failed jobs older than {days} days?",
                    default=False,
                ):
                    echo_info("Job cleanup cancelled")
                    return

            client = APIClient()
            success, response = await client.delete("jobs/cleanup", params={"days": days})

            if success:
                removed_count = response.get("removed_count", 0)
                echo_success(f"Cleaned up {removed_count} old job(s)")
            else:
                echo_error(f"Failed to cleanup jobs: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(cleanup_jobs())


@job.command()
@rich_help_option("-h", "--help")
def active():
    """Show currently active (running) jobs.

    Displays all jobs that are currently running with
    real-time progress information.
    """

    async def show_active_jobs():
        try:
            client = APIClient()
            success, response = await client.get("jobs/active")

            if success:
                jobs = response.get("jobs", [])

                if not jobs:
                    echo_info("No active jobs")
                    return

                echo_info(f"Found {len(jobs)} active job(s)")
                console.print()

                # Display each active job
                for job in jobs:
                    job_id = job["id"]
                    status = job["status"]
                    progress = int(job["progress"] * 100)
                    job_type = job.get("type", "unknown")
                    
                    # Get current phase from metadata
                    metadata = job.get("metadata", {})
                    phase = metadata.get("phase", "processing")
                    
                    # Create a mini status card
                    info_lines = [
                        f"[bold cyan]Job ID:[/bold cyan] {job_id}",
                        f"[bold cyan]Status:[/bold cyan] {status}",
                        f"[bold cyan]Progress:[/bold cyan] {progress}%",
                        f"[bold cyan]Type:[/bold cyan] {job_type}",
                        f"[bold cyan]Phase:[/bold cyan] {phase}",
                    ]
                    
                    if metadata.get("source"):
                        info_lines.append(f"[bold cyan]Source:[/bold cyan] {metadata['source']}")
                    
                    panel = Panel(
                        "\n".join(info_lines),
                        title=f"[bold yellow]Active Job[/bold yellow]",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                    console.print(panel)
                    console.print()

            else:
                echo_error(f"Failed to get active jobs: {response}")

        except Exception as e:
            echo_error(f"Connection error: {e}")
            echo_info("Make sure the server is running: ctx server up")

    asyncio.run(show_active_jobs())