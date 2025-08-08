"""Job status and management API endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from context_server.api.error_handlers import handle_document_errors
from context_server.core.database import DatabaseManager
from context_server.models.api.system import (
    ActiveJobsResponse,
    JobCancelResponse,
    JobCleanupResponse,
    JobStatus,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
@handle_document_errors("job_status")
async def get_job_status(
    job_id: str,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get the status of a processing job."""
    job_data = await db.get_job_status(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return structured job status response
    return JobStatusResponse(**job_data)


@router.get("/jobs/active", response_model=ActiveJobsResponse)
async def get_active_jobs(
    context_id: str = None,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get all active jobs, optionally filtered by context."""
    jobs = await db.get_active_jobs(context_id)

    return ActiveJobsResponse(
        active_jobs=[
            JobStatus(
                id=job["id"],
                type=job["type"],
                context=job.get("context_id", ""),
                status=job["status"],
                progress=job["progress"],
                started_at=job["started_at"],
            )
            for job in jobs
        ],
        total=len(jobs),
    )


@router.delete("/jobs/{job_id}", response_model=JobCancelResponse)
async def cancel_job(
    job_id: str,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Cancel a running job (mark as failed)."""
    # Check if job exists and is cancelable
    job_data = await db.get_job_status(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_data["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job_data['status']}",
        )

    # Mark job as failed
    await db.complete_job(job_id, error_message="Job cancelled by user")

    logger.info(f"Job {job_id} cancelled by user")

    return JobCancelResponse(
        message=f"Job {job_id} cancelled successfully",
        job_id=job_id,
        status="cancelled",
    )


@router.post("/admin/jobs/cleanup", response_model=JobCleanupResponse)
async def cleanup_old_jobs(
    days: int = 7,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Clean up completed jobs older than specified days (admin endpoint)."""
    deleted_count = await db.cleanup_old_jobs(days)

    return JobCleanupResponse(
        message=f"Cleaned up {deleted_count} old jobs",
        deleted_count=deleted_count,
        days=days,
    )
