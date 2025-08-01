"""Job status and management API endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from context_server.api.error_handlers import handle_document_errors
from context_server.core.database import DatabaseManager
from context_server.models.api.system import JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


@router.get("/jobs/{job_id}/status")
@handle_document_errors("job_status")
async def get_job_status(
    job_id: str,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get the status of a processing job."""
    job_data = await db.get_job_status(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return raw job data with all fields for CLI compatibility
    return job_data


@router.get("/jobs/active")
async def get_active_jobs(
    context_id: str = None,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get all active jobs, optionally filtered by context."""
    jobs = await db.get_active_jobs(context_id)

    return {
        "active_jobs": [
            {
                "id": job["id"],
                "type": job["type"],
                "context_id": job["context_id"],
                "status": job["status"],
                "progress": job["progress"],
                "started_at": job["started_at"],
                "updated_at": job["updated_at"],
                "metadata": job["metadata"],
            }
            for job in jobs
        ],
        "total": len(jobs),
    }


@router.delete("/jobs/{job_id}")
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

    return {
        "message": f"Job {job_id} cancelled successfully",
        "job_id": job_id,
        "status": "cancelled",
    }


@router.post("/admin/jobs/cleanup")
async def cleanup_old_jobs(
    days: int = 7,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Clean up completed jobs older than specified days (admin endpoint)."""
    deleted_count = await db.cleanup_old_jobs(days)

    return {
        "message": f"Cleaned up {deleted_count} old jobs",
        "deleted_count": deleted_count,
        "days": days,
    }
