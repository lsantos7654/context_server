"""Job tracking operations."""

import json
import logging
import uuid
from typing import Any

from context_server.core.database.base import DatabaseManagerBase
from context_server.core.database.utils import format_uuid, parse_metadata, parse_uuid

logger = logging.getLogger(__name__)


class JobManager(DatabaseManagerBase):
    """Manages job-related database operations."""

    async def create_job(
        self,
        job_id: str,
        job_type: str,
        context_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new processing job."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (id, type, context_id, status, metadata)
                VALUES ($1, $2, $3, 'pending', $4)
                """,
                job_id,
                job_type,
                uuid.UUID(context_id) if context_id else None,
                json.dumps(metadata or {}),
            )
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    async def update_job_progress(
        self,
        job_id: str,
        progress: float,
        status: str | None = None,
        metadata: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job progress and status."""
        async with self.pool.acquire() as conn:
            # Build dynamic update query
            updates = ["progress = $2", "updated_at = NOW()"]
            params = [job_id, progress]
            param_count = 2

            if status:
                param_count += 1
                updates.append(f"status = ${param_count}")
                params.append(status)

            if metadata:
                param_count += 1
                updates.append(f"metadata = ${param_count}")
                params.append(json.dumps(metadata))

            if error_message:
                param_count += 1
                updates.append(f"error_message = ${param_count}")
                params.append(error_message)

            if status in ["completed", "failed"]:
                updates.append("completed_at = NOW()")

            query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = $1"
            await conn.execute(query, *params)

        logger.debug(f"Updated job {job_id}: progress={progress}, status={status}")

    async def get_job_status(self, job_id: str) -> dict | None:
        """Get job status and details."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, type, context_id, status, progress,
                    started_at, updated_at, completed_at,
                    metadata, error_message, result_data
                FROM jobs
                WHERE id = $1
                """,
                job_id,
            )

            if not row:
                return None

            return {
                "id": row["id"],
                "type": row["type"],
                "context_id": str(row["context_id"]) if row["context_id"] else None,
                "status": row["status"],
                "progress": row["progress"],
                "started_at": (
                    row["started_at"].isoformat() if row["started_at"] else None
                ),
                "updated_at": (
                    row["updated_at"].isoformat() if row["updated_at"] else None
                ),
                "completed_at": (
                    row["completed_at"].isoformat() if row["completed_at"] else None
                ),
                "metadata": parse_metadata(row["metadata"]),
                "error_message": row["error_message"],
                "result_data": (
                    json.loads(row["result_data"]) if row["result_data"] else {}
                ),
            }

    async def complete_job(
        self,
        job_id: str,
        result_data: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark job as completed or failed."""
        status = "failed" if error_message else "completed"
        progress = 1.0

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE jobs
                SET status = $2, progress = $3, completed_at = NOW(),
                    updated_at = NOW(), result_data = $4, error_message = $5
                WHERE id = $1
                """,
                job_id,
                status,
                progress,
                json.dumps(result_data or {}),
                error_message,
            )

        logger.info(f"Completed job {job_id} with status: {status}")

    async def get_active_jobs(self, context_id: str | None = None) -> list[dict]:
        """Get all active (non-completed) jobs."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    id, type, context_id, status, progress,
                    started_at, updated_at, metadata
                FROM jobs
                WHERE status NOT IN ('completed', 'failed')
            """
            params = []

            if context_id:
                query += " AND context_id = $1"
                params.append(uuid.UUID(context_id))

            query += " ORDER BY started_at DESC"
            rows = await conn.fetch(query, *params)

            return [
                {
                    "id": row["id"],
                    "type": row["type"],
                    "context_id": str(row["context_id"]) if row["context_id"] else None,
                    "status": row["status"],
                    "progress": row["progress"],
                    "started_at": (
                        row["started_at"].isoformat() if row["started_at"] else None
                    ),
                    "updated_at": (
                        row["updated_at"].isoformat() if row["updated_at"] else None
                    ),
                    "metadata": parse_metadata(row["metadata"]),
                }
                for row in rows
            ]

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed')
                AND completed_at < NOW() - INTERVAL '%s days'
                """,
                days,
            )
            deleted_count = int(result.split()[-1])
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return deleted_count


__all__ = ["JobManager"]
