"""Context management API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from ..core.storage import DatabaseManager
from .models import ContextCreate, ContextMerge, ContextResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


@router.post("", response_model=ContextResponse, status_code=201)
async def create_context(
    context_data: ContextCreate, db: DatabaseManager = Depends(get_db_manager)
):
    """Create a new context."""
    try:
        # Check if context with same name already exists
        existing = await db.get_context_by_name(context_data.name)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Context with name '{context_data.name}' already exists",
            )

        context = await db.create_context(
            name=context_data.name,
            description=context_data.description,
            embedding_model=context_data.embedding_model,
        )

        logger.info(f"Created context: {context['name']} (ID: {context['id']})")
        return ContextResponse(**context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create context: {e}")
        raise HTTPException(status_code=500, detail="Failed to create context")


@router.get("", response_model=list[ContextResponse])
async def list_contexts(db: DatabaseManager = Depends(get_db_manager)):
    """List all contexts."""
    try:
        contexts = await db.get_contexts()
        return [ContextResponse(**ctx) for ctx in contexts]
    except Exception as e:
        logger.error(f"Failed to list contexts: {e}")
        raise HTTPException(status_code=500, detail="Failed to list contexts")


@router.get("/{context_name}", response_model=ContextResponse)
async def get_context(context_name: str, db: DatabaseManager = Depends(get_db_manager)):
    """Get a specific context by name."""
    try:
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        return ContextResponse(**context)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context {context_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get context")


@router.delete("/{context_name}", status_code=204)
async def delete_context(
    context_name: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Delete a context and all its data."""
    try:
        # Get context to find ID
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        success = await db.delete_context(context["id"])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete context")

        logger.info(f"Deleted context: {context_name}")
        return Response(status_code=204)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete context {context_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete context")


@router.post("/merge", status_code=201)
async def merge_contexts(
    merge_data: ContextMerge, db: DatabaseManager = Depends(get_db_manager)
):
    """Merge multiple contexts into a target context."""
    # TODO: Implement context merging
    # This is a complex operation that needs careful planning
    raise HTTPException(status_code=501, detail="Context merging not yet implemented")


@router.get("/{context_name}/export")
async def export_context(
    context_name: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Export context data."""
    # TODO: Implement context export using pg_dump
    raise HTTPException(status_code=501, detail="Context export not yet implemented")


@router.post("/import")
async def import_context(db: DatabaseManager = Depends(get_db_manager)):
    """Import context data."""
    # TODO: Implement context import from pg_dump
    raise HTTPException(status_code=501, detail="Context import not yet implemented")
