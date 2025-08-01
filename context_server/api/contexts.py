"""Context management API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from context_server.api.error_handlers import handle_context_errors
from context_server.core.database import DatabaseManager
from context_server.models.api.contexts import (
    ContextCreate,
    ContextMerge,
    ContextResponse,
)
from context_server.models.api.export import (
    ContextExport,
    ContextImportRequest,
    ContextImportResponse,
    ContextMergeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


@router.post("", response_model=ContextResponse, status_code=201)
@handle_context_errors("create")
async def create_context(
    context_data: ContextCreate, db: DatabaseManager = Depends(get_db_manager)
):
    """Create a new context."""
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


@router.get("", response_model=list[ContextResponse])
@handle_context_errors("list")
async def list_contexts(db: DatabaseManager = Depends(get_db_manager)):
    """List all contexts."""
    contexts = await db.get_contexts()
    return [ContextResponse(**ctx) for ctx in contexts]


@router.get("/{context_name}", response_model=ContextResponse)
@handle_context_errors("get")
async def get_context(context_name: str, db: DatabaseManager = Depends(get_db_manager)):
    """Get a specific context by name."""
    context = await db.get_context_by_name(context_name)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")

    return ContextResponse(**context)


@router.delete("/{context_name}", status_code=204)
@handle_context_errors("delete")
async def delete_context(
    context_name: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Delete a context and all its data."""
    # Get context to find ID
    context = await db.get_context_by_name(context_name)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")

    success = await db.delete_context(context["id"])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete context")

    logger.info(f"Deleted context: {context_name}")
    return Response(status_code=204)


@router.post("/merge", response_model=ContextMergeResponse, status_code=201)
@handle_context_errors("merge")
async def merge_contexts(
    merge_data: ContextMerge, db: DatabaseManager = Depends(get_db_manager)
):
    """Merge multiple contexts into a target context.

    Supports two merge modes:
    - union: Combine all documents from source contexts
    - intersection: Only keep documents that exist in ALL source contexts
    """
    try:
        result = await db.merge_contexts(
            source_contexts=merge_data.source_contexts,
            target_context=merge_data.target_context,
            mode=merge_data.mode.value,
        )

        logger.info(
            f"Merged contexts: {merge_data.source_contexts} -> {merge_data.target_context} ({merge_data.mode})"
        )
        return ContextMergeResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context merge failed: {e}")
        raise HTTPException(status_code=500, detail="Context merge failed")


@router.get("/{context_name}/export", response_model=ContextExport)
@handle_context_errors("export")
async def export_context(
    context_name: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Export complete context data for backup/migration.

    Returns all context data including documents, chunks, code snippets,
    and embeddings in a structured JSON format.
    """
    try:
        export_data = await db.export_context(context_name)

        logger.info(f"Exported context: {context_name}")
        return ContextExport(**export_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Context export failed: {e}")
        raise HTTPException(status_code=500, detail="Context export failed")


@router.post("/import", response_model=ContextImportResponse, status_code=201)
@handle_context_errors("import")
async def import_context(
    import_request: ContextImportRequest, db: DatabaseManager = Depends(get_db_manager)
):
    """Import context data from export.

    Supports importing complete context data with transaction safety.
    Can optionally overwrite existing contexts with the same name.
    """
    try:
        result = await db.import_context(
            {
                "context_data": import_request.context_data.dict(),
                "overwrite_existing": import_request.overwrite_existing,
            }
        )

        logger.info(f"Imported context: {result['context_name']}")
        return ContextImportResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context import failed: {e}")
        raise HTTPException(status_code=500, detail="Context import failed")
