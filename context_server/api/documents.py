"""Document management API endpoints."""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from ..core.cache import DocumentCacheService
from ..core.processing import DocumentProcessor
from ..core.storage import DatabaseManager
from .models import DocumentDelete, DocumentIngest, DocumentsResponse, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


def get_cache_service(request: Request) -> DocumentCacheService:
    """Dependency to get document cache service."""
    if not hasattr(request.app.state, "cache_service"):
        request.app.state.cache_service = DocumentCacheService()
    return request.app.state.cache_service


def get_processor(request: Request) -> DocumentProcessor:
    """Dependency to get document processor."""
    if not hasattr(request.app.state, "processor"):
        request.app.state.processor = DocumentProcessor()
    return request.app.state.processor


@router.post("/contexts/{context_name}/documents", status_code=202)
async def ingest_document(
    context_name: str,
    document_data: DocumentIngest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager),
    processor: DocumentProcessor = Depends(get_processor),
    cache_service: DocumentCacheService = Depends(get_cache_service),
):
    """Ingest a document into a context (async processing)."""
    try:
        # Verify context exists
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Create processing job
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{context_name}"

        # Start background processing
        background_tasks.add_task(
            _process_document_background,
            job_id,
            context,
            document_data,
            db,
            processor,
            cache_service,
        )

        logger.info(f"Started document ingestion job: {job_id}")

        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Document ingestion started for {document_data.source}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start document ingestion: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to start document ingestion"
        )


async def _process_document_background(
    job_id: str,
    context: dict,
    document_data: DocumentIngest,
    db: DatabaseManager,
    processor: DocumentProcessor,
    cache_service: DocumentCacheService = None,
):
    """Background task for document processing."""
    try:
        logger.info(
            f"Processing document: {document_data.source} for context {context['name']}"
        )

        # Process the document based on source type
        if document_data.source_type.value == "url":
            result = await processor.process_url(
                url=document_data.source, options=document_data.options
            )
        elif document_data.source_type.value == "file":
            result = await processor.process_file(
                file_path=document_data.source, options=document_data.options
            )
        elif document_data.source_type.value == "git":
            result = await processor.process_git_repo(
                repo_url=document_data.source, options=document_data.options
            )
        else:
            raise ValueError(f"Unsupported source type: {document_data.source_type}")

        # Store processed documents
        for doc in result.documents:
            doc_id = await db.create_document(
                context_id=context["id"],
                url=doc.url,
                title=doc.title,
                content=doc.content,
                metadata=doc.metadata,
                source_type=document_data.source_type.value,
            )

            # Store chunks with embeddings and line tracking
            for i, chunk in enumerate(doc.chunks):
                await db.create_chunk(
                    document_id=doc_id,
                    context_id=context["id"],
                    content=chunk.content,
                    embedding=chunk.embedding,
                    chunk_index=i,
                    metadata=chunk.metadata,
                    tokens=chunk.tokens,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )

            # Update document chunk count
            await db.update_document_chunk_count(doc_id)

            # Cache the document for fast line-based expansion
            if cache_service:
                try:
                    await cache_service.cache_document(doc_id, doc.content)
                    logger.debug(f"Cached document {doc_id} for line-based expansion")
                except Exception as e:
                    logger.warning(f"Failed to cache document {doc_id}: {e}")
                    # Don't fail the whole operation if caching fails

        logger.info(
            f"Completed processing job: {job_id}, processed {len(result.documents)} documents"
        )

    except Exception as e:
        logger.error(f"Failed to process document in job {job_id}: {e}")
        # TODO: Store job status/errors in database for tracking


@router.get("/contexts/{context_name}/documents", response_model=DocumentsResponse)
async def list_documents(
    context_name: str,
    offset: int = 0,
    limit: int = 50,
    db: DatabaseManager = Depends(get_db_manager),
):
    """List documents in a context."""
    try:
        # Verify context exists
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get documents
        result = await db.get_documents(context["id"], offset=offset, limit=limit)
        return DocumentsResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents for context {context_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.delete("/contexts/{context_name}/documents", status_code=200)
async def delete_documents(
    context_name: str,
    delete_data: DocumentDelete,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Delete documents from a context."""
    try:
        # Verify context exists
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Delete documents
        deleted_count = await db.delete_documents(
            context["id"], delete_data.document_ids
        )

        logger.info(f"Deleted {deleted_count} documents from context {context_name}")

        return {
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} documents",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete documents from context {context_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete documents")


@router.get("/contexts/{context_name}/documents/{doc_id}/raw")
async def get_document_raw(
    context_name: str, doc_id: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Get raw document content."""
    try:
        # Get context
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get document
        document = await db.get_document_by_id(context["id"], doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get document {doc_id} from context {context_name}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get document")
