"""Document management API endpoints."""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

# Cache service removed in simplification
from ..core.processing import DocumentProcessor
from ..core.storage import DatabaseManager
from .error_handlers import handle_document_errors
from .models import DocumentDelete, DocumentIngest, DocumentsResponse, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


# Cache service removed in simplification


def get_processor(request: Request) -> DocumentProcessor:
    """Dependency to get document processor."""
    if not hasattr(request.app.state, "processor"):
        request.app.state.processor = DocumentProcessor()
    return request.app.state.processor


@router.post("/contexts/{context_name}/documents", status_code=202)
@handle_document_errors("ingest")
async def ingest_document(
    context_name: str,
    document_data: DocumentIngest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager),
    processor: DocumentProcessor = Depends(get_processor),
):
    """Ingest a document into a context (async processing)."""
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
    )

    logger.info(f"Started document ingestion job: {job_id}")

    return {
        "job_id": job_id,
        "status": "processing",
        "message": f"Document ingestion started for {document_data.source}",
    }


async def _process_document_background(
    job_id: str,
    context: dict,
    document_data: DocumentIngest,
    db: DatabaseManager,
    processor: DocumentProcessor,
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

            # Store code snippets with embeddings
            for snippet in doc.code_snippets:
                await db.create_code_snippet(
                    document_id=doc_id,
                    context_id=context["id"],
                    content=snippet.content,
                    language=snippet.language,
                    embedding=snippet.embedding,
                    metadata=snippet.metadata,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                    char_start=snippet.char_start,
                    char_end=snippet.char_end,
                    snippet_type=snippet.metadata.get("type", "code_block"),
                )

            # Update document chunk count
            await db.update_document_chunk_count(doc_id)

            # Note: Documents are no longer cached during extraction
            # Caching now only happens during search when expand-context is requested
            # This improves memory usage and follows the intended design

        logger.info(
            f"Completed processing job: {job_id}, processed {len(result.documents)} documents"
        )

    except Exception as e:
        logger.error(f"Failed to process document in job {job_id}: {e}")
        # TODO: Store job status/errors in database for tracking


@router.get("/contexts/{context_name}/documents", response_model=DocumentsResponse)
@handle_document_errors("list")
async def list_documents(
    context_name: str,
    offset: int = 0,
    limit: int = 50,
    db: DatabaseManager = Depends(get_db_manager),
):
    """List documents in a context."""
    # Verify context exists
    context = await db.get_context_by_name(context_name)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")

    # Get documents
    result = await db.get_documents(context["id"], offset=offset, limit=limit)
    return DocumentsResponse(**result)


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


@router.get("/contexts/{context_name}/documents/{doc_id}/code-snippets")
async def list_document_code_snippets(
    context_name: str, doc_id: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Get all code snippets for a document."""
    try:
        # Get context
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get code snippets
        snippets = await db.get_code_snippets_by_document(doc_id, context["id"])

        return {
            "snippets": snippets,
            "total": len(snippets),
            "document_id": doc_id,
            "context_name": context_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get code snippets for document {doc_id} in context {context_name}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get code snippets")


@router.get("/contexts/{context_name}/code-snippets/{snippet_id}")
async def get_code_snippet(
    context_name: str, snippet_id: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Get a specific code snippet by ID."""
    try:
        # Get context
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get code snippet
        snippet = await db.get_code_snippet_by_id(snippet_id, context["id"])
        if not snippet:
            raise HTTPException(status_code=404, detail="Code snippet not found")

        return snippet

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get code snippet {snippet_id} from context {context_name}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get code snippet")
