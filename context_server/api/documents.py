"""Document management API endpoints."""

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from context_server.core.pipeline import DocumentProcessor
from context_server.core.database import DatabaseManager
from context_server.api.error_handlers import handle_document_errors
from context_server.models.api.documents import DocumentDelete, DocumentIngest, DocumentsResponse
from context_server.models.api.system import JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


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
    """Background task for document processing with real progress tracking and cancellation support."""
    try:
        # Add overall job timeout (30 minutes)
        job_timeout = 1800  # 30 minutes

        async def _check_cancellation():
            """Check if job has been cancelled"""
            job_status = await db.get_job_status(job_id)
            return job_status and job_status.get("status") == "cancelled"

        async def _process_with_cancellation_checks():
            # Check for cancellation before starting
            if await _check_cancellation():
                logger.info(f"Job {job_id} was cancelled before processing started")
                return

            # Create the job record
            await db.create_job(
                job_id=job_id,
                job_type="document_extraction",
                context_id=context["id"],
                metadata={
                    "source": document_data.source,
                    "source_type": document_data.source_type.value,
                    "options": document_data.options,
                },
            )

            logger.info(
                f"Processing document: {document_data.source} for context {context['name']}"
            )

            # Update job to processing status
            await db.update_job_progress(job_id, 0.1, status="processing")

            # Check for cancellation before main processing
            if await _check_cancellation():
                logger.info(f"Job {job_id} was cancelled during setup")
                return

            # Process the document based on source type
            if document_data.source_type.value == "url":
                result = await processor.process_url(
                    url=document_data.source,
                    options=document_data.options,
                    job_id=job_id,
                    db=db,
                )
            elif document_data.source_type.value == "file":
                result = await processor.process_file(
                    file_path=document_data.source,
                    options=document_data.options,
                    job_id=job_id,
                    db=db,
                )
            else:
                raise ValueError(
                    f"Unsupported source type: {document_data.source_type}"
                )

            # Check for cancellation after processing
            if await _check_cancellation():
                logger.info(f"Job {job_id} was cancelled after processing")
                return

            if not result.success:
                await db.complete_job(job_id, error_message=result.error)
                return

            # Process and store documents individually for fault tolerance
            await db.update_job_progress(
                job_id,
                0.8,
                metadata={
                    "phase": "storing_documents",
                    "documents_extracted": len(result.documents),
                },
            )

            total_docs = len(result.documents)
            stored_docs = 0
            failed_docs = []
            total_chunks = 0
            total_code_snippets = 0

            # Store processed documents one by one to preserve partial work
            for doc_idx, doc in enumerate(result.documents):
                # Check for cancellation before each document
                if await _check_cancellation():
                    logger.info(
                        f"Job {job_id} was cancelled during document storage ({stored_docs}/{total_docs} completed)"
                    )
                    return
                try:
                    # SIMPLIFIED APPROACH: Create document first, then store code snippets with real document_id
                    
                    # 1. Store main document with temporary original content
                    doc_id = await asyncio.wait_for(
                        db.create_document(
                            context_id=context["id"],
                            url=doc.url,
                            title=doc.title,
                            content=doc.content,  # Temporary - original content
                            metadata=doc.metadata,
                            source_type=document_data.source_type.value,
                        ),
                        timeout=30.0,  # 30 second timeout per document
                    )
                    
                    # 2. Store code snippets with real document_id to get real UUIDs
                    code_snippets_with_uuids = []
                    for snippet in doc.code_snippets:
                        # Generate preview for the code snippet using existing pipeline logic
                        preview = processor.code_extractor._generate_code_preview(snippet.content)
                        
                        # Store code snippet with real document_id
                        real_snippet_id = await db.create_code_snippet(
                            document_id=doc_id,
                            context_id=context["id"],
                            content=snippet.content,
                            embedding=snippet.embedding,
                            metadata=snippet.metadata,
                            start_line=snippet.start_line,
                            end_line=snippet.end_line,
                            char_start=snippet.char_start,
                            char_end=snippet.char_end,
                            preview=preview,
                            snippet_type=snippet.metadata.get("snippet_type", "code_block"),
                        )
                        
                        # Store snippet info with real UUID for placeholder creation
                        snippet_with_uuid = {
                            **snippet.metadata,
                            "content": snippet.content,
                            "uuid": real_snippet_id,
                        }
                        code_snippets_with_uuids.append(snippet_with_uuid)
                    
                    # 3. Create cleaned content with real UUIDs using the code extractor
                    cleaned_content = processor.code_extractor.create_cleaned_content_with_real_uuids(
                        doc.content, code_snippets_with_uuids
                    )
                    
                    # 4. Update document with cleaned content
                    async with db.pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE documents SET content = $1 WHERE id = $2",
                            cleaned_content,
                            uuid.UUID(doc_id)
                        )
                    
                    # 5. Store raw document content separately  
                    await asyncio.wait_for(
                        db.create_raw_document(
                            document_id=doc_id,
                            raw_content=doc.content,  # Original raw content
                        ),
                        timeout=30.0,  # 30 second timeout per document
                    )

                    # 6. Store chunks with code snippet references
                    chunk_count = len(doc.chunks)
                    for i, chunk in enumerate(doc.chunks):
                        # Find code snippet references in chunk content
                        chunk_code_snippet_ids = []
                        
                        # Check if this chunk content overlaps with any code snippet positions
                        for snippet_with_uuid in code_snippets_with_uuids:
                            snippet_start = snippet_with_uuid.get("start_pos", 0)
                            snippet_end = snippet_with_uuid.get("end_pos", 0)
                            chunk_start = chunk.char_start or 0
                            chunk_end = chunk.char_end or 0
                            
                            # If chunk and snippet overlap, link them
                            if (chunk_start <= snippet_end and chunk_end >= snippet_start):
                                chunk_code_snippet_ids.append(snippet_with_uuid["uuid"])
                        
                        await db.create_chunk(
                            document_id=doc_id,  # Link chunks to main document
                            context_id=context["id"],
                            content=chunk.content,
                            embedding=chunk.embedding,
                            chunk_index=i,
                            code_snippet_ids=chunk_code_snippet_ids,
                            metadata=chunk.metadata,
                            tokens=chunk.tokens,
                            title=chunk.title,
                            summary=chunk.summary,
                            summary_model=chunk.summary_model,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                        )

                    # Update document chunk count
                    await db.update_document_chunk_count(doc_id)

                    # Track successful storage
                    stored_docs += 1
                    total_chunks += chunk_count
                    total_code_snippets += len(doc.code_snippets)

                    logger.info(
                        f"Successfully stored document {doc_idx + 1}/{total_docs}: {doc.title}"
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"Timeout storing document {doc_idx + 1}/{total_docs}: {doc.title}"
                    )
                    failed_docs.append(
                        {"url": doc.url, "title": doc.title, "error": "Storage timeout"}
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to store document {doc_idx + 1}/{total_docs}: {doc.title} - {e}"
                    )
                    failed_docs.append(
                        {"url": doc.url, "title": doc.title, "error": str(e)}
                    )

                # Update progress after each document (successful or failed)
                storage_progress = 0.8 + (0.2 * ((doc_idx + 1) / total_docs))
                await db.update_job_progress(
                    job_id,
                    storage_progress,
                    metadata={
                        "phase": "storing_documents",
                        "stored_docs": stored_docs,
                        "failed_docs": len(failed_docs),
                        "total_docs": total_docs,
                    },
                )

            # Complete the job with comprehensive results
            result_data = {
                "documents_processed": stored_docs,
                "documents_failed": len(failed_docs),
                "total_chunks": total_chunks,
                "total_code_snippets": total_code_snippets,
            }

            # Add failure details if any
            if failed_docs:
                result_data["failed_documents"] = failed_docs
                logger.warning(
                    f"Job {job_id} completed with {len(failed_docs)} failed documents"
                )

            await db.complete_job(job_id, result_data=result_data)

            logger.info(
                f"Completed processing job: {job_id}, processed {stored_docs} documents"
            )

        # Apply timeout to the entire processing function
        try:
            await asyncio.wait_for(
                _process_with_cancellation_checks(), timeout=job_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Job {job_id} timed out after {job_timeout}s")
            await db.complete_job(
                job_id, error_message=f"Job timed out after {job_timeout} seconds"
            )

    except Exception as e:
        logger.error(f"Failed to process document in job {job_id}: {e}")
        # Store job failure in database
        await db.complete_job(job_id, error_message=str(e))


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
    context_name: str, 
    doc_id: str, 
    page_number: int = 1,
    page_size: int = 25000,
    raw: bool = False,  # False = cleaned content, True = raw content
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get raw document content with pagination support for Claude's 25k token limit."""
    try:
        # Get context
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get document
        document = await db.get_document_by_id(context["id"], doc_id, raw)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Handle pagination
        full_content = document.get("content", "")
        full_content_length = len(full_content)
        
        # Calculate pagination
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        
        # Get the page content
        page_content = full_content[start_index:end_index]
        
        # Calculate total pages
        total_pages = max(1, (full_content_length + page_size - 1) // page_size)
        
        # Validate page number
        if page_number > total_pages:
            raise HTTPException(
                status_code=400, 
                detail=f"Page {page_number} not found. Document has {total_pages} pages."
            )
        
        # Create paginated response
        paginated_document = {
            **document,
            "content": page_content,
            "full_content_length": full_content_length,
            "pagination": {
                "page_number": page_number,
                "page_size": page_size,
                "total_pages": total_pages,
                "current_page_length": len(page_content),
                "has_next_page": page_number < total_pages,
                "has_previous_page": page_number > 1,
                "content_truncated": full_content_length > page_size,
            }
        }

        return paginated_document

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


@router.get("/contexts/{context_name}/chunks/{chunk_id}")
async def get_chunk(
    context_name: str, chunk_id: str, db: DatabaseManager = Depends(get_db_manager)
):
    """Get a specific chunk by ID with full content and metadata."""
    try:
        # Get context
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get chunk
        chunk = await db.get_chunk_by_id(chunk_id, context["id"])
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        return chunk

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get chunk {chunk_id} from context {context_name}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get chunk")
