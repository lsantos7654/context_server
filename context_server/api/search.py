"""Search API endpoints."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from context_server.core.services.embeddings import EmbeddingService, VoyageEmbeddingService
from context_server.core.database import DatabaseManager
from context_server.api.error_handlers import handle_search_errors
from context_server.models.api.search import SearchRequest, SearchResponse, CodeSearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency to get database manager."""
    return request.app.state.db_manager


def get_embedding_service(request: Request) -> EmbeddingService:
    """Dependency to get embedding service."""
    if not hasattr(request.app.state, "embedding_service"):
        request.app.state.embedding_service = EmbeddingService()
    return request.app.state.embedding_service


def get_code_embedding_service(request: Request) -> VoyageEmbeddingService:
    """Dependency to get code embedding service."""
    if not hasattr(request.app.state, "code_embedding_service"):
        request.app.state.code_embedding_service = VoyageEmbeddingService()
    return request.app.state.code_embedding_service


@router.post("/contexts/{context_name}/search")
@handle_search_errors("search documents")
async def search_context(
    context_name: str,
    search_request: SearchRequest,
    format: str = "standard",  # "standard" or "compact"
    db: DatabaseManager = Depends(get_db_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Search documents within a context."""
    start_time = time.time()

    # Verify context exists
    context = await db.get_context_by_name(context_name)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")

    context_id = context["id"]
    results = []

    if search_request.mode.value == "vector":
        # Vector search
        query_embedding = await embedding_service.embed_text(search_request.query)
        results = await db.vector_search(
            context_id=context_id,
            query_embedding=query_embedding,
            limit=search_request.limit,
            min_similarity=0.1,  # Lower threshold for testing
            embedding_type="text",
        )

    elif search_request.mode.value == "fulltext":
        # Full-text search
        results = await db.fulltext_search(
            context_id=context_id,
            query=search_request.query,
            limit=search_request.limit,
        )

    elif search_request.mode.value == "hybrid":
        # Hybrid search - combine vector and full-text
        query_embedding = await embedding_service.embed_text(search_request.query)

        # Get results from both methods
        vector_results = await db.vector_search(
            context_id=context_id,
            query_embedding=query_embedding,
            limit=search_request.limit * 2,  # Get more for merging
            min_similarity=0.1,  # Lower threshold for testing
            embedding_type="text",
        )

        fulltext_results = await db.fulltext_search(
            context_id=context_id,
            query=search_request.query,
            limit=search_request.limit * 2,  # Get more for merging
        )

        # Merge and rank results
        results = _merge_search_results(
            vector_results, fulltext_results, search_request.limit
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported search mode: {search_request.mode}",
        )

    # Format results with enhanced metadata
    formatted_results = []
    for result in results:
        metadata = result.get("metadata", {})
        formatted_result = {
            "id": result["id"],
            "document_id": result.get("document_id"),
            "title": result["title"],
            "content": result["content"],
            "summary": result.get("summary"),
            "summary_model": result.get("summary_model"),
            "score": result["score"],
            "metadata": metadata,
            "url": result.get("url"),
        }
        
        # Extract useful metadata to top level for easier access
        formatted_result.update({
            "page_url": metadata.get("page_url", result.get("url")),
            "source_type": metadata.get("source_type"),
            "base_url": metadata.get("base_url"),
            "is_individual_page": metadata.get("is_individual_page", False),
            "source_title": metadata.get("source_title"),
        })
        formatted_results.append(formatted_result)

    execution_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Search completed: query='{search_request.query}', "
        f"mode={search_request.mode}, results={len(formatted_results)}, "
        f"time={execution_time_ms}ms"
    )

    # Return compact format if requested
    if format == "compact":
        compact_response = await db._transform_to_compact_format(
            formatted_results,
            query=search_request.query,
            mode=search_request.mode.value,
            execution_time_ms=execution_time_ms
        )
        return compact_response
    
    # Return standard format
    return SearchResponse(
        results=formatted_results,
        total=len(formatted_results),
        query=search_request.query,
        mode=search_request.mode.value,
        execution_time_ms=execution_time_ms,
    )


def _merge_search_results(
    vector_results: list, fulltext_results: list, limit: int
) -> list:
    """Merge vector and full-text search results using hybrid ranking."""
    # Create a map of results by chunk ID
    result_map = {}

    # Add vector results with weight
    for result in vector_results:
        chunk_id = result["id"]
        result_map[chunk_id] = {
            **result,
            "vector_score": result["score"],
            "fulltext_score": 0.0,
            "hybrid_score": result["score"] * 0.7,  # Weight vector results at 70%
        }

    # Add/update with full-text results
    for result in fulltext_results:
        chunk_id = result["id"]
        if chunk_id in result_map:
            # Combine scores
            result_map[chunk_id]["fulltext_score"] = result["score"]
            result_map[chunk_id]["hybrid_score"] = (
                result_map[chunk_id]["vector_score"] * 0.7
                + result["score"] * 0.3  # Weight full-text at 30%
            )
        else:
            # New result from full-text only
            result_map[chunk_id] = {
                **result,
                "vector_score": 0.0,
                "fulltext_score": result["score"],
                "hybrid_score": result["score"] * 0.3,  # Only full-text score
            }

    # Sort by hybrid score and return top results
    sorted_results = sorted(
        result_map.values(), key=lambda x: x["hybrid_score"], reverse=True
    )

    # Update final scores and return
    final_results = []
    for result in sorted_results[:limit]:
        result["score"] = result["hybrid_score"]  # Use hybrid score as final score
        final_results.append(result)

    return final_results


@router.post("/contexts/{context_name}/search/code")
@handle_search_errors("search code snippets")
async def search_code_snippets(
    context_name: str,
    search_request: SearchRequest,
    format: str = "standard",  # "standard" or "compact"
    db: DatabaseManager = Depends(get_db_manager),
    code_embedding_service: VoyageEmbeddingService = Depends(get_code_embedding_service),
):
    """Search code snippets within a context using code-optimized embeddings."""
    start_time = time.time()

    # Verify context exists
    context = await db.get_context_by_name(context_name)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")

    context_id = context["id"]
    results = []

    if search_request.mode.value == "vector":
        # Vector search using code embeddings
        query_embedding = await code_embedding_service.embed_text(search_request.query)
        results = await db.vector_search_code_snippets(
            context_id=context_id,
            query_embedding=query_embedding,
            limit=search_request.limit,
            min_similarity=0.1,  # Lower threshold for testing
        )

    elif search_request.mode.value == "fulltext":
        # Full-text search in code snippets
        results = await db.fulltext_search_code_snippets(
            context_id=context_id,
            query=search_request.query,
            limit=search_request.limit,
        )

    elif search_request.mode.value == "hybrid":
        # Hybrid search - combine vector and full-text for code
        query_embedding = await code_embedding_service.embed_text(search_request.query)

        # Get results from both methods
        vector_results = await db.vector_search_code_snippets(
            context_id=context_id,
            query_embedding=query_embedding,
            limit=search_request.limit * 2,  # Get more for merging
            min_similarity=0.1,  # Lower threshold for testing
        )

        fulltext_results = await db.fulltext_search_code_snippets(
            context_id=context_id,
            query=search_request.query,
            limit=search_request.limit * 2,  # Get more for merging
        )

        # Merge and rank results
        results = _merge_search_results(
            vector_results, fulltext_results, search_request.limit
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported search mode: {search_request.mode}",
        )

    # Format code search results with enhanced metadata
    formatted_results = []
    for result in results:
        metadata = result.get("metadata", {})
        
        # Clean metadata to remove deprecated fields
        clean_metadata = {}
        for key, value in metadata.items():
            # Skip deprecated fields that were used for chunks but not code snippets
            if key not in ["language", "start_line", "end_line", "char_start", "char_end", "type", "chunk_index"]:
                clean_metadata[key] = value
        
        formatted_result = {
            "id": result["id"],
            "document_id": result.get("document_id"),
            "content": result["content"],
            "score": result["score"],
            "line_count": result.get("line_count", len(result["content"].split('\n')) if result["content"] else 0),
            "metadata": clean_metadata,
            "url": result.get("url"),
        }
        
        # Only add optional fields if they have values
        page_url = metadata.get("page_url", result.get("url"))
        if page_url and page_url != result.get("url"):
            formatted_result["page_url"] = page_url
            
        if metadata.get("source_type"):
            formatted_result["source_type"] = metadata.get("source_type")
            
        if metadata.get("base_url"):
            formatted_result["base_url"] = metadata.get("base_url")
            
        if metadata.get("source_title"):
            formatted_result["source_title"] = metadata.get("source_title")
        formatted_results.append(formatted_result)

    execution_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Code search completed: query='{search_request.query}', "
        f"mode={search_request.mode}, results={len(formatted_results)}, "
        f"time={execution_time_ms}ms"
    )

    # Return compact format if requested
    if format == "compact":
        compact_response = db._transform_code_to_compact_format(
            formatted_results,
            query=search_request.query,
            execution_time_ms=execution_time_ms
        )
        return compact_response
    
    # Return standard format
    return CodeSearchResponse(
        results=formatted_results,
        total=len(formatted_results),
        query=search_request.query,
        mode=search_request.mode.value,
        execution_time_ms=execution_time_ms,
    )


@router.get("/contexts/{context_name}/search/suggestions")
async def get_search_suggestions(
    context_name: str,
    query: str,
    limit: int = 5,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get search query suggestions based on context content."""
    raise HTTPException(
        status_code=501, detail="Search suggestions not yet implemented"
    )
