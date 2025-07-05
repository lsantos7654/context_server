"""Search API endpoints."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.embeddings import EmbeddingService
from ..core.storage import DatabaseManager
from .models import SearchRequest, SearchResponse

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


@router.post("/contexts/{context_name}/search", response_model=SearchResponse)
async def search_context(
    context_name: str,
    search_request: SearchRequest,
    db: DatabaseManager = Depends(get_db_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Search documents within a context."""
    start_time = time.time()

    try:
        # Verify context exists
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        context_id = context["id"]
        results = []

        if search_request.mode.value == "vector":
            # Pure vector search
            query_embedding = await embedding_service.embed_text(search_request.query)
            results = await db.vector_search(
                context_id=context_id,
                query_embedding=query_embedding,
                limit=search_request.limit,
                min_similarity=0.1,  # Lower threshold for testing
                expand_context=search_request.expand_context,
            )

        elif search_request.mode.value == "fulltext":
            # Pure full-text search
            results = await db.fulltext_search(
                context_id=context_id,
                query=search_request.query,
                limit=search_request.limit,
                expand_context=search_request.expand_context,
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
                expand_context=search_request.expand_context,
            )

            fulltext_results = await db.fulltext_search(
                context_id=context_id,
                query=search_request.query,
                limit=search_request.limit * 2,  # Get more for merging
                expand_context=search_request.expand_context,
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

        # Handle load_full_doc option
        if search_request.load_full_doc and results:
            # Replace chunk content with full document content
            processed_docs = set()
            enhanced_results = []

            for result in results:
                doc_id = result.get("document_id")
                if doc_id and doc_id not in processed_docs:
                    # Get full document content
                    full_doc = await db.get_document_by_id(context_id, doc_id)
                    if full_doc:
                        enhanced_result = {
                            "id": result["id"],
                            "document_id": doc_id,
                            "title": full_doc["title"],
                            "content": full_doc["content"],  # Full document content
                            "score": result["score"],
                            "metadata": full_doc["metadata"],
                            "url": full_doc.get("url"),
                            "chunk_index": result.get("chunk_index"),
                            "content_type": "full_document",
                        }
                        enhanced_results.append(enhanced_result)
                        processed_docs.add(doc_id)

            formatted_results = enhanced_results
        else:
            # Format results normally
            formatted_results = [
                {
                    "id": result["id"],
                    "document_id": result.get("document_id"),
                    "title": result["title"],
                    "content": result["content"],
                    "score": result["score"],
                    "metadata": result["metadata"],
                    "url": result.get("url"),
                    "chunk_index": result.get("chunk_index"),
                    "content_type": "expanded_chunk"
                    if search_request.expand_context > 0
                    else "chunk",
                }
                for result in results
            ]

        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Search completed: query='{search_request.query}', "
            f"mode={search_request.mode}, results={len(formatted_results)}, "
            f"time={execution_time_ms}ms"
        )

        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query=search_request.query,
            mode=search_request.mode.value,
            execution_time_ms=execution_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for context {context_name}: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


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


@router.get("/contexts/{context_name}/search/suggestions")
async def get_search_suggestions(
    context_name: str,
    query: str,
    limit: int = 5,
    db: DatabaseManager = Depends(get_db_manager),
):
    """Get search query suggestions based on context content."""
    # TODO: Implement search suggestions
    # Could use document titles, common terms, etc.
    raise HTTPException(
        status_code=501, detail="Search suggestions not yet implemented"
    )
