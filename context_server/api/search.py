"""Search API endpoints."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.cache import DocumentCacheService
from ..core.enhanced_storage import EnhancedDatabaseManager
from ..core.expansion import ContextExpansionService
from ..core.multi_embedding_service import MultiEmbeddingService
from ..core.multi_modal_search import SearchEngine
from .models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_db_manager(request: Request) -> EnhancedDatabaseManager:
    """Dependency to get enhanced database manager."""
    if not hasattr(request.app.state, "enhanced_db_manager"):
        request.app.state.enhanced_db_manager = EnhancedDatabaseManager()
    return request.app.state.enhanced_db_manager


def get_search_engine(request: Request) -> SearchEngine:
    """Dependency to get multi-modal search engine."""
    if not hasattr(request.app.state, "search_engine"):
        embedding_service = MultiEmbeddingService()
        db_manager = get_db_manager(request)
        request.app.state.search_engine = SearchEngine(embedding_service, db_manager)
    return request.app.state.search_engine


async def get_cache_service(request: Request) -> DocumentCacheService:
    """Dependency to get cache service."""
    if not hasattr(request.app.state, "cache_service"):
        cache_service = DocumentCacheService()
        try:
            await cache_service.initialize()
            request.app.state.cache_service = cache_service
            logger.info("Cache service initialized successfully")
        except Exception as e:
            logger.warning(f"Cache service initialization failed: {e}")
            # Create a dummy cache service that doesn't actually cache
            request.app.state.cache_service = DocumentCacheService()
    return request.app.state.cache_service


async def get_expansion_service(request: Request) -> ContextExpansionService:
    """Dependency to get expansion service."""
    if not hasattr(request.app.state, "expansion_service"):
        db = get_db_manager(request)
        cache = await get_cache_service(request)
        request.app.state.expansion_service = ContextExpansionService(db, cache)
    return request.app.state.expansion_service


@router.post("/contexts/{context_name}/search", response_model=SearchResponse)
async def search_context(
    context_name: str,
    search_request: SearchRequest,
    db: EnhancedDatabaseManager = Depends(get_db_manager),
    search_engine: SearchEngine = Depends(get_search_engine),
):
    """Search documents within a context."""
    start_time = time.time()

    try:
        # Get expansion service - initialize cache service directly
        cache_service = DocumentCacheService()
        try:
            await cache_service.initialize()
            logger.info("Cache service initialized for search")
        except Exception as e:
            logger.warning(f"Cache service initialization failed: {e}")

        expansion_service = ContextExpansionService(db, cache_service)

        # Verify context exists
        context = await db.get_context_by_name(context_name)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        context_id = context["id"]

        # Use multi-modal search engine
        search_response = await search_engine.search(
            query=search_request.query,
            context_id=context_id,
            search_type=search_request.mode.value,
            limit=search_request.limit,
            filters={},
            enable_caching=True,
        )

        results = search_response.results

        # Add warning for large context expansions
        if search_request.expand_context > 50:
            logger.warning(
                f"Large context expansion requested: {search_request.expand_context} lines. "
                f"This may impact performance and memory usage."
            )

        # Apply line-based expansion if requested
        if search_request.expand_context > 0:
            # Expand results using line-based context expansion
            results = await expansion_service.expand_search_results(
                results,
                expand_lines=search_request.expand_context,
                prefer_boundaries=True,
            )

        # Format results with enhanced metadata
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.id,
                "document_id": result.document_id,
                "title": result.title,
                "content": result.content,
                "score": result.similarity_score,
                "metadata": {},  # Will be populated from SearchResult attributes
                "url": result.url,
                "chunk_index": getattr(result, "chunk_index", None),
                "content_type": result.content_type,
                "expansion_info": getattr(result, "expansion_info", None),
                # Extract useful metadata to top level for easier access
                "page_url": result.url,
                "source_type": getattr(result, "source_type", None),
                "programming_language": result.programming_language,
                "quality_score": result.quality_score,
                "matched_keywords": result.matched_keywords,
                "code_elements": result.code_elements,
                "api_references": result.api_references,
            }
            formatted_results.append(formatted_result)

        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Search completed: query='{search_request.query}', "
            f"mode={search_request.mode}, results={len(formatted_results)}, "
            f"time={execution_time_ms}ms"
        )

        return SearchResponse(
            results=formatted_results,
            total=search_response.total_results,
            query=search_request.query,
            mode=search_request.mode.value,
            execution_time_ms=search_response.search_time_ms,
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
    db: EnhancedDatabaseManager = Depends(get_db_manager),
):
    """Get search query suggestions based on context content."""
    # TODO: Implement search suggestions
    # Could use document titles, common terms, etc.
    raise HTTPException(
        status_code=501, detail="Search suggestions not yet implemented"
    )
