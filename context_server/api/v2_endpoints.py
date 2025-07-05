"""Version 2 API endpoints with enhanced search and metadata features."""

import logging
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.advanced_search_api import (
    AdvancedSearchAPI,
    AdvancedSearchResponse,
    ContentMetadata,
    ContextAnalytics,
)
from ..core.enhanced_storage import EnhancedDatabaseManager
from ..core.llm_endpoints import LLMOptimizedEndpoints
from ..core.multi_modal_search import SearchEngine
from ..core.relationship_mapping import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class EnhancedSearchRequest(BaseModel):
    """Request model for enhanced search."""

    query: str = Field(..., description="Search query")
    context_id: str = Field(..., description="Context to search in")
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of results"
    )
    include_recommendations: bool = Field(
        default=True, description="Include context recommendations"
    )
    include_clusters: bool = Field(
        default=True, description="Include related topic clusters"
    )
    include_graph_insights: bool = Field(
        default=True, description="Include knowledge graph insights"
    )
    enable_caching: bool = Field(
        default=True, description="Enable caching for performance"
    )


class SimilarContentRequest(BaseModel):
    """Request model for similar content search."""

    url: str = Field(..., description="URL to find similar content for")
    context_id: str = Field(..., description="Context to search in")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class TrendingTopicsRequest(BaseModel):
    """Request model for trending topics."""

    context_id: str = Field(..., description="Context to analyze")
    time_window_days: int = Field(
        default=30, ge=1, le=365, description="Time window in days"
    )


# API Router
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])


# Dependencies
def get_db_manager(request) -> EnhancedDatabaseManager:
    """Get the enhanced database manager."""
    if not hasattr(request.app.state, "enhanced_db_manager"):
        request.app.state.enhanced_db_manager = EnhancedDatabaseManager()
        # TODO: Initialize properly with async
    return request.app.state.enhanced_db_manager


def get_search_engine(request) -> SearchEngine:
    """Get the multi-modal search engine."""
    if not hasattr(request.app.state, "search_engine"):
        from ..core.multi_embedding_service import MultiEmbeddingService

        embedding_service = MultiEmbeddingService()
        db_manager = get_db_manager(request)
        request.app.state.search_engine = SearchEngine(embedding_service, db_manager)
    return request.app.state.search_engine


def get_knowledge_graph_builder(request) -> KnowledgeGraphBuilder:
    """Get the knowledge graph builder."""
    if not hasattr(request.app.state, "knowledge_graph_builder"):
        from ..core.multi_embedding_service import MultiEmbeddingService

        embedding_service = MultiEmbeddingService()
        db_manager = get_db_manager(request)
        request.app.state.knowledge_graph_builder = KnowledgeGraphBuilder(
            embedding_service, db_manager
        )
    return request.app.state.knowledge_graph_builder


def get_llm_endpoints(request) -> LLMOptimizedEndpoints:
    """Get the LLM-optimized endpoints service."""
    if not hasattr(request.app.state, "llm_endpoints"):
        from ..core.multi_embedding_service import MultiEmbeddingService

        embedding_service = MultiEmbeddingService()
        db_manager = get_db_manager(request)
        knowledge_graph_builder = get_knowledge_graph_builder(request)
        request.app.state.llm_endpoints = LLMOptimizedEndpoints(
            embedding_service, db_manager, knowledge_graph_builder
        )
    return request.app.state.llm_endpoints


async def get_advanced_search_api(request) -> AdvancedSearchAPI:
    """Get the advanced search API instance."""
    if not hasattr(request.app.state, "advanced_search_api"):
        search_engine = get_search_engine(request)
        db_manager = get_db_manager(request)
        llm_endpoints = get_llm_endpoints(request)
        knowledge_graph_builder = get_knowledge_graph_builder(request)
        request.app.state.advanced_search_api = AdvancedSearchAPI(
            search_engine, db_manager, llm_endpoints, knowledge_graph_builder
        )
    return request.app.state.advanced_search_api


@v2_router.post("/search/enhanced", response_model=Dict)
async def enhanced_search(
    request_data: EnhancedSearchRequest,
    request: Request,
) -> Dict:
    """
    Perform enhanced search with metadata, recommendations, and insights.

    This endpoint provides comprehensive search capabilities including:
    - Traditional search results
    - Search metadata and statistics
    - Related topic clusters
    - Context recommendations
    - Knowledge graph insights
    - Performance metrics
    """
    try:
        api = await get_advanced_search_api(request)
        response = await api.enhanced_search(
            query=request_data.query,
            context_id=request_data.context_id,
            limit=request_data.limit,
            include_recommendations=request_data.include_recommendations,
            include_clusters=request_data.include_clusters,
            include_graph_insights=request_data.include_graph_insights,
            enable_caching=request_data.enable_caching,
        )

        # Convert response to dictionary for JSON serialization
        return {
            "search_response": {
                "query": response.search_response.query,
                "results": [
                    {
                        "url": result.url,
                        "title": result.title,
                        "content": result.content,
                        "similarity_score": result.similarity_score,
                        "relevance_score": result.relevance_score,
                        "content_type": result.content_type,
                        "programming_language": result.programming_language,
                        "summary": result.summary,
                        "strategy_used": result.strategy_used,
                        "quality_score": result.quality_score,
                        "matched_keywords": result.matched_keywords,
                        "code_elements": result.code_elements,
                        "api_references": result.api_references,
                    }
                    for result in response.search_response.results
                ],
                "total_results": response.search_response.total_results,
                "strategies_used": response.search_response.strategies_used,
                "search_time_ms": response.search_response.search_time_ms,
                "result_quality_score": response.search_response.result_quality_score,
                "search_confidence": response.search_response.search_confidence,
            },
            "search_metadata": {
                "total_content_pieces": response.search_metadata.total_content_pieces,
                "content_type_distribution": response.search_metadata.content_type_distribution,
                "programming_language_distribution": response.search_metadata.programming_language_distribution,
                "quality_score_distribution": response.search_metadata.quality_score_distribution,
                "embedding_model_distribution": response.search_metadata.embedding_model_distribution,
                "average_content_quality": response.search_metadata.average_content_quality,
                "search_coverage_percentage": response.search_metadata.search_coverage_percentage,
                "cluster_coverage": response.search_metadata.cluster_coverage,
            },
            "related_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "name": cluster.name,
                    "description": cluster.description,
                    "content_count": len(cluster.content_urls),
                    "topic_keywords": cluster.topic_keywords,
                    "programming_languages": cluster.programming_languages,
                    "difficulty_level": cluster.difficulty_level,
                    "quality_score": cluster.quality_score,
                }
                for cluster in response.related_clusters
            ],
            "context_recommendations": (
                {
                    "primary_recommendations": [
                        {
                            "url": rec.url,
                            "title": rec.title,
                            "relevance_score": rec.relevance_score,
                            "recommendation_type": rec.recommendation_type,
                            "why_recommended": rec.why_recommended,
                            "difficulty_level": rec.difficulty_level,
                            "estimated_reading_time": rec.estimated_reading_time,
                        }
                        for rec in response.context_recommendations.primary_recommendations
                    ],
                    "learning_path": response.context_recommendations.learning_path,
                    "related_clusters": response.context_recommendations.related_clusters,
                    "knowledge_gaps": response.context_recommendations.knowledge_gaps,
                    "next_steps": response.context_recommendations.next_steps,
                    "total_score": response.context_recommendations.total_score,
                }
                if response.context_recommendations
                else None
            ),
            "knowledge_graph_insights": response.knowledge_graph_insights,
            "performance_metrics": {
                "total_search_time_ms": response.total_search_time_ms,
                "cache_hit_ratio": response.cache_hit_ratio,
                "search_strategy_effectiveness": response.search_strategy_effectiveness,
            },
        }

    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@v2_router.get("/content/{context_id}/metadata")
async def get_content_metadata(
    context_id: str,
    request: Request,
    url: Optional[str] = Query(None, description="Specific URL to get metadata for"),
) -> Union[Dict, List[Dict]]:
    """
    Get detailed metadata for content in a context.

    If URL is provided, returns metadata for that specific content piece.
    Otherwise, returns metadata for all content in the context.
    """
    try:
        api = await get_advanced_search_api(request)
        metadata = await api.get_content_metadata(context_id, url)

        if isinstance(metadata, ContentMetadata):
            # Single content metadata
            return {
                "url": metadata.url,
                "title": metadata.title,
                "content_type": metadata.content_type,
                "programming_language": metadata.programming_language,
                "quality_score": metadata.quality_score,
                "readability_score": metadata.readability_score,
                "complexity_indicators": metadata.complexity_indicators,
                "topic_keywords": metadata.topic_keywords,
                "code_elements": metadata.code_elements,
                "api_references": metadata.api_references,
                "embedding_model": metadata.embedding_model,
                "last_updated": metadata.last_updated.isoformat(),
                "relationship_count": metadata.relationship_count,
                "cluster_memberships": metadata.cluster_memberships,
            }
        else:
            # List of content metadata
            return [
                {
                    "url": meta.url,
                    "title": meta.title,
                    "content_type": meta.content_type,
                    "programming_language": meta.programming_language,
                    "quality_score": meta.quality_score,
                    "readability_score": meta.readability_score,
                    "complexity_indicators": meta.complexity_indicators,
                    "topic_keywords": meta.topic_keywords,
                    "code_elements": meta.code_elements,
                    "api_references": meta.api_references,
                    "embedding_model": meta.embedding_model,
                    "last_updated": meta.last_updated.isoformat(),
                    "relationship_count": meta.relationship_count,
                    "cluster_memberships": meta.cluster_memberships,
                }
                for meta in metadata
            ]

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Content metadata retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metadata retrieval failed: {str(e)}"
        )


@v2_router.get("/context/{context_id}/analytics")
async def get_context_analytics(
    context_id: str,
    request: Request,
) -> Dict:
    """
    Get comprehensive analytics for a context.

    Provides detailed insights including:
    - Content breakdown by type and language
    - Quality metrics and distributions
    - Knowledge graph statistics
    - Search performance metrics
    - Trending topics and temporal insights
    """
    try:
        api = await get_advanced_search_api(request)
        analytics = await api.get_context_analytics(context_id)

        return {
            "context_id": analytics.context_id,
            "content_overview": {
                "total_content_pieces": analytics.total_content_pieces,
                "content_breakdown": analytics.content_breakdown,
                "language_breakdown": analytics.language_breakdown,
            },
            "quality_metrics": {
                "average_quality_score": analytics.average_quality_score,
                "quality_distribution": analytics.quality_distribution,
                "average_readability_score": analytics.average_readability_score,
            },
            "knowledge_graph": {
                "total_relationships": analytics.total_relationships,
                "relationship_type_breakdown": analytics.relationship_type_breakdown,
                "total_clusters": analytics.total_clusters,
                "cluster_size_distribution": analytics.cluster_size_distribution,
                "graph_density": analytics.graph_density,
                "modularity_score": analytics.modularity_score,
            },
            "search_performance": {
                "most_searched_topics": analytics.most_searched_topics,
                "search_success_rate": analytics.search_success_rate,
                "average_result_relevance": analytics.average_result_relevance,
            },
            "temporal_insights": {
                "content_creation_timeline": analytics.content_creation_timeline,
                "search_activity_timeline": analytics.search_activity_timeline,
                "last_analysis_update": analytics.last_analysis_update.isoformat(),
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Context analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@v2_router.post("/search/similar")
async def search_similar_content(
    request_data: SimilarContentRequest,
    request: Request,
) -> Dict:
    """
    Find content similar to a specific URL.

    Uses the content's metadata, keywords, and embeddings to find
    semantically similar content within the same context.
    """
    try:
        api = await get_advanced_search_api(request)
        response = await api.search_similar_content(
            url=request_data.url,
            context_id=request_data.context_id,
            limit=request_data.limit,
        )

        # Convert response to dictionary (similar to enhanced_search)
        return {
            "reference_url": request_data.url,
            "similar_content": [
                {
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "similarity_score": result.similarity_score,
                    "relevance_score": result.relevance_score,
                    "content_type": result.content_type,
                    "programming_language": result.programming_language,
                    "summary": result.summary,
                    "quality_score": result.quality_score,
                }
                for result in response.search_response.results
            ],
            "search_metadata": {
                "total_similar_items": response.search_metadata.total_content_pieces,
                "content_type_distribution": response.search_metadata.content_type_distribution,
                "programming_language_distribution": response.search_metadata.programming_language_distribution,
                "average_similarity": response.search_metadata.average_content_quality,
            },
            "performance_metrics": {
                "search_time_ms": response.total_search_time_ms,
                "cache_hit_ratio": response.cache_hit_ratio,
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Similar content search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Similar content search failed: {str(e)}"
        )


@v2_router.post("/context/trending")
async def get_trending_topics(
    request_data: TrendingTopicsRequest,
    request: Request,
) -> Dict:
    """
    Get trending topics in a context based on content analysis and search activity.

    Returns topics ranked by frequency, trend scores, and relevance.
    """
    try:
        api = await get_advanced_search_api(request)
        trending = await api.get_trending_topics(
            context_id=request_data.context_id,
            time_window_days=request_data.time_window_days,
        )

        return {
            "context_id": request_data.context_id,
            "time_window_days": request_data.time_window_days,
            "trending_topics": trending,
            "analysis_summary": {
                "total_topics": len(trending),
                "topic_types": list(set(topic["type"] for topic in trending)),
                "top_trend_score": max(
                    (topic["trend_score"] for topic in trending), default=0.0
                ),
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Trending topics analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Trending topics analysis failed: {str(e)}"
        )


@v2_router.get("/health")
async def health_check() -> Dict:
    """Health check endpoint for API v2."""
    return {
        "status": "healthy",
        "version": "2.0",
        "features": [
            "enhanced_search",
            "content_metadata",
            "context_analytics",
            "similar_content_search",
            "trending_topics",
            "knowledge_graph_insights",
        ],
    }
