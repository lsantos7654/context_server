"""Unit tests for v2 API endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from context_server.api.v2_endpoints import (
    EnhancedSearchRequest,
    SimilarContentRequest,
    TrendingTopicsRequest,
    v2_router,
)
from context_server.core.advanced_search_api import (
    AdvancedSearchResponse,
    ContentMetadata,
    ContextAnalytics,
    SearchMetadata,
)
from context_server.core.llm_endpoints import (
    ContentRecommendation,
    ContextRecommendation,
)
from context_server.core.multi_modal_search import SearchResponse, SearchResult
from context_server.core.relationship_mapping import TopicCluster


class TestV2Endpoints:
    """Test V2 API endpoints functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_api = Mock()

    def create_sample_search_result(self, url: str, **kwargs):
        """Create a sample SearchResult for testing."""
        defaults = {
            "title": f"Title for {url}",
            "content": f"Content for {url}",
            "similarity_score": 0.8,
            "relevance_score": 0.8,
            "content_type": "code_example",
            "programming_language": "python",
            "summary": f"Summary for {url}",
            "strategy_used": "semantic_search",
            "embedding_model": "text-embedding-3-small",
            "quality_score": 0.8,
            "matched_keywords": ["python", "test"],
            "code_elements": ["def", "function"],
            "api_references": [],
        }
        defaults.update(kwargs)

        return SearchResult(url=url, **defaults)

    def test_enhanced_search_request_validation(self):
        """Test enhanced search request validation."""
        # Valid request
        valid_request = EnhancedSearchRequest(
            query="Python functions",
            context_id="test-context",
            limit=10,
        )
        assert valid_request.query == "Python functions"
        assert valid_request.context_id == "test-context"
        assert valid_request.limit == 10
        assert valid_request.include_recommendations is True  # Default value

        # Test limit validation
        with pytest.raises(ValueError):
            EnhancedSearchRequest(
                query="test",
                context_id="test",
                limit=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            EnhancedSearchRequest(
                query="test",
                context_id="test",
                limit=101,  # Above maximum
            )

    def test_similar_content_request_validation(self):
        """Test similar content request validation."""
        valid_request = SimilarContentRequest(
            url="https://example.com/test",
            context_id="test-context",
            limit=5,
        )
        assert valid_request.url == "https://example.com/test"
        assert valid_request.context_id == "test-context"
        assert valid_request.limit == 5

    def test_trending_topics_request_validation(self):
        """Test trending topics request validation."""
        valid_request = TrendingTopicsRequest(
            context_id="test-context",
            time_window_days=7,
        )
        assert valid_request.context_id == "test-context"
        assert valid_request.time_window_days == 7

        # Test time window validation
        with pytest.raises(ValueError):
            TrendingTopicsRequest(
                context_id="test",
                time_window_days=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            TrendingTopicsRequest(
                context_id="test",
                time_window_days=366,  # Above maximum
            )

    @pytest.mark.asyncio
    async def test_enhanced_search_endpoint_success(self):
        """Test successful enhanced search endpoint."""
        from context_server.api.v2_endpoints import enhanced_search

        # Mock request
        request = EnhancedSearchRequest(
            query="Python functions",
            context_id="test-context",
            limit=5,
        )

        # Mock API response
        mock_search_response = SearchResponse(
            query=request.query,
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result("https://example.com/1"),
                self.create_sample_search_result("https://example.com/2"),
            ],
            total_results=2,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        mock_advanced_response = AdvancedSearchResponse(
            search_response=mock_search_response,
            search_metadata=SearchMetadata(
                total_content_pieces=2,
                content_type_distribution={"code_example": 2},
                programming_language_distribution={"python": 2},
                quality_score_distribution={"high": 2, "medium": 0, "low": 0},
                embedding_model_distribution={"text-embedding-3-small": 2},
                average_content_quality=0.8,
                search_coverage_percentage=20.0,
                cluster_coverage={},
            ),
            related_clusters=[],
            context_recommendations=None,
            knowledge_graph_insights={},
            total_search_time_ms=120,
            cache_hit_ratio=0.0,
            search_strategy_effectiveness={"semantic_search": 1.0},
        )

        self.mock_api.enhanced_search = AsyncMock(return_value=mock_advanced_response)

        # Call endpoint
        result = await enhanced_search(request, self.mock_api)

        # Verify result structure
        assert isinstance(result, dict)
        assert "search_response" in result
        assert "search_metadata" in result
        assert "related_clusters" in result
        assert "performance_metrics" in result

        # Verify search response
        search_resp = result["search_response"]
        assert search_resp["query"] == request.query
        assert search_resp["total_results"] == 2
        assert len(search_resp["results"]) == 2

        # Verify search metadata
        metadata = result["search_metadata"]
        assert metadata["total_content_pieces"] == 2
        assert metadata["average_content_quality"] == 0.8

        # Verify performance metrics
        perf = result["performance_metrics"]
        assert perf["total_search_time_ms"] == 120
        assert perf["cache_hit_ratio"] == 0.0

    @pytest.mark.asyncio
    async def test_enhanced_search_endpoint_error(self):
        """Test enhanced search endpoint error handling."""
        from context_server.api.v2_endpoints import enhanced_search

        request = EnhancedSearchRequest(
            query="test",
            context_id="test-context",
        )

        # Mock API to raise exception
        self.mock_api.enhanced_search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Call endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await enhanced_search(request, self.mock_api)

        assert exc_info.value.status_code == 500
        assert "Search failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_content_metadata_single(self):
        """Test getting metadata for a single content piece."""
        from datetime import datetime

        from context_server.api.v2_endpoints import get_content_metadata

        context_id = "test-context"
        url = "https://example.com/test"

        # Mock content metadata
        mock_metadata = ContentMetadata(
            url=url,
            title="Test Content",
            content_type="code_example",
            programming_language="python",
            quality_score=0.8,
            readability_score=0.9,
            complexity_indicators=["medium"],
            topic_keywords=["python", "test"],
            code_elements=["def", "function"],
            api_references=[],
            embedding_model="test-model",
            last_updated=datetime(2023, 1, 1, 12, 0, 0),
            relationship_count=2,
            cluster_memberships=["cluster1"],
        )

        self.mock_api.get_content_metadata = AsyncMock(return_value=mock_metadata)

        # Call endpoint
        result = await get_content_metadata(context_id, url, self.mock_api)

        # Verify result
        assert isinstance(result, dict)
        assert result["url"] == url
        assert result["title"] == "Test Content"
        assert result["content_type"] == "code_example"
        assert result["programming_language"] == "python"
        assert result["relationship_count"] == 2
        assert result["cluster_memberships"] == ["cluster1"]

    @pytest.mark.asyncio
    async def test_get_content_metadata_all(self):
        """Test getting metadata for all content in a context."""
        from datetime import datetime

        from context_server.api.v2_endpoints import get_content_metadata

        context_id = "test-context"

        # Mock list of content metadata
        mock_metadata_list = [
            ContentMetadata(
                url="https://example.com/1",
                title="Content 1",
                content_type="code_example",
                programming_language="python",
                quality_score=0.8,
                readability_score=0.9,
                complexity_indicators=["low"],
                topic_keywords=["python"],
                code_elements=["def"],
                api_references=[],
                embedding_model="test-model",
                last_updated=datetime(2023, 1, 1, 12, 0, 0),
                relationship_count=1,
                cluster_memberships=["cluster1"],
            ),
            ContentMetadata(
                url="https://example.com/2",
                title="Content 2",
                content_type="tutorial",
                programming_language="javascript",
                quality_score=0.7,
                readability_score=0.8,
                complexity_indicators=["medium"],
                topic_keywords=["javascript"],
                code_elements=["function"],
                api_references=[],
                embedding_model="test-model",
                last_updated=datetime(2023, 1, 1, 12, 0, 0),
                relationship_count=0,
                cluster_memberships=[],
            ),
        ]

        self.mock_api.get_content_metadata = AsyncMock(return_value=mock_metadata_list)

        # Call endpoint
        result = await get_content_metadata(context_id, None, self.mock_api)

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/1"
        assert result[1]["url"] == "https://example.com/2"
        assert result[0]["programming_language"] == "python"
        assert result[1]["programming_language"] == "javascript"

    @pytest.mark.asyncio
    async def test_get_context_analytics(self):
        """Test getting context analytics."""
        from datetime import datetime

        from context_server.api.v2_endpoints import get_context_analytics

        context_id = "test-context"

        # Mock context analytics
        mock_analytics = ContextAnalytics(
            context_id=context_id,
            total_content_pieces=10,
            content_breakdown={"code_example": 5, "tutorial": 3, "api_reference": 2},
            language_breakdown={"python": 6, "javascript": 4},
            average_quality_score=0.8,
            quality_distribution={"high": 6, "medium": 3, "low": 1},
            average_readability_score=0.75,
            total_relationships=15,
            relationship_type_breakdown={
                "semantic_similarity": 8,
                "code_dependency": 4,
                "api_reference": 3,
            },
            total_clusters=3,
            cluster_size_distribution={"small": 1, "medium": 2, "large": 0},
            graph_density=0.6,
            modularity_score=0.8,
            most_searched_topics=[("python", 25), ("javascript", 18)],
            search_success_rate=0.85,
            average_result_relevance=0.78,
            content_creation_timeline={},
            search_activity_timeline={},
            last_analysis_update=datetime(2023, 1, 1, 12, 0, 0),
        )

        self.mock_api.get_context_analytics = AsyncMock(return_value=mock_analytics)

        # Call endpoint
        result = await get_context_analytics(context_id, self.mock_api)

        # Verify result structure
        assert isinstance(result, dict)
        assert result["context_id"] == context_id

        # Verify content overview
        content_overview = result["content_overview"]
        assert content_overview["total_content_pieces"] == 10
        assert content_overview["content_breakdown"]["code_example"] == 5

        # Verify quality metrics
        quality_metrics = result["quality_metrics"]
        assert quality_metrics["average_quality_score"] == 0.8

        # Verify knowledge graph metrics
        kg_metrics = result["knowledge_graph"]
        assert kg_metrics["total_relationships"] == 15
        assert kg_metrics["total_clusters"] == 3

    @pytest.mark.asyncio
    async def test_search_similar_content(self):
        """Test similar content search."""
        from context_server.api.v2_endpoints import search_similar_content

        request = SimilarContentRequest(
            url="https://example.com/reference",
            context_id="test-context",
            limit=5,
        )

        # Mock similar content response
        mock_search_response = SearchResponse(
            query="constructed similarity query",
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result("https://example.com/similar1"),
                self.create_sample_search_result("https://example.com/similar2"),
            ],
            total_results=2,
            strategies_used=["semantic_search"],
            search_time_ms=80,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.85,
            search_confidence=0.9,
        )

        mock_advanced_response = AdvancedSearchResponse(
            search_response=mock_search_response,
            search_metadata=SearchMetadata(
                total_content_pieces=2,
                content_type_distribution={"code_example": 2},
                programming_language_distribution={"python": 2},
                quality_score_distribution={"high": 2, "medium": 0, "low": 0},
                embedding_model_distribution={"text-embedding-3-small": 2},
                average_content_quality=0.85,
                search_coverage_percentage=15.0,
                cluster_coverage={},
            ),
            related_clusters=[],
            context_recommendations=None,
            knowledge_graph_insights={},
            total_search_time_ms=90,
            cache_hit_ratio=0.2,
            search_strategy_effectiveness={"semantic_search": 1.0},
        )

        self.mock_api.search_similar_content = AsyncMock(
            return_value=mock_advanced_response
        )

        # Call endpoint
        result = await search_similar_content(request, self.mock_api)

        # Verify result
        assert isinstance(result, dict)
        assert result["reference_url"] == request.url
        assert "similar_content" in result
        assert len(result["similar_content"]) == 2
        assert "search_metadata" in result
        assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_get_trending_topics(self):
        """Test trending topics endpoint."""
        from context_server.api.v2_endpoints import get_trending_topics

        request = TrendingTopicsRequest(
            context_id="test-context",
            time_window_days=7,
        )

        # Mock trending topics
        mock_trending = [
            {
                "topic": "python",
                "type": "programming_language",
                "frequency": 15,
                "trend_score": 0.75,
                "related_content_count": 15,
            },
            {
                "topic": "functions",
                "type": "keyword",
                "frequency": 12,
                "trend_score": 0.6,
                "related_content_count": 12,
            },
            {
                "topic": "javascript",
                "type": "programming_language",
                "frequency": 8,
                "trend_score": 0.4,
                "related_content_count": 8,
            },
        ]

        self.mock_api.get_trending_topics = AsyncMock(return_value=mock_trending)

        # Call endpoint
        result = await get_trending_topics(request, self.mock_api)

        # Verify result
        assert isinstance(result, dict)
        assert result["context_id"] == request.context_id
        assert result["time_window_days"] == 7
        assert "trending_topics" in result
        assert len(result["trending_topics"]) == 3

        # Verify analysis summary
        summary = result["analysis_summary"]
        assert summary["total_topics"] == 3
        assert "programming_language" in summary["topic_types"]
        assert "keyword" in summary["topic_types"]
        assert summary["top_trend_score"] == 0.75

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint."""
        from context_server.api.v2_endpoints import health_check

        result = await health_check()

        assert isinstance(result, dict)
        assert result["status"] == "healthy"
        assert result["version"] == "2.0"
        assert "features" in result
        assert "enhanced_search" in result["features"]
        assert "content_metadata" in result["features"]
        assert "context_analytics" in result["features"]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for various endpoints."""
        from context_server.api.v2_endpoints import (
            get_content_metadata,
            get_context_analytics,
        )

        # Test ValueError handling (404)
        self.mock_api.get_content_metadata = AsyncMock(
            side_effect=ValueError("Content not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_content_metadata("test-context", "nonexistent", self.mock_api)

        assert exc_info.value.status_code == 404
        assert "Content not found" in str(exc_info.value.detail)

        # Test general exception handling (500)
        self.mock_api.get_context_analytics = AsyncMock(
            side_effect=Exception("Database error")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_context_analytics("test-context", self.mock_api)

        assert exc_info.value.status_code == 500
        assert "Analytics failed" in str(exc_info.value.detail)
