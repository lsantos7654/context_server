"""Unit tests for advanced search API with enhanced metadata functionality."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.advanced_search_api import (
    AdvancedSearchAPI,
    AdvancedSearchResponse,
    ContentMetadata,
    ContextAnalytics,
    SearchMetadata,
)
from context_server.core.content_analysis import ContentAnalysis
from context_server.core.llm_endpoints import (
    ContentRecommendation,
    ContextRecommendation,
)
from context_server.core.multi_modal_search import SearchResponse, SearchResult
from context_server.core.relationship_mapping import (
    ContentRelationship,
    KnowledgeGraph,
    RelationshipType,
    TopicCluster,
)


class TestAdvancedSearchAPI:
    """Test advanced search API functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_engine = Mock()
        self.mock_database_manager = Mock()
        self.mock_llm_endpoints = Mock()
        self.mock_knowledge_graph_builder = Mock()

        self.api = AdvancedSearchAPI(
            search_engine=self.mock_search_engine,
            database_manager=self.mock_database_manager,
            llm_endpoints=self.mock_llm_endpoints,
            knowledge_graph_builder=self.mock_knowledge_graph_builder,
        )

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

    def create_sample_content_analysis(self, url: str, **kwargs):
        """Create a sample ContentAnalysis for testing."""
        # Handle primary_language -> programming_language mapping
        if "programming_language" in kwargs:
            kwargs["primary_language"] = kwargs.pop("programming_language")

        defaults = {
            "title": f"Title for {url}",
            "summary": f"Summary for {url}",
            "content_type": "code_example",
            "primary_language": "python",
            "topic_keywords": ["python", "function", "example"],
            "code_elements": ["def", "return"],
            "api_references": [],
            "complexity_indicators": ["medium"],
            "readability_score": 0.8,
            "quality_indicators": {"has_examples": True},
            "raw_content": f"Content for {url}",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        defaults.update(kwargs)

        return ContentAnalysis(url=url, **defaults)

    @pytest.mark.asyncio
    async def test_enhanced_search(self):
        """Test enhanced search functionality."""
        query = "Python function examples"
        context_id = "test-context"

        # Mock search response
        mock_search_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result("https://example.com/1"),
                self.create_sample_search_result(
                    "https://example.com/2", programming_language="javascript"
                ),
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

        self.mock_search_engine.search = AsyncMock(return_value=mock_search_response)

        # Mock database manager
        self.mock_database_manager.get_context_stats = AsyncMock(
            return_value={"total_chunks": 10}
        )
        self.mock_database_manager.load_topic_clusters = AsyncMock(return_value=[])
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[]
        )

        # Mock LLM endpoints
        mock_context_recommendation = ContextRecommendation(
            primary_recommendations=[],
            learning_path=[],
            related_clusters=[],
            knowledge_gaps=[],
            next_steps=[],
            total_score=0.8,
        )
        self.mock_llm_endpoints.recommend_context = AsyncMock(
            return_value=mock_context_recommendation
        )

        # Mock knowledge graph builder
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=KnowledgeGraph(
                relationships=[],
                clusters=[],
                total_nodes=10,
                total_edges=5,
                cluster_count=2,
                average_node_degree=1.0,
                graph_density=0.5,
                modularity_score=0.7,
                coverage_ratio=0.8,
            )
        )

        # Perform enhanced search
        response = await self.api.enhanced_search(query, context_id)

        # Verify response structure
        assert isinstance(response, AdvancedSearchResponse)
        assert response.search_response == mock_search_response
        assert isinstance(response.search_metadata, SearchMetadata)
        assert response.context_recommendations == mock_context_recommendation
        assert isinstance(response.knowledge_graph_insights, dict)
        assert response.total_search_time_ms >= 0  # Can be 0 for very fast operations
        assert 0.0 <= response.cache_hit_ratio <= 1.0

        # Verify search metadata
        metadata = response.search_metadata
        assert metadata.total_content_pieces == 2
        assert "code_example" in metadata.content_type_distribution
        assert "python" in metadata.programming_language_distribution
        assert "javascript" in metadata.programming_language_distribution

    @pytest.mark.asyncio
    async def test_enhanced_search_with_caching(self):
        """Test enhanced search with caching enabled."""
        query = "Python test"
        context_id = "test-context"

        # Setup mock response
        mock_search_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[self.create_sample_search_result("https://example.com/1")],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=50,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        self.mock_search_engine.search = AsyncMock(return_value=mock_search_response)
        self.mock_database_manager.get_context_stats = AsyncMock(
            return_value={"total_chunks": 5}
        )
        self.mock_database_manager.load_topic_clusters = AsyncMock(return_value=[])
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[]
        )
        self.mock_llm_endpoints.recommend_context = AsyncMock(
            return_value=ContextRecommendation(
                primary_recommendations=[],
                learning_path=[],
                related_clusters=[],
                knowledge_gaps=[],
                next_steps=[],
                total_score=0.8,
            )
        )
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=KnowledgeGraph(
                relationships=[],
                clusters=[],
                total_nodes=5,
                total_edges=3,
                cluster_count=1,
                average_node_degree=1.2,
                graph_density=0.6,
                modularity_score=0.8,
                coverage_ratio=0.9,
            )
        )

        # First call - should hit database
        response1 = await self.api.enhanced_search(
            query, context_id, enable_caching=True
        )
        assert response1.cache_hit_ratio == 0.0  # First call, no cache hits yet

        # Second call - should hit cache
        response2 = await self.api.enhanced_search(
            query, context_id, enable_caching=True
        )
        assert response2.cache_hit_ratio == 0.5  # 1 hit out of 2 total calls

        # Verify search engine was only called once
        self.mock_search_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_content_metadata_single(self):
        """Test getting metadata for a single content piece."""
        context_id = "test-context"
        url = "https://example.com/test"

        # Mock content analysis
        mock_analysis = self.create_sample_content_analysis(url)
        self.mock_database_manager.get_content_analysis = AsyncMock(
            return_value=mock_analysis
        )

        # Mock relationships and clusters
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[
                ContentRelationship(
                    source_url=url,
                    target_url="https://example.com/other",
                    relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                    strength=0.8,
                    confidence=0.9,
                    discovered_method="test",
                    supporting_evidence=[],
                    context_elements=[],
                    discovery_timestamp=0.0,
                )
            ]
        )

        self.mock_database_manager.load_topic_clusters = AsyncMock(
            return_value=[
                TopicCluster(
                    cluster_id="cluster1",
                    name="Python Functions",
                    description="Python function examples",
                    content_urls=[url, "https://example.com/other"],
                    topic_keywords=["python", "functions"],
                    programming_languages=["python"],
                    content_types=["code_example"],
                    difficulty_level="beginner",
                    coherence_score=0.8,
                    coverage_score=0.7,
                    quality_score=0.9,
                    related_clusters=[],
                    parent_cluster=None,
                    child_clusters=[],
                )
            ]
        )

        # Get content metadata
        metadata = await self.api.get_content_metadata(context_id, url)

        # Verify metadata structure
        assert isinstance(metadata, ContentMetadata)
        assert metadata.url == url
        assert metadata.content_type == "code_example"
        assert metadata.programming_language == "python"
        assert metadata.relationship_count == 1
        assert "cluster1" in metadata.cluster_memberships

    @pytest.mark.asyncio
    async def test_get_content_metadata_all(self):
        """Test getting metadata for all content in a context."""
        context_id = "test-context"

        # Mock multiple content analyses
        mock_analyses = [
            self.create_sample_content_analysis("https://example.com/1"),
            self.create_sample_content_analysis(
                "https://example.com/2", programming_language="javascript"
            ),
        ]
        self.mock_database_manager.get_all_content_analyses = AsyncMock(
            return_value=mock_analyses
        )

        # Mock empty relationships and clusters
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[]
        )
        self.mock_database_manager.load_topic_clusters = AsyncMock(return_value=[])

        # Get all content metadata
        metadata_list = await self.api.get_content_metadata(context_id)

        # Verify metadata list
        assert isinstance(metadata_list, list)
        assert len(metadata_list) == 2
        assert all(isinstance(metadata, ContentMetadata) for metadata in metadata_list)
        assert metadata_list[0].programming_language == "python"
        assert metadata_list[1].programming_language == "javascript"

    @pytest.mark.asyncio
    async def test_get_context_analytics(self):
        """Test getting comprehensive context analytics."""
        context_id = "test-context"

        # Mock content analyses
        mock_analyses = [
            self.create_sample_content_analysis(
                "https://example.com/1",
                content_type="code_example",
                primary_language="python",
            ),
            self.create_sample_content_analysis(
                "https://example.com/2",
                content_type="tutorial",
                primary_language="python",
            ),
            self.create_sample_content_analysis(
                "https://example.com/3",
                content_type="api_reference",
                primary_language="javascript",
            ),
        ]
        self.mock_database_manager.get_all_content_analyses = AsyncMock(
            return_value=mock_analyses
        )

        # Mock relationships
        mock_relationships = [
            ContentRelationship(
                source_url="https://example.com/1",
                target_url="https://example.com/2",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=0.0,
            )
        ]
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=mock_relationships
        )

        # Mock clusters
        mock_clusters = [
            TopicCluster(
                cluster_id="cluster1",
                name="Python Content",
                description="Python-related content",
                content_urls=["https://example.com/1", "https://example.com/2"],
                topic_keywords=["python"],
                programming_languages=["python"],
                content_types=["code_example", "tutorial"],
                difficulty_level="beginner",
                coherence_score=0.8,
                coverage_score=0.7,
                quality_score=0.9,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            )
        ]
        self.mock_database_manager.load_topic_clusters = AsyncMock(
            return_value=mock_clusters
        )

        # Mock knowledge graph
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=KnowledgeGraph(
                relationships=mock_relationships,
                clusters=mock_clusters,
                total_nodes=3,
                total_edges=1,
                cluster_count=1,
                average_node_degree=0.67,
                graph_density=0.33,
                modularity_score=0.8,
                coverage_ratio=0.67,
            )
        )

        # Get context analytics
        analytics = await self.api.get_context_analytics(context_id)

        # Verify analytics structure
        assert isinstance(analytics, ContextAnalytics)
        assert analytics.context_id == context_id
        assert analytics.total_content_pieces == 3
        assert analytics.content_breakdown["code_example"] == 1
        assert analytics.content_breakdown["tutorial"] == 1
        assert analytics.content_breakdown["api_reference"] == 1
        assert analytics.language_breakdown["python"] == 2
        assert analytics.language_breakdown["javascript"] == 1
        assert analytics.total_relationships == 1
        assert analytics.total_clusters == 1

    @pytest.mark.asyncio
    async def test_search_similar_content(self):
        """Test finding similar content to a specific URL."""
        context_id = "test-context"
        url = "https://example.com/reference"

        # Mock content analysis for reference URL
        mock_analysis = self.create_sample_content_analysis(
            url,
            summary="Python function documentation",
            topic_keywords=["python", "functions", "documentation"],
        )
        self.mock_database_manager.get_content_analysis = AsyncMock(
            return_value=mock_analysis
        )

        # Mock search response for similarity query
        mock_search_response = SearchResponse(
            query="Python function documentation python functions documentation",
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result(
                    "https://example.com/similar1", programming_language="python"
                ),
                self.create_sample_search_result(
                    "https://example.com/similar2", programming_language="python"
                ),
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

        self.mock_search_engine.search = AsyncMock(return_value=mock_search_response)
        self.mock_database_manager.get_context_stats = AsyncMock(
            return_value={"total_chunks": 10}
        )
        self.mock_database_manager.load_topic_clusters = AsyncMock(return_value=[])
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[]
        )
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=KnowledgeGraph(
                relationships=[],
                clusters=[],
                total_nodes=10,
                total_edges=5,
                cluster_count=2,
                average_node_degree=1.0,
                graph_density=0.5,
                modularity_score=0.7,
                coverage_ratio=0.8,
            )
        )

        # Search for similar content
        response = await self.api.search_similar_content(url, context_id)

        # Verify response
        assert isinstance(response, AdvancedSearchResponse)
        assert len(response.search_response.results) == 2
        assert all(
            result.programming_language == "python"
            for result in response.search_response.results
        )

        # Verify the search was called with constructed similarity query
        self.mock_search_engine.search.assert_called_once()
        call_kwargs = self.mock_search_engine.search.call_args.kwargs
        assert "python" in call_kwargs["query"]
        assert "functions" in call_kwargs["query"]

    @pytest.mark.asyncio
    async def test_get_trending_topics(self):
        """Test getting trending topics in a context."""
        context_id = "test-context"

        # Mock content analyses with various topics and languages
        mock_analyses = [
            self.create_sample_content_analysis(
                "https://example.com/1",
                topic_keywords=["python", "functions", "async"],
                primary_language="python",
            ),
            self.create_sample_content_analysis(
                "https://example.com/2",
                topic_keywords=["python", "classes", "oop"],
                primary_language="python",
            ),
            self.create_sample_content_analysis(
                "https://example.com/3",
                topic_keywords=["javascript", "functions", "promises"],
                primary_language="javascript",
            ),
            self.create_sample_content_analysis(
                "https://example.com/4",
                topic_keywords=["python", "functions", "decorators"],
                primary_language="python",
            ),
        ]
        self.mock_database_manager.get_all_content_analyses = AsyncMock(
            return_value=mock_analyses
        )

        # Get trending topics
        trending = await self.api.get_trending_topics(context_id)

        # Verify trending topics
        assert isinstance(trending, list)
        assert len(trending) > 0

        # Check that most frequent topics appear first
        topic_names = [topic["topic"] for topic in trending]
        assert "python" in topic_names  # Should be most frequent language
        assert "functions" in topic_names  # Should be most frequent keyword

        # Verify topic structure
        first_topic = trending[0]
        assert "topic" in first_topic
        assert "type" in first_topic
        assert "frequency" in first_topic
        assert "trend_score" in first_topic
        assert "related_content_count" in first_topic

    def test_calculate_cache_hit_ratio(self):
        """Test cache hit ratio calculation."""
        # Initial state - no requests
        assert self.api._calculate_cache_hit_ratio() == 0.0

        # Simulate cache misses
        self.api.cache_miss_count = 3
        assert self.api._calculate_cache_hit_ratio() == 0.0

        # Add cache hits
        self.api.cache_hit_count = 2
        expected_ratio = 2 / (2 + 3)  # 2 hits out of 5 total
        assert self.api._calculate_cache_hit_ratio() == expected_ratio

    def test_calculate_strategy_effectiveness(self):
        """Test search strategy effectiveness calculation."""
        # Mock search response with mixed strategies
        search_response = SearchResponse(
            query="test query",
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result(
                    "https://example.com/1", strategy_used="semantic_search"
                ),
                self.create_sample_search_result(
                    "https://example.com/2", strategy_used="semantic_search"
                ),
                self.create_sample_search_result(
                    "https://example.com/3", strategy_used="keyword_search"
                ),
            ],
            total_results=3,
            strategies_used=["semantic_search", "keyword_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        effectiveness = self.api._calculate_strategy_effectiveness(search_response)

        # Verify effectiveness calculation
        assert "semantic_search" in effectiveness
        assert "keyword_search" in effectiveness
        assert effectiveness["semantic_search"] == 2 / 3  # 2 out of 3 results
        assert effectiveness["keyword_search"] == 1 / 3  # 1 out of 3 results

    @pytest.mark.asyncio
    async def test_generate_search_metadata(self):
        """Test search metadata generation."""
        # Mock search response
        search_response = SearchResponse(
            query="test query",
            query_analysis=Mock(),
            results=[
                self.create_sample_search_result(
                    "https://example.com/1",
                    content_type="code_example",
                    programming_language="python",
                    quality_score=0.9,
                ),
                self.create_sample_search_result(
                    "https://example.com/2",
                    content_type="tutorial",
                    programming_language="python",
                    quality_score=0.7,
                ),
                self.create_sample_search_result(
                    "https://example.com/3",
                    content_type="api_reference",
                    programming_language="javascript",
                    quality_score=0.8,
                ),
            ],
            total_results=3,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        self.mock_database_manager.get_context_stats = AsyncMock(
            return_value={"total_chunks": 10}
        )

        # Generate search metadata
        metadata = await self.api._generate_search_metadata(
            search_response, "test-context"
        )

        # Verify metadata
        assert isinstance(metadata, SearchMetadata)
        assert metadata.total_content_pieces == 3
        assert metadata.content_type_distribution["code_example"] == 1
        assert metadata.content_type_distribution["tutorial"] == 1
        assert metadata.content_type_distribution["api_reference"] == 1
        assert metadata.programming_language_distribution["python"] == 2
        assert metadata.programming_language_distribution["javascript"] == 1
        assert metadata.quality_score_distribution["high"] == 2  # scores 0.9, 0.8
        assert metadata.quality_score_distribution["medium"] == 1  # score 0.7
        assert abs(metadata.average_content_quality - (0.9 + 0.7 + 0.8) / 3) < 0.001

    @pytest.mark.asyncio
    async def test_find_related_clusters(self):
        """Test finding clusters related to search results."""
        # Mock search response
        search_response = SearchResponse(
            query="test query",
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

        # Mock clusters - one related, one unrelated
        mock_clusters = [
            TopicCluster(
                cluster_id="related_cluster",
                name="Related Cluster",
                description="Contains search result URLs",
                content_urls=["https://example.com/1", "https://example.com/other"],
                topic_keywords=["python"],
                programming_languages=["python"],
                content_types=["code_example"],
                difficulty_level="beginner",
                coherence_score=0.8,
                coverage_score=0.7,
                quality_score=0.9,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            ),
            TopicCluster(
                cluster_id="unrelated_cluster",
                name="Unrelated Cluster",
                description="Does not contain search result URLs",
                content_urls=["https://example.com/different"],
                topic_keywords=["java"],
                programming_languages=["java"],
                content_types=["tutorial"],
                difficulty_level="intermediate",
                coherence_score=0.7,
                coverage_score=0.6,
                quality_score=0.8,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            ),
        ]

        self.mock_database_manager.load_topic_clusters = AsyncMock(
            return_value=mock_clusters
        )

        # Find related clusters
        related_clusters = await self.api._find_related_clusters(
            search_response, "test-context"
        )

        # Verify only related cluster is returned
        assert len(related_clusters) == 1
        assert related_clusters[0].cluster_id == "related_cluster"

    @pytest.mark.asyncio
    async def test_generate_graph_insights(self):
        """Test knowledge graph insights generation."""
        # Mock search response
        search_response = SearchResponse(
            query="test query",
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

        # Mock relationships involving search result URLs
        mock_relationships = [
            ContentRelationship(
                source_url="https://example.com/1",
                target_url="https://example.com/2",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=0.0,
            ),
            ContentRelationship(
                source_url="https://example.com/1",
                target_url="https://example.com/other",
                relationship_type=RelationshipType.CODE_DEPENDENCY,
                strength=0.7,
                confidence=0.8,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=0.0,
            ),
        ]

        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=mock_relationships
        )

        # Generate graph insights
        insights = await self.api._generate_graph_insights(
            search_response, "test-context"
        )

        # Verify insights
        assert isinstance(insights, dict)
        assert insights["total_relationships"] == 2
        assert "semantic_similarity" in insights["relationship_type_distribution"]
        assert "code_dependency" in insights["relationship_type_distribution"]
        assert insights["relationship_type_distribution"]["semantic_similarity"] == 1
        assert insights["relationship_type_distribution"]["code_dependency"] == 1
        assert len(insights["most_connected_content"]) > 0
        assert insights["average_relationship_strength"] == (0.8 + 0.7) / 2

    @pytest.mark.asyncio
    async def test_build_content_metadata(self):
        """Test building detailed content metadata."""
        context_id = "test-context"
        url = "https://example.com/test"

        # Mock content analysis
        content_analysis = self.create_sample_content_analysis(url)

        # Mock relationships
        mock_relationships = [
            ContentRelationship(
                source_url=url,
                target_url="https://example.com/other",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=0.0,
            )
        ]
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=mock_relationships
        )

        # Mock clusters
        mock_clusters = [
            TopicCluster(
                cluster_id="cluster1",
                name="Test Cluster",
                description="Test cluster",
                content_urls=[url],
                topic_keywords=["test"],
                programming_languages=["python"],
                content_types=["code_example"],
                difficulty_level="beginner",
                coherence_score=0.8,
                coverage_score=0.7,
                quality_score=0.9,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            )
        ]
        self.mock_database_manager.load_topic_clusters = AsyncMock(
            return_value=mock_clusters
        )

        # Build content metadata
        metadata = await self.api._build_content_metadata(content_analysis, context_id)

        # Verify metadata
        assert isinstance(metadata, ContentMetadata)
        assert metadata.url == url
        assert metadata.relationship_count == 1
        assert "cluster1" in metadata.cluster_memberships
        assert metadata.programming_language == "python"
        assert metadata.content_type == "code_example"
