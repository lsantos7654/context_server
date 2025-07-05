"""Unit tests for multi-modal search system."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.multi_modal_search import (
    SearchEngine,
    SearchResponse,
    SearchResult,
    SearchRouter,
    SearchStrategy,
)
from context_server.core.query_analysis import QueryAnalysis, QueryType, SearchIntent


class TestSearchRouter:
    """Test search strategy routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.mock_database_manager = Mock()
        self.router = SearchRouter(
            self.mock_embedding_service, self.mock_database_manager
        )

    def test_select_strategies_for_code_function(self):
        """Test strategy selection for code function queries."""
        analysis = QueryAnalysis(
            original_query="how to use fetch() function",
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.9,
            keywords=["fetch", "function"],
            code_elements=["fetch()"],
            programming_language="javascript",
            api_references=[],
            is_question=True,
            complexity_score=0.3,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategies = self.router.select_strategies(analysis)

        assert SearchStrategy.SEMANTIC_CODE_SEARCH in strategies
        assert SearchStrategy.LANGUAGE_SPECIFIC_SEARCH in strategies
        assert len(strategies) >= 2

    def test_select_strategies_for_api_reference(self):
        """Test strategy selection for API reference queries."""
        analysis = QueryAnalysis(
            original_query="REST API GET /users documentation",
            query_type=QueryType.API_REFERENCE,
            search_intent=SearchIntent.REFERENCE,
            confidence=0.9,
            keywords=["REST", "API", "users", "documentation"],
            code_elements=[],
            programming_language=None,
            api_references=["GET /users"],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.9,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategies = self.router.select_strategies(analysis)

        assert SearchStrategy.API_SEARCH in strategies
        assert SearchStrategy.STRUCTURED_SEARCH in strategies

    def test_select_strategies_for_tutorial(self):
        """Test strategy selection for tutorial queries."""
        analysis = QueryAnalysis(
            original_query="beginner guide to React hooks",
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["beginner", "guide", "React", "hooks"],
            code_elements=[],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.5,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategies = self.router.select_strategies(analysis)

        assert SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH in strategies
        assert SearchStrategy.HIERARCHICAL_SEARCH in strategies

    def test_select_strategies_for_complex_query(self):
        """Test strategy selection for complex queries."""
        analysis = QueryAnalysis(
            original_query="implement advanced concurrent processing with error handling",
            query_type=QueryType.CODE_PATTERN,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.7,
            keywords=[
                "implement",
                "advanced",
                "concurrent",
                "processing",
                "error",
                "handling",
            ],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.9,  # High complexity
            specificity_score=0.6,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategies = self.router.select_strategies(analysis)

        assert SearchStrategy.PROGRESSIVE_REFINEMENT in strategies

    def test_select_strategies_for_vague_query(self):
        """Test strategy selection for vague queries."""
        analysis = QueryAnalysis(
            original_query="programming stuff",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.5,
            keywords=["programming", "stuff"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.2,
            specificity_score=0.3,  # Low specificity
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategies = self.router.select_strategies(analysis)

        assert SearchStrategy.MULTI_STRATEGY_FUSION in strategies


class TestSearchEngine:
    """Test main search engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.mock_database_manager = Mock()
        self.search_engine = SearchEngine(
            self.mock_embedding_service, self.mock_database_manager
        )

        # Mock embedding service methods
        self.mock_embedding_service.embed_text = AsyncMock()

        # Mock database manager methods
        self.mock_database_manager.search_similar_content = AsyncMock()
        self.mock_database_manager.search_by_keywords = AsyncMock()

    @pytest.mark.asyncio
    async def test_search_with_semantic_strategy(self):
        """Test semantic search execution."""
        query = "Python list comprehension"

        # Mock embedding response
        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.1] * 1536,
            "model": "text-embedding-3-small",
            "success": True,
        }

        # Mock database search results
        self.mock_database_manager.search_similar_content.return_value = [
            {
                "id": "1",
                "content": "List comprehensions in Python are concise way to create lists",
                "title": "Python List Comprehensions",
                "url": "https://example.com/python-lists",
                "similarity": 0.9,
                "content_type": "tutorial",
                "primary_language": "python",
                "summary": "Tutorial on Python list comprehensions",
                "quality_score": 0.8,
                "metadata": {},
            }
        ]

        with patch.object(
            self.search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.CODE_PATTERN,
                search_intent=SearchIntent.LEARNING,
                confidence=0.8,
                keywords=["Python", "list", "comprehension"],
                code_elements=[],
                programming_language="python",
                api_references=[],
                is_question=False,
                complexity_score=0.4,
                specificity_score=0.7,
                suggested_strategies=[],
                expansion_terms=[],
                filters={"primary_language": "python"},
            )

            response = await self.search_engine.search(query, limit=10)

        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        assert response.query == query
        assert len(response.strategies_used) > 0

        # Check first result
        result = response.results[0]
        assert isinstance(result, SearchResult)
        assert (
            result.content
            == "List comprehensions in Python are concise way to create lists"
        )
        assert result.programming_language == "python"

    @pytest.mark.asyncio
    async def test_search_with_api_strategy(self):
        """Test API search execution."""
        query = "REST API documentation GET /users"

        # Mock embedding response
        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.2] * 1536,
            "model": "text-embedding-3-small",
            "success": True,
        }

        # Mock database search results for API content
        self.mock_database_manager.search_similar_content.return_value = [
            {
                "id": "2",
                "content": "GET /users endpoint returns list of all users",
                "title": "Users API Reference",
                "url": "https://api.example.com/docs/users",
                "similarity": 0.95,
                "content_type": "api_reference",
                "primary_language": None,
                "summary": "API documentation for user endpoints",
                "quality_score": 0.9,
                "metadata": {},
            }
        ]

        with patch.object(
            self.search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.API_REFERENCE,
                search_intent=SearchIntent.REFERENCE,
                confidence=0.9,
                keywords=["REST", "API", "documentation", "users"],
                code_elements=[],
                programming_language=None,
                api_references=["GET /users"],
                is_question=False,
                complexity_score=0.3,
                specificity_score=0.9,
                suggested_strategies=[],
                expansion_terms=[],
                filters={"content_type": "api_reference"},
            )

            response = await self.search_engine.search(query, limit=10)

        assert len(response.results) > 0

        # Should use API search strategy
        assert (
            "api_search" in response.strategies_used
            or "semantic_search" in response.strategies_used
        )

        result = response.results[0]
        assert result.content_type == "api_reference"
        assert "GET /users" in result.content

    @pytest.mark.asyncio
    async def test_semantic_code_search(self):
        """Test code-specific semantic search."""
        query = "async function JavaScript"

        # Mock code-specific embedding
        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.3] * 4096,
            "model": "embed-code-v3.0",
            "success": True,
        }

        self.mock_database_manager.search_similar_content.return_value = [
            {
                "id": "3",
                "content": "async function fetchData() { return await fetch('/api/data'); }",
                "title": "JavaScript Async Functions",
                "url": "https://example.com/js-async",
                "similarity": 0.85,
                "content_type": "code_example",
                "primary_language": "javascript",
                "summary": "Examples of async functions in JavaScript",
                "quality_score": 0.8,
                "metadata": {},
            }
        ]

        analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.8,
            keywords=["async", "function", "JavaScript"],
            code_elements=["async"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={"primary_language": "javascript"},
        )

        results = await self.search_engine._semantic_code_search(
            query, analysis, 10, {}
        )

        assert len(results) > 0
        result = results[0]
        assert result.strategy_used == "semantic_code_search"
        assert result.programming_language == "javascript"
        assert "async" in result.content

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid semantic + keyword search."""
        query = "Python web framework tutorial"

        # Mock semantic search results
        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.1] * 1536,
            "model": "text-embedding-3-small",
            "success": True,
        }

        self.mock_database_manager.search_similar_content.return_value = [
            {
                "id": "4",
                "content": "Django is a popular Python web framework",
                "title": "Django Tutorial",
                "url": "https://example.com/django",
                "similarity": 0.8,
                "content_type": "tutorial",
                "primary_language": "python",
                "summary": "Introduction to Django web framework",
                "quality_score": 0.8,
                "metadata": {},
            }
        ]

        # Mock keyword search results
        self.mock_database_manager.search_by_keywords.return_value = [
            {
                "id": "5",
                "content": "Flask is a lightweight web framework for Python",
                "title": "Flask Framework",
                "url": "https://example.com/flask",
                "similarity": 0.7,
                "content_type": "tutorial",
                "primary_language": "python",
                "summary": "Flask web framework basics",
                "quality_score": 0.7,
                "metadata": {},
            }
        ]

        analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["Python", "web", "framework", "tutorial"],
            code_elements=[],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.6,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        results = await self.search_engine._hybrid_search(query, analysis, 10, {})

        assert len(results) >= 2  # Should combine semantic + keyword results

        # Check that results are marked as hybrid
        for result in results:
            assert result.strategy_used == "hybrid_search"

    @pytest.mark.asyncio
    async def test_tutorial_prioritized_search(self):
        """Test tutorial-prioritized search for learning queries."""
        query = "beginner guide to machine learning"

        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.2] * 1536,
            "model": "text-embedding-3-small",
            "success": True,
        }

        self.mock_database_manager.search_similar_content.return_value = [
            {
                "id": "6",
                "content": "A complete beginner's guide to machine learning concepts",
                "title": "ML for Beginners",
                "url": "https://example.com/ml-guide",
                "similarity": 0.9,
                "content_type": "tutorial",
                "primary_language": "python",
                "summary": "Beginner-friendly machine learning tutorial",
                "quality_score": 0.9,
                "metadata": {},
            }
        ]

        analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["beginner", "guide", "machine", "learning"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.6,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        results = await self.search_engine._tutorial_prioritized_search(
            query, analysis, 10, {}
        )

        assert len(results) > 0
        result = results[0]
        assert result.strategy_used == "tutorial_prioritized_search"
        assert result.content_type == "tutorial"
        # Should have boosted relevance score for tutorial content
        assert result.relevance_score > 0.9

    def test_deduplicate_results(self):
        """Test result deduplication by URL."""
        results = [
            SearchResult(
                url="https://example.com/page1",
                title="Page 1",
                content="Content 1",
                similarity_score=0.9,
                relevance_score=0.9,
                content_type="tutorial",
                programming_language="python",
                summary="Summary 1",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.8,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/page1",  # Duplicate URL
                title="Page 1 (duplicate)",
                content="Content 1 duplicate",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="tutorial",
                programming_language="python",
                summary="Summary 1",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.7,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/page2",
                title="Page 2",
                content="Content 2",
                similarity_score=0.7,
                relevance_score=0.7,
                content_type="tutorial",
                programming_language="python",
                summary="Summary 2",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.8,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
        ]

        unique_results = self.search_engine._deduplicate_results(results)

        assert len(unique_results) == 2
        assert unique_results[0].url == "https://example.com/page1"
        assert unique_results[1].url == "https://example.com/page2"
        # Should keep the higher-scoring version
        assert unique_results[0].relevance_score == 0.9

    def test_rank_results(self):
        """Test result ranking with multi-factor scoring."""
        results = [
            SearchResult(
                url="https://example.com/low-quality",
                title="Low Quality",
                content="Basic content",
                similarity_score=0.6,
                relevance_score=0.6,
                content_type="general",
                programming_language=None,
                summary="Basic summary",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.3,  # Low quality
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/high-quality",
                title="High Quality",
                content="Comprehensive Python tutorial content",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="tutorial",
                programming_language="python",
                summary="High quality summary",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.9,  # High quality
                matched_keywords=["Python", "tutorial"],
                code_elements=[],
                api_references=[],
            ),
        ]

        analysis = QueryAnalysis(
            original_query="Python tutorial",
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["Python", "tutorial"],
            code_elements=[],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        ranked_results = self.search_engine._rank_results(results, analysis)

        # High quality result should be ranked first
        assert ranked_results[0].url == "https://example.com/high-quality"
        assert ranked_results[0].relevance_score > ranked_results[1].relevance_score

    def test_content_type_alignment_boost(self):
        """Test content type alignment scoring."""
        # Tutorial query should boost tutorial content
        assert self.search_engine._content_type_matches_query(
            "tutorial", QueryType.TUTORIAL
        )
        assert self.search_engine._content_type_matches_query(
            "api_reference", QueryType.API_REFERENCE
        )
        assert self.search_engine._content_type_matches_query(
            "code_example", QueryType.CODE_FUNCTION
        )

        # Misaligned types should not match
        assert not self.search_engine._content_type_matches_query(
            "general", QueryType.CODE_FUNCTION
        )

    def test_calculate_result_quality(self):
        """Test overall result quality calculation."""
        good_results = [
            SearchResult(
                url="https://example.com/1",
                title="Good Result 1",
                content="High quality content",
                similarity_score=0.9,
                relevance_score=0.9,
                content_type="tutorial",
                programming_language="python",
                summary="Good summary",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.9,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/2",
                title="Good Result 2",
                content="High quality content",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="api_reference",
                programming_language="javascript",
                summary="Good summary",
                strategy_used="semantic_search",
                embedding_model="model1",
                quality_score=0.8,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
        ]

        analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.8,
            keywords=["test"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.5,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        quality_score = self.search_engine._calculate_result_quality(
            good_results, analysis
        )

        assert quality_score > 0.8
        assert quality_score <= 1.0

    def test_calculate_search_confidence(self):
        """Test search confidence calculation."""
        analysis = QueryAnalysis(
            original_query="Python tutorial",
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["Python", "tutorial"],
            code_elements=[],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        # Test with good result count and multiple strategies
        high_confidence = self.search_engine._calculate_search_confidence(
            analysis,
            result_count=10,
            strategies_used=["semantic_search", "tutorial_prioritized_search"],
        )

        # Test with poor result count and single strategy
        low_confidence = self.search_engine._calculate_search_confidence(
            analysis, result_count=1, strategies_used=["semantic_search"]
        )

        assert high_confidence > low_confidence
        assert high_confidence <= 1.0
        assert low_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_search_with_empty_results(self):
        """Test search handling when no results are found."""
        query = "very specific nonexistent query"

        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.1] * 1536,
            "model": "text-embedding-3-small",
            "success": True,
        }

        # Return empty results
        self.mock_database_manager.search_similar_content.return_value = []
        self.mock_database_manager.search_by_keywords.return_value = []

        with patch.object(
            self.search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.GENERAL,
                search_intent=SearchIntent.EXPLORATION,
                confidence=0.5,
                keywords=["very", "specific", "nonexistent", "query"],
                code_elements=[],
                programming_language=None,
                api_references=[],
                is_question=False,
                complexity_score=0.4,
                specificity_score=0.8,
                suggested_strategies=[],
                expansion_terms=[],
                filters={},
            )

            response = await self.search_engine.search(query, limit=10)

        assert len(response.results) == 0
        assert response.total_results == 0
        assert len(response.refinement_suggestions) > 0  # Should suggest improvements
