"""Unit tests for progressive refinement engine."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.multi_modal_search import (
    SearchEngine,
    SearchResponse,
    SearchResult,
)
from context_server.core.progressive_refinement import (
    AdaptiveSearchOrchestrator,
    ProgressiveRefinementEngine,
    RefinementSession,
    RefinementStep,
    RefinementType,
)
from context_server.core.query_analysis import QueryAnalysis, QueryType, SearchIntent


class TestProgressiveRefinementEngine:
    """Test progressive refinement functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_engine = Mock(spec=SearchEngine)
        self.refinement_engine = ProgressiveRefinementEngine(self.mock_search_engine)

    def test_is_result_satisfactory_good_results(self):
        """Test satisfaction check with good results."""
        good_response = SearchResponse(
            query="Python tutorial",
            query_analysis=Mock(),
            results=[Mock()] * 5,  # 5 results
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,  # Good quality
            search_confidence=0.9,  # High confidence
        )

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

        is_satisfactory = self.refinement_engine._is_result_satisfactory(
            good_response, analysis
        )
        assert is_satisfactory is True

    def test_is_result_satisfactory_poor_results(self):
        """Test satisfaction check with poor results."""
        poor_response = SearchResponse(
            query="vague query",
            query_analysis=Mock(),
            results=[Mock()],  # Only 1 result
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.4,  # Poor quality
            search_confidence=0.5,  # Low confidence
        )

        analysis = QueryAnalysis(
            original_query="vague query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.5,
            keywords=["vague", "query"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.2,
            specificity_score=0.3,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        is_satisfactory = self.refinement_engine._is_result_satisfactory(
            poor_response, analysis
        )
        assert is_satisfactory is False

    def test_identify_search_issues_insufficient_results(self):
        """Test identification of insufficient results issue."""
        mock_result = Mock()
        mock_result.relevance_score = 0.7
        mock_result.content_type = "general"

        response = SearchResponse(
            query="test query",
            query_analysis=Mock(),
            results=[mock_result],  # Only 1 result
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.8,
        )

        analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.8,
            keywords=["test", "query"],
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

        issues = self.refinement_engine._identify_search_issues(response, analysis)
        assert "insufficient_results" in issues

    def test_identify_search_issues_poor_relevance(self):
        """Test identification of poor relevance issue."""
        # Create mock results with low relevance scores
        mock_results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.relevance_score = 0.3  # Low relevance
            mock_results.append(mock_result)

        response = SearchResponse(
            query="test query",
            query_analysis=Mock(),
            results=mock_results,
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.4,  # Low quality
            search_confidence=0.6,
        )

        analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.6,
            keywords=["test", "query"],
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

        issues = self.refinement_engine._identify_search_issues(response, analysis)
        assert "poor_relevance" in issues
        assert "low_quality" in issues

    def test_identify_search_issues_content_type_mismatch(self):
        """Test identification of content type mismatch."""
        # Create mock results with wrong content type
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.content_type = "general"  # Wrong type for API query
            mock_result.relevance_score = 0.8
            mock_results.append(mock_result)

        response = SearchResponse(
            query="API documentation",
            query_analysis=Mock(),
            results=mock_results,
            total_results=3,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.7,
            search_confidence=0.7,
        )

        analysis = QueryAnalysis(
            original_query="API documentation",
            query_type=QueryType.API_REFERENCE,  # Expects API content
            search_intent=SearchIntent.REFERENCE,
            confidence=0.8,
            keywords=["API", "documentation"],
            code_elements=[],
            programming_language=None,
            api_references=["GET /users"],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        issues = self.refinement_engine._identify_search_issues(response, analysis)
        assert "wrong_content_type" in issues

    def test_select_refinement_strategy_insufficient_results(self):
        """Test refinement strategy selection for insufficient results."""
        mock_result = Mock()
        mock_result.relevance_score = 0.7
        mock_result.content_type = "general"

        response = SearchResponse(
            query="test query",
            query_analysis=Mock(),
            results=[mock_result],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.8,
        )

        analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.8,
            keywords=["test", "query"],
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

        # Step 1 should try query expansion
        strategy = self.refinement_engine._select_refinement_strategy(
            response, analysis, 1
        )
        assert strategy == RefinementType.QUERY_EXPANSION

        # Step 2 should try filter modification
        strategy = self.refinement_engine._select_refinement_strategy(
            response, analysis, 2
        )
        assert strategy == RefinementType.FILTER_MODIFICATION

    def test_select_refinement_strategy_poor_relevance(self):
        """Test refinement strategy selection for poor relevance."""
        # Create mock results with low relevance
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.relevance_score = 0.3
            mock_results.append(mock_result)

        response = SearchResponse(
            query="vague query",
            query_analysis=Mock(),
            results=mock_results,
            total_results=3,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.4,
            search_confidence=0.5,
        )

        # Low specificity should trigger narrowing
        analysis = QueryAnalysis(
            original_query="vague query",
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.5,
            keywords=["vague", "query"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.2,
            specificity_score=0.2,  # Low specificity
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        strategy = self.refinement_engine._select_refinement_strategy(
            response, analysis, 1
        )
        assert strategy == RefinementType.QUERY_NARROWING

    @pytest.mark.asyncio
    async def test_apply_query_expansion(self):
        """Test query expansion refinement."""
        original_query = "Python functions"

        mock_result = Mock()
        mock_result.matched_keywords = ["python", "functions"]
        mock_result.relevance_score = 0.6

        response = SearchResponse(
            query=original_query,
            query_analysis=Mock(),
            results=[mock_result],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.6,
            search_confidence=0.6,
        )

        analysis = QueryAnalysis(
            original_query=original_query,
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.LEARNING,
            confidence=0.7,
            keywords=["Python", "functions"],
            code_elements=[],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.6,
            suggested_strategies=[],
            expansion_terms=["def", "lambda", "method"],
            filters={},
        )

        (
            expanded_query,
            filters,
            strategies,
        ) = await self.refinement_engine._apply_query_expansion(
            original_query, response, analysis
        )

        assert expanded_query != original_query
        assert len(expanded_query) > len(original_query)
        assert "def" in expanded_query or "lambda" in expanded_query
        assert "semantic_search" in strategies

    @pytest.mark.asyncio
    async def test_apply_query_narrowing(self):
        """Test query narrowing refinement."""
        original_query = "web framework"

        response = SearchResponse(
            query=original_query,
            query_analysis=Mock(),
            results=[Mock()] * 10,  # Many results but poor quality
            total_results=10,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.4,
            search_confidence=0.5,
        )

        analysis = QueryAnalysis(
            original_query=original_query,
            query_type=QueryType.GENERAL,
            search_intent=SearchIntent.EXPLORATION,
            confidence=0.5,
            keywords=["web", "framework"],
            code_elements=["fetch()"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.3,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        (
            narrowed_query,
            filters,
            strategies,
        ) = await self.refinement_engine._apply_query_narrowing(
            original_query, response, analysis
        )

        assert narrowed_query != original_query
        assert "javascript" in narrowed_query
        assert "fetch()" in narrowed_query
        assert "semantic_code_search" in strategies

    @pytest.mark.asyncio
    async def test_apply_filter_modification(self):
        """Test filter modification refinement."""
        original_query = "Python tutorial"
        original_filters = {}

        mock_result = Mock()
        mock_result.relevance_score = 0.7

        response = SearchResponse(
            query=original_query,
            query_analysis=Mock(),
            results=[mock_result],  # Too few results
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.7,
            search_confidence=0.7,
        )

        analysis = QueryAnalysis(
            original_query=original_query,
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

        (
            query,
            new_filters,
            strategies,
        ) = await self.refinement_engine._apply_filter_modification(
            original_query, original_filters, response, analysis
        )

        assert query == original_query  # Query shouldn't change
        assert "primary_language" in new_filters
        assert new_filters["primary_language"] == "python"
        # content_type filter is removed when there are too few results (< 2)
        assert "content_type" not in new_filters

    def test_calculate_improvement_positive(self):
        """Test improvement calculation with positive change."""
        before_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 2,
            total_results=2,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.5,
            search_confidence=0.6,
        )

        # Mock results with relevance scores
        before_results = []
        for i in range(2):
            mock_result = Mock()
            mock_result.relevance_score = 0.5
            before_results.append(mock_result)
        before_response.results = before_results

        after_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 5,
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,  # Improved
            search_confidence=0.9,  # Improved
        )

        # Mock results with better relevance scores
        after_results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.relevance_score = 0.8
            after_results.append(mock_result)
        after_response.results = after_results

        improvement = self.refinement_engine._calculate_improvement(
            before_response, after_response
        )

        assert improvement > 0.2  # Should show significant improvement

    def test_calculate_improvement_negative(self):
        """Test improvement calculation with negative change."""
        before_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 5,
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        # Mock good initial results
        before_results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.relevance_score = 0.8
            before_results.append(mock_result)
        before_response.results = before_results

        after_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.4,  # Degraded
            search_confidence=0.5,  # Degraded
        )

        # Mock worse results
        after_results = []
        mock_result = Mock()
        mock_result.relevance_score = 0.3
        after_results.append(mock_result)
        after_response.results = after_results

        improvement = self.refinement_engine._calculate_improvement(
            before_response, after_response
        )

        assert improvement < 0  # Should show degradation

    def test_calculate_step_confidence(self):
        """Test step confidence calculation."""
        response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 5,
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.8,
        )

        analysis = QueryAnalysis(
            original_query="test",
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

        # High improvement should boost confidence
        high_improvement_confidence = self.refinement_engine._calculate_step_confidence(
            0.3, response, analysis
        )

        # Low improvement should reduce confidence
        low_improvement_confidence = self.refinement_engine._calculate_step_confidence(
            -0.1, response, analysis
        )

        assert high_improvement_confidence > low_improvement_confidence
        assert high_improvement_confidence <= 1.0
        assert low_improvement_confidence >= 0.0

    def test_generate_step_reasoning(self):
        """Test step reasoning generation."""
        before_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 2,
            total_results=2,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.5,
            search_confidence=0.6,
        )

        after_response = SearchResponse(
            query="test",
            query_analysis=Mock(),
            results=[Mock()] * 5,
            total_results=5,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        reasoning = self.refinement_engine._generate_step_reasoning(
            RefinementType.QUERY_EXPANSION, 0.25, before_response, after_response
        )

        assert "expanded query" in reasoning
        assert "significantly improved" in reasoning
        assert "from 2 to 5 results" in reasoning
        assert "0.50 to 0.80" in reasoning

    @pytest.mark.asyncio
    async def test_refine_search_no_improvement_needed(self):
        """Test refinement when initial results are already satisfactory."""
        good_response = SearchResponse(
            query="Python tutorial",
            query_analysis=Mock(),
            results=[Mock()] * 8,
            total_results=8,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.9,
            search_confidence=0.9,
        )

        analysis = QueryAnalysis(
            original_query="Python tutorial",
            query_type=QueryType.TUTORIAL,
            search_intent=SearchIntent.LEARNING,
            confidence=0.9,
            keywords=["Python", "tutorial"],
            code_elements=[],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        self.mock_search_engine.search = AsyncMock(return_value=good_response)

        with patch.object(
            self.refinement_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = analysis

            session = await self.refinement_engine.refine_search("Python tutorial")

        assert isinstance(session, RefinementSession)
        assert session.original_query == "Python tutorial"
        assert len(session.refinement_steps) == 0  # No refinement needed
        assert session.final_response == good_response
        assert session.total_improvement == 0.0


class TestAdaptiveSearchOrchestrator:
    """Test adaptive search orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_engine = Mock(spec=SearchEngine)
        self.orchestrator = AdaptiveSearchOrchestrator(self.mock_search_engine)

    @pytest.mark.asyncio
    async def test_intelligent_search(self):
        """Test intelligent search orchestration."""
        query = "Python web development"
        target_quality = 0.8

        # Mock a refinement session
        mock_session = RefinementSession(
            original_query=query,
            original_analysis=Mock(),
            refinement_steps=[],
            final_response=SearchResponse(
                query=query,
                query_analysis=Mock(),
                results=[Mock()] * 7,
                total_results=7,
                strategies_used=["semantic_search"],
                search_time_ms=150,
                refinement_suggestions=[],
                expansion_suggestions=[],
                filter_suggestions={},
                result_quality_score=0.85,  # Above target
                search_confidence=0.9,
            ),
            total_improvement=0.2,
            session_confidence=0.9,
        )

        with patch.object(
            self.orchestrator.refinement_engine, "refine_search"
        ) as mock_refine:
            mock_refine.return_value = mock_session

            session = await self.orchestrator.intelligent_search(
                query, target_quality=target_quality
            )

        assert isinstance(session, RefinementSession)
        assert session.final_response.result_quality_score >= target_quality

    @pytest.mark.asyncio
    async def test_explain_search_process(self):
        """Test search process explanation generation."""
        query = "JavaScript async programming"

        # Create a sample refinement session
        original_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_PATTERN,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["JavaScript", "async", "programming"],
            code_elements=["async"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.6,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        refinement_step = RefinementStep(
            step_number=1,
            refinement_type=RefinementType.QUERY_EXPANSION,
            original_query=query,
            refined_query=f"{query} Promise await",
            filters_applied={},
            strategies_used=["semantic_search"],
            improvement_score=0.15,
            confidence=0.85,
            reasoning="Applied query expansion which improved search results",
        )

        final_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[Mock()] * 6,
            total_results=6,
            strategies_used=["semantic_search", "semantic_code_search"],
            search_time_ms=200,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.82,
            search_confidence=0.88,
        )

        session = RefinementSession(
            original_query=query,
            original_analysis=original_analysis,
            refinement_steps=[refinement_step],
            final_response=final_response,
            total_improvement=0.15,
            session_confidence=0.88,
        )

        explanation = await self.orchestrator.explain_search_process(session)

        assert isinstance(explanation, dict)
        assert explanation["original_query"] == query
        assert "query_understanding" in explanation
        assert explanation["query_understanding"]["type"] == "code_pattern"
        assert (
            explanation["query_understanding"]["programming_language"] == "javascript"
        )
        assert "refinement_process" in explanation
        assert len(explanation["refinement_process"]) == 1
        assert explanation["refinement_process"][0]["strategy"] == "query_expansion"
        assert "final_outcome" in explanation
        assert explanation["final_outcome"]["total_steps"] == 1
        assert explanation["final_outcome"]["result_count"] == 6
