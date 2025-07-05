"""Unit tests for semantic code search and API/function search functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.code_search import (
    APIFunctionSearchEngine,
    CodePattern,
    CodePatternAnalyzer,
    CodeSearchType,
    SemanticCodeSearchEngine,
)
from context_server.core.multi_modal_search import (
    SearchEngine,
    SearchResponse,
    SearchResult,
)
from context_server.core.query_analysis import QueryAnalysis, QueryType, SearchIntent


class TestCodePatternAnalyzer:
    """Test code pattern analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CodePatternAnalyzer()

    def test_analyze_function_query(self):
        """Test function pattern detection."""
        query = "how to define a Python function"

        # Mock query analysis
        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["define", "Python", "function"],
            code_elements=["def"],
            programming_language="python",
            api_references=[],
            is_question=True,
            complexity_score=0.3,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.FUNCTION_SIGNATURE
        assert pattern.confidence >= 0.5  # 0.3 base + 0.2 code elements boost
        assert pattern.programming_language == "python"
        assert len(pattern.matched_elements) > 0

    def test_analyze_class_query(self):
        """Test class pattern detection."""
        query = "JavaScript class inheritance example"

        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_CLASS,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["JavaScript", "class", "inheritance", "example"],
            code_elements=["class"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.CLASS_DEFINITION
        assert pattern.confidence >= 0.5  # 0.3 base + 0.2 code elements boost
        assert pattern.programming_language == "javascript"

    def test_analyze_api_query(self):
        """Test API pattern detection."""
        query = "REST API GET /users endpoint documentation"

        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.API_REFERENCE,
            search_intent=SearchIntent.REFERENCE,
            confidence=0.9,
            keywords=["REST", "API", "users", "endpoint", "documentation"],
            code_elements=[],
            programming_language=None,
            api_references=["GET /users"],
            is_question=False,
            complexity_score=0.3,
            specificity_score=0.9,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.API_ENDPOINT
        assert pattern.confidence >= 0.4
        assert "GET" in pattern.matched_elements or "/users" in pattern.matched_elements

    def test_analyze_error_handling_query(self):
        """Test error handling pattern detection."""
        query = "Python try catch exception handling"

        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_PATTERN,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.8,
            keywords=["Python", "try", "catch", "exception", "handling"],
            code_elements=["try", "except"],
            programming_language="python",
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.7,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.ERROR_HANDLING
        assert pattern.confidence > 0.2
        assert pattern.programming_language == "python"

    def test_analyze_algorithm_query(self):
        """Test algorithm pattern detection."""
        query = "binary search algorithm implementation"

        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_PATTERN,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.8,
            keywords=["binary", "search", "algorithm", "implementation"],
            code_elements=[],
            programming_language=None,
            api_references=[],
            is_question=False,
            complexity_score=0.6,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.ALGORITHM_IMPLEMENTATION
        assert pattern.confidence >= 0.25
        assert (
            "binary" in pattern.matched_elements or "search" in pattern.matched_elements
        )

    def test_analyze_design_pattern_query(self):
        """Test design pattern detection."""
        query = "singleton design pattern Java example"

        query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.CODE_PATTERN,
            search_intent=SearchIntent.LEARNING,
            confidence=0.8,
            keywords=["singleton", "design", "pattern", "Java", "example"],
            code_elements=[],
            programming_language="java",
            api_references=[],
            is_question=False,
            complexity_score=0.5,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        pattern = self.analyzer.analyze_code_query(query, query_analysis)

        assert pattern.pattern_type == CodeSearchType.DESIGN_PATTERN
        assert pattern.confidence >= 0.35
        assert pattern.programming_language == "java"

    def test_detect_frameworks(self):
        """Test framework detection in queries."""
        react_query = "React useState hook example"
        django_query = "Django models and views tutorial"
        express_query = "Express.js middleware setup"

        react_frameworks = self.analyzer._detect_frameworks(react_query)
        django_frameworks = self.analyzer._detect_frameworks(django_query)
        express_frameworks = self.analyzer._detect_frameworks(express_query)

        assert "react" in react_frameworks
        assert "django" in django_frameworks
        assert "express" in express_frameworks

    def test_extract_function_elements(self):
        """Test function element extraction."""
        query = "def calculate_sum(a, b) async function example"

        elements = self.analyzer._extract_function_elements(query)

        assert "calculate_sum" in elements or "def" in elements

    def test_extract_api_elements(self):
        """Test API element extraction."""
        query = "GET /api/users/{id} POST /api/posts"

        elements = self.analyzer._extract_api_elements(query)

        assert any("GET" in str(elem) or "/api/users" in str(elem) for elem in elements)

    def test_calculate_code_complexity(self):
        """Test code complexity calculation."""
        simple_query = "Python print function"
        complex_query = "advanced distributed concurrent algorithm optimization"

        simple_complexity = self.analyzer._calculate_code_complexity(
            simple_query, ["print"]
        )
        complex_complexity = self.analyzer._calculate_code_complexity(
            complex_query, ["algorithm", "optimization"]
        )

        assert complex_complexity > simple_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0

    def test_is_design_pattern_query(self):
        """Test design pattern query detection."""
        pattern_query = "observer pattern implementation"
        non_pattern_query = "simple function example"

        assert self.analyzer._is_design_pattern_query(pattern_query) is True
        assert self.analyzer._is_design_pattern_query(non_pattern_query) is False

    def test_is_library_usage_query(self):
        """Test library usage query detection."""
        usage_query = "how to use pandas DataFrame"
        non_usage_query = "mathematical algorithm"

        assert self.analyzer._is_library_usage_query(usage_query) is True
        assert self.analyzer._is_library_usage_query(non_usage_query) is False


class TestSemanticCodeSearchEngine:
    """Test semantic code search engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_engine = Mock(spec=SearchEngine)
        self.code_search_engine = SemanticCodeSearchEngine(self.mock_base_engine)

    @pytest.mark.asyncio
    async def test_semantic_code_search_function(self):
        """Test semantic code search for function queries."""
        query = "Python async function example"

        # Mock base search engine response
        mock_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[
                SearchResult(
                    url="https://example.com/async-func",
                    title="Async Functions in Python",
                    content="async def fetch_data(): return await request.get(url)",
                    similarity_score=0.9,
                    relevance_score=0.9,
                    content_type="code_example",
                    programming_language="python",
                    summary="Python async function example",
                    strategy_used="semantic_search",
                    embedding_model="test-model",
                    quality_score=0.8,
                    matched_keywords=["async", "python"],
                    code_elements=["async", "def"],
                    api_references=[],
                )
            ],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=100,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        self.mock_base_engine.search = AsyncMock(return_value=mock_response)

        with patch.object(
            self.code_search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.CODE_FUNCTION,
                search_intent=SearchIntent.LEARNING,
                confidence=0.8,
                keywords=["Python", "async", "function", "example"],
                code_elements=["async", "def"],
                programming_language="python",
                api_references=[],
                is_question=False,
                complexity_score=0.4,
                specificity_score=0.7,
                suggested_strategies=[],
                expansion_terms=[],
                filters={},
            )

            response = await self.code_search_engine.semantic_code_search(
                query, limit=10
            )

        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        assert response.results[0].content_type == "code_example"
        assert response.results[0].programming_language == "python"

    @pytest.mark.asyncio
    async def test_semantic_code_search_api(self):
        """Test semantic code search for API queries."""
        query = "REST API GET /users endpoint"

        mock_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[
                SearchResult(
                    url="https://api.example.com/docs",
                    title="Users API Documentation",
                    content="GET /users - Returns list of all users",
                    similarity_score=0.95,
                    relevance_score=0.95,
                    content_type="api_reference",
                    programming_language=None,
                    summary="API documentation for user endpoints",
                    strategy_used="api_search",
                    embedding_model="test-model",
                    quality_score=0.9,
                    matched_keywords=["GET", "users", "API"],
                    code_elements=[],
                    api_references=["GET /users"],
                )
            ],
            total_results=1,
            strategies_used=["api_search"],
            search_time_ms=80,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.9,
            search_confidence=0.95,
        )

        self.mock_base_engine.search = AsyncMock(return_value=mock_response)

        with patch.object(
            self.code_search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.API_REFERENCE,
                search_intent=SearchIntent.REFERENCE,
                confidence=0.9,
                keywords=["REST", "API", "GET", "users", "endpoint"],
                code_elements=[],
                programming_language=None,
                api_references=["GET /users"],
                is_question=False,
                complexity_score=0.3,
                specificity_score=0.9,
                suggested_strategies=[],
                expansion_terms=[],
                filters={},
            )

            response = await self.code_search_engine.semantic_code_search(
                query, limit=10
            )

        assert len(response.results) > 0
        assert response.results[0].content_type == "api_reference"
        assert "GET" in response.results[0].content

    def test_build_code_filters(self):
        """Test code filter building."""
        base_filters = {"some_filter": "value"}

        code_pattern = CodePattern(
            pattern_type=CodeSearchType.FUNCTION_SIGNATURE,
            confidence=0.8,
            matched_elements=["function", "async"],
            programming_language="javascript",
            framework_hints=["react"],
            complexity_score=0.4,
        )

        query_analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.IMPLEMENTATION,
            confidence=0.8,
            keywords=["test"],
            code_elements=["function"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.6,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        filters = self.code_search_engine._build_code_filters(
            base_filters, code_pattern, query_analysis
        )

        assert filters["min_code_percentage"] == 15
        assert filters["primary_language"] == "javascript"
        assert filters["content_type"] == "code_example"
        assert filters["framework"] == "react"
        assert filters["some_filter"] == "value"  # Original filter preserved

    def test_enhance_function_query(self):
        """Test function query enhancement."""
        query = "calculate sum"

        code_pattern = CodePattern(
            pattern_type=CodeSearchType.FUNCTION_SIGNATURE,
            confidence=0.8,
            matched_elements=["calculate"],
            programming_language="python",
            framework_hints=[],
            complexity_score=0.3,
        )

        enhanced = self.code_search_engine._enhance_function_query(query, code_pattern)

        assert "python" in enhanced.lower()
        assert "function" in enhanced.lower()

    def test_enhance_api_query(self):
        """Test API query enhancement."""
        query = "user endpoint documentation"

        code_pattern = CodePattern(
            pattern_type=CodeSearchType.API_ENDPOINT,
            confidence=0.8,
            matched_elements=["endpoint"],
            programming_language=None,
            framework_hints=[],
            complexity_score=0.3,
        )

        enhanced = self.code_search_engine._enhance_api_query(query, code_pattern)

        assert "api" in enhanced.lower()

    def test_enhance_function_results(self):
        """Test function result enhancement."""
        results = [
            SearchResult(
                url="https://example.com/1",
                title="Function Example",
                content="def calculate_sum(a, b): return a + b",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="code_example",
                programming_language="python",
                summary="Function example",
                strategy_used="semantic_search",
                embedding_model="test-model",
                quality_score=0.8,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/2",
                title="General Info",
                content="Some general information about functions",
                similarity_score=0.7,
                relevance_score=0.7,
                content_type="general",
                programming_language=None,
                summary="General info",
                strategy_used="semantic_search",
                embedding_model="test-model",
                quality_score=0.6,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
        ]

        code_pattern = CodePattern(
            pattern_type=CodeSearchType.FUNCTION_SIGNATURE,
            confidence=0.8,
            matched_elements=["calculate_sum"],
            programming_language="python",
            framework_hints=[],
            complexity_score=0.3,
        )

        enhanced_results = self.code_search_engine._enhance_function_results(
            results, code_pattern
        )

        # Code example with matching element should be ranked higher
        assert enhanced_results[0].content_type == "code_example"
        assert enhanced_results[0].relevance_score > enhanced_results[1].relevance_score


class TestAPIFunctionSearchEngine:
    """Test API and function search engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_engine = Mock(spec=SearchEngine)
        self.api_search_engine = APIFunctionSearchEngine(self.mock_base_engine)

    @pytest.mark.asyncio
    async def test_search_api_endpoints(self):
        """Test API endpoint search."""
        query = "GET /api/users/{id} endpoint"

        mock_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[
                SearchResult(
                    url="https://api.example.com/docs/users",
                    title="Users API",
                    content="GET /api/users/{id} - Retrieve user by ID",
                    similarity_score=0.95,
                    relevance_score=0.95,
                    content_type="api_reference",
                    programming_language=None,
                    summary="User API documentation",
                    strategy_used="api_search",
                    embedding_model="test-model",
                    quality_score=0.9,
                    matched_keywords=["GET", "users", "id"],
                    code_elements=[],
                    api_references=["GET /api/users/{id}"],
                )
            ],
            total_results=1,
            strategies_used=["api_search"],
            search_time_ms=90,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.9,
            search_confidence=0.95,
        )

        self.mock_base_engine.search = AsyncMock(return_value=mock_response)

        with patch.object(
            self.api_search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.API_REFERENCE,
                search_intent=SearchIntent.REFERENCE,
                confidence=0.9,
                keywords=["GET", "api", "users", "id", "endpoint"],
                code_elements=[],
                programming_language=None,
                api_references=["GET /api/users/{id}"],
                is_question=False,
                complexity_score=0.3,
                specificity_score=0.9,
                suggested_strategies=[],
                expansion_terms=[],
                filters={},
            )

            response = await self.api_search_engine.search_api_endpoints(
                query, limit=10
            )

        assert len(response.results) > 0
        assert response.results[0].content_type == "api_reference"
        assert "GET" in response.results[0].content
        assert "/api/users" in response.results[0].content

    @pytest.mark.asyncio
    async def test_search_function_documentation(self):
        """Test function documentation search."""
        query = "async function fetchData() documentation"

        mock_response = SearchResponse(
            query=query,
            query_analysis=Mock(),
            results=[
                SearchResult(
                    url="https://docs.example.com/async",
                    title="Async Functions",
                    content="async function fetchData() { return await fetch(url); }",
                    similarity_score=0.9,
                    relevance_score=0.9,
                    content_type="code_example",
                    programming_language="javascript",
                    summary="Async function documentation",
                    strategy_used="semantic_search",
                    embedding_model="test-model",
                    quality_score=0.8,
                    matched_keywords=["async", "function", "fetchData"],
                    code_elements=["async", "function"],
                    api_references=[],
                )
            ],
            total_results=1,
            strategies_used=["semantic_search"],
            search_time_ms=120,
            refinement_suggestions=[],
            expansion_suggestions=[],
            filter_suggestions={},
            result_quality_score=0.8,
            search_confidence=0.9,
        )

        self.mock_base_engine.search = AsyncMock(return_value=mock_response)

        with patch.object(
            self.api_search_engine.query_analyzer, "analyze_query"
        ) as mock_analyzer:
            mock_analyzer.return_value = QueryAnalysis(
                original_query=query,
                query_type=QueryType.CODE_FUNCTION,
                search_intent=SearchIntent.REFERENCE,
                confidence=0.8,
                keywords=["async", "function", "fetchData", "documentation"],
                code_elements=["async", "function"],
                programming_language="javascript",
                api_references=[],
                is_question=False,
                complexity_score=0.4,
                specificity_score=0.8,
                suggested_strategies=[],
                expansion_terms=[],
                filters={},
            )

            response = await self.api_search_engine.search_function_documentation(
                query, limit=10
            )

        assert len(response.results) > 0
        assert response.results[0].content_type == "code_example"
        assert response.results[0].programming_language == "javascript"
        assert "async" in response.results[0].content

    def test_extract_api_info(self):
        """Test API information extraction."""
        query = "GET /api/users POST /api/posts/{id}"

        api_info = self.api_search_engine._extract_api_info(query)

        assert "GET" in api_info["methods"]
        assert "POST" in api_info["methods"]
        assert any("/api/users" in path for path in api_info["paths"])
        assert any("/api/posts" in path for path in api_info["paths"])
        assert api_info["has_endpoint"] is True

    def test_extract_function_info(self):
        """Test function information extraction."""
        query = "async function calculateSum(a, b) method documentation"

        function_info = self.api_search_engine._extract_function_info(query)

        assert "calculateSum" in function_info["names"]
        assert "async" in function_info["modifiers"]
        assert function_info["has_function"] is True

    def test_build_api_filters(self):
        """Test API filter building."""
        base_filters = {"base_filter": "value"}
        api_info = {
            "methods": ["GET", "POST"],
            "paths": ["/api/users"],
            "parameters": ["id"],
            "has_endpoint": True,
        }

        filters = self.api_search_engine._build_api_filters(base_filters, api_info)

        assert filters["content_type"] == "api_reference"
        assert filters["http_method"] == "GET"  # First method
        assert filters["base_filter"] == "value"  # Original filter preserved

    def test_build_function_filters(self):
        """Test function filter building."""
        base_filters = {"base_filter": "value"}
        function_info = {
            "names": ["fetchData"],
            "signatures": [],
            "modifiers": ["async"],
            "has_function": True,
        }

        query_analysis = QueryAnalysis(
            original_query="test query",
            query_type=QueryType.CODE_FUNCTION,
            search_intent=SearchIntent.REFERENCE,
            confidence=0.8,
            keywords=["test"],
            code_elements=["function"],
            programming_language="javascript",
            api_references=[],
            is_question=False,
            complexity_score=0.4,
            specificity_score=0.8,
            suggested_strategies=[],
            expansion_terms=[],
            filters={},
        )

        filters = self.api_search_engine._build_function_filters(
            base_filters, function_info, query_analysis
        )

        assert filters["content_type"] == "code_example"
        assert filters["primary_language"] == "javascript"
        assert filters["min_code_percentage"] == 10
        assert filters["base_filter"] == "value"  # Original filter preserved

    def test_enhance_api_endpoint_results(self):
        """Test API endpoint result enhancement."""
        results = [
            SearchResult(
                url="https://api.example.com/users",
                title="Users API",
                content="GET /api/users - List all users",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="api_reference",
                programming_language=None,
                summary="Users API",
                strategy_used="api_search",
                embedding_model="test-model",
                quality_score=0.9,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/general",
                title="General Info",
                content="Some information about APIs",
                similarity_score=0.7,
                relevance_score=0.7,
                content_type="general",
                programming_language=None,
                summary="General info",
                strategy_used="semantic_search",
                embedding_model="test-model",
                quality_score=0.6,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
        ]

        api_info = {
            "methods": ["GET"],
            "paths": ["/api/users"],
            "parameters": [],
            "has_endpoint": True,
        }

        enhanced_results = self.api_search_engine._enhance_api_endpoint_results(
            results, api_info
        )

        # API reference with matching method should be ranked higher
        assert enhanced_results[0].content_type == "api_reference"
        assert enhanced_results[0].relevance_score > enhanced_results[1].relevance_score

    def test_enhance_function_documentation_results(self):
        """Test function documentation result enhancement."""
        results = [
            SearchResult(
                url="https://docs.example.com/calc",
                title="Calculate Function",
                content="function calculateSum(a, b) { return a + b; }",
                similarity_score=0.8,
                relevance_score=0.8,
                content_type="code_example",
                programming_language="javascript",
                summary="Function documentation",
                strategy_used="semantic_search",
                embedding_model="test-model",
                quality_score=0.8,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
            SearchResult(
                url="https://example.com/general",
                title="General Info",
                content="General function information",
                similarity_score=0.7,
                relevance_score=0.7,
                content_type="general",
                programming_language=None,
                summary="General info",
                strategy_used="semantic_search",
                embedding_model="test-model",
                quality_score=0.6,
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            ),
        ]

        function_info = {
            "names": ["calculateSum"],
            "signatures": [],
            "modifiers": [],
            "has_function": True,
        }

        enhanced_results = (
            self.api_search_engine._enhance_function_documentation_results(
                results, function_info
            )
        )

        # Code example with matching function name should be ranked higher
        assert enhanced_results[0].content_type == "code_example"
        assert enhanced_results[0].relevance_score > enhanced_results[1].relevance_score
