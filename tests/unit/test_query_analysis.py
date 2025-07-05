"""Unit tests for query analysis and classification."""

import pytest

from context_server.core.query_analysis import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryType,
    SearchIntent,
)


class TestQueryAnalyzer:
    """Test query analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_analyze_code_function_query(self):
        """Test analysis of code function queries."""
        query = "how to use the fetch() function in JavaScript"

        analysis = self.analyzer.analyze_query(query)

        assert isinstance(analysis, QueryAnalysis)
        assert analysis.query_type == QueryType.CODE_FUNCTION
        assert analysis.programming_language == "javascript"
        assert "fetch()" in analysis.code_elements
        assert analysis.confidence > 0.7

    def test_analyze_api_reference_query(self):
        """Test analysis of API reference queries."""
        query = "REST API documentation GET /users endpoint"

        analysis = self.analyzer.analyze_query(query)

        assert analysis.query_type == QueryType.API_REFERENCE
        assert len(analysis.api_references) > 0
        assert analysis.search_intent == SearchIntent.REFERENCE

    def test_analyze_tutorial_query(self):
        """Test analysis of tutorial queries."""
        query = "step by step tutorial how to build a React app"

        analysis = self.analyzer.analyze_query(query)

        assert analysis.query_type == QueryType.TUTORIAL
        assert analysis.search_intent == SearchIntent.LEARNING
        assert analysis.is_question is False
        assert "tutorial" in analysis.keywords

    def test_analyze_troubleshooting_query(self):
        """Test analysis of troubleshooting queries."""
        query = "fix TypeError: cannot read property of undefined in JavaScript"

        analysis = self.analyzer.analyze_query(query)

        assert analysis.query_type == QueryType.TROUBLESHOOTING
        assert analysis.search_intent == SearchIntent.DEBUGGING
        assert analysis.programming_language == "javascript"
        assert "fix" in analysis.keywords

    def test_analyze_class_query(self):
        """Test analysis of class-specific queries."""
        query = "Python class inheritance example with super()"

        analysis = self.analyzer.analyze_query(query)

        assert analysis.query_type == QueryType.CODE_CLASS
        assert analysis.programming_language == "python"
        assert "class" in analysis.keywords
        assert "super()" in analysis.code_elements

    def test_analyze_conceptual_query(self):
        """Test analysis of conceptual queries."""
        query = "what is dependency injection and why use it?"

        analysis = self.analyzer.analyze_query(query)

        assert analysis.query_type == QueryType.CONCEPTUAL
        assert analysis.search_intent == SearchIntent.LEARNING
        assert analysis.is_question is True
        assert "dependency" in analysis.keywords

    def test_extract_keywords(self):
        """Test keyword extraction."""
        query = "how to implement authentication in Django REST framework"

        keywords = self.analyzer._extract_keywords(query)

        assert "implement" in keywords
        assert "authentication" in keywords
        assert "django" in keywords
        assert "rest" in keywords
        assert "framework" in keywords
        # Stop words should be filtered out
        assert "how" not in keywords
        assert "to" not in keywords
        assert "in" not in keywords

    def test_extract_code_elements(self):
        """Test code element extraction."""
        query = "use fetch() method and async/await pattern"

        code_elements = self.analyzer._extract_code_elements(query)

        assert "fetch()" in code_elements
        assert "async" in code_elements

    def test_detect_programming_language(self):
        """Test programming language detection."""
        test_cases = [
            ("Python function def hello():", "python"),
            ("JavaScript const data = fetch()", "javascript"),
            ("Java public class Main", "java"),
            ("TypeScript interface User", "typescript"),
            ("golang func main()", "go"),
            ("Rust fn main()", "rust"),
            ("general programming theory", None),
        ]

        for query, expected_lang in test_cases:
            detected_lang = self.analyzer._detect_programming_language(query)
            assert detected_lang == expected_lang

    def test_extract_api_references(self):
        """Test API reference extraction."""
        query = "GET /api/users POST /users REST API endpoints"

        api_refs = self.analyzer._extract_api_references(query)

        assert any("GET" in ref for ref in api_refs)
        assert any("POST" in ref for ref in api_refs)
        assert len(api_refs) > 0

    def test_complexity_score_calculation(self):
        """Test query complexity scoring."""
        simple_query = "Python list"
        complex_query = "implement asynchronous concurrent processing algorithm with error handling using asyncio and multiprocessing"

        simple_analysis = self.analyzer.analyze_query(simple_query)
        complex_analysis = self.analyzer.analyze_query(complex_query)

        assert simple_analysis.complexity_score < complex_analysis.complexity_score
        assert complex_analysis.complexity_score > 0.5

    def test_specificity_score_calculation(self):
        """Test query specificity scoring."""
        vague_query = "how to do something with stuff"
        specific_query = "Array.prototype.map() function with arrow functions in ES6"

        vague_analysis = self.analyzer.analyze_query(vague_query)
        specific_analysis = self.analyzer.analyze_query(specific_query)

        assert vague_analysis.specificity_score < specific_analysis.specificity_score
        assert specific_analysis.specificity_score > 0.5

    def test_search_intent_detection(self):
        """Test search intent classification."""
        test_cases = [
            ("learn React hooks tutorial", SearchIntent.LEARNING),
            ("implement user authentication system", SearchIntent.IMPLEMENTATION),
            ("debug memory leak error", SearchIntent.DEBUGGING),
            ("API documentation reference", SearchIntent.REFERENCE),
            ("compare different frameworks", SearchIntent.EXPLORATION),
        ]

        for query, expected_intent in test_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.search_intent == expected_intent

    def test_question_detection(self):
        """Test question detection."""
        questions = [
            "What is React?",
            "How to implement authentication?",
            "Why use TypeScript?",
            "Which framework is better?",
            "Where to find documentation?",
        ]

        statements = [
            "React tutorial",
            "Implement authentication",
            "TypeScript benefits",
            "Framework comparison",
            "Documentation link",
        ]

        for question in questions:
            analysis = self.analyzer.analyze_query(question)
            assert analysis.is_question is True

        for statement in statements:
            analysis = self.analyzer.analyze_query(statement)
            assert analysis.is_question is False

    def test_suggested_strategies(self):
        """Test search strategy suggestions."""
        code_query = "Python class method decorator example"
        api_query = "REST API GET endpoint documentation"
        tutorial_query = "beginner guide to machine learning"

        code_analysis = self.analyzer.analyze_query(code_query)
        api_analysis = self.analyzer.analyze_query(api_query)
        tutorial_analysis = self.analyzer.analyze_query(tutorial_query)

        assert "semantic_code_search" in code_analysis.suggested_strategies
        assert "api_search" in api_analysis.suggested_strategies
        assert "tutorial_prioritized_search" in tutorial_analysis.suggested_strategies

    def test_expansion_terms_generation(self):
        """Test expansion terms generation."""
        query = "Python web framework"

        analysis = self.analyzer.analyze_query(query)

        assert len(analysis.expansion_terms) > 0
        # Should include language-specific expansions
        python_expansions = ["py", "python3", "pythonic"]
        assert any(term in analysis.expansion_terms for term in python_expansions)

    def test_filters_creation(self):
        """Test search filters creation."""
        code_query = "JavaScript async function example"

        analysis = self.analyzer.analyze_query(code_query)

        assert "primary_language" in analysis.filters
        assert analysis.filters["primary_language"] == "javascript"
        assert "content_type" in analysis.filters
        assert analysis.filters["content_type"] == "code_example"

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        high_confidence_query = "React useState hook tutorial"
        low_confidence_query = "something about programming"

        high_analysis = self.analyzer.analyze_query(high_confidence_query)
        low_analysis = self.analyzer.analyze_query(low_confidence_query)

        assert high_analysis.confidence > low_analysis.confidence
        assert high_analysis.confidence > 0.8
        assert low_analysis.confidence < 0.8

    def test_edge_cases(self):
        """Test edge cases and empty inputs."""
        # Empty query
        empty_analysis = self.analyzer.analyze_query("")
        assert empty_analysis.query_type == QueryType.GENERAL
        assert len(empty_analysis.keywords) == 0

        # Very short query
        short_analysis = self.analyzer.analyze_query("CSS")
        assert short_analysis.query_type == QueryType.GENERAL

        # Query with special characters
        special_analysis = self.analyzer.analyze_query(
            "node.js @decorator #comments $variables"
        )
        assert special_analysis.programming_language == "javascript"
        assert "@decorator" in special_analysis.code_elements

    def test_multiple_languages_in_query(self):
        """Test queries mentioning multiple programming languages."""
        query = "migrate from Python Django to JavaScript Node.js"

        analysis = self.analyzer.analyze_query(query)

        # Should detect the first mentioned language
        assert analysis.programming_language in ["python", "javascript"]
        assert "migrate" in analysis.keywords

    def test_code_pattern_detection(self):
        """Test detection of various code patterns."""
        patterns = [
            ("user.getName()", "function_call"),
            ("obj.method", "method_access"),
            ("class User:", "class_reference"),
            ("name = value", "variable_assignment"),
            ("import os", "import_statement"),
            ("@property", "decorator"),
        ]

        for code_snippet, pattern_type in patterns:
            query = f"how to use {code_snippet} in Python"
            analysis = self.analyzer.analyze_query(query)

            assert len(analysis.code_elements) > 0
            # Check if any part of the code snippet was detected
            found_match = any(
                code_snippet in element or element in code_snippet
                for element in analysis.code_elements
            )
            assert (
                found_match
            ), f"No match found for {code_snippet} in {analysis.code_elements}"

    def test_api_pattern_recognition(self):
        """Test API pattern recognition."""
        api_patterns = [
            "GET /users",
            "POST /api/auth",
            "REST endpoint",
            "API documentation",
            "/users/{id}",
        ]

        for pattern in api_patterns:
            query = f"how to implement {pattern}"
            analysis = self.analyzer.analyze_query(query)

            # Should be classified as API reference or have API references
            assert (
                analysis.query_type == QueryType.API_REFERENCE
                or len(analysis.api_references) > 0
            )
