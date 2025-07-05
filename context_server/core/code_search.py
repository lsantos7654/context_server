"""Advanced code search system with pattern recognition and API discovery."""

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .multi_modal_search import SearchEngine, SearchResponse, SearchResult
from .query_analysis import QueryAnalysis, QueryAnalyzer

logger = logging.getLogger(__name__)


class CodeSearchType(Enum):
    """Types of code-specific searches."""

    FUNCTION_SIGNATURE = "function_signature"  # Function definitions and usage
    CLASS_DEFINITION = "class_definition"  # Class structures and inheritance
    DESIGN_PATTERN = "design_pattern"  # Programming patterns and architectures
    ERROR_HANDLING = "error_handling"  # Exception handling and debugging
    ALGORITHM_IMPLEMENTATION = "algorithm_implementation"  # Specific algorithms
    API_ENDPOINT = "api_endpoint"  # REST API endpoints and usage
    LIBRARY_USAGE = "library_usage"  # Framework and library examples
    CODE_SNIPPET = "code_snippet"  # General code examples


@dataclass
class CodePattern:
    """Represents a recognized code pattern."""

    pattern_type: CodeSearchType
    confidence: float
    matched_elements: List[str]
    programming_language: Optional[str]
    framework_hints: List[str]
    complexity_score: float


@dataclass
class FunctionSignature:
    """Represents a function signature for search."""

    name: str
    parameters: List[str]
    return_type: Optional[str]
    programming_language: str
    modifiers: List[str]  # async, static, private, etc.
    documentation: str


@dataclass
class APIEndpoint:
    """Represents an API endpoint for search."""

    method: str  # GET, POST, PUT, DELETE
    path: str
    parameters: List[str]
    response_format: Optional[str]
    authentication_required: bool
    documentation: str


class CodePatternAnalyzer:
    """Analyzes queries to identify code patterns and search requirements."""

    def __init__(self):
        """Initialize pattern analyzer with recognition rules."""
        self.function_patterns = [
            r"\b(def|function|func|fn)\s+(\w+)",  # Function definitions
            r"(\w+)\s*\(",  # Function calls
            r"async\s+(\w+)",  # Async functions
            r"(\w+)\.(\w+)\(",  # Method calls
            r"\bfunction\b",  # General function keyword
            r"\bmethod\b",  # Method keyword
        ]

        self.class_patterns = [
            r"\b(class|struct|interface)\s+(\w+)",  # Class definitions
            r"extends\s+(\w+)",  # Inheritance
            r"implements\s+(\w+)",  # Interface implementation
            r"\bclass\b",  # General class keyword
            r"\binheritance\b",  # Inheritance keyword
        ]

        self.api_patterns = [
            r"\b(GET|POST|PUT|DELETE|PATCH)\s+([/\w-]+)",  # HTTP methods
            r"(/api/[\w/-]+)",  # API paths
            r"endpoint\s*[:\s]*([/\w-]+)",  # Endpoint descriptions
            r"(fetch|axios|request)\s*\(",  # HTTP clients
        ]

        self.error_patterns = [
            r"\b(try|catch|except|finally|throw|raise)",  # Error handling keywords
            r"(error|exception|Error|Exception)",  # Error types
            r"(catch|handle)\s+(\w+)",  # Exception handling
        ]

        self.algorithm_patterns = [
            r"\b(sort|search|tree|graph|hash|binary)",  # Algorithm keywords
            r"(recursive|iterative|dynamic programming)",  # Approach keywords
            r"(O\([^)]+\))",  # Big O notation
        ]

    def analyze_code_query(
        self, query: str, query_analysis: QueryAnalysis
    ) -> CodePattern:
        """Analyze query to identify code search patterns."""

        query_lower = query.lower()
        matched_elements = []
        framework_hints = []
        pattern_scores = {}

        # Function pattern detection
        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.function_patterns
        ):
            pattern_scores[CodeSearchType.FUNCTION_SIGNATURE] = 0.3
            matched_elements.extend(self._extract_function_elements(query))

        # Class pattern detection
        if any(
            re.search(pattern, query, re.IGNORECASE) for pattern in self.class_patterns
        ):
            pattern_scores[CodeSearchType.CLASS_DEFINITION] = 0.3
            matched_elements.extend(self._extract_class_elements(query))

        # API pattern detection
        if any(
            re.search(pattern, query, re.IGNORECASE) for pattern in self.api_patterns
        ):
            pattern_scores[CodeSearchType.API_ENDPOINT] = 0.4
            matched_elements.extend(self._extract_api_elements(query))

        # Error handling detection
        if any(
            re.search(pattern, query, re.IGNORECASE) for pattern in self.error_patterns
        ):
            pattern_scores[CodeSearchType.ERROR_HANDLING] = 0.2
            matched_elements.extend(self._extract_error_elements(query))

        # Algorithm detection
        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.algorithm_patterns
        ):
            pattern_scores[CodeSearchType.ALGORITHM_IMPLEMENTATION] = 0.25
            matched_elements.extend(self._extract_algorithm_elements(query))

        # Design pattern detection (check before library usage as it's more specific)
        if self._is_design_pattern_query(query):
            pattern_scores[CodeSearchType.DESIGN_PATTERN] = 0.35

        # Library usage detection
        if self._is_library_usage_query(query):
            pattern_scores[CodeSearchType.LIBRARY_USAGE] = 0.2

        # Framework detection
        framework_hints = self._detect_frameworks(query)

        # Select pattern type with highest score
        if pattern_scores:
            pattern_type = max(pattern_scores.keys(), key=lambda k: pattern_scores[k])
            confidence = pattern_scores[pattern_type]
        else:
            pattern_type = CodeSearchType.CODE_SNIPPET
            confidence = 0.0

        # Additional confidence boosts
        if framework_hints:
            confidence += 0.1

        # Boost confidence based on code elements in original analysis
        if query_analysis.code_elements:
            confidence += 0.2

        # Calculate complexity based on query elements
        complexity_score = self._calculate_code_complexity(query, matched_elements)

        return CodePattern(
            pattern_type=pattern_type,
            confidence=min(confidence, 1.0),
            matched_elements=list(set(matched_elements)),
            programming_language=query_analysis.programming_language,
            framework_hints=framework_hints,
            complexity_score=complexity_score,
        )

    def _extract_function_elements(self, query: str) -> List[str]:
        """Extract function-related elements from query."""
        elements = []

        for pattern in self.function_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 2:
                    elements.append(match.group(2))
                elif match.lastindex and match.lastindex >= 1:
                    elements.append(match.group(1))
                else:
                    # For patterns without groups, add the full match
                    elements.append(match.group(0))

        return elements

    def _extract_class_elements(self, query: str) -> List[str]:
        """Extract class-related elements from query."""
        elements = []

        for pattern in self.class_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 2:
                    elements.append(match.group(2))
                elif match.lastindex and match.lastindex >= 1:
                    elements.append(match.group(1))
                else:
                    # For patterns without groups, add the full match
                    elements.append(match.group(0))

        return elements

    def _extract_api_elements(self, query: str) -> List[str]:
        """Extract API-related elements from query."""
        elements = []

        for pattern in self.api_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 1:
                    elements.append(match.group(1))
                if match.lastindex and match.lastindex >= 2:
                    elements.append(match.group(2))

        return elements

    def _extract_error_elements(self, query: str) -> List[str]:
        """Extract error handling elements from query."""
        elements = []

        for pattern in self.error_patterns:
            logger.debug(f"Processing error pattern: {pattern}")
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    # Debug logging for match analysis
                    logger.debug(f"Error pattern match: {match.group(0)}")
                    logger.debug(f"Error pattern groups: {match.groups()}")
                    logger.debug(f"Error pattern group types: {[type(g) for g in match.groups()]}")
                    
                    # Add non-None groups to elements
                    for group in match.groups():
                        logger.debug(f"Processing error group: {repr(group)} (type: {type(group)})")
                        if group and group.strip():
                            elements.append(group.strip())
                            logger.debug(f"Added error element: {repr(group.strip())}")
                            
                except Exception as e:
                    logger.error(f"Error processing error pattern match: {e}")
                    logger.error(f"Match object: {match}")
                    logger.error(f"Match groups: {match.groups()}")
                    logger.error(f"Pattern: {pattern}")
                    raise

        return elements

    def _extract_algorithm_elements(self, query: str) -> List[str]:
        """Extract algorithm-related elements from query."""
        elements = []

        for pattern in self.algorithm_patterns:
            logger.debug(f"Processing algorithm pattern: {pattern}")
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    # Debug logging for match analysis
                    logger.debug(f"Algorithm pattern match: {match.group(0)}")
                    logger.debug(f"Algorithm pattern groups: {match.groups()}")
                    logger.debug(f"Algorithm pattern group types: {[type(g) for g in match.groups()]}")
                    
                    # Add non-None groups to elements
                    for group in match.groups():
                        logger.debug(f"Processing algorithm group: {repr(group)} (type: {type(group)})")
                        if group and group.strip():
                            elements.append(group.strip())
                            logger.debug(f"Added algorithm element: {repr(group.strip())}")
                            
                except Exception as e:
                    logger.error(f"Error processing algorithm pattern match: {e}")
                    logger.error(f"Match object: {match}")
                    logger.error(f"Match groups: {match.groups()}")
                    logger.error(f"Pattern: {pattern}")
                    raise

        return elements

    def _detect_frameworks(self, query: str) -> List[str]:
        """Detect framework mentions in the query."""
        frameworks = {
            "react": ["react", "jsx", "hooks", "usestate", "useeffect"],
            "angular": ["angular", "typescript", "component", "directive"],
            "vue": ["vue", "vuejs", "composition api"],
            "django": ["django", "models", "views", "templates"],
            "flask": ["flask", "route", "app.run"],
            "spring": ["spring", "boot", "mvc", "@autowired"],
            "express": ["express", "router", "middleware"],
            "fastapi": ["fastapi", "pydantic", "async def"],
            "tensorflow": ["tensorflow", "tf", "keras"],
            "pytorch": ["pytorch", "torch", "tensor"],
            "pandas": ["pandas", "dataframe", "series"],
            "numpy": ["numpy", "array", "ndarray"],
        }

        detected = []
        query_lower = query.lower()

        for framework, keywords in frameworks.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(framework)

        return detected

    def _is_design_pattern_query(self, query: str) -> bool:
        """Check if query is about design patterns."""
        pattern_keywords = [
            "singleton",
            "factory",
            "observer",
            "strategy",
            "adapter",
            "decorator",
            "facade",
            "proxy",
            "command",
            "state",
            "template",
            "builder",
            "prototype",
            "composite",
            "bridge",
            "flyweight",
            "chain of responsibility",
            "iterator",
            "mediator",
            "memento",
            "visitor",
            "design pattern",
            "architectural pattern",
        ]

        query_lower = query.lower()
        return any(pattern in query_lower for pattern in pattern_keywords)

    def _is_library_usage_query(self, query: str) -> bool:
        """Check if query is about library/framework usage."""
        usage_keywords = [
            "how to use",
            "example",
            "tutorial",
            "getting started",
            "import",
            "install",
            "setup",
            "configuration",
            "usage",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in usage_keywords)

    def _calculate_code_complexity(
        self, query: str, matched_elements: List[str]
    ) -> float:
        """Calculate complexity score for the code query."""
        complexity = 0.0

        # Base complexity from matched elements
        complexity += min(len(matched_elements) * 0.1, 0.3)

        # Complexity keywords
        complex_keywords = [
            "advanced",
            "complex",
            "optimization",
            "performance",
            "concurrent",
            "parallel",
            "distributed",
            "scalable",
            "enterprise",
            "production",
            "architecture",
        ]

        query_lower = query.lower()
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity += 0.15

        # Technical depth indicators
        technical_indicators = [
            "algorithm",
            "data structure",
            "pattern",
            "framework",
            "library",
            "API",
            "database",
            "async",
            "thread",
        ]

        for indicator in technical_indicators:
            if indicator in query_lower:
                complexity += 0.1

        return min(complexity, 1.0)


class SemanticCodeSearchEngine:
    """Advanced semantic search engine specialized for code queries."""

    def __init__(self, base_search_engine: SearchEngine):
        """Initialize with base search engine."""
        self.base_search_engine = base_search_engine
        self.pattern_analyzer = CodePatternAnalyzer()
        self.query_analyzer = QueryAnalyzer()

    async def semantic_code_search(
        self, query: str, limit: int = 10, filters: Dict[str, Any] = None
    ) -> SearchResponse:
        """Execute semantic code search with pattern recognition."""

        logger.info(f"Starting semantic code search for: '{query}'")

        # Analyze query for code patterns
        query_analysis = self.query_analyzer.analyze_query(query)
        code_pattern = self.pattern_analyzer.analyze_code_query(query, query_analysis)

        # Build enhanced filters for code search
        code_filters = self._build_code_filters(
            filters or {}, code_pattern, query_analysis
        )

        # Execute specialized search based on pattern type
        if code_pattern.pattern_type == CodeSearchType.FUNCTION_SIGNATURE:
            return await self._function_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        elif code_pattern.pattern_type == CodeSearchType.CLASS_DEFINITION:
            return await self._class_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        elif code_pattern.pattern_type == CodeSearchType.API_ENDPOINT:
            return await self._api_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        elif code_pattern.pattern_type == CodeSearchType.DESIGN_PATTERN:
            return await self._design_pattern_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        elif code_pattern.pattern_type == CodeSearchType.ERROR_HANDLING:
            return await self._error_handling_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        elif code_pattern.pattern_type == CodeSearchType.ALGORITHM_IMPLEMENTATION:
            return await self._algorithm_search(
                query, query_analysis, code_pattern, limit, code_filters
            )
        else:
            # Default code search
            return await self._general_code_search(
                query, query_analysis, code_pattern, limit, code_filters
            )

    def _build_code_filters(
        self,
        base_filters: Dict[str, Any],
        code_pattern: CodePattern,
        query_analysis: QueryAnalysis,
    ) -> Dict[str, Any]:
        """Build specialized filters for code searches."""

        filters = base_filters.copy()

        # Minimum code percentage for code searches
        filters["min_code_percentage"] = 15

        # Programming language filter
        if code_pattern.programming_language:
            filters["primary_language"] = code_pattern.programming_language

        # Content type preferences
        if code_pattern.pattern_type in [
            CodeSearchType.FUNCTION_SIGNATURE,
            CodeSearchType.CLASS_DEFINITION,
        ]:
            filters["content_type"] = "code_example"
        elif code_pattern.pattern_type == CodeSearchType.API_ENDPOINT:
            filters["content_type"] = "api_reference"
        elif code_pattern.pattern_type == CodeSearchType.DESIGN_PATTERN:
            filters["content_type"] = "tutorial"

        # Framework filters
        if code_pattern.framework_hints:
            filters["framework"] = code_pattern.framework_hints[0]

        return filters

    async def _function_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for function definitions and usage examples."""

        # Enhance query with function-specific terms
        enhanced_query = self._enhance_function_query(query, code_pattern)

        # Execute search with code-specific strategy
        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        # Post-process results for function relevance
        enhanced_results = self._enhance_function_results(
            response.results, code_pattern
        )

        # Update response
        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _class_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for class definitions and object-oriented patterns."""

        enhanced_query = self._enhance_class_query(query, code_pattern)

        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        enhanced_results = self._enhance_class_results(response.results, code_pattern)

        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _api_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for API endpoints and documentation."""

        # Force API content type
        filters["content_type"] = "api_reference"

        enhanced_query = self._enhance_api_query(query, code_pattern)

        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        enhanced_results = self._enhance_api_results(response.results, code_pattern)

        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _design_pattern_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for design patterns and architectural examples."""

        enhanced_query = self._enhance_pattern_query(query, code_pattern)

        # Prefer tutorial and code example content
        filters["content_type"] = "tutorial"

        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        enhanced_results = self._enhance_pattern_results(response.results, code_pattern)

        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _error_handling_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for error handling and debugging examples."""

        enhanced_query = self._enhance_error_query(query, code_pattern)

        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        enhanced_results = self._enhance_error_results(response.results, code_pattern)

        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _algorithm_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """Search for algorithm implementations and data structures."""

        enhanced_query = self._enhance_algorithm_query(query, code_pattern)

        response = await self.base_search_engine.search(
            enhanced_query, limit=limit, filters=filters
        )

        enhanced_results = self._enhance_algorithm_results(
            response.results, code_pattern
        )

        response.results = enhanced_results
        response.query = enhanced_query

        return response

    async def _general_code_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        code_pattern: CodePattern,
        limit: int,
        filters: Dict[str, Any],
    ) -> SearchResponse:
        """General code search with pattern awareness."""

        response = await self.base_search_engine.search(
            query, limit=limit, filters=filters
        )

        # Boost code-specific content
        for result in response.results:
            if result.content_type == "code_example":
                result.relevance_score *= 1.2
            if (
                code_pattern.programming_language
                and result.programming_language == code_pattern.programming_language
            ):
                result.relevance_score *= 1.1

        # Re-sort by relevance
        response.results.sort(key=lambda x: x.relevance_score, reverse=True)

        return response

    def _enhance_function_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for function search."""
        enhanced = query

        if code_pattern.programming_language:
            enhanced = f"{code_pattern.programming_language} {enhanced}"

        if "function" not in enhanced.lower() and "def" not in enhanced.lower():
            enhanced = f"{enhanced} function definition example"

        return enhanced

    def _enhance_class_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for class search."""
        enhanced = query

        if code_pattern.programming_language:
            enhanced = f"{code_pattern.programming_language} {enhanced}"

        if "class" not in enhanced.lower():
            enhanced = f"{enhanced} class definition example"

        return enhanced

    def _enhance_api_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for API search."""
        enhanced = query

        if "api" not in enhanced.lower():
            enhanced = f"{enhanced} API documentation"

        return enhanced

    def _enhance_pattern_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for design pattern search."""
        enhanced = query

        if code_pattern.programming_language:
            enhanced = f"{code_pattern.programming_language} {enhanced}"

        if "pattern" not in enhanced.lower():
            enhanced = f"{enhanced} design pattern example"

        return enhanced

    def _enhance_error_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for error handling search."""
        enhanced = query

        if code_pattern.programming_language:
            enhanced = f"{code_pattern.programming_language} {enhanced}"

        if not any(
            keyword in enhanced.lower()
            for keyword in ["error", "exception", "try", "catch"]
        ):
            enhanced = f"{enhanced} error handling example"

        return enhanced

    def _enhance_algorithm_query(self, query: str, code_pattern: CodePattern) -> str:
        """Enhance query for algorithm search."""
        enhanced = query

        if code_pattern.programming_language:
            enhanced = f"{code_pattern.programming_language} {enhanced}"

        if (
            "algorithm" not in enhanced.lower()
            and "implementation" not in enhanced.lower()
        ):
            enhanced = f"{enhanced} algorithm implementation"

        return enhanced

    def _enhance_function_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance function search results with better scoring."""

        for result in results:
            # Boost results with function definitions
            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.3

            # Boost results with actual code
            if result.content_type == "code_example":
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_class_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance class search results with better scoring."""

        for result in results:
            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.3

            if result.content_type == "code_example":
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_api_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance API search results with better scoring."""

        for result in results:
            if result.content_type == "api_reference":
                result.relevance_score *= 1.4

            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_pattern_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance design pattern search results with better scoring."""

        for result in results:
            if result.content_type == "tutorial":
                result.relevance_score *= 1.3

            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_error_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance error handling search results with better scoring."""

        for result in results:
            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.3

            if result.content_type == "code_example":
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_algorithm_results(
        self, results: List[SearchResult], code_pattern: CodePattern
    ) -> List[SearchResult]:
        """Enhance algorithm search results with better scoring."""

        for result in results:
            if any(
                element in result.content.lower()
                for element in code_pattern.matched_elements
            ):
                result.relevance_score *= 1.3

            if result.content_type == "code_example":
                result.relevance_score *= 1.2

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)


class APIFunctionSearchEngine:
    """Specialized search engine for API endpoints and function documentation."""

    def __init__(self, base_search_engine: SearchEngine):
        """Initialize with base search engine."""
        self.base_search_engine = base_search_engine
        self.query_analyzer = QueryAnalyzer()

    async def search_api_endpoints(
        self, query: str, limit: int = 10, filters: Dict[str, Any] = None
    ) -> SearchResponse:
        """Search for API endpoints and documentation."""

        logger.info(f"Starting API endpoint search for: '{query}'")

        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Extract API elements
        api_info = self._extract_api_info(query)

        # Build API-specific filters
        api_filters = self._build_api_filters(filters or {}, api_info)

        # Execute search
        response = await self.base_search_engine.search(
            query, limit=limit, filters=api_filters
        )

        # Enhance results for API relevance
        enhanced_results = self._enhance_api_endpoint_results(
            response.results, api_info
        )

        response.results = enhanced_results

        return response

    async def search_function_documentation(
        self, query: str, limit: int = 10, filters: Dict[str, Any] = None
    ) -> SearchResponse:
        """Search for function documentation and usage examples."""

        logger.info(f"Starting function documentation search for: '{query}'")

        # Analyze query for function patterns
        query_analysis = self.query_analyzer.analyze_query(query)
        function_info = self._extract_function_info(query)

        # Build function-specific filters
        function_filters = self._build_function_filters(
            filters or {}, function_info, query_analysis
        )

        # Execute search
        response = await self.base_search_engine.search(
            query, limit=limit, filters=function_filters
        )

        # Enhance results for function relevance
        enhanced_results = self._enhance_function_documentation_results(
            response.results, function_info
        )

        response.results = enhanced_results

        return response

    def _extract_api_info(self, query: str) -> Dict[str, Any]:
        """Extract API information from query."""

        api_info = {"methods": [], "paths": [], "parameters": [], "has_endpoint": False}

        # HTTP methods
        method_pattern = r"\b(GET|POST|PUT|DELETE|PATCH)\b"
        methods = re.findall(method_pattern, query.upper())
        api_info["methods"] = methods

        # API paths
        path_pattern = r"(/[a-zA-Z0-9/_-]*[a-zA-Z0-9/_-]+)"
        paths = re.findall(path_pattern, query)
        api_info["paths"] = paths

        # Parameter patterns
        param_pattern = r"\{([a-zA-Z0-9_]+)\}"
        parameters = re.findall(param_pattern, query)
        api_info["parameters"] = parameters

        # Check if query has endpoint indicators
        endpoint_indicators = ["endpoint", "api", "rest", "route", "url"]
        api_info["has_endpoint"] = any(
            indicator in query.lower() for indicator in endpoint_indicators
        )

        return api_info

    def _extract_function_info(self, query: str) -> Dict[str, Any]:
        """Extract function information from query."""

        function_info = {
            "names": [],
            "signatures": [],
            "modifiers": [],
            "has_function": False,
        }

        # Function names
        func_pattern = r"\b(\w+)\s*\("
        function_names = re.findall(func_pattern, query)
        function_info["names"] = function_names

        # Function modifiers
        modifier_pattern = r"\b(async|static|private|public|protected)\b"
        modifiers = re.findall(modifier_pattern, query.lower())
        function_info["modifiers"] = modifiers

        # Check if query has function indicators
        function_indicators = ["function", "method", "def", "fn", "func"]
        function_info["has_function"] = any(
            indicator in query.lower() for indicator in function_indicators
        )

        return function_info

    def _build_api_filters(
        self, base_filters: Dict[str, Any], api_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build filters for API search."""

        filters = base_filters.copy()

        # Force API content type
        filters["content_type"] = "api_reference"

        # Add method filters if specified
        if api_info["methods"]:
            filters["http_method"] = api_info["methods"][0]

        return filters

    def _build_function_filters(
        self,
        base_filters: Dict[str, Any],
        function_info: Dict[str, Any],
        query_analysis: QueryAnalysis,
    ) -> Dict[str, Any]:
        """Build filters for function search."""

        filters = base_filters.copy()

        # Prefer code examples and API references
        filters["content_type"] = "code_example"

        # Add language filter if detected
        if query_analysis.programming_language:
            filters["primary_language"] = query_analysis.programming_language

        # Require some code content
        filters["min_code_percentage"] = 10

        return filters

    def _enhance_api_endpoint_results(
        self, results: List[SearchResult], api_info: Dict[str, Any]
    ) -> List[SearchResult]:
        """Enhance API endpoint search results."""

        for result in results:
            # Boost API reference content
            if result.content_type == "api_reference":
                result.relevance_score *= 1.4

            # Boost results with matching HTTP methods
            if api_info["methods"]:
                for method in api_info["methods"]:
                    if method in result.content.upper():
                        result.relevance_score *= 1.2
                        break

            # Boost results with matching paths
            if api_info["paths"]:
                for path in api_info["paths"]:
                    if path in result.content:
                        result.relevance_score *= 1.3
                        break

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _enhance_function_documentation_results(
        self, results: List[SearchResult], function_info: Dict[str, Any]
    ) -> List[SearchResult]:
        """Enhance function documentation search results."""

        for result in results:
            # Boost code examples
            if result.content_type == "code_example":
                result.relevance_score *= 1.3

            # Boost API references
            if result.content_type == "api_reference":
                result.relevance_score *= 1.2

            # Boost results with matching function names
            if function_info["names"]:
                for name in function_info["names"]:
                    if name in result.content:
                        result.relevance_score *= 1.3
                        break

            # Boost results with matching modifiers
            if function_info["modifiers"]:
                for modifier in function_info["modifiers"]:
                    if modifier in result.content.lower():
                        result.relevance_score *= 1.1
                        break

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
