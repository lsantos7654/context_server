"""Query analysis and classification for multi-modal search."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of search queries for routing decisions."""

    CODE_FUNCTION = "code_function"  # Search for specific functions
    CODE_CLASS = "code_class"  # Search for classes/interfaces
    CODE_PATTERN = "code_pattern"  # Search for code patterns/snippets
    API_REFERENCE = "api_reference"  # Search for API documentation
    CONCEPTUAL = "conceptual"  # Search for concepts/explanations
    TUTORIAL = "tutorial"  # Search for how-to/tutorials
    TROUBLESHOOTING = "troubleshooting"  # Search for error solutions
    GENERAL = "general"  # General text search


class SearchIntent(Enum):
    """User intent behind the search query."""

    LEARNING = "learning"  # User wants to learn something
    IMPLEMENTATION = "implementation"  # User wants to implement something
    DEBUGGING = "debugging"  # User has a problem to solve
    REFERENCE = "reference"  # User needs quick reference
    EXPLORATION = "exploration"  # User is exploring/browsing


@dataclass
class QueryAnalysis:
    """Comprehensive analysis of a search query."""

    original_query: str
    query_type: QueryType
    search_intent: SearchIntent
    confidence: float

    # Extracted components
    keywords: List[str]
    code_elements: List[str]
    programming_language: Optional[str]
    api_references: List[str]

    # Query characteristics
    is_question: bool
    complexity_score: float  # 0.0 = simple, 1.0 = complex
    specificity_score: float  # 0.0 = vague, 1.0 = specific

    # Search strategy hints
    suggested_strategies: List[str]
    expansion_terms: List[str]
    filters: Dict[str, Any]


class QueryAnalyzer:
    """Analyzes search queries to determine optimal search strategies."""

    def __init__(self):
        """Initialize query analyzer with patterns and rules."""
        self._code_patterns = {
            "function_call": re.compile(r"\w+\s*\([^)]*\)", re.IGNORECASE),
            "method_access": re.compile(r"\w+\.\w+", re.IGNORECASE),
            "class_reference": re.compile(r"class\s+\w+", re.IGNORECASE),
            "variable_assignment": re.compile(r"\w+\s*=\s*", re.IGNORECASE),
            "import_statement": re.compile(r"import\s+\w+|from\s+\w+", re.IGNORECASE),
            "decorator": re.compile(r"@\w+", re.IGNORECASE),
        }

        self._language_indicators = {
            "python": [
                "python",
                "django",
                "flask",
                "pandas",
                "numpy",
                "def ",
                "import ",
                "__init__",
            ],
            "javascript": [
                "javascript",
                "js",
                "node",
                "react",
                "vue",
                "function ",
                "const ",
                "let ",
            ],
            "java": ["java", "spring", "class ", "public static", "package "],
            "typescript": ["typescript", "ts", "interface ", "type "],
            "go": ["golang", "func ", "package main"],  # Removed ambiguous 'go'
            "rust": ["rust", "cargo", "fn ", "let mut"],
            "c++": ["cpp", "c++", "std::", "#include"],
            "c#": ["csharp", "c#", "using System", "public class"],
        }

        self._intent_keywords = {
            SearchIntent.LEARNING: [
                "learn",
                "tutorial",
                "guide",
                "introduction",
                "basics",
                "beginner",
                "how to",
                "what is",
                "explain",
            ],
            SearchIntent.IMPLEMENTATION: [
                "implement",
                "create",
                "build",
                "develop",
                "code",
                "example",
                "sample",
                "template",
            ],
            SearchIntent.DEBUGGING: [
                "error",
                "bug",
                "issue",
                "problem",
                "fix",
                "debug",
                "troubleshoot",
                "not working",
                "failed",
            ],
            SearchIntent.REFERENCE: [
                "documentation",
                "docs",
                "api",
                "reference",
                "parameters",
                "syntax",
                "specification",
            ],
            SearchIntent.EXPLORATION: [
                "alternatives",
                "compare",
                "options",
                "best practices",
                "patterns",
                "overview",
            ],
        }

        self._api_patterns = [
            re.compile(r"api\s+\w+", re.IGNORECASE),
            re.compile(r"\w+\s+api", re.IGNORECASE),
            re.compile(r"endpoint", re.IGNORECASE),
            re.compile(r"GET|POST|PUT|DELETE|PATCH", re.IGNORECASE),
            re.compile(r"/\w+", re.IGNORECASE),  # URL patterns
        ]

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a search query and return comprehensive analysis."""

        logger.info(f"Analyzing query: '{query}'")

        # Basic preprocessing
        query_lower = query.lower().strip()

        # Extract components
        keywords = self._extract_keywords(query)
        code_elements = self._extract_code_elements(query)
        programming_language = self._detect_programming_language(query)
        api_references = self._extract_api_references(query)

        # Classify query type
        query_type = self._classify_query_type(query, code_elements, api_references)

        # Determine search intent
        search_intent = self._determine_search_intent(query)

        # Calculate scores
        is_question = self._is_question(query)
        complexity_score = self._calculate_complexity_score(query, code_elements)
        specificity_score = self._calculate_specificity_score(query, code_elements)
        confidence = self._calculate_confidence(
            query_type, search_intent, specificity_score
        )

        # Generate search strategy suggestions
        suggested_strategies = self._suggest_search_strategies(
            query_type, search_intent, complexity_score, programming_language
        )

        # Generate expansion terms
        expansion_terms = self._generate_expansion_terms(
            query, query_type, programming_language
        )

        # Create filters
        filters = self._create_filters(query_type, programming_language, api_references)

        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            search_intent=search_intent,
            confidence=confidence,
            keywords=keywords,
            code_elements=code_elements,
            programming_language=programming_language,
            api_references=api_references,
            is_question=is_question,
            complexity_score=complexity_score,
            specificity_score=specificity_score,
            suggested_strategies=suggested_strategies,
            expansion_terms=expansion_terms,
            filters=filters,
        )

        logger.info(
            f"Query analysis complete: type={query_type.value}, intent={search_intent.value}, confidence={confidence:.2f}"
        )

        return analysis

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "how",
            "to",
            "what",
            "where",
            "when",
            "why",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "for",
            "with",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }

        # Split and filter
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:10]  # Limit to top 10 keywords

    def _extract_code_elements(self, query: str) -> List[str]:
        """Extract code-related elements from query."""
        code_elements = []

        for pattern_name, pattern in self._code_patterns.items():
            matches = pattern.findall(query)
            for match in matches:
                code_elements.append(match.strip())

        # Also look for camelCase or snake_case identifiers
        identifier_pattern = re.compile(r"\b(?:[a-z]+[A-Z][a-zA-Z]*|[a-z]+_[a-z_]+)\b")
        identifiers = identifier_pattern.findall(query)
        code_elements.extend(identifiers)

        # Look for standalone keywords that might be code elements
        code_keywords = [
            "async",
            "await",
            "const",
            "let",
            "var",
            "def",
            "class",
            "interface",
            "type",
        ]
        query_words = re.findall(r"\b\w+\b", query.lower())
        for keyword in code_keywords:
            if keyword in query_words:
                code_elements.append(keyword)

        return list(set(code_elements))[:20]  # Remove duplicates, limit to 20

    def _detect_programming_language(self, query: str) -> Optional[str]:
        """Detect programming language mentioned in query."""
        query_lower = query.lower()

        # Check for explicit language mentions
        for language, indicators in self._language_indicators.items():
            for indicator in indicators:
                if indicator.lower() in query_lower:
                    return language

        return None

    def _extract_api_references(self, query: str) -> List[str]:
        """Extract API-related references from query."""
        api_refs = []

        for pattern in self._api_patterns:
            matches = pattern.findall(query)
            api_refs.extend(matches)

        # Look for REST method + endpoint patterns
        rest_pattern = re.compile(
            r"(GET|POST|PUT|DELETE|PATCH)\s+(/\S+)", re.IGNORECASE
        )
        rest_matches = rest_pattern.findall(query)
        for method, endpoint in rest_matches:
            api_refs.append(f"{method} {endpoint}")

        return list(set(api_refs))

    def _classify_query_type(
        self, query: str, code_elements: List[str], api_references: List[str]
    ) -> QueryType:
        """Classify the type of query based on content analysis."""
        query_lower = query.lower()

        # Check for API references
        if api_references or any(
            word in query_lower for word in ["api", "endpoint", "rest"]
        ):
            return QueryType.API_REFERENCE

        # Check for function-specific queries
        if any(pattern in query_lower for pattern in ["function", "method", "def "]):
            return QueryType.CODE_FUNCTION

        # Check for class-specific queries
        if any(pattern in query_lower for pattern in ["class", "interface", "struct"]):
            return QueryType.CODE_CLASS

        # Check for code patterns
        if code_elements or any(
            word in query_lower for word in ["code", "implement", "syntax"]
        ):
            return QueryType.CODE_PATTERN

        # Check for tutorial intent
        if any(
            word in query_lower
            for word in ["tutorial", "guide", "how to", "step by step"]
        ):
            return QueryType.TUTORIAL

        # Check for troubleshooting
        if any(
            word in query_lower for word in ["error", "bug", "issue", "problem", "fix"]
        ):
            return QueryType.TROUBLESHOOTING

        # Check for conceptual queries
        if any(
            word in query_lower for word in ["what is", "explain", "concept", "theory"]
        ):
            return QueryType.CONCEPTUAL

        return QueryType.GENERAL

    def _determine_search_intent(self, query: str) -> SearchIntent:
        """Determine the user's search intent."""
        query_lower = query.lower()

        # Score each intent based on keyword matches
        intent_scores = {}

        for intent, keywords in self._intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score

        # Return the highest scoring intent, or exploration as default
        if intent_scores:
            return max(intent_scores.keys(), key=lambda x: intent_scores[x])

        return SearchIntent.EXPLORATION

    def _is_question(self, query: str) -> bool:
        """Determine if the query is phrased as a question."""
        # Explicit question mark
        if "?" in query:
            return True

        # Question words at the beginning of query
        question_words = ["how", "what", "where", "when", "why", "which", "who"]
        query_lower = query.lower().strip()

        # Check if query starts with question words
        for word in question_words:
            if query_lower.startswith(word + " "):
                return True

        return False

    def _calculate_complexity_score(
        self, query: str, code_elements: List[str]
    ) -> float:
        """Calculate query complexity score (0.0 = simple, 1.0 = complex)."""
        score = 0.0

        # Length factor
        word_count = len(query.split())
        score += min(word_count / 20.0, 0.3)  # Max 0.3 for length

        # Code elements factor
        score += min(len(code_elements) / 10.0, 0.3)  # Max 0.3 for code complexity

        # Technical terms factor
        technical_terms = [
            "algorithm",
            "optimization",
            "architecture",
            "design pattern",
            "performance",
            "scalability",
            "concurrency",
            "asynchronous",
        ]
        tech_score = sum(1 for term in technical_terms if term in query.lower())
        score += min(tech_score / 5.0, 0.2)  # Max 0.2 for technical complexity

        # Multiple concepts factor
        concepts = ["and", "with", "using", "while", "during", "between"]
        concept_score = sum(1 for concept in concepts if concept in query.lower())
        score += min(concept_score / 3.0, 0.2)  # Max 0.2 for multiple concepts

        return min(score, 1.0)

    def _calculate_specificity_score(
        self, query: str, code_elements: List[str]
    ) -> float:
        """Calculate query specificity score (0.0 = vague, 1.0 = specific)."""
        score = 0.0

        # Specific code elements increase specificity
        score += min(len(code_elements) / 5.0, 0.4)

        # Specific function/method names
        function_pattern = re.compile(r"\w+\(\)", re.IGNORECASE)
        function_matches = len(function_pattern.findall(query))
        score += min(function_matches / 3.0, 0.3)

        # Version numbers, exact names increase specificity
        specific_patterns = [
            re.compile(r"\d+\.\d+"),  # Version numbers
            re.compile(r"[A-Z][a-zA-Z]*[A-Z]"),  # CamelCase names
            re.compile(r"[a-z_]+_[a-z_]+"),  # snake_case names
        ]

        for pattern in specific_patterns:
            matches = len(pattern.findall(query))
            score += min(matches / 5.0, 0.1)

        # Avoid vague terms
        vague_terms = ["thing", "stuff", "something", "anything", "way", "method"]
        vague_penalty = sum(1 for term in vague_terms if term in query.lower())
        score -= min(vague_penalty / 5.0, 0.2)

        return max(0.0, min(score, 1.0))

    def _calculate_confidence(
        self,
        query_type: QueryType,
        search_intent: SearchIntent,
        specificity_score: float,
    ) -> float:
        """Calculate confidence in the analysis."""
        base_confidence = 0.7

        # High specificity increases confidence
        base_confidence += specificity_score * 0.2

        # Strong indicators increase confidence
        if query_type != QueryType.GENERAL:
            base_confidence += 0.1

        if search_intent != SearchIntent.EXPLORATION:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _suggest_search_strategies(
        self,
        query_type: QueryType,
        search_intent: SearchIntent,
        complexity_score: float,
        programming_language: Optional[str],
    ) -> List[str]:
        """Suggest optimal search strategies based on analysis."""
        strategies = []

        # Base strategy selection
        if query_type in [
            QueryType.CODE_FUNCTION,
            QueryType.CODE_CLASS,
            QueryType.CODE_PATTERN,
        ]:
            strategies.append("semantic_code_search")
            if programming_language:
                strategies.append("language_specific_search")

        if query_type == QueryType.API_REFERENCE:
            strategies.append("api_search")
            strategies.append("structured_search")

        if query_type in [QueryType.CONCEPTUAL, QueryType.TUTORIAL]:
            strategies.append("semantic_search")
            strategies.append("hierarchical_search")

        if complexity_score > 0.7:
            strategies.append("progressive_refinement")
            strategies.append("multi_strategy_fusion")

        if search_intent == SearchIntent.LEARNING:
            strategies.append("tutorial_prioritized_search")

        # Default fallbacks
        if not strategies:
            strategies.extend(["hybrid_search", "semantic_search"])

        return strategies

    def _generate_expansion_terms(
        self, query: str, query_type: QueryType, programming_language: Optional[str]
    ) -> List[str]:
        """Generate terms to expand the search query."""
        expansion_terms = []

        # Language-specific expansions
        if programming_language:
            lang_synonyms = {
                "python": ["py", "python3", "pythonic"],
                "javascript": ["js", "ecmascript", "node.js"],
                "java": ["jvm", "openjdk"],
                "typescript": ["ts", "javascript"],
            }
            expansion_terms.extend(lang_synonyms.get(programming_language, []))

        # Query type specific expansions
        if query_type == QueryType.API_REFERENCE:
            expansion_terms.extend(["documentation", "endpoint", "reference", "spec"])
        elif query_type == QueryType.TUTORIAL:
            expansion_terms.extend(["guide", "example", "walkthrough", "step-by-step"])
        elif query_type == QueryType.TROUBLESHOOTING:
            expansion_terms.extend(["solution", "fix", "resolve", "debug"])

        return expansion_terms

    def _create_filters(
        self,
        query_type: QueryType,
        programming_language: Optional[str],
        api_references: List[str],
    ) -> Dict[str, Any]:
        """Create search filters based on query analysis."""
        filters = {}

        if programming_language:
            filters["primary_language"] = programming_language

        if query_type in [
            QueryType.CODE_PATTERN,
            QueryType.CODE_FUNCTION,
            QueryType.CODE_CLASS,
        ]:
            filters["content_type"] = "code_example"
            filters["min_code_percentage"] = 30
        elif query_type == QueryType.API_REFERENCE:
            filters["content_type"] = "api_reference"
        elif query_type == QueryType.TUTORIAL:
            filters["content_type"] = "tutorial"

        if api_references:
            filters["has_api_references"] = True

        return filters
