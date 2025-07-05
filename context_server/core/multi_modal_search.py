"""Multi-modal search system with intelligent routing and progressive refinement."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .content_analysis import ContentAnalysis
from .embedding_strategies import EmbeddingResult, EmbeddingStrategy
from .enhanced_storage import EnhancedDatabaseManager
from .multi_embedding_service import EmbeddingModel, MultiEmbeddingService
from .query_analysis import QueryAnalysis, QueryAnalyzer, QueryType, SearchIntent

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Available search strategies for different query types."""

    SEMANTIC_SEARCH = "semantic_search"  # Vector similarity search
    SEMANTIC_CODE_SEARCH = "semantic_code_search"  # Code-specific semantic search
    HYBRID_SEARCH = "hybrid_search"  # Semantic + keyword search
    API_SEARCH = "api_search"  # API documentation search
    STRUCTURED_SEARCH = "structured_search"  # Metadata-based search
    HIERARCHICAL_SEARCH = "hierarchical_search"  # Multi-level embedding search
    PROGRESSIVE_REFINEMENT = "progressive_refinement"  # Iterative search improvement
    MULTI_STRATEGY_FUSION = "multi_strategy_fusion"  # Combine multiple strategies
    TUTORIAL_PRIORITIZED_SEARCH = (
        "tutorial_prioritized_search"  # Learning-focused search
    )
    LANGUAGE_SPECIFIC_SEARCH = (
        "language_specific_search"  # Programming language filtered search
    )


@dataclass
class SearchResult:
    """Individual search result with enhanced metadata."""

    url: str
    title: str
    content: str
    similarity_score: float
    relevance_score: float  # Combined score incorporating multiple factors

    # Content metadata
    content_type: str
    programming_language: Optional[str]
    summary: str

    # Search metadata
    strategy_used: str
    embedding_model: str
    quality_score: float

    # Additional context
    matched_keywords: List[str]
    code_elements: List[str]
    api_references: List[str]

    # Ranking factors
    freshness_score: float = 1.0
    authority_score: float = 1.0
    completeness_score: float = 1.0


@dataclass
class SearchResponse:
    """Complete search response with results and metadata."""

    query: str
    query_analysis: QueryAnalysis
    results: List[SearchResult]
    total_results: int

    # Search execution metadata
    strategies_used: List[str]
    search_time_ms: int

    # Progressive refinement suggestions
    refinement_suggestions: List[str]
    expansion_suggestions: List[str]
    filter_suggestions: Dict[str, Any]

    # Quality indicators
    result_quality_score: float
    search_confidence: float


class SearchRouter:
    """Routes queries to optimal search strategies based on analysis."""

    def __init__(
        self,
        embedding_service: MultiEmbeddingService,
        database_manager: EnhancedDatabaseManager,
    ):
        """Initialize search router with required services."""
        self.embedding_service = embedding_service
        self.database_manager = database_manager
        self.query_analyzer = QueryAnalyzer()

        # Strategy priority mappings
        self.strategy_priorities = {
            QueryType.CODE_FUNCTION: [
                SearchStrategy.SEMANTIC_CODE_SEARCH,
                SearchStrategy.LANGUAGE_SPECIFIC_SEARCH,
                SearchStrategy.API_SEARCH,
                SearchStrategy.HYBRID_SEARCH,
            ],
            QueryType.CODE_CLASS: [
                SearchStrategy.SEMANTIC_CODE_SEARCH,
                SearchStrategy.STRUCTURED_SEARCH,
                SearchStrategy.LANGUAGE_SPECIFIC_SEARCH,
                SearchStrategy.HYBRID_SEARCH,
            ],
            QueryType.CODE_PATTERN: [
                SearchStrategy.SEMANTIC_CODE_SEARCH,
                SearchStrategy.HIERARCHICAL_SEARCH,
                SearchStrategy.HYBRID_SEARCH,
            ],
            QueryType.API_REFERENCE: [
                SearchStrategy.API_SEARCH,
                SearchStrategy.STRUCTURED_SEARCH,
                SearchStrategy.SEMANTIC_SEARCH,
            ],
            QueryType.CONCEPTUAL: [
                SearchStrategy.SEMANTIC_SEARCH,
                SearchStrategy.HIERARCHICAL_SEARCH,
                SearchStrategy.HYBRID_SEARCH,
            ],
            QueryType.TUTORIAL: [
                SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH,
                SearchStrategy.HIERARCHICAL_SEARCH,
                SearchStrategy.SEMANTIC_SEARCH,
            ],
            QueryType.TROUBLESHOOTING: [
                SearchStrategy.HYBRID_SEARCH,
                SearchStrategy.SEMANTIC_SEARCH,
                SearchStrategy.PROGRESSIVE_REFINEMENT,
            ],
            QueryType.GENERAL: [
                SearchStrategy.HYBRID_SEARCH,
                SearchStrategy.SEMANTIC_SEARCH,
            ],
        }

    def select_strategies(self, query_analysis: QueryAnalysis) -> List[SearchStrategy]:
        """Select optimal search strategies based on query analysis."""

        strategies = []
        base_strategies = self.strategy_priorities.get(
            query_analysis.query_type, [SearchStrategy.HYBRID_SEARCH]
        )

        # Add base strategies
        strategies.extend(base_strategies[:2])  # Top 2 strategies

        # Add complexity-based strategies
        if query_analysis.complexity_score > 0.7:
            if SearchStrategy.PROGRESSIVE_REFINEMENT not in strategies:
                strategies.append(SearchStrategy.PROGRESSIVE_REFINEMENT)

        if query_analysis.specificity_score < 0.4:
            if SearchStrategy.MULTI_STRATEGY_FUSION not in strategies:
                strategies.append(SearchStrategy.MULTI_STRATEGY_FUSION)

        # Intent-based additions
        if query_analysis.search_intent == SearchIntent.LEARNING:
            if SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH not in strategies:
                strategies.insert(0, SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH)

        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)

        logger.info(f"Selected strategies: {[s.value for s in unique_strategies]}")

        return unique_strategies


class SearchEngine:
    """Main search engine coordinating multiple search strategies."""

    def __init__(
        self,
        embedding_service: MultiEmbeddingService,
        database_manager: EnhancedDatabaseManager,
    ):
        """Initialize search engine with required components."""
        self.embedding_service = embedding_service
        self.database_manager = database_manager
        self.router = SearchRouter(embedding_service, database_manager)
        self.query_analyzer = QueryAnalyzer()

    async def search(
        self, query: str, limit: int = 10, filters: Dict[str, Any] = None
    ) -> SearchResponse:
        """Execute multi-modal search with intelligent strategy selection."""

        import time

        start_time = time.time()

        logger.info(f"Starting multi-modal search for: '{query}'")

        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Merge filters
        all_filters = {**(filters or {}), **query_analysis.filters}

        # Select search strategies
        strategies = self.router.select_strategies(query_analysis)

        # Execute search strategies
        all_results = []
        strategies_used = []

        for strategy in strategies:
            try:
                strategy_results = await self._execute_strategy(
                    strategy, query, query_analysis, limit, all_filters
                )
                all_results.extend(strategy_results)
                strategies_used.append(strategy.value)

            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue

        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query_analysis)

        # Limit results
        final_results = ranked_results[:limit]

        # Calculate search quality metrics
        result_quality_score = self._calculate_result_quality(
            final_results, query_analysis
        )
        search_confidence = self._calculate_search_confidence(
            query_analysis, len(final_results), strategies_used
        )

        # Generate refinement suggestions
        refinement_suggestions = self._generate_refinement_suggestions(
            query_analysis, final_results
        )
        expansion_suggestions = self._generate_expansion_suggestions(
            query_analysis, final_results
        )
        filter_suggestions = self._generate_filter_suggestions(
            query_analysis, final_results
        )

        search_time_ms = int((time.time() - start_time) * 1000)

        response = SearchResponse(
            query=query,
            query_analysis=query_analysis,
            results=final_results,
            total_results=len(unique_results),
            strategies_used=strategies_used,
            search_time_ms=search_time_ms,
            refinement_suggestions=refinement_suggestions,
            expansion_suggestions=expansion_suggestions,
            filter_suggestions=filter_suggestions,
            result_quality_score=result_quality_score,
            search_confidence=search_confidence,
        )

        logger.info(
            f"Search completed: {len(final_results)} results in {search_time_ms}ms"
        )

        return response

    async def _execute_strategy(
        self,
        strategy: SearchStrategy,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute a specific search strategy."""

        if strategy == SearchStrategy.SEMANTIC_SEARCH:
            return await self._semantic_search(query, query_analysis, limit, filters)
        elif strategy == SearchStrategy.SEMANTIC_CODE_SEARCH:
            return await self._semantic_code_search(
                query, query_analysis, limit, filters
            )
        elif strategy == SearchStrategy.HYBRID_SEARCH:
            return await self._hybrid_search(query, query_analysis, limit, filters)
        elif strategy == SearchStrategy.API_SEARCH:
            return await self._api_search(query, query_analysis, limit, filters)
        elif strategy == SearchStrategy.STRUCTURED_SEARCH:
            return await self._structured_search(query, query_analysis, limit, filters)
        elif strategy == SearchStrategy.HIERARCHICAL_SEARCH:
            return await self._hierarchical_search(
                query, query_analysis, limit, filters
            )
        elif strategy == SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH:
            return await self._tutorial_prioritized_search(
                query, query_analysis, limit, filters
            )
        elif strategy == SearchStrategy.LANGUAGE_SPECIFIC_SEARCH:
            return await self._language_specific_search(
                query, query_analysis, limit, filters
            )
        else:
            # Fallback to semantic search
            return await self._semantic_search(query, query_analysis, limit, filters)

    async def _semantic_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute semantic vector search."""

        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)

        if not query_embedding.get("success", False):
            return []

        # Search using database manager
        results = await self.database_manager.search_similar_content(
            embedding=query_embedding["embedding"],
            limit=limit * 2,  # Get more results for ranking
            filters=filters,
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_result = SearchResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                similarity_score=result.get("similarity", 0.0),
                relevance_score=result.get("similarity", 0.0),
                content_type=result.get("content_type", "general"),
                programming_language=result.get("primary_language"),
                summary=result.get("summary", ""),
                strategy_used=SearchStrategy.SEMANTIC_SEARCH.value,
                embedding_model=query_embedding.get("model", "unknown"),
                quality_score=result.get("quality_score", 0.8),
                matched_keywords=[],
                code_elements=[],
                api_references=[],
            )
            search_results.append(search_result)

        return search_results

    async def _semantic_code_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute code-specific semantic search."""

        # Force code-specific embedding model
        code_filters = {**filters, "min_code_percentage": 20}

        # Use code-specific model for embedding
        query_embedding = await self.embedding_service.embed_text(
            query, force_model=EmbeddingModel.COHERE_CODE
        )

        if not query_embedding.get("success", False):
            return []

        results = await self.database_manager.search_similar_content(
            embedding=query_embedding["embedding"],
            limit=limit * 2,
            filters=code_filters,
        )

        # Convert and boost code relevance
        search_results = []
        for result in results:
            relevance_boost = 1.0
            if result.get("content_type") in ["code_example", "api_reference"]:
                relevance_boost = 1.2

            search_result = SearchResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                similarity_score=result.get("similarity", 0.0),
                relevance_score=result.get("similarity", 0.0) * relevance_boost,
                content_type=result.get("content_type", "general"),
                programming_language=result.get("primary_language"),
                summary=result.get("summary", ""),
                strategy_used=SearchStrategy.SEMANTIC_CODE_SEARCH.value,
                embedding_model=query_embedding.get("model", "unknown"),
                quality_score=result.get("quality_score", 0.8),
                matched_keywords=[],
                code_elements=query_analysis.code_elements,
                api_references=query_analysis.api_references,
            )
            search_results.append(search_result)

        return search_results

    async def _hybrid_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute hybrid semantic + keyword search."""

        # Get semantic results
        semantic_results = await self._semantic_search(
            query, query_analysis, limit // 2, filters
        )

        # Get keyword-based results (using database text search if available)
        keyword_results = await self._keyword_search(
            query, query_analysis, limit // 2, filters
        )

        # Combine and return
        combined_results = semantic_results + keyword_results

        # Mark as hybrid strategy
        for result in combined_results:
            result.strategy_used = SearchStrategy.HYBRID_SEARCH.value
            # Boost relevance for exact keyword matches
            if any(
                keyword in result.content.lower() for keyword in query_analysis.keywords
            ):
                result.relevance_score *= 1.1

        return combined_results

    async def _keyword_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute keyword-based search."""

        # Use database manager's text search if available
        # This is a simplified implementation - in practice would use full-text search
        results = await self.database_manager.search_by_keywords(
            keywords=query_analysis.keywords, limit=limit, filters=filters
        )

        search_results = []
        for result in results:
            # Calculate keyword match score
            keyword_score = sum(
                1
                for keyword in query_analysis.keywords
                if keyword.lower() in result.get("content", "").lower()
            ) / max(len(query_analysis.keywords), 1)

            search_result = SearchResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                similarity_score=keyword_score,
                relevance_score=keyword_score,
                content_type=result.get("content_type", "general"),
                programming_language=result.get("primary_language"),
                summary=result.get("summary", ""),
                strategy_used="keyword_search",
                embedding_model="none",
                quality_score=result.get("quality_score", 0.7),
                matched_keywords=[
                    k
                    for k in query_analysis.keywords
                    if k.lower() in result.get("content", "").lower()
                ],
                code_elements=[],
                api_references=[],
            )
            search_results.append(search_result)

        return search_results

    async def _api_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute API-specific search."""

        api_filters = {**filters, "content_type": "api_reference"}

        return await self._semantic_search(query, query_analysis, limit, api_filters)

    async def _structured_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute structured metadata-based search."""

        # Search based on content analysis metadata
        structured_filters = filters.copy()

        if query_analysis.programming_language:
            structured_filters["primary_language"] = query_analysis.programming_language

        if query_analysis.code_elements:
            structured_filters["has_code_elements"] = True

        return await self._semantic_search(
            query, query_analysis, limit, structured_filters
        )

    async def _hierarchical_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute hierarchical multi-level search."""

        # Search both document and summary embeddings
        results = []

        # Document-level search
        doc_results = await self._semantic_search(
            query, query_analysis, limit // 2, filters
        )
        results.extend(doc_results)

        # Summary-level search (if available)
        summary_filters = {**filters, "embedding_type": "summary"}
        summary_results = await self._semantic_search(
            query, query_analysis, limit // 2, summary_filters
        )
        results.extend(summary_results)

        # Mark as hierarchical
        for result in results:
            result.strategy_used = SearchStrategy.HIERARCHICAL_SEARCH.value

        return results

    async def _tutorial_prioritized_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute tutorial-prioritized search for learning intent."""

        tutorial_filters = {**filters, "content_type": "tutorial"}

        results = await self._semantic_search(
            query, query_analysis, limit, tutorial_filters
        )

        # Boost tutorial content
        for result in results:
            result.strategy_used = SearchStrategy.TUTORIAL_PRIORITIZED_SEARCH.value
            if result.content_type == "tutorial":
                result.relevance_score *= 1.3

        return results

    async def _language_specific_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        limit: int,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Execute programming language-specific search."""

        if not query_analysis.programming_language:
            return []

        lang_filters = {
            **filters,
            "primary_language": query_analysis.programming_language,
        }

        results = await self._semantic_search(
            query, query_analysis, limit, lang_filters
        )

        for result in results:
            result.strategy_used = SearchStrategy.LANGUAGE_SPECIFIC_SEARCH.value

        return results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL."""

        seen_urls = set()
        unique_results = []

        # Sort by relevance score first to keep the best version
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)

        for result in sorted_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    def _rank_results(
        self, results: List[SearchResult], query_analysis: QueryAnalysis
    ) -> List[SearchResult]:
        """Rank results using multi-factor scoring."""

        for result in results:
            # Base relevance score
            final_score = result.relevance_score

            # Quality boost
            final_score *= 0.8 + 0.2 * result.quality_score

            # Content type alignment boost
            if self._content_type_matches_query(
                result.content_type, query_analysis.query_type
            ):
                final_score *= 1.15

            # Language alignment boost
            if (
                query_analysis.programming_language
                and result.programming_language == query_analysis.programming_language
            ):
                final_score *= 1.1

            # Keyword match boost
            keyword_matches = len(result.matched_keywords)
            if keyword_matches > 0:
                final_score *= 1.0 + 0.05 * keyword_matches

            # Code element match boost
            if query_analysis.code_elements and result.code_elements:
                final_score *= 1.1

            # Update relevance score
            result.relevance_score = final_score

        # Sort by final relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _content_type_matches_query(
        self, content_type: str, query_type: QueryType
    ) -> bool:
        """Check if content type aligns with query type."""

        alignment_map = {
            QueryType.CODE_FUNCTION: ["code_example", "api_reference"],
            QueryType.CODE_CLASS: ["code_example", "api_reference"],
            QueryType.CODE_PATTERN: ["code_example"],
            QueryType.API_REFERENCE: ["api_reference"],
            QueryType.TUTORIAL: ["tutorial"],
            QueryType.CONCEPTUAL: ["tutorial", "general"],
        }

        aligned_types = alignment_map.get(query_type, [])
        return content_type in aligned_types

    def _calculate_result_quality(
        self, results: List[SearchResult], query_analysis: QueryAnalysis
    ) -> float:
        """Calculate overall result quality score."""

        if not results:
            return 0.0

        # Average relevance score
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        # Average quality score
        avg_quality = sum(r.quality_score for r in results) / len(results)

        # Content type diversity
        content_types = set(r.content_type for r in results)
        diversity_bonus = min(len(content_types) / 3.0, 0.1)

        # Combine factors
        quality_score = avg_relevance * 0.5 + avg_quality * 0.4 + diversity_bonus

        return min(quality_score, 1.0)

    def _calculate_search_confidence(
        self,
        query_analysis: QueryAnalysis,
        result_count: int,
        strategies_used: List[str],
    ) -> float:
        """Calculate confidence in search results."""

        confidence = query_analysis.confidence

        # Result count factor
        if result_count >= 5:
            confidence *= 1.1
        elif result_count < 2:
            confidence *= 0.8

        # Strategy diversity factor
        if len(strategies_used) > 1:
            confidence *= 1.05

        return min(confidence, 1.0)

    def _generate_refinement_suggestions(
        self, query_analysis: QueryAnalysis, results: List[SearchResult]
    ) -> List[str]:
        """Generate suggestions for query refinement."""

        suggestions = []

        if query_analysis.specificity_score < 0.4:
            suggestions.append("Try adding more specific terms or function names")

        if query_analysis.programming_language is None and any(
            r.programming_language for r in results
        ):
            suggestions.append(
                "Specify a programming language to get more targeted results"
            )

        if len(results) < 3:
            suggestions.append("Try broader terms or remove specific constraints")

        return suggestions

    def _generate_expansion_suggestions(
        self, query_analysis: QueryAnalysis, results: List[SearchResult]
    ) -> List[str]:
        """Generate query expansion suggestions."""

        return query_analysis.expansion_terms[:5]

    def _generate_filter_suggestions(
        self, query_analysis: QueryAnalysis, results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Generate filter suggestions based on results."""

        suggestions = {}

        # Language filters
        languages = set(
            r.programming_language for r in results if r.programming_language
        )
        if languages:
            suggestions["programming_languages"] = list(languages)

        # Content type filters
        content_types = set(r.content_type for r in results)
        if len(content_types) > 1:
            suggestions["content_types"] = list(content_types)

        return suggestions
