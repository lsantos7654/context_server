"""Advanced search endpoints with enhanced metadata APIs."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from .content_analysis import ContentAnalysis
from .enhanced_storage import EnhancedDatabaseManager
from .llm_endpoints import ContextRecommendation, LLMOptimizedEndpoints
from .multi_modal_search import SearchEngine, SearchResponse
from .relationship_mapping import KnowledgeGraphBuilder, TopicCluster

logger = logging.getLogger(__name__)


@dataclass
class SearchMetadata:
    """Metadata about search results and content."""

    total_content_pieces: int
    content_type_distribution: Dict[str, int]
    programming_language_distribution: Dict[str, int]
    quality_score_distribution: Dict[str, int]  # ranges: "high", "medium", "low"
    embedding_model_distribution: Dict[str, int]
    average_content_quality: float
    search_coverage_percentage: float  # % of total content that matched
    cluster_coverage: Dict[str, int]  # cluster_id -> content_count


@dataclass
class AdvancedSearchResponse:
    """Enhanced search response with metadata and recommendations."""

    # Core search results
    search_response: SearchResponse
    search_metadata: SearchMetadata

    # Enhanced features
    related_clusters: List[TopicCluster]
    context_recommendations: Optional[ContextRecommendation]
    knowledge_graph_insights: Dict[str, Any]

    # Performance metrics
    total_search_time_ms: int
    cache_hit_ratio: float
    search_strategy_effectiveness: Dict[str, float]


@dataclass
class ContentMetadata:
    """Detailed metadata about content in a context."""

    url: str
    title: str
    content_type: str
    programming_language: Optional[str]
    quality_score: float
    readability_score: float
    complexity_indicators: List[str]
    topic_keywords: List[str]
    code_elements: List[str]
    api_references: List[str]
    embedding_model: str
    last_updated: datetime
    relationship_count: int  # Number of relationships this content has
    cluster_memberships: List[str]  # Cluster IDs this content belongs to


@dataclass
class ContextAnalytics:
    """Comprehensive analytics about a context."""

    context_id: str
    total_content_pieces: int
    content_breakdown: Dict[str, int]  # content_type -> count
    language_breakdown: Dict[str, int]  # language -> count

    # Quality metrics
    average_quality_score: float
    quality_distribution: Dict[str, int]
    average_readability_score: float

    # Knowledge graph metrics
    total_relationships: int
    relationship_type_breakdown: Dict[str, int]
    total_clusters: int
    cluster_size_distribution: Dict[str, int]  # size_range -> count
    graph_density: float
    modularity_score: float

    # Search performance metrics
    most_searched_topics: List[Tuple[str, int]]  # (topic, search_count)
    search_success_rate: float
    average_result_relevance: float

    # Temporal insights
    content_creation_timeline: Dict[str, int]  # date -> content_count
    search_activity_timeline: Dict[str, int]  # date -> search_count
    last_analysis_update: datetime


class AdvancedSearchAPI:
    """Advanced search API with enhanced metadata and analytics."""

    def __init__(
        self,
        search_engine: SearchEngine,
        database_manager: EnhancedDatabaseManager,
        llm_endpoints: LLMOptimizedEndpoints,
        knowledge_graph_builder: KnowledgeGraphBuilder,
    ):
        self.search_engine = search_engine
        self.database_manager = database_manager
        self.llm_endpoints = llm_endpoints
        self.knowledge_graph_builder = knowledge_graph_builder

        # Performance tracking
        self.search_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    async def enhanced_search(
        self,
        query: str,
        context_id: str,
        limit: int = 20,
        include_recommendations: bool = True,
        include_clusters: bool = True,
        include_graph_insights: bool = True,
        enable_caching: bool = True,
    ) -> AdvancedSearchResponse:
        """
        Perform enhanced search with metadata, recommendations, and insights.

        Args:
            query: Search query
            context_id: Context to search in
            limit: Maximum number of results
            include_recommendations: Whether to include context recommendations
            include_clusters: Whether to include related clusters
            include_graph_insights: Whether to include knowledge graph insights
            enable_caching: Whether to use caching for performance

        Returns:
            AdvancedSearchResponse with comprehensive search data
        """
        start_time = asyncio.get_event_loop().time()

        # Check cache first
        cache_key = f"{query}:{context_id}:{limit}"
        if enable_caching and cache_key in self.search_cache:
            self.cache_hit_count += 1
            cached_response = self.search_cache[cache_key]
            cached_response.cache_hit_ratio = self._calculate_cache_hit_ratio()
            return cached_response

        self.cache_miss_count += 1

        # Perform core search
        search_response = await self.search_engine.search(
            query=query, context_id=context_id, limit=limit
        )

        # Generate search metadata
        search_metadata = await self._generate_search_metadata(
            search_response, context_id
        )

        # Get related clusters if requested
        related_clusters = []
        if include_clusters:
            related_clusters = await self._find_related_clusters(
                search_response, context_id
            )

        # Get context recommendations if requested
        context_recommendations = None
        if include_recommendations:
            context_recommendations = await self.llm_endpoints.recommend_context(
                query, context_id, max_recommendations=5
            )

        # Generate knowledge graph insights if requested
        knowledge_graph_insights = {}
        if include_graph_insights:
            knowledge_graph_insights = await self._generate_graph_insights(
                search_response, context_id
            )

        # Calculate performance metrics
        end_time = asyncio.get_event_loop().time()
        total_search_time_ms = int((end_time - start_time) * 1000)

        # Create enhanced response
        enhanced_response = AdvancedSearchResponse(
            search_response=search_response,
            search_metadata=search_metadata,
            related_clusters=related_clusters,
            context_recommendations=context_recommendations,
            knowledge_graph_insights=knowledge_graph_insights,
            total_search_time_ms=total_search_time_ms,
            cache_hit_ratio=self._calculate_cache_hit_ratio(),
            search_strategy_effectiveness=self._calculate_strategy_effectiveness(
                search_response
            ),
        )

        # Cache the response
        if enable_caching:
            self.search_cache[cache_key] = enhanced_response

        return enhanced_response

    async def get_content_metadata(
        self, context_id: str, url: Optional[str] = None
    ) -> Union[ContentMetadata, List[ContentMetadata]]:
        """
        Get detailed metadata for content in a context.

        Args:
            context_id: Context to analyze
            url: Specific URL to get metadata for (optional)

        Returns:
            ContentMetadata or list of ContentMetadata objects
        """
        if url:
            # Get metadata for specific content
            content_analysis = await self.database_manager.get_content_analysis(
                context_id, url
            )
            if not content_analysis:
                raise ValueError(f"Content not found: {url}")

            return await self._build_content_metadata(content_analysis, context_id)
        else:
            # Get metadata for all content in context
            all_analyses = await self.database_manager.get_all_content_analyses(
                context_id
            )
            metadata_list = []

            for analysis in all_analyses:
                metadata = await self._build_content_metadata(analysis, context_id)
                metadata_list.append(metadata)

            return metadata_list

    async def get_context_analytics(self, context_id: str) -> ContextAnalytics:
        """
        Get comprehensive analytics for a context.

        Args:
            context_id: Context to analyze

        Returns:
            ContextAnalytics with detailed insights
        """
        # Get all content analyses
        all_analyses = await self.database_manager.get_all_content_analyses(context_id)

        if not all_analyses:
            raise ValueError(f"No content found in context: {context_id}")

        # Calculate content breakdown
        content_breakdown = {}
        language_breakdown = {}
        quality_scores = []
        readability_scores = []

        for analysis in all_analyses:
            # Content type breakdown
            content_type = getattr(analysis, "content_type", "general")
            content_breakdown[content_type] = content_breakdown.get(content_type, 0) + 1

            # Language breakdown
            if analysis.primary_language:
                language_breakdown[analysis.primary_language] = (
                    language_breakdown.get(analysis.primary_language, 0) + 1
                )

            # Quality and readability scores
            quality_scores.append(
                getattr(analysis, "quality_score", analysis.readability_score)
            )
            readability_scores.append(analysis.readability_score)

        # Calculate quality distribution
        quality_distribution = {"high": 0, "medium": 0, "low": 0}
        for score in quality_scores:
            if score >= 0.8:
                quality_distribution["high"] += 1
            elif score >= 0.5:
                quality_distribution["medium"] += 1
            else:
                quality_distribution["low"] += 1

        # Get knowledge graph metrics
        knowledge_graph = await self.knowledge_graph_builder.load_knowledge_graph()

        # Get relationships for this context
        relationships = await self.database_manager.load_content_relationships()
        context_relationships = [
            r
            for r in relationships
            if any(
                url in [a.url for a in all_analyses]
                for url in [r.source_url, r.target_url]
            )
        ]

        # Get clusters for this context
        clusters = await self.database_manager.load_topic_clusters()
        context_clusters = [
            c
            for c in clusters
            if any(url in [a.url for a in all_analyses] for url in c.content_urls)
        ]

        # Relationship type breakdown
        relationship_type_breakdown = {}
        for rel in context_relationships:
            rel_type = rel.relationship_type.value
            relationship_type_breakdown[rel_type] = (
                relationship_type_breakdown.get(rel_type, 0) + 1
            )

        # Cluster size distribution
        cluster_size_distribution = {"small": 0, "medium": 0, "large": 0}
        for cluster in context_clusters:
            size = len(cluster.content_urls)
            if size <= 3:
                cluster_size_distribution["small"] += 1
            elif size <= 10:
                cluster_size_distribution["medium"] += 1
            else:
                cluster_size_distribution["large"] += 1

        # Search performance metrics (mock data for now)
        most_searched_topics = [
            ("python", 45),
            ("javascript", 32),
            ("api", 28),
            ("functions", 22),
            ("algorithms", 18),
        ]

        return ContextAnalytics(
            context_id=context_id,
            total_content_pieces=len(all_analyses),
            content_breakdown=content_breakdown,
            language_breakdown=language_breakdown,
            average_quality_score=(
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            ),
            quality_distribution=quality_distribution,
            average_readability_score=(
                sum(readability_scores) / len(readability_scores)
                if readability_scores
                else 0.0
            ),
            total_relationships=len(context_relationships),
            relationship_type_breakdown=relationship_type_breakdown,
            total_clusters=len(context_clusters),
            cluster_size_distribution=cluster_size_distribution,
            graph_density=knowledge_graph.graph_density if knowledge_graph else 0.0,
            modularity_score=(
                knowledge_graph.modularity_score if knowledge_graph else 0.0
            ),
            most_searched_topics=most_searched_topics,
            search_success_rate=0.85,  # Mock data
            average_result_relevance=0.78,  # Mock data
            content_creation_timeline={},  # Would be populated from database timestamps
            search_activity_timeline={},  # Would be populated from search logs
            last_analysis_update=datetime.now(),
        )

    async def search_similar_content(
        self, url: str, context_id: str, limit: int = 10
    ) -> AdvancedSearchResponse:
        """
        Find content similar to a specific URL.

        Args:
            url: URL to find similar content for
            context_id: Context to search in
            limit: Maximum number of results

        Returns:
            AdvancedSearchResponse with similar content
        """
        # Get the content analysis for the given URL
        content_analysis = await self.database_manager.get_content_analysis(
            context_id, url
        )
        if not content_analysis:
            raise ValueError(f"Content not found: {url}")

        # Use the content's summary and keywords to build a similarity query
        query_parts = []
        if hasattr(content_analysis, "summary") and content_analysis.summary:
            query_parts.append(content_analysis.summary)
        if (
            hasattr(content_analysis, "topic_keywords")
            and content_analysis.topic_keywords
        ):
            query_parts.extend(content_analysis.topic_keywords[:5])  # Top 5 keywords

        similarity_query = " ".join(query_parts)

        # Perform enhanced search
        return await self.enhanced_search(
            query=similarity_query,
            context_id=context_id,
            limit=limit,
            include_recommendations=False,  # Not needed for similarity search
        )

    async def get_trending_topics(
        self, context_id: str, time_window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics in a context based on search activity and content updates.

        Args:
            context_id: Context to analyze
            time_window_days: Number of days to look back

        Returns:
            List of trending topic information
        """
        # Get recent content analyses
        all_analyses = await self.database_manager.get_all_content_analyses(context_id)

        # Analyze topic frequency
        topic_frequency = {}
        language_frequency = {}

        for analysis in all_analyses:
            # Count topic keywords
            if hasattr(analysis, "topic_keywords") and analysis.topic_keywords:
                for keyword in analysis.topic_keywords:
                    topic_frequency[keyword] = topic_frequency.get(keyword, 0) + 1

            # Count programming languages
            if analysis.primary_language:
                language_frequency[analysis.primary_language] = (
                    language_frequency.get(analysis.primary_language, 0) + 1
                )

        # Sort by frequency and create trending topics
        trending_topics = []

        # Add topic keywords
        for topic, count in sorted(
            topic_frequency.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            trending_topics.append(
                {
                    "topic": topic,
                    "type": "keyword",
                    "frequency": count,
                    "trend_score": count / len(all_analyses),
                    "related_content_count": count,
                }
            )

        # Add programming languages
        for language, count in sorted(
            language_frequency.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            trending_topics.append(
                {
                    "topic": language,
                    "type": "programming_language",
                    "frequency": count,
                    "trend_score": count / len(all_analyses),
                    "related_content_count": count,
                }
            )

        return trending_topics

    async def _generate_search_metadata(
        self, search_response: SearchResponse, context_id: str
    ) -> SearchMetadata:
        """Generate metadata about search results."""
        if not search_response.results:
            return SearchMetadata(
                total_content_pieces=0,
                content_type_distribution={},
                programming_language_distribution={},
                quality_score_distribution={"high": 0, "medium": 0, "low": 0},
                embedding_model_distribution={},
                average_content_quality=0.0,
                search_coverage_percentage=0.0,
                cluster_coverage={},
            )

        # Analyze search results
        content_types = {}
        languages = {}
        quality_distribution = {"high": 0, "medium": 0, "low": 0}
        embedding_models = {}
        quality_scores = []

        for result in search_response.results:
            # Content type distribution
            content_type = result.content_type or "general"
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Programming language distribution
            if result.programming_language:
                languages[result.programming_language] = (
                    languages.get(result.programming_language, 0) + 1
                )

            # Quality score distribution
            quality_score = result.quality_score
            quality_scores.append(quality_score)
            if quality_score >= 0.8:
                quality_distribution["high"] += 1
            elif quality_score >= 0.5:
                quality_distribution["medium"] += 1
            else:
                quality_distribution["low"] += 1

            # Embedding model distribution
            model = result.embedding_model or "unknown"
            embedding_models[model] = embedding_models.get(model, 0) + 1

        # Calculate coverage percentage
        total_content = await self.database_manager.get_context_stats(context_id)
        total_count = total_content.get("total_chunks", 1)
        coverage_percentage = (len(search_response.results) / total_count) * 100

        return SearchMetadata(
            total_content_pieces=len(search_response.results),
            content_type_distribution=content_types,
            programming_language_distribution=languages,
            quality_score_distribution=quality_distribution,
            embedding_model_distribution=embedding_models,
            average_content_quality=(
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            ),
            search_coverage_percentage=min(coverage_percentage, 100.0),
            cluster_coverage={},  # Would be populated with actual cluster data
        )

    async def _find_related_clusters(
        self, search_response: SearchResponse, context_id: str
    ) -> List[TopicCluster]:
        """Find clusters related to search results."""
        # Get all clusters
        all_clusters = await self.database_manager.load_topic_clusters()

        # Find clusters that contain URLs from search results
        result_urls = {result.url for result in search_response.results}
        related_clusters = []

        for cluster in all_clusters:
            # Check if cluster contains any of the result URLs
            if any(url in result_urls for url in cluster.content_urls):
                related_clusters.append(cluster)

        # Sort by relevance (clusters with more matching URLs first)
        related_clusters.sort(
            key=lambda c: len(set(c.content_urls) & result_urls), reverse=True
        )

        return related_clusters[:5]  # Return top 5 related clusters

    async def _generate_graph_insights(
        self, search_response: SearchResponse, context_id: str
    ) -> Dict[str, Any]:
        """Generate knowledge graph insights based on search results."""
        result_urls = {result.url for result in search_response.results}

        # Get relationships involving search result URLs
        all_relationships = await self.database_manager.load_content_relationships()
        relevant_relationships = [
            r
            for r in all_relationships
            if r.source_url in result_urls or r.target_url in result_urls
        ]

        # Analyze relationship patterns
        relationship_types = {}
        for rel in relevant_relationships:
            rel_type = rel.relationship_type.value
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        # Find most connected content
        connection_counts = {}
        for rel in relevant_relationships:
            connection_counts[rel.source_url] = (
                connection_counts.get(rel.source_url, 0) + 1
            )
            connection_counts[rel.target_url] = (
                connection_counts.get(rel.target_url, 0) + 1
            )

        most_connected = sorted(
            connection_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        return {
            "total_relationships": len(relevant_relationships),
            "relationship_type_distribution": relationship_types,
            "most_connected_content": most_connected,
            "average_relationship_strength": (
                sum(r.strength for r in relevant_relationships)
                / len(relevant_relationships)
                if relevant_relationships
                else 0.0
            ),
            "graph_connectivity": (
                len(relevant_relationships) / len(result_urls) if result_urls else 0.0
            ),
        }

    async def _build_content_metadata(
        self, content_analysis: ContentAnalysis, context_id: str
    ) -> ContentMetadata:
        """Build detailed metadata for a content piece."""
        # Get relationship count for this content
        all_relationships = await self.database_manager.load_content_relationships()
        relationship_count = sum(
            1
            for r in all_relationships
            if r.source_url == content_analysis.url
            or r.target_url == content_analysis.url
        )

        # Get cluster memberships
        all_clusters = await self.database_manager.load_topic_clusters()
        cluster_memberships = [
            cluster.cluster_id
            for cluster in all_clusters
            if content_analysis.url in cluster.content_urls
        ]

        return ContentMetadata(
            url=content_analysis.url,
            title=getattr(content_analysis, "title", ""),
            content_type=getattr(content_analysis, "content_type", "general"),
            programming_language=content_analysis.primary_language,
            quality_score=getattr(
                content_analysis, "quality_score", content_analysis.readability_score
            ),
            readability_score=content_analysis.readability_score,
            complexity_indicators=content_analysis.complexity_indicators,
            topic_keywords=getattr(content_analysis, "topic_keywords", []),
            code_elements=getattr(content_analysis, "code_elements", []),
            api_references=getattr(content_analysis, "api_references", []),
            embedding_model="text-embedding-3-small",  # Default model
            last_updated=datetime.now(),  # Would come from database
            relationship_count=relationship_count,
            cluster_memberships=cluster_memberships,
        )

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        return self.cache_hit_count / total_requests

    def _calculate_strategy_effectiveness(
        self, search_response: SearchResponse
    ) -> Dict[str, float]:
        """Calculate effectiveness of different search strategies."""
        strategies = {}
        total_results = len(search_response.results)

        if total_results == 0:
            return {}

        # Count results by strategy
        strategy_counts = {}
        for result in search_response.results:
            strategy = result.strategy_used
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Calculate effectiveness as percentage of total results
        for strategy, count in strategy_counts.items():
            strategies[strategy] = count / total_results

        return strategies
