"""Content relationship mapping and topic clustering system."""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .content_analysis import ContentAnalysis
from .enhanced_storage import EnhancedDatabaseManager
from .multi_embedding_service import MultiEmbeddingService

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between content pieces."""

    SEMANTIC_SIMILARITY = "semantic_similarity"  # Similar topics/concepts
    CODE_DEPENDENCY = "code_dependency"  # Code imports/dependencies
    API_REFERENCE = "api_reference"  # API usage relationships
    CONCEPTUAL_HIERARCHY = "conceptual_hierarchy"  # Parent-child concepts
    TUTORIAL_SEQUENCE = "tutorial_sequence"  # Learning progression
    CROSS_REFERENCE = "cross_reference"  # Explicit references
    LANGUAGE_VARIANT = "language_variant"  # Same concept, different languages
    VERSION_EVOLUTION = "version_evolution"  # Different versions of same content


@dataclass
class ContentRelationship:
    """Represents a relationship between two content pieces."""

    source_url: str
    target_url: str
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Relationship metadata
    discovered_method: str  # How the relationship was found
    supporting_evidence: List[str]  # Evidence for the relationship
    context_elements: List[str]  # Shared elements that support the relationship

    # Temporal information
    discovery_timestamp: float
    last_validated: Optional[float] = None


@dataclass
class TopicCluster:
    """Represents a cluster of related content around a topic."""

    cluster_id: str
    name: str
    description: str
    content_urls: List[str]

    # Cluster characteristics
    topic_keywords: List[str]
    programming_languages: List[str]
    content_types: List[str]
    difficulty_level: Optional[str]  # beginner, intermediate, advanced

    # Cluster metrics
    coherence_score: float  # How well content fits together
    coverage_score: float  # How comprehensive the cluster is
    quality_score: float  # Average quality of content in cluster

    # Cluster relationships
    related_clusters: List[str]  # IDs of related clusters
    parent_cluster: Optional[str]  # Hierarchical parent
    child_clusters: List[str]  # Hierarchical children


@dataclass
class KnowledgeGraph:
    """Complete knowledge graph of content relationships and clusters."""

    relationships: List[ContentRelationship]
    clusters: List[TopicCluster]

    # Graph statistics
    total_nodes: int  # Total content pieces
    total_edges: int  # Total relationships
    cluster_count: int
    average_node_degree: float

    # Quality metrics
    graph_density: float
    modularity_score: float  # How well-clustered the graph is
    coverage_ratio: float  # Percentage of content that's clustered


class RelationshipDetector:
    """Detects various types of relationships between content pieces."""

    def __init__(self, embedding_service: MultiEmbeddingService):
        """Initialize relationship detector with embedding service."""
        self.embedding_service = embedding_service

        # Detection thresholds
        self.semantic_similarity_threshold = 0.7
        self.code_dependency_threshold = 0.6
        self.api_reference_threshold = 0.8
        self.cross_reference_threshold = 0.9

    async def detect_relationships(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect all types of relationships between content pieces."""

        logger.info(
            f"Detecting relationships among {len(content_analyses)} content pieces"
        )

        relationships = []

        # Create lookup maps for efficient processing
        url_to_analysis = {analysis.url: analysis for analysis in content_analyses}

        # Detect different types of relationships
        semantic_rels = await self._detect_semantic_relationships(content_analyses)
        code_rels = self._detect_code_dependencies(content_analyses)
        api_rels = self._detect_api_references(content_analyses)
        concept_rels = self._detect_conceptual_hierarchy(content_analyses)
        tutorial_rels = self._detect_tutorial_sequences(content_analyses)
        cross_rels = self._detect_cross_references(content_analyses)
        language_rels = self._detect_language_variants(content_analyses)
        version_rels = self._detect_version_evolution(content_analyses)

        # Combine all relationships
        all_relationships = [
            *semantic_rels,
            *code_rels,
            *api_rels,
            *concept_rels,
            *tutorial_rels,
            *cross_rels,
            *language_rels,
            *version_rels,
        ]

        # Deduplicate and validate relationships
        validated_relationships = self._validate_and_deduplicate(all_relationships)

        logger.info(f"Detected {len(validated_relationships)} relationships")

        return validated_relationships

    async def _detect_semantic_relationships(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect semantic similarity relationships using embeddings."""

        relationships = []

        if len(content_analyses) < 2:
            return relationships

        # Get embeddings for all content
        embeddings_map = {}
        for analysis in content_analyses:
            if analysis.embedding:
                embeddings_map[analysis.url] = np.array(analysis.embedding)

        if len(embeddings_map) < 2:
            return relationships

        # Calculate pairwise similarities
        urls = list(embeddings_map.keys())
        embeddings = list(embeddings_map.values())

        similarity_matrix = cosine_similarity(embeddings)

        for i, url1 in enumerate(urls):
            for j, url2 in enumerate(urls):
                if i >= j:  # Avoid duplicates and self-comparisons
                    continue

                similarity = similarity_matrix[i][j]

                if similarity >= self.semantic_similarity_threshold:
                    analysis1 = next(a for a in content_analyses if a.url == url1)
                    analysis2 = next(a for a in content_analyses if a.url == url2)

                    # Find supporting evidence
                    evidence = self._find_semantic_evidence(analysis1, analysis2)
                    context_elements = self._find_shared_elements(analysis1, analysis2)

                    relationship = ContentRelationship(
                        source_url=url1,
                        target_url=url2,
                        relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                        strength=similarity,
                        confidence=min(similarity * 1.2, 1.0),
                        discovered_method="embedding_similarity",
                        supporting_evidence=evidence,
                        context_elements=context_elements,
                        discovery_timestamp=asyncio.get_event_loop().time(),
                    )
                    relationships.append(relationship)

        return relationships

    def _detect_code_dependencies(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect code dependency relationships (imports, function calls, etc.)."""

        relationships = []

        # Create maps of code elements
        import_map = defaultdict(list)  # module -> [urls that import it]
        function_map = defaultdict(list)  # function -> [urls that define it]
        class_map = defaultdict(list)  # class -> [urls that define it]

        for analysis in content_analyses:
            if analysis.code_elements:
                for element in analysis.code_elements:
                    element_lower = element.lower()
                    if any(
                        keyword in element_lower
                        for keyword in ["import", "require", "include"]
                    ):
                        import_map[element].append(analysis.url)
                    elif any(
                        keyword in element_lower
                        for keyword in ["def ", "function ", "func "]
                    ):
                        function_map[element].append(analysis.url)
                    elif any(
                        keyword in element_lower
                        for keyword in ["class ", "struct ", "interface "]
                    ):
                        class_map[element].append(analysis.url)

        # Find dependency relationships
        for analysis in content_analyses:
            if not analysis.code_elements:
                continue

            for element in analysis.code_elements:
                element_lower = element.lower()

                # Check if this content uses functions/classes defined elsewhere
                for func_def, defining_urls in function_map.items():
                    if analysis.url in defining_urls:
                        continue  # Skip self-definitions

                    # Extract function name from definition
                    func_def_lower = func_def.lower()
                    # Look for function name in function calls or references
                    if "def " in func_def_lower:
                        # Extract function name from "def add(a, b)" -> "add"
                        import re

                        match = re.search(r"def\s+(\w+)", func_def_lower)
                        if match:
                            func_name = match.group(1)
                            # Check if this function is called in current element
                            if func_name in element_lower and (
                                "(" in element_lower
                                or "call" in element_lower
                                or func_name + "(" in element_lower
                            ):
                                for defining_url in defining_urls:
                                    relationship = ContentRelationship(
                                        source_url=analysis.url,
                                        target_url=defining_url,
                                        relationship_type=RelationshipType.CODE_DEPENDENCY,
                                        strength=self._calculate_dependency_strength(
                                            element, func_def
                                        ),
                                        confidence=0.8,
                                        discovered_method="code_element_analysis",
                                        supporting_evidence=[
                                            f"Uses function: {func_name} defined in {func_def}"
                                        ],
                                        context_elements=[element, func_def],
                                        discovery_timestamp=asyncio.get_event_loop().time(),
                                    )
                                    relationships.append(relationship)

        return relationships

    def _detect_api_references(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect API reference relationships."""

        relationships = []

        # Create API endpoint map
        api_map = defaultdict(list)

        for analysis in content_analyses:
            if analysis.api_references:
                for api_ref in analysis.api_references:
                    api_map[api_ref].append(analysis.url)

        # Find API usage relationships
        for api_endpoint, urls in api_map.items():
            if len(urls) > 1:
                # Multiple pieces of content reference the same API
                for i, url1 in enumerate(urls):
                    for url2 in urls[i + 1 :]:
                        relationship = ContentRelationship(
                            source_url=url1,
                            target_url=url2,
                            relationship_type=RelationshipType.API_REFERENCE,
                            strength=0.9,
                            confidence=0.95,
                            discovered_method="api_reference_analysis",
                            supporting_evidence=[f"Both reference API: {api_endpoint}"],
                            context_elements=[api_endpoint],
                            discovery_timestamp=asyncio.get_event_loop().time(),
                        )
                        relationships.append(relationship)

        return relationships

    def _detect_conceptual_hierarchy(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect hierarchical relationships between concepts."""

        relationships = []

        # Hierarchy indicators
        parent_indicators = [
            "overview",
            "introduction",
            "basics",
            "fundamentals",
            "guide",
        ]
        child_indicators = [
            "advanced",
            "specific",
            "detailed",
            "example",
            "implementation",
        ]

        for analysis1 in content_analyses:
            title1 = analysis1.title.lower() if analysis1.title else ""
            summary1 = analysis1.summary.lower() if analysis1.summary else ""
            content1 = f"{title1} {summary1}"

            for analysis2 in content_analyses:
                if analysis1.url == analysis2.url:
                    continue

                title2 = analysis2.title.lower() if analysis2.title else ""
                summary2 = analysis2.summary.lower() if analysis2.summary else ""
                content2 = f"{title2} {summary2}"

                # Check for parent-child relationship
                is_parent = any(
                    indicator in content1 for indicator in parent_indicators
                )
                is_child = any(indicator in content2 for indicator in child_indicators)

                # Check for shared topic keywords
                shared_keywords = set(analysis1.topic_keywords or []) & set(
                    analysis2.topic_keywords or []
                )

                if is_parent and is_child and shared_keywords:
                    relationship = ContentRelationship(
                        source_url=analysis1.url,  # Parent
                        target_url=analysis2.url,  # Child
                        relationship_type=RelationshipType.CONCEPTUAL_HIERARCHY,
                        strength=len(shared_keywords)
                        / max(len(analysis1.topic_keywords or []), 1),
                        confidence=0.7,
                        discovered_method="conceptual_hierarchy_analysis",
                        supporting_evidence=[
                            f"Shared keywords: {list(shared_keywords)}"
                        ],
                        context_elements=list(shared_keywords),
                        discovery_timestamp=asyncio.get_event_loop().time(),
                    )
                    relationships.append(relationship)

        return relationships

    def _detect_tutorial_sequences(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect tutorial sequence relationships."""

        relationships = []

        # Tutorial sequence indicators
        sequence_patterns = [
            ["part 1", "part 2", "part 3"],
            ["chapter 1", "chapter 2", "chapter 3"],
            ["step 1", "step 2", "step 3"],
            ["lesson 1", "lesson 2", "lesson 3"],
            ["basic", "intermediate", "advanced"],
            ["introduction", "getting started", "examples"],
        ]

        for pattern in sequence_patterns:
            for i, keyword1 in enumerate(pattern[:-1]):
                keyword2 = pattern[i + 1]

                # Find content matching sequential keywords
                content1_matches = [
                    a
                    for a in content_analyses
                    if keyword1 in (a.title or "").lower()
                    or keyword1 in (a.summary or "").lower()
                ]
                content2_matches = [
                    a
                    for a in content_analyses
                    if keyword2 in (a.title or "").lower()
                    or keyword2 in (a.summary or "").lower()
                ]

                # Create sequence relationships
                for analysis1 in content1_matches:
                    for analysis2 in content2_matches:
                        # Check if they share topic keywords (same subject)
                        shared_keywords = set(analysis1.topic_keywords or []) & set(
                            analysis2.topic_keywords or []
                        )

                        if shared_keywords:
                            relationship = ContentRelationship(
                                source_url=analysis1.url,
                                target_url=analysis2.url,
                                relationship_type=RelationshipType.TUTORIAL_SEQUENCE,
                                strength=0.8,
                                confidence=0.75,
                                discovered_method="tutorial_sequence_analysis",
                                supporting_evidence=[
                                    f"Sequential: {keyword1} -> {keyword2}"
                                ],
                                context_elements=[keyword1, keyword2],
                                discovery_timestamp=asyncio.get_event_loop().time(),
                            )
                            relationships.append(relationship)

        return relationships

    def _detect_cross_references(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect explicit cross-references between content."""

        relationships = []

        # Look for URL mentions in content
        for analysis1 in content_analyses:
            content_text = (analysis1.raw_content or "").lower()

            for analysis2 in content_analyses:
                if analysis1.url == analysis2.url:
                    continue

                # Check if analysis2's URL or title is mentioned in analysis1's content
                url_mentioned = analysis2.url.lower() in content_text
                title_mentioned = (
                    analysis2.title and analysis2.title.lower() in content_text
                )

                if url_mentioned or title_mentioned:
                    evidence = []
                    if url_mentioned:
                        evidence.append(f"URL mentioned: {analysis2.url}")
                    if title_mentioned:
                        evidence.append(f"Title mentioned: {analysis2.title}")

                    relationship = ContentRelationship(
                        source_url=analysis1.url,
                        target_url=analysis2.url,
                        relationship_type=RelationshipType.CROSS_REFERENCE,
                        strength=0.95,
                        confidence=0.9,
                        discovered_method="cross_reference_analysis",
                        supporting_evidence=evidence,
                        context_elements=[analysis2.title or analysis2.url],
                        discovery_timestamp=asyncio.get_event_loop().time(),
                    )
                    relationships.append(relationship)

        return relationships

    def _detect_language_variants(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect content covering same concepts in different programming languages."""

        relationships = []

        # Find language variants by comparing all pairs
        for i, analysis1 in enumerate(content_analyses):
            if not analysis1.topic_keywords or not analysis1.primary_language:
                continue

            for analysis2 in content_analyses[i + 1 :]:
                if not analysis2.topic_keywords or not analysis2.primary_language:
                    continue

                # Skip if same language
                if analysis1.primary_language == analysis2.primary_language:
                    continue

                # Check for shared topic keywords (need at least 2 common keywords)
                shared_keywords = set(analysis1.topic_keywords) & set(
                    analysis2.topic_keywords
                )

                if len(shared_keywords) >= 2:
                    # Calculate strength based on keyword overlap
                    total_keywords = len(
                        set(analysis1.topic_keywords) | set(analysis2.topic_keywords)
                    )
                    strength = (
                        len(shared_keywords) / total_keywords
                        if total_keywords > 0
                        else 0.0
                    )

                    if strength >= 0.4:  # At least 40% keyword overlap
                        relationship = ContentRelationship(
                            source_url=analysis1.url,
                            target_url=analysis2.url,
                            relationship_type=RelationshipType.LANGUAGE_VARIANT,
                            strength=min(strength + 0.4, 1.0),  # Boost strength
                            confidence=0.8,
                            discovered_method="language_variant_analysis",
                            supporting_evidence=[
                                f"Same topic in {analysis1.primary_language} and {analysis2.primary_language}",
                                f"Shared keywords: {list(shared_keywords)}",
                            ],
                            context_elements=list(shared_keywords),
                            discovery_timestamp=asyncio.get_event_loop().time(),
                        )
                        relationships.append(relationship)

        return relationships

    def _detect_version_evolution(
        self, content_analyses: List[ContentAnalysis]
    ) -> List[ContentRelationship]:
        """Detect version evolution relationships."""

        relationships = []

        # Version indicators
        version_patterns = [
            r"v\d+",
            r"version \d+",
            r"\d+\.\d+",
            "deprecated",
            "legacy",
            "new",
            "updated",
        ]

        import re

        # Find content with version indicators
        versioned_content = []
        for analysis in content_analyses:
            content_text = f"{analysis.title or ''} {analysis.summary or ''}".lower()

            for pattern in version_patterns:
                if re.search(pattern, content_text):
                    versioned_content.append((analysis, pattern))
                    break

        # Create version evolution relationships
        for i, (analysis1, pattern1) in enumerate(versioned_content):
            for j, (analysis2, pattern2) in enumerate(versioned_content):
                if i >= j:
                    continue

                # Check if they cover similar topics
                shared_keywords = set(analysis1.topic_keywords or []) & set(
                    analysis2.topic_keywords or []
                )

                if shared_keywords:
                    relationship = ContentRelationship(
                        source_url=analysis1.url,
                        target_url=analysis2.url,
                        relationship_type=RelationshipType.VERSION_EVOLUTION,
                        strength=len(shared_keywords)
                        / max(len(analysis1.topic_keywords or []), 1),
                        confidence=0.6,
                        discovered_method="version_evolution_analysis",
                        supporting_evidence=[
                            f"Version patterns: {pattern1}, {pattern2}",
                            f"Shared keywords: {list(shared_keywords)}",
                        ],
                        context_elements=list(shared_keywords),
                        discovery_timestamp=asyncio.get_event_loop().time(),
                    )
                    relationships.append(relationship)

        return relationships

    def _find_semantic_evidence(
        self, analysis1: ContentAnalysis, analysis2: ContentAnalysis
    ) -> List[str]:
        """Find evidence supporting semantic similarity."""

        evidence = []

        # Shared keywords
        shared_keywords = set(analysis1.topic_keywords or []) & set(
            analysis2.topic_keywords or []
        )
        if shared_keywords:
            evidence.append(f"Shared keywords: {list(shared_keywords)}")

        # Same language
        if (
            analysis1.primary_language
            and analysis1.primary_language == analysis2.primary_language
        ):
            evidence.append(f"Same programming language: {analysis1.primary_language}")

        # Same content type
        if analysis1.content_type == analysis2.content_type:
            evidence.append(f"Same content type: {analysis1.content_type}")

        # Similar complexity
        if (
            analysis1.complexity_indicators
            and analysis2.complexity_indicators
            and abs(
                len(analysis1.complexity_indicators)
                - len(analysis2.complexity_indicators)
            )
            <= 1
        ):
            evidence.append("Similar complexity level")

        return evidence

    def _find_shared_elements(
        self, analysis1: ContentAnalysis, analysis2: ContentAnalysis
    ) -> List[str]:
        """Find shared elements between two content analyses."""

        elements = []

        # Shared code elements
        if analysis1.code_elements and analysis2.code_elements:
            shared_code = set(analysis1.code_elements) & set(analysis2.code_elements)
            elements.extend(shared_code)

        # Shared API references
        if analysis1.api_references and analysis2.api_references:
            shared_apis = set(analysis1.api_references) & set(analysis2.api_references)
            elements.extend(shared_apis)

        # Shared topic keywords
        if analysis1.topic_keywords and analysis2.topic_keywords:
            shared_topics = set(analysis1.topic_keywords) & set(
                analysis2.topic_keywords
            )
            elements.extend(shared_topics)

        return list(elements)

    def _calculate_dependency_strength(
        self, element: str, target_element: str
    ) -> float:
        """Calculate strength of code dependency relationship."""

        # Simple heuristic based on string similarity and context
        element_parts = element.lower().split()
        target_parts = target_element.lower().split()

        # Count shared words
        shared_words = len(set(element_parts) & set(target_parts))
        total_words = len(set(element_parts) | set(target_parts))

        if total_words == 0:
            return 0.0

        similarity = shared_words / total_words

        # Boost for exact matches
        if (
            element.lower() in target_element.lower()
            or target_element.lower() in element.lower()
        ):
            similarity += 0.3

        return min(similarity, 1.0)

    def _validate_and_deduplicate(
        self, relationships: List[ContentRelationship]
    ) -> List[ContentRelationship]:
        """Validate and deduplicate relationships."""

        # Remove duplicates based on source, target, and type
        seen = set()
        unique_relationships = []

        for rel in relationships:
            # Create bidirectional key for symmetric relationships
            if rel.relationship_type in [
                RelationshipType.SEMANTIC_SIMILARITY,
                RelationshipType.API_REFERENCE,
                RelationshipType.LANGUAGE_VARIANT,
            ]:
                key = tuple(
                    sorted([rel.source_url, rel.target_url])
                    + [rel.relationship_type.value]
                )
            else:
                key = (rel.source_url, rel.target_url, rel.relationship_type.value)

            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        # Filter by minimum confidence and strength
        validated_relationships = [
            rel
            for rel in unique_relationships
            if rel.confidence >= 0.5 and rel.strength >= 0.3
        ]

        return validated_relationships


class TopicClusteringEngine:
    """Clusters content into coherent topic groups."""

    def __init__(self, embedding_service: MultiEmbeddingService):
        """Initialize clustering engine."""
        self.embedding_service = embedding_service

        # Clustering parameters
        self.min_cluster_size = 3
        self.max_clusters = 50
        self.similarity_threshold = 0.6

    async def cluster_content(
        self,
        content_analyses: List[ContentAnalysis],
        relationships: List[ContentRelationship],
    ) -> List[TopicCluster]:
        """Cluster content into topic groups."""

        logger.info(f"Clustering {len(content_analyses)} content pieces")

        if len(content_analyses) < self.min_cluster_size:
            return []

        # Prepare embeddings
        embeddings_map = {}
        for analysis in content_analyses:
            if analysis.embedding:
                embeddings_map[analysis.url] = np.array(analysis.embedding)

        if len(embeddings_map) < self.min_cluster_size:
            return []

        # Perform clustering
        cluster_assignments = await self._perform_clustering(embeddings_map)

        # Create topic clusters
        clusters = self._create_topic_clusters(
            content_analyses, cluster_assignments, relationships
        )

        # Post-process clusters
        processed_clusters = self._post_process_clusters(clusters, relationships)

        logger.info(f"Created {len(processed_clusters)} topic clusters")

        return processed_clusters

    async def _perform_clustering(
        self, embeddings_map: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """Perform embedding-based clustering."""

        urls = list(embeddings_map.keys())
        embeddings = np.array(list(embeddings_map.values()))

        # Try DBSCAN first for automatic cluster number detection
        dbscan = DBSCAN(
            eps=1 - self.similarity_threshold,  # Convert similarity to distance
            min_samples=self.min_cluster_size,
            metric="cosine",
        )

        dbscan_labels = dbscan.fit_predict(embeddings)

        # If DBSCAN produces too few clusters, fall back to K-means
        unique_labels = set(dbscan_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise cluster

        if len(unique_labels) < 2:
            # Use K-means with estimated number of clusters
            n_clusters = min(max(len(urls) // 5, 2), self.max_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            cluster_labels = dbscan_labels

        # Create URL to cluster mapping
        cluster_assignments = {}
        for i, url in enumerate(urls):
            cluster_id = cluster_labels[i]
            if cluster_id != -1:  # Ignore noise points
                cluster_assignments[url] = cluster_id

        return cluster_assignments

    def _create_topic_clusters(
        self,
        content_analyses: List[ContentAnalysis],
        cluster_assignments: Dict[str, int],
        relationships: List[ContentRelationship],
    ) -> List[TopicCluster]:
        """Create topic cluster objects from cluster assignments."""

        # Group content by cluster
        cluster_groups = defaultdict(list)
        for analysis in content_analyses:
            if analysis.url in cluster_assignments:
                cluster_id = cluster_assignments[analysis.url]
                cluster_groups[cluster_id].append(analysis)

        clusters = []

        for cluster_id, analyses in cluster_groups.items():
            if len(analyses) < self.min_cluster_size:
                continue

            # Extract cluster characteristics
            cluster_urls = [a.url for a in analyses]
            topic_keywords = self._extract_cluster_keywords(analyses)
            programming_languages = self._extract_cluster_languages(analyses)
            content_types = self._extract_cluster_content_types(analyses)
            difficulty_level = self._determine_difficulty_level(analyses)

            # Calculate cluster metrics
            coherence_score = self._calculate_coherence_score(analyses)
            coverage_score = self._calculate_coverage_score(analyses, topic_keywords)
            quality_score = self._calculate_cluster_quality_score(analyses)

            # Generate cluster name and description
            cluster_name = self._generate_cluster_name(
                topic_keywords, programming_languages
            )
            cluster_description = self._generate_cluster_description(
                analyses, topic_keywords, programming_languages, content_types
            )

            cluster = TopicCluster(
                cluster_id=f"cluster_{cluster_id}",
                name=cluster_name,
                description=cluster_description,
                content_urls=cluster_urls,
                topic_keywords=topic_keywords,
                programming_languages=programming_languages,
                content_types=content_types,
                difficulty_level=difficulty_level,
                coherence_score=coherence_score,
                coverage_score=coverage_score,
                quality_score=quality_score,
                related_clusters=[],  # Will be filled in post-processing
                parent_cluster=None,
                child_clusters=[],
            )

            clusters.append(cluster)

        return clusters

    def _extract_cluster_keywords(self, analyses: List[ContentAnalysis]) -> List[str]:
        """Extract representative keywords for a cluster."""

        # Count keyword frequencies
        keyword_counts = defaultdict(int)
        for analysis in analyses:
            if analysis.topic_keywords:
                for keyword in analysis.topic_keywords:
                    keyword_counts[keyword] += 1

        # Select keywords that appear in at least 30% of content
        min_frequency = max(1, len(analyses) * 0.3)
        cluster_keywords = [
            keyword
            for keyword, count in keyword_counts.items()
            if count >= min_frequency
        ]

        # Sort by frequency and take top 10
        cluster_keywords.sort(key=lambda k: keyword_counts[k], reverse=True)
        return cluster_keywords[:10]

    def _extract_cluster_languages(self, analyses: List[ContentAnalysis]) -> List[str]:
        """Extract programming languages represented in cluster."""

        language_counts = defaultdict(int)
        for analysis in analyses:
            if analysis.primary_language:
                language_counts[analysis.primary_language] += 1

        # Return languages that appear in at least 20% of content
        min_frequency = max(1, len(analyses) * 0.2)
        languages = [
            lang for lang, count in language_counts.items() if count >= min_frequency
        ]

        # Sort by frequency
        languages.sort(key=lambda l: language_counts[l], reverse=True)
        return languages

    def _extract_cluster_content_types(
        self, analyses: List[ContentAnalysis]
    ) -> List[str]:
        """Extract content types represented in cluster."""

        type_counts = defaultdict(int)
        for analysis in analyses:
            type_counts[analysis.content_type] += 1

        # Return all content types
        content_types = list(type_counts.keys())
        content_types.sort(key=lambda t: type_counts[t], reverse=True)
        return content_types

    def _determine_difficulty_level(
        self, analyses: List[ContentAnalysis]
    ) -> Optional[str]:
        """Determine the difficulty level of cluster content."""

        # Count complexity indicators
        complexity_counts = defaultdict(int)

        for analysis in analyses:
            title_lower = (analysis.title or "").lower()
            summary_lower = (analysis.summary or "").lower()
            content = f"{title_lower} {summary_lower}"

            if any(
                word in content
                for word in ["basic", "intro", "beginner", "start", "guide"]
            ):
                complexity_counts["beginner"] += 1
            elif any(
                word in content for word in ["advanced", "expert", "complex", "deep"]
            ):
                complexity_counts["advanced"] += 1
            else:
                complexity_counts["intermediate"] += 1

        # Return the most common difficulty level
        if complexity_counts:
            return max(complexity_counts.keys(), key=lambda k: complexity_counts[k])

        return None

    def _calculate_coherence_score(self, analyses: List[ContentAnalysis]) -> float:
        """Calculate how coherent the cluster is (how well content fits together)."""

        if len(analyses) < 2:
            return 1.0

        # Calculate pairwise similarities using keywords
        similarities = []

        for i, analysis1 in enumerate(analyses):
            for analysis2 in analyses[i + 1 :]:
                # Keyword similarity
                keywords1 = set(analysis1.topic_keywords or [])
                keywords2 = set(analysis2.topic_keywords or [])

                if keywords1 or keywords2:
                    keyword_similarity = len(keywords1 & keywords2) / len(
                        keywords1 | keywords2
                    )
                else:
                    keyword_similarity = 0.0

                # Language similarity
                lang_similarity = (
                    1.0
                    if analysis1.primary_language == analysis2.primary_language
                    else 0.5
                )

                # Combined similarity
                combined_similarity = keyword_similarity * 0.7 + lang_similarity * 0.3
                similarities.append(combined_similarity)

        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_coverage_score(
        self, analyses: List[ContentAnalysis], topic_keywords: List[str]
    ) -> float:
        """Calculate how comprehensive the cluster's coverage is."""

        if not topic_keywords:
            return 0.5

        # Count how many different aspects are covered
        content_type_diversity = len(set(a.content_type for a in analyses))
        keyword_coverage = len(topic_keywords) / 10.0  # Normalize to max 10 keywords

        # Language diversity (but not too much - we want focused clusters)
        language_diversity = (
            min(len(set(a.primary_language for a in analyses if a.primary_language)), 3)
            / 3.0
        )

        coverage_score = (
            content_type_diversity * 0.4
            + keyword_coverage * 0.4
            + language_diversity * 0.2
        )

        return min(coverage_score, 1.0)

    def _calculate_cluster_quality_score(
        self, analyses: List[ContentAnalysis]
    ) -> float:
        """Calculate average quality score for cluster content."""

        if not analyses:
            return 0.0

        # Use quality score if available, otherwise estimate from other factors
        quality_scores = []

        for analysis in analyses:
            if (
                hasattr(analysis, "quality_score")
                and analysis.quality_score is not None
            ):
                quality_scores.append(analysis.quality_score)
            else:
                # Estimate quality from available metrics
                estimated_quality = 0.7  # Base score

                if analysis.topic_keywords and len(analysis.topic_keywords) > 3:
                    estimated_quality += 0.1

                if analysis.code_elements and len(analysis.code_elements) > 2:
                    estimated_quality += 0.1

                if analysis.summary and len(analysis.summary) > 100:
                    estimated_quality += 0.1

                quality_scores.append(min(estimated_quality, 1.0))

        return sum(quality_scores) / len(quality_scores)

    def _generate_cluster_name(
        self, topic_keywords: List[str], programming_languages: List[str]
    ) -> str:
        """Generate a descriptive name for the cluster."""

        if not topic_keywords and not programming_languages:
            return "General Content"

        name_parts = []

        # Add primary programming language
        if programming_languages:
            name_parts.append(programming_languages[0].title())

        # Add primary topic keywords (max 3)
        if topic_keywords:
            key_topics = [kw.title() for kw in topic_keywords[:3]]
            name_parts.extend(key_topics)

        return " ".join(name_parts) if name_parts else "Mixed Content"

    def _generate_cluster_description(
        self,
        analyses: List[ContentAnalysis],
        topic_keywords: List[str],
        programming_languages: List[str],
        content_types: List[str],
    ) -> str:
        """Generate a descriptive description for the cluster."""

        desc_parts = []

        # Content count
        desc_parts.append(f"Collection of {len(analyses)} content pieces")

        # Topic focus
        if topic_keywords:
            topics_str = ", ".join(topic_keywords[:5])
            desc_parts.append(f"focusing on {topics_str}")

        # Language focus
        if programming_languages:
            if len(programming_languages) == 1:
                desc_parts.append(f"in {programming_languages[0]}")
            else:
                langs_str = ", ".join(programming_languages[:3])
                desc_parts.append(f"covering {langs_str}")

        # Content types
        if content_types and len(content_types) > 1:
            types_str = ", ".join(content_types[:3])
            desc_parts.append(f"including {types_str}")

        return ". ".join(desc_parts) + "."

    def _post_process_clusters(
        self, clusters: List[TopicCluster], relationships: List[ContentRelationship]
    ) -> List[TopicCluster]:
        """Post-process clusters to add relationships and hierarchy."""

        # Find related clusters based on shared relationships
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i >= j:
                    continue

                # Count relationships between clusters
                shared_relationships = 0
                for rel in relationships:
                    if (
                        rel.source_url in cluster1.content_urls
                        and rel.target_url in cluster2.content_urls
                    ) or (
                        rel.source_url in cluster2.content_urls
                        and rel.target_url in cluster1.content_urls
                    ):
                        shared_relationships += 1

                # If clusters have significant relationships, mark them as related
                if shared_relationships >= 2:
                    cluster1.related_clusters.append(cluster2.cluster_id)
                    cluster2.related_clusters.append(cluster1.cluster_id)

        # Identify potential hierarchical relationships
        for cluster in clusters:
            # Look for parent clusters (more general topics)
            for other_cluster in clusters:
                if cluster.cluster_id == other_cluster.cluster_id:
                    continue

                # Check if other cluster has more general keywords
                shared_keywords = set(cluster.topic_keywords) & set(
                    other_cluster.topic_keywords
                )
                if (
                    len(shared_keywords) >= 2
                    and len(other_cluster.topic_keywords) < len(cluster.topic_keywords)
                    and other_cluster.difficulty_level in ["beginner", None]
                    and cluster.difficulty_level in ["intermediate", "advanced"]
                ):
                    cluster.parent_cluster = other_cluster.cluster_id
                    other_cluster.child_clusters.append(cluster.cluster_id)

        return clusters


class KnowledgeGraphBuilder:
    """Builds and manages the complete knowledge graph."""

    def __init__(
        self,
        embedding_service: MultiEmbeddingService,
        database_manager: EnhancedDatabaseManager,
    ):
        """Initialize knowledge graph builder."""
        self.embedding_service = embedding_service
        self.database_manager = database_manager
        self.relationship_detector = RelationshipDetector(embedding_service)
        self.clustering_engine = TopicClusteringEngine(embedding_service)

    async def build_knowledge_graph(
        self, content_analyses: List[ContentAnalysis]
    ) -> KnowledgeGraph:
        """Build complete knowledge graph from content analyses."""

        logger.info(
            f"Building knowledge graph for {len(content_analyses)} content pieces"
        )

        # Detect relationships
        relationships = await self.relationship_detector.detect_relationships(
            content_analyses
        )

        # Create topic clusters
        clusters = await self.clustering_engine.cluster_content(
            content_analyses, relationships
        )

        # Calculate graph statistics
        total_nodes = len(content_analyses)
        total_edges = len(relationships)
        cluster_count = len(clusters)

        # Calculate average node degree
        node_degrees = defaultdict(int)
        for rel in relationships:
            node_degrees[rel.source_url] += 1
            node_degrees[rel.target_url] += 1

        average_node_degree = sum(node_degrees.values()) / max(total_nodes, 1)

        # Calculate graph density
        max_possible_edges = total_nodes * (total_nodes - 1) / 2
        graph_density = total_edges / max(max_possible_edges, 1)

        # Calculate modularity (simplified)
        modularity_score = self._calculate_modularity(relationships, clusters)

        # Calculate coverage ratio
        clustered_urls = set()
        for cluster in clusters:
            clustered_urls.update(cluster.content_urls)
        coverage_ratio = len(clustered_urls) / max(total_nodes, 1)

        knowledge_graph = KnowledgeGraph(
            relationships=relationships,
            clusters=clusters,
            total_nodes=total_nodes,
            total_edges=total_edges,
            cluster_count=cluster_count,
            average_node_degree=average_node_degree,
            graph_density=graph_density,
            modularity_score=modularity_score,
            coverage_ratio=coverage_ratio,
        )

        logger.info(
            f"Knowledge graph built: {total_nodes} nodes, {total_edges} edges, "
            f"{cluster_count} clusters, density: {graph_density:.3f}"
        )

        return knowledge_graph

    def _calculate_modularity(
        self, relationships: List[ContentRelationship], clusters: List[TopicCluster]
    ) -> float:
        """Calculate modularity score (simplified version)."""

        if not clusters or not relationships:
            return 0.0

        # Create URL to cluster mapping
        url_to_cluster = {}
        for cluster in clusters:
            for url in cluster.content_urls:
                url_to_cluster[url] = cluster.cluster_id

        # Count edges within vs between clusters
        within_cluster_edges = 0
        total_edges = len(relationships)

        for rel in relationships:
            source_cluster = url_to_cluster.get(rel.source_url)
            target_cluster = url_to_cluster.get(rel.target_url)

            if source_cluster and target_cluster and source_cluster == target_cluster:
                within_cluster_edges += 1

        # Simple modularity approximation
        if total_edges == 0:
            return 0.0

        return within_cluster_edges / total_edges

    async def update_knowledge_graph(
        self, current_graph: KnowledgeGraph, new_content_analyses: List[ContentAnalysis]
    ) -> KnowledgeGraph:
        """Update existing knowledge graph with new content."""

        logger.info(
            f"Updating knowledge graph with {len(new_content_analyses)} new content pieces"
        )

        # Combine existing and new content for relationship detection
        # Note: In practice, you'd want to be more selective about which existing content to include
        all_content = new_content_analyses  # Simplified for this implementation

        # Rebuild graph (in practice, you'd want incremental updates)
        updated_graph = await self.build_knowledge_graph(all_content)

        return updated_graph

    async def save_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> bool:
        """Save knowledge graph to database."""

        try:
            # Save relationships
            for relationship in knowledge_graph.relationships:
                await self.database_manager.store_content_relationship(
                    source_url=relationship.source_url,
                    target_url=relationship.target_url,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    confidence=relationship.confidence,
                    metadata={
                        "discovered_method": relationship.discovered_method,
                        "supporting_evidence": relationship.supporting_evidence,
                        "context_elements": relationship.context_elements,
                        "discovery_timestamp": relationship.discovery_timestamp,
                    },
                )

            # Save clusters
            for cluster in knowledge_graph.clusters:
                await self.database_manager.store_topic_cluster(
                    cluster_id=cluster.cluster_id,
                    name=cluster.name,
                    description=cluster.description,
                    content_urls=cluster.content_urls,
                    metadata={
                        "topic_keywords": cluster.topic_keywords,
                        "programming_languages": cluster.programming_languages,
                        "content_types": cluster.content_types,
                        "difficulty_level": cluster.difficulty_level,
                        "coherence_score": cluster.coherence_score,
                        "coverage_score": cluster.coverage_score,
                        "quality_score": cluster.quality_score,
                        "related_clusters": cluster.related_clusters,
                        "parent_cluster": cluster.parent_cluster,
                        "child_clusters": cluster.child_clusters,
                    },
                )

            # Save graph statistics
            await self.database_manager.store_graph_statistics(
                total_nodes=knowledge_graph.total_nodes,
                total_edges=knowledge_graph.total_edges,
                cluster_count=knowledge_graph.cluster_count,
                average_node_degree=knowledge_graph.average_node_degree,
                graph_density=knowledge_graph.graph_density,
                modularity_score=knowledge_graph.modularity_score,
                coverage_ratio=knowledge_graph.coverage_ratio,
            )

            logger.info("Knowledge graph saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
            return False

    async def load_knowledge_graph(self) -> Optional[KnowledgeGraph]:
        """Load knowledge graph from database."""

        try:
            # Load relationships
            relationships = await self.database_manager.load_content_relationships()

            # Load clusters
            clusters = await self.database_manager.load_topic_clusters()

            # Load graph statistics
            stats = await self.database_manager.load_graph_statistics()

            if stats:
                knowledge_graph = KnowledgeGraph(
                    relationships=relationships,
                    clusters=clusters,
                    total_nodes=stats.get("total_nodes", 0),
                    total_edges=stats.get("total_edges", 0),
                    cluster_count=stats.get("cluster_count", 0),
                    average_node_degree=stats.get("average_node_degree", 0.0),
                    graph_density=stats.get("graph_density", 0.0),
                    modularity_score=stats.get("modularity_score", 0.0),
                    coverage_ratio=stats.get("coverage_ratio", 0.0),
                )

                logger.info(
                    f"Knowledge graph loaded: {len(relationships)} relationships, {len(clusters)} clusters"
                )
                return knowledge_graph

        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")

        return None
