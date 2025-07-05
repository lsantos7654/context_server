"""Unit tests for content relationship mapping and topic clustering functionality."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from context_server.core.content_analysis import ContentAnalysis
from context_server.core.relationship_mapping import (
    ContentRelationship,
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    RelationshipDetector,
    RelationshipType,
    TopicCluster,
    TopicClusteringEngine,
)


class TestRelationshipDetector:
    """Test relationship detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.detector = RelationshipDetector(self.mock_embedding_service)

    def create_sample_analysis(
        self,
        url: str,
        title: str = "",
        summary: str = "",
        topic_keywords: list = None,
        code_elements: list = None,
        api_references: list = None,
        primary_language: str = None,
        content_type: str = "general",
        embedding: list = None,
        raw_content: str = None,
    ):
        """Create a sample ContentAnalysis for testing."""
        return ContentAnalysis(
            url=url,
            title=title,
            summary=summary,
            content_type=content_type,
            primary_language=primary_language,
            topic_keywords=topic_keywords or [],
            code_elements=code_elements or [],
            api_references=api_references or [],
            complexity_indicators=[],
            readability_score=0.8,
            quality_indicators={},
            raw_content=raw_content or f"{title} {summary}",
            embedding=embedding or [0.1, 0.2, 0.3, 0.4, 0.5],
        )

    @pytest.mark.asyncio
    async def test_detect_semantic_relationships(self):
        """Test semantic similarity detection."""
        # Create content with similar embeddings
        analysis1 = self.create_sample_analysis(
            "http://example.com/1",
            "Python Functions",
            "How to define functions in Python",
            topic_keywords=["python", "functions", "programming"],
            embedding=[0.8, 0.6, 0.4, 0.2, 0.0],
        )
        analysis2 = self.create_sample_analysis(
            "http://example.com/2",
            "Function Examples",
            "Examples of Python function definitions",
            topic_keywords=["python", "functions", "examples"],
            embedding=[0.7, 0.5, 0.3, 0.1, 0.0],  # Similar to analysis1
        )
        analysis3 = self.create_sample_analysis(
            "http://example.com/3",
            "Java Classes",
            "Object-oriented programming in Java",
            topic_keywords=["java", "classes", "oop"],
            embedding=[0.1, 0.2, 0.8, 0.9, 0.7],  # Different from others
        )

        content_analyses = [analysis1, analysis2, analysis3]
        relationships = await self.detector._detect_semantic_relationships(
            content_analyses
        )

        # Should find relationship between analysis1 and analysis2 (similar embeddings)
        semantic_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.SEMANTIC_SIMILARITY
        ]
        assert len(semantic_rels) == 1

        rel = semantic_rels[0]
        assert {rel.source_url, rel.target_url} == {
            "http://example.com/1",
            "http://example.com/2",
        }
        assert rel.strength > 0.7  # High similarity
        assert rel.confidence > 0.7

    def test_detect_code_dependencies(self):
        """Test code dependency detection."""
        analysis1 = self.create_sample_analysis(
            "http://example.com/calculator",
            "Calculator Functions",
            code_elements=["def add(a, b)", "def multiply(a, b)"],
            primary_language="python",
        )
        analysis2 = self.create_sample_analysis(
            "http://example.com/usage",
            "Using Calculator",
            code_elements=[
                "import calculator",
                "calculator.add(1, 2)",
                "result = add(x, y)",
            ],
            primary_language="python",
        )

        content_analyses = [analysis1, analysis2]
        relationships = self.detector._detect_code_dependencies(content_analyses)

        # Should find dependency relationship
        code_deps = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.CODE_DEPENDENCY
        ]
        assert len(code_deps) > 0

        # Check that usage depends on definition
        dep_rel = next(
            (r for r in code_deps if r.source_url == "http://example.com/usage"), None
        )
        assert dep_rel is not None
        assert dep_rel.target_url == "http://example.com/calculator"

    def test_detect_api_references(self):
        """Test API reference detection."""
        analysis1 = self.create_sample_analysis(
            "http://example.com/api1",
            "User API Documentation",
            api_references=["GET /api/users", "POST /api/users"],
        )
        analysis2 = self.create_sample_analysis(
            "http://example.com/api2",
            "User Management Guide",
            api_references=["GET /api/users", "DELETE /api/users/{id}"],
        )
        analysis3 = self.create_sample_analysis(
            "http://example.com/different",
            "Product API",
            api_references=["GET /api/products"],
        )

        content_analyses = [analysis1, analysis2, analysis3]
        relationships = self.detector._detect_api_references(content_analyses)

        # Should find relationship between analysis1 and analysis2 (shared API)
        api_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.API_REFERENCE
        ]
        assert len(api_rels) == 1

        rel = api_rels[0]
        assert {rel.source_url, rel.target_url} == {
            "http://example.com/api1",
            "http://example.com/api2",
        }
        assert "GET /api/users" in rel.context_elements

    def test_detect_conceptual_hierarchy(self):
        """Test conceptual hierarchy detection."""
        analysis_parent = self.create_sample_analysis(
            "http://example.com/intro",
            "Introduction to Python Programming",
            "Basic Python programming guide",
            topic_keywords=["python", "programming", "basics"],
        )
        analysis_child = self.create_sample_analysis(
            "http://example.com/advanced",
            "Advanced Python Techniques",
            "Detailed examples of advanced Python",
            topic_keywords=["python", "programming", "advanced"],
        )

        content_analyses = [analysis_parent, analysis_child]
        relationships = self.detector._detect_conceptual_hierarchy(content_analyses)

        # Should find hierarchical relationship
        hierarchy_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.CONCEPTUAL_HIERARCHY
        ]
        assert len(hierarchy_rels) == 1

        rel = hierarchy_rels[0]
        assert rel.source_url == "http://example.com/intro"  # Parent
        assert rel.target_url == "http://example.com/advanced"  # Child

    def test_detect_tutorial_sequences(self):
        """Test tutorial sequence detection."""
        analysis1 = self.create_sample_analysis(
            "http://example.com/part1",
            "Python Tutorial Part 1",
            "Basic Python concepts",
            topic_keywords=["python", "tutorial", "basics"],
        )
        analysis2 = self.create_sample_analysis(
            "http://example.com/part2",
            "Python Tutorial Part 2",
            "Intermediate Python concepts",
            topic_keywords=["python", "tutorial", "intermediate"],
        )

        content_analyses = [analysis1, analysis2]
        relationships = self.detector._detect_tutorial_sequences(content_analyses)

        # Should find sequence relationship
        sequence_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.TUTORIAL_SEQUENCE
        ]
        assert len(sequence_rels) >= 1

        # Find the part 1 -> part 2 relationship
        part_rel = next(
            (
                r
                for r in sequence_rels
                if r.source_url == "http://example.com/part1"
                and r.target_url == "http://example.com/part2"
            ),
            None,
        )
        assert part_rel is not None

    def test_detect_cross_references(self):
        """Test cross-reference detection."""
        analysis1 = self.create_sample_analysis(
            "http://example.com/doc1",
            "Main Documentation",
            raw_content="See also: Advanced Guide and http://example.com/doc2",
        )
        analysis2 = self.create_sample_analysis(
            "http://example.com/doc2",
            "Advanced Guide",
            raw_content="Referenced documentation",
        )

        # Manually set raw_content since create_sample_analysis sets it automatically
        analysis1.raw_content = "See also: Advanced Guide and http://example.com/doc2"

        content_analyses = [analysis1, analysis2]
        relationships = self.detector._detect_cross_references(content_analyses)

        # Should find cross-reference relationship
        cross_refs = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.CROSS_REFERENCE
        ]
        assert len(cross_refs) == 1

        rel = cross_refs[0]
        assert rel.source_url == "http://example.com/doc1"
        assert rel.target_url == "http://example.com/doc2"

    def test_detect_language_variants(self):
        """Test language variant detection."""
        analysis_python = self.create_sample_analysis(
            "http://example.com/python",
            "Sorting in Python",
            topic_keywords=["sorting", "algorithms", "list"],
            primary_language="python",
        )
        analysis_java = self.create_sample_analysis(
            "http://example.com/java",
            "Sorting in Java",
            topic_keywords=["sorting", "algorithms", "array"],
            primary_language="java",
        )
        analysis_different = self.create_sample_analysis(
            "http://example.com/web",
            "Web Development",
            topic_keywords=["html", "css", "web"],
            primary_language="javascript",
        )

        content_analyses = [analysis_python, analysis_java, analysis_different]
        relationships = self.detector._detect_language_variants(content_analyses)

        # Should find language variant between Python and Java sorting
        language_variants = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.LANGUAGE_VARIANT
        ]
        assert len(language_variants) == 1

        rel = language_variants[0]
        assert {rel.source_url, rel.target_url} == {
            "http://example.com/python",
            "http://example.com/java",
        }
        assert "sorting" in rel.context_elements

    def test_detect_version_evolution(self):
        """Test version evolution detection."""
        analysis_v1 = self.create_sample_analysis(
            "http://example.com/v1",
            "API Documentation v1.0",
            "Legacy API version 1.0",
            topic_keywords=["api", "documentation", "rest"],
        )
        analysis_v2 = self.create_sample_analysis(
            "http://example.com/v2",
            "API Documentation v2.0",
            "Updated API version 2.0",
            topic_keywords=["api", "documentation", "rest"],
        )

        content_analyses = [analysis_v1, analysis_v2]
        relationships = self.detector._detect_version_evolution(content_analyses)

        # Should find version evolution relationship
        version_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.VERSION_EVOLUTION
        ]
        assert len(version_rels) == 1

        rel = version_rels[0]
        assert {rel.source_url, rel.target_url} == {
            "http://example.com/v1",
            "http://example.com/v2",
        }

    @pytest.mark.asyncio
    async def test_detect_relationships_integration(self):
        """Test full relationship detection integration."""
        # Create diverse content for comprehensive testing
        analyses = [
            self.create_sample_analysis(
                "http://example.com/1",
                "Python Functions Tutorial",
                topic_keywords=["python", "functions"],
                code_elements=["def example()"],
                embedding=[0.8, 0.2, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/2",
                "Python Function Examples",
                topic_keywords=["python", "functions"],
                code_elements=["example()", "function call"],
                embedding=[0.7, 0.3, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/3",
                "API Reference",
                api_references=["GET /api/users"],
                embedding=[0.1, 0.1, 0.8, 0.1, 0.1],
            ),
        ]

        relationships = await self.detector.detect_relationships(analyses)

        # Should detect multiple types of relationships
        assert len(relationships) > 0

        # Check for semantic similarity
        semantic_rels = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.SEMANTIC_SIMILARITY
        ]
        assert len(semantic_rels) > 0

        # Check for code dependencies
        code_deps = [
            r
            for r in relationships
            if r.relationship_type == RelationshipType.CODE_DEPENDENCY
        ]
        assert len(code_deps) > 0

    def test_validate_and_deduplicate(self):
        """Test relationship validation and deduplication."""
        import asyncio

        # Create duplicate and invalid relationships
        relationships = [
            ContentRelationship(
                source_url="http://example.com/1",
                target_url="http://example.com/2",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=asyncio.get_event_loop().time(),
            ),
            ContentRelationship(  # Duplicate (reversed)
                source_url="http://example.com/2",
                target_url="http://example.com/1",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.7,
                confidence=0.8,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=asyncio.get_event_loop().time(),
            ),
            ContentRelationship(  # Low confidence - should be filtered
                source_url="http://example.com/3",
                target_url="http://example.com/4",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.2,
                confidence=0.3,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=asyncio.get_event_loop().time(),
            ),
        ]

        validated = self.detector._validate_and_deduplicate(relationships)

        # Should remove duplicate and low-confidence relationship
        assert len(validated) == 1
        assert validated[0].confidence >= 0.5


class TestTopicClusteringEngine:
    """Test topic clustering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.engine = TopicClusteringEngine(self.mock_embedding_service)

    def create_sample_analysis(self, url: str, **kwargs):
        """Create a sample ContentAnalysis for testing."""
        defaults = {
            "title": f"Content {url}",
            "summary": f"Summary for {url}",
            "content_type": "general",
            "primary_language": None,
            "topic_keywords": [],
            "code_elements": [],
            "api_references": [],
            "complexity_indicators": [],
            "readability_score": 0.8,
            "quality_indicators": {},
            "raw_content": f"Content {url}",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        defaults.update(kwargs)

        return ContentAnalysis(url=url, **defaults)

    @pytest.mark.asyncio
    async def test_perform_clustering(self):
        """Test embedding-based clustering."""
        # Create embeddings map with distinct clusters
        embeddings_map = {
            "http://example.com/python1": np.array([0.9, 0.1, 0.1, 0.1, 0.1]),
            "http://example.com/python2": np.array([0.8, 0.2, 0.1, 0.1, 0.1]),
            "http://example.com/python3": np.array([0.7, 0.3, 0.1, 0.1, 0.1]),
            "http://example.com/java1": np.array([0.1, 0.1, 0.9, 0.1, 0.1]),
            "http://example.com/java2": np.array([0.1, 0.2, 0.8, 0.1, 0.1]),
            "http://example.com/java3": np.array([0.1, 0.1, 0.7, 0.2, 0.1]),
        }

        cluster_assignments = await self.engine._perform_clustering(embeddings_map)

        # Should create meaningful clusters
        assert len(cluster_assignments) > 0

        # URLs with similar embeddings should be in same cluster
        python_urls = [url for url in cluster_assignments.keys() if "python" in url]
        java_urls = [url for url in cluster_assignments.keys() if "java" in url]

        if len(python_urls) > 1:
            python_clusters = [cluster_assignments[url] for url in python_urls]
            assert len(set(python_clusters)) <= 2  # Should be mostly in same cluster

        if len(java_urls) > 1:
            java_clusters = [cluster_assignments[url] for url in java_urls]
            assert len(set(java_clusters)) <= 2  # Should be mostly in same cluster

    def test_extract_cluster_keywords(self):
        """Test cluster keyword extraction."""
        analyses = [
            self.create_sample_analysis(
                "http://example.com/1",
                topic_keywords=["python", "functions", "programming"],
            ),
            self.create_sample_analysis(
                "http://example.com/2",
                topic_keywords=["python", "functions", "examples"],
            ),
            self.create_sample_analysis(
                "http://example.com/3", topic_keywords=["python", "classes", "oop"]
            ),
        ]

        keywords = self.engine._extract_cluster_keywords(analyses)

        # "python" should be included (appears in all 3)
        assert "python" in keywords
        # "functions" should be included (appears in 2/3)
        assert "functions" in keywords

    def test_extract_cluster_languages(self):
        """Test cluster programming language extraction."""
        analyses = [
            self.create_sample_analysis(
                "http://example.com/1", primary_language="python"
            ),
            self.create_sample_analysis(
                "http://example.com/2", primary_language="python"
            ),
            self.create_sample_analysis(
                "http://example.com/3", primary_language="java"
            ),
        ]

        languages = self.engine._extract_cluster_languages(analyses)

        # Python should be primary language (appears most frequently)
        assert "python" in languages
        assert languages[0] == "python"

    def test_determine_difficulty_level(self):
        """Test difficulty level determination."""
        beginner_analyses = [
            self.create_sample_analysis(
                "http://example.com/1", title="Basic Python Guide"
            ),
            self.create_sample_analysis(
                "http://example.com/2", title="Introduction to Programming"
            ),
        ]

        advanced_analyses = [
            self.create_sample_analysis(
                "http://example.com/3", title="Advanced Python Techniques"
            ),
            self.create_sample_analysis(
                "http://example.com/4", title="Expert-level Programming"
            ),
        ]

        beginner_level = self.engine._determine_difficulty_level(beginner_analyses)
        advanced_level = self.engine._determine_difficulty_level(advanced_analyses)

        assert beginner_level == "beginner"
        assert advanced_level == "advanced"

    def test_calculate_coherence_score(self):
        """Test cluster coherence calculation."""
        # High coherence cluster (similar keywords and language)
        coherent_analyses = [
            self.create_sample_analysis(
                "http://example.com/1",
                topic_keywords=["python", "functions"],
                primary_language="python",
            ),
            self.create_sample_analysis(
                "http://example.com/2",
                topic_keywords=["python", "functions", "examples"],
                primary_language="python",
            ),
        ]

        # Low coherence cluster (different keywords and languages)
        incoherent_analyses = [
            self.create_sample_analysis(
                "http://example.com/3",
                topic_keywords=["python", "functions"],
                primary_language="python",
            ),
            self.create_sample_analysis(
                "http://example.com/4",
                topic_keywords=["html", "css"],
                primary_language="javascript",
            ),
        ]

        coherent_score = self.engine._calculate_coherence_score(coherent_analyses)
        incoherent_score = self.engine._calculate_coherence_score(incoherent_analyses)

        assert coherent_score > incoherent_score
        assert 0.0 <= coherent_score <= 1.0
        assert 0.0 <= incoherent_score <= 1.0

    def test_calculate_coverage_score(self):
        """Test cluster coverage calculation."""
        topic_keywords = ["python", "functions", "programming", "examples"]

        analyses = [
            self.create_sample_analysis(
                "http://example.com/1", content_type="tutorial"
            ),
            self.create_sample_analysis(
                "http://example.com/2", content_type="code_example"
            ),
            self.create_sample_analysis(
                "http://example.com/3", content_type="api_reference"
            ),
        ]

        coverage_score = self.engine._calculate_coverage_score(analyses, topic_keywords)

        assert 0.0 <= coverage_score <= 1.0
        # Should have good coverage with diverse content types and keywords
        assert coverage_score > 0.5

    def test_generate_cluster_name(self):
        """Test cluster name generation."""
        topic_keywords = ["functions", "programming", "examples"]
        programming_languages = ["python"]

        name = self.engine._generate_cluster_name(topic_keywords, programming_languages)

        assert "Python" in name
        assert any(keyword.title() in name for keyword in topic_keywords[:3])

    def test_generate_cluster_description(self):
        """Test cluster description generation."""
        analyses = [
            self.create_sample_analysis("http://example.com/1"),
            self.create_sample_analysis("http://example.com/2"),
            self.create_sample_analysis("http://example.com/3"),
        ]
        topic_keywords = ["python", "functions"]
        programming_languages = ["python"]
        content_types = ["tutorial", "code_example"]

        description = self.engine._generate_cluster_description(
            analyses, topic_keywords, programming_languages, content_types
        )

        assert "3 content pieces" in description
        assert "python" in description.lower()
        assert "functions" in description.lower()

    @pytest.mark.asyncio
    async def test_cluster_content_integration(self):
        """Test full content clustering integration."""
        # Create content analyses with embeddings for clustering
        analyses = [
            self.create_sample_analysis(
                "http://example.com/python1",
                topic_keywords=["python", "functions"],
                primary_language="python",
                embedding=[0.9, 0.1, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/python2",
                topic_keywords=["python", "functions"],
                primary_language="python",
                embedding=[0.8, 0.2, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/python3",
                topic_keywords=["python", "classes"],
                primary_language="python",
                embedding=[0.7, 0.3, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/java1",
                topic_keywords=["java", "classes"],
                primary_language="java",
                embedding=[0.1, 0.1, 0.9, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/java2",
                topic_keywords=["java", "oop"],
                primary_language="java",
                embedding=[0.1, 0.1, 0.8, 0.2, 0.1],
            ),
        ]

        relationships = []  # Empty for this test

        clusters = await self.engine.cluster_content(analyses, relationships)

        # Should create meaningful clusters
        assert len(clusters) > 0

        # Check cluster properties
        for cluster in clusters:
            assert cluster.cluster_id is not None
            assert cluster.name is not None
            assert cluster.description is not None
            assert len(cluster.content_urls) >= self.engine.min_cluster_size
            assert 0.0 <= cluster.coherence_score <= 1.0
            assert 0.0 <= cluster.coverage_score <= 1.0
            assert 0.0 <= cluster.quality_score <= 1.0


class TestKnowledgeGraphBuilder:
    """Test knowledge graph building functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.mock_database_manager = Mock()
        self.builder = KnowledgeGraphBuilder(
            self.mock_embedding_service, self.mock_database_manager
        )

    def create_sample_analysis(self, url: str, **kwargs):
        """Create a sample ContentAnalysis for testing."""
        defaults = {
            "title": f"Content {url}",
            "summary": f"Summary for {url}",
            "content_type": "general",
            "primary_language": None,
            "topic_keywords": [],
            "code_elements": [],
            "api_references": [],
            "complexity_indicators": [],
            "readability_score": 0.8,
            "quality_indicators": {},
            "raw_content": f"Content {url}",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        defaults.update(kwargs)

        return ContentAnalysis(url=url, **defaults)

    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self):
        """Test complete knowledge graph building."""
        # Create sample content analyses
        analyses = [
            self.create_sample_analysis(
                "http://example.com/1",
                topic_keywords=["python", "functions"],
                primary_language="python",
                embedding=[0.9, 0.1, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/2",
                topic_keywords=["python", "functions"],
                primary_language="python",
                embedding=[0.8, 0.2, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/3",
                topic_keywords=["python", "classes"],
                primary_language="python",
                embedding=[0.7, 0.3, 0.1, 0.1, 0.1],
            ),
            self.create_sample_analysis(
                "http://example.com/4",
                topic_keywords=["java", "oop"],
                primary_language="java",
                embedding=[0.1, 0.1, 0.9, 0.1, 0.1],
            ),
        ]

        knowledge_graph = await self.builder.build_knowledge_graph(analyses)

        # Verify graph structure
        assert isinstance(knowledge_graph, KnowledgeGraph)
        assert knowledge_graph.total_nodes == len(analyses)
        assert knowledge_graph.total_edges >= 0
        assert knowledge_graph.cluster_count >= 0
        assert 0.0 <= knowledge_graph.graph_density <= 1.0
        assert 0.0 <= knowledge_graph.modularity_score <= 1.0
        assert 0.0 <= knowledge_graph.coverage_ratio <= 1.0

    def test_calculate_modularity(self):
        """Test modularity calculation."""
        # Create sample relationships and clusters
        relationships = [
            ContentRelationship(
                source_url="http://example.com/1",
                target_url="http://example.com/2",
                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=[],
                context_elements=[],
                discovery_timestamp=0.0,
            )
        ]

        clusters = [
            TopicCluster(
                cluster_id="cluster_1",
                name="Test Cluster",
                description="Test cluster description",
                content_urls=["http://example.com/1", "http://example.com/2"],
                topic_keywords=["test"],
                programming_languages=["python"],
                content_types=["general"],
                difficulty_level="beginner",
                coherence_score=0.8,
                coverage_score=0.7,
                quality_score=0.9,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            )
        ]

        modularity = self.builder._calculate_modularity(relationships, clusters)

        # Should be high modularity (edge within cluster)
        assert 0.0 <= modularity <= 1.0
        assert modularity > 0.5  # Good clustering

    @pytest.mark.asyncio
    async def test_save_knowledge_graph(self):
        """Test knowledge graph saving."""
        # Mock database manager methods
        self.mock_database_manager.store_content_relationship = AsyncMock(
            return_value=True
        )
        self.mock_database_manager.store_topic_cluster = AsyncMock(return_value=True)
        self.mock_database_manager.store_graph_statistics = AsyncMock(return_value=True)

        # Create sample knowledge graph
        knowledge_graph = KnowledgeGraph(
            relationships=[],
            clusters=[],
            total_nodes=4,
            total_edges=2,
            cluster_count=1,
            average_node_degree=1.0,
            graph_density=0.3,
            modularity_score=0.8,
            coverage_ratio=0.9,
        )

        result = await self.builder.save_knowledge_graph(knowledge_graph)

        assert result is True
        self.mock_database_manager.store_graph_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_knowledge_graph(self):
        """Test knowledge graph loading."""
        # Mock database manager methods
        self.mock_database_manager.load_content_relationships = AsyncMock(
            return_value=[]
        )
        self.mock_database_manager.load_topic_clusters = AsyncMock(return_value=[])
        self.mock_database_manager.load_graph_statistics = AsyncMock(
            return_value={
                "total_nodes": 4,
                "total_edges": 2,
                "cluster_count": 1,
                "average_node_degree": 1.0,
                "graph_density": 0.3,
                "modularity_score": 0.8,
                "coverage_ratio": 0.9,
            }
        )

        knowledge_graph = await self.builder.load_knowledge_graph()

        assert knowledge_graph is not None
        assert knowledge_graph.total_nodes == 4
        assert knowledge_graph.total_edges == 2
        assert knowledge_graph.cluster_count == 1

    @pytest.mark.asyncio
    async def test_update_knowledge_graph(self):
        """Test knowledge graph updating."""
        current_graph = KnowledgeGraph(
            relationships=[],
            clusters=[],
            total_nodes=2,
            total_edges=1,
            cluster_count=1,
            average_node_degree=1.0,
            graph_density=0.5,
            modularity_score=0.7,
            coverage_ratio=0.8,
        )

        new_analyses = [
            self.create_sample_analysis(
                "http://example.com/new1",
                topic_keywords=["new", "content"],
                embedding=[0.5, 0.5, 0.1, 0.1, 0.1],
            )
        ]

        updated_graph = await self.builder.update_knowledge_graph(
            current_graph, new_analyses
        )

        assert isinstance(updated_graph, KnowledgeGraph)
        # Updated graph should reflect new content
        assert updated_graph.total_nodes >= 1
