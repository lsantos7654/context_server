"""Unit tests for enhanced embedding strategies."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from context_server.core.content_analysis import ContentAnalysis
from context_server.core.embedding_strategies import (
    CompositeEmbeddingGenerator,
    EmbeddingQualityAnalyzer,
    EmbeddingQualityMetrics,
    EmbeddingResult,
    EmbeddingStrategy,
    EnhancedEmbeddingStrategy,
    HierarchicalEmbedding,
    SummaryGenerator,
)
from context_server.core.multi_embedding_service import (
    EmbeddingModel,
    MultiEmbeddingService,
)


class TestSummaryGenerator:
    """Test summary generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SummaryGenerator()

    def test_generate_document_summary_basic(self):
        """Test basic document summary generation."""
        content = (
            "This is a tutorial about Python programming. Learn functions and classes."
        )
        title = "Python Tutorial"

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language="python",
            summary="Python programming tutorial",
            code_percentage=20.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["Python", "programming", "tutorial"],
            api_references=[],
        )

        summary = self.generator.generate_document_summary(content, analysis, title)

        assert "Document: Python Tutorial" in summary
        assert "Type: tutorial" in summary
        assert "Language: python" in summary
        assert "Key concepts: Python, programming, tutorial" in summary
        assert len(summary) >= self.generator.min_summary_length
        assert len(summary) <= self.generator.max_summary_length

    def test_generate_document_summary_high_code(self):
        """Test summary generation for code-heavy content."""
        content = "def hello(): pass\nclass Test: pass"

        # Mock code blocks
        from context_server.core.content_analysis import CodeBlock

        code_block = CodeBlock(
            language="python",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            functions=["hello", "test_func"],
            classes=["Test", "Helper"],
            imports=["sys", "os"],
        )

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Code example",
            code_percentage=80.0,
            code_blocks=[code_block],
            detected_patterns={},
            key_concepts=["Python"],
            api_references=[],
        )

        summary = self.generator.generate_document_summary(content, analysis)

        assert "Contains 80% code" in summary
        assert "Functions: hello, test_func" in summary
        assert "Classes: Test, Helper" in summary

    def test_generate_section_summary(self):
        """Test section summary generation."""
        content = "This section covers API endpoints and authentication."
        section_title = "API Reference"
        chunk_contents = [
            "GET /api/users - Returns user list",
            "POST /api/users - Creates new user",
            "Authentication via Bearer token",
        ]

        summary = self.generator.generate_section_summary(
            content, section_title, chunk_contents
        )

        assert "Section: API Reference" in summary
        assert len(summary) <= 300

    def test_extract_key_sentences(self):
        """Test key sentence extraction."""
        content = "First sentence is important. Second sentence provides context. Third sentence adds details. Fourth is less relevant."

        sentences = self.generator._extract_key_sentences(content, 2)

        assert len(sentences) <= 2
        assert all(len(s.strip()) > 0 for s in sentences)


class TestCompositeEmbeddingGenerator:
    """Test composite embedding generation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock multi-embedding service
        self.mock_embedding_service = Mock(spec=MultiEmbeddingService)
        self.mock_embedding_service.embed_text = AsyncMock()
        self.mock_embedding_service.route_content.return_value = (
            EmbeddingModel.OPENAI_SMALL
        )

        self.generator = CompositeEmbeddingGenerator(self.mock_embedding_service)

    @pytest.mark.asyncio
    async def test_generate_hierarchical_embedding(self):
        """Test hierarchical embedding generation."""
        content = "This is test content for hierarchical embedding."
        title = "Test Document"
        chunks = ["Chunk 1 content", "Chunk 2 content"]

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Test content",
            code_percentage=10.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["test"],
            api_references=[],
        )

        # Mock embedding responses
        self.mock_embedding_service.embed_text.side_effect = [
            # Document embedding
            {
                "embedding": [0.1] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            },
            # Summary embedding
            {
                "embedding": [0.2] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            },
            # Chunk 1 embedding
            {
                "embedding": [0.3] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            },
            # Chunk 2 embedding
            {
                "embedding": [0.4] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            },
        ]

        result = await self.generator.generate_hierarchical_embedding(
            content, title, chunks, analysis
        )

        assert isinstance(result, HierarchicalEmbedding)
        assert result.document_embedding is not None
        assert result.summary_embedding is not None
        assert len(result.chunk_embeddings) == 2
        assert result.composite_embedding is not None  # Default enabled

        # Verify embedding calls
        assert self.mock_embedding_service.embed_text.call_count == 4

    @pytest.mark.asyncio
    async def test_generate_composite_embedding(self):
        """Test composite embedding generation from multiple sources."""

        # Mock embeddings with same dimension
        embeddings = {
            "document": EmbeddingResult(
                embedding=[0.1] * 1536,
                model="text-embedding-3-small",
                dimension=1536,
                strategy="test",
                metadata={},
            ),
            "summary": EmbeddingResult(
                embedding=[0.2] * 1536,
                model="text-embedding-3-small",
                dimension=1536,
                strategy="test",
                metadata={},
            ),
        }

        analysis = ContentAnalysis(
            content_type="general",
            primary_language=None,
            summary="Test",
            code_percentage=0.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        result = await self.generator._generate_composite_embedding(
            embeddings, analysis, {}
        )

        assert isinstance(result, EmbeddingResult)
        assert result.model == "composite"
        assert result.dimension == 1536
        assert len(result.embedding) == 1536
        assert result.metadata["component_count"] == 2

    def test_calculate_composite_weights(self):
        """Test composite weight calculation."""

        embeddings = {
            "document": Mock(),
            "summary": Mock(),
            "chunk_0": Mock(),
            "chunk_1": Mock(),
        }

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Code",
            code_percentage=60.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        weights = self.generator._calculate_composite_weights(embeddings, analysis, {})

        # For code content, document should have higher weight
        assert weights["document"] == 0.5
        assert weights["summary"] == 0.2
        assert "chunk_0" in weights
        assert "chunk_1" in weights


class TestEmbeddingQualityAnalyzer:
    """Test embedding quality analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EmbeddingQualityAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_embedding_quality(self):
        """Test comprehensive embedding quality analysis."""

        # Create mock hierarchical embedding
        doc_embedding = EmbeddingResult(
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            dimension=1536,
            strategy="test",
            metadata={},
        )

        chunk_embeddings = [
            EmbeddingResult(
                embedding=[0.15] * 1536,  # Similar to document
                model="text-embedding-3-small",
                dimension=1536,
                strategy="test",
                metadata={},
            ),
            EmbeddingResult(
                embedding=[0.05] * 1536,  # Different from document
                model="text-embedding-3-small",
                dimension=1536,
                strategy="test",
                metadata={},
            ),
        ]

        hierarchical_embedding = HierarchicalEmbedding(
            document_embedding=doc_embedding, chunk_embeddings=chunk_embeddings
        )

        content = "Test content"
        chunks = ["Chunk 1", "Chunk 2"]

        metrics = await self.analyzer.analyze_embedding_quality(
            hierarchical_embedding, content, chunks
        )

        assert isinstance(metrics, EmbeddingQualityMetrics)
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.diversity_score <= 1.0
        assert 0.0 <= metrics.coverage_score <= 1.0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert 0.0 <= metrics.confidence_score <= 1.0

    def test_calculate_coherence(self):
        """Test coherence calculation between document and chunks."""

        doc_embedding = EmbeddingResult(
            embedding=[1.0, 0.0, 0.0],
            model="test",
            dimension=3,
            strategy="test",
            metadata={},
        )

        chunk_embeddings = [
            EmbeddingResult(
                embedding=[0.9, 0.1, 0.0],  # High similarity
                model="test",
                dimension=3,
                strategy="test",
                metadata={},
            ),
            EmbeddingResult(
                embedding=[0.0, 1.0, 0.0],  # Low similarity
                model="test",
                dimension=3,
                strategy="test",
                metadata={},
            ),
        ]

        coherence = self.analyzer._calculate_coherence(doc_embedding, chunk_embeddings)

        assert 0.0 <= coherence <= 1.0
        # Should be moderate due to mixed similarities
        assert 0.3 <= coherence <= 0.8

    def test_calculate_diversity(self):
        """Test diversity calculation among chunks."""

        chunk_embeddings = [
            EmbeddingResult(
                embedding=[1.0, 0.0, 0.0],
                model="test",
                dimension=3,
                strategy="test",
                metadata={},
            ),
            EmbeddingResult(
                embedding=[0.0, 1.0, 0.0],  # Orthogonal - high diversity
                model="test",
                dimension=3,
                strategy="test",
                metadata={},
            ),
        ]

        diversity = self.analyzer._calculate_diversity(chunk_embeddings)

        assert 0.0 <= diversity <= 1.0
        # Should be high due to orthogonal vectors
        assert diversity > 0.5


class TestEnhancedEmbeddingStrategy:
    """Test enhanced embedding strategy orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock multi-embedding service
        self.mock_embedding_service = Mock(spec=MultiEmbeddingService)
        self.mock_embedding_service.embed_text = AsyncMock()
        self.mock_embedding_service.embed_batch = AsyncMock()
        self.mock_embedding_service.route_content.return_value = (
            EmbeddingModel.OPENAI_SMALL
        )

        self.strategy_engine = EnhancedEmbeddingStrategy(self.mock_embedding_service)

    @pytest.mark.asyncio
    async def test_generate_adaptive_strategy_code_content(self):
        """Test adaptive strategy selection for code content."""

        content = "def hello(): pass\nclass Test: pass"
        title = "Code Example"
        chunks = ["def hello(): pass", "class Test: pass"]

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=70.0,  # High code percentage
            code_blocks=[],
            detected_patterns={},
            key_concepts=["Python"],
            api_references=[],
        )

        # Mock the hierarchical strategy call
        with patch.object(
            self.strategy_engine, "_generate_hierarchical_strategy"
        ) as mock_hierarchical:
            mock_hierarchical.return_value = {
                "strategy": "hierarchical",
                "primary_embedding": Mock(),
            }

            result = await self.strategy_engine.generate_enhanced_embeddings(
                content, title, chunks, analysis, EmbeddingStrategy.ADAPTIVE
            )

        assert result["adaptive_choice"] == "hierarchical"
        assert result["adaptive_reasoning"]["code_percentage"] == 70.0
        mock_hierarchical.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_adaptive_strategy_long_content(self):
        """Test adaptive strategy selection for long content."""

        content = "Tutorial content"
        title = "Long Tutorial"
        chunks = ["chunk"] * 15  # Many chunks

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Tutorial",
            code_percentage=5.0,  # Low code percentage
            code_blocks=[],
            detected_patterns={},
            key_concepts=["tutorial"],
            api_references=[],
        )

        # Mock the summary enhanced strategy call
        with patch.object(
            self.strategy_engine, "_generate_summary_enhanced_strategy"
        ) as mock_summary:
            mock_summary.return_value = {
                "strategy": "summary_enhanced",
                "primary_embedding": Mock(),
            }

            result = await self.strategy_engine.generate_enhanced_embeddings(
                content, title, chunks, analysis, EmbeddingStrategy.ADAPTIVE
            )

        assert result["adaptive_choice"] == "summary_enhanced"
        assert result["adaptive_reasoning"]["chunk_count"] == 15
        mock_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_hierarchical_strategy(self):
        """Test hierarchical strategy implementation."""

        content = "Test content"
        title = "Test Document"
        chunks = ["Chunk 1", "Chunk 2"]

        analysis = ContentAnalysis(
            content_type="general",
            primary_language=None,
            summary="Test",
            code_percentage=0.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        # Mock hierarchical embedding generation
        mock_hierarchical = HierarchicalEmbedding(
            document_embedding=EmbeddingResult(
                embedding=[0.1] * 1536,
                model="test",
                dimension=1536,
                strategy="hierarchical",
                metadata={},
            )
        )

        with patch.object(
            self.strategy_engine.composite_generator, "generate_hierarchical_embedding"
        ) as mock_gen:
            mock_gen.return_value = mock_hierarchical

            with patch.object(
                self.strategy_engine.quality_analyzer, "analyze_embedding_quality"
            ) as mock_quality:
                mock_quality.return_value = EmbeddingQualityMetrics(
                    coherence_score=0.8,
                    diversity_score=0.7,
                    coverage_score=0.9,
                    consistency_score=0.8,
                    confidence_score=0.8,
                    model_agreement={},
                )

                result = await self.strategy_engine._generate_hierarchical_strategy(
                    content, title, chunks, analysis, {}
                )

        assert result["strategy"] == "hierarchical"
        assert "hierarchical_embedding" in result
        assert "quality_metrics" in result
        assert result["primary_embedding"] == mock_hierarchical.document_embedding

    @pytest.mark.asyncio
    async def test_generate_summary_enhanced_strategy(self):
        """Test summary-enhanced strategy implementation."""

        content = "This is a comprehensive tutorial about advanced topics."
        title = "Advanced Tutorial"
        chunks = ["Chapter 1", "Chapter 2", "Chapter 3"]

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Advanced tutorial",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["advanced", "tutorial"],
            api_references=[],
        )

        # Mock embedding responses
        self.mock_embedding_service.embed_text.return_value = {
            "embedding": [0.1] * 1536,
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "success": True,
        }

        self.mock_embedding_service.embed_batch.return_value = [
            {
                "embedding": [0.2] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            }
        ] * 3

        result = await self.strategy_engine._generate_summary_enhanced_strategy(
            content, title, chunks, analysis, {}
        )

        assert result["strategy"] == "summary_enhanced"
        assert "enhanced_summary" in result
        assert "summary_embedding" in result
        assert "chunk_embeddings" in result
        assert result["primary_embedding"] == result["summary_embedding"]

    @pytest.mark.asyncio
    async def test_generate_composite_strategy(self):
        """Test composite strategy implementation."""

        content = "API documentation with examples"
        title = "API Reference"
        chunks = ["GET /users", "POST /users", "Examples"]

        analysis = ContentAnalysis(
            content_type="api_reference",
            primary_language="javascript",
            summary="API reference",
            code_percentage=30.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["API", "reference"],
            api_references=["GET", "POST"],
        )

        # Mock hierarchical embedding with composite
        mock_hierarchical = HierarchicalEmbedding(
            document_embedding=Mock(),
            summary_embedding=Mock(),
            chunk_embeddings=[Mock(), Mock(), Mock()],
            composite_embedding=EmbeddingResult(
                embedding=[0.5] * 1536,
                model="composite",
                dimension=1536,
                strategy="composite",
                metadata={"component_count": 3},
            ),
        )

        with patch.object(
            self.strategy_engine.composite_generator, "generate_hierarchical_embedding"
        ) as mock_gen:
            mock_gen.return_value = mock_hierarchical

            result = await self.strategy_engine._generate_composite_strategy(
                content, title, chunks, analysis, {}
            )

        assert result["strategy"] == "composite"
        assert "composite_embedding" in result
        assert "component_embeddings" in result
        assert result["primary_embedding"] == mock_hierarchical.composite_embedding
