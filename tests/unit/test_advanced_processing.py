"""Unit tests for advanced document processing with enhanced embedding strategies."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.advanced_processing import (
    AdvancedDocumentProcessor,
    AdvancedProcessedChunk,
    AdvancedProcessedDocument,
    AdvancedProcessingResult,
)
from context_server.core.content_analysis import ContentAnalysis
from context_server.core.embedding_strategies import (
    EmbeddingQualityMetrics,
    EmbeddingResult,
    EmbeddingStrategy,
    HierarchicalEmbedding,
)
from context_server.core.multi_embedding_service import MultiEmbeddingService


class TestAdvancedDocumentProcessor:
    """Test advanced document processor with enhanced embedding strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock multi-embedding service
        self.mock_embedding_service = Mock(spec=MultiEmbeddingService)

        # Create processor with mocked service
        self.processor = AdvancedDocumentProcessor(
            multi_embedding_service=self.mock_embedding_service,
            default_strategy=EmbeddingStrategy.ADAPTIVE,
            enable_quality_analysis=True,
        )

        # Mock extractor to avoid real HTTP requests
        self.processor.extractor.extract_from_url = AsyncMock()

        # Mock chunker
        self.mock_chunker_result = Mock()
        self.mock_chunker_result.content = "Test chunk content"
        self.mock_chunker_result.tokens = 10
        self.mock_chunker_result.metadata = {"chunk_index": 0}
        self.mock_chunker_result.start_line = 1
        self.mock_chunker_result.end_line = 1
        self.mock_chunker_result.char_start = 0
        self.mock_chunker_result.char_end = 18

    @pytest.mark.asyncio
    async def test_process_url_with_adaptive_strategy(self):
        """Test advanced URL processing with adaptive strategy."""

        # Mock extraction result
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.content = "Combined content"
        mock_extraction_result.metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "extracted_pages": [
                {
                    "url": "https://example.com/api",
                    "content": "API documentation with function definitions and examples",
                    "filename": "api.md",
                }
            ],
        }

        self.processor.extractor.extract_from_url.return_value = mock_extraction_result

        # Mock chunker
        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            mock_chunker.return_value = [self.mock_chunker_result]

            # Mock content analyzer
            with patch.object(
                self.processor.content_analyzer, "analyze_content"
            ) as mock_analyzer:
                mock_analysis = ContentAnalysis(
                    content_type="api_reference",
                    primary_language="javascript",
                    summary="API documentation",
                    code_percentage=35.0,
                    code_blocks=[],
                    detected_patterns={"async": ["async", "await"]},
                    key_concepts=["API", "documentation"],
                    api_references=["getData()", "postData()"],
                )
                mock_analyzer.return_value = mock_analysis

                # Mock embedding strategy engine
                with patch.object(
                    self.processor.embedding_strategy_engine,
                    "generate_enhanced_embeddings",
                ) as mock_strategy:
                    mock_strategy.return_value = {
                        "strategy": "composite",
                        "adaptive_choice": "composite",
                        "adaptive_reasoning": {
                            "code_percentage": 35.0,
                            "content_type": "api_reference",
                        },
                        "primary_embedding": {
                            "embedding": [0.1] * 1536,
                            "model": "text-embedding-3-small",
                            "dimension": 1536,
                            "success": True,
                        },
                        "chunk_embeddings": [
                            {
                                "embedding": [0.1] * 1536,
                                "model": "text-embedding-3-small",
                                "dimension": 1536,
                                "success": True,
                            }
                        ],
                    }

                    result = await self.processor.process_url("https://example.com")

        # Verify result structure
        assert isinstance(result, AdvancedProcessingResult)
        assert result.success is True
        assert len(result.documents) == 1
        assert result.total_chunks == 1
        assert "composite" in result.embedding_strategies_used

        # Check processing stats
        assert result.processing_stats["total_pages"] == 1
        assert result.processing_stats["api_reference_pages"] == 1
        assert "api_reference" in result.processing_stats["content_types_detected"]

        # Verify document
        doc = result.documents[0]
        assert isinstance(doc, AdvancedProcessedDocument)
        assert doc.url == "https://example.com/api"
        assert doc.content_analysis.content_type == "api_reference"
        assert doc.embedding_strategy == "composite"

        # Verify chunk
        assert len(doc.chunks) == 1
        chunk = doc.chunks[0]
        assert isinstance(chunk, AdvancedProcessedChunk)
        assert chunk.content == "Test chunk content"
        assert chunk.primary_embedding is not None

    @pytest.mark.asyncio
    async def test_process_url_with_explicit_strategy(self):
        """Test processing with explicitly specified strategy."""

        # Mock extraction result
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.content = "Tutorial content about programming"
        mock_extraction_result.metadata = {"source_type": "test"}

        self.processor.extractor.extract_from_url.return_value = mock_extraction_result

        # Mock chunker
        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            mock_chunker.return_value = [self.mock_chunker_result]

            # Mock content analyzer
            with patch.object(
                self.processor.content_analyzer, "analyze_content"
            ) as mock_analyzer:
                mock_analysis = ContentAnalysis(
                    content_type="tutorial",
                    primary_language=None,
                    summary="Programming tutorial",
                    code_percentage=15.0,
                    code_blocks=[],
                    detected_patterns={},
                    key_concepts=["programming", "tutorial"],
                    api_references=[],
                )
                mock_analyzer.return_value = mock_analysis

                # Mock embedding strategy engine
                with patch.object(
                    self.processor.embedding_strategy_engine,
                    "generate_enhanced_embeddings",
                ) as mock_strategy:
                    mock_strategy.return_value = {
                        "strategy": "summary_enhanced",
                        "enhanced_summary": "Enhanced tutorial summary",
                        "summary_embedding": {
                            "embedding": [0.2] * 1536,
                            "model": "text-embedding-3-small",
                            "dimension": 1536,
                            "success": True,
                        },
                        "chunk_embeddings": [
                            {
                                "embedding": [0.1] * 1536,
                                "model": "text-embedding-3-small",
                                "dimension": 1536,
                                "success": True,
                            }
                        ],
                        "primary_embedding": {
                            "embedding": [0.2] * 1536,
                            "model": "text-embedding-3-small",
                            "dimension": 1536,
                            "success": True,
                        },
                    }

                    # Process with explicit strategy
                    options = {
                        "embedding_strategy": "summary_enhanced",
                        "strategy_config": {"generate_composite": False},
                    }

                    result = await self.processor.process_url(
                        "https://example.com", options
                    )

        # Verify explicit strategy was used
        assert result.success is True
        assert "summary_enhanced" in result.embedding_strategies_used

        doc = result.documents[0]
        assert doc.embedding_strategy == "summary_enhanced"

        # Verify strategy engine was called with correct parameters
        mock_strategy.assert_called_once()
        call_args = mock_strategy.call_args
        assert call_args[1]["strategy"] == EmbeddingStrategy.SUMMARY_ENHANCED

    @pytest.mark.asyncio
    async def test_analyze_chunk_quality(self):
        """Test chunk quality analysis."""

        chunk = self.mock_chunker_result

        embedding = EmbeddingResult(
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            dimension=1536,
            strategy="test",
            metadata={},
        )

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Tutorial",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        quality_metrics = await self.processor._analyze_chunk_quality(
            chunk, embedding, analysis
        )

        assert isinstance(quality_metrics, EmbeddingQualityMetrics)
        assert (
            quality_metrics.confidence_score == 1.0
        )  # Should be boosted and clamped to 1.0
        assert quality_metrics.coherence_score > 0.0

    def test_analyze_processing_quality(self):
        """Test overall processing quality analysis."""

        # Create mock documents with quality metrics
        doc1 = AdvancedProcessedDocument(
            url="https://example.com/page1",
            title="Page 1",
            content="Content 1",
            chunks=[],
            metadata={},
            embedding_strategy="hierarchical",
            quality_metrics=EmbeddingQualityMetrics(
                coherence_score=0.8,
                diversity_score=0.7,
                coverage_score=0.9,
                consistency_score=0.8,
                confidence_score=0.8,
                model_agreement={},
            ),
        )

        doc2 = AdvancedProcessedDocument(
            url="https://example.com/page2",
            title="Page 2",
            content="Content 2",
            chunks=[
                AdvancedProcessedChunk(
                    content="Chunk content",
                    primary_embedding=EmbeddingResult(
                        embedding=[0.1] * 1536,
                        model="embed-code-v3.0",
                        dimension=4096,
                        strategy="adaptive",
                        metadata={},
                    ),
                    strategy_embeddings={},
                    quality_metrics=None,
                    metadata={},
                    tokens=10,
                )
            ],
            metadata={},
            embedding_strategy="adaptive",
            quality_metrics=EmbeddingQualityMetrics(
                coherence_score=0.9,
                diversity_score=0.8,
                coverage_score=0.85,
                consistency_score=0.9,
                confidence_score=0.9,
                model_agreement={},
            ),
        )

        documents = [doc1, doc2]
        processing_stats = {"high_code_content_pages": 1}

        quality_summary, suggestions = self.processor._analyze_processing_quality(
            documents, processing_stats
        )

        # Verify quality summary
        assert "average_quality_score" in quality_summary
        assert "strategy_distribution" in quality_summary
        assert "model_usage" in quality_summary
        assert quality_summary["total_documents"] == 2

        # Check average quality
        expected_avg = (0.8 + 0.9) / 2
        assert quality_summary["average_quality_score"] == expected_avg

        # Check strategy distribution
        assert quality_summary["strategy_distribution"]["hierarchical"] == 1
        assert quality_summary["strategy_distribution"]["adaptive"] == 1

        # Check model usage
        assert quality_summary["model_usage"]["embed-code-v3.0"] == 1

        # Verify suggestions
        assert isinstance(suggestions, list)

    def test_recommend_strategy(self):
        """Test strategy recommendation based on content analysis."""

        # Test code-heavy content
        code_analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Code",
            code_percentage=60.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        strategy = self.processor._recommend_strategy(code_analysis)
        assert strategy == "hierarchical"

        # Test API reference
        api_analysis = ContentAnalysis(
            content_type="api_reference",
            primary_language="javascript",
            summary="API",
            code_percentage=20.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        strategy = self.processor._recommend_strategy(api_analysis)
        assert strategy == "composite"

        # Test content with many concepts
        concept_analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Tutorial",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ],
            api_references=[],
        )

        strategy = self.processor._recommend_strategy(concept_analysis)
        assert strategy == "summary_enhanced"

        # Test general content
        general_analysis = ContentAnalysis(
            content_type="general",
            primary_language=None,
            summary="General",
            code_percentage=0.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["general"],
            api_references=[],
        )

        strategy = self.processor._recommend_strategy(general_analysis)
        assert strategy == "adaptive"

    @pytest.mark.asyncio
    async def test_get_strategy_recommendations(self):
        """Test strategy recommendation system."""

        content = "This is a comprehensive API documentation with code examples."

        analysis = ContentAnalysis(
            content_type="api_reference",
            primary_language="javascript",
            summary="API documentation",
            code_percentage=40.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=["API", "documentation"],
            api_references=["getData()", "postData()"],
        )

        # Mock content analyzer if no analysis provided
        with patch.object(
            self.processor.content_analyzer, "analyze_content"
        ) as mock_analyzer:
            mock_analyzer.return_value = analysis

            recommendations = await self.processor.get_strategy_recommendations(
                content, analysis
            )

        assert "primary_strategy" in recommendations
        assert "alternative_strategies" in recommendations
        assert "reasoning" in recommendations
        assert "expected_performance" in recommendations

        # Should recommend composite for API reference
        assert recommendations["primary_strategy"] == "composite"

        # Should include reasoning
        assert len(recommendations["reasoning"]) > 0

        # Should include performance expectations
        assert "quality_score" in recommendations["expected_performance"]
        assert "processing_time" in recommendations["expected_performance"]
        assert "storage_efficiency" in recommendations["expected_performance"]

    def test_create_advanced_page_metadata(self):
        """Test advanced page metadata creation."""

        batch_metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "total_links_found": 10,
            "extraction_time": "2024-01-01T10:00:00",
        }

        page_info = {
            "url": "https://example.com/tutorial",
            "content": "This is a comprehensive tutorial with many concepts",
            "filename": "tutorial.md",
        }

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Comprehensive tutorial",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[
                "tutorial",
                "comprehensive",
                "concepts",
                "learning",
                "guide",
                "examples",
                "practice",
                "advanced",
                "beginner",
            ],
            api_references=[],
        )

        metadata = self.processor._create_advanced_page_metadata(
            batch_metadata, page_info, analysis
        )

        # Should exclude batch statistics
        assert "total_links_found" not in metadata

        # Should include page-specific info
        assert metadata["content_length"] == len(page_info["content"])
        assert metadata["filename"] == "tutorial.md"

        # Should include content analysis
        assert metadata["content_type"] == "tutorial"
        assert metadata["summary"] == "Comprehensive tutorial"
        assert metadata["code_percentage"] == 5.0
        assert metadata["content_analysis_available"] is True

        # Should include strategy recommendation
        assert (
            metadata["recommended_embedding_strategy"] == "summary_enhanced"
        )  # Many concepts

    @pytest.mark.asyncio
    async def test_process_url_extraction_failure(self):
        """Test processing when extraction fails."""

        # Mock extraction failure
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error = "Network timeout"

        self.processor.extractor.extract_from_url.return_value = mock_extraction_result

        result = await self.processor.process_url("https://example.com")

        assert isinstance(result, AdvancedProcessingResult)
        assert result.success is False
        assert result.error == "Network timeout"
        assert len(result.documents) == 0


class TestAdvancedDataClasses:
    """Test advanced data classes."""

    def test_advanced_processed_chunk_creation(self):
        """Test AdvancedProcessedChunk creation."""

        embedding = EmbeddingResult(
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            dimension=1536,
            strategy="hierarchical",
            metadata={},
        )

        quality_metrics = EmbeddingQualityMetrics(
            coherence_score=0.8,
            diversity_score=0.7,
            coverage_score=0.9,
            consistency_score=0.8,
            confidence_score=0.8,
            model_agreement={},
        )

        chunk = AdvancedProcessedChunk(
            content="Test content",
            primary_embedding=embedding,
            strategy_embeddings={"hierarchical": "result"},
            quality_metrics=quality_metrics,
            metadata={"test": "value"},
            tokens=10,
        )

        assert chunk.content == "Test content"
        assert chunk.primary_embedding == embedding
        assert chunk.quality_metrics == quality_metrics
        assert chunk.strategy_embeddings["hierarchical"] == "result"

    def test_advanced_processed_document_creation(self):
        """Test AdvancedProcessedDocument creation."""

        doc_embedding = EmbeddingResult(
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            dimension=1536,
            strategy="hierarchical",
            metadata={},
        )

        quality_metrics = EmbeddingQualityMetrics(
            coherence_score=0.9,
            diversity_score=0.8,
            coverage_score=0.95,
            consistency_score=0.9,
            confidence_score=0.9,
            model_agreement={},
        )

        doc = AdvancedProcessedDocument(
            url="https://example.com",
            title="Test Document",
            content="Full content",
            chunks=[],
            metadata={"source": "test"},
            document_embedding=doc_embedding,
            embedding_strategy="hierarchical",
            quality_metrics=quality_metrics,
        )

        assert doc.url == "https://example.com"
        assert doc.document_embedding == doc_embedding
        assert doc.embedding_strategy == "hierarchical"
        assert doc.quality_metrics == quality_metrics

    def test_advanced_processing_result_creation(self):
        """Test AdvancedProcessingResult creation."""

        result = AdvancedProcessingResult(
            documents=[],
            success=True,
            embedding_strategies_used=["hierarchical", "adaptive"],
            total_chunks=50,
            processing_stats={"total_pages": 10, "average_quality_score": 0.85},
            quality_summary={
                "overall_quality": "high",
                "model_distribution": {"openai": 30, "cohere": 20},
            },
            optimization_suggestions=[
                "Consider using composite strategy for API documentation",
                "Summary enhancement recommended for long documents",
            ],
        )

        assert result.success is True
        assert len(result.embedding_strategies_used) == 2
        assert result.total_chunks == 50
        assert result.processing_stats["average_quality_score"] == 0.85
        assert len(result.optimization_suggestions) == 2
