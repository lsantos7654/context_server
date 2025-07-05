"""Unit tests for enhanced document processing with multi-embedding support."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.content_analysis import ContentAnalysis
from context_server.core.enhanced_processing import (
    EnhancedDocumentProcessor,
    EnhancedProcessedChunk,
    EnhancedProcessedDocument,
    EnhancedProcessingResult,
)
from context_server.core.multi_embedding_service import (
    ContentType,
    EmbeddingModel,
    MultiEmbeddingService,
)


class TestEnhancedDocumentProcessor:
    """Test enhanced document processor with multi-embedding support."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock multi-embedding service
        self.mock_embedding_service = Mock(spec=MultiEmbeddingService)
        self.mock_embedding_service.route_content.return_value = (
            EmbeddingModel.OPENAI_SMALL
        )
        self.mock_embedding_service.embed_batch = AsyncMock(
            return_value=[
                {
                    "embedding": [0.1] * 1536,
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                    "success": True,
                }
            ]
        )
        # Mock routing rules for get_embedding_stats test
        self.mock_embedding_service.routing_rules = {
            ContentType.CODE: EmbeddingModel.VOYAGE_CODE,
            ContentType.TUTORIAL: EmbeddingModel.OPENAI_SMALL,
        }

        # Create processor with mocked service
        self.processor = EnhancedDocumentProcessor(
            multi_embedding_service=self.mock_embedding_service,
            enable_multi_embedding=False,
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
    async def test_process_url_with_individual_pages_enhanced(self):
        """Test enhanced URL processing with individual pages."""

        # Mock extraction result with individual pages
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.content = "Combined content"
        mock_extraction_result.metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "total_links_found": 3,
            "successful_extractions": 2,
            "extracted_pages": [
                {
                    "url": "https://example.com/page1",
                    "content": "def hello():\n    print('Hello World')",
                    "filename": "page1.md",
                },
                {
                    "url": "https://example.com/page2",
                    "content": "This is a tutorial about Python functions.",
                    "filename": "page2.md",
                },
            ],
        }

        self.processor.extractor.extract_from_url.return_value = mock_extraction_result

        # Mock chunker to return our test chunk
        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            mock_chunker.return_value = [self.mock_chunker_result]

            # Mock content analyzer
            with patch.object(
                self.processor.content_analyzer, "analyze_content"
            ) as mock_analyzer:
                # Return different analyses for different content
                def analyze_side_effect(content):
                    if "def hello" in content:
                        # High code content for first page
                        return ContentAnalysis(
                            content_type="code_example",
                            primary_language="python",
                            summary="Python function example",
                            code_percentage=80.0,
                            code_blocks=[],
                            detected_patterns={"functions": ["hello"]},
                            key_concepts=["Python", "functions"],
                            api_references=["print()"],
                        )
                    else:
                        # Low code content for second page
                        return ContentAnalysis(
                            content_type="tutorial",
                            primary_language=None,
                            summary="Tutorial about Python functions",
                            code_percentage=5.0,
                            code_blocks=[],
                            detected_patterns={},
                            key_concepts=["Python", "tutorial"],
                            api_references=[],
                        )

                mock_analyzer.side_effect = analyze_side_effect

                result = await self.processor.process_url("https://example.com")

        # Verify result structure
        assert isinstance(result, EnhancedProcessingResult)
        assert result.success is True
        assert len(result.documents) == 2
        assert result.total_chunks == 2
        assert result.embedding_models_used == ["text-embedding-3-small"]

        # Check processing stats
        assert result.processing_stats["total_pages"] == 2
        assert (
            result.processing_stats["high_code_content_pages"] == 1
        )  # First page has 80% code
        assert "code_example" in result.processing_stats["content_types_detected"]

        # Verify first document (code)
        doc1 = result.documents[0]
        assert isinstance(doc1, EnhancedProcessedDocument)
        assert doc1.url == "https://example.com/page1"
        assert doc1.content_analysis.content_type == "code_example"
        assert doc1.content_analysis.code_percentage == 80.0
        assert doc1.primary_embedding_model == "text-embedding-3-small"

        # Verify chunks
        assert len(doc1.chunks) == 1
        chunk = doc1.chunks[0]
        assert isinstance(chunk, EnhancedProcessedChunk)
        assert chunk.content == "Test chunk content"
        assert "text-embedding-3-small" in chunk.embeddings
        assert len(chunk.embeddings["text-embedding-3-small"]) == 1536

    @pytest.mark.asyncio
    async def test_process_url_with_multi_embedding_enabled(self):
        """Test processing with multi-embedding enabled."""

        # Enable multi-embedding
        processor = EnhancedDocumentProcessor(
            multi_embedding_service=self.mock_embedding_service,
            enable_multi_embedding=True,
        )
        processor.extractor.extract_from_url = AsyncMock()

        # Mock multi-embedding results
        mock_multi_results = [
            {
                "embedding": [0.1] * 1536,
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "success": True,
            },
            {
                "embedding": [0.2] * 4096,
                "model": "embed-code-v3.0",
                "dimension": 4096,
                "success": True,
            },
        ]

        # Mock the multi-embedding generation method
        with patch.object(
            processor, "_generate_multi_embeddings_batch"
        ) as mock_multi_embed:
            mock_multi_embed.return_value = [
                {
                    "embeddings": {
                        "text-embedding-3-small": [0.1] * 1536,
                        "embed-code-v3.0": [0.2] * 4096,
                    },
                    "metadata": {
                        "text-embedding-3-small": {
                            "dimension": 1536,
                            "success": True,
                            "model": "text-embedding-3-small",
                        },
                        "embed-code-v3.0": {
                            "dimension": 4096,
                            "success": True,
                            "model": "embed-code-v3.0",
                        },
                    },
                }
            ]

            # Mock extraction result
            mock_extraction_result = Mock()
            mock_extraction_result.success = True
            mock_extraction_result.content = "Test content"
            mock_extraction_result.metadata = {"source_type": "test"}
            processor.extractor.extract_from_url.return_value = mock_extraction_result

            # Mock chunker
            with patch.object(processor.chunker, "chunk_text") as mock_chunker:
                mock_chunker.return_value = [self.mock_chunker_result]

                # Mock content analyzer
                with patch.object(
                    processor.content_analyzer, "analyze_content"
                ) as mock_analyzer:
                    mock_analysis = ContentAnalysis(
                        content_type="general",
                        primary_language=None,
                        summary="Test content",
                        code_percentage=0.0,
                        code_blocks=[],
                        detected_patterns={},
                        key_concepts=[],
                        api_references=[],
                    )
                    mock_analyzer.return_value = mock_analysis

                    result = await processor.process_url("https://example.com")

        # Verify multi-embedding results
        assert result.success is True
        assert len(result.documents) == 1

        chunk = result.documents[0].chunks[0]
        assert len(chunk.embeddings) == 2
        assert "text-embedding-3-small" in chunk.embeddings
        assert "embed-code-v3.0" in chunk.embeddings
        assert len(chunk.embeddings["text-embedding-3-small"]) == 1536
        assert len(chunk.embeddings["embed-code-v3.0"]) == 4096

    @pytest.mark.asyncio
    async def test_generate_optimal_embeddings_batch(self):
        """Test optimal embedding generation with content-aware routing."""

        chunks = [self.mock_chunker_result]

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=60.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        processing_stats = {"embedding_models_used": set()}

        # Mock embedding service response
        self.mock_embedding_service.embed_batch.return_value = [
            {
                "embedding": [0.2] * 4096,
                "model": "embed-code-v3.0",
                "dimension": 4096,
                "success": True,
            }
        ]

        results = await self.processor._generate_optimal_embeddings_batch(
            chunks, analysis, processing_stats
        )

        assert len(results) == 1
        result = results[0]

        assert "embed-code-v3.0" in result["embeddings"]
        assert len(result["embeddings"]["embed-code-v3.0"]) == 4096
        assert result["metadata"]["embed-code-v3.0"]["success"] is True
        assert "embed-code-v3.0" in processing_stats["embedding_models_used"]

    @pytest.mark.asyncio
    async def test_generate_multi_embeddings_batch(self):
        """Test multi-embedding generation for comparison."""

        chunks = [self.mock_chunker_result]

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=60.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        processing_stats = {"embedding_models_used": set()}

        # Mock multiple embedding responses
        self.mock_embedding_service.embed_batch.side_effect = [
            # OpenAI response
            [
                {
                    "embedding": [0.1] * 1536,
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                    "success": True,
                }
            ],
            # Cohere response
            [
                {
                    "embedding": [0.2] * 4096,
                    "model": "embed-code-v3.0",
                    "dimension": 4096,
                    "success": True,
                }
            ],
        ]

        results = await self.processor._generate_multi_embeddings_batch(
            chunks, analysis, processing_stats
        )

        assert len(results) == 1
        result = results[0]

        # Should have embeddings from both models
        assert "text-embedding-3-small" in result["embeddings"]
        assert "embed-code-v3.0" in result["embeddings"]
        assert len(result["embeddings"]["text-embedding-3-small"]) == 1536
        assert len(result["embeddings"]["embed-code-v3.0"]) == 4096

        # Check metadata for both models
        assert result["metadata"]["text-embedding-3-small"]["success"] is True
        assert result["metadata"]["embed-code-v3.0"]["success"] is True

    def test_create_enhanced_page_metadata(self):
        """Test enhanced page metadata creation with content analysis."""

        batch_metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "total_links_found": 10,
            "successful_extractions": 8,
            "extraction_time": "2024-01-01T10:00:00",
        }

        page_info = {
            "url": "https://example.com/api",
            "content": "API documentation with function definitions",
            "filename": "api.md",
        }

        analysis = ContentAnalysis(
            content_type="api_reference",
            primary_language="javascript",
            summary="JavaScript API documentation",
            code_percentage=30.0,
            code_blocks=[],
            detected_patterns={"async": ["async", "await"]},
            key_concepts=["API", "JavaScript", "functions"],
            api_references=["getData()", "postData()"],
        )

        metadata = self.processor._create_enhanced_page_metadata(
            batch_metadata, page_info, analysis
        )

        # Should exclude batch statistics
        assert "total_links_found" not in metadata
        assert "successful_extractions" not in metadata

        # Should include page-specific info
        assert metadata["content_length"] == len(page_info["content"])
        assert metadata["filename"] == "api.md"

        # Should include content analysis
        assert metadata["content_type"] == "api_reference"
        assert metadata["primary_language"] == "javascript"
        assert metadata["summary"] == "JavaScript API documentation"
        assert metadata["code_percentage"] == 30.0
        assert metadata["detected_patterns"] == {"async": ["async", "await"]}
        assert metadata["key_concepts"] == ["API", "JavaScript", "functions"]
        assert metadata["api_references"] == ["getData()", "postData()"]
        assert metadata["content_analysis_available"] is True

        # Should include code analysis for high code percentage
        assert "code_analysis" in metadata

    def test_create_enhanced_page_metadata_analysis_failure(self):
        """Test enhanced metadata creation when content analysis fails."""

        batch_metadata = {"source_type": "test"}
        page_info = {
            "url": "https://example.com/broken",
            "content": "Some content",
            "filename": "broken.md",
        }

        metadata = self.processor._create_enhanced_page_metadata(
            batch_metadata, page_info, None  # No analysis
        )

        # Should have fallback values
        assert metadata["content_type"] == "general"
        assert metadata["primary_language"] is None
        assert metadata["summary"] == "Content analysis unavailable"
        assert metadata["code_percentage"] == 0.0
        assert metadata["detected_patterns"] == {}
        assert metadata["key_concepts"] == []
        assert metadata["api_references"] == []
        assert metadata["content_analysis_available"] is False

    @pytest.mark.asyncio
    async def test_get_embedding_stats(self):
        """Test getting embedding service statistics."""

        # Mock health check and available models
        self.mock_embedding_service.health_check = AsyncMock(
            return_value={"text-embedding-3-small": True, "embed-code-v3.0": True}
        )
        self.mock_embedding_service.get_available_models.return_value = [
            "text-embedding-3-small",
            "embed-code-v3.0",
        ]

        stats = await self.processor.get_embedding_stats()

        assert "available_models" in stats
        assert "health_status" in stats
        assert "multi_embedding_enabled" in stats
        assert "routing_rules" in stats

        assert stats["multi_embedding_enabled"] is False
        assert "text-embedding-3-small" in stats["available_models"]
        assert stats["health_status"]["text-embedding-3-small"] is True

    @pytest.mark.asyncio
    async def test_process_url_extraction_failure(self):
        """Test enhanced processing when extraction fails."""

        # Mock extraction failure
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error = "Network error"

        self.processor.extractor.extract_from_url.return_value = mock_extraction_result

        result = await self.processor.process_url("https://example.com")

        assert isinstance(result, EnhancedProcessingResult)
        assert result.success is False
        assert result.error == "Network error"
        assert len(result.documents) == 0


class TestEnhancedDataClasses:
    """Test enhanced data classes."""

    def test_enhanced_processed_chunk_creation(self):
        """Test EnhancedProcessedChunk creation."""

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Test",
            code_percentage=50.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        chunk = EnhancedProcessedChunk(
            content="Test content",
            embeddings={
                "text-embedding-3-small": [0.1] * 1536,
                "embed-code-v3.0": [0.2] * 4096,
            },
            embedding_metadata={
                "text-embedding-3-small": {"dimension": 1536, "success": True},
                "embed-code-v3.0": {"dimension": 4096, "success": True},
            },
            metadata={"test": "value"},
            tokens=10,
            content_analysis=analysis,
        )

        assert chunk.content == "Test content"
        assert len(chunk.embeddings) == 2
        assert "text-embedding-3-small" in chunk.embeddings
        assert chunk.content_analysis.content_type == "code_example"

    def test_enhanced_processed_document_creation(self):
        """Test EnhancedProcessedDocument creation."""

        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Tutorial content",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        chunk = EnhancedProcessedChunk(
            content="Chunk content",
            embeddings={"text-embedding-3-small": [0.1] * 1536},
            embedding_metadata={"text-embedding-3-small": {"dimension": 1536}},
            metadata={},
            tokens=5,
        )

        doc = EnhancedProcessedDocument(
            url="https://example.com",
            title="Test Document",
            content="Full content",
            chunks=[chunk],
            metadata={"source": "test"},
            content_analysis=analysis,
            primary_embedding_model="text-embedding-3-small",
        )

        assert doc.url == "https://example.com"
        assert doc.content_analysis.content_type == "tutorial"
        assert doc.primary_embedding_model == "text-embedding-3-small"
        assert len(doc.chunks) == 1

    def test_enhanced_processing_result_creation(self):
        """Test EnhancedProcessingResult creation."""

        result = EnhancedProcessingResult(
            documents=[],
            success=True,
            embedding_models_used=["text-embedding-3-small", "embed-code-v3.0"],
            total_chunks=25,
            processing_stats={
                "total_pages": 5,
                "high_code_content_pages": 2,
                "content_types_detected": ["tutorial", "code_example"],
            },
        )

        assert result.success is True
        assert len(result.embedding_models_used) == 2
        assert result.total_chunks == 25
        assert result.processing_stats["total_pages"] == 5
