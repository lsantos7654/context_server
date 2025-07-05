"""Unit tests for document processing functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.content_analysis import ContentAnalysis
from context_server.core.processing import (
    DocumentProcessor,
    ProcessedChunk,
    ProcessedDocument,
    ProcessingResult,
)


class TestDocumentProcessor:
    """Test the DocumentProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock embedding service to avoid API calls in tests
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.embed_batch = AsyncMock(
            return_value=[[0.1] * 1536] * 5
        )

        self.processor = DocumentProcessor(
            embedding_service=self.mock_embedding_service
        )

    def test_create_page_metadata_basic(self):
        """Test basic page metadata creation."""
        batch_metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "total_links_found": 100,
            "filtered_links": 50,
            "successful_extractions": 45,
            "extraction_time": "2024-01-01T10:00:00",
            "extracted_pages": [],
        }

        page_info = {
            "url": "https://example.com/page1",
            "content": "This is a sample page with some content.",
            "filename": "page1.md",
        }

        page_metadata = self.processor._create_page_metadata(batch_metadata, page_info)

        # Should include base metadata but exclude batch statistics
        assert page_metadata["source_type"] == "crawl4ai"
        assert page_metadata["base_url"] == "https://example.com"
        assert page_metadata["content_length"] == len(page_info["content"])
        assert page_metadata["filename"] == "page1.md"

        # Should include content analysis fields
        assert "content_type" in page_metadata
        assert "summary" in page_metadata

        # Should NOT include batch statistics
        assert "total_links_found" not in page_metadata
        assert "filtered_links" not in page_metadata
        assert "successful_extractions" not in page_metadata
        assert "extracted_pages" not in page_metadata

    @patch("context_server.core.processing.ContentAnalyzer")
    def test_create_page_metadata_with_content_analysis(self, mock_analyzer_class):
        """Test page metadata creation with content analysis."""
        # Mock content analyzer
        mock_analyzer = Mock()
        mock_analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language="python",
            summary="A Python tutorial about functions",
            code_percentage=35.0,
            code_blocks=[],
            detected_patterns={"async": ["async", "await"]},
            key_concepts=["Python", "Functions"],
            api_references=["print()", "len()"],
        )
        mock_analyzer.analyze_content.return_value = mock_analysis
        mock_analyzer_class.return_value = mock_analyzer

        # Create new processor to use mocked analyzer
        processor = DocumentProcessor(embedding_service=self.mock_embedding_service)

        batch_metadata = {
            "source_type": "crawl4ai",
            "extraction_time": "2024-01-01T10:00:00",
        }

        page_info = {
            "url": "https://example.com/tutorial",
            "content": "def hello(): print('Hello World')",
            "filename": "tutorial.md",
        }

        page_metadata = processor._create_page_metadata(batch_metadata, page_info)

        # Should include content analysis results
        assert page_metadata["content_type"] == "tutorial"
        assert page_metadata["primary_language"] == "python"
        assert page_metadata["summary"] == "A Python tutorial about functions"
        assert page_metadata["code_percentage"] == 35.0
        assert page_metadata["detected_patterns"] == {"async": ["async", "await"]}
        assert page_metadata["key_concepts"] == ["Python", "Functions"]
        assert page_metadata["api_references"] == ["print()", "len()"]
        assert page_metadata["code_blocks_count"] == 0

    @patch("context_server.core.processing.ContentAnalyzer")
    def test_create_page_metadata_with_code_analysis(self, mock_analyzer_class):
        """Test page metadata creation with code analysis details."""
        # Mock content analyzer
        mock_analyzer = Mock()

        # Create mock code blocks
        from context_server.core.content_analysis import CodeBlock

        mock_code_blocks = [
            CodeBlock(
                language="python",
                content="def test(): pass",
                start_line=1,
                end_line=2,
                functions=["test", "main"],
                classes=["TestClass"],
                imports=["sys", "os"],
            )
        ]

        mock_analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code example",
            code_percentage=65.0,  # High code percentage
            code_blocks=mock_code_blocks,
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )
        mock_analyzer.analyze_content.return_value = mock_analysis
        mock_analyzer_class.return_value = mock_analyzer

        processor = DocumentProcessor(embedding_service=self.mock_embedding_service)

        batch_metadata = {"source_type": "crawl4ai"}
        page_info = {
            "url": "https://example.com/code",
            "content": "def test(): pass\nclass TestClass: pass",
            "filename": "code.md",
        }

        page_metadata = processor._create_page_metadata(batch_metadata, page_info)

        # Should include code analysis for high code percentage
        assert page_metadata["code_percentage"] == 65.0
        assert page_metadata["code_blocks_count"] == 1
        assert "code_analysis" in page_metadata

        code_analysis = page_metadata["code_analysis"]
        assert "test" in code_analysis["functions"]
        assert "main" in code_analysis["functions"]
        assert "TestClass" in code_analysis["classes"]
        assert "sys" in code_analysis["imports"]
        assert "os" in code_analysis["imports"]
        assert "python" in code_analysis["languages"]

    @patch("context_server.core.processing.ContentAnalyzer")
    def test_create_page_metadata_analysis_failure(self, mock_analyzer_class):
        """Test page metadata creation when content analysis fails."""
        # Mock content analyzer to raise exception
        mock_analyzer = Mock()
        mock_analyzer.analyze_content.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer

        processor = DocumentProcessor(embedding_service=self.mock_embedding_service)

        batch_metadata = {"source_type": "crawl4ai"}
        page_info = {
            "url": "https://example.com/broken",
            "content": "Some content",
            "filename": "broken.md",
        }

        page_metadata = processor._create_page_metadata(batch_metadata, page_info)

        # Should have fallback values when analysis fails
        assert page_metadata["content_type"] == "general"
        assert page_metadata["primary_language"] is None
        assert page_metadata["summary"] == "Content analysis unavailable"
        assert page_metadata["code_percentage"] == 0.0
        assert page_metadata["detected_patterns"] == {}
        assert page_metadata["key_concepts"] == []
        assert page_metadata["api_references"] == []
        assert page_metadata["code_blocks_count"] == 0

    def test_create_title_from_url_simple(self):
        """Test title creation from simple URL."""
        page_url = "https://docs.example.com/api/users"
        base_url = "https://docs.example.com"

        title = self.processor._create_title_from_url(page_url, base_url)

        assert "Users" in title
        assert "docs.example.com" in title

    def test_create_title_from_url_complex(self):
        """Test title creation from complex URL."""
        page_url = "https://api.example.com/docs/getting-started/quick-setup"
        base_url = "https://api.example.com"

        title = self.processor._create_title_from_url(page_url, base_url)

        assert "Quick Setup" in title or "quick-setup" in title
        assert "api.example.com" in title

    def test_create_title_from_url_root(self):
        """Test title creation from root URL."""
        page_url = "https://example.com/"
        base_url = "https://example.com"

        title = self.processor._create_title_from_url(page_url, base_url)

        assert "Documentation from" in title
        assert "example.com" in title

    def test_create_title_from_url_error_handling(self):
        """Test title creation handles malformed URLs."""
        page_url = "not-a-valid-url"
        base_url = "https://example.com"

        title = self.processor._create_title_from_url(page_url, base_url)

        # Should handle error gracefully and still return a title
        assert title is not None
        assert len(title) > 0

    @pytest.mark.asyncio
    async def test_process_content_basic(self):
        """Test basic content processing."""
        content = "This is some test content for processing."
        url = "https://example.com/test"
        title = "Test Document"
        metadata = {"source_type": "test"}

        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            # Mock chunker to return simple chunks
            from context_server.core.chunking import TextChunk

            mock_chunks = [
                TextChunk(
                    content="This is some test content",
                    tokens=25,
                    metadata={"chunk_index": 0},
                    start_line=1,
                    end_line=1,
                ),
                TextChunk(
                    content="for processing.",
                    tokens=15,
                    metadata={"chunk_index": 1},
                    start_line=2,
                    end_line=2,
                ),
            ]
            mock_chunker.return_value = mock_chunks

            result = await self.processor._process_content(
                content, url, title, metadata
            )

            assert isinstance(result, ProcessedDocument)
            assert result.url == url
            assert result.title == title
            assert result.content == content
            assert result.metadata == metadata
            assert len(result.chunks) == 2

            # Check chunks
            for i, chunk in enumerate(result.chunks):
                assert isinstance(chunk, ProcessedChunk)
                assert chunk.content == mock_chunks[i].content
                assert chunk.tokens == mock_chunks[i].tokens
                assert chunk.embedding == [0.1] * 1536  # Mocked embedding
                assert chunk.metadata["source_url"] == url
                assert chunk.metadata["source_title"] == title

    @pytest.mark.asyncio
    async def test_process_content_embedding_failure(self):
        """Test content processing when embeddings fail."""
        content = "Test content"
        url = "https://example.com/test"
        title = "Test"
        metadata = {}

        # Mock embedding service to fail
        self.mock_embedding_service.embed_batch.side_effect = Exception("API Error")

        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            from context_server.core.chunking import TextChunk

            mock_chunks = [
                TextChunk(
                    content="Test content", tokens=10, metadata={"chunk_index": 0}
                )
            ]
            mock_chunker.return_value = mock_chunks

            result = await self.processor._process_content(
                content, url, title, metadata
            )

            # Should create dummy embeddings and continue
            assert len(result.chunks) == 1
            assert result.chunks[0].embedding == [0.0] * 1536  # Dummy embedding

    @pytest.mark.asyncio
    async def test_process_url_with_individual_pages(self):
        """Test URL processing with individual pages."""
        # Mock the extractor directly on the processor instance
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.content = "Combined content"
        mock_extraction_result.metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com",
            "total_links_found": 5,
            "successful_extractions": 3,
            "extracted_pages": [
                {
                    "url": "https://example.com/page1",
                    "content": "Page 1 content",
                    "filename": "page1.md",
                },
                {
                    "url": "https://example.com/page2",
                    "content": "Page 2 content",
                    "filename": "page2.md",
                },
            ],
        }

        # Directly mock the extractor on the processor
        self.processor.extractor.extract_from_url = AsyncMock(
            return_value=mock_extraction_result
        )

        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            from context_server.core.chunking import TextChunk

            mock_chunker.return_value = [
                TextChunk(content="Test chunk", tokens=10, metadata={"chunk_index": 0})
            ]

            result = await self.processor.process_url("https://example.com")

            assert result.success is True
            assert len(result.documents) == 2  # One for each page

            # Check first document
            doc1 = result.documents[0]
            assert doc1.url == "https://example.com/page1"
            assert doc1.content == "Page 1 content"
            assert doc1.metadata["page_url"] == "https://example.com/page1"
            assert doc1.metadata["is_individual_page"] is True
            assert doc1.metadata["extraction_success"] is True

            # Should not contain batch statistics
            assert "total_links_found" not in doc1.metadata
            assert "successful_extractions" not in doc1.metadata

    @pytest.mark.asyncio
    async def test_process_url_extraction_failure(self):
        """Test URL processing when extraction fails."""
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error = "Extraction failed"

        self.processor.extractor.extract_from_url = AsyncMock(
            return_value=mock_extraction_result
        )

        result = await self.processor.process_url("https://example.com")

        assert result.success is False
        assert result.error == "Extraction failed"
        assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_process_url_no_individual_pages(self):
        """Test URL processing without individual pages (fallback to combined content)."""
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.content = "Combined content from all pages"
        mock_extraction_result.metadata = {
            "source_type": "crawl4ai",
            "base_url": "https://example.com"
            # No extracted_pages field
        }

        self.processor.extractor.extract_from_url = AsyncMock(
            return_value=mock_extraction_result
        )

        with patch.object(self.processor.chunker, "chunk_text") as mock_chunker:
            from context_server.core.chunking import TextChunk

            mock_chunker.return_value = [
                TextChunk(
                    content="Combined content", tokens=20, metadata={"chunk_index": 0}
                )
            ]

            result = await self.processor.process_url("https://example.com")

            assert result.success is True
            assert len(result.documents) == 1  # Single combined document

            doc = result.documents[0]
            assert doc.url == "https://example.com"
            assert doc.content == "Combined content from all pages"
            assert doc.title == "Documentation from https://example.com"


class TestProcessingDataClasses:
    """Test the processing data classes."""

    def test_processed_chunk_creation(self):
        """Test ProcessedChunk creation."""
        chunk = ProcessedChunk(
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "value"},
            tokens=10,
            start_line=1,
            end_line=2,
            char_start=0,
            char_end=12,
        )

        assert chunk.content == "Test content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata == {"test": "value"}
        assert chunk.tokens == 10
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.char_start == 0
        assert chunk.char_end == 12

    def test_processed_document_creation(self):
        """Test ProcessedDocument creation."""
        chunks = [
            ProcessedChunk(
                content="Chunk 1", embedding=[0.1] * 1536, metadata={}, tokens=5
            )
        ]

        doc = ProcessedDocument(
            url="https://example.com",
            title="Test Document",
            content="Full content",
            chunks=chunks,
            metadata={"source": "test"},
        )

        assert doc.url == "https://example.com"
        assert doc.title == "Test Document"
        assert doc.content == "Full content"
        assert len(doc.chunks) == 1
        assert doc.metadata == {"source": "test"}

    def test_processing_result_success(self):
        """Test ProcessingResult for successful processing."""
        docs = [
            ProcessedDocument(
                url="https://example.com",
                title="Test",
                content="Content",
                chunks=[],
                metadata={},
            )
        ]

        result = ProcessingResult(documents=docs, success=True)

        assert result.success is True
        assert result.error is None
        assert len(result.documents) == 1

    def test_processing_result_failure(self):
        """Test ProcessingResult for failed processing."""
        result = ProcessingResult(
            documents=[], success=False, error="Processing failed"
        )

        assert result.success is False
        assert result.error == "Processing failed"
        assert len(result.documents) == 0
