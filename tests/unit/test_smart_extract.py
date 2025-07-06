"""
Unit tests for smart_extract module.
"""

from unittest.mock import patch

import pytest
from src.core.crawl4ai_extraction import ExtractionResult
from src.smart_extract import SmartExtractor


class TestSmartExtractor:
    """Test SmartExtractor class."""

    def test_extractor_initialization(self, tmp_path):
        """Test extractor initialization."""
        extractor = SmartExtractor(str(tmp_path))

        assert extractor.output_dir == tmp_path
        assert extractor.clean is True
        assert extractor.max_pages == 50
        assert extractor.extractor is not None
        assert extractor.cleaner is not None

    def test_extractor_initialization_no_clean(self, tmp_path):
        """Test extractor initialization without cleaning."""
        extractor = SmartExtractor(str(tmp_path), clean=False)

        assert extractor.clean is False
        assert extractor.cleaner is None

    def test_extractor_initialization_custom_max_pages(self, tmp_path):
        """Test extractor initialization with custom max pages."""
        extractor = SmartExtractor(str(tmp_path), max_pages=100)

        assert extractor.max_pages == 100

    def test_output_directory_created(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "custom_output"
        SmartExtractor(str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    @pytest.mark.asyncio
    async def test_extract_success(self, tmp_path):
        """Test successful extraction."""
        extractor = SmartExtractor(str(tmp_path))

        # Mock the extractor
        mock_result = ExtractionResult(
            success=True,
            content="# Test Content\n\nSome test content.",
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = await extractor.extract("https://example.com")

            assert result is True

    @pytest.mark.asyncio
    async def test_extract_failure(self, tmp_path):
        """Test failed extraction."""
        extractor = SmartExtractor(str(tmp_path))

        # Mock failed result
        mock_result = ExtractionResult(success=False, error="Test error")

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = await extractor.extract("https://example.com")

            assert result is False

    @pytest.mark.asyncio
    async def test_extract_with_cleaning(self, tmp_path):
        """Test extraction with cleaning enabled."""
        extractor = SmartExtractor(str(tmp_path), clean=True)

        mock_result = ExtractionResult(
            success=True,
            content="# Test Content\n\n\n\nSome content with extra spaces   \n",
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = await extractor.extract("https://example.com")

            assert result is True

            # Check that cleaned content was saved
            output_file = tmp_path / "extracted_content.md"
            assert output_file.exists()

            content = output_file.read_text()
            # Should have cleaned trailing spaces
            lines = content.split("\n")
            for line in lines:
                if line:
                    assert not line.endswith(" ")

    @pytest.mark.asyncio
    async def test_extract_without_cleaning(self, tmp_path):
        """Test extraction without cleaning."""
        extractor = SmartExtractor(str(tmp_path), clean=False)

        mock_result = ExtractionResult(
            success=True,
            content="# Test Content",
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = await extractor.extract("https://example.com")

            assert result is True

    @pytest.mark.asyncio
    async def test_extract_no_content(self, tmp_path):
        """Test extraction with no content."""
        extractor = SmartExtractor(str(tmp_path))

        mock_result = ExtractionResult(
            success=True,
            content=None,
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = await extractor.extract("https://example.com")

            assert result is True

    def test_extract_sync(self, tmp_path):
        """Test synchronous extraction wrapper."""
        extractor = SmartExtractor(str(tmp_path))

        mock_result = ExtractionResult(
            success=True,
            content="# Test Content",
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = extractor.extract_sync("https://example.com")

            assert result is True

    def test_extract_sync_failure(self, tmp_path):
        """Test synchronous extraction failure."""
        extractor = SmartExtractor(str(tmp_path))

        mock_result = ExtractionResult(success=False, error="Test error")

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ):
            result = extractor.extract_sync("https://example.com")

            assert result is False

    def test_default_output_directory(self):
        """Test default output directory creation."""
        extractor = SmartExtractor()

        assert extractor.output_dir.name == "output"
        assert extractor.output_dir.exists()

        # Cleanup
        import shutil

        shutil.rmtree("output")

    @pytest.mark.asyncio
    async def test_extract_max_pages_passed(self, tmp_path):
        """Test that max_pages is passed to extractor."""
        extractor = SmartExtractor(str(tmp_path), max_pages=25)

        mock_result = ExtractionResult(
            success=True,
            content="# Test Content",
            metadata={"source_type": "test", "successful_extractions": 1},
        )

        with patch.object(
            extractor.extractor, "extract_from_url", return_value=mock_result
        ) as mock_extract:
            await extractor.extract("https://example.com")

            # Verify max_pages was passed
            mock_extract.assert_called_once_with("https://example.com", max_pages=25)
