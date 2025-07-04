"""
Integration tests for extraction functionality.
"""

import tempfile
from pathlib import Path

import pytest

from src.smart_extract import SmartExtractor


class TestExtractionIntegration:
    """Integration tests for the extraction system."""

    @pytest.mark.integration
    def test_full_extraction_pipeline(self):
        """Test the complete extraction pipeline with a simple example."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = SmartExtractor(tmp_dir, max_pages=1)

            # Test with a simple, reliable URL
            success = extractor.extract_sync("https://example.com")

            assert success is True

            # Check that output files were created
            output_dir = Path(tmp_dir)
            assert output_dir.exists()

            # Should have created some output files
            output_files = list(output_dir.glob("*.md"))
            assert len(output_files) > 0

            # Check that cleaned content was created
            cleaned_file = output_dir / "extracted_content.md"
            if cleaned_file.exists():
                content = cleaned_file.read_text()
                assert len(content) > 0
                assert "Example Domain" in content or "example" in content.lower()

    @pytest.mark.integration
    def test_extraction_without_cleaning(self):
        """Test extraction without cleaning enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = SmartExtractor(tmp_dir, clean=False, max_pages=1)

            success = extractor.extract_sync("https://example.com")

            assert success is True

            # Should still create output files
            output_dir = Path(tmp_dir)
            output_files = list(output_dir.glob("*.md"))
            assert len(output_files) > 0

    @pytest.mark.integration
    def test_extraction_invalid_url(self):
        """Test extraction with invalid URL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = SmartExtractor(tmp_dir, max_pages=1)

            # This should fail gracefully
            success = extractor.extract_sync(
                "https://this-domain-does-not-exist-12345.com"
            )

            # Should handle failure gracefully
            assert success is False

    @pytest.mark.integration
    def test_extraction_with_custom_max_pages(self):
        """Test extraction with custom max pages setting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = SmartExtractor(tmp_dir, max_pages=2)

            success = extractor.extract_sync("https://example.com")

            assert success is True
            assert extractor.max_pages == 2
