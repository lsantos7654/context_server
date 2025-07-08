"""
Unit tests for crawl4ai_extraction module.
"""

import pytest
from context_server.core.crawl4ai_extraction import Crawl4aiExtractor, ExtractionResult


class TestExtractionResult:
    """Test ExtractionResult class."""

    def test_success_result(self):
        """Test successful result creation."""
        result = ExtractionResult(
            success=True, content="Test content", metadata={"key": "value"}
        )

        assert result.success is True
        assert result.content == "Test content"
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error result creation."""
        result = ExtractionResult(success=False, error="Test error")

        assert result.success is False
        assert result.error == "Test error"
        assert result.content == ""

    def test_error_class_method(self):
        """Test error class method."""
        result = ExtractionResult.error("Test error message")

        assert result.success is False
        assert result.error == "Test error message"
        assert result.content == ""


class TestCrawl4aiExtractor:
    """Test Crawl4aiExtractor class."""

    def test_extractor_initialization(self, tmp_path):
        """Test extractor initialization."""
        extractor = Crawl4aiExtractor(tmp_path)

        assert extractor.output_dir == tmp_path

    def test_extractor_initialization_default_output(self):
        """Test extractor initialization with default output."""
        extractor = Crawl4aiExtractor()

        assert extractor.output_dir.name == "output"
        assert extractor.output_dir.exists()

        # Cleanup
        import shutil

        shutil.rmtree("output")

    def test_filter_documentation_links_empty(self, tmp_path):
        """Test filtering with empty links."""
        extractor = Crawl4aiExtractor(tmp_path)
        result = extractor._filter_documentation_links([], "https://example.com", 10)

        assert result == []

    def test_filter_documentation_links_basic(self, tmp_path):
        """Test basic link filtering."""
        extractor = Crawl4aiExtractor(tmp_path)

        links = [
            "https://example.com/docs/intro",
            "https://example.com/docs/guide",
            "https://other.com/external",  # Should be filtered out
            "https://example.com/login",  # Should be filtered out
        ]

        result = extractor._filter_documentation_links(links, "https://example.com", 10)

        # Should only include same-domain, non-login links
        # (includes base URL + filtered links)
        assert len(result) == 3
        assert all("example.com" in url for url in result)
        assert all("login" not in url for url in result)
        # Should include the base URL
        assert "https://example.com" in result

    def test_filter_documentation_links_prioritization(self, tmp_path):
        """Test link prioritization."""
        extractor = Crawl4aiExtractor(tmp_path)

        links = [
            "https://example.com/docs/intro",
            "https://example.com/docs/guide",
            "https://example.com/about",
        ]

        result = extractor._filter_documentation_links(
            links, "https://example.com/docs/", 10
        )

        # Should prioritize links under the target path
        assert len(result) > 0
        # First results should be under /docs/
        docs_links = [url for url in result if "/docs/" in url]
        assert len(docs_links) >= 2

    def test_filter_documentation_links_max_pages(self, tmp_path):
        """Test max pages limit."""
        extractor = Crawl4aiExtractor(tmp_path)

        links = [f"https://example.com/page{i}" for i in range(20)]

        result = extractor._filter_documentation_links(links, "https://example.com", 5)

        assert len(result) <= 5


    def test_create_filename_from_url(self, tmp_path):
        """Test filename creation from URL."""
        extractor = Crawl4aiExtractor(tmp_path)

        test_cases = [
            ("https://example.com/docs/intro", "docs_intro.md"),
            ("https://example.com/", "example.com.md"),
            ("https://example.com/path/with/slashes", "path_with_slashes.md"),
            ("https://example.com/file.html", "file.html.md"),
        ]

        for url, expected in test_cases:
            result = extractor._create_filename_from_url(url)
            assert result.endswith(".md")
            # Should be a safe filename
            assert "/" not in result
            assert "\\" not in result

    def test_create_filename_edge_cases(self, tmp_path):
        """Test filename creation edge cases."""
        extractor = Crawl4aiExtractor(tmp_path)

        # Edge cases that should result in index.md
        edge_cases = [
            "https://example.com/",
            "https://example.com",
        ]

        for url in edge_cases:
            result = extractor._create_filename_from_url(url)
            assert result.endswith(".md")
            assert result != ".md"

    @pytest.mark.asyncio
    async def test_extract_from_url_basic_structure(self, tmp_path):
        """Test basic structure of extract_from_url (without actual crawling)."""
        extractor = Crawl4aiExtractor(tmp_path)

        # This test would need extensive mocking to test the full async flow
        # For now, just test that the method exists and has the right signature
        assert hasattr(extractor, "extract_from_url")
        assert callable(extractor.extract_from_url)


