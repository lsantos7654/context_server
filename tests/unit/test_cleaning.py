"""
Unit tests for cleaning module.
"""

from src.core.cleaning import MarkdownCleaner


class TestMarkdownCleaner:
    """Test markdown cleaning functionality."""

    def test_cleaner_initialization(self):
        """Test cleaner initialization."""
        cleaner = MarkdownCleaner()
        assert cleaner is not None

    def test_clean_content_basic(self):
        """Test basic content cleaning."""
        cleaner = MarkdownCleaner()
        content = "# Title\n\nSome content here.\n\n\n\nMore content."
        result = cleaner.clean_content(content)

        assert "# Title" in result
        assert "Some content here." in result
        assert "More content." in result
        # Should reduce excessive blank lines
        assert "\n\n\n\n" not in result

    def test_clean_content_empty(self):
        """Test cleaning empty content."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean_content("")
        assert result == ""

    def test_clean_content_whitespace_only(self):
        """Test cleaning whitespace-only content."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean_content("   \n  \n   ")
        assert result.strip() == ""

    def test_clean_content_removes_trailing_whitespace(self):
        """Test that trailing whitespace is removed."""
        cleaner = MarkdownCleaner()
        content = "Line with trailing spaces   \nAnother line  \n"
        result = cleaner.clean_content(content)
        lines = result.split("\n")

        for line in lines:
            if line:  # Skip empty lines
                assert not line.endswith(" ")

    def test_clean_content_preserves_structure(self):
        """Test that basic markdown structure is preserved."""
        cleaner = MarkdownCleaner()
        content = """# Main Title

## Subtitle

Some paragraph text.

* List item 1
* List item 2

```code
Some code block
```

More text."""

        result = cleaner.clean_content(content)

        assert "# Main Title" in result
        assert "## Subtitle" in result
        assert "* List item 1" in result
        assert "```code" in result

    def test_clean_content_reduces_excessive_blank_lines(self):
        """Test that excessive blank lines are reduced."""
        cleaner = MarkdownCleaner()
        content = "Line 1\n\n\n\n\n\nLine 2"
        result = cleaner.clean_content(content)

        # Should not have more than 2 consecutive blank lines
        assert "\n\n\n\n" not in result

    def test_clean_content_handles_mixed_whitespace(self):
        """Test handling of mixed whitespace."""
        cleaner = MarkdownCleaner()
        content = "Line 1\t  \n\n  \t\nLine 2"
        result = cleaner.clean_content(content)

        lines = result.split("\n")
        # Should remove trailing whitespace including tabs
        for line in lines:
            if line:
                assert not line.endswith(" ") and not line.endswith("\t")

    def test_clean_content_preserves_code_blocks(self):
        """Test that code blocks are preserved."""
        cleaner = MarkdownCleaner()
        content = """Text before

```python
def hello():
    print("Hello")
```

Text after"""

        result = cleaner.clean_content(content)
        assert "```python" in result
        assert "def hello():" in result
        assert 'print("Hello")' in result

    def test_clean_content_handles_unicode(self):
        """Test handling of unicode characters."""
        cleaner = MarkdownCleaner()
        content = "# Title with émojis 🚀\n\nContent with ñiño and café."
        result = cleaner.clean_content(content)

        assert "émojis 🚀" in result
        assert "ñiño" in result
        assert "café" in result

    def test_clean_content_large_input(self):
        """Test cleaning large content."""
        cleaner = MarkdownCleaner()
        # Create large content
        content = "# Title\n\n" + "Content line.\n" * 1000
        result = cleaner.clean_content(content)

        assert "# Title" in result
        assert "Content line." in result
        assert len(result) > 0

    def test_clean_content_with_links(self):
        """Test cleaning content with markdown links."""
        cleaner = MarkdownCleaner()
        content = (
            "Check out [this link](https://example.com) and [another](./local.md)."
        )
        result = cleaner.clean_content(content)

        assert "[this link](https://example.com)" in result
        assert "[another](./local.md)" in result

    def test_clean_content_with_images(self):
        """Test cleaning content with markdown images."""
        cleaner = MarkdownCleaner()
        content = "Here's an image: ![alt text](image.png)"
        result = cleaner.clean_content(content)

        assert "![alt text](image.png)" in result
