"""Markdown content cleaning utilities."""

import re


class MarkdownCleaner:
    """Clean and normalize markdown content."""

    def __init__(self):
        """Initialize the markdown cleaner with noise removal patterns."""
        # Patterns for removing common documentation noise
        self.navigation_patterns = [
            r"(?i)(?:^|\n)(?:home|contents?|index|navigation|nav|menu)\s*[:\-]?\s*$",
            r"(?i)(?:^|\n)(?:previous|next|back|forward)\s*[:\|\-]?\s*.*$",
            r"(?i)(?:^|\n)breadcrumb[s]?\s*[:\-]?.*$",
            r"(?i)(?:^|\n)(?:table of contents?|toc)\s*[:\-]?.*$",
            r"(?i)(?:^|\n)on this page\s*[:\-]?.*$",
        ]

        # Patterns for removing repetitive or low-value content
        self.noise_patterns = [
            r"(?i)(?:^|\n)edit on github\s*$",
            r"(?i)(?:^|\n)improve this page\s*$",
            r"(?i)(?:^|\n)last updated[:\-]?\s*.*$",
            r"(?i)(?:^|\n)(?:copyright|©)\s*\d{4}.*$",
            r"(?i)(?:^|\n)print this page\s*$",
            r"(?i)(?:^|\n)share this.*$",
            r"(?i)(?:^|\n)feedback\s*[:\-]?.*$",
            r"(?i)(?:^|\n)see also\s*[:\-]?\s*$",  # Empty see also sections
            r"(?i)\n?copy\s*$",  # Copy button text
        ]

        # Patterns for cleaning up markdown artifacts
        self.markdown_artifact_patterns = [
            r"\[!\[.*?\]\(.*?\)\]\(.*?\)",  # Complex nested link/image patterns
            r"\[\s*\]",  # Empty links
            r"!\[\s*\]",  # Empty images
            r"\n\s*\|\s*\n",  # Empty table rows
            r"\n\s*\-{3,}\s*\n",  # Horizontal rules that are alone
        ]

    def clean_content(self, content: str) -> str:
        """Clean markdown content by removing noise, excessive whitespace and normalizing structure."""
        if not content or not content.strip():
            return ""

        # First pass: Remove noise patterns before line-by-line processing
        content = self._remove_noise_patterns(content)

        # Split into lines for processing
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip empty lines that are just whitespace
            if not line.strip():
                # Keep some empty lines for structure, but not too many consecutive ones
                if cleaned_lines and cleaned_lines[-1].strip():
                    cleaned_lines.append("")
                continue

            # Remove lines that are just navigation or noise
            if self._is_noise_line(line):
                continue

            # Clean individual line
            cleaned_line = self._clean_line(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        # Join lines and do final cleanup
        result = "\n".join(cleaned_lines)

        # Remove markdown artifacts
        for pattern in self.markdown_artifact_patterns:
            result = re.sub(pattern, "", result)

        # Normalize whitespace and line breaks
        result = self._normalize_whitespace(result)

        return result.strip()

    def _remove_noise_patterns(self, content: str) -> str:
        """Remove navigation and noise patterns from content."""
        # Apply all noise removal patterns
        for pattern in self.navigation_patterns + self.noise_patterns:
            content = re.sub(pattern, "", content, flags=re.MULTILINE)

        return content

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is likely noise/navigation."""
        line_lower = line.lower().strip()

        # Common noise indicators
        noise_indicators = [
            "edit this page",
            "improve this page",
            "last modified",
            "last updated",
            "print page",
            "share page",
            "feedback",
            "report issue",
            "edit on github",
            "view source",
            "page source",
            "contribute",
        ]

        # Check if line contains noise indicators
        for indicator in noise_indicators:
            if indicator in line_lower:
                return True

        # Check if line is just a navigation element
        if re.match(r"^[\s\|\-\>\<\[\]]+$", line):
            return True

        # Check if line is very short and likely navigation
        if len(line.strip()) < 3 and re.match(r"^[\w\s\-]+$", line.strip()):
            return True

        return False

    def _clean_line(self, line: str) -> str:
        """Clean an individual line of markdown."""
        # Remove excessive inline formatting
        # Fix multiple consecutive formatting markers
        line = re.sub(r"\*{3,}", "**", line)  # Reduce excessive bold
        line = re.sub(r"_{3,}", "__", line)  # Reduce excessive italics

        # Clean up link formatting
        # Remove empty links
        line = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", line)

        # Clean up excessive punctuation
        line = re.sub(r"[\.]{3,}", "...", line)  # Normalize ellipsis
        line = re.sub(r"[!]{2,}", "!", line)  # Reduce excessive exclamation
        line = re.sub(r"[\?]{2,}", "?", line)  # Reduce excessive questions

        # Remove trailing whitespace
        line = line.rstrip()

        return line

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace and line breaks."""
        # Remove excessive blank lines (more than 2 consecutive)
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Remove trailing spaces from lines
        lines = content.split("\n")
        lines = [line.rstrip() for line in lines]
        content = "\n".join(lines)

        # Ensure proper spacing around headers
        content = re.sub(r"\n(#+\s+.*?)\n", r"\n\n\1\n\n", content)

        # Clean up the result
        content = re.sub(r"\n{3,}", "\n\n", content)  # Max 2 consecutive newlines

        return content


__all__ = ["MarkdownCleaner"]
