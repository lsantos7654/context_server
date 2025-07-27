"""Markdown content cleaning utilities."""

import re


class MarkdownCleaner:
    """Clean and normalize markdown content."""

    def __init__(self):
        """Initialize the markdown cleaner with noise removal patterns."""
        # Patterns for removing common documentation noise
        self.navigation_patterns = [
            r'(?i)(?:^|\n)(?:home|contents?|index|navigation|nav|menu)\s*[:\-]?\s*$',
            r'(?i)(?:^|\n)(?:previous|next|back|forward)\s*[:\|\-]?\s*.*$',
            r'(?i)(?:^|\n)breadcrumb[s]?\s*[:\-]?.*$',
            r'(?i)(?:^|\n)(?:table of contents?|toc)\s*[:\-]?.*$',
            r'(?i)(?:^|\n)on this page\s*[:\-]?.*$',
        ]
        
        # Patterns for removing repetitive or low-value content
        self.noise_patterns = [
            r'(?i)(?:^|\n)edit on github\s*$',
            r'(?i)(?:^|\n)improve this page\s*$',
            r'(?i)(?:^|\n)last updated[:\-]?\s*.*$',
            r'(?i)(?:^|\n)(?:copyright|©)\s*\d{4}.*$',
            r'(?i)(?:^|\n)print this page\s*$',
            r'(?i)(?:^|\n)share this.*$',
            r'(?i)(?:^|\n)feedback\s*[:\-]?.*$',
            r'(?i)(?:^|\n)see also\s*[:\-]?\s*$',  # Empty see also sections
            r'(?i)\n?copy\s*$',  # Copy button text
        ]
        
        # Patterns for cleaning up markdown artifacts
        self.markdown_artifact_patterns = [
            r'\[!\[.*?\]\(.*?\)\]\(.*?\)',  # Complex nested link/image patterns
            r'\[\s*\]',  # Empty links
            r'!\[\s*\]',  # Empty images
            r'\n\s*\|\s*\n',  # Empty table rows
            r'\n\s*\-{3,}\s*\n',  # Horizontal rules that are alone
        ]

    def clean_content(self, content: str) -> str:
        """Clean markdown content by removing noise, excessive whitespace and normalizing structure."""
        if not content or not content.strip():
            return ""

        # First pass: Remove noise patterns before line-by-line processing
        content = self._remove_noise_patterns(content)

        # Split into lines for processing
        lines = content.splitlines()
        cleaned_lines = []
        consecutive_empty = 0
        in_code_block = False

        for line in lines:
            # Track code blocks to preserve their formatting
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                cleaned_lines.append(line.rstrip())
                consecutive_empty = 0
                continue

            if in_code_block:
                # Preserve code block content exactly
                cleaned_lines.append(line.rstrip())
                consecutive_empty = 0
            else:
                # Clean regular content
                stripped_line = line.rstrip()

                # Skip lines that are likely navigation or noise
                if self._is_noise_line(stripped_line):
                    continue

                if not stripped_line:
                    # Empty line
                    consecutive_empty += 1
                    # Allow max 2 consecutive empty lines
                    if consecutive_empty <= 2:
                        cleaned_lines.append("")
                else:
                    # Non-empty line
                    consecutive_empty = 0
                    cleaned_lines.append(stripped_line)

        # Join lines and clean up any remaining issues
        result = "\n".join(cleaned_lines)

        # Final cleanup: Remove markdown artifacts and excessive whitespace
        result = self._final_cleanup(result)

        return result

    def _remove_noise_patterns(self, content: str) -> str:
        """Remove common noise patterns from documentation content."""
        # Remove navigation patterns
        for pattern in self.navigation_patterns:
            content = re.sub(pattern, '', content)
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            content = re.sub(pattern, '', content)
            
        # Remove markdown artifacts
        for pattern in self.markdown_artifact_patterns:
            content = re.sub(pattern, '', content)
            
        return content

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is likely to be navigation or other noise."""
        if not line.strip():
            return False
            
        line_lower = line.lower().strip()
        
        # Short lines that are likely navigation
        if len(line_lower) < 3:
            return True
            
        # Common navigation indicators
        noise_indicators = [
            'skip to', 'jump to', 'go to', 'back to top', 'scroll to',
            'toggle', 'expand', 'collapse', 'show', 'hide',
            '»', '›', '‹', '«',  # Navigation arrows
            'permalink', 'anchor',
        ]
        
        for indicator in noise_indicators:
            if indicator in line_lower:
                return True
                
        # Lines that are just symbols or very short
        if re.match(r'^[\s\-_=\|•·]+$', line):
            return True
            
        return False

    def _final_cleanup(self, content: str) -> str:
        """Final cleanup of the content."""
        # Remove excessive consecutive newlines (more than 2)
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        
        # Remove standalone bullet points or list markers
        content = re.sub(r'^\s*[•·\-\*]\s*$', '', content, flags=re.MULTILINE)
        
        # Remove lines with just numbers (likely page numbers or section numbers alone)
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        
        # Clean up remaining empty lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Clean up any remaining trailing whitespace at the end
        content = content.rstrip()

        return content
