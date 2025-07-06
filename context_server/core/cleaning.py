"""Markdown content cleaning utilities."""

import re


class MarkdownCleaner:
    """Clean and normalize markdown content."""

    def __init__(self):
        """Initialize the markdown cleaner."""
        pass

    def clean_content(self, content: str) -> str:
        """Clean markdown content by removing excessive whitespace and normalizing structure."""
        if not content or not content.strip():
            return ""

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

        # Remove excessive consecutive newlines (more than 2)
        result = re.sub(r"\n{4,}", "\n\n\n", result)

        # Clean up any remaining trailing whitespace at the end
        result = result.rstrip()

        return result
