"""
Unified markdown cleaning system.

Consolidates MarkdownCleaner from extract.py with improved structure
following CLAUDE.md principles.
"""

import re
import shutil
from pathlib import Path

# No typing imports needed for Python 3.12+


class MarkdownCleaner:
    """
    Comprehensive markdown cleanup processor for LLM-friendly content.

    Consolidates and improves upon the MarkdownCleaner from extract.py
    with better separation of concerns and cleaner interface.
    """

    def __init__(self, backup_suffix: str = ".backup"):
        self.backup_suffix = backup_suffix
        self.stats = {
            "files_processed": 0,
            "files_backed_up": 0,
            "html_entities_fixed": 0,
            "navigation_sections_removed": 0,
            "paragraph_symbols_fixed": 0,
            "whitespace_normalized": 0,
            "duplicate_headings_removed": 0,
            "empty_sections_removed": 0,
        }

        # HTML entity mappings
        self.html_entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&nbsp;": " ",
            "&#39;": "'",
            "&#8217;": "'",
            "&#8220;": '"',
            "&#8221;": '"',
            "&#8211;": "–",
            "&#8212;": "—",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "…",
        }

    def clean_content(self, content: str) -> str:
        """
        Clean markdown content using all available cleaning methods.

        Args:
            content: Raw markdown content to clean

        Returns:
            Cleaned markdown content
        """
        # Apply all cleaning methods in sequence
        content = self.fix_html_entities(content)
        content = self.remove_navigation_sections(content)
        content = self.fix_paragraph_symbols(content)
        content = self.normalize_whitespace(content)
        content = self.remove_duplicate_headings(content)
        content = self.remove_empty_sections(content)

        return content

    def clean_file(self, file_path: Path, create_backup: bool = True) -> dict:
        """
        Clean a single markdown file.

        Args:
            file_path: Path to markdown file
            create_backup: Whether to create backup before cleaning

        Returns:
            Cleaning statistics for this file
        """
        try:
            if create_backup:
                self.backup_file(file_path)

            # Read and clean content
            content = file_path.read_text(encoding="utf-8")
            cleaned_content = self.clean_content(content)

            # Write cleaned content back
            file_path.write_text(cleaned_content, encoding="utf-8")
            self.stats["files_processed"] += 1

            return {"success": True, "file": str(file_path)}

        except Exception as e:
            return {"success": False, "file": str(file_path), "error": str(e)}

    def clean_directory(self, directory: Path, create_backups: bool = True) -> dict:
        """
        Clean all markdown files in a directory.

        Args:
            directory: Directory containing markdown files
            create_backups: Whether to create backups

        Returns:
            Overall cleaning statistics
        """
        markdown_files = list(directory.glob("*.md"))

        for file_path in markdown_files:
            self.clean_file(file_path, create_backups)

        return self.stats

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the original file in backup subdirectory."""
        # Create backup directory if it doesn't exist
        backup_dir = file_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)

        # Create backup path in the backup directory
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        self.stats["files_backed_up"] += 1
        return backup_path

    def fix_html_entities(self, content: str) -> str:
        """Replace HTML entities with proper characters."""
        original_content = content
        for entity, replacement in self.html_entities.items():
            content = content.replace(entity, replacement)

        if content != original_content:
            self.stats["html_entities_fixed"] += 1

        return content

    def remove_navigation_sections(self, content: str) -> str:
        """Remove redundant navigation sections."""
        original_content = content
        lines = content.split("\n")
        cleaned_lines = []
        skip_mode = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if we're starting a navigation section
            if line.strip() == "<!-- image -->" or (
                line.strip() == "- Home" and i < 200
            ):
                skip_mode = True

            # Look for the end of navigation section
            if skip_mode:
                # End navigation when we hit a proper heading or significant content
                if line.startswith("#") or (
                    line.strip()
                    and not line.startswith("-")
                    and not line.startswith(" ")
                    and len(line.strip()) > 50
                    and "Table of contents" not in line
                ):
                    skip_mode = False
                    cleaned_lines.append(line)
                # Skip this line if still in navigation
            else:
                cleaned_lines.append(line)

            i += 1

        result = "\n".join(cleaned_lines)
        if result != original_content:
            self.stats["navigation_sections_removed"] += 1

        return result

    def fix_paragraph_symbols(self, content: str) -> str:
        """Fix paragraph symbols and formatting."""
        original_content = content

        # Remove ¶ symbols and similar artifacts
        content = re.sub(r"¶\s*", "", content)

        # Fix common paragraph markers
        content = re.sub(r"§\s*", "## ", content)

        # Remove other common artifacts
        content = re.sub(r"\[edit\]", "", content)
        content = re.sub(r"\[source\]", "", content)

        if content != original_content:
            self.stats["paragraph_symbols_fixed"] += 1

        return content

    def normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace and line breaks."""
        original_content = content

        # Remove trailing whitespace
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Normalize multiple blank lines to maximum of 2
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Fix spacing around headings
        content = re.sub(r"\n(#{1,6}[^\n]*)\n{1}([^\n#])", r"\n\1\n\n\2", content)

        if content != original_content:
            self.stats["whitespace_normalized"] += 1

        return content

    def remove_duplicate_headings(self, content: str) -> str:
        """Remove duplicate consecutive headings."""
        original_content = content
        lines = content.split("\n")
        cleaned_lines = []
        last_heading = None

        for line in lines:
            if line.startswith("#"):
                # Extract heading text (without # symbols)
                heading_text = re.sub(r"^#+\s*", "", line).strip()
                if heading_text != last_heading:
                    cleaned_lines.append(line)
                    last_heading = heading_text
                # Skip duplicate heading
            else:
                cleaned_lines.append(line)
                last_heading = None  # Reset when we encounter non-heading

        result = "\n".join(cleaned_lines)
        if result != original_content:
            self.stats["duplicate_headings_removed"] += 1

        return result

    def remove_empty_sections(self, content: str) -> str:
        """Remove sections that are empty or contain only whitespace."""
        original_content = content

        # Remove empty sections (heading followed immediately by another heading or end)
        content = re.sub(r"(#{1,6}[^\n]*)\n+(?=#{1,6})", r"\1\n\n", content)

        # Remove headings at the very end with no content
        content = re.sub(r"\n(#{1,6}[^\n]*)\s*$", "", content)

        if content != original_content:
            self.stats["empty_sections_removed"] += 1

        return content
