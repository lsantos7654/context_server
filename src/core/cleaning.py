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
            "permalink_fragments_removed": 0,
            "code_line_numbers_removed": 0,
            "encoded_data_blobs_removed": 0,
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
        content = self.remove_permalink_fragments(content)
        content = self.remove_code_line_numbers(content)
        content = self.remove_encoded_data_blobs(content)
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
        in_navigation = False
        in_toc = False
        skip_until_content = True

        for line in lines:
            line_stripped = line.strip()

            # Skip initial navigation elements until we find content
            if skip_until_content:
                # Look for actual content markers (headings that aren't navigation)
                if line_stripped.startswith("# ") and not any(
                    nav in line_stripped.lower()
                    for nav in [
                        "skip to",
                        "search",
                        "cancel",
                        "clear",
                        "home",
                        "navigation",
                    ]
                ):
                    skip_until_content = False
                    cleaned_lines.append(line)
                continue

            # Skip common navigation/UI patterns
            skip_patterns = [
                "Skip to content",
                "Type to start searching",
                "Search ` `⌘``K`",
                "Cancel",
                "Clear",
                "Select theme DarkLightAuto",
                "Discourse",
                "On this page",
                "## On this page",
                "Table of contents",
                "Home",
                "Textual",
                "![logo]",
            ]

            if any(pattern in line_stripped for pattern in skip_patterns):
                continue

            # Skip navigation lists (detect by bullet point + link pattern)
            if line_stripped.startswith("* [") and "](" in line_stripped:
                # Check if this looks like navigation (common nav terms)
                nav_terms = [
                    "Home",
                    "Introduction",
                    "Tutorial",
                    "Guide",
                    "Widgets",
                    "Reference",
                    "API",
                    "How To",
                    "FAQ",
                    "Roadmap",
                    "Blog",
                    "Installation",
                    "Tutorials",
                    "Examples",
                    "Concepts",
                    "Recipes",
                    "Highlights",
                    "Showcase",
                    "Templates",
                    "Developer Guide",
                    "Getting started",
                    "Help",
                ]
                if any(term in line_stripped for term in nav_terms):
                    in_navigation = True
                    continue

            # Exit navigation when we hit content
            if in_navigation:
                if line_stripped.startswith("#") or (
                    line_stripped
                    and not line_stripped.startswith("*")
                    and not line_stripped.startswith(" ")
                    and len(line_stripped) > 30
                ):
                    in_navigation = False
                    cleaned_lines.append(line)
                else:
                    continue

            # Skip table of contents sections that are just navigation
            if (
                "table of contents" in line_stripped.lower()
                or line_stripped == "## On this page"
            ):
                in_toc = True
                continue

            if in_toc:
                if line_stripped.startswith("*") or line_stripped.startswith("-"):
                    continue
                elif line_stripped.startswith("#") or (
                    line_stripped and len(line_stripped) > 20
                ):
                    in_toc = False
                    cleaned_lines.append(line)
                else:
                    continue

            # Skip footer navigation
            if line_stripped.startswith("[ Previous ") or line_stripped.startswith(
                "[ Next "
            ):
                continue

            # Skip lines that are just site navigation repetition
            if line_stripped in ["Home", "Textual", "Guide", "Reference"]:
                continue

            # Skip empty lines at the start
            if not cleaned_lines and not line_stripped:
                continue

            cleaned_lines.append(line)

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

    def remove_permalink_fragments(self, content: str) -> str:
        """Remove permalink fragments like [¶](url "Permanent link")."""
        original_content = content

        # Remove permalink fragments (¶ symbol with links)
        content = re.sub(r'\[¶\]\([^)]+\s+"Permanent link"\)', "", content)

        if content != original_content:
            self.stats["permalink_fragments_removed"] += 1

        return content

    def remove_code_line_numbers(self, content: str) -> str:
        """Remove code line number links like [](__codelineno-X-Y)."""
        original_content = content

        # Remove code line number references
        content = re.sub(r"\[\]\([^)]*__codelineno-[^)]*\)", "", content)

        if content != original_content:
            self.stats["code_line_numbers_removed"] += 1

        return content

    def remove_encoded_data_blobs(self, content: str) -> str:
        """Remove large encoded data blobs (base64-like strings)."""
        original_content = content

        # Remove very long encoded strings (likely base64 or similar)
        # Look for lines with mostly base64 characters that are very long
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Check if line is very long and contains encoded data patterns
            # Pattern 1: Pure base64 (letters, numbers, +, /, =)
            # Pattern 2: Base64 with unicode escape sequences (like \u0001)
            is_encoded_blob = len(stripped) > 200 and (
                re.match(r"^[A-Za-z0-9+/=]+$", stripped)
                or re.match(r"^[A-Za-z0-9+/=\\xu0-9]+$", stripped)
                or ("eyJ" in stripped and len(stripped) > 500)  # JSON-like base64
            )

            if is_encoded_blob:
                # Skip this line as it's likely an encoded data blob
                continue
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines)
        if result != original_content:
            self.stats["encoded_data_blobs_removed"] += 1

        return result
