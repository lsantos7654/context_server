"""Text chunking utilities for processing documents."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata and line tracking."""

    content: str
    tokens: int
    metadata: dict
    start_line: int = None
    end_line: int = None
    char_start: int = None
    char_end: int = None


class TextChunker:
    """Chunks text into smaller pieces suitable for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks with overlaps and line tracking."""
        try:
            # Store original text and line mappings
            self.original_text = text
            self.lines = text.splitlines()
            self.line_to_char = self._build_line_to_char_mapping()

            # Clean the text first
            cleaned_text = self._clean_text(text)

            # Split by paragraphs first, then by sentences if needed
            chunk_strings = self._split_by_paragraphs(cleaned_text)

            # If chunks are still too large, split further
            final_chunk_strings = []
            for chunk in chunk_strings:
                if len(chunk) <= self.chunk_size:
                    final_chunk_strings.append(chunk)
                else:
                    # Split large chunks by sentences
                    sub_chunks = self._split_by_sentences(chunk)
                    final_chunk_strings.extend(sub_chunks)

            # Create TextChunk objects with line tracking
            text_chunks = []
            for i, chunk_content in enumerate(final_chunk_strings):
                chunk_content = chunk_content.strip()
                if chunk_content:  # Skip empty chunks
                    # Find line positions for this chunk
                    (
                        start_line,
                        end_line,
                        char_start,
                        char_end,
                    ) = self._find_chunk_position(chunk_content, i)

                    text_chunk = TextChunk(
                        content=chunk_content,
                        tokens=self._estimate_tokens(chunk_content),
                        metadata={"chunk_index": i, "method": "paragraph_sentence"},
                        start_line=start_line,
                        end_line=end_line,
                        char_start=char_start,
                        char_end=char_end,
                    )
                    text_chunks.append(text_chunk)

            logger.debug(
                f"Split text into {len(text_chunks)} chunks with line tracking"
            )
            return text_chunks

        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            # Fallback to simple splitting
            return self._simple_chunk_with_lines(text)

    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove very short lines that are likely artifacts
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if (
                len(line.strip()) > 10 or not line.strip()
            ):  # Keep longer lines or empty lines
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraphs, respecting chunk size."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Get last few sentences for overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentences when paragraphs are too large."""
        # Simple sentence splitting (could be improved with spacy/nltk)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text

        # Try to find sentence boundaries for natural overlap
        sentences = re.split(r"(?<=[.!?])\s+", text)

        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                if overlap_text:
                    overlap_text = sentence + " " + overlap_text
                else:
                    overlap_text = sentence
            else:
                break

        # If no sentence-based overlap, just take the last N characters
        if not overlap_text:
            overlap_text = text[-self.chunk_overlap :]

        return overlap_text

    def _simple_chunk(self, text: str) -> list[TextChunk]:
        """Simple chunking fallback method."""
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_content = text[i : i + self.chunk_size]
            if chunk_content.strip():
                chunk = TextChunk(
                    content=chunk_content.strip(),
                    tokens=self._estimate_tokens(chunk_content),
                    metadata={"chunk_index": len(chunks), "method": "simple"},
                )
                chunks.append(chunk)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def _build_line_to_char_mapping(self) -> list[tuple]:
        """Build mapping from line numbers to character positions."""
        line_to_char = []
        char_pos = 0

        for line_num, line in enumerate(self.lines):
            start_char = char_pos
            end_char = char_pos + len(line)
            line_to_char.append((start_char, end_char))
            # Add 1 for newline character (except last line)
            char_pos = end_char + (1 if line_num < len(self.lines) - 1 else 0)

        return line_to_char

    def _find_chunk_position(self, chunk_content: str, chunk_index: int) -> tuple:
        """
        Find the line and character positions of a chunk in the original text.

        Returns:
            tuple: (start_line, end_line, char_start, char_end)
        """
        try:
            # Find the chunk content in the original text
            # This is approximate since the content might have been cleaned
            chunk_start = self.original_text.find(
                chunk_content[:100]
            )  # Use first 100 chars for search

            if chunk_start == -1:
                # Fallback: estimate based on chunk index
                chars_per_chunk = len(self.original_text) // max(1, chunk_index + 1)
                chunk_start = chunk_index * chars_per_chunk
                chunk_end = min(
                    chunk_start + len(chunk_content), len(self.original_text)
                )
            else:
                chunk_end = chunk_start + len(chunk_content)

            # Find which lines these character positions correspond to
            start_line = self._char_to_line(chunk_start)
            end_line = self._char_to_line(chunk_end)

            return start_line, end_line, chunk_start, chunk_end

        except Exception as e:
            logger.warning(f"Could not determine chunk position: {e}")
            # Return estimated positions
            lines_per_chunk = max(1, len(self.lines) // max(1, chunk_index + 1))
            start_line = chunk_index * lines_per_chunk
            end_line = min(start_line + lines_per_chunk, len(self.lines) - 1)
            return start_line, end_line, 0, len(chunk_content)

    def _char_to_line(self, char_pos: int) -> int:
        """Convert character position to line number."""
        for line_num, (start_char, end_char) in enumerate(self.line_to_char):
            if start_char <= char_pos <= end_char:
                return line_num
        # If not found, return closest line
        return min(
            len(self.lines) - 1, max(0, char_pos // 80)
        )  # Assume ~80 chars per line

    def _simple_chunk_with_lines(self, text: str) -> list[TextChunk]:
        """Simple chunking fallback method with line tracking."""
        self.original_text = text
        self.lines = text.splitlines()
        self.line_to_char = self._build_line_to_char_mapping()

        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_content = text[i : i + self.chunk_size].strip()
            if chunk_content:
                start_line = self._char_to_line(i)
                end_line = self._char_to_line(min(i + self.chunk_size, len(text) - 1))

                chunk = TextChunk(
                    content=chunk_content,
                    tokens=self._estimate_tokens(chunk_content),
                    metadata={"chunk_index": len(chunks), "method": "simple"},
                    start_line=start_line,
                    end_line=end_line,
                    char_start=i,
                    char_end=min(i + self.chunk_size, len(text)),
                )
                chunks.append(chunk)

        return chunks
