"""Text chunking utilities for processing documents using LangChain."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata and line tracking."""

    content: str
    tokens: int
    metadata: dict
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class TextChunker:
    """Chunks text into smaller pieces suitable for embedding using LangChain."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, chunk_type: str = "text"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_type = chunk_type
        
        # Create LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
        try:
            # Store original text and line mappings for position tracking
            self.original_text = text
            self.lines = text.splitlines()
            self.line_to_char = self._build_line_to_char_mapping()

            # Use LangChain's RecursiveCharacterTextSplitter
            chunk_strings = self.text_splitter.split_text(text)

            # Create TextChunk objects with line tracking
            text_chunks = []
            for i, chunk_content in enumerate(chunk_strings):
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
                        metadata={
                            "chunk_index": i, 
                        },
                        start_line=start_line,
                        end_line=end_line,
                        char_start=char_start,
                        char_end=char_end,
                    )
                    text_chunks.append(text_chunk)

            logger.debug(
                f"Split text into {len(text_chunks)} chunks using LangChain RecursiveCharacterTextSplitter"
            )
            return text_chunks

        except Exception as e:
            logger.error(f"Failed to chunk text with LangChain: {e}")
            # Fallback to simple splitting
            return self._simple_chunk_with_lines(text)


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
