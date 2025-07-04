"""Text chunking utilities for processing documents."""

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    tokens: int
    metadata: dict


class TextChunker:
    """Chunks text into smaller pieces suitable for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks with overlaps."""
        try:
            # Clean the text first
            text = self._clean_text(text)

            # Split by paragraphs first, then by sentences if needed
            chunks = self._split_by_paragraphs(text)

            # If chunks are still too large, split further
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    final_chunks.append(chunk)
                else:
                    # Split large chunks by sentences
                    sub_chunks = self._split_by_sentences(chunk)
                    final_chunks.extend(sub_chunks)

            # Create TextChunk objects
            text_chunks = []
            for i, chunk_content in enumerate(final_chunks):
                if chunk_content.strip():  # Skip empty chunks
                    text_chunk = TextChunk(
                        content=chunk_content.strip(),
                        tokens=self._estimate_tokens(chunk_content),
                        metadata={"chunk_index": i, "method": "paragraph_sentence"},
                    )
                    text_chunks.append(text_chunk)

            logger.debug(f"Split text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            # Fallback to simple splitting
            return self._simple_chunk(text)

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

    def _split_by_paragraphs(self, text: str) -> List[str]:
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

    def _split_by_sentences(self, text: str) -> List[str]:
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

    def _simple_chunk(self, text: str) -> List[TextChunk]:
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
