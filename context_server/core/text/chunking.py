"""Text chunking utilities for processing documents using LangChain."""

import logging
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter

from context_server.models.domain.chunks import TextChunk

logger = logging.getLogger(__name__)


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
            text_chunks = self.text_splitter.split_text(text)
            
            result = []
            for i, chunk_text in enumerate(text_chunks):
                # Calculate line positions for this chunk
                start_line, end_line = self._find_line_range(chunk_text)
                char_start, char_end = self._find_char_range(chunk_text)
                
                # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                tokens = len(chunk_text) // 4
                
                # Create TextChunk with position tracking
                chunk = TextChunk(
                    content=chunk_text,
                    tokens=tokens,
                    metadata={
                        "chunk_index": i,
                        "chunk_type": self.chunk_type,
                        "chunk_size": len(chunk_text),
                        "chunk_overlap": self.chunk_overlap if i > 0 else 0,
                    },
                    start_line=start_line,
                    end_line=end_line,
                    char_start=char_start,
                    char_end=char_end,
                )
                result.append(chunk)
            
            logger.debug(f"Created {len(result)} chunks from {len(text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            # Return single chunk as fallback
            return [TextChunk(
                content=text,
                tokens=len(text) // 4,
                metadata={"chunk_index": 0, "chunk_type": self.chunk_type, "error": str(e)},
            )]

    def _build_line_to_char_mapping(self) -> list[int]:
        """Build mapping from line numbers to character positions."""
        char_positions = [0]  # Line 1 starts at character 0
        current_pos = 0
        
        for line in self.lines:
            current_pos += len(line) + 1  # +1 for newline character
            char_positions.append(current_pos)
            
        return char_positions

    def _find_line_range(self, chunk_text: str) -> tuple[int | None, int | None]:
        """Find the line range (1-based) for a chunk of text."""
        try:
            # Find the chunk text in the original text
            start_pos = self.original_text.find(chunk_text)
            if start_pos == -1:
                logger.debug("Could not find chunk in original text for line mapping")
                return None, None
                
            end_pos = start_pos + len(chunk_text)
            
            # Find which lines these positions correspond to
            start_line = None
            end_line = None
            
            for line_num, char_pos in enumerate(self.line_to_char):
                if start_line is None and char_pos <= start_pos < self.line_to_char[line_num + 1]:
                    start_line = line_num + 1  # Convert to 1-based
                    
                if end_pos <= self.line_to_char[line_num + 1]:
                    end_line = line_num + 1  # Convert to 1-based
                    break
                    
            return start_line, end_line
            
        except Exception as e:
            logger.debug(f"Could not determine line range: {e}")
            return None, None

    def _find_char_range(self, chunk_text: str) -> tuple[int | None, int | None]:
        """Find the character range for a chunk of text."""
        try:
            start_pos = self.original_text.find(chunk_text)
            if start_pos == -1:
                return None, None
                
            end_pos = start_pos + len(chunk_text)
            return start_pos, end_pos
            
        except Exception as e:
            logger.debug(f"Could not determine character range: {e}")
            return None, None

    def get_optimal_chunk_size(self, text: str, target_chunks: int = None) -> int:
        """Calculate optimal chunk size based on text length and target number of chunks."""
        text_length = len(text)
        
        if target_chunks:
            # Calculate chunk size to get approximately target_chunks
            optimal_size = text_length // target_chunks
            # Round to reasonable boundaries
            if optimal_size < 500:
                return 500
            elif optimal_size > 2000:
                return 2000
            else:
                return (optimal_size // 100) * 100  # Round to nearest 100
        
        # Default logic based on text length
        if text_length < 2000:
            return 500
        elif text_length < 10000:
            return 1000
        elif text_length < 50000:
            return 1500
        else:
            return 2000


__all__ = ["TextChunker", "TextChunk"]