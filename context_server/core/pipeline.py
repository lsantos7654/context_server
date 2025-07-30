"""Document processing pipeline - the main coordinator of the Context Server system.

This module orchestrates the three-document processing pipeline:
1. Original: Raw parsed markdown content  
2. Code Snippets: Extracted code blocks with voyage-code-3 embeddings
3. Cleaned Markdown: Text with code snippet placeholders for improved search
"""

import asyncio
import logging
import re

from pathlib import Path

from context_server.core.services.embeddings import EmbeddingService, VoyageEmbeddingService
from context_server.core.services.extraction import Crawl4aiExtractor
from context_server.core.services.llm import SummarizationService 
from context_server.core.text import TextChunker
from context_server.models.domain.documents import ProcessedDocument, ProcessedChunk, CodeSnippet, ProcessingResult

logger = logging.getLogger(__name__)


class CodeSnippetExtractor:
    """Extracts code snippets from text content."""

    def __init__(self):
        # Enhanced regex patterns for different code block formats
        self.markdown_code_pattern = re.compile(
            r"```(\w*)\n(.*?)\n```", re.DOTALL | re.MULTILINE
        )
        self.inline_code_pattern = re.compile(r"`([^`\n]+)`")

        # Additional patterns for various code block formats
        self.html_code_pattern = re.compile(
            r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", re.DOTALL | re.IGNORECASE
        )
        self.indented_code_pattern = re.compile(
            r"(?:^|\n)((?:    .*(?:\n|$))+)", re.MULTILINE
        )

    def extract_code_snippets(self, text: str) -> tuple[list[dict], str]:
        """Extract code snippets and return cleaned text with placeholders.
        
        Returns:
            tuple: (code_snippets, cleaned_text_with_placeholders)
        """
        code_snippets = []
        processed_text = text
        
        # Extract markdown code blocks
        for match in self.markdown_code_pattern.finditer(text):
            language = match.group(1).strip() or "text"
            code_content = match.group(2).strip()
            
            if self._is_valid_code_snippet(code_content):
                snippet_id = f"snippet_{len(code_snippets)}"
                
                snippet_info = {
                    "id": snippet_id,
                    "language": language,
                    "content": code_content,
                    "type": "code_block",
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "char_count": len(code_content),
                    "line_count": len(code_content.split('\n')),
                }
                
                code_snippets.append(snippet_info)
                
                # Create placeholder with metadata
                placeholder = self._create_code_placeholder(snippet_info)
                processed_text = processed_text.replace(match.group(0), placeholder, 1)
        
        # Extract inline code (only if significant)
        for match in self.inline_code_pattern.finditer(processed_text):
            code_content = match.group(1).strip()
            
            # Only extract inline code that looks substantial
            if len(code_content) > 20 and self._is_valid_code_snippet(code_content):
                snippet_id = f"snippet_{len(code_snippets)}"
                
                snippet_info = {
                    "id": snippet_id,
                    "language": "text",
                    "content": code_content,
                    "type": "inline_code",
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "char_count": len(code_content),
                    "line_count": 1,
                }
                
                code_snippets.append(snippet_info)
                
                # Create placeholder
                placeholder = self._create_code_placeholder(snippet_info)
                processed_text = processed_text.replace(match.group(0), placeholder, 1)
        
        logger.debug(f"Extracted {len(code_snippets)} code snippets")
        return code_snippets, processed_text

    def _is_valid_code_snippet(self, content: str) -> bool:
        """Check if content looks like a valid code snippet."""
        if not content or len(content.strip()) < 10:
            return False
            
        # Skip if it's just plain text
        code_indicators = [
            '{', '}', '(', ')', '[', ']', ';', '=', '->',
            'def ', 'function', 'class ', 'import ', 'from ',
            'const ', 'let ', 'var ', 'if ', 'for ', 'while ',
            'return ', 'print(', 'console.log', 'System.out'
        ]
        
        return any(indicator in content for indicator in code_indicators)

    def _create_code_placeholder(self, snippet_info: dict) -> str:
        """Create a structured placeholder for a code snippet."""
        # Generate a brief summary of the code content
        content = snippet_info["content"]
        summary = self._generate_code_summary(content)
        
        placeholder = (
            f"[CODE_SNIPPET: "
            f"language={snippet_info['language']}, "
            f"size={snippet_info['char_count']}_chars, "
            f"summary=\"{summary}\", "
            f"snippet_id={snippet_info['id']}]"
        )
        
        return placeholder

    def _generate_code_summary(self, code: str) -> str:
        """Generate a brief summary of code content."""
        lines = code.split('\n')
        first_meaningful_line = next(
            (line.strip() for line in lines if line.strip() and not line.strip().startswith('//')), 
            ""
        )
        
        if 'def ' in first_meaningful_line:
            return "Function definition"
        elif 'class ' in first_meaningful_line:
            return "Class definition"
        elif any(keyword in first_meaningful_line for keyword in ['import', 'from ', 'require']):
            return "Import statements"
        elif any(keyword in first_meaningful_line for keyword in ['const ', 'let ', 'var ']):
            return "Variable declaration"
        elif '{' in code and '}' in code:
            return "Code block with logic"
        else:
            return "Code snippet"


class DocumentProcessor:
    """Main document processor that orchestrates the three-document pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        code_embedding_service: VoyageEmbeddingService | None = None,
        summarization_service: SummarizationService | None = None,
    ):
        """Initialize the document processor with services."""
        self.embedding_service = embedding_service or EmbeddingService()
        self.code_embedding_service = code_embedding_service or VoyageEmbeddingService()
        self.summarization_service = summarization_service or SummarizationService()
        
        self.extractor = Crawl4aiExtractor()
        self.text_chunker = TextChunker()
        self.code_extractor = CodeSnippetExtractor()

    async def process_url(self, url: str, options: dict | None = None, job_id: str | None = None, db=None) -> ProcessingResult:
        """Process a URL through the complete pipeline."""
        try:
            logger.info(f"Starting URL processing: {url}")
            
            # Extract max_pages from options if provided, otherwise use default
            max_pages = options.get("max_pages", 50) if options else 50
            
            # Report progress if job tracking is enabled
            if job_id and db:
                await db.update_job_progress(job_id, 0.1, metadata={"phase": "extracting", "url": url})
            
            # Step 1: Extract content from URL
            extraction_result = await self.extractor.extract_url(url, max_pages)
            
            if not extraction_result.success:
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"Extraction failed: {extraction_result.error}"
                )
            
            # Step 2: Process the extracted content
            processed_doc = await self._process_content(
                url=url,
                title=extraction_result.metadata.get("title", "Untitled"),
                content=extraction_result.content,
                metadata=extraction_result.metadata
            )
            
            logger.info(f"Successfully processed URL: {url}")
            return ProcessingResult(
                documents=[processed_doc],
                success=True
            )
            
        except Exception as e:
            error_msg = f"Failed to process URL {url}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                documents=[],
                success=False,
                error=error_msg
            )

    async def process_file(self, file_path: str | Path, options: dict | None = None, job_id: str | None = None, db=None) -> ProcessingResult:
        """Process a local file through the complete pipeline."""
        try:
            logger.info(f"Starting file processing: {file_path}")
            
            # Report progress if job tracking is enabled
            if job_id and db:
                await db.update_job_progress(job_id, 0.1, metadata={"phase": "reading_file", "file_path": str(file_path)})
            
            # Step 1: Extract content from file
            extraction_result = await self.extractor.extract_file(file_path)
            
            if not extraction_result.success:
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"File extraction failed: {extraction_result.error}"
                )
            
            # Step 2: Process the extracted content
            processed_doc = await self._process_content(
                url=f"file://{file_path}",
                title=extraction_result.metadata.get("title", "Untitled"),
                content=extraction_result.content,
                metadata=extraction_result.metadata
            )
            
            logger.info(f"Successfully processed file: {file_path}")
            return ProcessingResult(
                documents=[processed_doc],
                success=True
            )
            
        except Exception as e:
            error_msg = f"Failed to process file {file_path}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                documents=[],
                success=False,
                error=error_msg
            )

    async def _process_content(
        self, url: str, title: str, content: str, metadata: dict
    ) -> ProcessedDocument:
        """Process content through the three-document pipeline."""
        
        # Step 1: Extract code snippets and create cleaned markdown
        code_snippets_info, cleaned_content = self.code_extractor.extract_code_snippets(content)
        
        # Step 2: Generate embeddings for code snippets
        code_snippets = []
        if code_snippets_info:
            logger.info(f"Processing {len(code_snippets_info)} code snippets")
            
            code_contents = [snippet["content"] for snippet in code_snippets_info]
            
            try:
                code_embeddings = await self.code_embedding_service.embed_batch(code_contents)
                
                for snippet_info, embedding in zip(code_snippets_info, code_embeddings):
                    if embedding:  # Only include if embedding was successful
                        code_snippet = CodeSnippet(
                            content=snippet_info["content"],
                            embedding=embedding,
                            metadata={
                                "language": snippet_info["language"],
                                "snippet_type": snippet_info["type"],
                                "char_count": snippet_info["char_count"],
                                "line_count": snippet_info["line_count"],
                            }
                        )
                        code_snippets.append(code_snippet)
                        
            except Exception as e:
                logger.warning(f"Code embedding failed: {e}")
                # Continue without code embeddings
        
        # Step 3: Chunk the cleaned content
        text_chunks = self.text_chunker.chunk_text(cleaned_content)
        
        # Step 4: Generate embeddings and summaries for text chunks
        processed_chunks = []
        if text_chunks:
            logger.info(f"Processing {len(text_chunks)} text chunks")
            
            # Generate embeddings
            chunk_contents = [chunk.content for chunk in text_chunks]
            
            try:
                # Generate embeddings and summaries concurrently
                embeddings_task = self.embedding_service.embed_batch(chunk_contents)
                summaries_task = self.summarization_service.summarize_batch(chunk_contents)
                
                embeddings, summaries = await asyncio.gather(
                    embeddings_task, summaries_task, return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(embeddings, Exception):
                    logger.warning(f"Text embedding failed: {embeddings}")
                    embeddings = [None] * len(chunk_contents)
                    
                if isinstance(summaries, Exception):
                    logger.warning(f"Summarization failed: {summaries}")
                    summaries = [None] * len(chunk_contents)
                
                # Create processed chunks
                for chunk, embedding, summary in zip(text_chunks, embeddings, summaries):
                    if embedding:  # Only include if embedding was successful
                        processed_chunk = ProcessedChunk(
                            content=chunk.content,
                            embedding=embedding,
                            metadata=chunk.metadata,
                            tokens=chunk.tokens,
                            summary=summary,
                            summary_model=self.summarization_service.model if summary else None,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                        )
                        processed_chunks.append(processed_chunk)
                        
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                # Continue with reduced functionality
        
        # Step 5: Create final processed document
        processed_document = ProcessedDocument(
            url=url,
            title=title,
            content=content,  # Keep original content
            chunks=processed_chunks,
            code_snippets=code_snippets,
            metadata={
                **metadata,
                "processing_stats": {
                    "original_length": len(content),
                    "cleaned_length": len(cleaned_content),
                    "total_chunks": len(processed_chunks),
                    "total_code_snippets": len(code_snippets),
                    "chunks_with_embeddings": len(processed_chunks),
                    "code_snippets_with_embeddings": len(code_snippets),
                }
            }
        )
        
        logger.info(f"Document processing complete: {len(processed_chunks)} chunks, {len(code_snippets)} code snippets")
        return processed_document


__all__ = ["DocumentProcessor", "CodeSnippetExtractor", "ProcessingResult"]