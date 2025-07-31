"""LLM-based summarization service for generating compact chunk summaries."""

import asyncio
import logging
import os

import openai
from openai import APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service for generating concise summaries of text chunks using LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        target_length: int = 600,  # Larger summaries for 2500 char chunks (3-4x smaller ratio)
    ):
        """Initialize the summarization service.
        
        Args:
            model: OpenAI model to use for summarization
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            target_length: Target summary length in characters (600 for detailed summaries)
        """
        self.model = model
        self.target_length = target_length
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. Summarization service will not work."
            )

        # Initialize OpenAI client with timeouts
        self.client = (
            openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=30.0,  # 30 second timeout for API calls
                max_retries=2,  # Automatic retries for transient failures
            )
            if self.api_key
            else None
        )

        # Default prompt for summarization
        self.summarization_prompt = """Summarize this documentation chunk in 3-5 clear, informative sentences (50-150 words). Focus on key concepts, actionable information, and main takeaways. Make it comprehensive but concise.

Content:
{text}

Summary:"""

        # Prompt for generating both title and summary
        self.title_summary_prompt = """For this documentation chunk, generate both a concise title and a detailed summary.

TITLE: Generate a descriptive title (5-10 words) that captures the main topic or concept.
SUMMARY: Generate a summary in 3-5 clear, informative sentences (50-150 words) focusing on key concepts, actionable information, and main takeaways.

Content:
{text}

Response format:
Title: [your title here]
Summary: [your summary here]"""

    async def summarize_chunk(self, text: str, timeout: float = 30.0) -> str | None:
        """Generate a summary for a text chunk.
        
        Args:
            text: The text content to summarize
            timeout: Timeout in seconds for the API call
            
        Returns:
            Summary string, or None if summarization failed
        """
        if not self.client:
            logger.warning("OpenAI client not configured, skipping summarization")
            return None

        if not text or not text.strip():
            return None

        # Skip summarization for very short text
        if len(text) <= self.target_length:
            logger.debug(f"Text too short ({len(text)} chars), skipping summarization")
            return None

        try:
            # Prepare the prompt
            prompt = self.summarization_prompt.format(text=text[:4000])  # Limit input size
            
            # Call OpenAI API with timeout
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,  # Allow for comprehensive 150-word summaries
                    temperature=0.3,  # Lower temperature for more consistent summaries
                ),
                timeout=timeout,
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Validate summary
            if not summary or len(summary) < 20:
                logger.warning("Generated summary too short, skipping")
                return None
                
            # No truncation for AI-generated summaries - they're already optimized
            # The AI was specifically instructed to generate appropriate length summaries
                
            logger.debug(f"Generated summary: {len(summary)} chars")
            return summary
            
        except (APITimeoutError, asyncio.TimeoutError):
            logger.warning(f"Summarization timed out after {timeout}s")
            return None
            
        except RateLimitError:
            logger.warning("Rate limit hit during summarization")
            return None
            
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return None

    async def generate_title_and_summary(self, text: str, timeout: float = 30.0) -> tuple[str | None, str | None]:
        """Generate both a title and summary for a text chunk.
        
        Args:
            text: The text content to process
            timeout: Timeout in seconds for the API call
            
        Returns:
            Tuple of (title, summary) - either can be None if generation fails
        """
        if not self.client:
            logger.warning("OpenAI client not configured, skipping title and summary generation")
            return None, None

        if not text or len(text.strip()) < 50:
            logger.debug("Text too short for title and summary generation")
            return None, None

        try:
            # Make API call
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates concise titles and informative summaries for documentation content."
                        },
                        {
                            "role": "user",
                            "content": self.title_summary_prompt.format(text=text)
                        }
                    ],
                    temperature=0.3,  # Low temperature for consistent output
                    max_tokens=200,   # Enough for both title and summary
                ),
                timeout=timeout
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Parse the response to extract title and summary
            title = None
            summary = None
            
            lines = full_response.split('\n')
            for line in lines:
                if line.startswith('Title:'):
                    title = line[6:].strip()
                elif line.startswith('Summary:'):
                    summary = line[8:].strip()
            
            # Validate results
            if title and len(title) < 5:
                title = None
            if summary and len(summary) < 20:
                summary = None
                
            logger.debug(f"Generated title: {title}, summary: {len(summary) if summary else 0} chars")
            return title, summary
            
        except (APITimeoutError, asyncio.TimeoutError):
            logger.warning(f"Title and summary generation timed out after {timeout}s")
            return None, None
            
        except RateLimitError:
            logger.warning("Rate limit hit during title and summary generation")
            return None, None
            
        except Exception as e:
            logger.warning(f"Title and summary generation failed: {e}")
            return None, None

    async def summarize_batch(
        self, 
        texts: list[str], 
        timeout: float = 120.0,
        max_concurrent: int = 3
    ) -> list[str | None]:
        """Generate summaries for multiple text chunks concurrently.
        
        Args:
            texts: List of text chunks to summarize
            timeout: Timeout in seconds for the entire batch operation
            max_concurrent: Maximum number of concurrent summarization requests
            
        Returns:
            List of summaries (same order as input), None for failed summarizations
        """
        if not self.client:
            logger.warning("OpenAI client not configured, skipping batch summarization")
            return [None] * len(texts)

        if not texts:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _summarize_with_semaphore(text: str, index: int) -> tuple[int, str | None]:
            """Summarize a single text with semaphore control."""
            async with semaphore:
                try:
                    summary = await self.summarize_chunk(text, timeout=30.0)
                    return index, summary
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk {index}: {e}")
                    return index, None

        try:
            # Execute all summarizations concurrently with timeout
            tasks = [
                _summarize_with_semaphore(text, i) 
                for i, text in enumerate(texts)
            ]
            
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results and maintain order
            summaries = [None] * len(texts)
            successful_count = 0
            
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    index, summary = result
                    summaries[index] = summary
                    if summary is not None:
                        successful_count += 1
                else:
                    logger.warning(f"Unexpected result format: {result}")
            
            logger.info(f"Batch summarization completed: {successful_count}/{len(texts)} successful")
            return summaries
            
        except asyncio.TimeoutError:
            logger.error(f"Batch summarization timed out after {timeout}s")
            # Return partial results if possible
            return [None] * len(texts)
            
        except Exception as e:
            logger.error(f"Batch summarization failed: {e}")
            return [None] * len(texts)

    async def generate_titles_and_summaries_batch(
        self, 
        texts: list[str], 
        timeout: float = 120.0,
        max_concurrent: int = 3
    ) -> list[tuple[str | None, str | None]]:
        """Generate titles and summaries for multiple text chunks concurrently.
        
        Args:
            texts: List of text chunks to process
            timeout: Timeout in seconds for the entire batch operation
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of (title, summary) tuples (same order as input), None values for failed generations
        """
        if not self.client:
            logger.warning("OpenAI client not configured, skipping batch title and summary generation")
            return [(None, None)] * len(texts)

        if not texts:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _generate_with_semaphore(text: str, index: int) -> tuple[int, tuple[str | None, str | None]]:
            """Generate title and summary for a single text with semaphore control."""
            async with semaphore:
                try:
                    title, summary = await self.generate_title_and_summary(text)
                    return index, (title, summary)
                except Exception as e:
                    logger.warning(f"Failed to generate title and summary for chunk {index}: {e}")
                    return index, (None, None)

        try:
            # Execute all generations concurrently with timeout
            tasks = [
                _generate_with_semaphore(text, i) 
                for i, text in enumerate(texts)
            ]
            
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results and maintain order
            title_summaries = [(None, None)] * len(texts)
            successful_count = 0
            
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    index, (title, summary) = result
                    title_summaries[index] = (title, summary)
                    if title or summary:
                        successful_count += 1
                else:
                    logger.warning(f"Unexpected result format: {result}")
            
            logger.info(f"Generated titles and summaries for {successful_count}/{len(texts)} chunks")
            return title_summaries
            
        except asyncio.TimeoutError:
            logger.error(f"Batch title and summary generation timed out after {timeout}s")
            return [(None, None)] * len(texts)
            
        except Exception as e:
            logger.error(f"Batch title and summary generation failed: {e}")
            return [(None, None)] * len(texts)

    async def health_check(self) -> bool:
        """Check if the summarization service is available."""
        if not self.client:
            return False

        try:
            # Try a simple summarization request
            test_summary = await self.summarize_chunk(
                "This is a test chunk for health checking the summarization service.", 
                timeout=10.0
            )
            return test_summary is not None
        except Exception as e:
            logger.error(f"Summarization service health check failed: {e}")
            return False


__all__ = ["SummarizationService"]