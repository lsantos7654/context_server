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
        target_length: int = 150,
    ):
        """Initialize the summarization service.
        
        Args:
            model: OpenAI model to use for summarization
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            target_length: Target summary length in characters (150 for 3-5 sentences)
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
                    max_tokens=100,  # Limit output size for summaries
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