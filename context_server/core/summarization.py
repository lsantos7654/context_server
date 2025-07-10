"""LLM-based summarization service for generating compact chunk summaries."""

import asyncio
import logging
import os
from typing import Optional

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
{content}

Summary:"""

        # Prompt for code snippet summarization
        self.code_summarization_prompt = """Analyze this code snippet and create a concise summary in 80-300 characters. Focus on what the code DOES, not what it is. Include function/class names if present.

Examples:
- "Configures AsyncWebCrawler with deep crawling strategy and content filtering"
- "Defines User class with authentication methods and password validation"  
- "Processes API response data and handles error cases with retry logic"

Code:
{code_content}

Summary:"""

    async def summarize_chunk(
        self, content: str, timeout: float = 30.0
    ) -> tuple[str | None, str | None]:
        """Generate a summary for a text chunk.
        
        Args:
            content: Text content to summarize
            timeout: API timeout in seconds
            
        Returns:
            tuple: (summary_text, error_message)
                   Returns (None, error) if summarization fails
        """
        if not self.client:
            return None, "OpenAI API key not configured"

        # Skip summarization for very short content
        if len(content) <= 150:
            return content[:self.target_length], None

        try:
            # Format the prompt
            prompt = self.summarization_prompt.format(content=content)

            # Call OpenAI API with retry logic
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            max_tokens=200,  # Increased for 3-5 sentence summaries
                            temperature=0.3,  # Lower temperature for consistent summaries
                        ),
                        timeout=timeout,
                    )

                    summary = response.choices[0].message.content.strip()

                    # No truncation for AI-generated summaries - they're already optimized
                    # The AI was specifically instructed to generate appropriate length summaries

                    logger.debug(
                        f"Generated summary: {len(content)} chars -> {len(summary)} chars"
                    )
                    return summary, None

                except RateLimitError:
                    if attempt < 2:
                        # Exponential backoff for rate limits
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/3"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded after retries"

                except APITimeoutError:
                    if attempt < 2:
                        logger.warning(f"Timeout on attempt {attempt + 1}/3, retrying...")
                        continue
                    else:
                        return None, "API timeout after retries"

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return None, str(e)

        return None, "All retry attempts failed"

    async def summarize_code_snippet(
        self, code_content: str, timeout: float = 15.0
    ) -> tuple[str | None, str | None]:
        """Generate a summary for a code snippet.
        
        Args:
            code_content: Code content to summarize
            timeout: API timeout in seconds (shorter than regular summarization)
            
        Returns:
            tuple: (summary_text, error_message)
                   Returns (None, error) if summarization fails
        """
        if not self.client:
            return None, "OpenAI API key not configured"

        # Skip summarization for very short code
        if len(code_content) <= 50:
            return code_content.strip(), None

        try:
            # Format the code summarization prompt
            prompt = self.code_summarization_prompt.format(code_content=code_content)

            # Call OpenAI API with retry logic
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            max_tokens=75,  # Limit for 80-300 char summaries
                            temperature=0.2,  # Lower temperature for consistent summaries
                        ),
                        timeout=timeout,
                    )

                    summary = response.choices[0].message.content.strip()
                    
                    # Ensure summary is within the expected range
                    if len(summary) > 300:
                        summary = summary[:297] + "..."
                    elif len(summary) < 80:
                        # If too short, pad with basic info
                        lines = code_content.split('\n')
                        line_count = len([line for line in lines if line.strip()])
                        summary += f" ({line_count} lines)"

                    logger.debug(
                        f"Generated code summary: {len(code_content)} chars -> {len(summary)} chars"
                    )
                    return summary, None

                except RateLimitError:
                    if attempt < 2:
                        # Exponential backoff for rate limits
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/3"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded after retries"

                except APITimeoutError:
                    if attempt < 2:
                        logger.warning(f"Timeout on attempt {attempt + 1}/3, retrying...")
                        continue
                    else:
                        return None, "API timeout after retries"

        except Exception as e:
            logger.error(f"Code summarization failed: {e}")
            return None, str(e)

        return None, "All retry attempts failed"

    async def summarize_chunks_batch(
        self, chunks: list[tuple[str, str]], batch_size: int = 5
    ) -> list[tuple[str, str | None, str | None]]:
        """Summarize multiple chunks in batches.
        
        Args:
            chunks: List of (chunk_id, content) tuples
            batch_size: Number of chunks to process concurrently
            
        Returns:
            List of (chunk_id, summary, error) tuples
        """
        results = []

        # Process chunks in batches to avoid overwhelming the API
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(f"Processing summarization batch {i//batch_size + 1}")

            # Create tasks for concurrent processing
            tasks = [
                self.summarize_chunk(content) for chunk_id, content in batch
            ]

            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results with chunk IDs
            for (chunk_id, content), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append((chunk_id, None, str(result)))
                else:
                    summary, error = result
                    results.append((chunk_id, summary, error))

            # Small delay between batches to be nice to the API
            if i + batch_size < len(chunks):
                await asyncio.sleep(1)

        return results

    def get_fallback_summary(self, content: str) -> str:
        """Generate a fallback summary by truncating content.
        
        Args:
            content: Original text content
            
        Returns:
            Truncated content with ellipsis
        """
        if len(content) <= self.target_length:
            return content

        # Try to truncate at sentence boundary for fallback
        truncated = content[: self.target_length * 2]  # Allow more length for fallback
        last_period = truncated.rfind(".")
        last_exclamation = truncated.rfind("!")
        last_question = truncated.rfind("?")

        # Find the last sentence ending
        last_sentence_end = max(last_period, last_exclamation, last_question)

        if last_sentence_end > self.target_length:
            # Truncate at sentence boundary if it's reasonable length
            return content[: last_sentence_end + 1]
        else:
            # Truncate at character boundary with ellipsis
            return content[: self.target_length * 2 - 3] + "..."