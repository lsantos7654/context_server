"""Embedding service for generating vector embeddings."""

import asyncio
import logging
import os
from typing import Optional

import httpx
import openai
from openai import APITimeoutError, RateLimitError

try:
    import voyageai
except ImportError:
    voyageai = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(
        self, model: str = "text-embedding-3-large", api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. Embedding service will not work."
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

        # Embedding dimensions for different models
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    async def embed_text(self, text: str, timeout: float = 30.0) -> list[float]:
        """Generate embedding for a single text with timeout and retry logic."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        # Clean and truncate text if needed
        text = self._prepare_text(text)

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Use asyncio.wait_for to enforce total timeout
                response = await asyncio.wait_for(
                    self.client.embeddings.create(model=self.model, input=text),
                    timeout=timeout,
                )

                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding for text (length: {len(text)})")
                return embedding

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff for rate limits
                    delay = base_delay * (2**attempt) + (
                        attempt * 5
                    )  # 1, 7, 17 seconds
                    logger.warning(
                        f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise

            except (APITimeoutError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Timeout error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Embedding generation timed out after {max_retries} attempts"
                    )
                    raise

            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                raise

    async def embed_batch(
        self, texts: list[str], timeout: float = 300.0, batch_timeout: float = 60.0
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts with comprehensive timeout and retry logic."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        # Clean and prepare texts
        prepared_texts = [self._prepare_text(text) for text in texts]

        # OpenAI has limits on batch size, so we'll process in chunks
        max_batch_size = 50  # More conservative limit to reduce timeout risk
        all_embeddings = []

        total_batches = (len(prepared_texts) + max_batch_size - 1) // max_batch_size

        try:
            # Overall timeout for the entire batch operation using wait_for
            async def _process_all_batches():
                for batch_idx in range(0, len(prepared_texts), max_batch_size):
                    batch = prepared_texts[batch_idx : batch_idx + max_batch_size]
                    batch_num = (batch_idx // max_batch_size) + 1

                    logger.info(
                        f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)"
                    )

                    # Retry logic for each batch
                    max_retries = 3
                    base_delay = 2.0

                    for attempt in range(max_retries):
                        try:
                            # Individual batch timeout
                            response = await asyncio.wait_for(
                                self.client.embeddings.create(
                                    model=self.model, input=batch
                                ),
                                timeout=batch_timeout,
                            )

                            batch_embeddings = [
                                data.embedding for data in response.data
                            ]
                            all_embeddings.extend(batch_embeddings)

                            logger.debug(
                                f"Generated {len(batch_embeddings)} embeddings in batch {batch_num}"
                            )
                            break  # Success, move to next batch

                        except RateLimitError as e:
                            if attempt < max_retries - 1:
                                # Progressive delay for rate limits
                                delay = base_delay * (2**attempt) + (
                                    batch_num * 2
                                )  # Account for batch position
                                logger.warning(
                                    f"Rate limit hit on batch {batch_num}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    f"Rate limit exceeded for batch {batch_num} after {max_retries} attempts"
                                )
                                raise

                        except (APITimeoutError, asyncio.TimeoutError) as e:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2**attempt)
                                logger.warning(
                                    f"Timeout on batch {batch_num}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    f"Batch {batch_num} timed out after {max_retries} attempts"
                                )
                                # For batch operations, we can continue with partial results
                                logger.warning(
                                    f"Continuing with partial embeddings ({len(all_embeddings)} so far)"
                                )
                                break

                        except Exception as e:
                            logger.error(
                                f"Failed to generate embeddings for batch {batch_num}: {e}"
                            )
                            # For batch operations, we can continue with partial results
                            logger.warning(
                                f"Skipping failed batch {batch_num}, continuing with partial results"
                            )
                            break

                    # Adaptive delay between batches based on success/failure
                    if batch_idx + max_batch_size < len(prepared_texts):
                        # Longer delay if we've had recent failures
                        delay = 0.5 if len(all_embeddings) > batch_idx else 2.0
                        await asyncio.sleep(delay)

            # Apply overall timeout to the entire process
            await asyncio.wait_for(_process_all_batches(), timeout=timeout)

        except asyncio.TimeoutError:
            logger.error(
                f"Total batch operation timed out after {timeout}s. Returning partial results ({len(all_embeddings)} embeddings)"
            )
            if not all_embeddings:
                raise

        logger.info(
            f"Generated {len(all_embeddings)} embeddings total (requested: {len(texts)})"
        )
        return all_embeddings

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding generation."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate if too long (OpenAI has token limits)
        # For text-embedding-3-large, limit is around 8192 tokens
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = 30000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text from {len(text)} to {max_chars} characters")

        return text

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.dimensions.get(self.model, 1536)

    async def health_check(self) -> bool:
        """Check if the embedding service is available."""
        if not self.client:
            return False

        try:
            # Try a simple embedding request
            await self.embed_text("test")
            return True
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False


class VoyageEmbeddingService:
    """Service for generating code embeddings using Voyage AI API."""

    def __init__(
        self, model: str = "voyage-code-3", api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")

        if not self.api_key:
            logger.warning(
                "No Voyage AI API key provided. Code embedding service will not work."
            )

        # Initialize Voyage client
        self.client = (
            voyageai.AsyncClient(api_key=self.api_key)
            if self.api_key and voyageai
            else None
        )

        if not voyageai:
            logger.warning(
                "VoyageAI package not installed. Code embedding service will not work."
            )

        # Embedding dimensions for different models
        self.dimensions = {
            "voyage-code-3": 2048,
            "voyage-3": 1024,
            "voyage-3-lite": 512,
        }

    async def embed_text(self, text: str, timeout: float = 30.0) -> list[float]:
        """Generate embedding for a single text with timeout and retry logic."""
        if not self.client:
            raise ValueError("Voyage AI API key not configured or package not installed")

        # Clean and truncate text if needed
        text = self._prepare_text(text)

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Use asyncio.wait_for to enforce total timeout
                response = await asyncio.wait_for(
                    self.client.embed(
                        texts=[text], 
                        model=self.model,
                        output_dimension=self.get_dimension(),
                        input_type="document",
                        output_dtype="float"
                    ),
                    timeout=timeout,
                )

                embedding = response.embeddings[0]
                logger.debug(f"Generated code embedding for text (length: {len(text)})")
                return embedding

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff for failures
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Voyage AI embedding error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Voyage AI embedding failed after {max_retries} attempts: {e}")
                    raise

    async def embed_batch(
        self, texts: list[str], timeout: float = 300.0, batch_timeout: float = 60.0
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts with comprehensive timeout and retry logic."""
        if not self.client:
            raise ValueError("Voyage AI API key not configured or package not installed")

        # Clean and prepare texts
        prepared_texts = [self._prepare_text(text) for text in texts]

        # Voyage AI has batch limits, so we'll process in chunks
        max_batch_size = 20  # Conservative limit for code embeddings
        all_embeddings = []

        total_batches = (len(prepared_texts) + max_batch_size - 1) // max_batch_size

        try:
            # Overall timeout for the entire batch operation
            async def _process_all_batches():
                for batch_idx in range(0, len(prepared_texts), max_batch_size):
                    batch = prepared_texts[batch_idx : batch_idx + max_batch_size]
                    batch_num = (batch_idx // max_batch_size) + 1

                    logger.info(
                        f"Processing code embedding batch {batch_num}/{total_batches} ({len(batch)} texts)"
                    )

                    # Retry logic for each batch
                    max_retries = 3
                    base_delay = 2.0

                    for attempt in range(max_retries):
                        try:
                            # Individual batch timeout
                            response = await asyncio.wait_for(
                                self.client.embed(
                                    texts=batch, 
                                    model=self.model,
                                    output_dimension=self.get_dimension(),
                                    input_type="document",
                                    output_dtype="float"
                                ),
                                timeout=batch_timeout,
                            )

                            batch_embeddings = response.embeddings
                            all_embeddings.extend(batch_embeddings)

                            logger.debug(
                                f"Generated {len(batch_embeddings)} code embeddings in batch {batch_num}"
                            )
                            break  # Success, move to next batch

                        except Exception as e:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2**attempt)
                                logger.warning(
                                    f"Voyage AI batch {batch_num} error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}"
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    f"Voyage AI batch {batch_num} failed after {max_retries} attempts: {e}"
                                )
                                # For batch operations, we can continue with partial results
                                logger.warning(
                                    f"Skipping failed batch {batch_num}, continuing with partial results"
                                )
                                break

                    # Adaptive delay between batches
                    if batch_idx + max_batch_size < len(prepared_texts):
                        await asyncio.sleep(0.5)

            # Apply overall timeout to the entire process
            await asyncio.wait_for(_process_all_batches(), timeout=timeout)

        except asyncio.TimeoutError:
            logger.error(
                f"Total batch operation timed out after {timeout}s. Returning partial results ({len(all_embeddings)} embeddings)"
            )
            if not all_embeddings:
                raise

        logger.info(
            f"Generated {len(all_embeddings)} code embeddings total (requested: {len(texts)})"
        )
        return all_embeddings

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding generation."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate if too long (Voyage AI has token limits)
        # For voyage-code-3, limit is around 16k tokens
        max_chars = 60000  # Conservative limit for code
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated code text from {len(text)} to {max_chars} characters")

        return text

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.dimensions.get(self.model, 2048)

    async def health_check(self) -> bool:
        """Check if the embedding service is available."""
        if not self.client:
            return False

        try:
            # Try a simple embedding request
            await self.embed_text("test code")
            return True
        except Exception as e:
            logger.error(f"Voyage AI embedding service health check failed: {e}")
            return False
