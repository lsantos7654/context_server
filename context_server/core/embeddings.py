"""Embedding service for generating vector embeddings."""

import asyncio
import logging
import os

import httpx
import openai

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. Embedding service will not work."
            )

        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.api_key) if self.api_key else None

        # Embedding dimensions for different models
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        try:
            # Clean and truncate text if needed
            text = self._prepare_text(text)

            response = await self.client.embeddings.create(model=self.model, input=text)

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)})")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        try:
            # Clean and prepare texts
            prepared_texts = [self._prepare_text(text) for text in texts]

            # OpenAI has limits on batch size, so we'll process in chunks
            max_batch_size = 100  # Conservative limit
            all_embeddings = []

            for i in range(0, len(prepared_texts), max_batch_size):
                batch = prepared_texts[i : i + max_batch_size]

                response = await self.client.embeddings.create(
                    model=self.model, input=batch
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch")

                # Small delay between batches to respect rate limits
                if i + max_batch_size < len(prepared_texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Generated {len(all_embeddings)} embeddings total")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding generation."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate if too long (OpenAI has token limits)
        # For text-embedding-3-small, limit is around 8192 tokens
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
