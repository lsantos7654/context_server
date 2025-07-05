"""Multi-embedding service with code-specific models and content-aware routing."""

import asyncio
import logging
import os
from enum import Enum
from typing import Any, List, Optional, Union

import httpx
import openai
from openai import OpenAI

from .content_analysis import ContentAnalysis

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models."""

    # OpenAI models (general purpose)
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_ADA = "text-embedding-ada-002"

    # Code-specific models
    VOYAGE_CODE = "voyage-code-3"  # Voyage AI's code embedding model


class ContentType(Enum):
    """Content types for routing."""

    GENERAL = "general"
    CODE = "code"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"


class EmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        raise NotImplementedError

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        raise NotImplementedError


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: str | None = None
    ):
        super().__init__(api_key)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("No OpenAI API key provided")

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
            text = self._prepare_text(text)
            response = await self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        try:
            prepared_texts = [self._prepare_text(text) for text in texts]

            # Process in batches to respect API limits
            max_batch_size = 100
            all_embeddings = []

            for i in range(0, len(prepared_texts), max_batch_size):
                batch = prepared_texts[i : i + max_batch_size]
                response = await self.client.embeddings.create(
                    model=self.model, input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Rate limiting
                if i + max_batch_size < len(prepared_texts):
                    await asyncio.sleep(0.1)

            return all_embeddings
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimensions.get(self.model, 1536)

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding."""
        text = " ".join(text.split())
        max_chars = 30000  # Conservative limit for OpenAI
        if len(text) > max_chars:
            text = text[:max_chars]
        return text


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embedding provider for code-specific embeddings."""

    def __init__(self, model: str = "voyage-code-3", api_key: str | None = None):
        super().__init__(api_key)
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")

        if not self.api_key:
            logger.warning("No Voyage API key provided")

        # voyage-code-3 supports multiple dimensions, default to 1024
        self.dimension = 1024

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using Voyage AI with retry logic."""
        if not self.api_key:
            raise ValueError("Voyage API key not configured")

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": [self._prepare_text(text)],
                            "input_type": "document",  # For indexing documents
                            "output_dimension": self.dimension,
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 429 and attempt < max_retries - 1:
                        # Rate limited, wait and retry with exponential backoff
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Rate limited by Voyage API, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["data"][0]["embedding"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited by Voyage API, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Voyage embedding failed: {e}")
                    raise
            except Exception as e:
                logger.error(f"Voyage embedding failed: {e}")
                raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using Voyage AI with retry logic."""
        if not self.api_key:
            raise ValueError("Voyage API key not configured")

        prepared_texts = [self._prepare_text(text) for text in texts]

        # Optimize batch size based on Voyage limits: 2,000 RPM, 3M TPM
        max_batch_size = 96  # Balanced for efficiency while staying under limits
        all_embeddings = []

        for i in range(0, len(prepared_texts), max_batch_size):
            batch = prepared_texts[i : i + max_batch_size]

            # Retry logic for each batch
            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.voyageai.com/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": self.model,
                                "input": batch,
                                "input_type": "document",
                                "output_dimension": self.dimension,
                            },
                            timeout=60.0,
                        )

                        if response.status_code == 429 and attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                f"Rate limited by Voyage API, retrying batch in {delay}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue

                        response.raise_for_status()
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break  # Success, exit retry loop

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Rate limited by Voyage API, retrying batch in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Voyage batch embedding failed: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Voyage batch embedding failed: {e}")
                    raise

            # Rate limiting between batches - optimized for 2,000 RPM
            if i + max_batch_size < len(prepared_texts):
                await asyncio.sleep(0.1)  # Minimal delay - we have 2,000 RPM available

        return all_embeddings

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding."""
        text = " ".join(text.split())
        # Voyage has generous token limits
        max_chars = 120000  # 120K token limit
        if len(text) > max_chars:
            text = text[:max_chars]
        return text


class MultiEmbeddingService:
    """Multi-embedding service with content-aware routing."""

    def __init__(self):
        # Initialize providers
        self.providers = {
            EmbeddingModel.OPENAI_SMALL: OpenAIProvider("text-embedding-3-small"),
            EmbeddingModel.OPENAI_LARGE: OpenAIProvider("text-embedding-3-large"),
            EmbeddingModel.VOYAGE_CODE: VoyageProvider("voyage-code-3"),
        }

        # Default routing rules based on content analysis
        self.routing_rules = {
            ContentType.CODE: EmbeddingModel.VOYAGE_CODE,
            ContentType.API_REFERENCE: EmbeddingModel.VOYAGE_CODE,
            ContentType.TUTORIAL: EmbeddingModel.OPENAI_SMALL,
            ContentType.DOCUMENTATION: EmbeddingModel.OPENAI_SMALL,
            ContentType.GENERAL: EmbeddingModel.OPENAI_SMALL,
        }

        # Fallback model if primary fails
        self.fallback_model = EmbeddingModel.OPENAI_SMALL

        logger.info("Multi-embedding service initialized")

    def route_content(
        self,
        content_analysis: ContentAnalysis | None = None,
        content_type: str | None = None,
        code_percentage: float = 0.0,
    ) -> EmbeddingModel:
        """Route content to the appropriate embedding model."""

        # Use content analysis if available
        if content_analysis:
            content_type = content_analysis.content_type
            code_percentage = content_analysis.code_percentage

        # Determine content type for routing
        if code_percentage > 30:  # High code content
            routing_type = ContentType.CODE
        elif content_type == "api_reference":
            routing_type = ContentType.API_REFERENCE
        elif content_type == "tutorial":
            routing_type = ContentType.TUTORIAL
        elif content_type in ["concept_explanation", "troubleshooting"]:
            routing_type = ContentType.DOCUMENTATION
        else:
            routing_type = ContentType.GENERAL

        model = self.routing_rules.get(routing_type, self.fallback_model)

        logger.debug(
            f"Routed content to {model.value} (type: {routing_type.value}, code%: {code_percentage:.1f})"
        )
        return model

    async def embed_text(
        self,
        text: str,
        content_analysis: ContentAnalysis | None = None,
        force_model: EmbeddingModel | None = None,
    ) -> dict[str, Any]:
        """Generate embedding for text with intelligent routing."""

        # Determine which model to use
        model = force_model or self.route_content(content_analysis)
        provider = self.providers[model]

        try:
            embedding = await provider.embed_text(text)
            return {
                "embedding": embedding,
                "model": model.value,
                "dimension": provider.get_dimension(),
                "success": True,
            }
        except Exception as e:
            logger.warning(f"Primary model {model.value} failed, trying fallback: {e}")

            # Try fallback model if primary fails
            if model != self.fallback_model:
                try:
                    fallback_provider = self.providers[self.fallback_model]
                    embedding = await fallback_provider.embed_text(text)
                    return {
                        "embedding": embedding,
                        "model": self.fallback_model.value,
                        "dimension": fallback_provider.get_dimension(),
                        "success": True,
                        "used_fallback": True,
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")

            # Create dummy embedding as last resort
            dimension = provider.get_dimension()
            return {
                "embedding": [0.0] * dimension,
                "model": model.value,
                "dimension": dimension,
                "success": False,
                "error": str(e),
            }

    async def embed_batch(
        self,
        texts: list[str],
        content_analyses: list[ContentAnalysis | None] | None = None,
        force_model: EmbeddingModel | None = None,
    ) -> list[dict[str, Any]]:
        """Generate embeddings for batch with intelligent routing."""

        if force_model:
            # All texts use the same model
            try:
                provider = self.providers[force_model]
                embeddings = await provider.embed_batch(texts)
                return [
                    {
                        "embedding": emb,
                        "model": force_model.value,
                        "dimension": provider.get_dimension(),
                        "success": True,
                    }
                    for emb in embeddings
                ]
            except Exception as e:
                logger.error(f"Batch embedding with {force_model.value} failed: {e}")
                # Create dummy embeddings
                dimension = provider.get_dimension()
                return [
                    {
                        "embedding": [0.0] * dimension,
                        "model": force_model.value,
                        "dimension": dimension,
                        "success": False,
                        "error": str(e),
                    }
                    for _ in texts
                ]

        # Route each text individually
        results = []
        for i, text in enumerate(texts):
            analysis = (
                content_analyses[i]
                if content_analyses and i < len(content_analyses)
                else None
            )
            result = await self.embed_text(text, analysis)
            results.append(result)

            # Small delay to be respectful to APIs
            if i < len(texts) - 1:
                await asyncio.sleep(0.05)

        return results

    async def embed_with_multiple_models(
        self, text: str, models: list[EmbeddingModel]
    ) -> dict[str, dict[str, Any]]:
        """Generate embeddings using multiple models for comparison."""
        results = {}

        for model in models:
            try:
                result = await self.embed_text(text, force_model=model)
                results[model.value] = result
            except Exception as e:
                logger.error(f"Failed to embed with {model.value}: {e}")
                provider = self.providers[model]
                results[model.value] = {
                    "embedding": [0.0] * provider.get_dimension(),
                    "model": model.value,
                    "dimension": provider.get_dimension(),
                    "success": False,
                    "error": str(e),
                }

        return results

    def get_available_models(self) -> list[str]:
        """Get list of available embedding models."""
        available = []
        for model, provider in self.providers.items():
            if hasattr(provider, "api_key") and provider.api_key:
                available.append(model.value)
        return available

    async def health_check(self) -> dict[str, bool]:
        """Check health of all embedding providers."""
        health_status = {}

        for model, provider in self.providers.items():
            try:
                if hasattr(provider, "api_key") and not provider.api_key:
                    health_status[model.value] = False
                    continue

                # Try a simple embedding
                await provider.embed_text("test")
                health_status[model.value] = True
            except Exception as e:
                logger.warning(f"Health check failed for {model.value}: {e}")
                health_status[model.value] = False

        return health_status
