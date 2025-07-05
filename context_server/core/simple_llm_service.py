"""Simple LLM service interface for content analysis."""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleLLMService:
    """Simple LLM service for content analysis tasks."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI API key and model."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.available = self.api_key is not None

        if not self.available:
            logger.warning("No OpenAI API key found - LLM analysis will be disabled")

    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from LLM."""

        if not self.available:
            raise Exception("LLM service not available - no API key")

        try:
            # Try to import and use OpenAI
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical content analyzer. Provide accurate, structured responses in the requested format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            logger.error(
                "OpenAI library not installed - install with: pip install openai"
            )
            raise Exception("OpenAI library not available")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        if not self.available:
            return False

        try:
            # Simple test query
            response = await self.generate_response("Test", max_tokens=10)
            return len(response) > 0
        except Exception:
            return False
