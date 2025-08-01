"""HTTP client for Context Server API communication."""

import json
import logging
from typing import Any, Optional, Union

import httpx

from .config import Config

logger = logging.getLogger(__name__)


class ContextServerError(Exception):
    """Exception raised when Context Server API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ContextServerClient:
    """Simple HTTP client for Context Server API."""

    def __init__(self, config: Config):
        """Initialize the client with configuration."""
        self.config = config
        self.base_url = config.context_server_url.rstrip("/")

    async def post(
        self, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None
    ) -> dict[str, Any]:
        """Make a POST request to the Context Server API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.post(
                    url,
                    json=data,
                    params=params,
                    headers={"Content-Type": "application/json"},
                )
                return self._handle_response(response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    async def get(
        self, endpoint: str, params: Optional[dict] = None
    ) -> Union[dict[str, Any], list[Any]]:
        """Make a GET request to the Context Server API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.get(url, params=params)
                return self._handle_response(response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    async def delete(self, endpoint: str) -> Optional[dict[str, Any]]:
        """Make a DELETE request to the Context Server API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.delete(url)

                # DELETE might return 204 (No Content)
                if response.status_code == 204:
                    return {"success": True}

                return self._handle_response(response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    def _handle_response(
        self, response: httpx.Response
    ) -> Union[dict[str, Any], list[Any]]:
        """Handle HTTP response and extract JSON data."""
        try:
            if response.status_code >= 400:
                # Try to extract error details from response
                error_details = {}
                try:
                    error_data = response.json()
                    error_details = error_data if isinstance(error_data, dict) else {}
                except Exception:
                    pass

                error_message = error_details.get(
                    "detail", f"HTTP {response.status_code}"
                )
                raise ContextServerError(
                    error_message,
                    status_code=response.status_code,
                    details=error_details,
                )

            # Try to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                # If no JSON content, return success indicator
                return {"success": True}

        except ContextServerError:
            raise
        except Exception as e:
            raise ContextServerError(f"Failed to parse response: {str(e)}")

    async def health_check(self) -> bool:
        """Check if Context Server is healthy and reachable."""
        try:
            response = await self.get("/health")
            return isinstance(response, dict) and response.get("status") != "unhealthy"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
