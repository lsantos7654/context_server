"""HTTP client for Context Server API communication."""

import json
import logging

import httpx
from pydantic import BaseModel

from context_server.mcp_server.config import Config
from context_server.models.api.contexts import (
    ContextDeleteResponse,
    ContextListResponse,
    ContextResponse,
)
from context_server.models.api.documents import (
    ChunkResponse,
    CodeSnippetResponse,
    CodeSnippetsResponse,
    DirectoryExtractionResponse,
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentsResponse,
)
from context_server.models.api.search import (
    CodeSearchResponse,
    CompactCodeSearchResponse,
    CompactSearchResponse,
    SearchResponse,
)
from context_server.models.api.system import (
    ActiveJobsResponse,
    JobCancelResponse,
    JobCleanupResponse,
    JobCreateResponse,
    JobStatusResponse,
)

# Union type for all possible API responses
APIResponse = (
    ContextResponse
    | ContextListResponse
    | ContextDeleteResponse
    | DocumentContentResponse
    | DocumentsResponse
    | DocumentDeleteResponse
    | ChunkResponse
    | CodeSnippetResponse
    | CodeSnippetsResponse
    | CompactSearchResponse
    | CompactCodeSearchResponse
    | SearchResponse
    | CodeSearchResponse
    | DirectoryExtractionResponse
    | JobCreateResponse
    | JobStatusResponse
    | JobCancelResponse
    | JobCleanupResponse
    | ActiveJobsResponse
    | dict  # Raw dict for unknown responses
    | list  # List responses
)

logger = logging.getLogger(__name__)


class ContextServerError(Exception):
    """Exception raised when Context Server API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict | None = None,
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
        self, endpoint: str, data: dict | None = None, params: dict | None = None
    ) -> APIResponse:
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
                raw_response = self._handle_response(response)
                return self._parse_response(endpoint, "POST", raw_response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    async def get(self, endpoint: str, params: dict | None = None) -> APIResponse:
        """Make a GET request to the Context Server API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.get(url, params=params)
                raw_response = self._handle_response(response)
                return self._parse_response(endpoint, "GET", raw_response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    async def delete(self, endpoint: str) -> APIResponse | None:
        """Make a DELETE request to the Context Server API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.delete(url)

                # DELETE might return 204 (No Content)
                if response.status_code == 204:
                    # For DELETE operations, return appropriate success response
                    if "/contexts/" in endpoint:
                        return ContextDeleteResponse(
                            success=True, message="Context deleted"
                        )
                    elif "/documents/" in endpoint:
                        return DocumentDeleteResponse(
                            success=True, message="Documents deleted"
                        )
                    else:
                        return {"success": True}

                raw_response = self._handle_response(response)
                return self._parse_response(endpoint, "DELETE", raw_response)

        except httpx.TimeoutException:
            raise ContextServerError(f"Request timeout: {url}")
        except httpx.ConnectError:
            raise ContextServerError(
                f"Cannot connect to Context Server at {self.base_url}"
            )
        except Exception as e:
            raise ContextServerError(f"Request failed: {str(e)}")

    def _handle_response(self, response: httpx.Response) -> dict | list:
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

    def _parse_response(
        self, endpoint: str, method: str, raw_response: dict | list
    ) -> APIResponse:
        """Parse raw HTTP response into appropriate Pydantic model based on endpoint."""
        try:
            # Handle list responses (mainly for contexts)
            if isinstance(raw_response, list):
                if "/api/contexts" in endpoint or "/contexts" in endpoint:
                    contexts = [ContextResponse(**ctx) for ctx in raw_response]
                    return ContextListResponse(contexts=contexts, total=len(contexts))
                return raw_response  # Fallback for other lists

            # Handle dict responses based on endpoint patterns
            if not isinstance(raw_response, dict):
                return raw_response

            # Context endpoints
            if "/api/contexts" in endpoint or "/contexts" in endpoint:
                if method == "POST":
                    return ContextResponse(**raw_response)
                elif method == "GET" and not endpoint.endswith("/contexts"):
                    return ContextResponse(**raw_response)
                elif method == "DELETE":
                    return ContextDeleteResponse(**raw_response)

            # Document endpoints
            elif "/documents" in endpoint:
                if method == "POST":
                    # Document ingestion returns job info, use JobCreateResponse
                    if "job_id" in raw_response:
                        return JobCreateResponse(**raw_response)
                    # Directory extraction uses specific response
                    elif "processed_files" in raw_response:
                        return DirectoryExtractionResponse(**raw_response)
                    # Fallback to raw response for other POST operations
                    return raw_response
                elif method == "GET":
                    if "documents" in endpoint and not endpoint.split("/")[-1]:
                        return DocumentsResponse(**raw_response)
                    else:
                        return DocumentContentResponse(**raw_response)
                elif method == "DELETE":
                    return DocumentDeleteResponse(**raw_response)

            # Search endpoints
            elif "/search" in endpoint:
                if "/code" in endpoint:
                    # Check if compact format was requested
                    if "format=compact" in str(endpoint):
                        return CompactCodeSearchResponse(**raw_response)
                    return CodeSearchResponse(**raw_response)
                else:
                    # Check if compact format was requested
                    if "format=compact" in str(endpoint):
                        return CompactSearchResponse(**raw_response)
                    return SearchResponse(**raw_response)

            # Chunk endpoints
            elif "/chunk/" in endpoint:
                return ChunkResponse(**raw_response)

            # Code snippet endpoints
            elif "/snippet" in endpoint:
                if endpoint.endswith("/snippets"):
                    return CodeSnippetsResponse(**raw_response)
                return CodeSnippetResponse(**raw_response)

            # Job endpoints
            elif "/jobs" in endpoint:
                if method == "POST":
                    return JobCreateResponse(**raw_response)
                elif "/cancel" in endpoint:
                    return JobCancelResponse(**raw_response)
                elif "/cleanup" in endpoint:
                    return JobCleanupResponse(**raw_response)
                elif endpoint.endswith("/jobs"):
                    return ActiveJobsResponse(**raw_response)
                else:
                    return JobStatusResponse(**raw_response)

            # Fallback to raw response for unknown endpoints
            return raw_response

        except Exception as e:
            logger.warning(f"Failed to parse response for {endpoint}: {e}")
            # Return raw response as fallback
            return raw_response

    async def health_check(self) -> bool:
        """Check if Context Server is healthy and reachable."""
        try:
            response = await self.get("/health")
            return isinstance(response, dict) and response.get("status") != "unhealthy"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
