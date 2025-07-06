"""Error handling decorators and utilities for FastAPI endpoints."""

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_api_errors(
    operation_name: str,
    not_found_message: str = None,
    conflict_message: str = None,
    default_error_message: str = None,
) -> Callable[[F], F]:
    """Decorator for standardized API error handling.

    Args:
        operation_name: Name of the operation for logging (e.g., "create context")
        not_found_message: Custom message for 404 errors
        conflict_message: Custom message for 409 errors
        default_error_message: Custom message for 500 errors

    Usage:
        @handle_api_errors("create context", conflict_message="Context already exists")
        async def create_context(context_data: ContextCreate, db: DatabaseManager):
            # Your endpoint logic here
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPExceptions (they're already handled)
                raise
            except Exception as e:
                # Log the error with context
                logger.error(f"Failed to {operation_name}: {e}", exc_info=True)

                # Determine appropriate error response
                error_msg = str(e).lower()

                if "not found" in error_msg and not_found_message:
                    raise HTTPException(status_code=404, detail=not_found_message)
                elif "already exists" in error_msg and conflict_message:
                    raise HTTPException(status_code=409, detail=conflict_message)
                elif "not found" in error_msg:
                    raise HTTPException(status_code=404, detail=f"Resource not found")
                elif "already exists" in error_msg:
                    raise HTTPException(
                        status_code=409, detail=f"Resource already exists"
                    )
                else:
                    # Generic server error
                    error_detail = (
                        default_error_message or f"Failed to {operation_name}"
                    )
                    raise HTTPException(status_code=500, detail=error_detail)

        return wrapper

    return decorator


def handle_context_errors(operation: str) -> Callable[[F], F]:
    """Specialized decorator for context-related operations."""
    return handle_api_errors(
        operation_name=f"{operation} context",
        not_found_message="Context not found",
        conflict_message="Context already exists",
        default_error_message=f"Failed to {operation} context",
    )


def handle_document_errors(operation: str) -> Callable[[F], F]:
    """Specialized decorator for document-related operations."""
    return handle_api_errors(
        operation_name=f"{operation} document",
        not_found_message="Document not found",
        conflict_message="Document already exists",
        default_error_message=f"Failed to {operation} document",
    )


def handle_search_errors(operation: str = "search") -> Callable[[F], F]:
    """Specialized decorator for search-related operations."""
    return handle_api_errors(
        operation_name=operation,
        not_found_message="Context not found",
        default_error_message=f"Failed to perform {operation}",
    )


class APIErrorHandler:
    """Context manager for standardized error handling in FastAPI endpoints."""

    def __init__(
        self,
        operation_name: str,
        not_found_message: str = None,
        conflict_message: str = None,
        default_error_message: str = None,
    ):
        self.operation_name = operation_name
        self.not_found_message = not_found_message
        self.conflict_message = conflict_message
        self.default_error_message = default_error_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True

        if issubclass(exc_type, HTTPException):
            # Let HTTPExceptions pass through
            return False

        # Log the error
        logger.error(f"Failed to {self.operation_name}: {exc_val}", exc_info=True)

        # Determine appropriate error response
        error_msg = str(exc_val).lower()

        if "not found" in error_msg and self.not_found_message:
            raise HTTPException(status_code=404, detail=self.not_found_message)
        elif "already exists" in error_msg and self.conflict_message:
            raise HTTPException(status_code=409, detail=self.conflict_message)
        elif "not found" in error_msg:
            raise HTTPException(status_code=404, detail="Resource not found")
        elif "already exists" in error_msg:
            raise HTTPException(status_code=409, detail="Resource already exists")
        else:
            # Generic server error
            error_detail = (
                self.default_error_message or f"Failed to {self.operation_name}"
            )
            raise HTTPException(status_code=500, detail=error_detail)
