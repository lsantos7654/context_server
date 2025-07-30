"""Database utility functions to eliminate code duplication."""

import json
import uuid


def convert_embedding_to_postgres(embedding: list[float]) -> str:
    """Convert Python embedding list to PostgreSQL halfvec format.
    
    Args:
        embedding: List of float values representing the embedding vector
        
    Returns:
        String representation suitable for PostgreSQL halfvec type
        
    Example:
        >>> convert_embedding_to_postgres([0.1, 0.2, 0.3])
        '[0.1,0.2,0.3]'
    """
    return "[" + ",".join(map(str, embedding)) + "]"


def parse_metadata(metadata_json: str | None) -> dict:
    """Safely parse JSON metadata with fallback to empty dict.
    
    Args:
        metadata_json: JSON string or None
        
    Returns:
        Parsed dictionary or empty dict if None/invalid
        
    Example:
        >>> parse_metadata('{"key": "value"}')
        {'key': 'value'}
        >>> parse_metadata(None)
        {}
    """
    return json.loads(metadata_json) if metadata_json else {}


def format_uuid(uuid_value) -> str:
    """Consistently format UUID to string.
    
    Args:
        uuid_value: UUID object or string
        
    Returns:
        String representation of UUID
        
    Example:
        >>> format_uuid(uuid.uuid4())
        'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    """
    return str(uuid_value)


def parse_uuid(uuid_str: str) -> uuid.UUID:
    """Consistently parse string to UUID object.
    
    Args:
        uuid_str: String representation of UUID
        
    Returns:
        UUID object
        
    Example:
        >>> parse_uuid('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
        UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
    """
    return uuid.UUID(uuid_str)


__all__ = [
    "convert_embedding_to_postgres",
    "parse_metadata", 
    "format_uuid",
    "parse_uuid"
]