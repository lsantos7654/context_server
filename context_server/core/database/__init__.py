"""Database operations for Context Server - Split from monolithic storage.py.

This package organizes database operations into focused modules:
- connection.py: Connection management & health checks
- schema.py: Schema creation & migrations
- models/: Entity CRUD operations (contexts, documents, chunks, etc.)
- search/: Search operations (vector, fulltext, formatters)
- operations/: Helper operations (code preview, summaries)

The main DatabaseManager class composes all these modules while maintaining
the exact same API as the original storage.py for backward compatibility.
"""

from context_server.core.database.connection import DatabaseManager

__all__ = ["DatabaseManager"]
