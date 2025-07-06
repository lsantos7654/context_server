#!/usr/bin/env python3
"""Startup script for the Context Server MCP server."""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from context_server.mcp_server.main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)
