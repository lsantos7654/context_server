"""Main MCP server for Context Server integration."""

import asyncio
import logging
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .client import ContextServerClient, ContextServerError
from .config import Config
from .tools import ContextServerTools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize server
server = Server("context-server")

# Global variables for configuration and tools
config: Config
client: ContextServerClient
tools: ContextServerTools


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        # Context Management Tools
        types.Tool(
            name="create_context",
            description="Create a new context for storing documentation. Claude can use this to set up organized documentation spaces.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for the context (e.g., 'python-docs', 'ratatui-examples')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of what this context contains",
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model to use for vector search",
                        "default": "text-embedding-3-small",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="list_contexts",
            description="List all available contexts with their metadata. Use this to see what documentation is already available.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_context",
            description="Get detailed information about a specific context including document count and size.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of the context to examine",
                    }
                },
                "required": ["context_name"],
            },
        ),
        types.Tool(
            name="delete_context",
            description="Delete a context and all its data. Use with caution as this cannot be undone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of the context to delete",
                    }
                },
                "required": ["context_name"],
            },
        ),
        # Document Ingestion Tools
        types.Tool(
            name="extract_url",
            description="Extract and index documentation from a website URL. Claude can autonomously scrape documentation sites to help with development.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to store the documentation",
                    },
                    "url": {
                        "type": "string",
                        "description": "Website URL to scrape (e.g., https://docs.python.org, https://ratatui.rs)",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum number of pages to crawl",
                        "default": 50,
                    },
                },
                "required": ["context_name", "url"],
            },
        ),
        types.Tool(
            name="extract_file",
            description="Extract and index content from a local file. Supports text, markdown, and reStructuredText files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to store the content",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to local file (txt, md, rst supported)",
                    },
                },
                "required": ["context_name", "file_path"],
            },
        ),
        # Search and Retrieval Tools
        types.Tool(
            name="search_context",
            description="Search for content within a context with compact summaries. Returns LLM-generated summaries (~100 chars) instead of full content for faster responses. Use get_document to retrieve full content when needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to search",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text (e.g., 'table widget example', 'async function patterns')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "vector", "fulltext"],
                        "description": "Search mode - hybrid (recommended), vector (semantic), or fulltext (exact)",
                        "default": "hybrid",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["context_name", "query"],
            },
        ),
        types.Tool(
            name="get_document",
            description="Get the full raw content of a specific document with pagination support for Claude's 25k token limit. Use this when search_context returns summaries and you need the complete document text for detailed analysis or code examples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context containing the document",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "ID of the document to retrieve (from search results)",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Page number to retrieve (1-based). Default is 1.",
                        "default": 1,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of characters per page. Default is 25000 for Claude's context limit.",
                        "default": 25000,
                    },
                },
                "required": ["context_name", "doc_id"],
            },
        ),
        types.Tool(
            name="get_code_snippets",
            description="Get all code snippets from a specific document. This returns executable code examples with line numbers and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context containing the document",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "ID of the document (from search results)",
                    },
                },
                "required": ["context_name", "doc_id"],
            },
        ),
        types.Tool(
            name="get_code_snippet",
            description="Get a specific code snippet by ID. This returns ready-to-use code with language and metadata information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context containing the snippet",
                    },
                    "snippet_id": {
                        "type": "string",
                        "description": "ID of the code snippet to retrieve (from search results metadata)",
                    },
                },
                "required": ["context_name", "snippet_id"],
            },
        ),
        types.Tool(
            name="search_code",
            description="Search for code snippets within a context using code-optimized embeddings (voyage-code-3). This is specialized for finding code examples, functions, and programming patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to search",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text focused on code (e.g., 'function definition', 'error handling', 'async pattern')",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language filter (e.g., 'python', 'javascript', 'rust')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["context_name", "query"],
            },
        ),
        # Job Management Tools
        types.Tool(
            name="get_job_status",
            description="Check the status and progress of a document extraction job. Use this to monitor long-running extractions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "ID of the job to check (returned from extract_url or extract_file)",
                    }
                },
                "required": ["job_id"],
            },
        ),
        types.Tool(
            name="cancel_job",
            description="Cancel a running document extraction job. Use this to stop long-running or stuck extractions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "ID of the job to cancel",
                    }
                },
                "required": ["job_id"],
            },
        ),
        types.Tool(
            name="cleanup_completed_jobs",
            description="Clean up old completed or failed jobs to save database space. Use this for maintenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Remove jobs completed/failed more than this many days ago",
                        "default": 7,
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_active_jobs",
            description="Get all currently active/running jobs, optionally filtered by context. Use this to monitor ongoing extractions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "Optional context ID to filter jobs",
                    }
                },
                "required": [],
            },
        ),
        # Utility Tools
        types.Tool(
            name="list_documents",
            description="List documents in a context with pagination. Use this to see what content is available.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to list documents from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of documents to skip (for pagination)",
                        "default": 0,
                    },
                },
                "required": ["context_name"],
            },
        ),
        types.Tool(
            name="delete_documents",
            description="Delete specific documents from a context. Use this to clean up outdated or incorrect content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_name": {
                        "type": "string",
                        "description": "Name of context to delete documents from",
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to delete",
                    },
                },
                "required": ["context_name", "document_ids"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle MCP tool calls."""
    try:
        # Check if Context Server is healthy
        if not await client.health_check():
            return [
                types.TextContent(
                    type="text",
                    text="Error: Context Server is not reachable. Please ensure the Context Server is running at http://localhost:8000",
                )
            ]

        # Route tool calls to appropriate methods
        if name == "create_context":
            result = await tools.create_context(**arguments)
        elif name == "list_contexts":
            result = await tools.list_contexts()
        elif name == "get_context":
            result = await tools.get_context(**arguments)
        elif name == "delete_context":
            result = await tools.delete_context(**arguments)
        elif name == "extract_url":
            result = await tools.extract_url(**arguments)
        elif name == "extract_file":
            result = await tools.extract_file(**arguments)
        elif name == "search_context":
            result = await tools.search_context(**arguments)
        elif name == "get_document":
            result = await tools.get_document(**arguments)
        elif name == "get_code_snippets":
            result = await tools.get_code_snippets(**arguments)
        elif name == "get_code_snippet":
            result = await tools.get_code_snippet(**arguments)
        elif name == "search_code":
            result = await tools.search_code(**arguments)
        elif name == "get_job_status":
            result = await tools.get_job_status(**arguments)
        elif name == "cancel_job":
            result = await tools.cancel_job(**arguments)
        elif name == "cleanup_completed_jobs":
            result = await tools.cleanup_completed_jobs(**arguments)
        elif name == "get_active_jobs":
            result = await tools.get_active_jobs(**arguments)
        elif name == "list_documents":
            result = await tools.list_documents(**arguments)
        elif name == "delete_documents":
            result = await tools.delete_documents(**arguments)
        else:
            return [
                types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")
            ]

        # Format successful response
        import json

        response_text = json.dumps(result, indent=2, ensure_ascii=False)

        return [types.TextContent(type="text", text=response_text)]

    except ContextServerError as e:
        # Handle Context Server specific errors
        error_message = f"Context Server Error: {e.message}"
        if e.status_code:
            error_message += f" (HTTP {e.status_code})"

        return [types.TextContent(type="text", text=error_message)]

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return [
            types.TextContent(
                type="text", text=f"Error: Tool execution failed - {str(e)}"
            )
        ]


async def main():
    """Main entry point for the MCP server."""
    global config, client, tools

    # Initialize configuration
    config = Config()
    logger.info(
        f"Starting MCP server with Context Server at {config.context_server_url}"
    )

    # Initialize client and tools
    client = ContextServerClient(config)
    tools = ContextServerTools(client)

    # Check initial connection
    if await client.health_check():
        logger.info("Successfully connected to Context Server")
    else:
        logger.warning(
            "Context Server not reachable - tools will check connection before each call"
        )

    # Run the MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=config.mcp_server_name,
                server_version=config.mcp_server_version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli_main():
    """Synchronous entry point for script generation."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
