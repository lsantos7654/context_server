# Context Server MCP Integration

A Model Context Protocol (MCP) server that enables Claude to autonomously manage documentation contexts, scrape websites, and retrieve code snippets through the Context Server API.

## Overview

This MCP server provides Claude with 17 essential tools to:
- **Create and manage contexts** for organizing documentation
- **Extract documentation** from websites and files with real-time job tracking
- **Monitor and control background jobs** with status checks and cancellation
- **Search and retrieve content** with hybrid vector/fulltext search
- **Access code snippets** with metadata and line numbers
- **Manage documents** with listing and deletion capabilities

## Quick Start

### 1. Prerequisites

- Context Server running at `http://localhost:8000`
- Python 3.12+ with the Context Server environment activated

### 2. Install Dependencies

```bash
# Activate the Context Server environment
source .venv/bin/activate

# Install MCP dependencies
pip install mcp>=1.0.0
```

### 3. Start the MCP Server

```bash
# Option 1: Using the script entry point
context-server-mcp

# Option 2: Using the startup script
python scripts/start_mcp_server.py

# Option 3: Direct module execution
python -m context_server.mcp_server.main
```

### 4. Configure Claude

Add the MCP server to your Claude configuration:

```json
{
  "mcpServers": {
    "context-server": {
      "command": "python",
      "args": ["/path/to/context_server/scripts/start_mcp_server.py"],
      "env": {}
    }
  }
}
```

## Available Tools

### Context Management

#### `create_context(name, description?, embedding_model?)`
Create a new documentation context.

```typescript
await create_context("ratatui-docs", "Ratatui terminal UI documentation")
```

#### `list_contexts()`
List all available contexts with metadata.

#### `get_context(context_name)`
Get detailed information about a specific context.

#### `delete_context(context_name)`
Delete a context and all its data (use with caution).

### Document Ingestion

#### `extract_url(context_name, url, max_pages?)`
Extract and index documentation from a website.

```typescript
await extract_url("ratatui-docs", "https://ratatui.rs/", 50)
```

#### `extract_file(context_name, file_path)`
Extract content from local files (PDF, txt, md, rst supported).

```typescript
await extract_file("my-docs", "/path/to/document.pdf")
```

### Job Management

#### `get_job_status(job_id)`
Check the status and progress of a document extraction job.

```typescript
const status = await get_job_status("job_20241206_143021_docs")
// Returns: { status: "processing", progress: 0.65, metadata: { phase: "chunking_and_embedding", processed_pages: 13, total_pages: 20 } }
```

#### `cancel_job(job_id)`
Cancel a running document extraction job.

```typescript
await cancel_job("job_20241206_143021_docs")
```

#### `get_active_jobs(context_id?)`
Get all currently active/running jobs, optionally filtered by context.

```typescript
const activeJobs = await get_active_jobs()
// or filter by context
const contextJobs = await get_active_jobs("ctx_abc123")
```

#### `cleanup_completed_jobs(days?)`
Clean up old completed or failed jobs (default: 7 days).

```typescript
await cleanup_completed_jobs(30) // Clean jobs older than 30 days
```

### Search and Retrieval

#### `search_context(context_name, query, mode?, limit?)`
Search for content within a context.

```typescript
// Hybrid search (recommended)
await search_context("ratatui-docs", "table widget examples", "hybrid", 5)

// Semantic search
await search_context("ratatui-docs", "layout containers", "vector", 10)

// Exact text search
await search_context("ratatui-docs", "pub fn new()", "fulltext", 3)
```

#### `get_document(context_name, doc_id)`
Get the full raw content of a specific document.

#### `get_code_snippets(context_name, doc_id)`
Get all code snippets from a document with metadata.

#### `get_code_snippet(context_name, snippet_id)`
Get a specific code snippet by ID.

### Utilities

#### `list_documents(context_name, limit?, offset?)`
List documents in a context with pagination.

#### `delete_documents(context_name, document_ids)`
Delete specific documents from a context.

## Usage Examples

### Complete Workflow: Building a Ratatui Application

```typescript
// 1. Create a context for Ratatui documentation
await create_context("ratatui-docs", "Ratatui terminal UI framework documentation")

// 2. Extract the official documentation (returns job_id for tracking)
const extractResult = await extract_url("ratatui-docs", "https://ratatui.rs/", 50)
const jobId = extractResult.job_id

// 3. Monitor extraction progress
let status = await get_job_status(jobId)
while (status.status === "processing") {
  console.log(`Progress: ${Math.round(status.progress * 100)}% - ${status.metadata?.phase || 'processing'}`)
  await new Promise(resolve => setTimeout(resolve, 2000)) // Wait 2 seconds
  status = await get_job_status(jobId)
}

if (status.status === "completed") {
  console.log(`Extraction completed! Processed ${status.result_data?.documents_processed} documents`)
  
  // 4. Search for table widget examples
  const searchResults = await search_context(
    "ratatui-docs",
    "table widget implementation",
    "hybrid",
    5
  )

  // 5. Get specific code snippets
  const snippets = await get_code_snippets("ratatui-docs", searchResults.results[0].document_id)

  // 6. Retrieve a specific code snippet for implementation
  const tableCode = await get_code_snippet("ratatui-docs", snippets.snippets[0].id)
} else {
  console.log(`Extraction failed: ${status.error_message}`)
}
```

### Managing Documentation and Jobs

```typescript
// List all available contexts
const contexts = await list_contexts()

// Get context details
const contextInfo = await get_context("ratatui-docs")

// List documents in a context
const documents = await list_documents("ratatui-docs", 20, 0)

// Clean up outdated documents
await delete_documents("ratatui-docs", ["doc-id-1", "doc-id-2"])

// Monitor all active extraction jobs
const activeJobs = await get_active_jobs()
console.log(`${activeJobs.total} jobs currently running`)

// Cancel a stuck job if needed
if (activeJobs.active_jobs.some(job => job.status === "processing" && /* job is stuck */)) {
  await cancel_job("stuck-job-id")
}

// Clean up old completed jobs (weekly maintenance)
await cleanup_completed_jobs(7)
```

## Architecture

### Components

- **`main.py`** - MCP server entry point with tool registration
- **`tools.py`** - Implementation of all 17 MCP tools
- **`client.py`** - HTTP client for Context Server API communication
- **`config.py`** - Configuration management

### Design Principles

1. **Lean and Simple** - Direct API mapping without unnecessary complexity
2. **Local-First** - Optimized for local Context Server deployment
3. **Error Resilient** - Graceful handling of API errors with clear messages
4. **Claude-Optimized** - Tool descriptions and responses designed for Claude

### No Caching

This MCP server doesn't implement caching because:
- All communication is local (fast response times)
- Context Server handles its own caching
- Simplifies the architecture
- Reduces memory usage

## Error Handling

The MCP server provides detailed error messages for common issues:

```
Context Server Error: Context 'my-docs' not found (HTTP 404)
Context Server Error: Cannot connect to Context Server at http://localhost:8000
Context Server Error: Context 'my-docs' already exists
```

## Configuration

### Environment Variables

- `CONTEXT_SERVER_URL` - Context Server URL (default: `http://localhost:8000`)
- `MCP_LOG_LEVEL` - Logging level (default: `INFO`)

### Default Settings

```python
DEFAULT_CONFIG = {
    "context_server_url": "http://localhost:8000",
    "mcp_server_name": "context-server",
    "mcp_server_version": "0.1.0",
    "log_level": logging.INFO,
    "request_timeout": 30.0,
}
```

## Troubleshooting

### Common Issues

#### "Cannot connect to Context Server"
- Ensure Context Server is running: `ctx server status`
- Check the URL: `curl http://localhost:8000/health`
- Verify network connectivity

#### "Context not found"
- List available contexts: `await list_contexts()`
- Check context name spelling
- Create the context if it doesn't exist

#### "Tool execution failed"
- Check Context Server logs: `ctx server logs`
- Verify API endpoint availability
- Ensure proper authentication if required

### Debug Mode

For detailed logging, set the log level to DEBUG:

```python
from context_server.mcp_server.config import Config

config = Config(log_level=logging.DEBUG)
```

## Future Enhancements

### Phase 2 (When Backend Supports)
- Git repository extraction tool
- Context merging capabilities
- Advanced document filtering

### Phase 3 (Intelligence Features)
- Content gap detection
- Related component suggestions
- Implementation pattern recognition

### Phase 4 (Performance)
- Response caching if needed
- Batch operations
- Connection pooling

## Development

### Testing the MCP Server

```bash
# Start Context Server
ctx server up

# Start MCP server in another terminal
python scripts/start_mcp_server.py

# Test with Claude or MCP client
```

### Adding New Tools

1. Implement the tool method in `ContextServerTools`
2. Add the tool definition in `handle_list_tools()`
3. Route the tool call in `handle_call_tool()`
4. Update documentation

## Integration with Claude Code

This MCP server enables Claude to:

1. **Autonomous Documentation Management** - Create contexts and extract documentation without user setup
2. **Code Discovery** - Find and retrieve executable code examples
3. **Implementation Assistance** - Access specific code snippets with metadata
4. **Documentation Navigation** - Search and browse large documentation sets efficiently

The result is a seamless experience where Claude can help developers build applications by autonomously managing and accessing documentation resources.
