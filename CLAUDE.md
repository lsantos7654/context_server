# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Server is a modern RAG (Retrieval-Augmented Generation) system that provides document extraction, semantic search, and MCP (Model Context Protocol) integration. The system is built with FastAPI, PostgreSQL + pgvector, and uses multiple embedding models for optimal search performance.

**Key Architecture Components:**
- **FastAPI REST API** (`context_server/api/`) - Core API endpoints for document management and search
- **CLI Interface** (`context_server/cli/`) - Rich command-line interface using Click and Rich
- **MCP Server** (`context_server/mcp_server/`) - Model Context Protocol integration for Claude
- **Core Processing** (`context_server/core/`) - Document processing, embeddings, and storage logic

## Essential Commands

### Development Environment
```bash
# Initial setup
make init                    # Create venv, install dependencies with uv
source .venv/bin/activate   # Activate virtual environment

# Docker services  
make up                     # Start PostgreSQL + API server
make down                   # Stop services
make restart                # Restart after code changes
make logs                   # View API server logs
make status                 # Check server health

# Testing and quality
make test                   # Run pytest with coverage
make test-watch            # Run tests in watch mode
make format                # Format with black + isort
make lint                  # Run flake8, mypy, bandit
make quality-check         # Full pipeline: format + lint + test
```

### CLI Operations
```bash
# Context management
ctx context create my-docs              # Create new context
ctx context list                        # List all contexts
ctx context delete my-docs             # Delete context and data

# Document extraction
ctx extract https://docs.python.org my-docs     # Extract from URL
ctx extract ./local-dir my-docs --source-type local  # Extract local directory

# Search and retrieval
ctx search query "async patterns" my-docs       # Search with summaries
ctx get document my-docs doc-id-123            # Get full document content
ctx search code "error handling" my-docs       # Search code snippets

# Job monitoring
ctx job status job-id-123              # Check extraction job status
ctx job cancel job-id-123              # Cancel running job
```

## Code Architecture

### Three-Document Processing Pipeline
The system processes each source into three distinct document types:
1. **Original** - Raw parsed markdown content  
2. **Code Snippets** - Extracted code blocks with language metadata
3. **Cleaned Markdown** - Text content with code snippet placeholders

### Dual Embedding Strategy
- **Documents**: OpenAI `text-embedding-3-large` (3072 dimensions) for semantic text search
- **Code**: Voyage AI `voyage-code-3` (2048 dimensions) for code-optimized search

### Database Schema (PostgreSQL + pgvector)
```sql
-- Core tables using halfvec for memory efficiency
contexts (id, name, description, embedding_model)
documents (id, context_id, url, title, content, document_type, metadata)
chunks (id, document_id, content, embedding halfvec(3072), summary, metadata)
code_snippets (id, document_id, content, language, embedding halfvec(2048), metadata)
jobs (id, type, status, progress, metadata)
```

### MCP Integration
The MCP server (`context_server/mcp_server/main.py`) provides 15 tools for Claude integration:
- Context management: create, list, get, delete contexts
- Document ingestion: extract_url, extract_file, extract_local_directory  
- Search & retrieval: search_context, search_code, get_document, get_code_snippet, get_chunk
- Job management: get_job_status, cancel_job, cleanup_completed_jobs, get_active_jobs

## Key Configuration

### Environment Variables
Required in `.env`:
```bash
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
DATABASE_URL=postgresql://context_user:password@localhost:5432/context_server
```

### Text Processing Settings
- **Regular text chunking**: 1000 chars, 200 overlap (LangChain RecursiveCharacterTextSplitter)
- **Code chunking**: 700 chars, 150 overlap
- **Summary length**: 3-5 sentences (50-150 words)
- **MCP pagination**: 25,000 characters per page (Claude's token limit)

## Testing Strategy

```bash
# Test categories (configured in pyproject.toml)
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests  
pytest -m slow          # Slow/long-running tests

# Coverage reporting
pytest --cov=context_server --cov-report=html
```

## Common Development Patterns

### API Error Handling
All API endpoints use structured error responses with appropriate HTTP status codes. MCP tools wrap API calls with `ContextServerError` handling.

### Async Operations
Document extraction jobs run asynchronously with progress tracking. Use job management endpoints to monitor long-running operations.

### Search Optimization
- Use `search_context` with `format=compact` for MCP responses (returns summaries)
- Use `get_document` with pagination for full content retrieval
- Use `search_code` for code-specific queries with optimized embeddings

### Code Snippet Placeholders
In cleaned markdown documents, code blocks are replaced with structured metadata:
```markdown
[CODE_SNIPPET: language=python, size=245_chars, summary="Function to handle user authentication with JWT tokens", snippet_id=uuid-here]
```

## Important Implementation Notes

- Always check server health with `ctx server status` before operations
- Use `make restart` after modifying API code (auto-reload enabled)
- MCP server runs separately via `context-server-mcp` command
- Database migrations are handled automatically by SQLAlchemy
- File extraction supports common text formats: .md, .txt, .rst, .py, .js, .ts, etc.
- URL extraction uses crawl4ai with configurable page limits
- All embedding operations are batched for efficiency

## Project Status

The system is in active enhancement phase (v0.2.0 development) with focus on improved chunking, dual embeddings, and MCP pagination support. The core RAG functionality is stable and production-ready.

## Intelligent Documentation Gap Detection & Extraction

When you encounter missing information while helping users:

### 1. Detect Documentation Gaps
If search results are insufficient for specific concepts (e.g., "BFSDeepCrawlStrategy", "specific API methods"):

### 2. Intelligent Documentation Discovery
Instead of blindly increasing max-pages, use this systematic approach:

```bash
# 1. Fetch the site's sitemap to understand structure
WebFetch("https://docs.example.com/sitemap.xml", "Extract all documentation URLs and categorize by topic")

# 2. Identify the specific section needed
# From sitemap results, find relevant URLs like:
# - https://docs.crawl4ai.com/core/deep-crawling/
# - https://docs.fastapi.com/advanced/security/
# - https://docs.react.dev/reference/hooks/

# 3. Extract the specific missing documentation
mcp__context-server__extract_url(
    context_name="existing-context-name",
    url="https://docs.example.com/specific/missing/section/",
    max_pages=10  # Focused extraction, not blanket increase
)
```

### 3. Proactive Documentation Management
- **Always check**: Before saying "I can't find information about X"
- **Auto-discover**: Use sitemaps, documentation indexes, or API references
- **Targeted extraction**: Extract specific missing sections, not entire sites again
- **Update contexts**: Add new pages to existing contexts rather than creating duplicates

### 4. Context Optimization Workflow
```
User asks about missing concept
↓
Search existing contexts first
↓
If insufficient: Analyze what's missing specifically  
↓
Use WebFetch to find the right documentation section
↓
Extract targeted pages with extract_url
↓
Re-search with enhanced context
↓
Provide complete answer
```

### 5. Optimal Context Server Usage Patterns

#### Search-First Workflow
1. **search_context** or **search_code** - Get summaries and identify relevant items
2. **get_document** or **get_code_snippet** - Retrieve specific content using IDs from search
3. **get_chunk** - Get detailed chunk content when summaries aren't sufficient

#### Context Management Best Practices
- Create focused contexts per framework/library: `fastapi-docs`, `crawl4ai-docs`, `react-docs`
- Use descriptive names that indicate the scope and purpose
- Extract with targeted approach rather than blanket high page limits
- Re-extract specific sections when documentation is updated or gaps are found

#### Documentation Site Discovery
- Check for `/sitemap.xml` for comprehensive URL lists
- Look for `/api/`, `/reference/`, `/docs/` pattern documentation
- Search for "API Reference", "Documentation Index", or "Table of Contents" pages
- Use domain-specific knowledge (e.g., GitHub repos often have docs in `/docs/` folder)

### 6. Advanced Context Enhancement Techniques

#### Iterative Documentation Building
```bash
# Start with core documentation
mcp__context-server__create_context(name="framework-docs", description="Core framework documentation")
mcp__context-server__extract_url(context_name="framework-docs", url="https://docs.framework.com/", max_pages=20)

# Identify gaps through usage
mcp__context-server__search_context(context_name="framework-docs", query="specific missing concept")

# If gaps found, use WebFetch to find specific sections
WebFetch("https://docs.framework.com/sitemap.xml", "Find URLs for 'specific missing concept'")

# Extract targeted sections
mcp__context-server__extract_url(context_name="framework-docs", url="https://docs.framework.com/advanced/specific-concept/", max_pages=5)
```

#### Multi-Source Documentation Strategy
- Primary source: Official documentation
- Secondary sources: GitHub README files, API references
- Tertiary sources: Community wikis, tutorial sites
- Always prioritize official docs over community content

### 7. Troubleshooting Missing Information  
If you can't find specific information after initial search:

1. **Check extraction completeness**: Verify if the context has sufficient pages for the topic
2. **Try multiple search strategies**: Use different modes (vector, fulltext, hybrid) and synonyms
3. **Analyze sitemap structure**: Use WebFetch to understand the documentation organization
4. **Extract missing sections**: Add specific pages that contain the needed information
5. **Use get_document for full content**: When summaries don't provide enough detail
6. **Cross-reference sources**: Check if information exists elsewhere in the ecosystem

### 8. Context Server Anti-Patterns to Avoid
❌ **Don't use list_documents** - Overwhelming and not actionable
❌ **Don't extract entire sites blindly** - Use targeted approach
❌ **Don't create duplicate contexts** - Enhance existing ones
❌ **Don't ignore sitemaps** - They provide the roadmap to complete documentation
❌ **Don't rely on single search terms** - Try variations and synonyms