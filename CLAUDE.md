# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Server is a modern, intelligent documentation RAG system with FastAPI backend, PostgreSQL + pgvector storage, and semantic search capabilities. It provides native MCP (Model Context Protocol) support for seamless Claude integration.

**Key Features:**
- Three-document pipeline: original content, code snippets (voyage-code-3), cleaned text (text-embedding-3-large)
- Hybrid semantic search with PostgreSQL + pgvector + halfvec optimization
- Intelligent web crawling with crawl4ai
- Async job system with real-time progress tracking
- Full Docker containerization

## Development Commands

### Environment Setup
```bash
# Using Makefile (recommended)
make init              # Initialize development environment
make up                # Start PostgreSQL + API in Docker
make down              # Stop services
make restart           # Restart services

# Using CLI directly
uv sync --extra dev    # Install dependencies
ctx server up          # Start services
ctx server down        # Stop services
ctx server status      # Check health
```

### Testing & Quality
```bash
make test              # Run tests with coverage
make test-watch        # Run tests in watch mode
make format            # Format with black + isort
make lint              # Run flake8, mypy, bandit
make quality-check     # Run format + lint + test
```

### CLI Operations
```bash
# Context management
ctx context create my-docs --description "Documentation context"
ctx context list
ctx context delete old-context

# Document extraction
ctx extract https://docs.python.org/ my-docs --max-pages 50
ctx extract ./local-files/ my-docs --source-type local

# Search operations
ctx search query "async functions" my-docs --limit 5
ctx search code "impl Error" my-docs --language rust

# Server management
ctx server logs        # View API logs
ctx server shell       # Connect to PostgreSQL
```

## Architecture Overview

### Core Components

**Processing Pipeline** (`context_server/core/pipeline.py`):
- Three-document processing: original → code snippets → cleaned content
- Parallel document processing with concurrency control
- Real-time job progress tracking

**Database Layer** (`context_server/core/database/`):
- PostgreSQL with pgvector extension for embeddings
- Models: contexts, documents, chunks, code_snippets, jobs
- Search manager with hybrid (semantic + full-text) search

**API Layer** (`context_server/api/`):
- FastAPI with async support
- Routers: contexts, documents, search, jobs
- Health checks and global exception handling

**MCP Server** (`context_server/mcp_server/`):
- Native Model Context Protocol integration
- Tools for context management, document extraction, search
- Optimized for Claude integration

### Key Services

**Embeddings** (`context_server/core/services/embeddings/`):
- OpenAI text-embedding-3-large for documents (3072 dimensions)
- Voyage AI voyage-code-3 for code snippets (2048 dimensions)

**Extraction** (`context_server/core/services/extraction/`):
- Crawl4ai for intelligent web scraping
- URL discovery and content cleaning
- Multi-page extraction with parallel processing

**Text Processing** (`context_server/core/text/`):
- LangChain RecursiveCharacterTextSplitter for chunking
- Code-aware chunking with placeholder system
- Text cleaning and preprocessing

## Database Schema

**Key Tables:**
- `contexts`: Documentation namespaces with embedding model config
- `documents`: Original content with metadata and source URLs
- `chunks`: Text chunks with embeddings and AI-generated summaries
- `code_snippets`: Extracted code with voyage-code-3 embeddings
- `jobs`: Async processing jobs with progress tracking

**Extensions:**
- `pgvector` with `halfvec` optimization for embeddings
- Full-text search capabilities

## Development Patterns

### Code Snippet Processing
The system uses a three-stage approach:
1. Extract code snippets during processing
2. Generate voyage-code-3 embeddings
3. Replace original code with placeholders containing real UUIDs

### Async Job Pattern
Long-running operations (URL extraction) use async jobs:
- Job creation returns `job_id`
- Progress tracking with metadata
- Status endpoints for monitoring

### Search Modes
- `hybrid`: Semantic + keyword search (recommended)
- `vector`: Pure semantic search
- `fulltext`: Keyword-only search

### Error Handling
- Structured exceptions with status codes
- Global exception handler in FastAPI
- Graceful degradation when services fail

## Configuration

**Environment Variables:**
```bash
OPENAI_API_KEY=your_openai_key
VOYAGE_API_KEY=your_voyage_key
DATABASE_URL=postgresql://context_user:context_password@localhost:5432/context_server
CONTEXT_SERVER_HOST=localhost
CONTEXT_SERVER_PORT=8000
```

**Default Chunk Sizes:**
- Text chunks: 4000 chars with 800 overlap
- Code chunks: 700 chars with 150 overlap

## Testing

Run the full test suite:
```bash
make test
```

Test markers available:
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow-running tests

## Docker Services

**Services defined in docker-compose.yml:**
- `postgres`: PostgreSQL 15 with pgvector extension
- `api`: Context Server API (built from Dockerfile)

**Ports:**
- API: http://localhost:8000
- PostgreSQL: localhost:5432
- API Docs: http://localhost:8000/docs

## Common Issues

**Database Connection**: Ensure PostgreSQL is running via `make up`
**API Keys**: Verify OPENAI_API_KEY and VOYAGE_API_KEY in .env file
**Port Conflicts**: Check if ports 8000 or 5432 are already in use
**Memory**: Large document processing may require sufficient RAM

## File Structure Notes

- `context_server/cli/`: Click-based CLI with rich output formatting
- `context_server/api/`: FastAPI routers and main application
- `context_server/core/`: Core business logic and services
- `context_server/mcp_server/`: MCP protocol implementation
- `context_server/models/`: Pydantic models for API and domain objects
- `scripts/`: Utility scripts for development

## Context Server User Guide

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

### 8. Context Server Workflow Patterns

#### Discovery-First Development
```
1. Check existing contexts before creating new ones
2. Search current contexts for gaps
3. Extract only missing documentation sections
4. Monitor long-running jobs
5. Maintain context hygiene over time
```

#### Progressive Information Retrieval
```
search_context (summaries) → get_chunk (targeted details) → get_document (comprehensive context)
```

#### When to Use Each Method
- **Chunks**: Quick reference, API details, specific code examples
- **Documents**: Complex concepts, tutorials, comprehensive understanding
- **Multiple chunks**: Comparing approaches, scanning multiple solutions

#### Context Management Philosophy
**Conservative approach for sustainable development sessions:**
- Start with summaries to scan and identify relevance
- Use chunks for targeted information without context flooding
- Retrieve full documents only when chunks are insufficient
- Maintain context window space for iterative development

#### Documentation Gap Workflow
```
User asks about missing concept
↓
Search existing contexts first
↓
If insufficient → Identify specific missing sections
↓
Extract targeted documentation (not entire sites)
↓
Re-search with enhanced context
↓
Provide complete answer
```

#### Maintenance Patterns
- **Regular cleanup**: Remove completed jobs and outdated contexts
- **Targeted updates**: Re-extract specific sections, not entire documentation sites
- **Context focus**: One context per framework/library for clarity

#### Anti-Patterns
❌ **Extracting entire sites** - Use targeted extraction
❌ **Creating duplicate contexts** - Enhance existing ones
❌ **Ignoring job monitoring** - Large extractions need oversight
❌ **Skipping context discovery** - Always check what exists first
❌ **Single search terms** - Try variations and synonyms
