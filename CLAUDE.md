# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Server is a modern documentation RAG system with FastAPI backend, PostgreSQL + pgvector storage, and semantic search capabilities. Built specifically for Claude integration via MCP (Model Context Protocol).

**Core Architecture**: Three-document pipeline processing:
1. **Original Document**: Raw parsed markdown content
2. **Code Snippets**: Extracted code blocks with voyage-code-3 embeddings  
3. **Cleaned Text**: Text with code snippet placeholders for improved search

**Dual Embedding Strategy**: 
- OpenAI text-embedding-3-large for documents
- Voyage AI voyage-code-3 for code snippets

## Essential Commands

### Development Environment
```bash
# Install with uv package manager  
uv sync --extra dev

# Start services (PostgreSQL + API)
make up

# Development server (with reload)
make restart

# Environment setup 
make init

# Clean caches
make clean
```

### Testing & Quality
```bash
# Run tests with coverage
make test

# Format code (black + isort)
make format

# Run linting (flake8, mypy, bandit)
make lint

# Full quality check
make quality-check

# Watch mode for development
make test-watch
```

### Docker Operations
```bash
# Check server status
make status

# View API logs  
make logs

# Database shell
make db-shell

# Reset database (destroys all data)
make db-reset
```

### CLI Usage (ctx command)
```bash
# Context management
ctx context create my-docs --description "Documentation context"
ctx context list
ctx context delete old-context

# Document extraction
ctx extract https://docs.python.org/ my-docs --max-pages 50
ctx extract ./my-project/ local-docs --source-type local

# Search operations  
ctx search query "async functions" my-docs --limit 5
ctx search code "impl Error" my-docs --language rust
```

## Key Architecture Components

### Core Pipeline (`context_server/core/pipeline.py`)
- **DocumentProcessor**: Main orchestrator of three-document processing
- **CodeSnippetExtractor**: Extracts and processes code blocks
- Text chunking with LangChain RecursiveCharacterTextSplitter
- Concurrent embedding generation and summarization

### MCP Integration (`context_server/mcp_server/`)
- **tools.py**: Complete MCP tool implementations for Claude
- **client.py**: HTTP client for Context Server API
- Native Model Context Protocol support for seamless AI workflows

### Database Layer (`context_server/core/database/`)
- PostgreSQL with pgvector extension
- Async SQLAlchemy models
- Hybrid search (semantic + keyword) capabilities
- Optimized chunking and embedding storage

### API Layer (`context_server/api/`)
- FastAPI application with async support
- RESTful endpoints for contexts, documents, search
- Job system for async processing with progress tracking
- Error handling and validation

## Important Development Notes

### Database Schema
- Uses pgvector with halfvec optimization for performance
- Three main tables: documents, chunks, code_snippets  
- UUID-based relationships between documents and code snippets
- Embedding vectors stored as halfvec for memory efficiency

### Text Processing
- **Text chunks**: 4000 chars, 800 overlap (20% for context continuity)
- **Code chunks**: 700 chars, 150 overlap (optimized for code blocks)
- Code snippet placeholders link chunks to code for enhanced search

### API Configuration
- Server runs on localhost:8000 by default
- API docs available at /docs and /redoc
- Database URL: postgresql://context_user:context_password@localhost:5432/context_server

### Environment Variables
Required in `.env` file:
- `OPENAI_API_KEY`: For text embeddings and summarization
- `VOYAGE_API_KEY`: For code embeddings (voyage-code-3)
- `DATABASE_URL`: PostgreSQL connection string (optional, defaults to Docker setup)

### Testing
- Uses pytest with async support
- Test markers: unit, integration, slow
- Coverage reports generated in htmlcov/
- Factory-boy for test data generation

### Code Quality
- Black formatter (88 char line length)
- isort for import sorting  
- mypy for type checking (not strict, allows untyped defs)
- bandit for security scanning
- pre-commit hooks configured

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