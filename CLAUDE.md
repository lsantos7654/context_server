# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Server is a modern CLI-based documentation RAG system with FastAPI backend, PostgreSQL + pgvector storage, and semantic search capabilities. It provides MCP (Model Context Protocol) tools for Claude integration and implements a three-document processing pipeline with dual embedding models.

## Key Architecture Components

### Three-Document Processing Pipeline
The system processes sources into three distinct document types:
1. **Original**: Raw parsed markdown content
2. **Code Snippets**: Extracted code blocks with voyage-code-3 embeddings  
3. **Cleaned Markdown**: Text with code snippet placeholders for improved search

### Dual Embedding Strategy
- **Documents**: OpenAI text-embedding-3-large (3072 dims)
- **Code**: Voyage AI voyage-code-3 (2048 dims)
- **Storage**: PostgreSQL with pgvector halfvec support

### Core Services
- **Chunking**: LangChain RecursiveCharacterTextSplitter (context_server/core/chunking.py)
- **Processing**: Three-document pipeline (context_server/core/processing.py)
- **Embeddings**: Dual service implementation (context_server/core/embeddings.py)
- **MCP Tools**: Claude integration layer (context_server/mcp_server/tools.py)

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
make init
source .venv/bin/activate

# Alternative manual setup
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,test]"
```

### Docker Services
```bash
make up            # Start PostgreSQL + API server
make down          # Stop services
make restart       # Restart services
make logs          # View API logs
make status        # Check server health
```

### Testing and Quality
```bash
make test          # Run tests with coverage
make test-watch    # Run tests in watch mode
make format        # Format with black and isort
make lint          # Run flake8, mypy, bandit
make quality-check # Run format, lint, and test
```

### CLI Usage
```bash
ctx --help                                    # Show all commands
ctx server up                                # Start services
ctx context create my-docs                   # Create context
ctx docs extract https://docs.rust-lang.org rust-docs  # Extract docs
ctx search query "async functions" my-docs   # Search context
```

## Project Structure

```
context_server/
├── cli/           # CLI commands and interface (Click-based)
├── api/           # FastAPI REST endpoints
├── core/          # Core business logic
│   ├── chunking.py          # LangChain text splitting
│   ├── embeddings.py        # OpenAI + Voyage AI services  
│   ├── processing.py        # Three-document pipeline
│   ├── storage.py           # PostgreSQL + pgvector
│   └── crawl4ai_extraction.py  # Web scraping
├── mcp_server/    # MCP protocol implementation
└── tests/         # Test suite with pytest
```

## Important Implementation Details

### Database Schema
Uses PostgreSQL with pgvector halfvec support for efficient vector storage. Key tables:
- `contexts`: Context metadata with embedding model configuration
- `documents`: Three document types (original, code_snippets, cleaned_markdown)  
- `chunks`: Text chunks with halfvec(3072) embeddings
- `code_snippets`: Code blocks with halfvec(2048) embeddings

### Code Snippet Placeholders
In cleaned markdown documents, code blocks are replaced with structured metadata:
```markdown
[CODE_SNIPPET: language=python, size=245_chars, summary="Function description", snippet_id=uuid-here]
```

### MCP Integration
Provides comprehensive MCP tools for Claude including:
- Context management (create, list, delete)
- Document extraction (URL and file-based)
- Dual search (document and code search)
- Paginated document retrieval (handles Claude's 25k token limit)

## Testing Strategy

The project uses pytest with async support and multiple test categories:
- `pytest -m unit` - Unit tests
- `pytest -m integration` - Integration tests  
- `pytest -m slow` - Slow/comprehensive tests
- `pytest --cov=context_server --cov-report=html` - Coverage reports

## Environment Variables

Required in `.env` file:
- `OPENAI_API_KEY` - For text-embedding-3-large
- `VOYAGE_API_KEY` - For voyage-code-3 code embeddings
- `DATABASE_URL` - PostgreSQL connection string

## Current Enhancement Phase

The project is actively enhancing the core architecture with:
- Modern chunking strategies using LangChain
- Three-document processing pipeline
- Advanced embedding models with dual strategies
- MCP document pagination for large documents
- Enhanced search with separate code endpoints

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