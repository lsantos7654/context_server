# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

**CRITICAL: Always activate the virtual environment first:**
```bash
source .venv/bin/activate
```

### Development Setup
```bash
make init              # Initialize development environment
make test              # Run tests with coverage
make test-watch        # Run tests in watch mode during development
make format            # Format code with black and isort
make lint              # Run linting (flake8, mypy, bandit)
make quality-check     # Run format, lint, and test
```

### Context Server Operations
```bash
# Server management
ctx server up          # Start Docker services (PostgreSQL + API)
ctx server down        # Stop all services
ctx server restart     # Restart after code changes
ctx server logs        # View logs (essential for debugging)
ctx server status      # Check service health

# Context management  
ctx context create my-docs              # Create new context
ctx context list                       # List all contexts
ctx context delete my-docs             # Delete context

# Document extraction and search
ctx docs extract https://example.com my-docs    # Extract from URL
ctx docs extract ./path/to/files my-docs        # Extract from local files
ctx search query "search term" my-docs          # Search within context
```

## Architecture Overview

### Core Components
- **API Layer** (`context_server/api/`): FastAPI REST endpoints
- **Core Business Logic** (`context_server/core/`): Document processing, embeddings, storage
- **CLI Interface** (`context_server/cli/`): Command-line tools
- **MCP Server** (`context_server/mcp_server/`): Model Context Protocol integration

### Key Processing Pipeline
1. **Extraction**: crawl4ai processes URLs/files into markdown
2. **Three-Document Creation**: 
   - Original: Raw parsed markdown
   - Code Snippets: Extracted code blocks with metadata
   - Cleaned Text: Markdown with code snippet placeholders
3. **Chunking**: LangChain RecursiveCharacterTextSplitter (1000 chars for text, 700 for code)
4. **Embeddings**: 
   - Documents: OpenAI text-embedding-3-large (3072 dims)
   - Code: Voyage AI voyage-code-3 (2048 dims)
5. **Storage**: PostgreSQL with pgvector halfvec support

### Database Schema
```sql
contexts (id, name, description, embedding_model)
documents (id, context_id, url, title, content, document_type, metadata)
chunks (id, document_id, content, embedding halfvec(3072), summary, metadata)
code_snippets (id, document_id, content, language, embedding halfvec(2048), metadata)
```

## Development Workflow

### Essential Setup (Once)
```bash
make init
source .venv/bin/activate
```

### Daily Development
```bash
# 1. ALWAYS start with virtual environment
source .venv/bin/activate

# 2. Start services
ctx server up

# 3. Monitor during development
make test-watch
ctx server logs

# 4. After code changes
ctx server restart

# 5. Before committing
make quality-check
```

## Troubleshooting

### Common Issues
1. **Command not found**: Activate virtual environment with `source .venv/bin/activate`
2. **Import errors**: Check `ctx server logs` for Python import issues
3. **Connection errors**: Ensure services are running with `ctx server up`
4. **Database issues**: Check PostgreSQL container status

### Debug Steps
```bash
source .venv/bin/activate
ctx server logs              # Check for errors
ctx server restart           # Apply code changes
ctx context list             # Verify database connection
```

## Key Files and Architecture

### Core Processing (`context_server/core/`)
- `processing.py`: Document processing pipeline with three-document approach
- `chunking.py`: LangChain RecursiveCharacterTextSplitter integration
- `embeddings.py`: OpenAI + Voyage AI embedding services
- `storage.py`: PostgreSQL operations with pgvector
- `crawl4ai_extraction.py`: Web scraping and content extraction

### API Endpoints (`context_server/api/`)
- `main.py`: FastAPI application with lifespan management
- `contexts.py`: Context CRUD operations
- `documents.py`: Document management and extraction
- `search.py`: Search endpoints (vector, fulltext, hybrid)

### CLI Commands (`context_server/cli/`)
- `main.py`: CLI entry point with command routing
- `commands/`: Individual command implementations (server, context, docs, search)

## Code Snippet Placeholders

In cleaned markdown documents, code blocks are replaced with metadata:
```markdown
[CODE_SNIPPET: language=python, size=245_chars, summary="Function description", snippet_id=uuid-here]
```

## Testing

- Tests are in `tests/` directory
- Use `pytest` with asyncio support
- Run with `make test` or `make test-watch`
- Coverage reports generated with `make test`

## Environment Variables

Set in `.env` file:
- `OPENAI_API_KEY`: Required for document embeddings
- `VOYAGE_API_KEY`: Required for code embeddings
- `DATABASE_URL`: PostgreSQL connection string