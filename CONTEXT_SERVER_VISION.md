# Context Server: Modular Documentation RAG System with MCP Integration

## Project Overview

Context Server is a local, containerized system that extracts, processes, and indexes documentation into segregated vector databases, making them accessible through Claude Code via MCP. It provides a flexible API for managing multiple documentation contexts and enables real-time semantic search across your indexed content.

## Core Principles

1. **Local-first**: Runs entirely on your machine via Docker Compose
2. **Vector-based**: Uses PostgreSQL with pgvector for efficient semantic search
3. **API-centric**: FastAPI provides full control over data management
4. **No scheduling**: Manual control over when to crawl/update
5. **Multi-context**: Separate vector databases for different projects
6. **Developer-focused**: Built for technical users who want control

## Architecture Decision: PostgreSQL with pgvector

### Why pgvector is Better for Context Server

**Advantages of pgvector:**
1. **Context Isolation**: Each context = separate PostgreSQL schema for clean segregation
2. **SQL Power**: Rich metadata queries, filtering, and joins
3. **Simple Operations**: Standard CRUD operations for all management tasks
4. **Transactional Safety**: ACID guarantees for context merging/deletion
5. **Backup/Export**: Standard pg_dump/pg_restore for data management
6. **Hybrid Search**: Combine vector similarity with full-text search
7. **Familiar Tooling**: Leverage existing PostgreSQL ecosystem

**Why Not Graph Database:**
- Most documentation queries are simple semantic searches
- Complex graph traversal rarely needed in practice
- Vector similarity handles "related concepts" well
- SQL is more familiar and debuggable
- Easier context management and data operations

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Server                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌──────────────────┐         │
│  │   FastAPI       │         │   Extraction     │         │
│  │   Admin API     │────────▶│   Pipeline       │         │
│  └─────────────────┘         └──────────────────┘         │
│           │                           │                     │
│           │                           ▼                     │
│           │                   ┌──────────────────┐         │
│           │                   │  PostgreSQL      │         │
│           └──────────────────▶│   +pgvector      │         │
│                               └──────────────────┘         │
│                                       │                     │
│                               ┌───────┴────────┐           │
│                               │                │           │
│                        ┌──────▼─────┐   ┌─────▼──────┐    │
│                        │  Schema A  │   │ Schema B   │    │
│                        │   (Rust)   │   │ (Python)   │    │
│                        └────────────┘   └────────────┘    │
│                                                             │
│  ┌─────────────────┐                                       │
│  │   MCP Server    │◀──────────────────────────────────────│
│  └─────────────────┘                                       │
│           │                                                 │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  Claude Code  │
    └───────────────┘
```

## Core Components

### 1. FastAPI Admin API
```
Purpose: Primary interface for all data management
Endpoints:
  - Context management (create, list, delete, merge)
  - Document ingestion (URL, file upload, git repo)
  - Search and exploration
  - Raw data inspection
  - System logs and monitoring
```

### 2. Extraction Pipeline
```
Purpose: Consolidated extraction capabilities
Components:
  - smart_extract.py (consolidated intelligent extraction)
  - cleanup_markdown.py (content cleaning)
  - Unified processors from embed.py (merged and simplified)
```

### 3. PostgreSQL + pgvector Storage
```
Purpose: Vector database storage and retrieval
Features:
  - Multiple isolated contexts (schemas)
  - Vector similarity search
  - Metadata filtering with SQL
  - Full-text search combination
  - Export/import with pg_dump
```

### 4. MCP Server
```
Purpose: Bridge to Claude Code
Tools:
  - search(query, context, mode)
  - get_document(id, context)
  - list_contexts()
```

## API Design (FastAPI)

### Context Management

```python
# Create new context
POST /api/contexts
{
  "name": "rust-docs",
  "description": "Rust standard library documentation",
  "embedding_model": "text-embedding-3-small"  # optional
}

# List all contexts
GET /api/contexts
Response: [
  {
    "id": "ctx_123",
    "name": "rust-docs",
    "created_at": "2024-01-15T10:30:00Z",
    "document_count": 1543,
    "size_mb": 245.3,
    "last_updated": "2024-01-15T10:30:00Z"
  }
]

# Delete context
DELETE /api/contexts/{context_id}

# Merge contexts
POST /api/contexts/merge
{
  "source_contexts": ["ctx_123", "ctx_456"],
  "target_context": "ctx_789",
  "mode": "union"  # or "intersection"
}

# Export context
GET /api/contexts/{context_id}/export
Response: Binary data (pg_dump SQL file)

# Import context
POST /api/contexts/import
Body: multipart/form-data with SQL dump file
```

### Document Management

```python
# Add documents to context
POST /api/contexts/{context_id}/documents
{
  "source_type": "url" | "file" | "git",
  "source": "https://docs.rs/tokio/latest/tokio/",
  "options": {
    "max_depth": 3,
    "include_pattern": "*.html",
    "exclude_pattern": "*test*"
  }
}

# List documents in context
GET /api/contexts/{context_id}/documents
Response: {
  "documents": [
    {
      "id": "doc_123",
      "url": "https://docs.rs/tokio/latest/tokio/",
      "title": "tokio - Rust",
      "indexed_at": "2024-01-15T10:30:00Z",
      "chunks": 45,
      "metadata": {...}
    }
  ]
}

# Delete documents
DELETE /api/contexts/{context_id}/documents
{
  "document_ids": ["doc_123", "doc_456"]
}

# Get raw document
GET /api/contexts/{context_id}/documents/{doc_id}/raw
Response: Original markdown/text content
```

### Search and Query

```python
# Search within context
POST /api/contexts/{context_id}/search
{
  "query": "async trait implementation",
  "mode": "hybrid",  # vector, fulltext, hybrid
  "limit": 10,
  "include_raw": false
}

# Get system logs
GET /api/logs
{
  "level": "INFO",
  "since": "2024-01-15T10:00:00Z",
  "limit": 100
}

# Get processing status
GET /api/status
Response: {
  "active_jobs": [
    {
      "id": "job_123",
      "type": "extraction",
      "context": "rust-docs",
      "progress": 0.75,
      "status": "Processing page 750/1000"
    }
  ]
}
```

## Why No Streaming/WebSocket?

For this use case, traditional REST is sufficient:

1. **Document processing is batch-oriented**: You add a source and wait for completion
2. **Search results are finite**: Even with 50 results, response time is sub-second
3. **No real-time requirements**: Documentation doesn't change in real-time
4. **Simpler architecture**: REST is easier to test, debug, and integrate
5. **MCP handles the streaming**: If needed, MCP protocol can handle streaming to Claude

We can always add WebSocket support later if real-time features become necessary.

## Embedding Strategy for Code

### Recommended Approach: Hybrid Embeddings

```python
embedding_models = {
    "text": "text-embedding-3-small",  # General documentation
    "code": "codegen-350M-multi",      # Code snippets
    "hybrid": "instructor-xl"          # Handles both well
}
```

### Why Consider Multiple Models:
1. **Code-specific models** understand syntax, not just semantics
2. **Different chunk types** benefit from different embeddings
3. **Language-specific models** (e.g., CodeBERT for Python/JS)

### Practical Implementation:
- Start with OpenAI's text-embedding-3-small (good baseline)
- Add code-specific embeddings later if needed
- Store model version with embeddings for future migration

## Data Management Features

### Inspection Capabilities

```python
# View extraction logs
GET /api/contexts/{context_id}/logs

# Inspect raw extracted data
GET /api/contexts/{context_id}/documents/{doc_id}/debug
Response: {
  "raw_html": "...",
  "extracted_markdown": "...",
  "chunks": [...],
  "metadata": {...},
  "processing_time": 1.23
}

# Browse document structure
GET /api/contexts/{context_id}/documents?limit=50&offset=0
GET /api/contexts/{context_id}/documents/{doc_id}/chunks

# Export for analysis
GET /api/contexts/{context_id}/export/csv
GET /api/contexts/{context_id}/export/json
```

## MCP Tools (Python Implementation)

```python
# mcp_tools.py
from typing import List, Optional, Dict
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    context: str
    mode: str = "hybrid"
    limit: int = 10

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    metadata: Dict

async def search(params: SearchParams) -> List[SearchResult]:
    """Search documents in specified context"""
    # Implementation

class DocumentParams(BaseModel):
    id: str
    context: str

async def get_document(params: DocumentParams) -> Dict:
    """Retrieve full document by ID"""
    # Implementation

async def list_contexts() -> List[Dict]:
    """List available contexts"""
    # Implementation
```

## Phase 0: Repository Analysis and Reuse Plan

### What to Keep:
1. **Extraction Pipeline**
   - `smart_extract.py` - Core extraction logic
   - `utils/sitemap.py` - Sitemap parsing
   - `cleanup_markdown.py` - Markdown cleaning

2. **Processing Components**
   - PDF processing logic with table extraction
   - Different handlers for different file types
   - Git repository handling
   - Document cleaning and chunking

3. **Vector Database Components**
   - Embedding generation logic
   - PostgreSQL schema design
   - Vector similarity search

### What to Remove:
1. **UI Components**
   - `chat.py` - Streamlit interface
   - `graph.py` - Neo4j visualization
   - PyVis dependencies

2. **Unused Features**
   - Image processing (EasyOCR)
   - Tree-sitter code chunking
   - Multiple visualization libraries

3. **Keep for PDF Processing**
   - Table extraction capabilities
   - Multi-modal document handlers

### What to Refactor:
1. **embed.py** → Split into:
   - `processors/base.py` - Abstract processor interface
   - `processors/pdf.py` - PDF handling with table extraction
   - `processors/url.py` - Web extraction (primary use case)
   - `processors/text.py` - Plain text processing
   - `processors/git.py` - Repository processing

2. **Create New Structure**:
   ```
   context_server/
   ├── api/
   │   ├── main.py         # FastAPI app
   │   ├── contexts.py     # Context endpoints
   │   ├── documents.py    # Document endpoints
   │   └── models.py       # Pydantic models
   ├── core/
   │   ├── extraction/     # Existing extractors
   │   ├── processing/     # Document processors
   │   └── storage/        # PostgreSQL/pgvector wrapper
   ├── mcp/
   │   └── server.py       # MCP implementation
   ├── tests/
   │   ├── test_api.py     # API tests
   │   ├── test_processors.py # Processor tests
   │   └── fixtures/       # Test data
   ├── Makefile            # Development commands
   ├── CLAUDE.md           # Development guidelines
   ├── audit.md            # Code audit template
   └── docker-compose.yml
   ```

## Simplified Development Phases

### Phase 0: Massive Consolidation (Week 0)
- [ ] **Audit existing codebase** using audit.md template
- [ ] **Remove specialized extractors** (rustdoc_json_extractor.py, etc.)
- [ ] **Consolidate duplicate code** following CLAUDE.md principles
- [ ] **Delete overly specific implementations** that don't fit plugin architecture
- [ ] **Merge similar processors** into generic, configurable handlers
- [ ] **Extract common patterns** into reusable utilities
- [ ] **Simplify complex modules** by breaking them down
- [ ] **Create clean abstractions** for remaining functionality

### Phase 1: Foundation (Week 1)
- [ ] Analyze and document consolidated components
- [ ] Create new project structure
- [ ] Set up FastAPI with basic endpoints
- [ ] Refactor embed.py into modular processors
- [ ] Create context management with PostgreSQL/pgvector

### Phase 2: Core Features (Week 2)
- [ ] Implement document ingestion API
- [ ] Add search functionality
- [ ] Create data inspection endpoints
- [ ] Set up Docker Compose
- [ ] Add comprehensive logging

### Phase 3: MCP Integration (Week 3)
- [ ] Implement MCP server
- [ ] Create search/get/list tools
- [ ] Test with Claude Code
- [ ] Add export/import functionality
- [ ] Create basic documentation

### Phase 4: Polish & Deploy (Week 4)
- [ ] Error handling and validation
- [ ] Performance optimization
- [ ] Create example workflows
- [ ] Write API documentation
- [ ] Package for easy deployment

## Docker Compose Setup

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/context_server
      - LOG_LEVEL=INFO
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/logs
    depends_on:
      - postgres
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  mcp:
    build: .
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    command: python -m mcp.server

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=context_server
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
  logs:
```

## Makefile Integration

```makefile
# Makefile for Context Server

.PHONY: help up down logs test clean extract search

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

up: ## Start the server
	docker-compose up -d
	@echo "Server started at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

down: ## Stop the server
	docker-compose down

logs: ## Show server logs
	docker-compose logs -f api

test: ## Run tests
	docker-compose exec api pytest tests/

test-coverage: ## Run tests with coverage
	docker-compose exec api pytest tests/ --cov=context_server --cov-report=html

clean: ## Clean up containers and volumes
	docker-compose down -v
	docker system prune -f

db-shell: ## Connect to PostgreSQL shell
	docker-compose exec postgres psql -U user -d context_server

db-migrate: ## Run database migrations
	docker-compose exec api python -m alembic upgrade head

db-reset: ## Reset database (WARNING: destroys all data)
	docker-compose exec postgres psql -U user -d context_server -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"
	docker-compose exec api python -m alembic upgrade head

# Extraction commands
extract: ## Extract from URL (usage: make extract URL=https://example.com CONTEXT=my-context)
	@if [ -z "$(URL)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make extract URL=https://example.com CONTEXT=my-context"; \
		exit 1; \
	fi
	curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/documents \
		-H "Content-Type: application/json" \
		-d '{"source_type": "url", "source": "$(URL)"}'

extract-file: ## Extract from file (usage: make extract-file FILE=path/to/file.pdf CONTEXT=my-context)
	@if [ -z "$(FILE)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make extract-file FILE=path/to/file.pdf CONTEXT=my-context"; \
		exit 1; \
	fi
	curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/documents \
		-F "file=@$(FILE)" \
		-F "source_type=file"

# Search commands
search: ## Search in context (usage: make search QUERY="rust async" CONTEXT=my-context)
	@if [ -z "$(QUERY)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make search QUERY=\"rust async\" CONTEXT=my-context"; \
		exit 1; \
	fi
	curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/search \
		-H "Content-Type: application/json" \
		-d '{"query": "$(QUERY)", "mode": "hybrid", "limit": 5}' | jq .

# Context management
list-contexts: ## List all contexts
	curl -s http://localhost:8000/api/contexts | jq .

create-context: ## Create new context (usage: make create-context NAME=my-context DESC="My documentation")
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make create-context NAME=my-context DESC=\"My documentation\""; \
		exit 1; \
	fi
	curl -X POST http://localhost:8000/api/contexts \
		-H "Content-Type: application/json" \
		-d '{"name": "$(NAME)", "description": "$(DESC)"}' | jq .

delete-context: ## Delete context (usage: make delete-context CONTEXT=my-context)
	@if [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make delete-context CONTEXT=my-context"; \
		exit 1; \
	fi
	curl -X DELETE http://localhost:8000/api/contexts/$(CONTEXT)

# Development commands
dev: ## Start development server with hot reload
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

format: ## Format code
	docker-compose exec api black .
	docker-compose exec api isort .

lint: ## Run linting
	docker-compose exec api flake8 .
	docker-compose exec api mypy .

# Utility commands
status: ## Show server status
	curl -s http://localhost:8000/api/status | jq .

health: ## Check server health
	curl -s http://localhost:8000/health

export-context: ## Export context (usage: make export-context CONTEXT=my-context)
	@if [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make export-context CONTEXT=my-context"; \
		exit 1; \
	fi
	curl -O http://localhost:8000/api/contexts/$(CONTEXT)/export

# Examples
example-rust: ## Example: scrape Rust std docs
	make create-context NAME=rust-std DESC="Rust standard library"
	make extract URL=https://doc.rust-lang.org/std/ CONTEXT=rust-std
	make search QUERY="HashMap" CONTEXT=rust-std

example-fastapi: ## Example: scrape FastAPI docs
	make create-context NAME=fastapi DESC="FastAPI documentation"
	make extract URL=https://fastapi.tiangolo.com/ CONTEXT=fastapi
	make search QUERY="async dependency injection" CONTEXT=fastapi
```

## Usage Examples

```bash
# Create a new context for Rust documentation
curl -X POST http://localhost:8000/api/contexts \
  -H "Content-Type: application/json" \
  -d '{"name": "rust-std", "description": "Rust standard library"}'

# Add documentation to context
curl -X POST http://localhost:8000/api/contexts/ctx_123/documents \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "url",
    "source": "https://doc.rust-lang.org/std/"
  }'

# Search within context
curl -X POST http://localhost:8000/api/contexts/ctx_123/search \
  -H "Content-Type: application/json" \
  -d '{"query": "HashMap concurrent access", "mode": "hybrid"}'

# Export context for sharing
curl -O http://localhost:8000/api/contexts/ctx_123/export

# View processing logs
curl http://localhost:8000/api/logs?level=INFO&limit=50
```

## Primary Use Case: Documentation Scraping

**Core Workflow:**
1. **Input**: Provide URL to documentation site
2. **Extract**: Intelligently scrape the site content
3. **Clean**: Remove navigation, ads, format for embedding
4. **Embed**: Convert to vectors and store in PostgreSQL
5. **Search**: Query via API or MCP for relevant content

**Example Usage:**
```bash
# Create context for Rust documentation
make create-context NAME=rust-std DESC="Rust standard library"

# Extract documentation
make extract URL=https://doc.rust-lang.org/std/ CONTEXT=rust-std

# Search for specific topics
make search QUERY="HashMap concurrent access" CONTEXT=rust-std
```

## Success Criteria for POC

1. **Basic Functionality**
   - Can create/delete contexts
   - Can ingest documentation from URLs
   - Can search and retrieve results
   - MCP tools work with Claude Code

2. **Performance**
   - Search returns in < 500ms
   - Can handle 10k+ documents per context
   - Extraction processes 100+ pages/minute

3. **Developer Experience**
   - Clear API documentation
   - Helpful error messages
   - Easy to inspect data
   - Simple deployment via Makefile

## Future Considerations (Post-POC)

1. **Enhanced Extractors**
   - GitHub wiki support
   - Confluence/Notion integration
   - API specification parsing (OpenAPI)

2. **Advanced Features**
   - Incremental updates
   - Duplicate detection
   - Context versioning
   - Query analytics

3. **Performance**
   - Caching layer
   - Parallel processing
   - Background job queue

## Development Guidelines (See CLAUDE.md)

### Code Quality Standards
- Use Python built-in types instead of `typing` module
- Focus on scalable, reusable code patterns
- Avoid code duplication - refactor when patterns emerge
- Create abstract base classes for different processor types
- Use dependency injection for better testability

### Architecture Principles
- **Separation of Concerns**: Each module has a single responsibility
- **Plugin Architecture**: Easy to add new document processors
- **Configuration-driven**: Minimize hardcoded values
- **Error Handling**: Graceful degradation and helpful error messages
- **Observability**: Comprehensive logging and metrics

### File Type Handlers
```python
# Example processor architecture
class DocumentProcessor:
    def can_handle(self, file_type: str) -> bool: ...
    def process(self, content: bytes) -> ProcessedDocument: ...

# Specific implementations
class PDFProcessor(DocumentProcessor):  # With table extraction
class URLProcessor(DocumentProcessor):  # Primary use case
class TextProcessor(DocumentProcessor): # Plain text files
```

## Conclusion

Context Server provides a pragmatic, developer-controlled approach to building a personal documentation knowledge base. By focusing on API-first design, multiple contexts, and vector-based storage with PostgreSQL, it offers flexibility and power while remaining simple to deploy and use. The reuse of existing extraction components accelerates development while the move to FastAPI and removal of UI components simplifies the architecture for its intended use case with Claude Code.
