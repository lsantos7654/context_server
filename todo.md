# Context Server - Development Todo

## Current Status

### ✅ Phase 0: Consolidation (COMPLETED)

- [x] Audited existing codebase structure and identified components
- [x] Removed specialized extractors that don't fit plugin architecture
- [x] Consolidated duplicate code following CLAUDE.md principles
- [x] Merged similar processors into generic, configurable handlers
- [x] Extracted common patterns into reusable utilities
- [x] Simplified complex modules by breaking them down
- [x] Created clean abstractions for remaining functionality

### ✅ Extraction Pipeline (COMPLETED)

- [x] Smart extraction system with automatic method detection
- [x] Sitemap processing for bulk documentation extraction
- [x] High-quality content extraction using Docling
- [x] Markdown cleaning and formatting
- [x] Table detection and processing
- [x] Batch processing with caching for performance
- [x] `make extract` command with URL parameter
- [x] Successfully tested with:
  - ✅ https://textual.textualize.io/sitemap.xml (350+ pages)
  - ✅ https://textual.textualize.io/api/app/ (smart sitemap detection)
  - ⚠️ https://docs.rs/ratatouille/latest/ratatouille/ (complex sitemap edge case)

### 📊 What We Have

- **Sophisticated extraction system** with UnifiedDocumentExtractor
- **Plugin architecture** with ProcessorFactory
- **High-quality output** with proper markdown formatting
- **Smart detection** between sitemap and single page extraction
- **Efficient processing** with batching and caching
- **Clean file organization** in output/ directory
- **Pre-commit hooks** and development environment setup

## 🎯 Next Phase: Context Server API

### 🚧 Current Priority: Phase 1 - Foundation

- [ ] Analyze and document consolidated components
- [ ] Create new project structure for API layer
- [ ] Set up FastAPI with basic endpoints
- [ ] Refactor existing processors into modular API-ready components
- [ ] Create context management with PostgreSQL/pgvector

### 📋 Todo List

#### High Priority

- [ ] **Design FastAPI application structure**

  - [ ] Create `api/` directory with main.py, contexts.py, documents.py
  - [ ] Design Pydantic models for request/response
  - [ ] Set up basic health check and info endpoints

- [ ] **Database setup**

  - [ ] Set up PostgreSQL with pgvector extension
  - [ ] Create database schema for contexts and documents
  - [ ] Implement context isolation using schemas
  - [ ] Create migration system with Alembic

- [ ] **Context management endpoints**
  - [ ] POST /api/contexts (create context)
  - [ ] GET /api/contexts (list contexts)
  - [ ] DELETE /api/contexts/{id} (delete context)
  - [ ] GET /api/contexts/{id}/export (export context)

#### Medium Priority

- [ ] **Document ingestion API**

  - [ ] POST /api/contexts/{id}/documents (add documents)
  - [ ] GET /api/contexts/{id}/documents (list documents)
  - [ ] DELETE /api/contexts/{id}/documents (remove documents)
  - [ ] Integration with existing extraction pipeline

- [ ] **Search functionality**

  - [ ] Implement vector embedding generation
  - [ ] POST /api/contexts/{id}/search (search documents)
  - [ ] Support for hybrid search (vector + full-text)

- [ ] **Docker Compose setup**
  - [ ] Create docker-compose.yml for local development
  - [ ] PostgreSQL + pgvector service
  - [ ] FastAPI service
  - [ ] Volume mounts and environment configuration

#### Low Priority

- [ ] **Handle edge cases in extraction**

  - [ ] Improve complex sitemap handling (docs.rs case)
  - [ ] Add retry mechanisms for failed extractions
  - [ ] Rate limiting for respectful crawling

- [ ] **Enhanced Makefile commands**

  - [ ] `make up` - Start services with docker-compose
  - [ ] `make create-context NAME=name DESC="description"`
  - [ ] `make search QUERY="query" CONTEXT=context-name`

- [ ] **MCP Integration (Phase 3)**
  - [ ] Implement MCP server
  - [ ] Create search/get/list tools for Claude Code
  - [ ] Test integration with Claude Code

## 📁 Current Architecture

```
context_server/
├── src/
│   ├── core/
│   │   ├── extraction.py        # UnifiedDocumentExtractor
│   │   ├── processors.py        # ProcessorFactory + processors
│   │   ├── cleaning.py          # MarkdownCleaner
│   │   └── logging.py           # Structured logging
│   ├── utils/
│   │   ├── sitemap.py           # Sitemap processing
│   │   └── segment_tables.py    # Table extraction
│   ├── smart_extract.py         # CLI interface
│   └── core/cli.py              # Unified CLI (needs API integration)
├── output/                      # Extracted content (350+ files)
├── Makefile                     # Development commands + extract
├── CLAUDE.md                    # Development guidelines
└── todo.md                      # This file
```

## 🎯 Vision Alignment

We're building towards the Context Server Vision:

- **Local-first**: Docker Compose for local deployment
- **Vector-based**: PostgreSQL + pgvector for semantic search
- **API-centric**: FastAPI for full control over data management
- **Multi-context**: Separate schemas for different projects
- **Developer-focused**: Clean APIs and comprehensive tooling

## 📈 Progress Metrics

- **Phase 0**: 100% Complete ✅
- **Extraction Pipeline**: 95% Complete ✅ (edge cases remain)
- **Overall Progress**: ~30% towards full Context Server MVP
- **Next Milestone**: FastAPI foundation + database setup

---

_Last Updated: 2025-01-04_
_Current Focus: Planning Phase 1 - FastAPI Foundation_

alright so here is the next few things I want to add to this project

1. Documents returned sometimes are too big for the mcp server. Claude
   has a limit of 25,000 tokens. Can we implement a way to have pagination.
2. When we ingest a document can we have it split into 3 separate
   documents. (1) is the original parsed markdown file (already implemented)
   (2) would be just the code snippet (already implemented) (3) would be
   just the markdown file without code snippets. (needs to be implemented).

   markdown file with no code snippet implementation:
3. Wherever the code snippet was removed from I want you to replace it
   with some metadata. It will have a summary of what the code was, size of
   the code, language of the code, and most importantly the code snippet id
   that claude could use to get the full code snippet.
4. After processing this document, this is the document that I want you
   to use to chunk and then embed.
5. This should be the document retrieved in the cli when we query for
   items as well. This should happen naturally since this would be the │
   document that was used for chunking.

   Furthermore I want to replace our current embedding model. Instead of │
   using `text-embedding-3-small` instead I want to use │
   `text-embedding-3-large`. In order to do this we will need to store our │
   embeddings in `halfvec` │

````│
vector: Supports up to 2,000 dimensions, according to pgvector             │
documentation.                                                             │
halfvec: Supports up to 4,000 dimensions, according to pgvector            │
documentation.                                                             │
```                                                                        │
                                                                           │
Lastly I want you to use                                                   │
````
