# Context Server

A local, containerized documentation RAG system with MCP integration. Extract, process, and search documentation using PostgreSQL + pgvector for vector storage.

## Quick Start

1. **Prerequisites**
   ```bash
   # Install Docker and Docker Compose
   # Get an OpenAI API key for embeddings
   ```

2. **Setup**
   ```bash
   # Clone and setup
   git clone <your-repo>
   cd context_server

   # Configure environment
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Start the server**
   ```bash
   make up
   ```

   The API will be available at:
   - **API**: http://localhost:8000
   - **Docs**: http://localhost:8000/docs
   - **Database**: localhost:5432

4. **Try an example**
   ```bash
   # Create context and extract FastAPI docs
   make example-fastapi

   # Search the documentation
   make search QUERY="dependency injection" CONTEXT=fastapi
   ```

## Core Commands

### Server Management
```bash
make up          # Start server
make down        # Stop server
make logs        # View logs
make status      # Check health
make restart     # Restart server
```

### Context Management
```bash
make create-context NAME=my-docs DESC="My documentation"
make list-contexts
make delete-context NAME=my-docs
```

### Document Extraction
```bash
# Extract from URL
make extract URL=https://docs.python.org/ CONTEXT=python-docs

# Extract from file
make extract-file FILE=./document.pdf CONTEXT=my-docs

# List documents
make list-documents CONTEXT=my-docs
```

### Search
```bash
# Hybrid search (recommended)
make search QUERY="async functions" CONTEXT=python-docs

# Vector-only search
make search QUERY="async functions" CONTEXT=python-docs MODE=vector

# Full-text search
make search QUERY="async functions" CONTEXT=python-docs MODE=fulltext
```

## API Endpoints

The FastAPI server provides a full REST API:

- `POST /api/contexts` - Create context
- `GET /api/contexts` - List contexts
- `DELETE /api/contexts/{name}` - Delete context
- `POST /api/contexts/{name}/documents` - Add documents
- `GET /api/contexts/{name}/documents` - List documents
- `POST /api/contexts/{name}/search` - Search documents

See the interactive docs at http://localhost:8000/docs

## Architecture

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
└─────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Install development dependencies
make init

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Quality check (format + lint + test)
make quality-check
```

## Examples

### Rust Documentation
```bash
make example-rust
make search QUERY="HashMap" CONTEXT=rust-std
```

### FastAPI Documentation
```bash
make example-fastapi
make search QUERY="dependency injection" CONTEXT=fastapi
```

### Textual Documentation
```bash
make example-textual
make search QUERY="widgets" CONTEXT=textual
```

## Features

- ✅ **Multiple Contexts**: Separate vector databases for different projects
- ✅ **Hybrid Search**: Combine vector similarity with full-text search
- ✅ **Smart Extraction**: Uses crawl4ai for JavaScript-rendered sites
- ✅ **Clean Content**: Advanced markdown cleaning and noise reduction
- ✅ **Docker Ready**: Easy deployment with Docker Compose
- ✅ **REST API**: Full API for integration with other tools
- 🚧 **MCP Integration**: Bridge to Claude Code (Phase 3)
- 🚧 **Export/Import**: Context backup and sharing
- 🚧 **Git Repositories**: Extract from code repositories

## Troubleshooting

### Server won't start
```bash
# Check if ports are available
lsof -i :8000 -i :5432

# View logs
make logs
make logs-db
```

### No search results
```bash
# Check if OpenAI API key is set
grep OPENAI_API_KEY .env

# Check document count
make list-documents CONTEXT=your-context
```

### Database issues
```bash
# Reset database (WARNING: destroys all data)
make db-reset

# Connect to database shell
make db-shell
```

For more details, see the [Vision Document](CONTEXT_SERVER_VISION.md).
