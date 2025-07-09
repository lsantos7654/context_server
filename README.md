# Context Server - Documentation RAG System

A modern CLI for documentation RAG system with FastAPI backend, PostgreSQL + pgvector storage, and semantic search capabilities.

## 🚀 Current Enhancement Phase

We're currently implementing major improvements to the Context Server architecture. This README tracks our progress through the enhancement phase.

### Enhancement Goals

1. **Modern Chunking Strategy** - Replace custom chunking with LangChain's RecursiveCharacterTextSplitter
2. **MCP Document Pagination** - Handle Claude's 25k token limit with smart pagination
3. **Three-Document Pipeline** - Process sources into original, code snippets, and cleaned markdown
4. **Advanced Embeddings** - Upgrade to text-embedding-3-large (3072 dims) + voyage-code-3 for code
5. **Enhanced Search** - Separate document and code search endpoints with optimized models

### Current Status

#### ✅ Completed
- Initial project analysis and planning
- Enhancement plan approved
- Added langchain and voyageai dependencies
- Replaced chunking strategy with LangChain RecursiveCharacterTextSplitter
- Updated database schema to use halfvec and add document_type field
- Created VoyageAI embedding service for code snippets
- Updated embedding service to support text-embedding-3-large
- Modified document processing to create 3-document pipeline
- Added code snippet metadata placeholders to cleaned markdown

#### 🚧 In Progress
- Enhancing summarization to generate 3-5 sentences

#### 📋 Planned
- MCP pagination support
- Separate code search endpoint
- Comprehensive testing

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   URL/File      │    │   Processing    │    │   Storage       │
│   Extraction    │───▶│   Pipeline      │───▶│   PostgreSQL    │
│   (crawl4ai)    │    │                 │    │   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Three Documents │
                       │  Per Source:     │
                       │  • Original      │
                       │  • Code Snippets │
                       │  • Cleaned Text  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Embeddings    │
                       │  • text-embed-  │
                       │    3-large      │
                       │  • voyage-code-3│
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Search APIs   │
                       │  • Document     │
                       │  • Code         │
                       │  • MCP Tools    │
                       └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- PostgreSQL with pgvector extension
- Docker (for containerized services)
- OpenAI API key
- Voyage AI API key

### Setup
```bash
# 1. Clone and setup environment
git clone <repository-url>
cd context_server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e .

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your API keys and database URL

# 4. Start services
ctx server up

# 5. Verify installation
ctx --help
```

## Usage

### Basic Commands
```bash
# Server management
ctx server up              # Start Docker services
ctx server down            # Stop services  
ctx server restart         # Restart after changes
ctx server logs           # View logs

# Context management
ctx context create my-docs              # Create context
ctx context list                       # List contexts
ctx context delete my-docs            # Delete context

# Document extraction
ctx docs extract https://docs.rust-lang.org rust-docs
ctx docs extract ./my-project my-code --source-type local
ctx docs list my-docs                  # List documents

# Search
ctx search query "async functions" my-docs
ctx search query "error handling" my-docs --limit 10
```

### MCP Integration
The server provides MCP (Model Context Protocol) tools for Claude integration:

```python
# Available MCP tools
- create_context(name, description, embedding_model)
- list_contexts()
- get_context(context_name)
- delete_context(context_name)
- extract_url(context_name, url, max_pages)
- extract_file(context_name, file_path)
- search_context(context_name, query, mode, limit)
- get_document(context_name, doc_id, page_number, page_size)  # New: Paginated
- search_code(context_name, query, language, limit)           # New: Code search
```

## Technical Specifications

### Current Configuration
- **Text Chunking**: LangChain RecursiveCharacterTextSplitter
  - Regular text: 1000 chars, 200 overlap
  - Code: 700 chars, 150 overlap
- **Embeddings**: 
  - Documents: OpenAI text-embedding-3-large (3072 dims)
  - Code: Voyage AI voyage-code-3 (2048 dims)
- **Storage**: PostgreSQL with pgvector halfvec support
- **Summarization**: 3-5 sentence summaries (50-150 words)

### Database Schema
```sql
-- Core tables with halfvec support
contexts (id, name, description, embedding_model)
documents (id, context_id, url, title, content, document_type, metadata)
chunks (id, document_id, content, embedding halfvec(3072), summary, metadata)
code_snippets (id, document_id, content, language, embedding halfvec(2048), metadata)
jobs (id, type, status, progress, metadata)
```

### Document Processing Pipeline
1. **URL/File Extraction** - crawl4ai processes source content
2. **Content Cleaning** - Remove noise, normalize structure
3. **Code Extraction** - Separate code blocks from text
4. **Three-Document Creation**:
   - Original: Raw parsed markdown
   - Code Snippets: Extracted code blocks
   - Cleaned Markdown: Text with code snippet placeholders
5. **Chunking** - LangChain RecursiveCharacterTextSplitter
6. **Embedding Generation** - Dual embedding models
7. **Storage** - PostgreSQL with vector indexes

### Code Snippet Placeholders
In cleaned markdown documents, code blocks are replaced with metadata:
```markdown
[CODE_SNIPPET: language=python, size=245_chars, summary="Function to handle user authentication with JWT tokens, validates credentials against database, and returns authentication status", snippet_id=uuid-here]
```

## API Endpoints

### Core API
- `GET /api/contexts` - List contexts
- `POST /api/contexts` - Create context
- `GET /api/contexts/{name}` - Get context details
- `DELETE /api/contexts/{name}` - Delete context

### Document Management
- `POST /api/contexts/{name}/documents` - Extract document
- `GET /api/contexts/{name}/documents` - List documents
- `GET /api/contexts/{name}/documents/{id}/raw` - Get document (paginated)
- `DELETE /api/contexts/{name}/documents` - Delete documents

### Search
- `GET /api/contexts/{name}/search` - Search documents
- `GET /api/contexts/{name}/search/code` - Search code snippets
- `GET /api/contexts/{name}/documents/{id}/code-snippets` - List code snippets
- `GET /api/contexts/{name}/code-snippets/{id}` - Get code snippet

### Job Management
- `GET /api/jobs/{id}/status` - Check job status
- `POST /api/jobs/{id}/cancel` - Cancel job
- `DELETE /api/jobs/cleanup` - Clean completed jobs

## Development

### Project Structure
```
context_server/
├── cli/                 # CLI commands and interface
├── api/                 # FastAPI REST endpoints
├── core/               # Core business logic
│   ├── chunking.py     # LangChain text splitting
│   ├── embeddings.py   # OpenAI + Voyage AI services
│   ├── processing.py   # Document processing pipeline  
│   ├── storage.py      # PostgreSQL + pgvector
│   └── crawl4ai_extraction.py  # Web scraping
├── mcp_server/         # MCP protocol implementation
└── tests/              # Test suite
```

### Enhancement Implementation Log

#### Phase 1: Foundation (Completed)
- [x] **Dependencies**: Add langchain and voyageai to pyproject.toml
- [x] **Chunking**: Replace custom TextChunker with RecursiveCharacterTextSplitter
- [x] **Database**: Migrate schema to halfvec support
- [x] **Embeddings**: Implement dual embedding services

#### Phase 2: Core Features (In Progress)
- [x] **Document Pipeline**: Implement three-document processing
- [x] **Code Placeholders**: Add metadata placeholders in cleaned markdown
- [ ] **Summarization**: Enhance to 3-5 sentences
- [ ] **Storage**: Update all database operations

#### Phase 3: Search & MCP
- [ ] **Code Search**: Implement separate code search endpoint
- [ ] **Pagination**: Add MCP document pagination
- [ ] **MCP Tools**: Update all MCP tools with new features
- [ ] **Testing**: Comprehensive test suite

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow

# Run with coverage
pytest --cov=context_server --cov-report=html
```

## Troubleshooting

### Common Issues
1. **Command not found**: Always activate virtual environment first
   ```bash
   source .venv/bin/activate
   ```

2. **Import errors**: Check server logs and restart services
   ```bash
   ctx server logs
   ctx server restart
   ```

3. **Connection errors**: Ensure PostgreSQL is running
   ```bash
   ctx server up
   ctx server status
   ```

4. **API key issues**: Verify environment variables in `.env`
   ```bash
   echo $OPENAI_API_KEY
   echo $VOYAGE_API_KEY
   ```

### Debug Commands
```bash
# Check server status
ctx server status

# View detailed logs
ctx server logs

# Test database connection
ctx context list

# Test embedding services
ctx docs extract https://example.com test-context --max-pages 1
```

## License
MIT License - See LICENSE file for details

## Changelog

### v0.2.0 (In Development)
- 🔄 **Enhanced Chunking**: LangChain RecursiveCharacterTextSplitter
- 🔄 **Advanced Embeddings**: text-embedding-3-large + voyage-code-3
- 🔄 **Three-Document Pipeline**: Original, code snippets, cleaned markdown
- 🔄 **MCP Pagination**: Support for large documents
- 🔄 **Code Search**: Separate code search with optimized embeddings
- 🔄 **Enhanced Summaries**: 3-5 sentence summaries

### v0.1.0 (Current)
- ✅ **Core RAG System**: FastAPI + PostgreSQL + pgvector
- ✅ **Document Extraction**: crawl4ai web scraping
- ✅ **Vector Search**: OpenAI embeddings with similarity search
- ✅ **MCP Integration**: Claude-compatible tools
- ✅ **CLI Interface**: Comprehensive command-line tools
- ✅ **Job Management**: Async processing with progress tracking

---

*Last updated: 2025-01-09*
*Enhancement phase: Active development*