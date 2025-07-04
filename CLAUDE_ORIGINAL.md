# Python Development Guidelines

## Project Overview

This document outlines coding standards, architectural principles, and development practices for Python projects focused on scalability, maintainability, and developer productivity.

## Core Principles

### 1. Scalable Code Patterns
- **Prefer composition over inheritance**
- **Use dependency injection for testability**
- **Create abstract base classes for extensibility**
- **Implement plugin architectures for processors**

### 2. Code Reuse Over Duplication
- **Extract common patterns into utilities**
- **Create shared interfaces for similar components**
- **Refactor when you see the same code in 3+ places**
- **Use mixins for cross-cutting concerns**

### 3. Clean Architecture
- **Separate concerns by layer (API, Business Logic, Data)**
- **Keep business logic framework-agnostic**
- **Use interfaces to define contracts**
- **Make dependencies explicit**

## Python Coding Standards

### Type Hints
```python
# ✅ DO: Use built-in types (Python 3.9+)
def process_documents(items: list[dict[str, str]]) -> dict[str, int]:
    pass

# ❌ DON'T: Use typing module unless necessary
from typing import List, Dict
def process_documents(items: List[Dict[str, str]]) -> Dict[str, int]:
    pass

# ✅ DO: Use typing for complex types
from typing import Protocol, TypeVar
class DocumentProcessor(Protocol):
    def process(self, content: bytes) -> ProcessedDocument: ...
```

### Error Handling
```python
# ✅ DO: Create specific exception types
class DocumentProcessingError(Exception):
    def __init__(self, message: str, document_id: str):
        self.document_id = document_id
        super().__init__(message)

# ✅ DO: Use context managers for resource cleanup
async def process_document(url: str) -> ProcessedDocument:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return await process_response(response)

# ❌ DON'T: Catch generic exceptions without re-raising
try:
    result = process_document(url)
except Exception:
    pass  # Silent failure is bad
```

### Logging
```python
# ✅ DO: Use structured logging
import structlog
logger = structlog.get_logger()

async def extract_document(url: str, context_id: str):
    logger.info("Starting document extraction", 
                url=url, context_id=context_id)
    try:
        result = await extractor.extract(url)
        logger.info("Document extraction completed", 
                    url=url, document_count=len(result.documents))
        return result
    except Exception as e:
        logger.error("Document extraction failed", 
                     url=url, error=str(e))
        raise
```

### Configuration
```python
# ✅ DO: Use Pydantic for configuration
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

# ✅ DO: Inject configuration
class DocumentProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
```

## Architecture Patterns

### Document Processors
```python
# Base processor interface
class DocumentProcessor:
    def can_handle(self, content_type: str, url: str = None) -> bool:
        """Check if this processor can handle the content type"""
        raise NotImplementedError
    
    async def process(self, content: bytes, metadata: dict) -> ProcessedDocument:
        """Process document content into structured format"""
        raise NotImplementedError

# Specific implementations
class PDFProcessor(DocumentProcessor):
    def can_handle(self, content_type: str, url: str = None) -> bool:
        return content_type == "application/pdf"
    
    async def process(self, content: bytes, metadata: dict) -> ProcessedDocument:
        # PDF-specific processing with table extraction
        pass

class URLProcessor(DocumentProcessor):
    def can_handle(self, content_type: str, url: str = None) -> bool:
        return content_type == "text/html"
    
    async def process(self, content: bytes, metadata: dict) -> ProcessedDocument:
        # Web scraping and cleaning
        pass

# Processor factory
class ProcessorFactory:
    def __init__(self, processors: list[DocumentProcessor]):
        self.processors = processors
    
    def get_processor(self, content_type: str, url: str = None) -> DocumentProcessor:
        for processor in self.processors:
            if processor.can_handle(content_type, url):
                return processor
        raise ValueError(f"No processor found for {content_type}")
```

### Repository Pattern
```python
# ✅ DO: Use repository pattern for data access
class ContextRepository:
    def __init__(self, db: Database):
        self.db = db
    
    async def create_context(self, context: CreateContextRequest) -> Context:
        # Database operations
        pass
    
    async def get_context(self, context_id: str) -> Context | None:
        # Database operations
        pass
    
    async def list_contexts(self) -> list[Context]:
        # Database operations
        pass

# Business logic layer
class ContextService:
    def __init__(self, repository: ContextRepository):
        self.repository = repository
    
    async def create_context(self, request: CreateContextRequest) -> Context:
        # Business logic, validation, etc.
        return await self.repository.create_context(request)
```

### Dependency Injection
```python
# ✅ DO: Use dependency injection container
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()
    
    # Database
    database = providers.Singleton(
        Database,
        url=config.database_url
    )
    
    # Repositories
    context_repository = providers.Factory(
        ContextRepository,
        db=database
    )
    
    # Services
    context_service = providers.Factory(
        ContextService,
        repository=context_repository
    )
    
    # Processors
    pdf_processor = providers.Factory(PDFProcessor)
    url_processor = providers.Factory(URLProcessor)
    
    processor_factory = providers.Factory(
        ProcessorFactory,
        processors=providers.List(
            pdf_processor,
            url_processor
        )
    )
```

## FastAPI Best Practices

### API Structure
```python
# ✅ DO: Organize endpoints by domain
# api/contexts.py
from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide

router = APIRouter(prefix="/api/contexts", tags=["contexts"])

@router.post("/")
@inject
async def create_context(
    request: CreateContextRequest,
    service: ContextService = Depends(Provide[Container.context_service])
) -> Context:
    return await service.create_context(request)

@router.get("/{context_id}")
@inject
async def get_context(
    context_id: str,
    service: ContextService = Depends(Provide[Container.context_service])
) -> Context:
    result = await service.get_context(context_id)
    if not result:
        raise HTTPException(status_code=404, detail="Context not found")
    return result
```

### Request/Response Models
```python
# ✅ DO: Use Pydantic models for API contracts
from pydantic import BaseModel, Field

class CreateContextRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    embedding_model: str = Field(default="text-embedding-3-small")

class Context(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int
    size_mb: float
    
    class Config:
        from_attributes = True
```

## Database Design

### Schema Organization
```sql
-- Each context gets its own schema
CREATE SCHEMA context_{context_id};

-- Documents table per context
CREATE TABLE context_{context_id}.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT,
    title TEXT,
    content TEXT,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_documents_embedding ON context_{context_id}.documents 
USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX idx_documents_metadata ON context_{context_id}.documents 
USING gin (metadata);
```

### Migrations
```python
# ✅ DO: Use Alembic for database migrations
# alembic/versions/001_initial_schema.py
def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.create_table(
        'contexts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
```

## Testing Strategy

### Unit Tests
```python
# ✅ DO: Test business logic independently
import pytest
from unittest.mock import AsyncMock

class TestContextService:
    @pytest.fixture
    def mock_repository(self):
        return AsyncMock(spec=ContextRepository)
    
    @pytest.fixture
    def service(self, mock_repository):
        return ContextService(mock_repository)
    
    async def test_create_context(self, service, mock_repository):
        # Arrange
        request = CreateContextRequest(name="test", description="test context")
        expected_context = Context(id="ctx_123", name="test", ...)
        mock_repository.create_context.return_value = expected_context
        
        # Act
        result = await service.create_context(request)
        
        # Assert
        assert result == expected_context
        mock_repository.create_context.assert_called_once_with(request)
```

### Integration Tests
```python
# ✅ DO: Test full workflows
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

class TestContextAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_create_and_retrieve_context(self, client):
        # Create context
        response = client.post("/api/contexts", json={
            "name": "test-context",
            "description": "Test context"
        })
        assert response.status_code == 201
        context = response.json()
        
        # Retrieve context
        response = client.get(f"/api/contexts/{context['id']}")
        assert response.status_code == 200
        assert response.json()["name"] == "test-context"
```

## Performance Guidelines

### Database Optimization
```python
# ✅ DO: Use connection pooling
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# ✅ DO: Use bulk operations for large datasets
async def bulk_insert_embeddings(documents: list[ProcessedDocument]):
    async with async_session() as session:
        await session.execute(
            insert(Document).values([doc.dict() for doc in documents])
        )
        await session.commit()
```

### Caching Strategy
```python
# ✅ DO: Cache expensive operations
from functools import lru_cache
import asyncio

class EmbeddingService:
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def get_embedding_sync(self, text: str) -> list[float]:
        # Expensive embedding calculation
        pass
    
    async def get_embedding(self, text: str) -> list[float]:
        # Run in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_embedding_sync, text
        )
```

## Security Considerations

### Input Validation
```python
# ✅ DO: Validate and sanitize inputs
from pydantic import BaseModel, validator, HttpUrl

class ExtractRequest(BaseModel):
    url: HttpUrl
    max_depth: int = Field(default=3, ge=1, le=10)
    
    @validator('url')
    def validate_url(cls, v):
        # Only allow certain domains for security
        allowed_domains = ['docs.rs', 'doc.rust-lang.org', 'fastapi.tiangolo.com']
        if v.host not in allowed_domains:
            raise ValueError('Domain not allowed')
        return v
```

### Rate Limiting
```python
# ✅ DO: Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/contexts/{context_id}/documents")
@limiter.limit("10/minute")
async def add_document(request: Request, ...):
    # API endpoint with rate limiting
    pass
```

## Monitoring and Observability

### Metrics
```python
# ✅ DO: Add metrics for important operations
from prometheus_client import Counter, Histogram, Gauge

extraction_counter = Counter('documents_extracted_total', 'Total documents extracted')
extraction_duration = Histogram('document_extraction_duration_seconds', 'Time spent extracting documents')
active_contexts = Gauge('active_contexts_total', 'Number of active contexts')

class DocumentService:
    async def extract_document(self, url: str):
        with extraction_duration.time():
            result = await self._extract(url)
            extraction_counter.inc()
            return result
```

### Health Checks
```python
# ✅ DO: Implement comprehensive health checks
@router.get("/health")
async def health_check():
    checks = {
        "database": await check_database_health(),
        "embedding_service": await check_embedding_service(),
        "storage": await check_storage_health()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
    )
```

## Development Workflow

### Code Quality
```bash
# ✅ DO: Use these tools in CI/CD
make format  # black + isort
make lint    # flake8 + mypy
make test    # pytest with coverage
```

### Development Workflow
```bash
# ✅ DO: Use conventional commit messages (without git commands)
# Examples of good commit messages:
# "feat: add PDF table extraction support"
# "fix: handle malformed URLs in extraction" 
# "refactor: extract common processor interface"
```

## Refactoring Guidelines

### When to Refactor
1. **Rule of Three**: If you copy code 3 times, extract it
2. **Long Methods**: Methods over 20 lines should be split
3. **Large Classes**: Classes with more than 10 methods need splitting
4. **Complex Conditionals**: Extract complex logic into methods
5. **Duplicate Logic**: Similar code in multiple places

### Refactoring Process
1. **Write tests first** to ensure behavior doesn't change
2. **Make small, incremental changes**
3. **Run tests after each change**
4. **Update documentation** when interfaces change
5. **Get code review** for architectural changes

## Common Patterns

### Factory Pattern
```python
# ✅ DO: Use factory pattern for object creation
class DocumentProcessorFactory:
    def __init__(self):
        self.processors = {
            'application/pdf': PDFProcessor,
            'text/html': URLProcessor,
            'text/plain': TextProcessor
        }
    
    def create_processor(self, content_type: str) -> DocumentProcessor:
        processor_class = self.processors.get(content_type)
        if not processor_class:
            raise ValueError(f"Unsupported content type: {content_type}")
        return processor_class()
```

### Strategy Pattern
```python
# ✅ DO: Use strategy pattern for algorithms
class SearchStrategy:
    def search(self, query: str, context_id: str) -> list[SearchResult]:
        raise NotImplementedError

class VectorSearchStrategy(SearchStrategy):
    def search(self, query: str, context_id: str) -> list[SearchResult]:
        # Vector similarity search
        pass

class FullTextSearchStrategy(SearchStrategy):
    def search(self, query: str, context_id: str) -> list[SearchResult]:
        # Full-text search
        pass

class HybridSearchStrategy(SearchStrategy):
    def __init__(self, vector_strategy: VectorSearchStrategy, 
                 fulltext_strategy: FullTextSearchStrategy):
        self.vector_strategy = vector_strategy
        self.fulltext_strategy = fulltext_strategy
    
    def search(self, query: str, context_id: str) -> list[SearchResult]:
        # Combine both strategies
        pass
```

## Dependency Management

### Use pyproject.toml over requirements.txt
```toml
# ✅ DO: Use pyproject.toml for modern Python projects
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-project"
version = "0.1.0"
description = "Your project description"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
]

test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Modern Python Environment Management

#### Use uv over pip
uv is a fast Python package installer and resolver, written in Rust. It's significantly faster than pip and provides better dependency resolution.

```bash
# ✅ DO: Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# ✅ DO: Use uv for all package management
uv pip install -e .              # Install in development mode
uv pip install -e ".[dev]"        # Install with dev dependencies
uv pip install -e ".[test]"       # Install with test dependencies
uv pip install -e ".[dev,test]"   # Install with all optional dependencies

# For production
uv pip install .
```

### Virtual Environment Setup

#### Makefile-based Environment Management
Create a comprehensive Makefile that handles virtual environment creation and management:

```makefile
# Project configuration
VENV_NAME := .venv
PYTHON := python3.9
VENV_PYTHON := $(VENV_NAME)/bin/python
VENV_PIP := $(VENV_NAME)/bin/uv pip
VENV_ACTIVATE := source $(VENV_NAME)/bin/activate

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help init clean test test-coverage lint format pre-commit-install

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Environment setup
init: ## Initialize project (create venv, install deps)
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Virtual environment already exists$(NC)"; \
		echo "$(GREEN)Activating existing environment and updating dependencies...$(NC)"; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
	else \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)Installing uv...$(NC)"; \
		$(VENV_NAME)/bin/pip install uv; \
		echo "$(GREEN)Installing project dependencies...$(NC)"; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
		echo "$(GREEN)Installing pre-commit hooks...$(NC)"; \
		$(VENV_ACTIVATE) && pre-commit install; \
		$(VENV_ACTIVATE) && pre-commit install --hook-type commit-msg; \
	fi
	@echo "$(GREEN)Setup complete! Activate with: source $(VENV_NAME)/bin/activate$(NC)"

venv: ## Create virtual environment only
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_NAME); \
		$(VENV_NAME)/bin/pip install uv; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(NC)"; \
	fi

install: venv ## Install project dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"

install-prod: venv ## Install production dependencies only
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(VENV_ACTIVATE) && uv pip install -e .

# Development tools
format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	$(VENV_ACTIVATE) && black .
	$(VENV_ACTIVATE) && isort .

lint: ## Run linting
	@echo "$(GREEN)Running linters...$(NC)"
	$(VENV_ACTIVATE) && flake8 .
	$(VENV_ACTIVATE) && mypy .
	$(VENV_ACTIVATE) && bandit -r . -x tests/

# Testing
test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(VENV_ACTIVATE) && pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(VENV_ACTIVATE) && pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(VENV_ACTIVATE) && pytest-watch tests/ -- -v

test-specific: ## Run specific test (usage: make test-specific TEST=test_name)
	@if [ -z "$(TEST)" ]; then \
		echo "$(RED)Usage: make test-specific TEST=test_name$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Running specific test: $(TEST)$(NC)"
	$(VENV_ACTIVATE) && pytest tests/ -k "$(TEST)" -v

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	$(VENV_ACTIVATE) && pre-commit install
	$(VENV_ACTIVATE) && pre-commit install --hook-type commit-msg

pre-commit-run: ## Run pre-commit on all files
	$(VENV_ACTIVATE) && pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	$(VENV_ACTIVATE) && pre-commit autoupdate

# Quality checks
quality-check: format lint test pre-commit-run ## Run full quality check

# Cleanup
clean: ## Clean up generated files and caches
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete
	find . -type d -name "*.egg-info" -delete

clean-venv: ## Remove virtual environment
	@echo "$(RED)Removing virtual environment...$(NC)"
	rm -rf $(VENV_NAME)

reset: clean-venv init ## Reset environment (remove venv and recreate)

# Development server (if applicable)
dev: ## Start development server
	$(VENV_ACTIVATE) && python -m src.main

# Environment info
env-info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python version: $$($(VENV_PYTHON) --version)"
	@echo "Virtual environment: $(VENV_NAME)"
	@echo "Activated: $$([ "$$VIRTUAL_ENV" ] && echo "Yes" || echo "No")"
	@if [ -f "$(VENV_PYTHON)" ]; then \
		echo "Installed packages:"; \
		$(VENV_ACTIVATE) && uv pip list; \
	fi

# Dependency management
deps-update: ## Update all dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(VENV_ACTIVATE) && uv pip install --upgrade -e ".[dev,test]"

deps-audit: ## Audit dependencies for security issues
	@echo "$(GREEN)Auditing dependencies...$(NC)"
	$(VENV_ACTIVATE) && uv pip install pip-audit
	$(VENV_ACTIVATE) && pip-audit

# Project shortcuts
build: ## Build the project
	$(VENV_ACTIVATE) && python -m build

publish: ## Publish to PyPI (requires authentication)
	$(VENV_ACTIVATE) && python -m twine upload dist/*

# Help with activation
activate: ## Show activation command
	@echo "$(GREEN)To activate the virtual environment, run:$(NC)"
	@echo "source $(VENV_NAME)/bin/activate"
```

#### Key Features of this Makefile:

1. **Smart `make init`**:
   - Creates venv if it doesn't exist
   - Sources existing venv if it does exist
   - Installs all dependencies with uv
   - Sets up pre-commit hooks automatically

2. **Always uses uv** for package management
3. **Environment-aware testing** - all commands source the venv
4. **Comprehensive commands** for development workflow
5. **Color-coded output** for better UX
6. **Dependency management** with uv
7. **Security auditing** capabilities

### Testing Best Practices

#### Test Structure and Organization
```
project/
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_services.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── fixtures/
│       ├── sample_data.json
│       └── test_files/
└── src/
    └── your_package/
```

#### Enhanced pyproject.toml for Testing
```toml
[project.optional-dependencies]
test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",      # Parallel test execution
    "pytest-watch>=4.2.0",      # Watch mode
    "pytest-mock>=3.11.0",      # Better mocking
    "factory-boy>=3.3.0",       # Test data factories
    "faker>=19.6.0",             # Fake data generation
    "httpx>=0.25.0",             # Async HTTP client for testing
    "respx>=0.20.0",             # HTTP mocking
]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "external: marks tests that require external services",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/migrations/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

#### Common Testing Patterns

##### Async Testing
```python
# ✅ DO: Test async functions properly
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_function():
    mock_service = AsyncMock()
    mock_service.fetch_data.return_value = {"data": "test"}
    
    result = await your_async_function(mock_service)
    
    assert result == {"data": "test"}
    mock_service.fetch_data.assert_called_once()
```

##### Fixture Usage
```python
# ✅ DO: Use fixtures for test setup
# conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_database():
    return AsyncMock()

@pytest.fixture
async def client():
    # Setup test client
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture(scope="session")
def test_data():
    return {
        "users": [{"id": 1, "name": "Test User"}],
        "documents": [{"id": 1, "title": "Test Doc"}]
    }
```

##### Parameterized Tests
```python
# ✅ DO: Use parametrization for multiple test cases
@pytest.mark.parametrize("input_value,expected", [
    ("valid_input", True),
    ("invalid_input", False),
    ("", False),
    (None, False),
])
def test_validation_function(input_value, expected):
    result = validate_input(input_value)
    assert result == expected
```

##### Integration Testing
```python
# ✅ DO: Test API endpoints end-to-end
@pytest.mark.integration
async def test_create_context_endpoint(client, test_database):
    response = await client.post("/api/contexts", json={
        "name": "test-context",
        "description": "Test context"
    })
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-context"
    
    # Verify in database
    context = await test_database.get_context(data["id"])
    assert context is not None
```

#### Test Commands and Usage

```bash
# ✅ DO: Use make commands for consistent test execution

# Basic test run
make test

# Test with coverage
make test-coverage

# Run specific test
make test-specific TEST=test_user_creation

# Run tests in watch mode (for development)
make test-watch

# Run only fast tests (skip slow/integration tests)
source .venv/bin/activate && pytest -m "not slow"

# Run tests in parallel
source .venv/bin/activate && pytest -n auto

# Run tests with verbose output
source .venv/bin/activate && pytest -v -s

# Run only integration tests
source .venv/bin/activate && pytest -m integration

# Generate coverage report and open in browser
make test-coverage && open htmlcov/index.html
```

#### Test Environment Variables
```bash
# ✅ DO: Use .env.test for test-specific configuration
# .env.test
DATABASE_URL=sqlite:///test.db
LOG_LEVEL=DEBUG
TESTING=true
OPENAI_API_KEY=test-key
```

#### Continuous Integration Testing
```yaml
# ✅ DO: Include in CI/CD pipeline
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Create venv and install dependencies
      run: |
        python -m venv .venv
        source .venv/bin/activate
        uv pip install -e ".[dev,test]"
    
    - name: Run quality checks
      run: |
        source .venv/bin/activate
        make quality-check
```

### Development Workflow Summary

```bash
# ✅ DO: Standard development workflow

# 1. Initial setup (only once)
make init

# 2. Activate environment (each session)
source .venv/bin/activate

# 3. Development cycle
make test-watch  # Keep running in one terminal
# Make code changes
# Tests run automatically

# 4. Before committing
make quality-check

# 5. Update dependencies (occasionally)
make deps-update

# 6. Clean up (when needed)
make clean
```

This setup ensures that every Python project has a consistent, professional development environment with proper testing, linting, and dependency management using modern tools like uv.

## Code Quality Automation

### Pre-commit Setup

Pre-commit hooks automatically run linters and formatters before each commit, ensuring consistent code quality.

#### Installation and Setup
```bash
# ✅ DO: Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Optional: Install commit-msg hook for conventional commits
pre-commit install --hook-type commit-msg
```

#### .pre-commit-config.yaml
```yaml
# ✅ DO: Create .pre-commit-config.yaml in project root
repos:
  # Black - Python code formatter
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.9

  # isort - Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black", "--line-length", "88"]

  # flake8 - Linting
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [
          "flake8-docstrings",
          "flake8-bugbear", 
          "flake8-comprehensions",
          "flake8-simplify"
        ]
        args: [
          "--max-line-length=88",
          "--extend-ignore=E203,W503",
          "--max-complexity=10"
        ]

  # mypy - Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-pyyaml]
        args: [--strict]

  # General hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files
      - id: debug-statements
      - id: check-docstring-first

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-r", ".", "-x", "tests/"]

  # Conventional commits (optional)
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [optional-scope]
```

#### Enhanced pyproject.toml with Pre-commit
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0", 
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "flake8-docstrings>=1.7.0",
    "flake8-bugbear>=23.9.0",
    "flake8-comprehensions>=3.14.0",
    "flake8-simplify>=0.20.0",
    "mypy>=1.6.0",
    "bandit>=1.7.5",
    "pre-commit>=3.5.0",
]

# Additional tool configurations
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
max-complexity = 10
docstring-convention = "google"

[tool.bandit]
exclude_dirs = ["tests", "test_*.py", "*_test.py"]
skips = ["B101", "B601"]
```

#### Usage Commands
```bash
# Run pre-commit on all files (useful for first setup)
pre-commit run --all-files

# Run specific hook
pre-commit run black
pre-commit run mypy

# Update hook versions
pre-commit autoupdate

# Bypass hooks (emergency only)
git commit --no-verify -m "emergency fix"

# Test hooks without committing
pre-commit run --all-files --verbose
```

### IDE Integration

#### VS Code Settings (.vscode/settings.json)
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.banditEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
```

#### .flake8 Configuration File
```ini
# ✅ DO: Create .flake8 file for project-specific settings
[flake8]
max-line-length = 88
extend-ignore = E203, W503
max-complexity = 10
docstring-convention = google
exclude = 
    .git,
    __pycache__,
    .pytest_cache,
    .mypy_cache,
    .venv,
    venv,
    build,
    dist
per-file-ignores =
    __init__.py:F401
    tests/*:D
```

### Makefile Integration
```makefile
# ✅ DO: Add pre-commit targets to Makefile
.PHONY: pre-commit-install pre-commit-run pre-commit-update

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	pre-commit autoupdate

format: ## Format code
	black .
	isort .

lint: ## Run all linters
	flake8 .
	mypy .
	bandit -r . -x tests/

quality-check: format lint pre-commit-run ## Run full quality check
```

### Benefits of Pre-commit

1. **Consistency**: Ensures all team members follow the same code standards
2. **Early Detection**: Catches issues before they reach CI/CD
3. **Productivity**: Automates formatting and basic fixes
4. **Security**: Runs security checks with bandit
5. **Documentation**: Enforces docstring standards
6. **Type Safety**: Validates type hints with mypy

### Troubleshooting Pre-commit

```bash
# ✅ DO: Common fixes for pre-commit issues

# Clear pre-commit cache
pre-commit clean

# Reinstall hooks
pre-commit uninstall
pre-commit install

# Skip failing hooks temporarily
SKIP=mypy git commit -m "fix: temporary mypy bypass"

# Debug specific hook
pre-commit run mypy --verbose --all-files

# Update Python version in hooks
pre-commit run --all-files
```

This document serves as the foundation for maintaining high code quality and architectural consistency throughout any Python project. Follow these guidelines to ensure the codebase remains maintainable, testable, and scalable.