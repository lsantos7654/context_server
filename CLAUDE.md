# Context Server Development Guidelines

## Project Overview

This is the Context Server project - a modern CLI for documentation RAG system with FastAPI backend, PostgreSQL + pgvector storage, and semantic search capabilities. This document outlines coding standards, architectural principles, and development practices.

## Essential Setup Commands

**IMPORTANT:** Always activate the virtual environment before running any commands:

```bash
# REQUIRED: Activate virtual environment first
source .venv/bin/activate

# Then run Context Server commands
ctx --help                    # Show available commands
ctx server up                 # Start services (Docker)
ctx server logs              # Check server logs (troubleshooting)
ctx server restart           # Restart after code changes
ctx context list             # List available contexts
ctx search query "term" docs  # Search documentation
```

## Troubleshooting

### Common Issues
1. **Command not found**: Activate `.venv` first with `source .venv/bin/activate`
2. **Import errors**: Check `ctx server logs` for Python import issues
3. **Connection errors**: Ensure server is running with `ctx server up`
4. **API changes**: Restart server with `ctx server restart` after code changes

### Debug Steps
```bash
source .venv/bin/activate
ctx server logs              # Check for errors
ctx server restart           # Apply code changes
ctx context list             # Verify database connection
```

## Quick Setup

```bash
# 1. Clone project and setup environment
make init

# 2. Development workflow
make test-watch  # Keep running during development
make quality-check  # Before committing

# 3. Common commands
make test           # Run tests
make test-coverage  # Test with coverage
make format         # Format code
make lint          # Run linters
```

## Core Principles

### 1. Scalable Code Patterns
- **Prefer composition over inheritance**
- **Use dependency injection for testability**
- **Create abstract base classes for extensibility**
- **Implement plugin architectures**

### 2. Code Reuse Over Duplication
- **Extract common patterns into utilities**
- **Refactor when you see the same code in 3+ places**
- **Use mixins for cross-cutting concerns**

### 3. Clean Architecture
- **Separate concerns by layer (API, Business Logic, Data)**
- **Keep business logic framework-agnostic**
- **Use interfaces to define contracts**
- **Make dependencies explicit**

## Python Standards

### Type Hints
```python
# ✅ DO: Use built-in types (Python 3.9+)
def process_items(items: list[dict[str, str]]) -> dict[str, int]:
    pass

# ✅ DO: Use typing for complex types only
from typing import Protocol
class Processor(Protocol):
    def process(self, content: bytes) -> ProcessedData: ...
```

### Error Handling
```python
# ✅ DO: Create specific exception types
class ProcessingError(Exception):
    def __init__(self, message: str, item_id: str):
        self.item_id = item_id
        super().__init__(message)

# ✅ DO: Use context managers
async def process_item(url: str) -> ProcessedItem:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return await process_response(response)
```

### Logging
```python
# ✅ DO: Use structured logging
import structlog
logger = structlog.get_logger()

async def process_item(url: str, context_id: str):
    logger.info("Starting processing", url=url, context_id=context_id)
    try:
        result = await processor.process(url)
        logger.info("Processing completed", url=url, count=len(result.items))
        return result
    except Exception as e:
        logger.error("Processing failed", url=url, error=str(e))
        raise
```

## Modern Python Setup

### Use uv over pip
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use uv for all package operations
uv pip install -e ".[dev,test]"
```

### Makefile Template
```makefile
# Project configuration
VENV_NAME := .venv
PYTHON := python3.9
VENV_ACTIVATE := source $(VENV_NAME)/bin/activate

# Colors
GREEN := \033[0;32m
NC := \033[0m

.PHONY: help init test test-coverage format lint quality-check clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

init: ## Initialize project (create venv, install deps)
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "$(GREEN)Updating existing environment...$(NC)"; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
	else \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_NAME); \
		$(VENV_NAME)/bin/pip install uv; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
		$(VENV_ACTIVATE) && pre-commit install; \
	fi

test: ## Run tests
	$(VENV_ACTIVATE) && pytest tests/ -v

test-coverage: ## Run tests with coverage
	$(VENV_ACTIVATE) && pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-watch: ## Run tests in watch mode
	$(VENV_ACTIVATE) && pytest-watch tests/ -- -v

format: ## Format code
	$(VENV_ACTIVATE) && black .
	$(VENV_ACTIVATE) && isort .

lint: ## Run linting
	$(VENV_ACTIVATE) && flake8 .
	$(VENV_ACTIVATE) && mypy .
	$(VENV_ACTIVATE) && bandit -r . -x tests/

quality-check: format lint test ## Run full quality check

clean: ## Clean up caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete

clean-venv: ## Remove virtual environment
	rm -rf $(VENV_NAME)

reset: clean-venv init ## Reset environment
```

## pyproject.toml Template

```toml
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
    "httpx>=0.25.0",
]

[project.optional-dependencies]
dev = [
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
    "bandit>=1.7.5",
    "pre-commit>=3.5.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-watch>=4.2.0",
    "factory-boy>=3.3.0",
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
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Pre-commit Setup

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        args: [--strict]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: debug-statements

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-r", ".", "-x", "tests/"]
```

## Architecture Patterns

### Plugin Architecture
```python
# Base interface
class Processor:
    def can_handle(self, content_type: str) -> bool:
        raise NotImplementedError

    async def process(self, content: bytes, metadata: dict) -> ProcessedData:
        raise NotImplementedError

# Factory pattern
class ProcessorFactory:
    def __init__(self, processors: list[Processor]):
        self.processors = processors

    def get_processor(self, content_type: str) -> Processor:
        for processor in self.processors:
            if processor.can_handle(content_type):
                return processor
        raise ValueError(f"No processor found for {content_type}")
```

### Repository Pattern
```python
class Repository:
    def __init__(self, db: Database):
        self.db = db

    async def create(self, item: CreateRequest) -> Item:
        # Database operations
        pass

    async def get(self, item_id: str) -> Item | None:
        # Database operations
        pass

# Business logic layer
class Service:
    def __init__(self, repository: Repository):
        self.repository = repository

    async def create_item(self, request: CreateRequest) -> Item:
        # Business logic, validation, etc.
        return await self.repository.create(request)
```

### Dependency Injection
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    database = providers.Singleton(Database, url=config.database_url)
    repository = providers.Factory(Repository, db=database)
    service = providers.Factory(Service, repository=repository)
```

## Testing Patterns

### Test Structure
```
tests/
├── conftest.py          # Shared fixtures
├── unit/
│   ├── test_services.py
│   └── test_models.py
├── integration/
│   └── test_api.py
└── fixtures/
    └── sample_data.json
```

### Common Patterns
```python
# Async testing
@pytest.mark.asyncio
async def test_async_function():
    mock_service = AsyncMock()
    mock_service.fetch_data.return_value = {"data": "test"}

    result = await your_async_function(mock_service)

    assert result == {"data": "test"}
    mock_service.fetch_data.assert_called_once()

# Parameterized tests
@pytest.mark.parametrize("input_value,expected", [
    ("valid", True),
    ("invalid", False),
    ("", False),
])
def test_validation(input_value, expected):
    result = validate_input(input_value)
    assert result == expected

# Fixtures
@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
```

## Context Server Development Workflow

```bash
# 1. Initial setup (once)
make init

# 2. Daily development (ALWAYS start with this)
source .venv/bin/activate  # REQUIRED for every terminal session

# 3. Context Server operations
ctx server up              # Start Docker services
ctx server logs           # Monitor for issues
ctx context list          # Verify connection

# 4. Development workflow
make test-watch           # Keep running during development
ctx server restart        # After code changes

# 5. Before committing
make quality-check

# 6. Common tasks
make test                 # Run tests
make test-coverage        # Test with coverage
make format              # Format code
make lint               # Run linters
```

## Context Server Specific Commands

### Server Management
```bash
source .venv/bin/activate  # ALWAYS FIRST
ctx server up              # Start all services (PostgreSQL, Redis, API)
ctx server down            # Stop all services
ctx server restart         # Restart after code changes
ctx server logs           # View logs (essential for debugging)
ctx server status          # Check service health
```

### Context & Document Management
```bash
ctx context create my-docs              # Create new context
ctx context list                       # List all contexts
ctx docs extract https://... my-docs    # Extract documentation
ctx docs list my-docs                  # List documents in context
```

### Search & Testing
```bash
ctx search query "widget" my-docs       # Basic search
ctx search query "async" docs --limit 5 # Limited results
```

## Refactoring Guidelines

### When to Refactor
1. **Rule of Three**: If you copy code 3 times, extract it
2. **Long Methods**: Methods over 20 lines should be split
3. **Large Classes**: Classes with more than 10 methods need splitting
4. **Complex Conditionals**: Extract complex logic into methods

### Refactoring Process
1. **Write tests first** to ensure behavior doesn't change
2. **Make small, incremental changes**
3. **Run tests after each change**
4. **Update documentation** when interfaces change

## Key Takeaways

- **ALWAYS activate `.venv`**: `source .venv/bin/activate` before any commands
- **Check logs first**: Use `ctx server logs` for troubleshooting
- **Restart after changes**: `ctx server restart` applies code changes
- Use **modern tools**: uv, pre-commit, pytest
- **Standardize with Makefiles** for consistent development experience
- **Always work in virtual environments**
- **Test everything** with proper async patterns
- **Format automatically** with black and isort
- **Refactor aggressively** to avoid code duplication
- **Use dependency injection** for testable code
- **Create plugin architectures** for extensibility

## Context Server Architecture Notes

### Current State (Post-Simplification)
- **No Redis caching** - Removed for simplicity
- **No expand-context** - Removed broken feature
- **Clean APIs** - Search returns compact responses (~2KB vs 300KB)
- **Separation of concerns**: Search for chunks, `/raw` for full documents
- **Ready for Claude Code/MCP integration**

### Key Files
- `context_server/core/storage.py` - Database operations with metadata filtering
- `context_server/api/search.py` - Search endpoints (vector, fulltext, hybrid)
- `context_server/api/documents.py` - Document management APIs
- `context_server/cli/` - CLI commands and interface

This setup provides a solid foundation for the Context Server project while keeping complexity manageable.

## Project Setup

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing
.tox/
.nox/
.coverage
.pytest_cache/
cover/
htmlcov/
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/

# Linting and type checking
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/
.pytype/
cython_debug/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Output directories
output/
extracted/
temp/
tmp/

# Environment variables
.env.local
.env.*.local

# Database
*.db
*.sqlite
*.sqlite3

# Documentation builds
docs/_build/
site/

# Jupyter Notebook
.ipynb_checkpoints

# Pre-commit
.pre-commit-cache/
```
