# Python Development Guidelines

## Project Overview

This document outlines coding standards, architectural principles, and development practices for Python projects focused on scalability, maintainability, and developer productivity.

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

## Development Workflow

```bash
# 1. Initial setup (once)
make init

# 2. Daily development
source .venv/bin/activate  # Each session
make test-watch           # Keep running during development

# 3. Before committing
make quality-check

# 4. Common tasks
make test                 # Run tests
make test-coverage        # Test with coverage
make format              # Format code
make lint               # Run linters
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

- Use **modern tools**: uv, pre-commit, pytest
- **Standardize with Makefiles** for consistent development experience
- **Always work in virtual environments** 
- **Test everything** with proper async patterns
- **Format automatically** with black and isort
- **Refactor aggressively** to avoid code duplication
- **Use dependency injection** for testable code
- **Create plugin architectures** for extensibility

This setup provides a solid foundation for any Python project while keeping complexity manageable.