# Context Server - Simple Development Makefile

# Project configuration
VENV_NAME := .venv
PYTHON := python3
VENV_ACTIVATE := source $(VENV_NAME)/bin/activate

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

.PHONY: help init test format lint quality-check clean extract up down logs db-shell

help: ## Show available commands
	@echo "Context Server - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

init: ## Initialize project (create venv, install deps)
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Updating environment...$(NC)"; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
	else \
		echo "$(GREEN)Creating environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_NAME); \
		$(VENV_NAME)/bin/pip install uv; \
		$(VENV_ACTIVATE) && uv pip install -e ".[dev,test]"; \
		$(VENV_ACTIVATE) && pre-commit install; \
	fi
	@echo "$(GREEN)Ready! Run: source $(VENV_NAME)/bin/activate$(NC)"

test: ## Run tests with coverage
	$(VENV_ACTIVATE) && pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-no-cov: ## Run tests without coverage
	$(VENV_ACTIVATE) && pytest tests/ -v

test-watch: ## Run tests in watch mode
	$(VENV_ACTIVATE) && pytest-watch tests/ -- -v

format: ## Format code
	$(VENV_ACTIVATE) && black .
	$(VENV_ACTIVATE) && isort .

lint: ## Run linting
	$(VENV_ACTIVATE) && flake8 .
	$(VENV_ACTIVATE) && mypy .
	$(VENV_ACTIVATE) && bandit -r src/ --skip B101

quality-check: format lint test ## Run full quality check

clean: ## Clean up caches
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name ".pytest_cache" -delete
	find . -name ".mypy_cache" -delete

clean-venv: ## Remove virtual environment
	rm -rf $(VENV_NAME)

reset: clean-venv init ## Reset environment

# Docker commands
up: ## Start the Context Server
	@echo "$(GREEN)Starting Context Server...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Server started!$(NC)"
	@echo "API available at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"
	@echo "Database at: localhost:5432"

down: ## Stop the Context Server
	@echo "$(YELLOW)Stopping Context Server...$(NC)"
	docker-compose down

logs: ## Show server logs
	docker-compose logs -f api

logs-db: ## Show database logs
	docker-compose logs -f postgres

db-shell: ## Connect to PostgreSQL shell
	docker-compose exec postgres psql -U context_user -d context_server

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(YELLOW)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose exec postgres psql -U context_user -d context_server -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public; CREATE EXTENSION IF NOT EXISTS vector;"; \
		echo "$(GREEN)Database reset complete$(NC)"; \
	else \
		echo "Cancelled"; \
	fi

restart: ## Restart the Context Server
	$(MAKE) down && $(MAKE) up

status: ## Show server status
	@curl -s http://localhost:8000/health | jq . || echo "Server not running or jq not installed"

# Context management commands
create-context: ## Create new context (usage: make create-context NAME=my-context DESC="My documentation")
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make create-context NAME=my-context DESC=\"My documentation\""; \
		exit 1; \
	fi
	@curl -X POST http://localhost:8000/api/contexts \
		-H "Content-Type: application/json" \
		-d '{"name": "$(NAME)", "description": "$(DESC)"}' | jq .

list-contexts: ## List all contexts
	@curl -s http://localhost:8000/api/contexts | jq .

delete-context: ## Delete context (usage: make delete-context NAME=my-context)
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make delete-context NAME=my-context"; \
		exit 1; \
	fi
	@curl -X DELETE http://localhost:8000/api/contexts/$(NAME)
	@echo "$(GREEN)Context '$(NAME)' deleted$(NC)"

# Document management commands
extract: ## Extract from URL to context (usage: make extract URL=https://example.com CONTEXT=my-context)
	@if [ -z "$(URL)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make extract URL=https://example.com CONTEXT=my-context [MAX_PAGES=50]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make extract URL=https://ratatui.rs/ CONTEXT=ratatui"; \
		echo "  make extract URL=https://fastapi.tiangolo.com/ CONTEXT=fastapi"; \
		exit 1; \
	fi
	@echo "$(GREEN)Extracting from: $(URL) to context: $(CONTEXT)$(NC)"
	@curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/documents \
		-H "Content-Type: application/json" \
		-d '{"source_type": "url", "source": "$(URL)", "options": {"max_pages": $(or $(MAX_PAGES),50)}}' | jq .

extract-file: ## Extract from file (usage: make extract-file FILE=path/to/file.pdf CONTEXT=my-context)
	@if [ -z "$(FILE)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make extract-file FILE=path/to/file.pdf CONTEXT=my-context"; \
		exit 1; \
	fi
	@echo "$(GREEN)Extracting file: $(FILE) to context: $(CONTEXT)$(NC)"
	@curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/documents \
		-F "file=@$(FILE)" \
		-F "source_type=file" | jq .

list-documents: ## List documents in context (usage: make list-documents CONTEXT=my-context)
	@if [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make list-documents CONTEXT=my-context"; \
		exit 1; \
	fi
	@curl -s http://localhost:8000/api/contexts/$(CONTEXT)/documents | jq .

# Search commands
search: ## Search in context (usage: make search QUERY="rust async" CONTEXT=my-context)
	@if [ -z "$(QUERY)" ] || [ -z "$(CONTEXT)" ]; then \
		echo "Usage: make search QUERY=\"rust async\" CONTEXT=my-context [MODE=hybrid]"; \
		echo ""; \
		echo "Search modes: vector, fulltext, hybrid"; \
		exit 1; \
	fi
	@echo "$(GREEN)Searching for: $(QUERY) in context: $(CONTEXT)$(NC)"
	@curl -X POST http://localhost:8000/api/contexts/$(CONTEXT)/search \
		-H "Content-Type: application/json" \
		-d '{"query": "$(QUERY)", "mode": "$(or $(MODE),hybrid)", "limit": 5}' | jq .

# Example workflows
example-rust: ## Example: scrape Rust std docs
	@echo "$(GREEN)Setting up Rust documentation example...$(NC)"
	$(MAKE) create-context NAME=rust-std DESC="Rust standard library documentation"
	$(MAKE) extract URL=https://doc.rust-lang.org/std/ CONTEXT=rust-std MAX_PAGES=20
	@echo "$(GREEN)Example setup complete! Try: make search QUERY=\"HashMap\" CONTEXT=rust-std$(NC)"

example-fastapi: ## Example: scrape FastAPI docs
	@echo "$(GREEN)Setting up FastAPI documentation example...$(NC)"
	$(MAKE) create-context NAME=fastapi DESC="FastAPI web framework documentation"
	$(MAKE) extract URL=https://fastapi.tiangolo.com/ CONTEXT=fastapi MAX_PAGES=30
	@echo "$(GREEN)Example setup complete! Try: make search QUERY=\"dependency injection\" CONTEXT=fastapi$(NC)"

example-textual: ## Example: scrape Textual docs
	@echo "$(GREEN)Setting up Textual documentation example...$(NC)"
	$(MAKE) create-context NAME=textual DESC="Textual TUI framework documentation"
	$(MAKE) extract URL=https://textual.textualize.io/ CONTEXT=textual MAX_PAGES=25
	@echo "$(GREEN)Example setup complete! Try: make search QUERY=\"widgets\" CONTEXT=textual$(NC)"
