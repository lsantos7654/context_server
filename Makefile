# Context Server - Development Environment & Docker Management

# Project configuration
VENV_NAME := .venv
PYTHON := python3
VENV_ACTIVATE := source $(VENV_NAME)/bin/activate

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

.PHONY: help init test test-watch format lint quality-check clean reset up down restart logs status db-shell db-reset

help: ## Show available commands
	@echo "Context Server - Development Commands"
	@echo ""
	@echo "Environment Setup:"
	@grep -E '^(init|clean|reset):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Testing & Quality:"
	@grep -E '^(test|test-watch|format|lint|quality-check):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Docker Operations:"
	@grep -E '^(up|down|restart|logs|status|db-shell|db-reset):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "For Context/Document management, use the CLI:"
	@echo "  ctx --help              # Show all CLI commands"
	@echo "  ctx context create      # Create contexts"
	@echo "  ctx docs extract        # Extract documentation"
	@echo "  ctx search query        # Search contexts"

# Environment Setup
init: ## Initialize development environment (create venv, install deps)
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

clean: ## Clean up caches and temporary files
	@echo "$(YELLOW)Cleaning up caches...$(NC)"
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@find . -name ".pytest_cache" -delete
	@find . -name ".mypy_cache" -delete
	@find . -name "htmlcov" -delete
	@echo "$(GREEN)Cleanup complete$(NC)"

reset: ## Reset environment (removes venv and recreates)
	@echo "$(YELLOW)Resetting environment...$(NC)"
	@rm -rf $(VENV_NAME)
	@$(MAKE) init

# Testing & Quality
test: ## Run tests with coverage
	$(VENV_ACTIVATE) && pytest tests/ --cov=context_server --cov-report=html --cov-report=term-missing -v

test-watch: ## Run tests in watch mode (for development)
	$(VENV_ACTIVATE) && pytest-watch tests/ -- -v

format: ## Format code with black and isort
	$(VENV_ACTIVATE) && black .
	$(VENV_ACTIVATE) && isort .

lint: ## Run linting (flake8, mypy, bandit)
	$(VENV_ACTIVATE) && flake8 .
	$(VENV_ACTIVATE) && mypy .
	$(VENV_ACTIVATE) && bandit -r context_server/ --skip B101

quality-check: format lint test ## Run full quality check (format, lint, test)

# Docker Operations
up: ## Start Context Server (PostgreSQL + API)
	@echo "$(GREEN)Starting Context Server...$(NC)"
	@docker-compose up -d --build
	@echo "$(GREEN)Server started!$(NC)"
	@echo "API available at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"
	@echo "Database at: localhost:5432"

down: ## Stop Context Server
	@echo "$(YELLOW)Stopping Context Server...$(NC)"
	@docker-compose down

restart: ## Restart Context Server
	@$(MAKE) down && $(MAKE) up

logs: ## Show API server logs
	@docker-compose logs -f api

status: ## Check server status
	@echo "$(GREEN)Checking server status...$(NC)"
	@curl -s http://localhost:8000/health | jq . || echo "Server not running (install jq for formatted output)"

# Database Operations
db-shell: ## Connect to PostgreSQL shell
	@docker-compose exec postgres psql -U context_user -d context_server

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
