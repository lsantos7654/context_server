# Context Server - Simple Development Makefile

# Project configuration
VENV_NAME := .venv
PYTHON := python3
VENV_ACTIVATE := source $(VENV_NAME)/bin/activate

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

.PHONY: help init test format lint quality-check clean

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

test: ## Run tests
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
