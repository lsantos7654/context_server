# Context Server Refactoring TODO List

*Generated from comprehensive code audit - Last updated: 2025-01-07*

This document tracks the refactoring tasks identified during the codebase audit. Tasks are organized by priority and completion status.

## Progress Summary

- **Total Tasks**: 33
- **Completed**: 12 ✅
- **High Priority Pending**: 2 🔥
- **Medium Priority Pending**: 18 ⚡
- **Low Priority Pending**: 3 💡

---

## 🔥 High Priority Tasks (4 pending)

### Code Cleanup & Legacy Removal

- [x] **cleanup-legacy-code**: Delete legacy src/ directory and associated test files
- [x] **cleanup-build-artifacts**: Delete build artifacts: htmlcov/, dist/, output/, logs/ directories
- [x] **cleanup-documentation**: Delete duplicate documentation files: CLI_README.md, CLAUDE_CODE_SETUP.md, MCP_IMPLEMENTATION_SUMMARY.md, claude-workflow-analysis.md, notes.md
- [x] **remove-redis-infrastructure**: Remove Redis from docker-compose.yml, pyproject.toml dependencies, and environment files

### Code Duplication Elimination

- [x] **extract-http-utilities**: Create shared HTTP request utility in cli/utils.py to eliminate duplication across CLI commands
- [x] **extract-context-completion**: Move context name completion function to shared cli/utils.py
- [x] **create-api-error-decorator**: Create API error handling decorator for FastAPI endpoints to standardize error responses

### Code Standards & Architecture

- [x] **audit-cli-consistency**: Audit CLI commands for consistent help text, styling, and remove obsolete options
- [x] **fix-type-hints-builtin**: Update all type hints to use built-in types (dict, list, tuple) instead of typing.Dict, List, Tuple
- [x] **fix-processing-imports**: Remove sys.path.append('/app/src') from processing.py and update imports
- [ ] **enforce-code-standards**: Run pre-commit on all files and fix any linting/formatting issues

### Testing Infrastructure

- [ ] **unit-tests-api**: Write comprehensive unit tests for all API endpoints
- [ ] **unit-tests-cli**: Write unit tests for CLI commands and utilities
- [ ] **integration-tests-api**: Write integration tests for API with database interactions
- [ ] **integration-tests-cli**: Write integration tests for CLI commands with real server interactions

---

## ⚡ Medium Priority Tasks (18 pending)

### Architecture Improvements

- [ ] **consolidate-api-dependencies**: Create api/dependencies.py to consolidate FastAPI dependency injection functions
- [ ] **enhance-table-display**: Create TableBuilder class in cli/output/tables.py to eliminate Rich table duplication
- [ ] **implement-repository-pattern**: Separate repository and service layers in core/storage.py for better architecture
- [ ] **add-structured-logging**: Implement structured logging with structlog throughout the codebase
- [ ] **create-custom-exceptions**: Create specific exception types for different error scenarios

### Configuration & Build

- [x] **update-pyproject-coverage**: Update pyproject.toml coverage source from 'src' to 'context_server'
- [ ] **docker-compose-validation**: Validate docker-compose configuration after Redis removal and ensure all services work correctly

### Quality & Performance

- [ ] **optimize-database-queries**: Review and optimize database queries for potential N+1 issues and add indexes
- [ ] **add-input-validation**: Enhance input validation in API endpoints and CLI commands

### CLI Improvements

- [ ] **cli-help-standardization**: Standardize CLI help text format, examples, and option descriptions across all commands
- [ ] **remove-obsolete-cli-options**: Remove CLI options that no longer make sense (e.g., Redis-related options, expand-context)

### Testing Setup

- [ ] **test-fixtures-creation**: Create comprehensive test fixtures for contexts, documents, and search data
- [ ] **api-test-client-setup**: Set up proper test client with async database for API testing
- [ ] **cli-test-runner-setup**: Set up CLI test runner using Click's testing utilities
- [ ] **mcp-integration-tests**: Write integration tests for MCP server functionality

---

## 💡 Low Priority Tasks (3 pending)

- [ ] **performance-monitoring**: Add performance monitoring and logging for slow operations
- [ ] **update-dockerfile-deps**: Remove any Redis-related dependencies or configurations from Dockerfile

---

## ✅ Completed Tasks (8)

- [x] **simplify-makefile**: Simplify Makefile to focus on environment setup and Docker operations, remove CLI-duplicated features
- [x] **update-makefile-targets**: Update Makefile test-coverage target to use correct source paths
- [x] **cleanup-legacy-code**: Delete legacy src/ directory and associated test files
- [x] **cleanup-build-artifacts**: Delete build artifacts: htmlcov/, dist/, output/, logs/ directories
- [x] **cleanup-documentation**: Delete duplicate documentation files: CLI_README.md, CLAUDE_CODE_SETUP.md, MCP_IMPLEMENTATION_SUMMARY.md, claude-workflow-analysis.md, notes.md
- [x] **remove-redis-infrastructure**: Remove Redis from docker-compose.yml, pyproject.toml dependencies, and environment files
- [x] **fix-type-hints-builtin**: Update all type hints to use built-in types (dict, list, tuple) instead of typing.Dict, List, Tuple
- [x] **fix-processing-imports**: Remove sys.path.append('/app/src') from processing.py and update imports
- [x] **update-pyproject-coverage**: Update pyproject.toml coverage source from 'src' to 'context_server'

---

## Implementation Guidelines

### Before Starting
1. **Always activate virtual environment**: `source .venv/bin/activate`
2. **Follow CLAUDE.md**: Reference @CLAUDE.md for coding standards
3. **Write tests first**: Ensure behavior doesn't change during refactoring

### Development Workflow
```bash
# 1. Start development
source .venv/bin/activate
make up                    # Start services

# 2. Make changes following CLAUDE.md standards
# 3. Test changes
make test                  # Run tests
make quality-check         # Format, lint, test

# 4. After completing tasks
pre-commit run --all-files # Fix all issues
make test                  # Verify no regressions
```

### Priority Execution Order

**Phase 1 - Cleanup (High Priority)**
1. Delete legacy code and artifacts
2. Remove Redis infrastructure
3. Fix import paths and type hints

**Phase 2 - Duplication Elimination (High Priority)**
1. Extract shared utilities
2. Create error handling decorators
3. Standardize CLI patterns

**Phase 3 - Testing Infrastructure (High Priority)**
1. Set up test frameworks
2. Write comprehensive test suites
3. Ensure all functionality is covered

**Phase 4 - Architecture Improvements (Medium Priority)**
1. Implement repository pattern
2. Add structured logging
3. Create custom exceptions

**Phase 5 - Polish & Performance (Low Priority)**
1. Performance monitoring
2. Final optimizations
3. Documentation updates

---

## Notes

- **Redis Removal**: Confirmed Redis is not used in code, safe to remove from infrastructure
- **Legacy src/ Directory**: Contains 1000+ lines of outdated code that needs removal
- **Type Hints**: Audit found extensive use of `typing.Dict/List` instead of `dict/list`
- **Code Duplication**: 6 major patterns identified affecting 200-300 lines
- **File Cleanup**: 500+ files can be safely deleted

## Success Metrics

- [ ] Reduce codebase complexity by 500+ files
- [ ] Eliminate 300+ lines of duplicated code
- [ ] Achieve 90%+ test coverage
- [ ] All pre-commit hooks pass
- [ ] Zero linting/type checking errors
- [ ] All TODO comments resolved
