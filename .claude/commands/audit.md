# Python Project Code Audit Template

## Instructions for Claude Code

Please perform a comprehensive audit of this Python codebase using this template. Focus on identifying opportunities for improvement, code duplication, and architectural simplifications.

## 1. Codebase Overview Analysis

### Current Structure Assessment
- [ ] Map out the current file structure and dependencies
- [ ] Identify the main components and their responsibilities
- [ ] Document the data flow between components
- [ ] List all external dependencies and their usage

### Questions to Answer:
1. What is the overall architecture pattern being used?
2. Are there clear separations between layers (API, business logic, data)?
3. Which components have the most dependencies?
4. Are there any circular dependencies?

## 2. Code Duplication Analysis

### Search for Duplication Patterns
- [ ] **Similar Functions**: Look for functions with similar logic but different names
- [ ] **Copy-Paste Code**: Identify code blocks that appear in multiple files
- [ ] **Similar Classes**: Find classes with overlapping responsibilities
- [ ] **Configuration Duplication**: Check for repeated configuration patterns

### Specific Areas to Check:
```python
# Example patterns to look for:
# 1. Error handling patterns
try:
    # operation
except Exception as e:
    logger.error(f"Failed to X: {e}")
    # similar error handling in multiple places

# 2. Database connection patterns
async with get_db_session() as session:
    # similar database operations

# 3. Validation patterns
if not input_value:
    raise ValueError("Input required")
# repeated validation logic

# 4. Processing patterns
def process_X(data):
    cleaned = clean_data(data)
    validated = validate_data(cleaned)
    return transform_data(validated)
# similar processing pipelines
```

### Questions to Answer:
1. What code patterns are repeated 3+ times?
2. Which utility functions could be extracted?
3. Are there similar but not identical implementations?
4. What validation logic is duplicated?

## 3. Architecture Improvement Opportunities

### Current vs. Ideal Architecture
- [ ] **Single Responsibility**: Do classes/functions have one clear purpose?
- [ ] **Dependency Inversion**: Are dependencies injected rather than hardcoded?
- [ ] **Interface Segregation**: Are interfaces focused and minimal?
- [ ] **Open/Closed**: Can new features be added without modifying existing code?

### Specific Improvements to Consider:
```python
# 1. Extract interfaces/protocols
class DocumentProcessor(Protocol):
    def process(self, content: bytes) -> ProcessedDocument: ...

# 2. Use dependency injection
class Service:
    def __init__(self, repository: Repository, logger: Logger):
        self.repository = repository
        self.logger = logger

# 3. Create factory patterns
class ProcessorFactory:
    def create_processor(self, file_type: str) -> DocumentProcessor: ...

# 4. Implement strategy patterns
class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str) -> list[Result]: ...
```

### Questions to Answer:
1. Which components violate single responsibility principle?
2. Where can we introduce better abstractions?
3. What hard dependencies should be injected?
4. Which classes are doing too much?

## 4. Performance and Scalability Issues

### Resource Usage Analysis
- [ ] **Memory Usage**: Identify operations that load large amounts of data
- [ ] **Database Queries**: Look for N+1 queries or missing indexes
- [ ] **Blocking Operations**: Find synchronous operations that should be async
- [ ] **Caching Opportunities**: Identify expensive operations that could be cached

### Specific Areas to Check:
```python
# 1. Large data loading
documents = []
for doc_id in doc_ids:
    doc = await get_document(doc_id)  # N+1 query
    documents.append(doc)

# 2. Missing batch operations
for document in documents:
    await process_document(document)  # Should be batched

# 3. No caching
def expensive_operation(input_data):
    # Heavy computation without caching
    pass

# 4. Synchronous file operations
with open(file_path, 'r') as f:  # Should be async
    content = f.read()
```

### Questions to Answer:
1. What operations are potential bottlenecks?
2. Where can we implement batch processing?
3. What expensive operations should be cached?
4. Are there memory leaks or excessive memory usage?

## 5. Error Handling and Logging

### Error Handling Patterns
- [ ] **Consistent Error Types**: Are custom exceptions used appropriately?
- [ ] **Error Propagation**: Are errors handled at the right level?
- [ ] **Recovery Strategies**: Are there retry mechanisms where appropriate?
- [ ] **User-Friendly Messages**: Are error messages helpful for debugging?

### Logging Quality
- [ ] **Structured Logging**: Is structured logging used consistently?
- [ ] **Log Levels**: Are appropriate log levels used?
- [ ] **Context Information**: Do logs include relevant context?
- [ ] **Performance Impact**: Are there too many/expensive log statements?

### Questions to Answer:
1. Where is error handling inconsistent?
2. What errors are swallowed without proper handling?
3. Where do we need better error context?
4. Are there missing error scenarios?

## 6. Configuration and Settings Management

### Configuration Analysis
- [ ] **Centralized Config**: Is configuration managed in one place?
- [ ] **Environment-Specific**: Can settings be easily changed per environment?
- [ ] **Validation**: Are configuration values validated?
- [ ] **Documentation**: Are configuration options documented?

### Questions to Answer:
1. Where is configuration scattered throughout the code?
2. What hardcoded values should be configurable?
3. Are there missing validation for configuration?
4. How can configuration be simplified?

## 7. Testing Coverage and Quality

### Test Analysis
- [ ] **Coverage Gaps**: What code lacks test coverage?
- [ ] **Test Quality**: Are tests testing behavior or implementation?
- [ ] **Test Organization**: Are tests well-organized and maintainable?
- [ ] **Integration Tests**: Are there sufficient integration tests?

### Questions to Answer:
1. What critical paths are untested?
2. Where do we need more integration tests?
3. What tests are brittle or hard to maintain?
4. Are there missing edge case tests?

## 8. Documentation and Code Clarity

### Code Readability
- [ ] **Naming**: Are variables, functions, and classes well-named?
- [ ] **Comments**: Are complex logic sections documented?
- [ ] **Type Hints**: Are type hints comprehensive and accurate?
- [ ] **Function Size**: Are functions too long or complex?

### Questions to Answer:
1. What code is hard to understand?
2. Where are better names needed?
3. What complex logic needs documentation?
4. Which functions should be broken down?

## 9. Security Considerations

### Security Analysis
- [ ] **Input Validation**: Is user input properly validated?
- [ ] **SQL Injection**: Are there potential SQL injection vulnerabilities?
- [ ] **File Operations**: Are file operations secure?
- [ ] **Error Information**: Do error messages leak sensitive information?

### Questions to Answer:
1. Where is input validation missing or insufficient?
2. What operations could be exploited?
3. Are there information disclosure vulnerabilities?
4. What sensitive data is logged or exposed?

## 10. Recommendations and Action Plan

### Priority Classification
Use this framework to classify findings:

**P0 - Critical (Fix Immediately):**
- Security vulnerabilities
- Data loss risks
- Performance bottlenecks causing failures

**P1 - High (Fix Soon):**
- Significant code duplication
- Major architectural issues
- Error handling gaps

**P2 - Medium (Fix When Convenient):**
- Minor duplication
- Code clarity issues
- Missing tests

**P3 - Low (Nice to Have):**
- Style improvements
- Documentation gaps
- Minor optimizations

### Refactoring Strategy
For each recommendation, provide:
1. **Current State**: What the code looks like now
2. **Target State**: What it should look like
3. **Migration Path**: Steps to safely refactor
4. **Risk Assessment**: What could go wrong
5. **Testing Strategy**: How to verify the refactor

### Example Recommendation Format:
```markdown
#### Issue: Duplicate Processing Logic
**Priority**: P1 - High
**Files Affected**: src/processors/handler_a.py, src/processors/handler_b.py, src/processors/handler_c.py

**Current State**:
Each processor has similar validation and error handling logic.

**Target State**:
Extract common processing logic into a base class or mixin.

**Migration Path**:
1. Create AbstractProcessor base class
2. Extract common methods (validate_input, handle_errors, etc.)
3. Refactor existing processors to inherit from base class
4. Update tests to cover new structure

**Risk Assessment**: Low - mostly moving existing code
**Testing Strategy**: Verify all existing tests pass, add tests for base class
```

## 11. Audit Execution Commands

Run these commands to gather information for the audit:

```bash
# Find duplicate code patterns
grep -r "async def" src/ | cut -d: -f2 | sort | uniq -c | sort -nr

# Find large files that might need splitting
find src/ -name "*.py" -exec wc -l {} + | sort -nr | head -20

# Find complex functions (high cyclomatic complexity)
# Use tools like radon or flake8-complexity

# Find TODO/FIXME comments
grep -r "TODO\|FIXME\|XXX" src/

# Check import dependencies
python -m pydeps src/ --show-deps

# Find unused imports
python -m unimport --check src/

# Find code duplication
python -m duplicated src/
```

## 12. Follow-up Actions

After completing the audit:

1. **Prioritize Issues**: Rank findings by impact and effort
2. **Create Tasks**: Break down refactoring into manageable tasks
3. **Update Documentation**: Reflect architectural decisions
4. **Set up Monitoring**: Add metrics for identified issues
5. **Schedule Reviews**: Plan regular architecture reviews

## Deliverables

Provide the following outputs:

1. **Executive Summary**: High-level findings and recommendations (1-2 pages)
2. **Detailed Analysis**: Section-by-section audit results
3. **Prioritized Action Plan**: Specific tasks with priorities and estimates
4. **Refactoring Guide**: Step-by-step instructions for major changes
5. **Code Examples**: Before/after code snippets for key improvements

Remember: The goal is not just to find problems, but to provide actionable solutions that improve the codebase's maintainability, performance, and reliability.
