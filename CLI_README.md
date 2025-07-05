# Context Server CLI

A modern command-line interface for the Context Server documentation RAG system.

## Installation

### From Source (Development)

```bash
# Clone and install in development mode
git clone <repository-url>
cd context_server
source .venv/bin/activate
pip install -e .
```

### From Wheel

```bash
# Install from wheel file
uv add ./dist/context_server-0.1.0-py3-none-any.whl

# Or with pip
pip install ./dist/context_server-0.1.0-py3-none-any.whl
```

## Quick Start

1. **Start the server:**
   ```bash
   context-server server up
   ```

2. **Create a context:**
   ```bash
   context-server context create my-docs "My documentation collection"
   ```

3. **Extract documentation:**
   ```bash
   context-server docs extract https://fastapi.tiangolo.com/ my-docs
   ```

4. **Search documentation:**
   ```bash
   context-server search query "async functions" my-docs
   ```

## Command Groups

### Server Management (`server`)
- `up` - Start Docker services
- `down` - Stop Docker services
- `status` - Check server health
- `logs` - View service logs
- `shell` - Database shell access

### Context Management (`context`)
- `create` - Create new contexts
- `list` - List all contexts
- `info` - Show context details
- `delete` - Delete contexts

### Document Management (`docs`)
- `extract` - Extract from URLs/files/directories
- `list` - List documents in context
- `delete` - Delete documents
- `count` - Count documents

### Search (`search`)
- `query` - Search with different modes
- `interactive` - Interactive search session

### Development (`dev`)
- `init` - Setup development environment
- `test` - Run tests
- `format` - Format code
- `lint` - Run linters
- `quality` - Full quality check

### Examples (`examples`)
- `rust` - Rust std library docs
- `fastapi` - FastAPI framework docs
- `python` - Python std library docs
- `textual` - Textual TUI framework docs
- `django` - Django framework docs
- `custom` - Interactive custom setup

### Shell Completion (`completion`)
- `install` - Install shell completion
- `uninstall` - Remove shell completion
- `status` - Show installation status
- `show` - Display completion script

## Configuration

The CLI supports configuration via:
1. Command line arguments (highest priority)
2. Environment variables (`CONTEXT_SERVER_*`)
3. Config file (`~/.context-server/config.yaml`)
4. Defaults (lowest priority)

View current configuration:
```bash
context-server config
```

## Examples

### Basic Workflow
```bash
# Start server
context-server server up

# Create context
context-server context create rust-docs "Rust documentation"

# Extract documentation from URL
context-server docs extract https://doc.rust-lang.org/std/ rust-docs --max-pages 20

# Extract local directory
context-server docs extract ./my-project my-project --source-type local --include-patterns "*.py" "*.md"

# Search
context-server search query "HashMap" rust-docs --mode hybrid

# List contexts
context-server context list
```

### Development Workflow
```bash
# Initialize development environment
context-server dev init

# Run tests
context-server dev test --coverage

# Format code
context-server dev format

# Full quality check
context-server dev quality
```

### Local Directory Extraction
```bash
# Extract Python project with default patterns
context-server docs extract ./my-python-project my-code --source-type local

# Extract with custom patterns
context-server docs extract ./docs doc-site \
  --include-patterns "*.md" "*.rst" \
  --exclude-patterns "build/*" "_site/*"

# Extract and save locally
context-server docs extract ./source my-backup \
  --output-path ./extracted-docs \
  --include-patterns "*.py" "*.md"
```

### Shell Completion Setup
```bash
# Install completion for your shell
context-server completion install

# Install for specific shell
context-server completion install --shell zsh

# Check installation status
context-server completion status

# Show completion script (for manual setup)
context-server completion show --shell bash
```

The completion system includes several advanced features:

- **Dynamic Context Completion**: Context names are fetched from the server in real-time
- **Command Availability Checks**: Completion scripts only activate if commands are available
- **Smart Filtering**: Completions are filtered based on what you've already typed
- **Multi-shell Support**: Works seamlessly across bash, zsh, and fish shells

Example of context name completion:
```bash
# Type and press TAB to see available contexts
ctx search query "async" <TAB>
# Shows: fastapi  python-std  rust-std  my-docs

ctx context delete <TAB>
# Shows all your existing contexts for safe deletion
```

### Quick Examples
```bash
# Set up Rust documentation
context-server examples rust

# Set up FastAPI documentation
context-server examples fastapi

# Interactive search
context-server search interactive my-docs
```

## Search Modes

- **vector** - Pure semantic similarity search
- **fulltext** - Traditional keyword search
- **hybrid** - Combines vector + fulltext (recommended)

## Output Formats

Most commands support multiple output formats:
- **rich** - Beautiful terminal output (default)
- **table** - Simple table format
- **json** - Machine-readable JSON

Example:
```bash
context-server context list --format json
```

## Short Alias

You can use `ctx` as a short alias for `context-server`:

```bash
ctx server up
ctx context create my-docs
ctx search query "async" my-docs
```

## Getting Help

Help is available throughout the CLI with both `-h` and `--help`:

```bash
# Main help
context-server -h
context-server --help
ctx -h

# Command group help
context-server server -h
context-server docs -h
context-server examples -h

# Individual command help
context-server docs extract -h
context-server search query -h
context-server context create -h
context-server server up -h
```

## Server Requirements

- Docker and Docker Compose
- PostgreSQL with pgvector extension
- OpenAI API key (for embeddings)

## Features

- ✅ Rich terminal output with colors and tables
- ✅ Progress bars for long-running operations
- ✅ Multiple search modes (vector, fulltext, hybrid)
- ✅ Context isolation for organizing documentation
- ✅ Comprehensive development commands
- ✅ Pre-configured examples for popular frameworks
- ✅ Interactive search sessions with real-time results
- ✅ Local directory extraction with file patterns
- ✅ Smart shell completion for bash, zsh, and fish
- ✅ Dynamic context name completion from server
- ✅ Robust completion scripts with command availability checks
- ✅ Configurable via files and environment variables
- ✅ Cross-platform compatibility
- ✅ Built with modern Python tooling

## Migration from Makefile

If you're migrating from the old Makefile, here are the equivalent commands:

| Makefile | CLI Command |
|----------|-------------|
| `make up` | `context-server server up` |
| `make down` | `context-server server down` |
| `make create-context NAME=x` | `context-server context create x` |
| `make extract URL=x CONTEXT=y` | `context-server docs extract x y` |
| `make search QUERY=x CONTEXT=y` | `context-server search query x y` |
| `make test` | `context-server dev test` |
| `make format` | `context-server dev format` |
| `make lint` | `context-server dev lint` |

## Contributing

The CLI is built with:
- [Click](https://click.palletsprojects.com/) for command-line interface
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [Pydantic](https://docs.pydantic.dev/) for configuration management
- [HTTPX](https://www.python-httpx.org/) for API communication

To contribute:
1. Set up development environment: `context-server dev init`
2. Make your changes
3. Run quality checks: `context-server dev quality`
4. Submit a pull request
