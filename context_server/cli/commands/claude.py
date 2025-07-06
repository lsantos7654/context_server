"""Claude integration commands for Context Server CLI."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import click
from rich.console import Console
from rich.syntax import Syntax

from ..config import get_api_base_url, get_config
from ..utils import (
    check_api_health,
    check_docker_compose_running,
    confirm_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    run_command,
)

console = Console()


@click.group()
@click.help_option("-h", "--help")
def claude():
    """Claude integration setup and testing.

    Commands for setting up Claude integration with Context Server
    via MCP (Model Context Protocol) server configuration.

    Examples:
        ctx claude install                  # Install MCP configuration
        ctx claude config                   # Show current configuration
        ctx claude test                     # Test MCP server connection
    """
    pass


@claude.command()
@click.option(
    "--claude-config-dir",
    default=None,
    help="Claude configuration directory (auto-detected if not provided)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing configuration files",
)
@click.option(
    "--show-paths",
    is_flag=True,
    help="Show detailed path information during installation",
)
@click.help_option("-h", "--help")
def install(claude_config_dir, overwrite, show_paths):
    """Install Context Server MCP integration for Claude.

    Sets up ~/.config/claude/config.json with Context Server MCP server configuration.
    This is all you need - no background processes or daemons required.
    """
    echo_info("Installing Context Server MCP integration for Claude...")

    # Detect Claude configuration directory
    if claude_config_dir is None:
        claude_config_dir = _detect_claude_config_dir()

    if claude_config_dir is None:
        # Create default Claude config directory
        default_path = Path.home() / ".config" / "claude"
        default_path.mkdir(parents=True, exist_ok=True)
        claude_config_dir = str(default_path)
        echo_info(f"Created Claude configuration directory: {claude_config_dir}")

    claude_config_path = Path(claude_config_dir)
    echo_info(f"Using Claude configuration directory: {claude_config_path}")

    # Get MCP server script path
    mcp_script_path = _get_mcp_script_path()
    if not mcp_script_path:
        echo_error("MCP server script not found")
        echo_info("Make sure Context Server is properly installed")
        return

    if show_paths:
        echo_info(f"Debug: MCP script path resolved to: {mcp_script_path}")
        echo_info(f"Debug: Script exists: {mcp_script_path.exists()}")
        echo_info(f"Debug: Script is file: {mcp_script_path.is_file()}")
        try:
            import context_server

            package_path = Path(context_server.__file__).parent.parent
            echo_info(f"Debug: Context Server package found at: {package_path}")
        except ImportError:
            echo_info("Debug: Context Server package not importable")

    # Create Claude configuration file
    claude_config_file = claude_config_path / "config.json"

    # Load existing configuration or create new
    if claude_config_file.exists() and not overwrite:
        try:
            with open(claude_config_file, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            config = {}

        echo_info(f"Updating existing configuration: {claude_config_file}")
    else:
        config = {}
        echo_info(f"Creating new configuration: {claude_config_file}")

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add Context Server MCP configuration
    config["mcpServers"]["context-server"] = {
        "command": "python",
        "args": [str(mcp_script_path)],
        "env": {"CONTEXT_SERVER_URL": "http://localhost:8000"},
    }

    # Write configuration
    try:
        with open(claude_config_file, "w") as f:
            json.dump(config, f, indent=2)

        echo_success(f"✅ Claude MCP configuration installed: {claude_config_file}")
    except Exception as e:
        echo_error(f"Failed to write configuration: {e}")
        return

    # Show what was configured
    echo_info("Configuration details:")
    echo_info(f"  • MCP Server: context-server")
    echo_info(f"  • Script: {mcp_script_path}")
    echo_info(f"  • Context Server URL: http://localhost:8000")

    # Show next steps
    echo_success("🎉 Installation complete!")
    echo_info("Next steps:")
    echo_info("  1. Start Context Server: ctx server up")
    echo_info("  2. Restart Claude (if running)")
    echo_info("  3. Test: Ask Claude to 'list all contexts'")
    echo_info("")
    echo_info("💡 The MCP server will start automatically when Claude connects")


@claude.command()
@click.help_option("-h", "--help")
def config():
    """Show current Claude and MCP configuration."""
    echo_info("Context Server MCP Configuration")
    echo_info("=" * 40)

    # Show Context Server status
    async def check_server():
        healthy, error = await check_api_health()
        if healthy:
            echo_success("Context Server: Running")
        else:
            echo_error(f"Context Server: Not available ({error})")
        return healthy

    server_running = asyncio.run(check_server())

    # Show MCP server configuration
    mcp_script_path = _get_mcp_script_path()
    if mcp_script_path:
        echo_success(f"MCP Server Script: {mcp_script_path}")
    else:
        echo_error("MCP Server Script: Not found")

    # Show Claude configuration
    claude_config_dir = _detect_claude_config_dir()
    if not claude_config_dir:
        claude_config_dir = str(Path.home() / ".config" / "claude")

    claude_config_file = Path(claude_config_dir) / "config.json"
    if claude_config_file.exists():
        echo_success(f"Claude Configuration: {claude_config_file}")

        # Show MCP server configuration
        try:
            with open(claude_config_file, "r") as f:
                config = json.load(f)

            if "mcpServers" in config and "context-server" in config["mcpServers"]:
                echo_success("Context Server MCP: Configured")
                server_config = config["mcpServers"]["context-server"]
                echo_info(f"  Command: {server_config['command']}")
                echo_info(f"  Script: {server_config['args'][0]}")
                echo_info(
                    f"  URL: {server_config.get('env', {}).get('CONTEXT_SERVER_URL', 'Not set')}"
                )
            else:
                echo_warning("Context Server MCP: Not configured")
                echo_info("Run 'ctx claude install' to set up the configuration")
        except Exception as e:
            echo_error(f"Failed to read Claude configuration: {e}")
    else:
        echo_warning(f"Claude Configuration: Not found at {claude_config_file}")
        echo_info("Run 'ctx claude install' to create the configuration")

    # Show available tools
    echo_info("\nAvailable MCP Tools:")
    tools = [
        "create_context - Create new documentation contexts",
        "extract_url - Extract documentation from websites",
        "search_context - Search with hybrid vector/fulltext search",
        "get_document - Retrieve full document content",
        "get_code_snippet - Get specific code snippets",
        "list_contexts - List all available contexts",
        "list_documents - List documents in a context",
        "delete_context - Remove contexts (use with caution)",
        "And 5 more utility tools...",
    ]

    for tool in tools:
        echo_info(f"  • {tool}")


@claude.command()
@click.help_option("-h", "--help")
def test():
    """Test MCP server connection and functionality."""
    echo_info("Testing Context Server MCP integration...")

    # Test 1: Check if Context Server is running
    echo_info("1. Testing Context Server connection...")

    async def check_server():
        healthy, error = await check_api_health()
        if healthy:
            echo_success("Context Server is running")
            return True
        else:
            echo_error(f"Context Server is not available: {error}")
            echo_info("Start Context Server with: ctx server up")
            return False

    if not asyncio.run(check_server()):
        return

    # Test 2: Check MCP server script
    echo_info("2. Testing MCP server script...")
    mcp_script_path = _get_mcp_script_path()
    if mcp_script_path:
        echo_success(f"MCP server script found: {mcp_script_path}")
    else:
        echo_error("MCP server script not found")
        return

    # Test 3: Test MCP server startup (quick test)
    echo_info("3. Testing MCP server startup...")
    try:
        # Test MCP server functionality directly using the client
        from context_server.mcp_server.client import ContextServerClient
        from context_server.mcp_server.config import Config
        from context_server.mcp_server.tools import ContextServerTools

        config = Config()
        client = ContextServerClient(config)
        tools = ContextServerTools(client)

        # Test basic MCP functionality
        async def test_mcp():
            try:
                # Test health check
                healthy = await client.health_check()
                if not healthy:
                    return False, "Context Server not reachable"

                # Test listing contexts
                contexts = await tools.list_contexts()
                return True, f"MCP tools working - found {len(contexts)} contexts"

            except Exception as e:
                return False, f"MCP test failed: {str(e)}"

        success, message = asyncio.run(test_mcp())
        if success:
            echo_success(f"MCP integration test passed! {message}")
        else:
            echo_error(f"MCP integration test failed: {message}")
            return

    except ImportError as e:
        echo_error(f"MCP modules not available: {e}")
        return
    except Exception as e:
        echo_error(f"Failed to run MCP test: {e}")
        return

    # Test 4: Check Claude configuration
    echo_info("4. Checking Claude configuration...")
    claude_config_dir = _detect_claude_config_dir()
    if claude_config_dir:
        claude_config_file = Path(claude_config_dir) / "claude_desktop_config.json"
        if claude_config_file.exists():
            try:
                with open(claude_config_file, "r") as f:
                    config = json.load(f)

                if "mcpServers" in config and "context-server" in config["mcpServers"]:
                    echo_success("Claude configuration is properly set up")
                else:
                    echo_warning("Context Server MCP not configured in Claude")
                    echo_info("Run 'ctx claude init' to set up the configuration")
            except Exception as e:
                echo_error(f"Failed to read Claude configuration: {e}")
        else:
            echo_warning("Claude configuration file not found")
            echo_info("Run 'ctx claude init' to create the configuration")
    else:
        echo_warning("Claude configuration directory not found")

    echo_success("MCP integration test completed!")
    echo_info("If all tests passed, Claude should be able to use Context Server tools")


def _detect_claude_config_dir() -> Optional[str]:
    """Detect Claude configuration directory."""
    possible_paths = [
        # macOS
        Path.home() / "Library" / "Application Support" / "Claude",
        # Linux
        Path.home() / ".config" / "claude",
        # Windows
        Path.home() / "AppData" / "Roaming" / "Claude",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def _get_mcp_script_path() -> Optional[Path]:
    """Get the MCP server script path, working from any directory.

    Returns:
        Path to start_mcp_server.py if found, None otherwise

    Strategy:
        1. Use installed package location (works anywhere)
        2. Fall back to current directory (development mode)
        3. Return None if not found
    """
    try:
        # Method 1: Use the installed package path (preferred)
        import context_server

        project_root = Path(context_server.__file__).parent.parent
        script_path = project_root / "scripts" / "start_mcp_server.py"

        if script_path.exists() and script_path.is_file():
            return script_path

        # Method 2: Fallback to current working directory (for development)
        fallback_path = Path.cwd() / "scripts" / "start_mcp_server.py"
        if fallback_path.exists() and fallback_path.is_file():
            return fallback_path

        return None

    except ImportError:
        # Method 3: Last resort - try current directory
        fallback_path = Path.cwd() / "scripts" / "start_mcp_server.py"
        if fallback_path.exists() and fallback_path.is_file():
            return fallback_path
        return None


def _create_claude_documentation(
    readme_path: Path, claude_config_file: Path, mcp_script_path: Path
) -> None:
    """Create comprehensive Claude setup documentation."""
    content = f"""# Claude Integration Setup for Context Server

This document provides complete setup instructions for integrating Claude with Context Server using the MCP (Model Context Protocol) server.

## ✅ Setup Complete

Your Claude integration has been automatically configured with the following settings:

- **Claude Configuration**: `{claude_config_file}`
- **MCP Server Script**: `{mcp_script_path}`
- **Context Server URL**: `{get_api_base_url()}`

## 🚀 Getting Started

### 1. Start Context Server

```bash
# Always start Context Server first
ctx server up
```

### 2. Start Claude Desktop

Restart Claude Desktop application to load the new MCP server configuration.

### 3. Verify Integration

In Claude, you should see the Context Server MCP tools available. Test with:

```
Can you create a context called "test-docs" and list all available contexts?
```

## 🛠️ Available Tools

Claude now has access to these Context Server tools:

### Context Management
- `create_context()` - Create new documentation contexts
- `list_contexts()` - List all available contexts
- `get_context()` - Get context details and metadata
- `delete_context()` - Remove contexts (use with caution)

### Document Ingestion
- `extract_url()` - Extract documentation from websites
- `extract_file()` - Process local files (PDF, txt, md, rst)

### Search & Retrieval
- `search_context()` - Hybrid vector/fulltext search
- `get_document()` - Retrieve full document content
- `get_code_snippets()` - Get all code snippets from document
- `get_code_snippet()` - Get specific code snippet by ID

### Utilities
- `list_documents()` - List documents with pagination
- `delete_documents()` - Remove specific documents

## 📋 Example Workflows

### Building a Ratatui Application

```
I want to build a terminal UI application using Ratatui. Can you:

1. Create a context called "ratatui-docs"
2. Extract documentation from https://ratatui.rs/
3. Search for table widget examples
4. Get the code snippets for implementing a table
```

### Managing Documentation

```
Can you:
1. List all my contexts
2. Show me what documents are in my "rust-docs" context
3. Search for "async patterns" in that context
4. Get the full content of the most relevant document
```

## 🔧 Troubleshooting

### Claude doesn't see the MCP tools

1. **Check Context Server is running**:
   ```bash
   ctx server status
   ```

2. **Verify MCP configuration**:
   ```bash
   ctx claude config
   ```

3. **Test MCP integration**:
   ```bash
   ctx claude test
   ```

4. **Restart Claude Desktop** completely

### Connection errors

1. **Check Context Server health**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check MCP server logs**:
   ```bash
   ctx server logs
   ```

3. **Manual MCP server test**:
   ```bash
   ctx claude start
   ```

### Configuration issues

1. **Reconfigure Claude integration**:
   ```bash
   ctx claude init --overwrite
   ```

2. **Check configuration file**:
   ```bash
   cat "{claude_config_file}"
   ```

## 🎯 Advanced Usage

### Autonomous Documentation Management

Claude can now autonomously:
- Create contexts for different projects
- Extract documentation from websites
- Search and discover relevant code examples
- Navigate large documentation sets
- Provide implementation-ready code snippets

### Development Workflow

1. **Project Setup**: Ask Claude to create a context for your project
2. **Documentation Extraction**: Let Claude extract relevant documentation
3. **Code Discovery**: Search for implementation patterns and examples
4. **Implementation**: Get specific code snippets with metadata

## 📚 Resources

- [Context Server Documentation](./README.md)
- [MCP Server Documentation](./README_MCP.md)
- [CLI Commands Reference](./context_server/cli/)

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Run `ctx claude test` for diagnostic information
3. Check Context Server logs with `ctx server logs`
4. Ensure you're using the latest version of both Claude and Context Server

---

*Generated by Context Server CLI v{_get_version()}*
"""

    try:
        with open(readme_path, "w") as f:
            f.write(content)
        echo_success(f"Created documentation: {readme_path}")
    except Exception as e:
        echo_warning(f"Could not create documentation: {e}")


def _get_version() -> str:
    """Get Context Server version."""
    try:
        from context_server import __version__

        return __version__
    except ImportError:
        return "unknown"
