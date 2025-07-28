"""CLAUDE.md template for intelligent documentation strategy."""

CLAUDE_MD_DOCUMENTATION_STRATEGY = """

## Intelligent Documentation Gap Detection & Extraction

When you encounter missing information while helping users:

### 1. Detect Documentation Gaps
If search results are insufficient for specific concepts (e.g., "BFSDeepCrawlStrategy", "specific API methods"):

### 2. Intelligent Documentation Discovery
Instead of blindly increasing max-pages, use this systematic approach:

```bash
# 1. Fetch the site's sitemap to understand structure
WebFetch("https://docs.example.com/sitemap.xml", "Extract all documentation URLs and categorize by topic")

# 2. Identify the specific section needed
# From sitemap results, find relevant URLs like:
# - https://docs.crawl4ai.com/core/deep-crawling/
# - https://docs.fastapi.com/advanced/security/
# - https://docs.react.dev/reference/hooks/

# 3. Extract the specific missing documentation
mcp__context-server__extract_url(
    context_name="existing-context-name",
    url="https://docs.example.com/specific/missing/section/",
    max_pages=10  # Focused extraction, not blanket increase
)
```

### 3. Proactive Documentation Management
- **Always check**: Before saying "I can't find information about X"
- **Auto-discover**: Use sitemaps, documentation indexes, or API references
- **Targeted extraction**: Extract specific missing sections, not entire sites again
- **Update contexts**: Add new pages to existing contexts rather than creating duplicates

### 4. Context Optimization Workflow
```
User asks about missing concept
↓
Search existing contexts first
↓
If insufficient: Analyze what's missing specifically  
↓
Use WebFetch to find the right documentation section
↓
Extract targeted pages with extract_url
↓
Re-search with enhanced context
↓
Provide complete answer
```

### 5. Optimal Context Server Usage Patterns

#### Search-First Workflow
1. **search_context** or **search_code** - Get summaries and identify relevant items
2. **get_document** or **get_code_snippet** - Retrieve specific content using IDs from search
3. **get_chunk** - Get detailed chunk content when summaries aren't sufficient

#### Context Management Best Practices
- Create focused contexts per framework/library: `fastapi-docs`, `crawl4ai-docs`, `react-docs`
- Use descriptive names that indicate the scope and purpose
- Extract with targeted approach rather than blanket high page limits
- Re-extract specific sections when documentation is updated or gaps are found

#### Documentation Site Discovery
- Check for `/sitemap.xml` for comprehensive URL lists
- Look for `/api/`, `/reference/`, `/docs/` pattern documentation
- Search for "API Reference", "Documentation Index", or "Table of Contents" pages
- Use domain-specific knowledge (e.g., GitHub repos often have docs in `/docs/` folder)

### 6. Advanced Context Enhancement Techniques

#### Iterative Documentation Building
```bash
# Start with core documentation
mcp__context-server__create_context(name="framework-docs", description="Core framework documentation")
mcp__context-server__extract_url(context_name="framework-docs", url="https://docs.framework.com/", max_pages=20)

# Identify gaps through usage
mcp__context-server__search_context(context_name="framework-docs", query="specific missing concept")

# If gaps found, use WebFetch to find specific sections
WebFetch("https://docs.framework.com/sitemap.xml", "Find URLs for 'specific missing concept'")

# Extract targeted sections
mcp__context-server__extract_url(context_name="framework-docs", url="https://docs.framework.com/advanced/specific-concept/", max_pages=5)
```

#### Multi-Source Documentation Strategy
- Primary source: Official documentation
- Secondary sources: GitHub README files, API references
- Tertiary sources: Community wikis, tutorial sites
- Always prioritize official docs over community content

### 7. Troubleshooting Missing Information  
If you can't find specific information after initial search:

1. **Check extraction completeness**: Verify if the context has sufficient pages for the topic
2. **Try multiple search strategies**: Use different modes (vector, fulltext, hybrid) and synonyms
3. **Analyze sitemap structure**: Use WebFetch to understand the documentation organization
4. **Extract missing sections**: Add specific pages that contain the needed information
5. **Use get_document for full content**: When summaries don't provide enough detail
6. **Cross-reference sources**: Check if information exists elsewhere in the ecosystem

### 8. Context Server Anti-Patterns to Avoid
❌ **Don't use list_documents** - Overwhelming and not actionable
❌ **Don't extract entire sites blindly** - Use targeted approach
❌ **Don't create duplicate contexts** - Enhance existing ones
❌ **Don't ignore sitemaps** - They provide the roadmap to complete documentation
❌ **Don't rely on single search terms** - Try variations and synonyms
"""


def get_claude_md_template() -> str:
    """Get the complete CLAUDE.md documentation strategy template."""
    return CLAUDE_MD_DOCUMENTATION_STRATEGY.strip()


def should_create_claude_md(project_dir: str) -> bool:
    """Check if CLAUDE.md exists in the project directory."""
    from pathlib import Path
    claude_md_path = Path(project_dir) / "CLAUDE.md"
    return not claude_md_path.exists()


def append_to_claude_md(project_dir: str, content: str) -> bool:
    """Append content to existing CLAUDE.md file."""
    from pathlib import Path
    try:
        claude_md_path = Path(project_dir) / "CLAUDE.md"
        with open(claude_md_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + content)
        return True
    except Exception:
        return False


def create_claude_md(project_dir: str) -> bool:
    """Create new CLAUDE.md file with the template content."""
    from pathlib import Path
    try:
        claude_md_path = Path(project_dir) / "CLAUDE.md"
        with open(claude_md_path, "w", encoding="utf-8") as f:
            f.write("# Claude Code Instructions\n\n")
            f.write("This file provides guidance to Claude Code when working with this project.\n")
            f.write(get_claude_md_template())
        return True
    except Exception:
        return False