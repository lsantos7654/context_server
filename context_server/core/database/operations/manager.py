"""Operations manager - placeholder for full implementation."""

class OperationsManager:
    """Manages helper operations like metadata filtering and result transformation."""
    
    def __init__(self, summarization_service=None):
        self.pool = None  # Will be injected by DatabaseManager
        self.summarization_service = summarization_service
    
    # TODO: Implement all operations from storage.py
    # - filter_metadata_for_search, transform_to_compact_format, transform_code_to_compact_format
    
    def filter_metadata_for_search(self, metadata: dict) -> dict:
        """Placeholder - implement from storage.py"""
        # For now, return simplified metadata to avoid breaking the system
        return {
            "document": metadata.get("document", {}),
            "chunk": metadata.get("chunk", {}),
            "code_snippets": metadata.get("code_snippets", []),
        }
    
    async def transform_to_compact_format(self, results: list[dict], query: str = "", mode: str = "hybrid", execution_time_ms: int = 0) -> dict:
        """Transform full search results to compact MCP format.
        
        This is the single source of truth for transforming search results 
        to the compact format used by both CLI and MCP server.
        """
        compact_results = []
        
        for result in results:
            # Use summary if available, otherwise truncate content
            display_content = result.get("summary", "")
            if not display_content:
                content = result.get("content", "")
                display_content = content[:150] + "..." if len(content) > 150 else content
            # No truncation for summaries - they're already AI-generated and meaningful
            
            # Get code snippets count and IDs from metadata
            metadata = result.get("metadata", {})
            code_snippets = metadata.get("code_snippets", [])
            code_snippets_count = len(code_snippets) if code_snippets else 0
            
            # Extract code snippet metadata for direct access
            code_snippet_ids = []
            if code_snippets:
                for snippet in code_snippets:
                    if isinstance(snippet, dict) and "id" in snippet:
                        snippet_id = snippet["id"]
                        
                        # Use content directly from snippet data if available
                        snippet_content = snippet.get("content", "")
                        
                        if snippet_content:
                            # Calculate lines and chars from available content
                            lines = snippet_content.split('\n')
                            line_count = len([line for line in lines if line.strip()])
                            char_count = len(snippet_content)
                            
                            # Generate preview: show first 8 lines of actual code
                            if line_count <= 8:
                                preview = snippet_content.strip()
                            else:
                                preview_lines = []
                                for line in lines[:8]:
                                    preview_lines.append(line)
                                preview = '\n'.join(preview_lines).strip()
                        else:
                            # Fallback for when no content is available in metadata
                            line_count = 0
                            char_count = 0
                            preview = snippet.get("preview", "")
                        
                        snippet_obj = {
                            "id": snippet_id,
                            "lines": line_count,
                            "chars": char_count,
                            "preview": preview
                        }
                        code_snippet_ids.append(snippet_obj)
            
            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                "summary": display_content,
                "score": result.get("score"),
                "url": result.get("url"),
                "code_snippets_count": code_snippets_count,
                "code_snippet_ids": code_snippet_ids,
                "content_type": result.get("content_type", "chunk"),
            }
            compact_results.append(compact_result)
        
        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": mode,
            "execution_time_ms": execution_time_ms,
            "note": "Content summarized for quick scanning. Use get_document for full content."
        }
    
    def transform_code_to_compact_format(self, results: list[dict], query: str = "", execution_time_ms: int = 0) -> dict:
        """Transform code search results to compact MCP format.
        
        This is the single source of truth for transforming code search results
        to the compact format used by both CLI and MCP server.
        """
        compact_results = []
        
        for result in results:
            # Use content directly since code snippets are already concise
            display_content = result.get("content", "")
            
            # Truncate very long code snippets
            if len(display_content) > 500:
                display_content = display_content[:500] + "..."
            
            # Calculate line count from content
            line_count = len(display_content.split('\n')) if display_content else 0
            
            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                "content": display_content,
                "snippet_type": result.get("snippet_type", "code_block"),
                "score": result.get("score"),
                "url": result.get("url"),
                "line_count": line_count,
                "content_type": "code_snippet",
            }
            compact_results.append(compact_result)
        
        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": "hybrid",
            "execution_time_ms": execution_time_ms,
            "note": "Code search using voyage-code-3 embeddings for enhanced code understanding."
        }


__all__ = ["OperationsManager"]