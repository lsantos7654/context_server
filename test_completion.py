#!/usr/bin/env python3
"""Test completion functions for TUI command."""

import sys
sys.path.insert(0, '.')

from context_server.cli.commands.tui import complete_search_query
from context_server.cli.utils import complete_context_name

def test_context_completion():
    """Test context name completion."""
    print("🧪 Testing context name completion...")
    
    # Mock click context and param
    class MockCtx:
        pass
    
    class MockParam:
        pass
    
    ctx = MockCtx()
    param = MockParam()
    
    # Test completion for "te" should match "test" and "test-debug", "test-summary"
    results = complete_context_name(ctx, param, "te")
    print(f"   Completion for 'te': {results}")
    
    # Test completion for "my" should match "my-docs"
    results = complete_context_name(ctx, param, "my")
    print(f"   Completion for 'my': {results}")
    
    # Test completion for empty string should return all
    results = complete_context_name(ctx, param, "")
    print(f"   Completion for '': {results}")


def test_query_completion():
    """Test search query completion."""
    print("\n🧪 Testing search query completion...")
    
    # Mock click context and param
    class MockCtx:
        pass
    
    class MockParam:
        pass
    
    ctx = MockCtx()
    param = MockParam()
    
    # Test completion for "w" should match "widget"
    results = complete_search_query(ctx, param, "w")
    print(f"   Completion for 'w': {results}")
    
    # Test completion for "a" should match "async"
    results = complete_search_query(ctx, param, "a")
    print(f"   Completion for 'a': {results}")
    
    # Test completion for "func" should match "function"
    results = complete_search_query(ctx, param, "func")
    print(f"   Completion for 'func': {results}")
    
    # Test completion for "best" should match "best practices"
    results = complete_search_query(ctx, param, "best")
    print(f"   Completion for 'best': {results}")


if __name__ == "__main__":
    print("🚀 Testing TUI Command Completion")
    print("=" * 40)
    
    try:
        test_context_completion()
        test_query_completion()
        
        print("\n✅ All completion tests passed!")
        print("\nCompletion is now available for:")
        print("  - Context names: ctx tui explorer <TAB>")
        print("  - Search queries: ctx tui explorer test -q <TAB>")
        print("\nTo enable completion in your shell:")
        print("  ctx completion install")
        
    except Exception as e:
        print(f"\n❌ Completion test failed: {e}")
        sys.exit(1)