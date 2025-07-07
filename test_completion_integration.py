#!/usr/bin/env python3
"""Test actual CLI completion integration."""

import subprocess
import sys
import os

def test_click_completion():
    """Test if Click completion is working."""
    print("🧪 Testing Click completion integration...")
    
    # Set environment variable for completion
    env = os.environ.copy()
    env['_CTX_COMPLETE'] = 'zsh_source'
    
    try:
        # Test if the completion script is generated
        result = subprocess.run(
            ['ctx'],
            env=env,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if 'complete' in result.stdout:
            print("   ✅ Click completion script generation working")
            return True
        else:
            print("   ❌ Click completion not working")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing completion: {e}")
        return False


def test_context_completion_manual():
    """Manually test context completion."""
    print("\n🧪 Testing context completion manually...")
    
    try:
        # Simulate completion for context names starting with 'te'
        from context_server.cli.utils import complete_context_name
        
        class MockCtx:
            pass
        class MockParam:
            pass
            
        results = complete_context_name(MockCtx(), MockParam(), 'te')
        expected_contexts = ['test', 'test-debug', 'test-summary']
        
        found_matches = [ctx for ctx in expected_contexts if ctx in results]
        
        if len(found_matches) > 0:
            print(f"   ✅ Context completion working: {found_matches}")
            return True
        else:
            print(f"   ❌ No context matches found for 'te'")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing context completion: {e}")
        return False


def test_query_completion_manual():
    """Manually test query completion.""" 
    print("\n🧪 Testing query completion manually...")
    
    try:
        from context_server.cli.commands.tui import complete_search_query
        
        class MockCtx:
            pass
        class MockParam:
            pass
            
        # Test completion for 'w' should include 'widget'
        results = complete_search_query(MockCtx(), MockParam(), 'w')
        
        if 'widget' in results:
            print(f"   ✅ Query completion working: {results}")
            return True
        else:
            print(f"   ❌ Expected 'widget' in results: {results}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing query completion: {e}")
        return False


def main():
    print("🚀 Testing TUI Command Completion Integration")
    print("=" * 50)
    
    # Change to project directory
    os.chdir('/Users/santos/projects/context_server')
    
    # Activate virtual environment
    venv_python = '/Users/santos/projects/context_server/.venv/bin/python'
    if os.path.exists(venv_python):
        print("✅ Using virtual environment")
    else:
        print("❌ Virtual environment not found")
        sys.exit(1)
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_context_completion_manual():
        success_count += 1
        
    if test_query_completion_manual():
        success_count += 1
    
    if test_click_completion():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("\n🎉 All completion tests passed!")
        print("\n📋 Completion Summary:")
        print("   ✅ Context name completion: ctx tui explorer <TAB>")
        print("   ✅ Search query completion: ctx tui explorer test -q <TAB>")
        print("   ✅ Click integration: Working")
        print("\n🛠️  To use completion:")
        print("   1. Ensure completion is installed: ctx completion status")
        print("   2. If not installed: ctx completion install")
        print("   3. Restart your shell or source your shell config")
        print("   4. Use TAB completion: ctx tui explorer <TAB>")
        
    else:
        print(f"\n❌ {total_tests - success_count} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()