#!/usr/bin/env python3
"""Integration test for the Context Server MCP server."""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from context_server.mcp_server.client import ContextServerClient, ContextServerError
from context_server.mcp_server.config import Config
from context_server.mcp_server.tools import ContextServerTools


async def test_basic_functionality():
    """Test basic MCP server functionality."""
    print("🧪 Testing Context Server MCP Integration")
    print("=" * 50)

    # Initialize components
    config = Config()
    client = ContextServerClient(config)
    tools = ContextServerTools(client)

    # Test 1: Health Check
    print("\n1. Testing Context Server connection...")
    try:
        healthy = await client.health_check()
        if healthy:
            print("✅ Context Server is healthy and reachable")
        else:
            print("❌ Context Server is not healthy")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Context Server: {e}")
        print("   Make sure Context Server is running with: ctx server up")
        return False

    # Test 2: List Contexts
    print("\n2. Testing context listing...")
    try:
        contexts = await tools.list_contexts()
        print(f"✅ Found {len(contexts)} existing contexts")
        for ctx in contexts:
            print(f"   - {ctx['name']}: {ctx.get('document_count', 0)} documents")
    except Exception as e:
        print(f"❌ Failed to list contexts: {e}")
        return False

    # Test 3: Create Test Context
    test_context_name = "mcp-test-context"
    print(f"\n3. Testing context creation...")
    try:
        # Try to delete existing test context first
        try:
            await tools.delete_context(test_context_name)
            print(f"   Deleted existing test context")
        except ContextServerError:
            pass  # Context doesn't exist, that's fine

        # Create new test context
        result = await tools.create_context(
            test_context_name, "Test context for MCP integration testing"
        )
        print(f"✅ Created test context: {result['name']}")
        print(f"   ID: {result['id']}")
        print(f"   Created: {result['created_at']}")

    except Exception as e:
        print(f"❌ Failed to create test context: {e}")
        return False

    # Test 4: Get Context Details
    print(f"\n4. Testing context retrieval...")
    try:
        context_info = await tools.get_context(test_context_name)
        print(f"✅ Retrieved context details:")
        print(f"   Name: {context_info['name']}")
        print(f"   Documents: {context_info['document_count']}")
        print(f"   Size: {context_info['size_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ Failed to get context details: {e}")
        return False

    # Test 5: Search (should return no results for empty context)
    print(f"\n5. Testing search functionality...")
    try:
        search_results = await tools.search_context(
            test_context_name, "test query", mode="hybrid", limit=5
        )
        results_count = len(search_results.get("results", []))
        print(
            f"✅ Search completed: {results_count} results (expected 0 for empty context)"
        )

    except Exception as e:
        print(f"❌ Failed to search context: {e}")
        return False

    # Test 6: List Documents
    print(f"\n6. Testing document listing...")
    try:
        documents = await tools.list_documents(test_context_name, limit=10)
        doc_count = len(documents.get("documents", []))
        print(f"✅ Listed documents: {doc_count} found (expected 0 for empty context)")

    except Exception as e:
        print(f"❌ Failed to list documents: {e}")
        return False

    # Test 7: Cleanup
    print(f"\n7. Cleaning up test context...")
    try:
        await tools.delete_context(test_context_name)
        print(f"✅ Deleted test context successfully")

    except Exception as e:
        print(f"❌ Failed to delete test context: {e}")
        # Don't return False here, as the main tests passed

    print("\n" + "=" * 50)
    print("🎉 All basic functionality tests passed!")
    print("   The MCP server is ready for Claude integration.")
    return True


async def test_url_extraction():
    """Test URL extraction if user wants to try it."""
    print("\n" + "=" * 50)
    print("🌐 Optional: Test URL Extraction")
    print("This will create a test context and extract a small documentation site.")

    response = input("\nWould you like to test URL extraction? (y/N): ").strip().lower()
    if response != "y":
        print("Skipping URL extraction test.")
        return True

    # Initialize components
    config = Config()
    client = ContextServerClient(config)
    tools = ContextServerTools(client)

    test_context_name = "mcp-url-test"

    try:
        # Clean up any existing test context
        try:
            await tools.delete_context(test_context_name)
        except ContextServerError:
            pass

        # Create test context
        print(f"\nCreating test context: {test_context_name}")
        await tools.create_context(test_context_name, "Test context for URL extraction")

        # Extract a simple documentation page
        test_url = "https://httpbin.org/json"  # Simple, predictable JSON response
        print(f"\nExtracting URL: {test_url}")
        print("(This is a simple JSON endpoint for testing)")

        extraction_result = await tools.extract_url(
            test_context_name, test_url, max_pages=1
        )

        print(f"✅ URL extraction started:")
        print(f"   Job ID: {extraction_result.get('job_id')}")
        print(f"   Status: {extraction_result.get('status')}")

        # Wait a moment for processing
        print("\nWaiting 5 seconds for processing...")
        await asyncio.sleep(5)

        # Check if documents were created
        documents = await tools.list_documents(test_context_name)
        doc_count = len(documents.get("documents", []))
        print(f"✅ Found {doc_count} documents after extraction")

        if doc_count > 0:
            # Test search
            search_results = await tools.search_context(
                test_context_name, "json", limit=3
            )
            results_count = len(search_results.get("results", []))
            print(f"✅ Search found {results_count} results")

            if results_count > 0:
                # Test getting document content
                first_result = search_results["results"][0]
                doc_id = first_result.get("document_id")
                if doc_id:
                    document = await tools.get_document(test_context_name, doc_id)
                    content_length = len(document.get("content", ""))
                    print(f"✅ Retrieved document: {content_length} characters")

        # Cleanup
        print(f"\nCleaning up test context...")
        await tools.delete_context(test_context_name)
        print(f"✅ Cleanup completed")

        print("\n🎉 URL extraction test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ URL extraction test failed: {e}")
        # Try to clean up
        try:
            await tools.delete_context(test_context_name)
        except:
            pass
        return False


async def main():
    """Main test function."""
    print("Context Server MCP Integration Test")
    print("This will test the MCP server's ability to communicate with Context Server.")
    print("Make sure Context Server is running: ctx server up")

    # Run basic functionality tests
    basic_success = await test_basic_functionality()

    if not basic_success:
        print("\n❌ Basic functionality tests failed.")
        print("   Please check Context Server status and try again.")
        return 1

    # Run optional URL extraction test
    await test_url_extraction()

    print("\n" + "=" * 50)
    print("✨ MCP Integration Test Complete!")
    print("\nNext steps:")
    print("1. Configure Claude to use this MCP server")
    print(
        "2. Test with Claude by asking it to create contexts and extract documentation"
    )
    print(
        "3. Try: 'Can you create a context called my-docs and extract https://example.com?'"
    )

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)
