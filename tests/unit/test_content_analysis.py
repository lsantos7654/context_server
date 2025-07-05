"""Unit tests for content analysis functionality."""

import pytest

from context_server.core.content_analysis import (
    CodeBlock,
    ContentAnalysis,
    ContentAnalyzer,
)


class TestContentAnalyzer:
    """Test the ContentAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ContentAnalyzer()

    def test_analyze_content_basic(self):
        """Test basic content analysis."""
        content = """
        # Introduction to Python

        This is a tutorial about Python programming.
        You will learn how to write functions and classes.

        ```python
        def hello_world():
            print("Hello, World!")

        class Person:
            def __init__(self, name):
                self.name = name
        ```

        This example shows basic Python syntax.
        """

        analysis = self.analyzer.analyze_content(content)

        assert isinstance(analysis, ContentAnalysis)
        assert analysis.content_type in ["tutorial", "code_example"]
        assert analysis.primary_language == "python"
        assert analysis.code_percentage > 0
        assert len(analysis.code_blocks) == 1
        assert analysis.summary is not None
        assert len(analysis.summary) > 0

    def test_extract_code_blocks_python(self):
        """Test Python code block extraction."""
        content = """
        ```python
        import requests
        from typing import List

        def fetch_data(url: str) -> dict:
            response = requests.get(url)
            return response.json()

        class DataProcessor:
            def process(self, data):
                return data
        ```
        """

        code_blocks = self.analyzer._extract_code_blocks(content)

        assert len(code_blocks) == 1
        block = code_blocks[0]
        assert block.language == "python"
        assert "fetch_data" in block.functions
        assert "DataProcessor" in block.classes
        assert "requests" in block.imports
        assert "typing" in block.imports

    def test_extract_code_blocks_javascript(self):
        """Test JavaScript code block extraction."""
        content = """
        ```javascript
        const express = require('express');

        function startServer(port) {
            const app = express();
            return app.listen(port);
        }

        class ApiClient {
            constructor(baseUrl) {
                this.baseUrl = baseUrl;
            }
        }

        const handleData = (data) => {
            return data.map(item => item.id);
        };
        ```
        """

        code_blocks = self.analyzer._extract_code_blocks(content)

        assert len(code_blocks) == 1
        block = code_blocks[0]
        assert block.language == "javascript"
        assert "startServer" in block.functions
        assert "handleData" in block.functions
        assert "ApiClient" in block.classes
        assert "express" in block.imports

    def test_classify_content_type_tutorial(self):
        """Test tutorial content classification."""
        content = """
        # Getting Started Tutorial

        This walkthrough will guide you through the basics.
        Learn how to use this API step by step.
        """

        code_percentage = 5.0  # Low code content
        content_type = self.analyzer._classify_content_type(content, code_percentage)

        assert content_type == "tutorial"

    def test_classify_content_type_api_reference(self):
        """Test API reference content classification."""
        content = """
        # API Reference

        ## function getData(parameters)

        Returns data from the endpoint.

        Parameters:
        - url: string - The API endpoint
        - options: object - Request options

        Returns: Promise<Response>
        """

        code_percentage = 10.0
        content_type = self.analyzer._classify_content_type(content, code_percentage)

        assert content_type == "api_reference"

    def test_classify_content_type_code_example(self):
        """Test code example content classification."""
        content = """
        # Example Implementation

        ```python
        # This is a large code block
        def complex_function():
            data = fetch_data()
            processed = process_data(data)
            return save_results(processed)

        # Multiple functions and examples
        def another_function():
            pass
        ```

        See the example above for implementation details.
        """

        code_percentage = 60.0  # High code content
        content_type = self.analyzer._classify_content_type(content, code_percentage)

        assert content_type == "code_example"

    def test_detect_primary_language_from_code_blocks(self):
        """Test language detection from code blocks."""
        content = """
        ```python
        def test():
            pass
        ```

        ```python
        class Test:
            pass
        ```

        ```javascript
        function test() {}
        ```
        """

        code_blocks = self.analyzer._extract_code_blocks(content)
        language = self.analyzer._detect_primary_language(content, code_blocks)

        assert language == "python"  # Most common in code blocks

    def test_detect_primary_language_from_patterns(self):
        """Test language detection from content patterns."""
        content = """
        Install the package with npm install.
        Use require() to import modules.
        Define functions with function keyword.
        """

        language = self.analyzer._detect_primary_language(content, [])

        assert language == "javascript"

    def test_calculate_code_percentage(self):
        """Test code percentage calculation."""
        content = "Regular text. ```python\ncode here\n``` More text."
        code_blocks = [
            CodeBlock(
                language="python",
                content="code here",
                start_line=1,
                end_line=3,
                functions=[],
                classes=[],
                imports=[],
            )
        ]

        percentage = self.analyzer._calculate_code_percentage(content, code_blocks)

        expected = (len("code here") / len(content)) * 100
        assert abs(percentage - expected) < 0.1

    def test_extract_programming_patterns(self):
        """Test programming pattern extraction."""
        content = """
        Use async/await for asynchronous operations.
        Handle errors with try/catch blocks.
        Use arrays and maps for data structures.
        Implement the observer pattern.
        """

        patterns = self.analyzer._extract_programming_patterns(content, [])

        assert "async" in patterns.get("async", [])
        assert "try" in patterns.get("error_handling", [])
        assert "array" in patterns.get("data_structures", [])
        assert "observer" in patterns.get("design_patterns", [])

    def test_extract_key_concepts(self):
        """Test key concept extraction."""
        content = """
        This tutorial covers React components and TypeScript.
        Learn about React hooks and TypeScript types.
        React makes building UIs easier.
        """

        concepts = self.analyzer._extract_key_concepts(content)

        assert "React" in concepts
        assert "TypeScript" in concepts

    def test_extract_api_references(self):
        """Test API reference extraction."""
        content = """
        Call getData() to fetch information.
        Use app.listen(3000) to start the server.
        The @route decorator defines endpoints.
        Access /api/users for user data.
        """

        api_refs = self.analyzer._extract_api_references(content)

        # Should find method calls and API endpoints
        assert len(api_refs) > 0
        # Check for some expected patterns
        found_patterns = any(
            "getData" in ref or "listen" in ref or "@route" in ref or "/api/" in ref
            for ref in api_refs
        )
        assert found_patterns

    def test_generate_summary_with_title(self):
        """Test summary generation with title."""
        content = """
        # Python Functions Tutorial

        This comprehensive guide explains how to write
        effective functions in Python. You'll learn about
        parameters, return values, and best practices.
        """

        summary = self.analyzer._generate_summary(content, "tutorial")

        assert "Tutorial:" in summary
        assert "Python Functions" in summary or "comprehensive guide" in summary
        assert len(summary) <= 300

    def test_empty_content_analysis(self):
        """Test analysis of empty content."""
        analysis = self.analyzer.analyze_content("")

        assert analysis.content_type == "general"
        assert analysis.primary_language is None
        assert analysis.code_percentage == 0.0
        assert len(analysis.code_blocks) == 0

    def test_analysis_error_handling(self):
        """Test that analysis gracefully handles errors."""
        # Malformed content that might cause issues
        content = (
            "```python\nunclosed code block\n\n```javascript\nprint('mixed content')\n"
        )

        # Should not raise exception
        analysis = self.analyzer.analyze_content(content)

        assert isinstance(analysis, ContentAnalysis)
        # Should still provide some meaningful data
        assert analysis.content_type is not None
        assert analysis.code_percentage >= 0

    def test_multiple_languages_detection(self):
        """Test detection when multiple languages are present."""
        content = """
        ```python
        def python_func():
            pass
        ```

        ```javascript
        function jsFunc() {}
        ```

        ```typescript
        interface User {
            name: string;
        }
        ```

        This shows Python, JavaScript, and TypeScript examples.
        """

        analysis = self.analyzer.analyze_content(content)

        assert len(analysis.code_blocks) == 3
        # Should detect the first or most prominent language
        assert analysis.primary_language in ["python", "javascript", "typescript"]

        # Check that all languages are detected in code blocks
        languages = [block.language for block in analysis.code_blocks]
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages

    def test_troubleshooting_content_classification(self):
        """Test troubleshooting content classification."""
        content = """
        # Common Problems and Solutions

        If you encounter an error, try these troubleshooting steps:
        1. Debug the connection issue
        2. Fix the configuration problem

        FAQ: How to resolve authentication errors?
        """

        content_type = self.analyzer._classify_content_type(content, 0)

        assert content_type == "troubleshooting"

    def test_concept_explanation_classification(self):
        """Test concept explanation content classification."""
        content = """
        # Understanding Microservices Architecture

        This concept overview explains the principles of
        microservices design patterns. The architectural
        approach involves breaking down applications.
        """

        content_type = self.analyzer._classify_content_type(content, 0)

        assert content_type == "concept_explanation"


class TestCodeBlock:
    """Test the CodeBlock dataclass."""

    def test_code_block_creation(self):
        """Test CodeBlock instantiation."""
        block = CodeBlock(
            language="python",
            content="def test(): pass",
            start_line=5,
            end_line=8,
            functions=["test"],
            classes=[],
            imports=[],
        )

        assert block.language == "python"
        assert block.content == "def test(): pass"
        assert block.start_line == 5
        assert block.end_line == 8
        assert "test" in block.functions
        assert len(block.classes) == 0
        assert len(block.imports) == 0


class TestContentAnalysis:
    """Test the ContentAnalysis dataclass."""

    def test_content_analysis_creation(self):
        """Test ContentAnalysis instantiation."""
        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language="python",
            summary="A Python tutorial",
            code_percentage=25.0,
            code_blocks=[],
            detected_patterns={"async": ["async", "await"]},
            key_concepts=["Python", "Functions"],
            api_references=["getData()"],
        )

        assert analysis.content_type == "tutorial"
        assert analysis.primary_language == "python"
        assert analysis.summary == "A Python tutorial"
        assert analysis.code_percentage == 25.0
        assert len(analysis.code_blocks) == 0
        assert "async" in analysis.detected_patterns
        assert "Python" in analysis.key_concepts
        assert "getData()" in analysis.api_references
