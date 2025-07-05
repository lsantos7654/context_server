"""Content analysis for intelligent metadata extraction and classification."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """Represents a code block found in content."""

    language: str
    content: str
    start_line: int
    end_line: int
    functions: List[str]
    classes: List[str]
    imports: List[str]


@dataclass
class ContentAnalysis:
    """Complete analysis of page content."""

    content_type: str
    primary_language: Optional[str]
    summary: str
    code_percentage: float
    code_blocks: List[CodeBlock]
    detected_patterns: Dict[str, List[str]]
    key_concepts: List[str]
    api_references: List[str]


class ContentAnalyzer:
    """Analyzes content to extract metadata for LLM optimization."""

    def __init__(self):
        # Content type patterns
        self.content_type_patterns = {
            "tutorial": [
                r"(?i)\b(tutorial|walkthrough|guide|how to|step by step)\b",
                r"(?i)\b(getting started|quick start|introduction)\b",
                r"(?i)\b(learn|learning|example|examples)\b",
            ],
            "api_reference": [
                r"(?i)\b(api|reference|documentation|docs)\b",
                r"(?i)\b(function|method|class|interface)\b",
                r"(?i)\b(parameters?|arguments?|returns?)\b",
                r"(?i)\b(endpoint|route|path)\b",
            ],
            "code_example": [
                r"(?i)\b(example|sample|demo|snippet)\b",
                r"(?i)\b(implementation|usage|use case)\b",
                r"```[\s\S]*?```",  # Code blocks
                r"(?i)\b(see also|related|similar)\b",
            ],
            "troubleshooting": [
                r"(?i)\b(troubleshoot|debug|error|problem|issue|fix)\b",
                r"(?i)\b(common problems|faq|frequently asked)\b",
                r"(?i)\b(solution|workaround|resolve)\b",
            ],
            "concept_explanation": [
                r"(?i)\b(concept|theory|principle|overview)\b",
                r"(?i)\b(understand|explanation|definition)\b",
                r"(?i)\b(architecture|design|pattern)\b",
            ],
        }

        # Programming language patterns
        self.language_patterns = {
            "python": [
                r"\bimport\s+\w+",
                r"\bfrom\s+\w+\s+import",
                r"\bdef\s+\w+\s*\(",
                r"\bclass\s+\w+\s*:",
                r"\.py\b",
                r"python",
                r"pip\s+install",
            ],
            "javascript": [
                r"\bfunction\s+\w+\s*\(",
                r"\bconst\s+\w+\s*=",
                r"\blet\s+\w+\s*=",
                r"\bvar\s+\w+\s*=",
                r"\.js\b",
                r"npm\s+install",
                r"require\s*\(",
                r"import\s+.*\s+from",
            ],
            "typescript": [
                r"\binterface\s+\w+",
                r"\btype\s+\w+\s*=",
                r"\.ts\b",
                r"\.tsx\b",
                r"typescript",
                r":\s*\w+\s*[=;,)]",  # Type annotations
            ],
            "java": [
                r"\bpublic\s+class\s+\w+",
                r"\bprivate\s+\w+",
                r"\bpublic\s+static\s+void\s+main",
                r"\.java\b",
                r"import\s+java\.",
            ],
            "go": [
                r"\bpackage\s+\w+",
                r"\bfunc\s+\w+\s*\(",
                r"\.go\b",
                r"go\s+mod",
                r"go\s+get",
            ],
            "rust": [
                r"\bfn\s+\w+\s*\(",
                r"\bstruct\s+\w+",
                r"\bimpl\s+\w+",
                r"\.rs\b",
                r"cargo\s+",
            ],
            "c": [r"#include\s*<", r"\bint\s+main\s*\(", r"\.c\b", r"\.h\b"],
            "cpp": [
                r"#include\s*<",
                r"\bnamespace\s+\w+",
                r"\.cpp\b",
                r"\.hpp\b",
                r"std::",
            ],
        }

        # API and function patterns
        self.api_patterns = [
            r"\b\w+\.\w+\s*\(",  # method calls
            r"\b\w+\s*\([^)]*\)\s*{",  # function definitions
            r"@\w+",  # decorators/annotations
            r"/api/\w+",  # API endpoints
            r"https?://[^\s]+/api/",  # API URLs
        ]

    def analyze_content(self, content: str) -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        try:
            # Extract code blocks first
            code_blocks = self._extract_code_blocks(content)

            # Calculate code percentage
            code_percentage = self._calculate_code_percentage(content, code_blocks)

            # Classify content type
            content_type = self._classify_content_type(content, code_percentage)

            # Detect primary programming language
            primary_language = self._detect_primary_language(content, code_blocks)

            # Generate summary
            summary = self._generate_summary(content, content_type)

            # Extract patterns and concepts
            detected_patterns = self._extract_programming_patterns(content, code_blocks)
            key_concepts = self._extract_key_concepts(content)
            api_references = self._extract_api_references(content)

            return ContentAnalysis(
                content_type=content_type,
                primary_language=primary_language,
                summary=summary,
                code_percentage=code_percentage,
                code_blocks=code_blocks,
                detected_patterns=detected_patterns,
                key_concepts=key_concepts,
                api_references=api_references,
            )

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            # Return minimal analysis on failure
            return ContentAnalysis(
                content_type="unknown",
                primary_language=None,
                summary="Content analysis unavailable",
                code_percentage=0.0,
                code_blocks=[],
                detected_patterns={},
                key_concepts=[],
                api_references=[],
            )

    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract and analyze code blocks from markdown content."""
        code_blocks = []

        # Find markdown code blocks
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "unknown"
            code_content = match.group(2).strip()

            # Calculate line positions
            start_pos = match.start()
            lines_before = content[:start_pos].count("\n")
            lines_in_block = code_content.count("\n")

            # Extract code elements
            functions = self._extract_functions(code_content, language)
            classes = self._extract_classes(code_content, language)
            imports = self._extract_imports(code_content, language)

            code_block = CodeBlock(
                language=language.lower(),
                content=code_content,
                start_line=lines_before + 1,
                end_line=lines_before + lines_in_block + 3,  # +3 for ``` lines
                functions=functions,
                classes=classes,
                imports=imports,
            )
            code_blocks.append(code_block)

        return code_blocks

    def _extract_functions(self, code: str, language: str) -> List[str]:
        """Extract function names from code."""
        functions = []
        language = language.lower()

        if language in ["python", "py"]:
            functions.extend(re.findall(r"\bdef\s+(\w+)\s*\(", code))
        elif language in ["javascript", "js", "typescript", "ts"]:
            functions.extend(re.findall(r"\bfunction\s+(\w+)\s*\(", code))
            functions.extend(re.findall(r"\b(\w+)\s*=\s*\([^)]*\)\s*=>", code))
            functions.extend(re.findall(r"\bconst\s+(\w+)\s*=\s*\([^)]*\)\s*=>", code))
        elif language == "java":
            functions.extend(
                re.findall(r"\b(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(", code)
            )
        elif language == "go":
            functions.extend(re.findall(r"\bfunc\s+(\w+)\s*\(", code))
        elif language == "rust":
            functions.extend(re.findall(r"\bfn\s+(\w+)\s*\(", code))

        return list(set(functions))  # Remove duplicates

    def _extract_classes(self, code: str, language: str) -> List[str]:
        """Extract class names from code."""
        classes = []
        language = language.lower()

        if language in ["python", "py"]:
            classes.extend(re.findall(r"\bclass\s+(\w+)\s*[:(]", code))
        elif language in ["javascript", "js", "typescript", "ts"]:
            classes.extend(re.findall(r"\bclass\s+(\w+)\s*[{(]", code))
        elif language == "java":
            classes.extend(re.findall(r"\b(?:public|private)?\s*class\s+(\w+)", code))
        elif language == "rust":
            classes.extend(re.findall(r"\bstruct\s+(\w+)", code))

        return list(set(classes))

    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        language = language.lower()

        if language in ["python", "py"]:
            imports.extend(re.findall(r"\bimport\s+([\w.]+)", code))
            imports.extend(re.findall(r"\bfrom\s+([\w.]+)\s+import", code))
        elif language in ["javascript", "js", "typescript", "ts"]:
            imports.extend(
                re.findall(r"\bimport\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", code)
            )
            imports.extend(re.findall(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]", code))
        elif language == "java":
            imports.extend(re.findall(r"\bimport\s+([\w.]+)", code))
        elif language == "go":
            imports.extend(re.findall(r"\bimport\s+[\"']([^\"']+)[\"']", code))

        return list(set(imports))

    def _calculate_code_percentage(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> float:
        """Calculate what percentage of content is code."""
        if not content.strip():
            return 0.0

        total_chars = len(content)
        code_chars = sum(len(block.content) for block in code_blocks)

        return (code_chars / total_chars) * 100 if total_chars > 0 else 0.0

    def _classify_content_type(self, content: str, code_percentage: float) -> str:
        """Classify the type of content."""
        content_lower = content.lower()

        # Score each content type
        scores = {}
        for content_type, patterns in self.content_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            scores[content_type] = score

        # Adjust scores based on code percentage
        if code_percentage > 30:
            scores["code_example"] += 5
        if code_percentage > 50:
            scores["code_example"] += 10

        # Return the highest scoring type
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return "general"

    def _detect_primary_language(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> Optional[str]:
        """Detect the primary programming language."""
        # First, check code blocks
        if code_blocks:
            languages = [
                block.language for block in code_blocks if block.language != "unknown"
            ]
            if languages:
                # Return most common language in code blocks
                from collections import Counter

                return Counter(languages).most_common(1)[0][0]

        # Fall back to pattern matching
        content_lower = content.lower()
        scores = {}

        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            scores[language] = score

        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return None

    def _generate_summary(self, content: str, content_type: str) -> str:
        """Generate a concise summary of the content."""
        # Clean content for analysis
        lines = content.split("\n")

        # Look for title or header
        title = ""
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
            elif line and len(line) > 10 and not line.startswith("```"):
                title = line[:100]  # First substantial line
                break

        # Extract first paragraph for summary
        paragraphs = content.split("\n\n")
        first_paragraph = ""
        for para in paragraphs:
            para = para.strip()
            if para and not para.startswith("#") and not para.startswith("```"):
                first_paragraph = para[:200]  # Limit length
                break

        # Create summary based on content type
        if content_type == "tutorial":
            prefix = "Tutorial: "
        elif content_type == "api_reference":
            prefix = "API Reference: "
        elif content_type == "code_example":
            prefix = "Code Example: "
        elif content_type == "troubleshooting":
            prefix = "Troubleshooting: "
        else:
            prefix = ""

        summary = prefix + (title or first_paragraph or "Content analysis")
        return summary[:300]  # Limit to 300 characters

    def _extract_programming_patterns(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> Dict[str, List[str]]:
        """Extract programming patterns from content."""
        patterns = {
            "async": [],
            "error_handling": [],
            "data_structures": [],
            "design_patterns": [],
        }

        content_lower = content.lower()

        # Async patterns
        async_patterns = [
            "async",
            "await",
            "promise",
            "callback",
            "asynchronous",
            "coroutine",
            "future",
            "concurrent",
        ]
        for pattern in async_patterns:
            if pattern in content_lower:
                patterns["async"].append(pattern)

        # Error handling
        error_patterns = [
            "try",
            "catch",
            "except",
            "error",
            "exception",
            "throw",
            "raise",
            "handle",
            "panic",
        ]
        for pattern in error_patterns:
            if pattern in content_lower:
                patterns["error_handling"].append(pattern)

        # Data structures
        data_patterns = [
            "array",
            "list",
            "dict",
            "map",
            "set",
            "queue",
            "stack",
            "tree",
            "graph",
            "hash",
        ]
        for pattern in data_patterns:
            if pattern in content_lower:
                patterns["data_structures"].append(pattern)

        # Design patterns
        design_patterns = [
            "singleton",
            "factory",
            "observer",
            "decorator",
            "adapter",
            "strategy",
            "command",
            "mvc",
        ]
        for pattern in design_patterns:
            if pattern in content_lower:
                patterns["design_patterns"].append(pattern)

        # Remove empty categories and duplicates
        return {k: list(set(v)) for k, v in patterns.items() if v}

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts and terms."""
        # Simple concept extraction based on frequency and capitalization
        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", content)

        # Filter common words and get most frequent
        common_words = {
            "The",
            "This",
            "That",
            "With",
            "From",
            "When",
            "Where",
            "What",
            "How",
            "Why",
            "You",
            "Your",
            "Code",
            "Example",
        }

        concepts = [word for word in words if word not in common_words]

        # Count frequency and return top concepts
        from collections import Counter

        concept_counts = Counter(concepts)
        return [concept for concept, count in concept_counts.most_common(10)]

    def _extract_api_references(self, content: str) -> List[str]:
        """Extract API references and function calls."""
        api_refs = []

        for pattern in self.api_patterns:
            matches = re.findall(pattern, content)
            api_refs.extend(matches)

        # Clean and deduplicate
        cleaned_refs = []
        for ref in api_refs:
            ref = ref.strip()
            if len(ref) > 2 and ref not in cleaned_refs:
                cleaned_refs.append(ref)

        return cleaned_refs[:20]  # Limit to top 20
