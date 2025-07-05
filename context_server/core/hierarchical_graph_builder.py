"""Hierarchical Knowledge Graph Builder for Document → Chunk → Code Block relationships."""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

# Type imports only - avoid circular import
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import asyncpg
import numpy as np

from .content_analysis import ContentAnalysis

if TYPE_CHECKING:
    from .enhanced_storage import EnhancedDatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the hierarchical knowledge graph."""

    id: str
    node_type: str  # 'document', 'chunk', 'code_block'
    title: str
    content: str
    summary: Optional[str] = None

    # Hierarchical relationships
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    depth_level: int = 0

    # Content characteristics
    primary_language: Optional[str] = None
    content_type: Optional[str] = None
    complexity_score: float = 0.0
    quality_score: float = 0.0

    # Metadata
    key_concepts: List[str] = None
    extracted_keywords: List[str] = None
    code_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.key_concepts is None:
            self.key_concepts = []
        if self.extracted_keywords is None:
            self.extracted_keywords = []
        if self.code_metadata is None:
            self.code_metadata = {}


@dataclass
class GraphRelationship:
    """Represents a relationship between nodes in the knowledge graph."""

    source_node_id: str
    target_node_id: str
    relationship_type: str
    strength: float
    confidence: float

    discovery_method: str
    supporting_evidence: List[str] = None
    extraction_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.extraction_metadata is None:
            self.extraction_metadata = {}


class HierarchicalGraphBuilder:
    """Builds and manages hierarchical knowledge graphs for document content."""

    def __init__(self, database_manager: "EnhancedDatabaseManager"):
        self.db_manager = database_manager

        # Code extraction patterns
        self.code_patterns = {
            "python": {
                "functions": re.compile(r"def\s+(\w+)\s*\([^)]*\):", re.MULTILINE),
                "classes": re.compile(r"class\s+(\w+)\s*(?:\([^)]*\))?:", re.MULTILINE),
                "imports": re.compile(
                    r"(?:from\s+\S+\s+)?import\s+([^#\n]+)", re.MULTILINE
                ),
            },
            "rust": {
                "functions": re.compile(
                    r"fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)", re.MULTILINE
                ),
                "structs": re.compile(r"struct\s+(\w+)", re.MULTILINE),
                "impls": re.compile(r"impl\s+(?:<[^>]*>)?\s*(\w+)", re.MULTILINE),
                "uses": re.compile(r"use\s+([^;]+);", re.MULTILINE),
            },
            "javascript": {
                "functions": re.compile(
                    r"function\s+(\w+)\s*\([^)]*\)|(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                    re.MULTILINE,
                ),
                "classes": re.compile(r"class\s+(\w+)", re.MULTILINE),
                "imports": re.compile(
                    r'import\s+(?:{[^}]+}|\w+)\s+from\s+[\'"][^\'"]+[\'"]', re.MULTILINE
                ),
            },
        }

    async def initialize_schema(self):
        """Initialize the hierarchical graph schema in the database."""
        try:
            # Read and execute the schema file
            schema_file = (
                "/Users/santos/projects/context_server/hierarchical_graph_schema.sql"
            )
            with open(schema_file, "r") as f:
                schema_sql = f.read()

            async with self.db_manager.pool.acquire() as conn:
                await conn.execute(schema_sql)

            logger.info("Hierarchical graph schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hierarchical graph schema: {e}")
            raise

    async def build_document_hierarchy(
        self,
        document_id: str,
        title: str,
        content: str,
        chunks: List[Dict],
        content_analysis: Optional[ContentAnalysis] = None,
    ) -> Dict[str, Any]:
        """Build complete hierarchical graph for a document."""

        try:
            # Create document node
            doc_node = await self._create_document_node(
                document_id, title, content, content_analysis
            )

            # Create chunk nodes and extract code blocks
            chunk_nodes = []
            code_block_nodes = []

            for chunk_data in chunks:
                chunk_node = await self._create_chunk_node(
                    chunk_data, doc_node.id, content_analysis
                )
                chunk_nodes.append(chunk_node)

                # Extract code blocks from chunk
                code_blocks = await self._extract_code_blocks_from_chunk(
                    chunk_data, chunk_node.id
                )
                code_block_nodes.extend(code_blocks)

            # Build containment relationships
            containment_rels = await self._build_containment_relationships(
                doc_node, chunk_nodes, code_block_nodes
            )

            # Build semantic relationships
            semantic_rels = await self._build_semantic_relationships(
                doc_node, chunk_nodes, code_block_nodes
            )

            # Build implementation relationships (tutorial ↔ code)
            impl_rels = await self._build_implementation_relationships(
                chunk_nodes, code_block_nodes
            )

            # Update graph statistics
            await self._update_graph_statistics()

            result = {
                "document_node": doc_node,
                "chunk_nodes": chunk_nodes,
                "code_block_nodes": code_block_nodes,
                "relationships": {
                    "containment": containment_rels,
                    "semantic": semantic_rels,
                    "implementation": impl_rels,
                },
                "total_nodes": 1 + len(chunk_nodes) + len(code_block_nodes),
                "total_relationships": len(containment_rels)
                + len(semantic_rels)
                + len(impl_rels),
            }

            logger.info(
                f"Built hierarchy for document {title}: "
                f"{result['total_nodes']} nodes, {result['total_relationships']} relationships"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to build document hierarchy for {title}: {e}")
            raise

    async def _create_document_node(
        self,
        document_id: str,
        title: str,
        content: str,
        content_analysis: Optional[ContentAnalysis],
    ) -> GraphNode:
        """Create a document node in the graph."""

        node_id = str(uuid.uuid4())

        # Extract key concepts and keywords
        key_concepts = []
        keywords = []

        if content_analysis:
            key_concepts = content_analysis.key_concepts[:10]
            # Extract keywords from content using simple tokenization
            keywords = self._extract_keywords(content)[:15]

        # Create summary (first paragraph or first 200 chars)
        summary = self._create_summary(content)

        node = GraphNode(
            id=node_id,
            node_type="document",
            title=title,
            content=content,
            summary=summary,
            document_id=document_id,
            depth_level=0,
            primary_language=content_analysis.primary_language
            if content_analysis
            else None,
            content_type=content_analysis.content_type
            if content_analysis
            else "general",
            quality_score=0.8,  # Default quality score
            key_concepts=key_concepts,
            extracted_keywords=keywords,
        )

        # Store in database
        await self._store_node(node)

        return node

    async def _create_chunk_node(
        self,
        chunk_data: Dict,
        parent_node_id: str,
        content_analysis: Optional[ContentAnalysis],
    ) -> GraphNode:
        """Create a chunk node in the graph."""

        node_id = str(uuid.uuid4())
        content = chunk_data.get("content", "")

        # Extract keywords and analyze content
        keywords = self._extract_keywords(content)[:10]
        summary = self._create_summary(content, max_length=100)

        # Determine if chunk contains significant code
        code_percentage = self._calculate_code_percentage(content)

        node = GraphNode(
            id=node_id,
            node_type="chunk",
            title=f"Chunk {chunk_data.get('chunk_index', 'N/A')}",
            content=content,
            summary=summary,
            document_id=chunk_data.get("document_id"),
            chunk_id=chunk_data.get("id"),
            parent_node_id=parent_node_id,
            depth_level=1,
            primary_language=self._detect_chunk_language(content),
            content_type="code" if code_percentage > 30 else "text",
            complexity_score=min(len(content) / 1000.0, 1.0),
            quality_score=0.7,
            extracted_keywords=keywords,
        )

        await self._store_node(node)

        return node

    async def _extract_code_blocks_from_chunk(
        self, chunk_data: Dict, parent_node_id: str
    ) -> List[GraphNode]:
        """Extract code blocks from a chunk and create nodes."""

        content = chunk_data.get("content", "")
        code_blocks = []

        # Find code blocks using various patterns
        code_block_patterns = [
            re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL),  # Markdown code blocks
            re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL),  # HTML code blocks
            re.compile(r"`([^`]+)`"),  # Inline code
        ]

        block_index = 0
        for pattern in code_block_patterns:
            for match in pattern.finditer(content):
                language = match.group(1) if pattern.groups >= 2 else None
                code_content = match.group(2) if pattern.groups >= 2 else match.group(1)

                # Skip very short code snippets
                if len(code_content.strip()) < 10:
                    continue

                # Analyze code structure
                code_metadata = self._analyze_code_structure(code_content, language)

                node_id = str(uuid.uuid4())
                node = GraphNode(
                    id=node_id,
                    node_type="code_block",
                    title=f"Code Block {block_index}",
                    content=code_content.strip(),
                    summary=self._create_code_summary(code_content, language),
                    document_id=chunk_data.get("document_id"),
                    chunk_id=chunk_data.get("id"),
                    parent_node_id=parent_node_id,
                    depth_level=2,
                    primary_language=language
                    or self._detect_chunk_language(code_content),
                    content_type="code",
                    complexity_score=self._calculate_code_complexity(code_content),
                    quality_score=self._calculate_code_quality(code_content),
                    code_metadata=code_metadata,
                )

                await self._store_node(node)
                await self._store_code_block(
                    node, chunk_data.get("id"), match.start(), match.end()
                )

                code_blocks.append(node)
                block_index += 1

        return code_blocks

    async def _build_containment_relationships(
        self,
        doc_node: GraphNode,
        chunk_nodes: List[GraphNode],
        code_block_nodes: List[GraphNode],
    ) -> List[GraphRelationship]:
        """Build hierarchical containment relationships."""

        relationships = []

        # Document contains chunks
        for chunk in chunk_nodes:
            rel = GraphRelationship(
                source_node_id=doc_node.id,
                target_node_id=chunk.id,
                relationship_type="CONTAINS",
                strength=1.0,
                confidence=1.0,
                discovery_method="structural",
                supporting_evidence=["hierarchical_structure"],
            )
            await self._store_relationship(rel)
            relationships.append(rel)

        # Chunks contain code blocks
        for code_block in code_block_nodes:
            parent_chunk = next(
                (c for c in chunk_nodes if c.chunk_id == code_block.chunk_id), None
            )
            if parent_chunk:
                rel = GraphRelationship(
                    source_node_id=parent_chunk.id,
                    target_node_id=code_block.id,
                    relationship_type="CONTAINS",
                    strength=1.0,
                    confidence=1.0,
                    discovery_method="structural",
                    supporting_evidence=["code_extraction"],
                )
                await self._store_relationship(rel)
                relationships.append(rel)

        return relationships

    async def _build_semantic_relationships(
        self,
        doc_node: GraphNode,
        chunk_nodes: List[GraphNode],
        code_block_nodes: List[GraphNode],
    ) -> List[GraphRelationship]:
        """Build semantic similarity relationships using keyword overlap."""

        relationships = []
        all_nodes = [doc_node] + chunk_nodes + code_block_nodes

        # Compare all pairs of nodes for semantic similarity
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes[i + 1 :]:
                # Skip if they're in a containment relationship
                if node1.parent_node_id == node2.id or node2.parent_node_id == node1.id:
                    continue

                # Calculate similarity based on keyword overlap
                similarity = self._calculate_semantic_similarity(node1, node2)

                if similarity > 0.3:  # Threshold for creating relationship
                    rel = GraphRelationship(
                        source_node_id=node1.id,
                        target_node_id=node2.id,
                        relationship_type="SIMILAR_TO",
                        strength=similarity,
                        confidence=min(similarity * 1.2, 1.0),
                        discovery_method="keyword_similarity",
                        supporting_evidence=[
                            f"keyword_overlap_{similarity:.2f}",
                            f"shared_concepts_{len(set(node1.key_concepts) & set(node2.key_concepts))}",
                        ],
                    )
                    await self._store_relationship(rel)
                    relationships.append(rel)

        return relationships

    async def _build_implementation_relationships(
        self, chunk_nodes: List[GraphNode], code_block_nodes: List[GraphNode]
    ) -> List[GraphRelationship]:
        """Build relationships between tutorial text and code implementations."""

        relationships = []

        for chunk in chunk_nodes:
            # Skip chunks that are primarily code themselves
            if chunk.content_type == "code":
                continue

            for code_block in code_block_nodes:
                # Check if chunk explains or references the code block
                explanation_score = self._calculate_explanation_score(chunk, code_block)

                if explanation_score > 0.4:
                    rel = GraphRelationship(
                        source_node_id=chunk.id,
                        target_node_id=code_block.id,
                        relationship_type="EXPLAINS",
                        strength=explanation_score,
                        confidence=explanation_score * 0.8,
                        discovery_method="text_analysis",
                        supporting_evidence=[
                            f"explanation_score_{explanation_score:.2f}",
                            "tutorial_to_code_mapping",
                        ],
                    )
                    await self._store_relationship(rel)
                    relationships.append(rel)

                # Check reverse relationship (code exemplifies concept)
                if explanation_score > 0.5:
                    rel = GraphRelationship(
                        source_node_id=code_block.id,
                        target_node_id=chunk.id,
                        relationship_type="EXEMPLIFIES",
                        strength=explanation_score * 0.9,
                        confidence=explanation_score * 0.7,
                        discovery_method="text_analysis",
                        supporting_evidence=[
                            f"exemplification_score_{explanation_score:.2f}",
                            "code_to_concept_mapping",
                        ],
                    )
                    await self._store_relationship(rel)
                    relationships.append(rel)

        return relationships

    # Helper methods for analysis and storage

    def _extract_keywords(self, content: str, max_keywords: int = 20) -> List[str]:
        """Extract meaningful keywords from content."""
        # Simple keyword extraction - remove stop words and get meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
        }

        # Extract words, filter stop words, and get most frequent
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", content.lower())
        keywords = [w for w in words if w not in stop_words]

        # Count frequency and return top keywords
        from collections import Counter

        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(max_keywords)]

    def _create_summary(self, content: str, max_length: int = 200) -> str:
        """Create a summary of content."""
        # Simple summarization - take first paragraph or truncate
        paragraphs = content.split("\n\n")
        first_para = paragraphs[0].strip()

        if len(first_para) <= max_length:
            return first_para

        # Truncate at word boundary
        words = first_para.split()
        summary = ""
        for word in words:
            if len(summary + word + " ") > max_length:
                break
            summary += word + " "

        return summary.strip() + "..." if summary else content[:max_length] + "..."

    def _calculate_code_percentage(self, content: str) -> float:
        """Calculate percentage of content that looks like code."""
        code_indicators = [
            r"\w+\s*\([^)]*\)",  # function calls
            r"\w+\.\w+",  # method access
            r"[{}();]",  # code punctuation
            r"^\s*\w+\s*=",  # assignments
            r"if\s+\w+",  # conditionals
            r"for\s+\w+",  # loops
            r"import\s+\w+",  # imports
            r"class\s+\w+",  # class definitions
            r"def\s+\w+",  # function definitions
        ]

        total_matches = 0
        for pattern in code_indicators:
            total_matches += len(re.findall(pattern, content, re.MULTILINE))

        # Rough heuristic: more than 1 code indicator per 50 characters suggests code
        lines = len(content.split("\n"))
        if lines == 0:
            return 0.0

        code_density = total_matches / max(lines, 1)
        return min(code_density * 20, 100.0)  # Scale to percentage

    def _detect_chunk_language(self, content: str) -> Optional[str]:
        """Detect programming language in content."""
        language_patterns = {
            "python": [r"def\s+\w+", r"import\s+\w+", r"class\s+\w+", r"if\s+__name__"],
            "rust": [r"fn\s+\w+", r"struct\s+\w+", r"impl\s+\w+", r"use\s+\w+"],
            "javascript": [r"function\s+\w+", r"const\s+\w+", r"let\s+\w+", r"=>\s*{"],
            "java": [r"public\s+class", r"public\s+static", r"import\s+java"],
            "c++": [r"#include", r"std::", r"public:", r"class\s+\w+"],
        }

        scores = {}
        for lang, patterns in language_patterns.items():
            score = sum(
                len(re.findall(pattern, content, re.IGNORECASE)) for pattern in patterns
            )
            if score > 0:
                scores[lang] = score

        return max(scores, key=scores.get) if scores else None

    def _analyze_code_structure(
        self, code_content: str, language: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze code structure and extract functions, classes, etc."""
        if not language or language not in self.code_patterns:
            return {}

        patterns = self.code_patterns[language]
        metadata = {}

        for element_type, pattern in patterns.items():
            matches = pattern.findall(code_content)
            if matches:
                # Flatten nested matches and clean up
                flat_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        flat_matches.extend([m for m in match if m])
                    else:
                        flat_matches.append(match)
                metadata[element_type] = flat_matches[:10]  # Limit to 10 items

        return metadata

    def _create_code_summary(self, code_content: str, language: Optional[str]) -> str:
        """Create a summary for code content."""
        lines = code_content.strip().split("\n")

        # Look for comments that might explain the code
        comments = []
        comment_patterns = [
            r"//\s*(.+)",  # Single line comments
            r"#\s*(.+)",  # Python/bash comments
            r"/\*\s*(.+?)\s*\*/",  # Multi-line comments
        ]

        for pattern in comment_patterns:
            comments.extend(re.findall(pattern, code_content, re.DOTALL))

        if comments:
            return comments[0][:100] + "..." if len(comments[0]) > 100 else comments[0]

        # Fallback: describe code structure
        if language:
            return f"{language.title()} code ({len(lines)} lines)"
        else:
            return f"Code snippet ({len(lines)} lines)"

    def _calculate_code_complexity(self, code_content: str) -> float:
        """Calculate code complexity score."""
        # Simple complexity heuristics
        lines = len(code_content.split("\n"))

        # Count complexity indicators
        complexity_patterns = [
            r"if\s+",  # Conditionals
            r"for\s+",  # Loops
            r"while\s+",  # Loops
            r"try\s+",  # Exception handling
            r"catch\s+",  # Exception handling
            r"match\s+",  # Pattern matching
        ]

        complexity_score = 0
        for pattern in complexity_patterns:
            complexity_score += len(re.findall(pattern, code_content, re.IGNORECASE))

        # Normalize by lines of code
        if lines > 0:
            normalized_complexity = complexity_score / lines
            return min(normalized_complexity * 5, 1.0)  # Scale to 0-1

        return 0.0

    def _calculate_code_quality(self, code_content: str) -> float:
        """Calculate code quality score."""
        # Simple quality heuristics
        lines = code_content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return 0.0

        quality_score = 0.5  # Base score

        # Bonus for comments
        comment_lines = len(
            [line for line in lines if line.strip().startswith(("#", "//", "/*"))]
        )
        if comment_lines > 0:
            quality_score += min(comment_lines / len(non_empty_lines), 0.3)

        # Bonus for consistent indentation
        indents = [
            len(line) - len(line.lstrip()) for line in non_empty_lines if line.strip()
        ]
        if indents and len(set(indents)) <= 3:  # Consistent indentation levels
            quality_score += 0.1

        # Penalty for very long lines
        long_lines = len([line for line in lines if len(line) > 120])
        if long_lines > 0:
            quality_score -= min(long_lines / len(non_empty_lines), 0.2)

        return max(0.0, min(quality_score, 1.0))

    def _calculate_semantic_similarity(
        self, node1: GraphNode, node2: GraphNode
    ) -> float:
        """Calculate semantic similarity between two nodes."""
        # Keyword overlap
        keywords1 = set(node1.extracted_keywords)
        keywords2 = set(node2.extracted_keywords)

        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        keyword_similarity = len(intersection) / len(union) if union else 0.0

        # Concept overlap
        concepts1 = set(node1.key_concepts)
        concepts2 = set(node2.key_concepts)

        concept_similarity = 0.0
        if concepts1 and concepts2:
            concept_intersection = concepts1 & concepts2
            concept_union = concepts1 | concepts2
            concept_similarity = len(concept_intersection) / len(concept_union)

        # Combine similarities
        return (keyword_similarity * 0.7) + (concept_similarity * 0.3)

    def _calculate_explanation_score(
        self, text_node: GraphNode, code_node: GraphNode
    ) -> float:
        """Calculate how well a text node explains a code node."""
        text_content = text_node.content.lower()

        # Check if text contains references to code elements
        score = 0.0

        # Check for code keywords in text
        if code_node.code_metadata:
            for element_type, elements in code_node.code_metadata.items():
                for element in elements[:5]:  # Check first 5 elements
                    if element.lower() in text_content:
                        score += 0.2

        # Check for programming language mentions
        if (
            code_node.primary_language
            and code_node.primary_language.lower() in text_content
        ):
            score += 0.1

        # Check for common tutorial/explanation words near code references
        explanation_words = [
            "example",
            "function",
            "method",
            "class",
            "implement",
            "code",
            "snippet",
        ]
        for word in explanation_words:
            if word in text_content:
                score += 0.05

        return min(score, 1.0)

    async def _store_node(self, node: GraphNode):
        """Store a graph node in the database."""
        async with self.db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes (
                    id, node_type, document_id, chunk_id, title, content, summary,
                    depth_level, parent_node_id, primary_language, content_type,
                    complexity_score, quality_score, key_concepts, extracted_keywords,
                    code_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                node.id,
                node.node_type,
                node.document_id,
                node.chunk_id,
                node.title,
                node.content,
                node.summary,
                node.depth_level,
                node.parent_node_id,
                node.primary_language,
                node.content_type,
                node.complexity_score,
                node.quality_score,
                json.dumps(node.key_concepts),
                json.dumps(node.extracted_keywords),
                json.dumps(node.code_metadata),
            )

    async def _store_relationship(self, relationship: GraphRelationship):
        """Store a graph relationship in the database."""
        async with self.db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_relationships (
                    source_node_id, target_node_id, relationship_type, strength, confidence,
                    discovery_method, supporting_evidence, extraction_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (source_node_id, target_node_id, relationship_type)
                DO UPDATE SET
                    strength = EXCLUDED.strength,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                """,
                relationship.source_node_id,
                relationship.target_node_id,
                relationship.relationship_type,
                relationship.strength,
                relationship.confidence,
                relationship.discovery_method,
                json.dumps(relationship.supporting_evidence),
                json.dumps(relationship.extraction_metadata),
            )

    async def _store_code_block(
        self, node: GraphNode, chunk_id: str, start_pos: int, end_pos: int
    ):
        """Store code block details in the database."""
        async with self.db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO code_blocks (
                    node_id, chunk_id, code_content, language, functions, classes,
                    imports, variables, char_start, char_end, complexity_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                node.id,
                chunk_id,
                node.content,
                node.primary_language,
                json.dumps(node.code_metadata.get("functions", [])),
                json.dumps(node.code_metadata.get("classes", [])),
                json.dumps(node.code_metadata.get("imports", [])),
                json.dumps(node.code_metadata.get("variables", [])),
                start_pos,
                end_pos,
                node.complexity_score,
            )

    async def _update_graph_statistics(self):
        """Update graph statistics in the database."""
        async with self.db_manager.pool.acquire() as conn:
            await conn.execute("SELECT update_hierarchical_graph_stats()")

    async def get_node_hierarchy(self, document_id: str) -> Dict[str, Any]:
        """Get the complete hierarchy for a document."""
        async with self.db_manager.pool.acquire() as conn:
            # Get document node
            doc_node = await conn.fetchrow(
                "SELECT * FROM graph_nodes WHERE document_id = $1 AND node_type = 'document'",
                document_id,
            )

            if not doc_node:
                return {"error": "Document not found in graph"}

            # Get all child nodes
            children = await conn.fetch(
                "SELECT * FROM get_node_children($1)", doc_node["id"]
            )

            # Get relationships
            relationships = await conn.fetch(
                """
                SELECT r.*, rt.description, rt.search_boost
                FROM graph_relationships r
                JOIN relationship_types rt ON r.relationship_type = rt.type_name
                WHERE r.source_node_id = $1 OR r.target_node_id = $1
                ORDER BY r.strength DESC
                """,
                doc_node["id"],
            )

            return {
                "document": dict(doc_node),
                "children": [dict(child) for child in children],
                "relationships": [dict(rel) for rel in relationships],
                "total_nodes": 1 + len(children),
                "total_relationships": len(relationships),
            }

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        async with self.db_manager.pool.acquire() as conn:
            # Get latest statistics
            stats = await conn.fetchrow(
                "SELECT * FROM hierarchical_graph_stats ORDER BY created_at DESC LIMIT 1"
            )

            # Get relationship summary
            rel_summary = await conn.fetch("SELECT * FROM relationship_summary")

            return {
                "statistics": dict(stats) if stats else {},
                "relationship_summary": [dict(rel) for rel in rel_summary],
            }
