"""Unit tests for LLM-optimized endpoints functionality."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from context_server.core.llm_endpoints import (
    ContentRecommendation,
    ContextRecommendation,
    ContextualAnswer,
    LLMOptimizedEndpoints,
)
from context_server.core.relationship_mapping import (
    ContentRelationship,
    KnowledgeGraph,
    RelationshipType,
    TopicCluster,
)


class TestLLMOptimizedEndpoints:
    """Test LLM-optimized endpoints functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = Mock()
        self.mock_database_manager = Mock()
        self.mock_knowledge_graph_builder = Mock()

        self.endpoints = LLMOptimizedEndpoints(
            self.mock_embedding_service,
            self.mock_database_manager,
            self.mock_knowledge_graph_builder,
        )

    def create_sample_content_item(
        self,
        url: str,
        title: str = "",
        content: str = "",
        content_type: str = "general",
        score: float = 0.8,
    ) -> dict:
        """Create a sample content item for testing."""
        return {
            "url": url,
            "title": title or f"Title for {url}",
            "content": content or f"Content for {url}",
            "content_type": content_type,
            "score": score,
            "summary": f"Summary for {title or url}",
        }

    def create_sample_knowledge_graph(self) -> KnowledgeGraph:
        """Create a sample knowledge graph for testing."""
        relationships = [
            ContentRelationship(
                source_url="http://example.com/tutorial1",
                target_url="http://example.com/tutorial2",
                relationship_type=RelationshipType.TUTORIAL_SEQUENCE,
                strength=0.9,
                confidence=0.8,
                discovered_method="test",
                supporting_evidence=["sequential content"],
                context_elements=["part 1", "part 2"],
                discovery_timestamp=0.0,
            ),
            ContentRelationship(
                source_url="http://example.com/code1",
                target_url="http://example.com/code2",
                relationship_type=RelationshipType.CODE_DEPENDENCY,
                strength=0.8,
                confidence=0.9,
                discovered_method="test",
                supporting_evidence=["import statement"],
                context_elements=["function call"],
                discovery_timestamp=0.0,
            ),
        ]

        clusters = [
            TopicCluster(
                cluster_id="cluster_1",
                name="Python Basics",
                description="Basic Python programming concepts",
                content_urls=[
                    "http://example.com/tutorial1",
                    "http://example.com/tutorial2",
                ],
                topic_keywords=["python", "basics"],
                programming_languages=["python"],
                content_types=["tutorial"],
                difficulty_level="beginner",
                coherence_score=0.9,
                coverage_score=0.8,
                quality_score=0.85,
                related_clusters=[],
                parent_cluster=None,
                child_clusters=[],
            )
        ]

        return KnowledgeGraph(
            relationships=relationships,
            clusters=clusters,
            total_nodes=4,
            total_edges=2,
            cluster_count=1,
            average_node_degree=1.0,
            graph_density=0.3,
            modularity_score=0.8,
            coverage_ratio=0.9,
        )

    @pytest.mark.asyncio
    async def test_analyze_question_how_to(self):
        """Test question analysis for how-to questions."""
        question = "How do I create a Python function?"

        analysis = await self.endpoints._analyze_question(question)

        assert analysis["type"] == "how_to"
        assert analysis["intent"] == "procedural"
        assert analysis["requires_code"] is True
        assert "python" in analysis["entities"]

    @pytest.mark.asyncio
    async def test_analyze_question_what_is(self):
        """Test question analysis for definitional questions."""
        question = "What is a Python class?"

        analysis = await self.endpoints._analyze_question(question)

        assert analysis["type"] == "what_is"
        assert analysis["intent"] == "definitional"
        assert "python" in analysis["entities"]

    @pytest.mark.asyncio
    async def test_analyze_question_troubleshooting(self):
        """Test question analysis for troubleshooting questions."""
        question = "Error in my JavaScript code not working"

        analysis = await self.endpoints._analyze_question(question)

        assert analysis["type"] == "troubleshooting"
        assert analysis["intent"] == "problem_solving"
        assert analysis["requires_code"] is True
        assert "javascript" in analysis["entities"]

    @pytest.mark.asyncio
    async def test_analyze_question_code_example(self):
        """Test question analysis for code example requests."""
        question = "Show me example of React component"

        analysis = await self.endpoints._analyze_question(question)

        assert analysis["type"] == "code_example"
        assert analysis["intent"] == "implementation"
        assert analysis["requires_code"] is True
        assert "react" in analysis["entities"]

    @pytest.mark.asyncio
    async def test_analyze_question_comparison(self):
        """Test question analysis for comparison questions."""
        question = "What's the difference between Python and JavaScript?"

        analysis = await self.endpoints._analyze_question(question)

        assert analysis["type"] == "comparison"
        assert analysis["intent"] == "analytical"
        assert "python" in analysis["entities"]
        assert "javascript" in analysis["entities"]

    def test_extract_entities(self):
        """Test entity extraction from text."""
        text = "How do I use React with TypeScript and Node.js?"

        entities = self.endpoints._extract_entities(text)

        assert "react" in entities
        assert "typescript" in entities
        assert "node.js" in entities or "nodejs" in entities

    @pytest.mark.asyncio
    async def test_find_relevant_content(self):
        """Test finding relevant content using multiple strategies."""
        question = "How to create Python functions?"
        question_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        context_id = "test_context"
        question_analysis = {
            "type": "how_to",
            "intent": "procedural",
            "entities": ["python", "functions"],
            "requires_code": True,
        }

        # Mock database search results
        semantic_results = [
            self.create_sample_content_item(
                "http://example.com/python1",
                "Python Functions",
                content_type="tutorial",
            ),
            self.create_sample_content_item(
                "http://example.com/python2",
                "Function Examples",
                content_type="code_example",
            ),
        ]

        keyword_results = [
            self.create_sample_content_item(
                "http://example.com/python3", "Python Basics", content_type="tutorial"
            )
        ]

        type_results = [
            self.create_sample_content_item(
                "http://example.com/tutorial1",
                "Programming Tutorial",
                content_type="tutorial",
            )
        ]

        self.mock_database_manager.semantic_search = AsyncMock(
            return_value=semantic_results
        )
        self.mock_database_manager.keyword_search = AsyncMock(
            return_value=keyword_results
        )
        self.mock_database_manager.search_by_content_type = AsyncMock(
            return_value=type_results
        )

        results = await self.endpoints._find_relevant_content(
            question, question_embedding, context_id, question_analysis, 10
        )

        assert len(results) > 0
        assert any(item["url"] == "http://example.com/python1" for item in results)
        self.mock_database_manager.semantic_search.assert_called_once()
        self.mock_database_manager.keyword_search.assert_called()

    @pytest.mark.asyncio
    async def test_enhance_with_relationships(self):
        """Test enhancing content with relationship context."""
        content = [
            self.create_sample_content_item(
                "http://example.com/tutorial1", "Tutorial Part 1"
            )
        ]

        knowledge_graph = self.create_sample_knowledge_graph()
        intent = "procedural"

        enhanced = await self.endpoints._enhance_with_relationships(
            content, knowledge_graph, intent
        )

        assert len(enhanced) == 1
        assert "related_content" in enhanced[0]
        # Should find tutorial sequence relationship
        related = enhanced[0]["related_content"]
        assert any(rel["relationship"] == "next_step" for rel in related)

    @pytest.mark.asyncio
    async def test_generate_contextual_answer_how_to(self):
        """Test generating contextual answer for how-to question."""
        question = "How do I create a Python function?"
        relevant_content = [
            self.create_sample_content_item(
                "http://example.com/python1",
                "Python Functions",
                "To create a function in Python, use the def keyword. Step 1: Define function. Step 2: Add parameters.",
                "tutorial",
                0.9,
            ),
            self.create_sample_content_item(
                "http://example.com/python2",
                "Function Examples",
                "```python\ndef hello():\n    print('Hello')\n```",
                "code_example",
                0.8,
            ),
        ]

        question_analysis = {
            "type": "how_to",
            "intent": "procedural",
            "entities": ["python", "function"],
            "requires_code": True,
        }

        answer = await self.endpoints._generate_contextual_answer(
            question, relevant_content, question_analysis, include_code=True
        )

        assert isinstance(answer, ContextualAnswer)
        assert "function" in answer.answer.lower()
        assert len(answer.supporting_sources) == 2
        assert len(answer.code_examples) > 0
        assert answer.confidence > 0.0
        assert len(answer.follow_up_questions) > 0
        assert len(answer.reasoning_chain) > 0

    @pytest.mark.asyncio
    async def test_generate_contextual_answer_what_is(self):
        """Test generating contextual answer for definitional question."""
        question = "What is a Python class?"
        relevant_content = [
            self.create_sample_content_item(
                "http://example.com/python1",
                "Python Classes",
                "A class is a blueprint for creating objects. Classes define attributes and methods.",
                "concept_explanation",
                0.9,
            )
        ]

        question_analysis = {
            "type": "what_is",
            "intent": "definitional",
            "entities": ["python", "class"],
            "requires_code": False,
        }

        answer = await self.endpoints._generate_contextual_answer(
            question, relevant_content, question_analysis, include_code=False
        )

        assert isinstance(answer, ContextualAnswer)
        assert "class" in answer.answer.lower()
        assert "blueprint" in answer.answer.lower()
        assert len(answer.supporting_sources) == 1
        assert answer.confidence > 0.0

    @pytest.mark.asyncio
    async def test_generate_contextual_answer_troubleshooting(self):
        """Test generating contextual answer for troubleshooting question."""
        question = "My Python code has an error"
        relevant_content = [
            self.create_sample_content_item(
                "http://example.com/debug1",
                "Python Debugging",
                "Common solution: Check syntax. Fix indentation errors. Verify variable names.",
                "troubleshooting",
                0.8,
            )
        ]

        question_analysis = {
            "type": "troubleshooting",
            "intent": "problem_solving",
            "entities": ["python", "error"],
            "requires_code": True,
        }

        answer = await self.endpoints._generate_contextual_answer(
            question, relevant_content, question_analysis, include_code=True
        )

        assert isinstance(answer, ContextualAnswer)
        assert "solution" in answer.answer.lower()
        assert len(answer.supporting_sources) == 1
        assert answer.confidence > 0.0

    @pytest.mark.asyncio
    async def test_answer_question_integration(self):
        """Test full answer_question integration."""
        question = "How to create Python functions?"
        context_id = "test_context"

        # Mock embedding service
        self.mock_embedding_service.embed_single = AsyncMock(
            return_value={"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "model": "test-model"}
        )

        # Mock database search
        search_results = [
            self.create_sample_content_item(
                "http://example.com/python1",
                "Python Functions",
                content_type="tutorial",
            )
        ]
        self.mock_database_manager.semantic_search = AsyncMock(
            return_value=search_results
        )
        self.mock_database_manager.keyword_search = AsyncMock(return_value=[])
        self.mock_database_manager.search_by_content_type = AsyncMock(return_value=[])

        # Mock knowledge graph
        knowledge_graph = self.create_sample_knowledge_graph()
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=knowledge_graph
        )

        answer = await self.endpoints.answer_question(question, context_id)

        assert isinstance(answer, ContextualAnswer)
        assert len(answer.answer) > 0
        assert answer.confidence >= 0.0
        self.mock_embedding_service.embed_single.assert_called_once()
        self.mock_knowledge_graph_builder.load_knowledge_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_user_intent_learn(self):
        """Test analyzing user intent for learning."""
        user_query = "I want to learn Python programming"
        user_level = "beginner"

        analysis = await self.endpoints._analyze_user_intent(user_query, user_level)

        assert analysis["intent"] == "learn"
        assert analysis["user_level"] == "beginner"
        assert "python" in analysis["entities"]
        assert "tutorial" in analysis["preferred_content_types"]
        assert analysis["complexity_preference"] == "low"

    @pytest.mark.asyncio
    async def test_analyze_user_intent_implement(self):
        """Test analyzing user intent for implementation."""
        user_query = "I need to build a web application with React"
        user_level = "intermediate"

        analysis = await self.endpoints._analyze_user_intent(user_query, user_level)

        assert analysis["intent"] == "implement"
        assert analysis["user_level"] == "intermediate"
        assert "react" in analysis["entities"]
        assert "code_example" in analysis["preferred_content_types"]
        assert analysis["complexity_preference"] == "medium"

    @pytest.mark.asyncio
    async def test_analyze_user_intent_troubleshoot(self):
        """Test analyzing user intent for troubleshooting."""
        user_query = "I need to fix an error in my JavaScript code"
        user_level = "advanced"

        analysis = await self.endpoints._analyze_user_intent(user_query, user_level)

        assert analysis["intent"] == "troubleshoot"
        assert analysis["user_level"] == "advanced"
        assert "javascript" in analysis["entities"]
        assert "troubleshooting" in analysis["preferred_content_types"]
        assert analysis["complexity_preference"] == "high"

    @pytest.mark.asyncio
    async def test_find_diverse_content(self):
        """Test finding diverse content for recommendations."""
        user_query = "Python programming"
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        context_id = "test_context"
        query_analysis = {
            "intent": "learn",
            "user_level": "beginner",
            "entities": ["python"],
            "preferred_content_types": ["tutorial", "concept_explanation"],
        }

        # Mock different search strategies
        semantic_results = [
            self.create_sample_content_item(
                "http://example.com/python1", "Python Basics", content_type="tutorial"
            )
        ]
        type_results = [
            self.create_sample_content_item(
                "http://example.com/python2",
                "Python Concepts",
                content_type="concept_explanation",
            )
        ]
        entity_results = [
            self.create_sample_content_item(
                "http://example.com/python3", "Python Guide", content_type="tutorial"
            )
        ]

        self.mock_database_manager.semantic_search = AsyncMock(
            return_value=semantic_results
        )
        self.mock_database_manager.search_by_content_type = AsyncMock(
            return_value=type_results
        )
        self.mock_database_manager.keyword_search = AsyncMock(
            return_value=entity_results
        )

        results = await self.endpoints._find_diverse_content(
            user_query, query_embedding, context_id, query_analysis, 10
        )

        assert len(results) > 0
        # Should have diverse content types
        content_types = {item["content_type"] for item in results}
        assert len(content_types) > 1

    @pytest.mark.asyncio
    async def test_generate_content_recommendations(self):
        """Test generating content recommendations."""
        content_candidates = [
            self.create_sample_content_item(
                "http://example.com/tutorial1",
                "Python Tutorial",
                content_type="tutorial",
                score=0.9,
            ),
            self.create_sample_content_item(
                "http://example.com/example1",
                "Python Examples",
                content_type="code_example",
                score=0.8,
            ),
            self.create_sample_content_item(
                "http://example.com/ref1",
                "Python Reference",
                content_type="api_reference",
                score=0.7,
            ),
        ]

        query_analysis = {
            "intent": "learn",
            "user_level": "beginner",
            "entities": ["python"],
            "preferred_content_types": ["tutorial", "code_example"],
        }

        knowledge_graph = self.create_sample_knowledge_graph()

        recommendations = await self.endpoints._generate_content_recommendations(
            content_candidates, query_analysis, knowledge_graph, 5
        )

        assert len(recommendations) > 0
        assert all(isinstance(rec, ContentRecommendation) for rec in recommendations)

        # Check properties
        for rec in recommendations:
            assert rec.url.startswith("http://")
            assert len(rec.title) > 0
            assert 0.0 <= rec.relevance_score <= 1.0
            assert rec.recommendation_type in [
                "tutorial",
                "example",
                "reference",
                "troubleshooting",
            ]
            assert len(rec.why_recommended) > 0

    def test_determine_recommendation_type(self):
        """Test determining recommendation type."""
        assert (
            self.endpoints._determine_recommendation_type("tutorial", "learn")
            == "tutorial"
        )
        assert (
            self.endpoints._determine_recommendation_type("code_example", "implement")
            == "example"
        )
        assert (
            self.endpoints._determine_recommendation_type("api_reference", "explore")
            == "reference"
        )
        assert (
            self.endpoints._determine_recommendation_type(
                "troubleshooting", "troubleshoot"
            )
            == "troubleshooting"
        )

    def test_generate_recommendation_explanation(self):
        """Test generating recommendation explanations."""
        candidate = self.create_sample_content_item(
            "http://example.com/test", "Test", content_type="tutorial"
        )
        query_analysis = {"intent": "learn", "entities": ["python"]}

        explanation = self.endpoints._generate_recommendation_explanation(
            candidate, "tutorial", query_analysis
        )

        assert len(explanation) > 0
        assert "step-by-step" in explanation.lower()
        assert "python" in explanation.lower()

    def test_find_related_content_urls(self):
        """Test finding related content URLs."""
        url = "http://example.com/tutorial1"
        knowledge_graph = self.create_sample_knowledge_graph()

        related_urls = self.endpoints._find_related_content_urls(url, knowledge_graph)

        assert len(related_urls) > 0
        assert "http://example.com/tutorial2" in related_urls

    def test_estimate_difficulty(self):
        """Test estimating content difficulty."""
        # Beginner content
        beginner_content = self.create_sample_content_item(
            "http://example.com/basic", "Basic Python Guide"
        )
        assert self.endpoints._estimate_difficulty(beginner_content) == "beginner"

        # Advanced content
        advanced_content = self.create_sample_content_item(
            "http://example.com/advanced", "Advanced Python Optimization"
        )
        assert self.endpoints._estimate_difficulty(advanced_content) == "advanced"

        # Intermediate content (default)
        intermediate_content = self.create_sample_content_item(
            "http://example.com/guide", "Python Programming Guide"
        )
        assert (
            self.endpoints._estimate_difficulty(intermediate_content) == "intermediate"
        )

    @pytest.mark.asyncio
    async def test_build_learning_path(self):
        """Test building a learning path."""
        recommendations = [
            ContentRecommendation(
                url="http://example.com/tutorial1",
                title="Basic Python",
                relevance_score=0.9,
                recommendation_type="tutorial",
                why_recommended="Good for beginners",
                related_content=[],
                difficulty_level="beginner",
            ),
            ContentRecommendation(
                url="http://example.com/example1",
                title="Python Examples",
                relevance_score=0.8,
                recommendation_type="example",
                why_recommended="Practical examples",
                related_content=[],
                difficulty_level="intermediate",
            ),
            ContentRecommendation(
                url="http://example.com/advanced1",
                title="Advanced Python",
                relevance_score=0.7,
                recommendation_type="tutorial",
                why_recommended="For experts",
                related_content=[],
                difficulty_level="advanced",
            ),
        ]

        knowledge_graph = self.create_sample_knowledge_graph()
        user_level = "beginner"

        learning_path = await self.endpoints._build_learning_path(
            recommendations, knowledge_graph, user_level
        )

        assert len(learning_path) > 0
        assert isinstance(learning_path, list)
        assert all(isinstance(url, str) for url in learning_path)

    def test_order_path_by_relationships(self):
        """Test ordering path by tutorial sequence relationships."""
        urls = ["http://example.com/tutorial1", "http://example.com/tutorial2"]
        knowledge_graph = self.create_sample_knowledge_graph()

        ordered_path = self.endpoints._order_path_by_relationships(
            urls, knowledge_graph
        )

        assert len(ordered_path) == 2
        # Should maintain or properly order based on relationships
        assert "http://example.com/tutorial1" in ordered_path
        assert "http://example.com/tutorial2" in ordered_path

    @pytest.mark.asyncio
    async def test_identify_gaps_and_steps(self):
        """Test identifying knowledge gaps and next steps."""
        user_query = "Learn Python programming"
        recommendations = [
            ContentRecommendation(
                url="http://example.com/tutorial1",
                title="Python Tutorial",
                relevance_score=0.9,
                recommendation_type="tutorial",
                why_recommended="Good tutorial",
                related_content=[],
            )
        ]

        knowledge_graph = self.create_sample_knowledge_graph()
        query_analysis = {
            "intent": "learn",
            "entities": ["python"],
            "preferred_content_types": ["tutorial", "code_example", "troubleshooting"],
        }

        result = await self.endpoints._identify_gaps_and_steps(
            user_query, recommendations, knowledge_graph, query_analysis
        )

        assert "related_clusters" in result
        assert "knowledge_gaps" in result
        assert "next_steps" in result
        assert isinstance(result["related_clusters"], list)
        assert isinstance(result["knowledge_gaps"], list)
        assert isinstance(result["next_steps"], list)

    @pytest.mark.asyncio
    async def test_recommend_context_integration(self):
        """Test full recommend_context integration."""
        user_query = "Learn Python programming"
        context_id = "test_context"
        user_level = "beginner"

        # Mock embedding service
        self.mock_embedding_service.embed_single = AsyncMock(
            return_value={"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "model": "test-model"}
        )

        # Mock database searches
        search_results = [
            self.create_sample_content_item(
                "http://example.com/python1", "Python Tutorial", content_type="tutorial"
            ),
            self.create_sample_content_item(
                "http://example.com/python2",
                "Python Examples",
                content_type="code_example",
            ),
        ]

        self.mock_database_manager.semantic_search = AsyncMock(
            return_value=search_results
        )
        self.mock_database_manager.search_by_content_type = AsyncMock(
            return_value=search_results
        )
        self.mock_database_manager.keyword_search = AsyncMock(
            return_value=search_results
        )

        # Mock knowledge graph
        knowledge_graph = self.create_sample_knowledge_graph()
        self.mock_knowledge_graph_builder.load_knowledge_graph = AsyncMock(
            return_value=knowledge_graph
        )

        result = await self.endpoints.recommend_context(
            user_query, context_id, user_level
        )

        assert isinstance(result, ContextRecommendation)
        assert len(result.primary_recommendations) > 0
        assert isinstance(result.learning_path, list)
        assert isinstance(result.related_clusters, list)
        assert isinstance(result.knowledge_gaps, list)
        assert isinstance(result.next_steps, list)
        assert 0.0 <= result.total_score <= 1.0

        self.mock_embedding_service.embed_single.assert_called_once()
        self.mock_knowledge_graph_builder.load_knowledge_graph.assert_called_once()

    def test_extract_procedural_info(self):
        """Test extracting procedural information."""
        content_texts = [
            "Step 1: Install Python. Step 2: Create a file. Then run the program.",
            "First, open your editor. Next, write the code. Finally, save the file.",
        ]

        result = self.endpoints._extract_procedural_info(content_texts)

        assert len(result) > 0
        assert "step" in result.lower() or "first" in result.lower()

    def test_extract_definitional_info(self):
        """Test extracting definitional information."""
        content_texts = [
            "A function is a reusable block of code that performs a specific task.",
            "Python is a high-level programming language known for its simplicity.",
        ]

        result = self.endpoints._extract_definitional_info(content_texts)

        assert len(result) > 0
        assert "function" in result.lower() or "python" in result.lower()

    def test_extract_solution_info(self):
        """Test extracting solution information."""
        content_texts = [
            "Solution: Check your syntax for missing parentheses.",
            "To fix this error, verify your variable names.",
            "The problem can be resolved by updating your imports.",
        ]

        result = self.endpoints._extract_solution_info(content_texts)

        assert len(result) > 0
        assert "solution" in result.lower() or "fix" in result.lower()

    def test_extract_general_info(self):
        """Test extracting general information."""
        content_texts = [
            "Python is a versatile programming language. It's great for beginners and experts alike.",
            "Web development with Python involves using frameworks like Django or Flask.",
        ]

        result = self.endpoints._extract_general_info(content_texts)

        assert len(result) > 0
        assert "python" in result.lower()

    def test_extract_related_topics(self):
        """Test extracting related topics."""
        content_texts = [
            "Python programming involves Variables, Functions, and Classes. Django is a popular framework."
        ]
        entities = ["python", "programming"]

        topics = self.endpoints._extract_related_topics(content_texts, entities)

        assert len(topics) > 0
        assert "python" in topics
        assert "Variables" in topics or "Functions" in topics or "Classes" in topics

    def test_generate_follow_up_questions(self):
        """Test generating follow-up questions."""
        original_question = "How do I create Python functions?"
        question_analysis = {"type": "how_to", "entities": ["python", "functions"]}
        related_topics = ["Variables", "Classes"]

        follow_ups = self.endpoints._generate_follow_up_questions(
            original_question, question_analysis, related_topics
        )

        assert len(follow_ups) > 0
        assert any("python" in q.lower() for q in follow_ups)

    def test_calculate_answer_confidence(self):
        """Test calculating answer confidence."""
        # High quality sources
        high_quality_sources = [
            {"relevance_score": 0.9, "content_type": "tutorial"},
            {"relevance_score": 0.8, "content_type": "code_example"},
        ]

        question_analysis = {"type": "how_to"}

        confidence = self.endpoints._calculate_answer_confidence(
            high_quality_sources, question_analysis
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high for good sources

        # Low quality sources
        low_quality_sources = [{"relevance_score": 0.3, "content_type": "general"}]

        low_confidence = self.endpoints._calculate_answer_confidence(
            low_quality_sources, question_analysis
        )

        assert 0.0 <= low_confidence <= 1.0
        assert low_confidence < confidence  # Should be lower

    @pytest.mark.asyncio
    async def test_answer_question_error_handling(self):
        """Test error handling in answer_question."""
        question = "Test question"
        context_id = "test_context"

        # Mock embedding service to raise error
        self.mock_embedding_service.embed_single = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        result = await self.endpoints.answer_question(question, context_id)

        assert isinstance(result, ContextualAnswer)
        assert result.confidence == 0.0
        assert "error" in result.answer.lower()
        assert len(result.reasoning_chain) > 0
        assert "Error:" in result.reasoning_chain[0]

    @pytest.mark.asyncio
    async def test_recommend_context_error_handling(self):
        """Test error handling in recommend_context."""
        user_query = "Test query"
        context_id = "test_context"

        # Mock embedding service to raise error
        self.mock_embedding_service.embed_single = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        result = await self.endpoints.recommend_context(user_query, context_id)

        assert isinstance(result, ContextRecommendation)
        assert len(result.primary_recommendations) == 0
        assert len(result.learning_path) == 0
        assert result.total_score == 0.0
