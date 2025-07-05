"""LLM-optimized endpoints for question-answering and context recommendation."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .content_analysis import ContentAnalysis
from .enhanced_storage import EnhancedDatabaseManager
from .multi_embedding_service import MultiEmbeddingService
from .relationship_mapping import KnowledgeGraphBuilder, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class ContextualAnswer:
    """Represents an answer with supporting context."""

    answer: str
    confidence: float
    supporting_sources: List[Dict[str, Any]]
    related_topics: List[str]
    code_examples: List[str]
    follow_up_questions: List[str]
    reasoning_chain: List[str]


@dataclass
class ContentRecommendation:
    """Represents a content recommendation."""

    url: str
    title: str
    relevance_score: float
    recommendation_type: str  # "tutorial", "reference", "example", "troubleshooting"
    why_recommended: str
    related_content: List[str]
    difficulty_level: Optional[str] = None
    estimated_reading_time: Optional[int] = None


@dataclass
class ContextRecommendation:
    """Comprehensive context recommendation including content and learning path."""

    primary_recommendations: List[ContentRecommendation]
    learning_path: List[str]  # Ordered sequence of URLs for progressive learning
    related_clusters: List[str]  # Topic cluster IDs
    knowledge_gaps: List[str]  # Identified gaps in current context
    next_steps: List[str]  # Suggested next actions
    total_score: float


class LLMOptimizedEndpoints:
    """Service for LLM-optimized question-answering and recommendations."""

    def __init__(
        self,
        embedding_service: MultiEmbeddingService,
        database_manager: EnhancedDatabaseManager,
        knowledge_graph_builder: KnowledgeGraphBuilder,
    ):
        self.embedding_service = embedding_service
        self.database_manager = database_manager
        self.knowledge_graph_builder = knowledge_graph_builder

        # Question analysis patterns
        self.question_patterns = {
            "how_to": [
                r"how\s+(?:do\s+i|to|can\s+i)\s+",
                r"what\s+(?:is\s+the\s+)?(?:best\s+)?way\s+to\s+",
                r"steps\s+to\s+",
                r"guide\s+(?:for|to)\s+",
            ],
            "what_is": [
                r"what\s+is\s+",
                r"define\s+",
                r"explain\s+",
                r"meaning\s+of\s+",
            ],
            "troubleshooting": [
                r"error\s+",
                r"not\s+working\s+",
                r"problem\s+with\s+",
                r"fix\s+",
                r"debug\s+",
                r"issue\s+with\s+",
            ],
            "code_example": [
                r"example\s+(?:of|for)\s+",
                r"sample\s+code\s+",
                r"implementation\s+of\s+",
                r"code\s+(?:for|to)\s+",
            ],
            "comparison": [
                r"(?:difference\s+between|vs\.?|versus|compare)\s+",
                r"which\s+(?:is\s+)?better\s+",
                r"pros\s+and\s+cons\s+",
            ],
        }

        logger.info("LLM-optimized endpoints initialized")

    async def answer_question(
        self,
        question: str,
        context_id: str,
        max_sources: int = 5,
        include_code: bool = True,
    ) -> ContextualAnswer:
        """
        Generate a contextual answer to a question using available knowledge.

        Args:
            question: The question to answer
            context_id: Context/workspace ID for scoped search
            max_sources: Maximum number of supporting sources
            include_code: Whether to include code examples

        Returns:
            ContextualAnswer with comprehensive response
        """
        try:
            logger.info(f"Answering question: {question[:100]}...")

            # Analyze question to understand intent and type
            question_analysis = await self._analyze_question(question)

            # Generate embedding for semantic search
            question_embedding = await self.embedding_service.embed_single(
                question, content_analysis=None
            )

            # Find relevant content using multiple strategies
            relevant_content = await self._find_relevant_content(
                question,
                question_embedding["embedding"],
                context_id,
                question_analysis,
                max_sources * 2,  # Get more initially, then filter
            )

            # Load knowledge graph for relationship context
            knowledge_graph = await self.knowledge_graph_builder.load_knowledge_graph()

            # Enhance content with relationship context
            enhanced_content = await self._enhance_with_relationships(
                relevant_content, knowledge_graph, question_analysis["intent"]
            )

            # Generate contextual answer
            answer = await self._generate_contextual_answer(
                question,
                enhanced_content[:max_sources],
                question_analysis,
                include_code,
            )

            logger.info(
                f"Generated answer with {len(answer.supporting_sources)} sources"
            )
            return answer

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return ContextualAnswer(
                answer="I apologize, but I encountered an error while processing your question.",
                confidence=0.0,
                supporting_sources=[],
                related_topics=[],
                code_examples=[],
                follow_up_questions=[],
                reasoning_chain=[f"Error: {str(e)}"],
            )

    async def recommend_context(
        self,
        user_query: str,
        context_id: str,
        user_level: str = "intermediate",
        max_recommendations: int = 10,
    ) -> ContextRecommendation:
        """
        Recommend relevant content and learning paths based on user query.

        Args:
            user_query: What the user is interested in learning/exploring
            context_id: Context/workspace ID for scoped recommendations
            user_level: User's skill level (beginner, intermediate, advanced)
            max_recommendations: Maximum number of content recommendations

        Returns:
            ContextRecommendation with comprehensive suggestions
        """
        try:
            logger.info(f"Generating recommendations for: {user_query[:100]}...")

            # Analyze user query for intent and topics
            query_analysis = await self._analyze_user_intent(user_query, user_level)

            # Generate embedding for semantic matching
            query_embedding = await self.embedding_service.embed_single(
                user_query, content_analysis=None
            )

            # Find relevant content across different types
            content_candidates = await self._find_diverse_content(
                user_query,
                query_embedding["embedding"],
                context_id,
                query_analysis,
                max_recommendations * 3,  # Get more for better filtering
            )

            # Load knowledge graph for clustering and relationships
            knowledge_graph = await self.knowledge_graph_builder.load_knowledge_graph()

            # Generate diverse recommendations
            recommendations = await self._generate_content_recommendations(
                content_candidates, query_analysis, knowledge_graph, max_recommendations
            )

            # Build learning path
            learning_path = await self._build_learning_path(
                recommendations, knowledge_graph, user_level
            )

            # Identify knowledge gaps and next steps
            gaps_and_steps = await self._identify_gaps_and_steps(
                user_query, recommendations, knowledge_graph, query_analysis
            )

            # Calculate total recommendation score
            total_score = (
                sum(rec.relevance_score for rec in recommendations)
                / len(recommendations)
                if recommendations
                else 0.0
            )

            result = ContextRecommendation(
                primary_recommendations=recommendations,
                learning_path=learning_path,
                related_clusters=gaps_and_steps["related_clusters"],
                knowledge_gaps=gaps_and_steps["knowledge_gaps"],
                next_steps=gaps_and_steps["next_steps"],
                total_score=total_score,
            )

            logger.info(
                f"Generated {len(recommendations)} recommendations with learning path of {len(learning_path)} steps"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ContextRecommendation(
                primary_recommendations=[],
                learning_path=[],
                related_clusters=[],
                knowledge_gaps=[],
                next_steps=[],
                total_score=0.0,
            )

    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to understand intent and type."""
        import re

        question_lower = question.lower()

        # Detect question type
        question_type = "general"
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    question_type = q_type
                    break
            if question_type != "general":
                break

        # Extract key entities and topics
        entities = self._extract_entities(question)

        # Determine intent
        intent_map = {
            "how_to": "procedural",
            "what_is": "definitional",
            "troubleshooting": "problem_solving",
            "code_example": "implementation",
            "comparison": "analytical",
        }
        intent = intent_map.get(question_type, "informational")

        return {
            "type": question_type,
            "intent": intent,
            "entities": entities,
            "requires_code": question_type
            in ["how_to", "code_example", "troubleshooting"],
            "complexity": "high"
            if any(
                word in question_lower
                for word in ["advanced", "complex", "optimization", "performance"]
            )
            else "medium",
        }

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities and technical terms from text."""
        import re

        # Extract programming languages
        languages = re.findall(
            r"\b(?:python|javascript|java|typescript|go|rust|c\+\+|c#|php|ruby|swift|kotlin)\b",
            text.lower(),
        )

        # Extract frameworks/libraries
        frameworks = re.findall(
            r"\b(?:react|vue|angular|django|flask|express|spring|node\.?js|tensorflow|pytorch)\b",
            text.lower(),
        )

        # Extract technical terms (capitalized words)
        tech_terms = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)

        # Extract API-like terms
        api_terms = re.findall(
            r"\b\w+(?:API|api|endpoint|method|function)\b", text.lower()
        )

        entities = languages + frameworks + tech_terms + api_terms
        return list(set(entities))

    async def _find_relevant_content(
        self,
        question: str,
        question_embedding: List[float],
        context_id: str,
        question_analysis: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find content relevant to the question using multiple strategies."""

        # Strategy 1: Semantic similarity search
        semantic_results = await self.database_manager.semantic_search(
            query_embedding=question_embedding,
            context_id=context_id,
            limit=limit // 2,
            similarity_threshold=0.7,
        )

        # Strategy 2: Keyword-based search for entities
        keyword_results = []
        if question_analysis["entities"]:
            for entity in question_analysis["entities"][:3]:  # Top 3 entities
                entity_results = await self.database_manager.keyword_search(
                    query=entity, context_id=context_id, limit=5
                )
                keyword_results.extend(entity_results)

        # Strategy 3: Content type specific search
        if question_analysis["type"] in ["how_to", "code_example"]:
            type_results = await self.database_manager.search_by_content_type(
                content_types=["tutorial", "code_example"],
                context_id=context_id,
                limit=limit // 4,
            )
            keyword_results.extend(type_results)
        elif question_analysis["type"] == "troubleshooting":
            type_results = await self.database_manager.search_by_content_type(
                content_types=["troubleshooting", "api_reference"],
                context_id=context_id,
                limit=limit // 4,
            )
            keyword_results.extend(type_results)

        # Combine and deduplicate results
        all_results = semantic_results + keyword_results
        seen_urls = set()
        unique_results = []

        for result in all_results:
            url = result.get("url", result.get("source_url"))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # Sort by relevance score
        unique_results.sort(
            key=lambda x: x.get("score", x.get("similarity", 0)), reverse=True
        )

        return unique_results[:limit]

    async def _enhance_with_relationships(
        self, content: List[Dict[str, Any]], knowledge_graph: Any, intent: str
    ) -> List[Dict[str, Any]]:
        """Enhance content with relationship context from knowledge graph."""

        enhanced_content = []

        for item in content:
            url = item.get("url", item.get("source_url"))
            if not url:
                enhanced_content.append(item)
                continue

            # Find relationships for this content
            related_content = []

            for relationship in knowledge_graph.relationships:
                if relationship.source_url == url:
                    # Add relationship context based on intent
                    if (
                        intent == "procedural"
                        and relationship.relationship_type
                        == RelationshipType.TUTORIAL_SEQUENCE
                    ):
                        related_content.append(
                            {
                                "url": relationship.target_url,
                                "relationship": "next_step",
                                "strength": relationship.strength,
                            }
                        )
                    elif (
                        intent == "implementation"
                        and relationship.relationship_type
                        == RelationshipType.CODE_DEPENDENCY
                    ):
                        related_content.append(
                            {
                                "url": relationship.target_url,
                                "relationship": "dependency",
                                "strength": relationship.strength,
                            }
                        )
                    elif (
                        relationship.relationship_type
                        == RelationshipType.SEMANTIC_SIMILARITY
                        and relationship.strength > 0.8
                    ):
                        related_content.append(
                            {
                                "url": relationship.target_url,
                                "relationship": "similar_content",
                                "strength": relationship.strength,
                            }
                        )

            # Add relationship context to item
            enhanced_item = {**item, "related_content": related_content}
            enhanced_content.append(enhanced_item)

        return enhanced_content

    async def _generate_contextual_answer(
        self,
        question: str,
        relevant_content: List[Dict[str, Any]],
        question_analysis: Dict[str, Any],
        include_code: bool,
    ) -> ContextualAnswer:
        """Generate a contextual answer from relevant content."""

        # Extract supporting sources
        supporting_sources = []
        all_content_text = []
        code_examples = []

        for item in relevant_content:
            source_info = {
                "url": item.get("url", item.get("source_url", "")),
                "title": item.get("title", ""),
                "content_type": item.get("content_type", "general"),
                "relevance_score": item.get("score", item.get("similarity", 0.0)),
                "summary": item.get("summary", item.get("content", "")[:200]),
            }
            supporting_sources.append(source_info)

            # Collect content for answer generation
            content_text = item.get("content", "")
            all_content_text.append(content_text)

            # Extract code examples if requested
            if include_code and item.get("content_type") in [
                "code_example",
                "tutorial",
            ]:
                # Simple code block extraction
                import re

                code_blocks = re.findall(
                    r"```[\w]*\n(.*?)\n```", content_text, re.DOTALL
                )
                code_examples.extend(code_blocks[:2])  # Limit per source

        # Generate answer based on question type and intent
        answer = await self._synthesize_answer(
            question, all_content_text, question_analysis
        )

        # Generate related topics
        related_topics = self._extract_related_topics(
            all_content_text, question_analysis["entities"]
        )

        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(
            question, question_analysis, related_topics
        )

        # Create reasoning chain
        reasoning_chain = [
            f"Analyzed question type: {question_analysis['type']}",
            f"Found {len(supporting_sources)} relevant sources",
            f"Identified {len(question_analysis['entities'])} key entities",
            f"Generated answer using {question_analysis['intent']} approach",
        ]

        # Calculate confidence based on source quality and relevance
        confidence = self._calculate_answer_confidence(
            supporting_sources, question_analysis
        )

        return ContextualAnswer(
            answer=answer,
            confidence=confidence,
            supporting_sources=supporting_sources,
            related_topics=related_topics[:5],
            code_examples=code_examples[:3],
            follow_up_questions=follow_up_questions[:3],
            reasoning_chain=reasoning_chain,
        )

    async def _synthesize_answer(
        self, question: str, content_texts: List[str], question_analysis: Dict[str, Any]
    ) -> str:
        """Synthesize an answer from available content."""

        # For now, provide a structured answer based on content type
        # In a full implementation, this would use an LLM to synthesize content

        if not content_texts:
            return "I don't have enough information in the current context to answer this question."

        question_type = question_analysis["type"]

        if question_type == "how_to":
            return f"""Based on the available documentation, here's how to approach this:

{self._extract_procedural_info(content_texts)}

This approach is derived from the relevant documentation in your context."""

        elif question_type == "what_is":
            return f"""Based on the documentation in your context:

{self._extract_definitional_info(content_texts)}

This definition is compiled from the available sources."""

        elif question_type == "troubleshooting":
            return f"""Here are solutions based on the troubleshooting information in your context:

{self._extract_solution_info(content_texts)}

These solutions are from the documentation and examples in your workspace."""

        else:
            # General informational response
            combined_info = self._extract_general_info(content_texts)
            return f"""Based on the relevant documentation in your context:

{combined_info}

This information is compiled from the sources that best match your question."""

    def _extract_procedural_info(self, content_texts: List[str]) -> str:
        """Extract step-by-step procedural information."""
        steps = []
        step_indicators = ["step", "first", "then", "next", "finally", "1.", "2.", "3."]

        for content in content_texts[:3]:  # Analyze top 3 sources
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    any(indicator in line.lower() for indicator in step_indicators)
                    and len(line) > 20
                ):
                    steps.append(line)
                    if len(steps) >= 5:  # Limit steps
                        break

        if steps:
            return "\n".join(f"• {step}" for step in steps[:5])
        else:
            return "The available documentation contains relevant information, but specific steps may need to be extracted from the full content."

    def _extract_definitional_info(self, content_texts: List[str]) -> str:
        """Extract definitional information."""
        definitions = []

        for content in content_texts[:2]:  # Top 2 sources for definitions
            # Look for definition patterns
            sentences = content.split(".")
            for sentence in sentences:
                if len(sentence.strip()) > 30 and len(sentence.strip()) < 200:
                    definitions.append(sentence.strip())
                    break  # One definition per source

        if definitions:
            return ". ".join(definitions)
        else:
            return "The concept is referenced in the available documentation. Please refer to the supporting sources for detailed information."

    def _extract_solution_info(self, content_texts: List[str]) -> str:
        """Extract troubleshooting and solution information."""
        solutions = []
        solution_indicators = [
            "solution",
            "fix",
            "resolve",
            "error",
            "problem",
            "issue",
        ]

        for content in content_texts[:3]:
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    any(indicator in line.lower() for indicator in solution_indicators)
                    and len(line) > 20
                ):
                    solutions.append(line)
                    if len(solutions) >= 3:
                        break

        if solutions:
            return "\n".join(f"• {solution}" for solution in solutions)
        else:
            return "The documentation contains troubleshooting information. Please check the supporting sources for specific solutions."

    def _extract_general_info(self, content_texts: List[str]) -> str:
        """Extract general informational content."""
        # Get the most relevant snippets from top sources
        snippets = []

        for content in content_texts[:2]:
            # Take first substantial paragraph
            paragraphs = content.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50 and len(para) < 300:
                    snippets.append(para)
                    break

        if snippets:
            return "\n\n".join(snippets)
        else:
            return "Relevant information is available in the supporting sources."

    def _extract_related_topics(
        self, content_texts: List[str], entities: List[str]
    ) -> List[str]:
        """Extract related topics from content."""
        import re

        all_text = " ".join(content_texts)

        # Extract capitalized terms that might be topics
        topics = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", all_text)

        # Filter and combine with entities
        filtered_topics = [
            topic
            for topic in topics
            if len(topic) > 3 and topic not in ["The", "This", "That"]
        ]
        all_topics = list(set(entities + filtered_topics))

        return all_topics[:10]

    def _generate_follow_up_questions(
        self,
        original_question: str,
        question_analysis: Dict[str, Any],
        related_topics: List[str],
    ) -> List[str]:
        """Generate relevant follow-up questions."""

        follow_ups = []
        q_type = question_analysis["type"]
        entities = question_analysis["entities"][:2]  # Top 2 entities

        if q_type == "how_to":
            if entities:
                follow_ups.append(f"What are best practices for {entities[0]}?")
                follow_ups.append(f"Are there any common issues with {entities[0]}?")
        elif q_type == "what_is":
            if entities:
                follow_ups.append(f"How do I implement {entities[0]}?")
                follow_ups.append(f"What are examples of {entities[0]}?")
        elif q_type == "troubleshooting":
            follow_ups.append("How can I prevent this issue in the future?")
            follow_ups.append("Are there alternative approaches?")

        # Add topic-based follow-ups
        if related_topics:
            follow_ups.append(f"How does this relate to {related_topics[0]}?")

        return follow_ups[:3]

    def _calculate_answer_confidence(
        self,
        supporting_sources: List[Dict[str, Any]],
        question_analysis: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the generated answer."""

        if not supporting_sources:
            return 0.0

        # Base confidence on source quality
        avg_relevance = sum(
            source.get("relevance_score", 0) for source in supporting_sources
        ) / len(supporting_sources)

        # Boost confidence for specific question types with appropriate content
        type_boost = 0.0
        content_types = [
            source.get("content_type", "") for source in supporting_sources
        ]

        if question_analysis["type"] == "how_to" and "tutorial" in content_types:
            type_boost = 0.1
        elif (
            question_analysis["type"] == "code_example"
            and "code_example" in content_types
        ):
            type_boost = 0.1
        elif (
            question_analysis["type"] == "troubleshooting"
            and "troubleshooting" in content_types
        ):
            type_boost = 0.1

        # Number of sources boost
        source_boost = min(len(supporting_sources) * 0.05, 0.2)

        confidence = min(avg_relevance + type_boost + source_boost, 1.0)
        return round(confidence, 2)

    async def _analyze_user_intent(
        self, user_query: str, user_level: str
    ) -> Dict[str, Any]:
        """Analyze user query to understand learning intent and preferences."""

        query_lower = user_query.lower()

        # Detect learning intent (order matters - most specific first)
        learning_intents = [
            (
                "troubleshoot",
                ["fix", "debug", "solve", "resolve", "error", "problem", "issue"],
            ),
            ("learn", ["learn", "understand", "study", "master"]),
            ("implement", ["build", "create", "implement", "develop", "code"]),
            ("explore", ["explore", "discover", "find", "research", "investigate"]),
        ]

        intent = "explore"  # default
        for intent_type, keywords in learning_intents:
            if any(keyword in query_lower for keyword in keywords):
                intent = intent_type
                break

        # Extract topics and technologies
        entities = self._extract_entities(user_query)

        # Determine preferred content types based on intent and level
        preferred_types = []
        if intent == "learn":
            if user_level == "beginner":
                preferred_types = ["tutorial", "concept_explanation"]
            else:
                preferred_types = ["tutorial", "api_reference", "concept_explanation"]
        elif intent == "implement":
            preferred_types = ["code_example", "tutorial", "api_reference"]
        elif intent == "troubleshoot":
            preferred_types = ["troubleshooting", "api_reference", "code_example"]
        else:  # explore
            preferred_types = ["concept_explanation", "tutorial", "api_reference"]

        return {
            "intent": intent,
            "user_level": user_level,
            "entities": entities,
            "preferred_content_types": preferred_types,
            "complexity_preference": "low"
            if user_level == "beginner"
            else "medium"
            if user_level == "intermediate"
            else "high",
        }

    async def _find_diverse_content(
        self,
        user_query: str,
        query_embedding: List[float],
        context_id: str,
        query_analysis: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find diverse content covering different aspects of the query."""

        content_candidates = []

        # Strategy 1: Semantic search
        semantic_results = await self.database_manager.semantic_search(
            query_embedding=query_embedding,
            context_id=context_id,
            limit=limit // 2,
            similarity_threshold=0.6,
        )
        content_candidates.extend(semantic_results)

        # Strategy 2: Content type specific searches
        for content_type in query_analysis["preferred_content_types"]:
            type_results = await self.database_manager.search_by_content_type(
                content_types=[content_type], context_id=context_id, limit=5
            )
            content_candidates.extend(type_results)

        # Strategy 3: Entity-based searches
        for entity in query_analysis["entities"][:3]:
            entity_results = await self.database_manager.keyword_search(
                query=entity, context_id=context_id, limit=3
            )
            content_candidates.extend(entity_results)

        # Deduplicate and sort
        seen_urls = set()
        unique_candidates = []

        for candidate in content_candidates:
            url = candidate.get("url", candidate.get("source_url"))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_candidates.append(candidate)

        # Sort by relevance
        unique_candidates.sort(
            key=lambda x: x.get("score", x.get("similarity", 0)), reverse=True
        )

        return unique_candidates[:limit]

    async def _generate_content_recommendations(
        self,
        content_candidates: List[Dict[str, Any]],
        query_analysis: Dict[str, Any],
        knowledge_graph: Any,
        max_recommendations: int,
    ) -> List[ContentRecommendation]:
        """Generate diverse content recommendations with explanations."""

        recommendations = []
        content_type_counts = {}

        for candidate in content_candidates:
            if len(recommendations) >= max_recommendations:
                break

            content_type = candidate.get("content_type", "general")

            # Ensure diversity by limiting same content types
            if content_type_counts.get(content_type, 0) >= max_recommendations // 3:
                continue

            # Calculate relevance score
            base_score = candidate.get("score", candidate.get("similarity", 0.0))

            # Boost score based on content type preference
            type_boost = 0.0
            if content_type in query_analysis["preferred_content_types"]:
                type_boost = 0.1

            # Boost score based on user level appropriateness
            level_boost = 0.0
            title = candidate.get("title", "").lower()
            if query_analysis["user_level"] == "beginner" and any(
                word in title
                for word in ["basic", "intro", "beginner", "getting started"]
            ):
                level_boost = 0.1
            elif query_analysis["user_level"] == "advanced" and any(
                word in title for word in ["advanced", "expert", "optimization"]
            ):
                level_boost = 0.1

            relevance_score = min(base_score + type_boost + level_boost, 1.0)

            # Determine recommendation type
            rec_type = self._determine_recommendation_type(
                content_type, query_analysis["intent"]
            )

            # Generate explanation
            why_recommended = self._generate_recommendation_explanation(
                candidate, content_type, query_analysis
            )

            # Find related content
            related_content = self._find_related_content_urls(
                candidate.get("url", candidate.get("source_url", "")), knowledge_graph
            )

            # Estimate reading time (rough)
            content_length = len(candidate.get("content", ""))
            estimated_time = max(1, content_length // 1000)  # ~1000 chars per minute

            recommendation = ContentRecommendation(
                url=candidate.get("url", candidate.get("source_url", "")),
                title=candidate.get("title", "Untitled"),
                relevance_score=relevance_score,
                recommendation_type=rec_type,
                why_recommended=why_recommended,
                related_content=related_content,
                difficulty_level=self._estimate_difficulty(candidate),
                estimated_reading_time=estimated_time,
            )

            recommendations.append(recommendation)
            content_type_counts[content_type] = (
                content_type_counts.get(content_type, 0) + 1
            )

        # Sort by relevance score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)

        return recommendations

    def _determine_recommendation_type(self, content_type: str, intent: str) -> str:
        """Determine the type of recommendation based on content and intent."""

        if content_type == "tutorial":
            return "tutorial"
        elif content_type == "api_reference":
            return "reference"
        elif content_type == "code_example":
            return "example"
        elif content_type == "troubleshooting":
            return "troubleshooting"
        elif intent == "learn":
            return "tutorial"
        elif intent == "implement":
            return "example"
        else:
            return "reference"

    def _generate_recommendation_explanation(
        self,
        candidate: Dict[str, Any],
        content_type: str,
        query_analysis: Dict[str, Any],
    ) -> str:
        """Generate explanation for why content is recommended."""

        explanations = []

        # Content type based explanation
        if content_type == "tutorial":
            explanations.append("provides step-by-step guidance")
        elif content_type == "code_example":
            explanations.append("includes practical code examples")
        elif content_type == "api_reference":
            explanations.append("offers comprehensive API documentation")
        elif content_type == "troubleshooting":
            explanations.append("helps solve common issues")

        # Intent based explanation
        if query_analysis["intent"] == "learn":
            explanations.append("suitable for learning the concepts")
        elif query_analysis["intent"] == "implement":
            explanations.append("helpful for implementation")

        # Entity based explanation
        if query_analysis["entities"]:
            explanations.append(f"covers {query_analysis['entities'][0]}")

        if explanations:
            return f"Recommended because it {' and '.join(explanations)}."
        else:
            return "Matches your query and interests."

    def _find_related_content_urls(self, url: str, knowledge_graph: Any) -> List[str]:
        """Find URLs of related content using knowledge graph."""

        related_urls = []

        for relationship in knowledge_graph.relationships:
            if relationship.source_url == url and relationship.strength > 0.7:
                related_urls.append(relationship.target_url)
            elif relationship.target_url == url and relationship.strength > 0.7:
                related_urls.append(relationship.source_url)

        return related_urls[:3]  # Limit to top 3

    def _estimate_difficulty(self, content: Dict[str, Any]) -> str:
        """Estimate content difficulty level."""

        title = content.get("title", "").lower()
        content_text = content.get("content", "").lower()

        # Check for beginner indicators
        beginner_indicators = [
            "basic",
            "intro",
            "beginner",
            "getting started",
            "simple",
            "easy",
        ]
        if any(indicator in title for indicator in beginner_indicators):
            return "beginner"

        # Check for advanced indicators
        advanced_indicators = [
            "advanced",
            "expert",
            "complex",
            "optimization",
            "performance",
            "deep dive",
        ]
        if any(indicator in title for indicator in advanced_indicators):
            return "advanced"

        # Check content complexity
        complex_terms = [
            "algorithm",
            "architecture",
            "scalability",
            "concurrency",
            "distributed",
        ]
        if any(term in content_text for term in complex_terms):
            return "advanced"

        return "intermediate"  # default

    async def _build_learning_path(
        self,
        recommendations: List[ContentRecommendation],
        knowledge_graph: Any,
        user_level: str,
    ) -> List[str]:
        """Build an ordered learning path from recommendations."""

        if not recommendations:
            return []

        # Separate content by type and difficulty
        tutorials = [
            rec for rec in recommendations if rec.recommendation_type == "tutorial"
        ]
        examples = [
            rec for rec in recommendations if rec.recommendation_type == "example"
        ]
        references = [
            rec for rec in recommendations if rec.recommendation_type == "reference"
        ]

        learning_path = []

        # Start with beginner tutorials if user is beginner
        if user_level == "beginner":
            beginner_tutorials = [
                t for t in tutorials if t.difficulty_level == "beginner"
            ]
            learning_path.extend([t.url for t in beginner_tutorials[:2]])

        # Add concept explanations and intermediate tutorials
        intermediate_content = [
            rec
            for rec in recommendations
            if rec.difficulty_level in ["intermediate", None]
            and rec.recommendation_type in ["tutorial", "reference"]
        ]
        learning_path.extend([rec.url for rec in intermediate_content[:3]])

        # Add practical examples
        learning_path.extend([ex.url for ex in examples[:2]])

        # Add advanced content if user is not beginner
        if user_level != "beginner":
            advanced_content = [
                rec for rec in recommendations if rec.difficulty_level == "advanced"
            ]
            learning_path.extend([rec.url for rec in advanced_content[:2]])

        # Use knowledge graph to order based on tutorial sequences
        ordered_path = self._order_path_by_relationships(learning_path, knowledge_graph)

        return ordered_path[:10]  # Limit path length

    def _order_path_by_relationships(
        self, urls: List[str], knowledge_graph: Any
    ) -> List[str]:
        """Order URLs based on tutorial sequence relationships."""

        if len(urls) <= 1:
            return urls

        # Build dependency graph
        dependencies = {}
        for url in urls:
            dependencies[url] = []

        for relationship in knowledge_graph.relationships:
            if (
                relationship.relationship_type == RelationshipType.TUTORIAL_SEQUENCE
                and relationship.source_url in dependencies
                and relationship.target_url in dependencies
            ):
                dependencies[relationship.target_url].append(relationship.source_url)

        # Topological sort for ordering
        ordered = []
        visited = set()

        def visit(url):
            if url in visited:
                return
            visited.add(url)
            for dep in dependencies[url]:
                visit(dep)
            ordered.append(url)

        for url in urls:
            visit(url)

        return ordered

    async def _identify_gaps_and_steps(
        self,
        user_query: str,
        recommendations: List[ContentRecommendation],
        knowledge_graph: Any,
        query_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify knowledge gaps and suggest next steps."""

        # Find related clusters
        related_clusters = []
        recommended_urls = {rec.url for rec in recommendations}

        for cluster in knowledge_graph.clusters:
            overlap = len(set(cluster.content_urls) & recommended_urls)
            if overlap > 0:
                related_clusters.append(cluster.cluster_id)

        # Identify potential knowledge gaps
        knowledge_gaps = []
        entities = query_analysis["entities"]

        # Check if we have all content types user might need
        available_types = {rec.recommendation_type for rec in recommendations}
        needed_types = set(query_analysis["preferred_content_types"])
        missing_types = needed_types - available_types

        for missing_type in missing_types:
            if missing_type == "tutorial":
                knowledge_gaps.append(
                    f"Could benefit from more tutorial content about {entities[0] if entities else 'the topic'}"
                )
            elif missing_type == "code_example":
                knowledge_gaps.append(f"Could use more practical code examples")
            elif missing_type == "troubleshooting":
                knowledge_gaps.append(f"May need troubleshooting guides")

        # Suggest next steps based on intent
        next_steps = []
        intent = query_analysis["intent"]

        if intent == "learn":
            next_steps.append("Start with the tutorial content in your learning path")
            next_steps.append("Practice with the provided code examples")
            user_level = query_analysis.get("user_level", "intermediate")
            if user_level == "beginner":
                next_steps.append(
                    "Focus on understanding basic concepts before moving to advanced topics"
                )
        elif intent == "implement":
            next_steps.append("Review the code examples and API references")
            next_steps.append("Start with a simple implementation")
            next_steps.append("Refer to troubleshooting guides if you encounter issues")
        elif intent == "troubleshoot":
            next_steps.append("Check the troubleshooting content first")
            next_steps.append("Review related API documentation")
            next_steps.append("Look for similar issues in the code examples")
        else:  # explore
            next_steps.append("Browse the recommended content to understand the topic")
            next_steps.append("Follow the learning path for structured exploration")
            next_steps.append("Dive deeper into areas that interest you most")

        return {
            "related_clusters": related_clusters[:5],
            "knowledge_gaps": knowledge_gaps[:3],
            "next_steps": next_steps[:3],
        }
