"""Progressive refinement engine for iterative search improvement."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .multi_modal_search import SearchEngine, SearchResponse, SearchResult
from .query_analysis import QueryAnalysis, QueryAnalyzer

logger = logging.getLogger(__name__)


class RefinementType(Enum):
    """Types of refinement strategies."""

    QUERY_EXPANSION = "query_expansion"  # Add related terms
    QUERY_NARROWING = "query_narrowing"  # Remove broad terms
    STRATEGY_ADJUSTMENT = "strategy_adjustment"  # Change search strategies
    FILTER_MODIFICATION = "filter_modification"  # Adjust search filters
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"  # Improve semantic understanding


@dataclass
class RefinementStep:
    """Individual refinement step with metadata."""

    step_number: int
    refinement_type: RefinementType
    original_query: str
    refined_query: str
    filters_applied: Dict[str, Any]
    strategies_used: List[str]
    improvement_score: float  # How much this step improved results
    confidence: float
    reasoning: str


@dataclass
class RefinementSession:
    """Complete refinement session tracking all steps."""

    original_query: str
    original_analysis: QueryAnalysis
    refinement_steps: List[RefinementStep]
    final_response: SearchResponse
    total_improvement: float
    session_confidence: float


class ProgressiveRefinementEngine:
    """Engine for progressive search refinement with feedback loops."""

    def __init__(self, search_engine: SearchEngine):
        """Initialize refinement engine with search engine."""
        self.search_engine = search_engine
        self.query_analyzer = QueryAnalyzer()

        # Refinement configuration
        self.max_refinement_steps = 3
        self.min_improvement_threshold = 0.1
        self.confidence_threshold = 0.8

    async def refine_search(
        self, query: str, filters: Dict[str, Any] = None, target_result_count: int = 10
    ) -> RefinementSession:
        """Execute progressive search refinement until satisfactory results."""

        logger.info(f"Starting progressive refinement for query: '{query}'")

        # Initial search and analysis
        original_analysis = self.query_analyzer.analyze_query(query)
        initial_response = await self.search_engine.search(
            query, target_result_count, filters
        )

        # Track refinement session
        session = RefinementSession(
            original_query=query,
            original_analysis=original_analysis,
            refinement_steps=[],
            final_response=initial_response,
            total_improvement=0.0,
            session_confidence=initial_response.search_confidence,
        )

        # Check if initial results are satisfactory
        if self._is_result_satisfactory(initial_response, original_analysis):
            logger.info("Initial results satisfactory, no refinement needed")
            return session

        # Progressive refinement loop
        current_query = query
        current_filters = filters or {}
        current_response = initial_response
        base_quality = initial_response.result_quality_score

        for step_num in range(1, self.max_refinement_steps + 1):
            logger.info(f"Refinement step {step_num}")

            # Analyze current results and determine refinement strategy
            refinement_strategy = self._select_refinement_strategy(
                current_response, original_analysis, step_num
            )

            if not refinement_strategy:
                logger.info("No beneficial refinement strategy found")
                break

            # Apply refinement
            refined_query, refined_filters, strategies = await self._apply_refinement(
                refinement_strategy,
                current_query,
                current_filters,
                current_response,
                original_analysis,
            )

            # Execute refined search
            refined_response = await self.search_engine.search(
                refined_query, target_result_count, refined_filters
            )

            # Evaluate improvement
            improvement_score = self._calculate_improvement(
                current_response, refined_response
            )
            step_confidence = self._calculate_step_confidence(
                improvement_score, refined_response, original_analysis
            )

            # Create refinement step record
            step = RefinementStep(
                step_number=step_num,
                refinement_type=refinement_strategy,
                original_query=current_query,
                refined_query=refined_query,
                filters_applied=refined_filters,
                strategies_used=strategies,
                improvement_score=improvement_score,
                confidence=step_confidence,
                reasoning=self._generate_step_reasoning(
                    refinement_strategy,
                    improvement_score,
                    current_response,
                    refined_response,
                ),
            )

            session.refinement_steps.append(step)

            # Check if refinement was beneficial
            if improvement_score > self.min_improvement_threshold:
                current_query = refined_query
                current_filters = refined_filters
                current_response = refined_response
                session.total_improvement += improvement_score
                session.final_response = refined_response
                session.session_confidence = step_confidence

                logger.info(
                    f"Step {step_num} improved results by {improvement_score:.3f}"
                )

                # Check if we've reached satisfactory results
                if self._is_result_satisfactory(refined_response, original_analysis):
                    logger.info("Satisfactory results achieved")
                    break
            else:
                logger.info(f"Step {step_num} did not improve results significantly")
                # Don't apply this refinement
                break

        logger.info(
            f"Refinement complete: {len(session.refinement_steps)} steps, "
            f"total improvement: {session.total_improvement:.3f}"
        )

        return session

    def _is_result_satisfactory(
        self, response: SearchResponse, original_analysis: QueryAnalysis
    ) -> bool:
        """Determine if search results are satisfactory."""

        # Quality thresholds
        min_quality_score = 0.7
        min_confidence = 0.8
        min_results = 3

        # Check basic metrics
        if (
            response.result_quality_score >= min_quality_score
            and response.search_confidence >= min_confidence
            and len(response.results) >= min_results
        ):
            return True

        # Content-specific satisfaction criteria
        if original_analysis.query_type.value in ["code_function", "api_reference"]:
            # For specific queries, prioritize precision over quantity
            if response.result_quality_score >= 0.8 and len(response.results) >= 2:
                return True

        return False

    def _select_refinement_strategy(
        self,
        current_response: SearchResponse,
        original_analysis: QueryAnalysis,
        step_number: int,
    ) -> Optional[RefinementType]:
        """Select the most appropriate refinement strategy."""

        # Analyze current search issues
        issues = self._identify_search_issues(current_response, original_analysis)

        # Strategy selection based on issues and step number
        if "insufficient_results" in issues:
            if step_number == 1:
                return RefinementType.QUERY_EXPANSION
            else:
                return RefinementType.FILTER_MODIFICATION

        elif "poor_relevance" in issues:
            if original_analysis.specificity_score < 0.4:
                return RefinementType.QUERY_NARROWING
            else:
                return RefinementType.SEMANTIC_ENHANCEMENT

        elif "wrong_content_type" in issues:
            return RefinementType.STRATEGY_ADJUSTMENT

        elif "low_quality" in issues:
            return RefinementType.SEMANTIC_ENHANCEMENT

        # No clear strategy identified
        return None

    def _identify_search_issues(
        self, response: SearchResponse, original_analysis: QueryAnalysis
    ) -> List[str]:
        """Identify specific issues with current search results."""

        issues = []

        # Insufficient results
        if len(response.results) < 3:
            issues.append("insufficient_results")

        # Poor relevance scores
        if response.results and max(r.relevance_score for r in response.results) < 0.6:
            issues.append("poor_relevance")

        # Low overall quality
        if response.result_quality_score < 0.6:
            issues.append("low_quality")

        # Content type mismatch
        expected_types = self._get_expected_content_types(original_analysis)
        actual_types = set(r.content_type for r in response.results)
        if expected_types and not any(t in actual_types for t in expected_types):
            issues.append("wrong_content_type")

        # Language mismatch
        if original_analysis.programming_language and not any(
            r.programming_language == original_analysis.programming_language
            for r in response.results
        ):
            issues.append("language_mismatch")

        return issues

    def _get_expected_content_types(self, analysis: QueryAnalysis) -> List[str]:
        """Get expected content types based on query analysis."""

        type_mapping = {
            "code_function": ["code_example", "api_reference"],
            "code_class": ["code_example", "api_reference"],
            "code_pattern": ["code_example"],
            "api_reference": ["api_reference"],
            "tutorial": ["tutorial"],
            "conceptual": ["tutorial", "general"],
        }

        return type_mapping.get(analysis.query_type.value, [])

    async def _apply_refinement(
        self,
        strategy: RefinementType,
        query: str,
        filters: Dict[str, Any],
        response: SearchResponse,
        original_analysis: QueryAnalysis,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Apply the selected refinement strategy."""

        if strategy == RefinementType.QUERY_EXPANSION:
            return await self._apply_query_expansion(query, response, original_analysis)

        elif strategy == RefinementType.QUERY_NARROWING:
            return await self._apply_query_narrowing(query, response, original_analysis)

        elif strategy == RefinementType.STRATEGY_ADJUSTMENT:
            return await self._apply_strategy_adjustment(
                query, filters, response, original_analysis
            )

        elif strategy == RefinementType.FILTER_MODIFICATION:
            return await self._apply_filter_modification(
                query, filters, response, original_analysis
            )

        elif strategy == RefinementType.SEMANTIC_ENHANCEMENT:
            return await self._apply_semantic_enhancement(
                query, filters, response, original_analysis
            )

        else:
            return query, filters, []

    async def _apply_query_expansion(
        self, query: str, response: SearchResponse, original_analysis: QueryAnalysis
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Expand query with related terms."""

        expansion_terms = []

        # Use original expansion terms
        expansion_terms.extend(original_analysis.expansion_terms[:3])

        # Extract terms from existing results
        if response.results:
            # Get common keywords from top results
            top_results = response.results[:3]
            for result in top_results:
                if result.matched_keywords:
                    expansion_terms.extend(result.matched_keywords[:2])

        # Create expanded query
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms[:5])}"
        else:
            expanded_query = query

        return expanded_query, {}, ["semantic_search", "hybrid_search"]

    async def _apply_query_narrowing(
        self, query: str, response: SearchResponse, original_analysis: QueryAnalysis
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Narrow query by adding specific constraints."""

        narrowed_query = query

        # Add programming language if detected
        if original_analysis.programming_language:
            narrowed_query = f"{original_analysis.programming_language} {query}"

        # Add content type specifier
        if original_analysis.query_type.value in ["code_function", "api_reference"]:
            narrowed_query = f"{narrowed_query} example"

        # Add specific code elements
        if original_analysis.code_elements:
            code_element = original_analysis.code_elements[0]
            narrowed_query = f"{narrowed_query} {code_element}"

        return narrowed_query, {}, ["semantic_code_search", "structured_search"]

    async def _apply_strategy_adjustment(
        self,
        query: str,
        filters: Dict[str, Any],
        response: SearchResponse,
        original_analysis: QueryAnalysis,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Adjust search strategies based on content type needs."""

        # Determine new strategies based on query type
        if original_analysis.query_type.value == "api_reference":
            new_strategies = ["api_search", "structured_search"]
        elif original_analysis.query_type.value in ["code_function", "code_class"]:
            new_strategies = ["semantic_code_search", "language_specific_search"]
        elif original_analysis.query_type.value == "tutorial":
            new_strategies = ["tutorial_prioritized_search", "hierarchical_search"]
        else:
            new_strategies = ["semantic_search", "hybrid_search"]

        return query, filters, new_strategies

    async def _apply_filter_modification(
        self,
        query: str,
        filters: Dict[str, Any],
        response: SearchResponse,
        original_analysis: QueryAnalysis,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Modify search filters to improve results."""

        new_filters = filters.copy()

        # Add language filter
        if original_analysis.programming_language:
            new_filters["primary_language"] = original_analysis.programming_language

        # Add content type filter
        expected_types = self._get_expected_content_types(original_analysis)
        if expected_types:
            new_filters["content_type"] = expected_types[0]

        # Adjust code percentage filter
        if original_analysis.code_elements:
            new_filters["min_code_percentage"] = 10

        # Remove overly restrictive filters if too few results
        if len(response.results) < 2:
            new_filters.pop("min_code_percentage", None)
            new_filters.pop("content_type", None)

        return query, new_filters, []

    async def _apply_semantic_enhancement(
        self,
        query: str,
        filters: Dict[str, Any],
        response: SearchResponse,
        original_analysis: QueryAnalysis,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Enhance semantic understanding of the query."""

        # Add semantic context terms
        context_terms = []

        if original_analysis.search_intent.value == "learning":
            context_terms.extend(["tutorial", "guide", "example"])
        elif original_analysis.search_intent.value == "implementation":
            context_terms.extend(["implementation", "code", "example"])
        elif original_analysis.search_intent.value == "debugging":
            context_terms.extend(["solution", "fix", "error"])

        # Enhance query with context
        if context_terms:
            enhanced_query = f"{query} {context_terms[0]}"
        else:
            enhanced_query = query

        return enhanced_query, filters, ["hierarchical_search", "semantic_search"]

    def _calculate_improvement(
        self, before: SearchResponse, after: SearchResponse
    ) -> float:
        """Calculate improvement score between two search responses."""

        # Quality improvement
        quality_improvement = after.result_quality_score - before.result_quality_score

        # Confidence improvement
        confidence_improvement = after.search_confidence - before.search_confidence

        # Result count improvement (with diminishing returns)
        count_before = len(before.results)
        count_after = len(after.results)
        count_improvement = 0.0

        if count_before < 3 and count_after > count_before:
            count_improvement = min((count_after - count_before) / 5.0, 0.2)

        # Relevance improvement (top result)
        relevance_improvement = 0.0
        if before.results and after.results:
            top_before = max(r.relevance_score for r in before.results)
            top_after = max(r.relevance_score for r in after.results)
            relevance_improvement = (top_after - top_before) * 0.5

        # Combine improvements
        total_improvement = (
            quality_improvement * 0.4
            + confidence_improvement * 0.3
            + count_improvement * 0.2
            + relevance_improvement * 0.1
        )

        return total_improvement

    def _calculate_step_confidence(
        self,
        improvement_score: float,
        response: SearchResponse,
        original_analysis: QueryAnalysis,
    ) -> float:
        """Calculate confidence in the refinement step."""

        base_confidence = response.search_confidence

        # Boost confidence based on improvement
        if improvement_score > 0.2:
            base_confidence *= 1.1
        elif improvement_score < 0:
            base_confidence *= 0.9

        # Factor in result quality
        base_confidence *= 0.8 + 0.2 * response.result_quality_score

        return min(base_confidence, 1.0)

    def _generate_step_reasoning(
        self,
        strategy: RefinementType,
        improvement_score: float,
        before: SearchResponse,
        after: SearchResponse,
    ) -> str:
        """Generate human-readable reasoning for the refinement step."""

        if improvement_score > 0.2:
            outcome = "significantly improved"
        elif improvement_score > 0.1:
            outcome = "improved"
        elif improvement_score > 0:
            outcome = "slightly improved"
        else:
            outcome = "did not improve"

        strategy_descriptions = {
            RefinementType.QUERY_EXPANSION: "expanded query with related terms",
            RefinementType.QUERY_NARROWING: "narrowed query with specific constraints",
            RefinementType.STRATEGY_ADJUSTMENT: "adjusted search strategies",
            RefinementType.FILTER_MODIFICATION: "modified search filters",
            RefinementType.SEMANTIC_ENHANCEMENT: "enhanced semantic understanding",
        }

        strategy_desc = strategy_descriptions.get(strategy, "applied refinement")

        result_change = f"from {len(before.results)} to {len(after.results)} results"
        quality_change = f"quality score from {before.result_quality_score:.2f} to {after.result_quality_score:.2f}"

        return f"Applied {strategy_desc} which {outcome} search results ({result_change}, {quality_change})"


class AdaptiveSearchOrchestrator:
    """High-level orchestrator that combines search and refinement."""

    def __init__(self, search_engine: SearchEngine):
        """Initialize orchestrator with search engine."""
        self.search_engine = search_engine
        self.refinement_engine = ProgressiveRefinementEngine(search_engine)

    async def intelligent_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        target_quality: float = 0.8,
        max_iterations: int = 3,
    ) -> RefinementSession:
        """Execute intelligent search with automatic refinement."""

        logger.info(
            f"Starting intelligent search: '{query}' (target quality: {target_quality})"
        )

        # Execute progressive refinement
        session = await self.refinement_engine.refine_search(
            query, filters, target_result_count=10
        )

        # Check if target quality achieved
        if session.final_response.result_quality_score >= target_quality:
            logger.info(
                f"Target quality achieved: {session.final_response.result_quality_score:.3f}"
            )
        else:
            logger.info(
                f"Target quality not reached: {session.final_response.result_quality_score:.3f}"
            )

        return session

    async def explain_search_process(
        self, session: RefinementSession
    ) -> Dict[str, Any]:
        """Generate explanation of the search process for transparency."""

        explanation = {
            "original_query": session.original_query,
            "query_understanding": {
                "type": session.original_analysis.query_type.value,
                "intent": session.original_analysis.search_intent.value,
                "confidence": session.original_analysis.confidence,
                "programming_language": session.original_analysis.programming_language,
                "complexity": session.original_analysis.complexity_score,
            },
            "refinement_process": [],
            "final_outcome": {
                "total_steps": len(session.refinement_steps),
                "total_improvement": session.total_improvement,
                "final_quality": session.final_response.result_quality_score,
                "final_confidence": session.session_confidence,
                "result_count": len(session.final_response.results),
            },
        }

        # Add step-by-step refinement details
        for step in session.refinement_steps:
            step_info = {
                "step_number": step.step_number,
                "strategy": step.refinement_type.value,
                "reasoning": step.reasoning,
                "improvement": step.improvement_score,
                "confidence": step.confidence,
            }
            explanation["refinement_process"].append(step_info)

        return explanation
