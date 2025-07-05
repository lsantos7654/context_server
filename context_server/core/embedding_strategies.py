"""Enhanced embedding strategies with summary embeddings and composite approaches."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .content_analysis import ContentAnalysis
from .multi_embedding_service import EmbeddingModel, MultiEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingStrategy(Enum):
    """Available embedding strategies."""

    SINGLE = "single"  # Single optimal embedding per chunk
    MULTI_MODEL = "multi_model"  # Multiple models per chunk
    HIERARCHICAL = "hierarchical"  # Document + chunk embeddings
    SUMMARY_ENHANCED = "summary_enhanced"  # Summary + chunk embeddings
    COMPOSITE = "composite"  # Combined approach with weighted vectors
    ADAPTIVE = "adaptive"  # Strategy chosen based on content analysis


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""

    embedding: List[float]
    model: str
    dimension: int
    strategy: str
    metadata: Dict[str, Any]
    confidence: float = 1.0
    fallback_used: bool = False


@dataclass
class HierarchicalEmbedding:
    """Hierarchical embedding structure."""

    document_embedding: EmbeddingResult
    summary_embedding: Optional[EmbeddingResult] = None
    chunk_embeddings: List[EmbeddingResult] = None
    section_embeddings: List[EmbeddingResult] = None
    composite_embedding: Optional[EmbeddingResult] = None


@dataclass
class EmbeddingQualityMetrics:
    """Metrics for embedding quality assessment."""

    coherence_score: float  # How well chunks relate to document
    diversity_score: float  # Embedding vector diversity
    coverage_score: float  # Content coverage quality
    consistency_score: float  # Cross-model consistency
    confidence_score: float  # Overall confidence
    model_agreement: Dict[str, float]  # Agreement between models


class SummaryGenerator:
    """Generates intelligent summaries for embedding."""

    def __init__(self):
        self.max_summary_length = 500
        self.min_summary_length = 100

    def generate_document_summary(
        self, content: str, content_analysis: ContentAnalysis = None, title: str = None
    ) -> str:
        """Generate document-level summary optimized for embeddings."""

        # Start with title if available
        summary_parts = []
        if title:
            summary_parts.append(f"Document: {title}")

        # Add content type and language context
        if content_analysis:
            if content_analysis.content_type != "general":
                summary_parts.append(f"Type: {content_analysis.content_type}")

            if content_analysis.primary_language:
                summary_parts.append(f"Language: {content_analysis.primary_language}")

            # Add existing summary if good quality
            if content_analysis.summary and len(content_analysis.summary) > 50:
                summary_parts.append(content_analysis.summary)

            # Add key concepts
            if content_analysis.key_concepts:
                concepts = ", ".join(content_analysis.key_concepts[:5])
                summary_parts.append(f"Key concepts: {concepts}")

            # Add code-specific information
            if content_analysis.code_percentage > 20:
                summary_parts.append(
                    f"Contains {content_analysis.code_percentage:.0f}% code"
                )

                if content_analysis.code_blocks:
                    functions = []
                    classes = []
                    for block in content_analysis.code_blocks:
                        functions.extend(
                            block.functions[:3]
                        )  # Top 3 functions per block
                        classes.extend(block.classes[:2])  # Top 2 classes per block

                    if functions:
                        summary_parts.append(f"Functions: {', '.join(functions[:8])}")
                    if classes:
                        summary_parts.append(f"Classes: {', '.join(classes[:5])}")

        # Extract key sentences from content if summary is too short
        summary_text = " | ".join(summary_parts)
        if len(summary_text) < self.min_summary_length:
            key_sentences = self._extract_key_sentences(content, 3)
            if key_sentences:
                summary_text += " | " + " ".join(key_sentences)

        # Truncate if too long
        if len(summary_text) > self.max_summary_length:
            summary_text = summary_text[: self.max_summary_length - 3] + "..."

        return summary_text

    def generate_section_summary(
        self, content: str, section_title: str = None, chunk_contents: List[str] = None
    ) -> str:
        """Generate section-level summary for hierarchical embeddings."""

        summary_parts = []

        if section_title:
            summary_parts.append(f"Section: {section_title}")

        # Extract key information from chunk contents
        if chunk_contents:
            # Find common themes/patterns across chunks
            all_content = " ".join(chunk_contents)
            key_sentences = self._extract_key_sentences(all_content, 2)
            summary_parts.extend(key_sentences)

        summary = " | ".join(summary_parts)

        # Limit length
        if len(summary) > 300:
            summary = summary[:297] + "..."

        return summary

    def _extract_key_sentences(self, content: str, max_sentences: int = 3) -> List[str]:
        """Extract key sentences from content using simple heuristics."""

        sentences = content.split(". ")
        if len(sentences) <= max_sentences:
            return sentences

        # Score sentences by length and position (favor beginning)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 200:  # Skip very short/long
                continue

            # Simple scoring: favor earlier sentences and moderate length
            position_score = max(0.1, 1.0 - (i / len(sentences)))
            length_score = min(1.0, len(sentence) / 100)  # Optimal around 100 chars

            score = position_score * length_score
            scored_sentences.append((score, sentence))

        # Return top sentences
        scored_sentences.sort(reverse=True)
        return [sent for _, sent in scored_sentences[:max_sentences]]


class CompositeEmbeddingGenerator:
    """Generates composite embeddings from multiple sources."""

    def __init__(self, multi_embedding_service: MultiEmbeddingService):
        self.embedding_service = multi_embedding_service
        self.summary_generator = SummaryGenerator()

    async def generate_hierarchical_embedding(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis = None,
        strategy_config: Dict = None,
    ) -> HierarchicalEmbedding:
        """Generate hierarchical embeddings at multiple levels."""

        config = strategy_config or {}

        # Generate document summary
        doc_summary = self.summary_generator.generate_document_summary(
            content, content_analysis, title
        )

        # Generate embeddings concurrently
        embedding_tasks = []

        # Document-level embedding (full content)
        doc_task = self._embed_with_strategy(
            content, content_analysis, "document", config
        )
        embedding_tasks.append(("document", doc_task))

        # Summary embedding
        summary_task = self._embed_with_strategy(
            doc_summary, content_analysis, "summary", config
        )
        embedding_tasks.append(("summary", summary_task))

        # Chunk embeddings
        for i, chunk in enumerate(chunks):
            chunk_task = self._embed_with_strategy(
                chunk, content_analysis, f"chunk_{i}", config
            )
            embedding_tasks.append((f"chunk_{i}", chunk_task))

        # Wait for all embeddings
        results = {}
        for name, task in embedding_tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Failed to generate {name} embedding: {e}")
                results[name] = None

        # Generate composite embedding if enabled
        composite_embedding = None
        if config.get("generate_composite", True):
            composite_embedding = await self._generate_composite_embedding(
                results, content_analysis, config
            )

        return HierarchicalEmbedding(
            document_embedding=results.get("document"),
            summary_embedding=results.get("summary"),
            chunk_embeddings=[
                results[k] for k in results.keys() if k.startswith("chunk_")
            ],
            composite_embedding=composite_embedding,
        )

    async def _embed_with_strategy(
        self, text: str, content_analysis: ContentAnalysis, level: str, config: Dict
    ) -> EmbeddingResult:
        """Generate embedding using specified strategy."""

        strategy = config.get("strategy", EmbeddingStrategy.ADAPTIVE)

        if strategy == EmbeddingStrategy.ADAPTIVE:
            # Choose strategy based on content and level
            if level == "summary" or (
                content_analysis and content_analysis.code_percentage < 10
            ):
                model = EmbeddingModel.OPENAI_SMALL
            else:
                model = self.embedding_service.route_content(content_analysis)
        else:
            model = config.get("force_model", EmbeddingModel.OPENAI_SMALL)

        try:
            result = await self.embedding_service.embed_text(
                text, content_analysis, force_model=model
            )

            return EmbeddingResult(
                embedding=result["embedding"],
                model=result["model"],
                dimension=result["dimension"],
                strategy=(
                    strategy.value
                    if isinstance(strategy, EmbeddingStrategy)
                    else str(strategy)
                ),
                metadata={
                    "level": level,
                    "text_length": len(text),
                    "success": result["success"],
                },
                fallback_used=result.get("used_fallback", False),
            )

        except Exception as e:
            logger.error(f"Embedding generation failed for {level}: {e}")
            raise

    async def _generate_composite_embedding(
        self,
        embeddings: Dict[str, EmbeddingResult],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> EmbeddingResult:
        """Generate composite embedding from multiple sources."""

        # Filter valid embeddings
        valid_embeddings = {k: v for k, v in embeddings.items() if v is not None}

        if not valid_embeddings:
            raise ValueError("No valid embeddings to compose")

        # Define weights based on content type
        weights = self._calculate_composite_weights(
            valid_embeddings, content_analysis, config
        )

        # Find common dimension (use the most common)
        dimensions = [emb.dimension for emb in valid_embeddings.values()]
        target_dim = max(set(dimensions), key=dimensions.count)

        # Normalize and weight embeddings
        weighted_embeddings = []
        total_weight = 0

        for name, embedding in valid_embeddings.items():
            if embedding.dimension != target_dim:
                # Skip incompatible dimensions for now
                # In production, could implement dimension transformation
                continue

            weight = weights.get(name, 0.1)
            normalized = np.array(embedding.embedding) / np.linalg.norm(
                embedding.embedding
            )
            weighted_embeddings.append(normalized * weight)
            total_weight += weight

        if not weighted_embeddings:
            raise ValueError("No compatible embeddings for composition")

        # Generate composite vector
        composite_vector = sum(weighted_embeddings) / total_weight
        composite_vector = composite_vector / np.linalg.norm(composite_vector)

        return EmbeddingResult(
            embedding=composite_vector.tolist(),
            model="composite",
            dimension=target_dim,
            strategy="composite",
            metadata={
                "component_count": len(weighted_embeddings),
                "weights": weights,
                "source_models": [emb.model for emb in valid_embeddings.values()],
            },
            confidence=min([emb.confidence for emb in valid_embeddings.values()]),
        )

    def _calculate_composite_weights(
        self,
        embeddings: Dict[str, EmbeddingResult],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, float]:
        """Calculate weights for composite embedding generation."""

        weights = {}

        # Base weights
        base_weights = {
            "document": 0.4,  # Full document context
            "summary": 0.3,  # Key information distilled
        }

        # Add chunk weights
        chunk_count = len([k for k in embeddings.keys() if k.startswith("chunk_")])
        if chunk_count > 0:
            chunk_weight = 0.3 / chunk_count  # Distribute chunk weight
            for k in embeddings.keys():
                if k.startswith("chunk_"):
                    base_weights[k] = chunk_weight

        # Adjust weights based on content analysis
        if content_analysis:
            if content_analysis.code_percentage > 50:
                # For code-heavy content, favor document embedding
                base_weights["document"] = 0.5
                base_weights["summary"] = 0.2
            elif content_analysis.content_type == "tutorial":
                # For tutorials, favor summary
                base_weights["summary"] = 0.4
                base_weights["document"] = 0.3

        # Apply user config overrides
        user_weights = config.get("composite_weights", {})
        weights.update(base_weights)
        weights.update(user_weights)

        return weights


class EmbeddingQualityAnalyzer:
    """Analyzes and measures embedding quality."""

    def __init__(self):
        self.coherence_threshold = 0.7
        self.diversity_threshold = 0.3

    async def analyze_embedding_quality(
        self,
        hierarchical_embedding: HierarchicalEmbedding,
        content: str,
        chunks: List[str],
    ) -> EmbeddingQualityMetrics:
        """Analyze quality of generated embeddings."""

        # Calculate coherence (how well chunks relate to document)
        coherence_score = self._calculate_coherence(
            hierarchical_embedding.document_embedding,
            hierarchical_embedding.chunk_embeddings,
        )

        # Calculate diversity (embedding vector diversity)
        diversity_score = self._calculate_diversity(
            hierarchical_embedding.chunk_embeddings
        )

        # Calculate coverage (how well embeddings cover content)
        coverage_score = self._calculate_coverage(
            hierarchical_embedding, content, chunks
        )

        # Calculate consistency across models
        consistency_score = self._calculate_consistency(
            hierarchical_embedding.chunk_embeddings
        )

        # Calculate model agreement
        model_agreement = self._calculate_model_agreement(
            hierarchical_embedding.chunk_embeddings
        )

        # Overall confidence
        confidence_score = (coherence_score + coverage_score + consistency_score) / 3

        return EmbeddingQualityMetrics(
            coherence_score=coherence_score,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            confidence_score=confidence_score,
            model_agreement=model_agreement,
        )

    def _calculate_coherence(
        self, doc_embedding: EmbeddingResult, chunk_embeddings: List[EmbeddingResult]
    ) -> float:
        """Calculate coherence between document and chunk embeddings."""

        if not doc_embedding or not chunk_embeddings:
            return 0.0

        doc_vec = np.array(doc_embedding.embedding)
        similarities = []

        for chunk_emb in chunk_embeddings:
            if chunk_emb and chunk_emb.dimension == doc_embedding.dimension:
                chunk_vec = np.array(chunk_emb.embedding)
                similarity = np.dot(doc_vec, chunk_vec) / (
                    np.linalg.norm(doc_vec) * np.linalg.norm(chunk_vec)
                )
                similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else 0.0

    def _calculate_diversity(self, chunk_embeddings: List[EmbeddingResult]) -> float:
        """Calculate diversity among chunk embeddings."""

        if len(chunk_embeddings) < 2:
            return 1.0

        valid_embeddings = [emb for emb in chunk_embeddings if emb is not None]
        if len(valid_embeddings) < 2:
            return 1.0

        similarities = []
        for i in range(len(valid_embeddings)):
            for j in range(i + 1, len(valid_embeddings)):
                emb1, emb2 = valid_embeddings[i], valid_embeddings[j]
                if emb1.dimension == emb2.dimension:
                    vec1 = np.array(emb1.embedding)
                    vec2 = np.array(emb2.embedding)
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    similarities.append(similarity)

        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity
        return float(max(0.0, diversity))  # Ensure non-negative

    def _calculate_coverage(
        self,
        hierarchical_embedding: HierarchicalEmbedding,
        content: str,
        chunks: List[str],
    ) -> float:
        """Calculate how well embeddings cover the content."""

        # Simple heuristic: if we have embeddings at multiple levels, coverage is good
        coverage_factors = []

        # Document level
        if hierarchical_embedding.document_embedding:
            coverage_factors.append(0.4)

        # Summary level
        if hierarchical_embedding.summary_embedding:
            coverage_factors.append(0.3)

        # Chunk level
        chunk_coverage = len(
            [emb for emb in (hierarchical_embedding.chunk_embeddings or []) if emb]
        ) / max(1, len(chunks))
        coverage_factors.append(chunk_coverage * 0.3)

        return float(sum(coverage_factors))

    def _calculate_consistency(self, chunk_embeddings: List[EmbeddingResult]) -> float:
        """Calculate consistency of embeddings from same model."""

        if not chunk_embeddings:
            return 1.0

        # Group by model
        model_groups = {}
        for emb in chunk_embeddings:
            if emb:
                if emb.model not in model_groups:
                    model_groups[emb.model] = []
                model_groups[emb.model].append(emb)

        # Calculate consistency within each model group
        model_consistencies = []
        for model, embeddings in model_groups.items():
            if len(embeddings) >= 2:
                # Calculate average pairwise similarity within model
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        emb1, emb2 = embeddings[i], embeddings[j]
                        vec1 = np.array(emb1.embedding)
                        vec2 = np.array(emb2.embedding)
                        similarity = np.dot(vec1, vec2) / (
                            np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        )
                        similarities.append(similarity)

                if similarities:
                    model_consistencies.append(np.mean(similarities))

        result = float(np.mean(model_consistencies)) if model_consistencies else 1.0
        return min(1.0, result)  # Clamp to avoid floating point precision issues

    def _calculate_model_agreement(
        self, chunk_embeddings: List[EmbeddingResult]
    ) -> Dict[str, float]:
        """Calculate agreement between different models."""

        model_agreement = {}

        # Group embeddings by model
        model_groups = {}
        for emb in chunk_embeddings:
            if emb:
                if emb.model not in model_groups:
                    model_groups[emb.model] = []
                model_groups[emb.model].append(emb.embedding)

        models = list(model_groups.keys())

        # Calculate agreement between each pair of models
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                embeddings1 = model_groups[model1]
                embeddings2 = model_groups[model2]

                # Compare embeddings at same positions
                similarities = []
                for k in range(min(len(embeddings1), len(embeddings2))):
                    # Note: This is simplified - in practice need to handle dimension differences
                    if len(embeddings1[k]) == len(embeddings2[k]):
                        vec1 = np.array(embeddings1[k])
                        vec2 = np.array(embeddings2[k])
                        similarity = np.dot(vec1, vec2) / (
                            np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        )
                        similarities.append(similarity)

                if similarities:
                    agreement_key = f"{model1}_vs_{model2}"
                    model_agreement[agreement_key] = float(np.mean(similarities))

        return model_agreement


class EnhancedEmbeddingStrategy:
    """Main orchestrator for enhanced embedding strategies."""

    def __init__(self, multi_embedding_service: MultiEmbeddingService):
        self.embedding_service = multi_embedding_service
        self.composite_generator = CompositeEmbeddingGenerator(multi_embedding_service)
        self.quality_analyzer = EmbeddingQualityAnalyzer()
        self.summary_generator = SummaryGenerator()

    async def generate_enhanced_embeddings(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        strategy: EmbeddingStrategy = EmbeddingStrategy.ADAPTIVE,
        config: Dict = None,
    ) -> Dict[str, Any]:
        """Generate embeddings using enhanced strategies."""

        config = config or {}

        if strategy == EmbeddingStrategy.SINGLE:
            return await self._generate_single_strategy(
                content, chunks, content_analysis, config
            )
        elif strategy == EmbeddingStrategy.MULTI_MODEL:
            return await self._generate_multi_model_strategy(
                content, chunks, content_analysis, config
            )
        elif strategy == EmbeddingStrategy.HIERARCHICAL:
            return await self._generate_hierarchical_strategy(
                content, title, chunks, content_analysis, config
            )
        elif strategy == EmbeddingStrategy.SUMMARY_ENHANCED:
            return await self._generate_summary_enhanced_strategy(
                content, title, chunks, content_analysis, config
            )
        elif strategy == EmbeddingStrategy.COMPOSITE:
            return await self._generate_composite_strategy(
                content, title, chunks, content_analysis, config
            )
        elif strategy == EmbeddingStrategy.ADAPTIVE:
            return await self._generate_adaptive_strategy(
                content, title, chunks, content_analysis, config
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _generate_adaptive_strategy(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Adaptively choose the best strategy based on content."""

        # Analyze content to choose optimal strategy
        if content_analysis.code_percentage > 40:
            # Code-heavy content benefits from hierarchical approach
            chosen_strategy = EmbeddingStrategy.HIERARCHICAL
        elif len(chunks) > 10:
            # Long documents benefit from summary enhancement
            chosen_strategy = EmbeddingStrategy.SUMMARY_ENHANCED
        elif content_analysis.content_type in ["api_reference", "tutorial"]:
            # Structured content benefits from composite approach
            chosen_strategy = EmbeddingStrategy.COMPOSITE
        else:
            # Default to hierarchical for comprehensive coverage
            chosen_strategy = EmbeddingStrategy.HIERARCHICAL

        logger.info(
            f"Adaptive strategy chose: {chosen_strategy.value} for content type: {content_analysis.content_type}"
        )

        # Apply chosen strategy
        result = await self.generate_enhanced_embeddings(
            content, title, chunks, content_analysis, chosen_strategy, config
        )

        # Add adaptive metadata
        result["adaptive_choice"] = chosen_strategy.value
        result["adaptive_reasoning"] = {
            "code_percentage": content_analysis.code_percentage,
            "chunk_count": len(chunks),
            "content_type": content_analysis.content_type,
        }

        return result

    async def _generate_hierarchical_strategy(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Generate hierarchical embeddings."""

        hierarchical_embedding = (
            await self.composite_generator.generate_hierarchical_embedding(
                content, title, chunks, content_analysis, config
            )
        )

        # Analyze quality
        quality_metrics = await self.quality_analyzer.analyze_embedding_quality(
            hierarchical_embedding, content, chunks
        )

        return {
            "strategy": "hierarchical",
            "hierarchical_embedding": hierarchical_embedding,
            "quality_metrics": quality_metrics,
            "primary_embedding": hierarchical_embedding.document_embedding,
            "summary_embedding": hierarchical_embedding.summary_embedding,
            "chunk_embeddings": hierarchical_embedding.chunk_embeddings,
            "composite_embedding": hierarchical_embedding.composite_embedding,
        }

    async def _generate_summary_enhanced_strategy(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Generate summary-enhanced embeddings."""

        # Generate enhanced summary
        enhanced_summary = self.summary_generator.generate_document_summary(
            content, content_analysis, title
        )

        # Generate embeddings for summary and chunks
        summary_result = await self.embedding_service.embed_text(
            enhanced_summary, content_analysis
        )

        chunk_results = await self.embedding_service.embed_batch(
            chunks, content_analyses=[content_analysis] * len(chunks)
        )

        return {
            "strategy": "summary_enhanced",
            "enhanced_summary": enhanced_summary,
            "summary_embedding": summary_result,
            "chunk_embeddings": chunk_results,
            "primary_embedding": summary_result,  # Use summary as primary
        }

    async def _generate_composite_strategy(
        self,
        content: str,
        title: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Generate composite embeddings with advanced weighting."""

        # Force composite generation
        config["generate_composite"] = True

        hierarchical_embedding = (
            await self.composite_generator.generate_hierarchical_embedding(
                content, title, chunks, content_analysis, config
            )
        )

        return {
            "strategy": "composite",
            "composite_embedding": hierarchical_embedding.composite_embedding,
            "component_embeddings": {
                "document": hierarchical_embedding.document_embedding,
                "summary": hierarchical_embedding.summary_embedding,
                "chunks": hierarchical_embedding.chunk_embeddings,
            },
            "primary_embedding": hierarchical_embedding.composite_embedding,
        }

    async def _generate_single_strategy(
        self,
        content: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Generate single optimal embeddings (baseline)."""

        chunk_results = await self.embedding_service.embed_batch(
            chunks, content_analyses=[content_analysis] * len(chunks)
        )

        return {
            "strategy": "single",
            "chunk_embeddings": chunk_results,
            "primary_embedding": chunk_results[0] if chunk_results else None,
        }

    async def _generate_multi_model_strategy(
        self,
        content: str,
        chunks: List[str],
        content_analysis: ContentAnalysis,
        config: Dict,
    ) -> Dict[str, Any]:
        """Generate multi-model embeddings for comparison."""

        models = config.get(
            "models", [EmbeddingModel.OPENAI_SMALL, EmbeddingModel.COHERE_CODE]
        )

        multi_model_results = {}
        for model in models:
            model_results = await self.embedding_service.embed_batch(
                chunks, force_model=model
            )
            multi_model_results[model.value] = model_results

        return {
            "strategy": "multi_model",
            "multi_model_embeddings": multi_model_results,
            "primary_embedding": (
                list(multi_model_results.values())[0][0]
                if multi_model_results
                else None
            ),
        }
