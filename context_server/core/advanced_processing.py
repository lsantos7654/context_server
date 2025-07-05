"""Advanced document processing with enhanced embedding strategies."""

import asyncio
import logging

# Import existing extraction functionality
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append("/app/src")
from src.core.crawl4ai_extraction import Crawl4aiExtractor

from .chunking import TextChunker
from .content_analysis import ContentAnalysis, ContentAnalyzer
from .embedding_strategies import (
    EmbeddingQualityMetrics,
    EmbeddingResult,
    EmbeddingStrategy,
    EnhancedEmbeddingStrategy,
    HierarchicalEmbedding,
)
from .enhanced_processing import EnhancedProcessedChunk, EnhancedProcessedDocument
from .multi_embedding_service import MultiEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class AdvancedProcessedChunk:
    """Advanced processed chunk with enhanced embedding strategies."""

    content: str
    primary_embedding: EmbeddingResult
    strategy_embeddings: Dict[str, Any]  # Results from different strategies
    quality_metrics: EmbeddingQualityMetrics
    metadata: dict
    tokens: int
    start_line: int = None
    end_line: int = None
    char_start: int = None
    char_end: int = None
    content_analysis: ContentAnalysis = None


@dataclass
class AdvancedProcessedDocument:
    """Advanced processed document with hierarchical embeddings."""

    url: str
    title: str
    content: str
    chunks: List[AdvancedProcessedChunk]
    metadata: dict
    content_analysis: ContentAnalysis = None

    # Hierarchical embeddings
    document_embedding: EmbeddingResult = None
    summary_embedding: EmbeddingResult = None
    composite_embedding: EmbeddingResult = None

    # Strategy results
    embedding_strategy: str = None
    strategy_results: Dict[str, Any] = None
    quality_metrics: EmbeddingQualityMetrics = None


@dataclass
class AdvancedProcessingResult:
    """Advanced processing result with comprehensive embedding analysis."""

    documents: List[AdvancedProcessedDocument]
    success: bool
    error: str | None = None

    # Enhanced metadata
    embedding_strategies_used: List[str] = None
    total_chunks: int = 0
    processing_stats: dict = None
    quality_summary: dict = None
    optimization_suggestions: List[str] = None


class AdvancedDocumentProcessor:
    """Advanced document processor with enhanced embedding strategies."""

    def __init__(
        self,
        multi_embedding_service: MultiEmbeddingService | None = None,
        default_strategy: EmbeddingStrategy = EmbeddingStrategy.ADAPTIVE,
        enable_quality_analysis: bool = True,
    ):
        """
        Initialize advanced document processor.

        Args:
            multi_embedding_service: Multi-embedding service instance
            default_strategy: Default embedding strategy to use
            enable_quality_analysis: Whether to perform quality analysis
        """
        self.multi_embedding_service = (
            multi_embedding_service or MultiEmbeddingService()
        )
        self.default_strategy = default_strategy
        self.enable_quality_analysis = enable_quality_analysis

        self.chunker = TextChunker()
        self.extractor = Crawl4aiExtractor()
        self.content_analyzer = ContentAnalyzer()
        self.embedding_strategy_engine = EnhancedEmbeddingStrategy(
            self.multi_embedding_service
        )

        logger.info(
            f"Advanced processor initialized (strategy: {default_strategy.value}, quality_analysis: {enable_quality_analysis})"
        )

    async def process_url(
        self, url: str, options: dict | None = None
    ) -> AdvancedProcessingResult:
        """Process a URL with advanced embedding strategies."""
        try:
            logger.info(f"Advanced processing URL: {url}")

            # Extract processing options
            strategy = (
                EmbeddingStrategy(
                    options.get("embedding_strategy", self.default_strategy.value)
                )
                if options
                else self.default_strategy
            )
            max_pages = options.get("max_pages", 50) if options else 50
            strategy_config = options.get("strategy_config", {}) if options else {}

            # Use existing extraction pipeline
            result = await self.extractor.extract_from_url(url, max_pages=max_pages)

            if not result.success:
                return AdvancedProcessingResult(
                    documents=[], success=False, error=result.error
                )

            # Track processing stats
            processing_stats = {
                "total_pages": 0,
                "embedding_strategies_used": set(),
                "content_types_detected": set(),
                "high_code_content_pages": 0,
                "api_reference_pages": 0,
                "average_quality_score": 0.0,
                "optimization_applied": [],
            }

            # Process documents
            documents = []
            extracted_pages = result.metadata.get("extracted_pages", [])

            if extracted_pages:
                processing_stats["total_pages"] = len(extracted_pages)

                # Process each page as a separate document
                for page_info in extracted_pages:
                    page_url = page_info["url"]
                    page_content = page_info["content"]

                    # Perform content analysis
                    try:
                        content_analysis = self.content_analyzer.analyze_content(
                            page_content
                        )
                        processing_stats["content_types_detected"].add(
                            content_analysis.content_type
                        )

                        if content_analysis.code_percentage > 30:
                            processing_stats["high_code_content_pages"] += 1
                        if content_analysis.content_type == "api_reference":
                            processing_stats["api_reference_pages"] += 1

                    except Exception as e:
                        logger.warning(f"Content analysis failed for {page_url}: {e}")
                        content_analysis = None

                    # Create page-specific metadata
                    page_metadata = self._create_advanced_page_metadata(
                        result.metadata, page_info, content_analysis
                    )
                    page_metadata["page_url"] = page_url
                    page_metadata["is_individual_page"] = True
                    page_metadata["extraction_success"] = True

                    # Create document title
                    page_title = self._create_title_from_url(page_url, url)

                    document = await self._process_content_advanced(
                        content=page_content,
                        url=page_url,
                        title=page_title,
                        metadata=page_metadata,
                        content_analysis=content_analysis,
                        strategy=strategy,
                        strategy_config=strategy_config,
                        processing_stats=processing_stats,
                    )
                    documents.append(document)
            else:
                # Fallback to single document
                processing_stats["total_pages"] = 1

                try:
                    content_analysis = self.content_analyzer.analyze_content(
                        result.content
                    )
                    processing_stats["content_types_detected"].add(
                        content_analysis.content_type
                    )
                except Exception as e:
                    logger.warning(f"Content analysis failed for combined content: {e}")
                    content_analysis = None

                document = await self._process_content_advanced(
                    content=result.content,
                    url=url,
                    title=f"Documentation from {url}",
                    metadata=result.metadata,
                    content_analysis=content_analysis,
                    strategy=strategy,
                    strategy_config=strategy_config,
                    processing_stats=processing_stats,
                )
                documents = [document]

            # Calculate quality summary and optimization suggestions
            (
                quality_summary,
                optimization_suggestions,
            ) = self._analyze_processing_quality(documents, processing_stats)

            # Finalize stats
            processing_stats["content_types_detected"] = list(
                processing_stats["content_types_detected"]
            )
            processing_stats["embedding_strategies_used"] = list(
                processing_stats["embedding_strategies_used"]
            )
            total_chunks = sum(len(doc.chunks) for doc in documents)

            return AdvancedProcessingResult(
                documents=documents,
                success=True,
                embedding_strategies_used=processing_stats["embedding_strategies_used"],
                total_chunks=total_chunks,
                processing_stats=processing_stats,
                quality_summary=quality_summary,
                optimization_suggestions=optimization_suggestions,
            )

        except Exception as e:
            logger.error(f"Advanced processing failed for URL {url}: {e}")
            return AdvancedProcessingResult(documents=[], success=False, error=str(e))

    async def _process_content_advanced(
        self,
        content: str,
        url: str,
        title: str,
        metadata: dict,
        content_analysis: ContentAnalysis | None,
        strategy: EmbeddingStrategy,
        strategy_config: dict,
        processing_stats: dict,
    ) -> AdvancedProcessedDocument:
        """Process content with advanced embedding strategies."""
        try:
            # Split content into chunks
            chunks = self.chunker.chunk_text(content)
            chunk_contents = [chunk.content for chunk in chunks]

            # Generate embeddings using enhanced strategies
            strategy_results = (
                await self.embedding_strategy_engine.generate_enhanced_embeddings(
                    content=content,
                    title=title,
                    chunks=chunk_contents,
                    content_analysis=content_analysis,
                    strategy=strategy,
                    config=strategy_config,
                )
            )

            processing_stats["embedding_strategies_used"].add(
                strategy_results["strategy"]
            )

            # Extract embeddings based on strategy
            document_embedding = strategy_results.get("primary_embedding")
            summary_embedding = strategy_results.get("summary_embedding")
            composite_embedding = strategy_results.get("composite_embedding")

            # Process chunks with advanced embeddings
            processed_chunks = []
            chunk_embeddings = strategy_results.get("chunk_embeddings", [])

            for i, chunk in enumerate(chunks):
                # Get chunk embedding
                chunk_embedding = None
                if i < len(chunk_embeddings):
                    chunk_emb_data = chunk_embeddings[i]
                    if isinstance(chunk_emb_data, dict):
                        # Convert dict to EmbeddingResult
                        chunk_embedding = EmbeddingResult(
                            embedding=chunk_emb_data["embedding"],
                            model=chunk_emb_data["model"],
                            dimension=chunk_emb_data["dimension"],
                            strategy=strategy_results["strategy"],
                            metadata=chunk_emb_data.get("metadata", {}),
                            fallback_used=chunk_emb_data.get("used_fallback", False),
                        )
                    else:
                        chunk_embedding = chunk_emb_data

                # Generate quality metrics if enabled
                quality_metrics = None
                if self.enable_quality_analysis and chunk_embedding:
                    quality_metrics = await self._analyze_chunk_quality(
                        chunk, chunk_embedding, content_analysis
                    )

                processed_chunk = AdvancedProcessedChunk(
                    content=chunk.content,
                    primary_embedding=chunk_embedding,
                    strategy_embeddings=strategy_results,
                    quality_metrics=quality_metrics,
                    metadata={
                        **chunk.metadata,
                        "source_url": url,
                        "source_title": title,
                        "embedding_strategy": strategy_results["strategy"],
                    },
                    tokens=chunk.tokens,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    content_analysis=content_analysis,
                )
                processed_chunks.append(processed_chunk)

            # Generate document-level quality metrics
            document_quality = None
            if (
                self.enable_quality_analysis
                and "hierarchical_embedding" in strategy_results
            ):
                document_quality = strategy_results.get("quality_metrics")

            logger.info(
                f"Advanced processing completed: {len(processed_chunks)} chunks for {title} using {strategy_results['strategy']} strategy"
            )

            return AdvancedProcessedDocument(
                url=url,
                title=title,
                content=content,
                chunks=processed_chunks,
                metadata=metadata,
                content_analysis=content_analysis,
                document_embedding=document_embedding,
                summary_embedding=summary_embedding,
                composite_embedding=composite_embedding,
                embedding_strategy=strategy_results["strategy"],
                strategy_results=strategy_results,
                quality_metrics=document_quality,
            )

        except Exception as e:
            logger.error(f"Advanced content processing failed for {title}: {e}")
            raise

    async def _analyze_chunk_quality(
        self, chunk, embedding: EmbeddingResult, content_analysis: ContentAnalysis
    ) -> EmbeddingQualityMetrics:
        """Analyze quality of individual chunk embedding."""

        # Simple quality metrics for individual chunks
        confidence_score = 1.0 if embedding and embedding.embedding else 0.0

        # Adjust confidence based on content analysis alignment
        if content_analysis and embedding:
            if content_analysis.code_percentage > 30 and "code" in embedding.model:
                confidence_score = min(
                    1.0, confidence_score * 1.1
                )  # Boost for appropriate model selection
            elif (
                content_analysis.code_percentage < 10
                and "text-embedding" in embedding.model
            ):
                confidence_score = min(
                    1.0, confidence_score * 1.1
                )  # Boost for appropriate model selection

        return EmbeddingQualityMetrics(
            coherence_score=confidence_score,
            diversity_score=0.8,  # Default assumption
            coverage_score=confidence_score,
            consistency_score=confidence_score,
            confidence_score=confidence_score,
            model_agreement={},
        )

    def _analyze_processing_quality(
        self, documents: List[AdvancedProcessedDocument], processing_stats: dict
    ) -> tuple[dict, List[str]]:
        """Analyze overall processing quality and suggest optimizations."""

        # Calculate quality metrics
        total_quality_scores = []
        strategy_distribution = {}
        model_usage = {}

        for doc in documents:
            if doc.quality_metrics:
                total_quality_scores.append(doc.quality_metrics.confidence_score)

            strategy = doc.embedding_strategy
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

            for chunk in doc.chunks:
                if chunk.primary_embedding:
                    model = chunk.primary_embedding.model
                    model_usage[model] = model_usage.get(model, 0) + 1

        avg_quality = (
            sum(total_quality_scores) / len(total_quality_scores)
            if total_quality_scores
            else 0.0
        )
        processing_stats["average_quality_score"] = avg_quality

        quality_summary = {
            "average_quality_score": avg_quality,
            "strategy_distribution": strategy_distribution,
            "model_usage": model_usage,
            "total_documents": len(documents),
        }

        # Generate optimization suggestions
        suggestions = []

        if avg_quality < 0.7:
            suggestions.append(
                "Consider using hierarchical or composite embedding strategies for better quality"
            )

        if processing_stats.get("high_code_content_pages", 0) > 0:
            if "embed-code" not in str(model_usage):
                suggestions.append(
                    "Code-specific embedding models recommended for code-heavy content"
                )

        if len(documents) > 10:
            suggestions.append(
                "Consider using summary-enhanced strategy for large document collections"
            )

        if "adaptive" not in strategy_distribution:
            suggestions.append(
                "Adaptive strategy can automatically optimize embedding approach"
            )

        return quality_summary, suggestions

    def _create_advanced_page_metadata(
        self,
        batch_metadata: dict,
        page_info: dict,
        content_analysis: ContentAnalysis | None,
    ) -> dict:
        """Create advanced page metadata with embedding strategy information."""

        # Start with basic metadata (excluding batch statistics)
        batch_only_fields = {
            "total_links_found",
            "filtered_links",
            "successful_extractions",
            "extracted_pages",
        }

        page_metadata = {
            key: value
            for key, value in batch_metadata.items()
            if key not in batch_only_fields
        }

        # Add page-specific information
        page_metadata.update(
            {
                "content_length": len(page_info.get("content", "")),
                "filename": page_info.get("filename"),
                "processing_time": batch_metadata.get("extraction_time"),
            }
        )

        # Integrate content analysis results
        if content_analysis:
            page_metadata.update(
                {
                    "content_type": content_analysis.content_type,
                    "primary_language": content_analysis.primary_language,
                    "summary": content_analysis.summary,
                    "code_percentage": content_analysis.code_percentage,
                    "detected_patterns": content_analysis.detected_patterns,
                    "key_concepts": content_analysis.key_concepts[:5],
                    "api_references": content_analysis.api_references[:10],
                    "code_blocks_count": len(content_analysis.code_blocks),
                    "content_analysis_available": True,
                    "recommended_embedding_strategy": self._recommend_strategy(
                        content_analysis
                    ),
                }
            )

            # Add detailed code analysis for high-code content
            if content_analysis.code_percentage > 10:
                page_metadata["code_analysis"] = {
                    "functions": [
                        func
                        for block in content_analysis.code_blocks
                        for func in block.functions
                    ],
                    "classes": [
                        cls
                        for block in content_analysis.code_blocks
                        for cls in block.classes
                    ],
                    "imports": [
                        imp
                        for block in content_analysis.code_blocks
                        for imp in block.imports
                    ],
                    "languages": list(
                        set(block.language for block in content_analysis.code_blocks)
                    ),
                }
        else:
            # Set defaults if analysis failed
            page_metadata.update(
                {
                    "content_type": "general",
                    "primary_language": None,
                    "summary": "Content analysis unavailable",
                    "code_percentage": 0.0,
                    "detected_patterns": {},
                    "key_concepts": [],
                    "api_references": [],
                    "code_blocks_count": 0,
                    "content_analysis_available": False,
                    "recommended_embedding_strategy": "single",
                }
            )

        return page_metadata

    def _recommend_strategy(self, content_analysis: ContentAnalysis) -> str:
        """Recommend optimal embedding strategy based on content analysis."""

        if content_analysis.code_percentage > 40:
            return "hierarchical"
        elif content_analysis.content_type == "api_reference":
            return "composite"
        elif (
            content_analysis.content_type == "tutorial"
            and len(content_analysis.key_concepts) > 8
        ):
            return "summary_enhanced"
        elif content_analysis.content_type == "tutorial":
            return "composite"
        elif len(content_analysis.key_concepts) > 8:
            return "summary_enhanced"
        else:
            return "adaptive"

    def _create_title_from_url(self, page_url: str, base_url: str) -> str:
        """Create a meaningful title from the page URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(page_url)

            # Extract path parts and create a title
            path_parts = [part for part in parsed.path.strip("/").split("/") if part]

            if path_parts:
                # Use the last meaningful part of the path
                title_part = path_parts[-1].replace("-", " ").replace("_", " ").title()
                if title_part and title_part.lower() not in ["index", "home", "main"]:
                    return f"{title_part} - {parsed.netloc}"

            # Fallback to just the domain and path
            if parsed.path and parsed.path != "/":
                return f"{parsed.netloc}{parsed.path}"

            return f"Documentation from {page_url}"

        except Exception:
            return f"Documentation from {page_url}"

    async def get_strategy_recommendations(
        self, content: str, content_analysis: ContentAnalysis = None
    ) -> Dict[str, Any]:
        """Get strategy recommendations for given content."""

        if not content_analysis:
            content_analysis = self.content_analyzer.analyze_content(content)

        recommendations = {
            "primary_strategy": self._recommend_strategy(content_analysis),
            "alternative_strategies": [],
            "reasoning": {},
            "expected_performance": {},
        }

        # Add reasoning
        if content_analysis.code_percentage > 40:
            recommendations["reasoning"][
                "hierarchical"
            ] = "High code content benefits from multi-level embeddings"
            recommendations["alternative_strategies"].append("composite")

        if content_analysis.content_type == "api_reference":
            recommendations["reasoning"][
                "composite"
            ] = "API documentation benefits from combined representations"
            recommendations["alternative_strategies"].append("hierarchical")

        if len(content.split()) > 2000:
            recommendations["reasoning"][
                "summary_enhanced"
            ] = "Long content benefits from summary distillation"
            recommendations["alternative_strategies"].append("hierarchical")

        # Add performance expectations
        recommendations["expected_performance"] = {
            "quality_score": 0.8 if content_analysis.content_type != "general" else 0.7,
            "processing_time": "medium" if len(content.split()) > 1000 else "fast",
            "storage_efficiency": "high"
            if recommendations["primary_strategy"] in ["single", "summary_enhanced"]
            else "medium",
        }

        return recommendations
