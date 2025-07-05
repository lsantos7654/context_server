"""Enhanced document processing with multi-embedding support and content-aware routing."""

import asyncio
import logging

# Import existing extraction functionality
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append("/app/src")
from src.core.crawl4ai_extraction import Crawl4aiExtractor

from .chunking import TextChunker
from .content_analysis import ContentAnalysis, ContentAnalyzer
from .hierarchical_graph_builder import HierarchicalGraphBuilder
from .multi_embedding_service import EmbeddingModel, MultiEmbeddingService
from .processing import ProcessedChunk, ProcessedDocument, ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedProcessedChunk:
    """Enhanced processed chunk with multiple embeddings support."""

    content: str
    embeddings: Dict[str, List[float]]  # model_name -> embedding
    embedding_metadata: Dict[str, Dict]  # model_name -> metadata
    metadata: dict
    tokens: int
    start_line: int = None
    end_line: int = None
    char_start: int = None
    char_end: int = None
    content_analysis: ContentAnalysis = None


@dataclass
class EnhancedProcessedDocument:
    """Enhanced processed document with content analysis and multi-embedding support."""

    url: str
    title: str
    content: str
    chunks: List[EnhancedProcessedChunk]
    metadata: dict
    content_analysis: ContentAnalysis = None
    primary_embedding_model: str = None


@dataclass
class EnhancedProcessingResult:
    """Enhanced result with additional analysis metadata."""

    documents: List[EnhancedProcessedDocument]
    success: bool
    error: str | None = None
    embedding_models_used: List[str] = None
    total_chunks: int = 0
    processing_stats: dict = None


class EnhancedDocumentProcessor:
    """Enhanced document processor with multi-embedding and hierarchical graph building."""

    def __init__(
        self,
        multi_embedding_service: MultiEmbeddingService | None = None,
        database_manager=None,
        enable_multi_embedding: bool = False,
        enable_graph_building: bool = True,
    ):
        """
        Initialize enhanced document processor.

        Args:
            multi_embedding_service: Multi-embedding service instance
            database_manager: Database manager for graph building
            enable_multi_embedding: Whether to generate multiple embeddings per chunk
            enable_graph_building: Whether to build hierarchical knowledge graphs
        """
        self.multi_embedding_service = (
            multi_embedding_service or MultiEmbeddingService()
        )
        self.enable_multi_embedding = enable_multi_embedding
        self.enable_graph_building = enable_graph_building

        self.chunker = TextChunker()
        self.extractor = Crawl4aiExtractor()

        # Initialize content analyzer with LLM support
        try:
            from .simple_llm_service import SimpleLLMService

            llm_service = SimpleLLMService()
            self.content_analyzer = ContentAnalyzer(
                llm_service=llm_service, use_llm_analysis=True
            )
            logger.info("ContentAnalyzer initialized with LLM support")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM service: {e}")
            self.content_analyzer = ContentAnalyzer(use_llm_analysis=False)
            logger.info("ContentAnalyzer initialized with fallback analysis")

        # Initialize graph builder if database manager is provided
        self.graph_builder = None
        if database_manager and enable_graph_building:
            self.graph_builder = HierarchicalGraphBuilder(database_manager)

        logger.info(
            f"Enhanced processor initialized (multi-embedding: {enable_multi_embedding}, "
            f"graph-building: {enable_graph_building})"
        )

    async def initialize_graph_schema(self):
        """Initialize the hierarchical graph schema."""
        if self.graph_builder:
            await self.graph_builder.initialize_schema()
            logger.info("Hierarchical graph schema initialized")
        else:
            logger.warning("Graph builder not available - schema not initialized")

    async def process_url(
        self, url: str, options: dict | None = None
    ) -> EnhancedProcessingResult:
        """Process a URL with enhanced content analysis and multi-embedding."""
        try:
            logger.info(f"Enhanced processing URL: {url}")

            # Use existing extraction pipeline
            max_pages = options.get("max_pages", 50) if options else 50
            result = await self.extractor.extract_from_url(url, max_pages=max_pages)

            if not result.success:
                return EnhancedProcessingResult(
                    documents=[], success=False, error=result.error
                )

            # Track processing stats
            processing_stats = {
                "total_pages": 0,
                "embedding_models_used": set(),
                "content_types_detected": set(),
                "high_code_content_pages": 0,
                "api_reference_pages": 0,
            }

            # Create separate documents for each extracted page if available
            documents = []
            extracted_pages = result.metadata.get("extracted_pages", [])

            if extracted_pages:
                processing_stats["total_pages"] = len(extracted_pages)

                # Process each page as a separate document
                for page_info in extracted_pages:
                    page_url = page_info["url"]
                    page_content = page_info["content"]

                    # Perform content analysis first
                    try:
                        content_analysis = await self.content_analyzer.analyze_content(
                            page_content, page_url
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
                    page_metadata = self._create_enhanced_page_metadata(
                        result.metadata, page_info, content_analysis
                    )
                    page_metadata["page_url"] = page_url
                    page_metadata["is_individual_page"] = True
                    page_metadata["extraction_success"] = True

                    # Create document title from page URL
                    page_title = self._create_title_from_url(page_url, url)

                    document = await self._process_content_enhanced(
                        content=page_content,
                        url=page_url,
                        title=page_title,
                        metadata=page_metadata,
                        content_analysis=content_analysis,
                        processing_stats=processing_stats,
                    )
                    documents.append(document)
            else:
                # Fallback to single document if no individual pages
                processing_stats["total_pages"] = 1

                # Analyze combined content
                try:
                    content_analysis = await self.content_analyzer.analyze_content(
                        result.content, url
                    )
                    processing_stats["content_types_detected"].add(
                        content_analysis.content_type
                    )
                except Exception as e:
                    logger.warning(f"Content analysis failed for combined content: {e}")
                    content_analysis = None

                document = await self._process_content_enhanced(
                    content=result.content,
                    url=url,
                    title=f"Documentation from {url}",
                    metadata=result.metadata,
                    content_analysis=content_analysis,
                    processing_stats=processing_stats,
                )
                documents = [document]

            # Finalize processing stats
            processing_stats["content_types_detected"] = list(
                processing_stats["content_types_detected"]
            )
            processing_stats["embedding_models_used"] = list(
                processing_stats["embedding_models_used"]
            )
            total_chunks = sum(len(doc.chunks) for doc in documents)

            return EnhancedProcessingResult(
                documents=documents,
                success=True,
                embedding_models_used=processing_stats["embedding_models_used"],
                total_chunks=total_chunks,
                processing_stats=processing_stats,
            )

        except Exception as e:
            logger.error(f"Enhanced processing failed for URL {url}: {e}")
            return EnhancedProcessingResult(documents=[], success=False, error=str(e))

    async def _process_content_enhanced(
        self,
        content: str,
        url: str,
        title: str,
        metadata: dict,
        content_analysis: ContentAnalysis | None = None,
        processing_stats: dict | None = None,
    ) -> EnhancedProcessedDocument:
        """Process content with enhanced embedding and analysis."""
        try:
            # Split content into chunks
            chunks = self.chunker.chunk_text(content)

            # Determine primary embedding model based on content analysis
            primary_model = self.multi_embedding_service.route_content(content_analysis)
            if processing_stats:
                processing_stats["embedding_models_used"].add(primary_model.value)

            # Process chunks with enhanced embeddings
            processed_chunks = []

            # Process chunks in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                # Generate embeddings for the batch
                if self.enable_multi_embedding:
                    # Generate multiple embeddings per chunk
                    chunk_embeddings = await self._generate_multi_embeddings_batch(
                        batch, content_analysis, processing_stats
                    )
                else:
                    # Generate single optimal embedding per chunk
                    chunk_embeddings = await self._generate_optimal_embeddings_batch(
                        batch, content_analysis, processing_stats
                    )

                # Create enhanced processed chunks
                for chunk, embeddings_data in zip(batch, chunk_embeddings):
                    processed_chunk = EnhancedProcessedChunk(
                        content=chunk.content,
                        embeddings=embeddings_data["embeddings"],
                        embedding_metadata=embeddings_data["metadata"],
                        metadata={
                            **chunk.metadata,
                            "source_url": url,
                            "source_title": title,
                        },
                        tokens=chunk.tokens,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        content_analysis=content_analysis,
                    )
                    processed_chunks.append(processed_chunk)

                # Small delay between batches
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)

            logger.info(
                f"Enhanced processing completed: {len(processed_chunks)} chunks for {title}"
            )

            document = EnhancedProcessedDocument(
                url=url,
                title=title,
                content=content,
                chunks=processed_chunks,
                metadata=metadata,
                content_analysis=content_analysis,
                primary_embedding_model=primary_model.value,
            )

            # Build hierarchical knowledge graph if enabled
            if self.graph_builder and self.enable_graph_building:
                try:
                    # Prepare chunk data for graph builder
                    chunk_data_for_graph = []
                    for chunk in processed_chunks:
                        chunk_data_for_graph.append(
                            {
                                "id": str(uuid.uuid4()),  # Generate chunk ID for graph
                                "document_id": metadata.get("document_id", url),
                                "content": chunk.content,
                                "chunk_index": len(chunk_data_for_graph),
                                "metadata": chunk.metadata,
                            }
                        )

                    # Build document hierarchy in graph
                    graph_result = await self.graph_builder.build_document_hierarchy(
                        document_id=metadata.get("document_id", url),
                        title=title,
                        content=content,
                        chunks=chunk_data_for_graph,
                        content_analysis=content_analysis,
                    )

                    # Add graph statistics to metadata
                    metadata["graph_nodes_created"] = graph_result["total_nodes"]
                    metadata["graph_relationships_created"] = graph_result[
                        "total_relationships"
                    ]
                    metadata["hierarchical_graph_built"] = True

                    logger.info(
                        f"Built knowledge graph for {title}: "
                        f"{graph_result['total_nodes']} nodes, "
                        f"{graph_result['total_relationships']} relationships"
                    )

                except Exception as e:
                    logger.error(f"Failed to build knowledge graph for {title}: {e}")
                    metadata["hierarchical_graph_built"] = False
                    metadata["graph_error"] = str(e)

            return document

        except Exception as e:
            logger.error(f"Enhanced content processing failed for {title}: {e}")
            raise

    async def _generate_multi_embeddings_batch(
        self,
        chunks: List,
        content_analysis: ContentAnalysis | None,
        processing_stats: dict | None,
    ) -> List[Dict]:
        """Generate multiple embeddings per chunk for comparison/experimentation."""

        # Models to generate embeddings with
        models_to_use = [EmbeddingModel.OPENAI_SMALL, EmbeddingModel.VOYAGE_CODE]

        results = []
        chunk_texts = [chunk.content for chunk in chunks]

        # Generate embeddings with each model
        embeddings_by_model = {}
        for model in models_to_use:
            try:
                if processing_stats:
                    processing_stats["embedding_models_used"].add(model.value)

                model_results = await self.multi_embedding_service.embed_batch(
                    chunk_texts, force_model=model
                )
                embeddings_by_model[model.value] = model_results
            except Exception as e:
                logger.warning(f"Failed to generate embeddings with {model.value}: {e}")
                # Create dummy embeddings
                provider = self.multi_embedding_service.providers[model]
                dummy_results = [
                    {
                        "embedding": [0.0] * provider.get_dimension(),
                        "model": model.value,
                        "dimension": provider.get_dimension(),
                        "success": False,
                        "error": str(e),
                    }
                    for _ in chunk_texts
                ]
                embeddings_by_model[model.value] = dummy_results

        # Combine results per chunk
        for i in range(len(chunks)):
            chunk_data = {"embeddings": {}, "metadata": {}}

            for model_name, model_results in embeddings_by_model.items():
                if i < len(model_results):
                    result = model_results[i]
                    chunk_data["embeddings"][model_name] = result["embedding"]
                    chunk_data["metadata"][model_name] = {
                        "dimension": result["dimension"],
                        "success": result["success"],
                        "model": result["model"],
                    }
                    if "error" in result:
                        chunk_data["metadata"][model_name]["error"] = result["error"]

            results.append(chunk_data)

        return results

    async def _generate_optimal_embeddings_batch(
        self,
        chunks: List,
        content_analysis: ContentAnalysis | None,
        processing_stats: dict | None,
    ) -> List[Dict]:
        """Generate single optimal embedding per chunk using content-aware routing."""

        chunk_texts = [chunk.content for chunk in chunks]

        try:
            # Generate embeddings using intelligent routing
            embedding_results = await self.multi_embedding_service.embed_batch(
                chunk_texts, content_analyses=[content_analysis] * len(chunk_texts)
            )

            # Track which models were used
            if processing_stats:
                for result in embedding_results:
                    processing_stats["embedding_models_used"].add(result["model"])

            # Format results
            results = []
            for result in embedding_results:
                chunk_data = {
                    "embeddings": {result["model"]: result["embedding"]},
                    "metadata": {
                        result["model"]: {
                            "dimension": result["dimension"],
                            "success": result["success"],
                            "model": result["model"],
                        }
                    },
                }

                if "error" in result:
                    chunk_data["metadata"][result["model"]]["error"] = result["error"]
                if "used_fallback" in result:
                    chunk_data["metadata"][result["model"]]["used_fallback"] = result[
                        "used_fallback"
                    ]

                results.append(chunk_data)

            return results

        except Exception as e:
            logger.error(f"Failed to generate optimal embeddings: {e}")
            # Create dummy embeddings with fallback model
            fallback_provider = self.multi_embedding_service.providers[
                self.multi_embedding_service.fallback_model
            ]
            results = []
            for _ in chunk_texts:
                chunk_data = {
                    "embeddings": {
                        self.multi_embedding_service.fallback_model.value: [0.0]
                        * fallback_provider.get_dimension()
                    },
                    "metadata": {
                        self.multi_embedding_service.fallback_model.value: {
                            "dimension": fallback_provider.get_dimension(),
                            "success": False,
                            "model": self.multi_embedding_service.fallback_model.value,
                            "error": str(e),
                        }
                    },
                }
                results.append(chunk_data)
            return results

    def _create_enhanced_page_metadata(
        self,
        batch_metadata: dict,
        page_info: dict,
        content_analysis: ContentAnalysis | None,
    ) -> dict:
        """Create enhanced page metadata with content analysis integration."""

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
                }
            )

        return page_metadata

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

    async def get_embedding_stats(self) -> dict:
        """Get statistics about embedding service health and usage."""
        health_status = await self.multi_embedding_service.health_check()
        available_models = self.multi_embedding_service.get_available_models()

        return {
            "available_models": available_models,
            "health_status": health_status,
            "multi_embedding_enabled": self.enable_multi_embedding,
            "routing_rules": {
                content_type.value: model.value
                for content_type, model in self.multi_embedding_service.routing_rules.items()
            },
        }
