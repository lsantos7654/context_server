"""Enhanced PostgreSQL storage with multi-embedding support."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
from asyncpg import Pool

from .enhanced_processing import EnhancedProcessedChunk, EnhancedProcessedDocument

logger = logging.getLogger(__name__)


class EnhancedDatabaseManager:
    """Enhanced database manager with multi-embedding support."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/context_server"
        )
        self.pool: Pool | None = None

    async def initialize(self):
        """Initialize database with enhanced schema for multi-embeddings."""
        max_retries = 30
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url, min_size=2, max_size=10, command_timeout=30
                )

                # Create enhanced schema
                await self._create_enhanced_schema()
                logger.info("Enhanced database initialized successfully")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to initialize enhanced database after {max_retries} attempts: {e}"
                    )
                    raise

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
            logger.info("Enhanced database connection closed")

    async def _create_enhanced_schema(self):
        """Create enhanced database schema with multi-embedding support."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create or update contexts table with enhanced metadata
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
                    embedding_strategy VARCHAR(50) DEFAULT 'single',  -- 'single', 'multi', 'adaptive'
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    processing_stats JSONB DEFAULT '{}'
                )
                """
            )

            # Add new columns to existing contexts table
            await conn.execute(
                """
                DO $$ BEGIN
                    ALTER TABLE contexts ADD COLUMN IF NOT EXISTS embedding_strategy VARCHAR(50) DEFAULT 'single';
                    ALTER TABLE contexts ADD COLUMN IF NOT EXISTS processing_stats JSONB DEFAULT '{}';
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END $$;
                """
            )

            # Create enhanced documents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    content_analysis JSONB DEFAULT '{}',  -- Store content analysis results
                    primary_embedding_model VARCHAR(100),
                    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    chunk_count INTEGER DEFAULT 0,
                    source_type VARCHAR(20) NOT NULL,
                    UNIQUE(context_id, url)
                )
                """
            )

            # Add new columns to existing documents table
            await conn.execute(
                """
                DO $$ BEGIN
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_analysis JSONB DEFAULT '{}';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS primary_embedding_model VARCHAR(100);
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END $$;
                """
            )

            # Create enhanced chunks table with multiple embedding support
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector(1536),  -- Primary embedding (backward compatibility)
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    tokens INTEGER,
                    start_line INTEGER,
                    end_line INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                )
                """
            )

            # Create new table for multiple embeddings per chunk
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_name VARCHAR(100) NOT NULL,
                    embedding vector(4096),  -- Support larger dimensions (Cohere: 4096)
                    dimension INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(chunk_id, model_name)
                )
                """
            )

            # Create content analysis table for structured storage
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content_analyses (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content_type VARCHAR(50),
                    primary_language VARCHAR(50),
                    summary TEXT,
                    code_percentage FLOAT DEFAULT 0.0,
                    detected_patterns JSONB DEFAULT '{}',
                    key_concepts JSONB DEFAULT '[]',
                    api_references JSONB DEFAULT '[]',
                    code_blocks_count INTEGER DEFAULT 0,
                    code_analysis JSONB DEFAULT '{}',
                    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id)
                )
                """
            )

            # Create knowledge graph tables
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content_relationships (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_url TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    relationship_type VARCHAR(50) NOT NULL,
                    strength FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(source_url, target_url, relationship_type)
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_clusters (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    cluster_id VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    content_urls JSONB DEFAULT '[]',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_statistics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    total_nodes INTEGER NOT NULL,
                    total_edges INTEGER NOT NULL,
                    cluster_count INTEGER NOT NULL,
                    average_node_degree FLOAT NOT NULL,
                    graph_density FLOAT NOT NULL,
                    modularity_score FLOAT NOT NULL,
                    coverage_ratio FLOAT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                """
            )

            # Create indexes for performance

            # Original indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_context_id ON chunks(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_context_id ON documents(context_id)"
            )

            # New indexes for multi-embedding support
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model_name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk_id ON chunk_embeddings(chunk_id)"
            )

            # Flexible embedding index - will need to create specific indexes per model/dimension
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_vector_1536 ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WHERE dimension = 1536"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_vector_4096 ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WHERE dimension = 4096"
            )

            # Content analysis indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_analyses_content_type ON content_analyses(content_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_analyses_language ON content_analyses(primary_language)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_analyses_code_percentage ON content_analyses(code_percentage)"
            )

            # Full-text search indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks USING gin(to_tsvector('english', content))"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', content))"
            )

            # Knowledge graph indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_relationships_source ON content_relationships(source_url)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_relationships_target ON content_relationships(target_url)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_relationships_type ON content_relationships(relationship_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic_clusters_cluster_id ON topic_clusters(cluster_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_statistics_created_at ON graph_statistics(created_at)"
            )

            logger.info("Enhanced database schema created successfully")

    async def create_enhanced_document(
        self, context_id: str, document: EnhancedProcessedDocument
    ) -> str:
        """Create document with enhanced processing results."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert main document
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documents (
                        context_id, url, title, content, metadata,
                        content_analysis, primary_embedding_model, source_type
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (context_id, url) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        content_analysis = EXCLUDED.content_analysis,
                        primary_embedding_model = EXCLUDED.primary_embedding_model,
                        indexed_at = NOW()
                    RETURNING id
                    """,
                    uuid.UUID(context_id),
                    document.url,
                    document.title,
                    document.content,
                    json.dumps(document.metadata),
                    json.dumps(
                        document.content_analysis.__dict__
                        if document.content_analysis
                        else {}
                    ),
                    document.primary_embedding_model,
                    document.metadata.get("source_type", "unknown"),
                )

                # Store content analysis separately for structured queries
                if document.content_analysis:
                    await self._store_content_analysis(
                        conn, doc_id, document.content_analysis
                    )

                # Store chunks with multiple embeddings
                for chunk_idx, chunk in enumerate(document.chunks):
                    await self._store_enhanced_chunk(
                        conn, doc_id, context_id, chunk, chunk_idx
                    )

                # Update document chunk count
                await conn.execute(
                    "UPDATE documents SET chunk_count = $1 WHERE id = $2",
                    len(document.chunks),
                    doc_id,
                )

                # Update context statistics
                await self._update_context_stats(conn, context_id)

                return str(doc_id)

    async def _store_content_analysis(
        self, conn, document_id: uuid.UUID, analysis
    ) -> None:
        """Store content analysis in structured format."""
        await conn.execute(
            """
            INSERT INTO content_analyses (
                document_id, content_type, primary_language, summary,
                code_percentage, detected_patterns, key_concepts,
                api_references, code_blocks_count, code_analysis
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (document_id) DO UPDATE SET
                content_type = EXCLUDED.content_type,
                primary_language = EXCLUDED.primary_language,
                summary = EXCLUDED.summary,
                code_percentage = EXCLUDED.code_percentage,
                detected_patterns = EXCLUDED.detected_patterns,
                key_concepts = EXCLUDED.key_concepts,
                api_references = EXCLUDED.api_references,
                code_blocks_count = EXCLUDED.code_blocks_count,
                code_analysis = EXCLUDED.code_analysis,
                analyzed_at = NOW()
            """,
            document_id,
            analysis.content_type,
            analysis.primary_language,
            analysis.summary,
            analysis.code_percentage,
            json.dumps(analysis.detected_patterns),
            json.dumps(analysis.key_concepts),
            json.dumps(analysis.api_references),
            len(analysis.code_blocks),
            json.dumps(getattr(analysis, "code_analysis", {})),
        )

    async def _store_enhanced_chunk(
        self,
        conn,
        document_id: uuid.UUID,
        context_id: str,
        chunk: EnhancedProcessedChunk,
        chunk_index: int,
    ) -> None:
        """Store chunk with multiple embeddings."""

        # Get primary embedding for backward compatibility
        primary_embedding = None
        primary_model = None

        if chunk.embeddings:
            # Use the first available embedding as primary
            primary_model = list(chunk.embeddings.keys())[0]
            primary_embedding = chunk.embeddings[primary_model]

        if not primary_embedding:
            # Create dummy embedding if none available
            primary_embedding = [0.0] * 1536
            primary_model = "dummy"

        # Convert primary embedding to PostgreSQL vector format
        primary_embedding_str = "[" + ",".join(map(str, primary_embedding)) + "]"

        # Insert main chunk record
        chunk_id = await conn.fetchval(
            """
            INSERT INTO chunks (
                document_id, context_id, content, embedding, chunk_index,
                metadata, tokens, start_line, end_line, char_start, char_end
            )
            VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                tokens = EXCLUDED.tokens,
                start_line = EXCLUDED.start_line,
                end_line = EXCLUDED.end_line,
                char_start = EXCLUDED.char_start,
                char_end = EXCLUDED.char_end
            RETURNING id
            """,
            document_id,
            uuid.UUID(context_id),
            chunk.content,
            primary_embedding_str,
            chunk_index,
            json.dumps(chunk.metadata or {}),
            chunk.tokens,
            chunk.start_line,
            chunk.end_line,
            chunk.char_start,
            chunk.char_end,
        )

        # Store all embeddings in separate table
        for model_name, embedding in chunk.embeddings.items():
            embedding_metadata = chunk.embedding_metadata.get(model_name, {})
            dimension = embedding_metadata.get("dimension", len(embedding))

            # Pad or truncate embedding to fit vector(4096)
            if len(embedding) > 4096:
                embedding = embedding[:4096]
            elif len(embedding) < 4096:
                embedding = embedding + [0.0] * (4096 - len(embedding))

            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            await conn.execute(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id, model_name, embedding, dimension, metadata
                )
                VALUES ($1, $2, $3::vector, $4, $5)
                ON CONFLICT (chunk_id, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    dimension = EXCLUDED.dimension,
                    metadata = EXCLUDED.metadata
                """,
                chunk_id,
                model_name,
                embedding_str,
                dimension,
                json.dumps(embedding_metadata),
            )

    async def _update_context_stats(self, conn, context_id: str) -> None:
        """Update context statistics."""
        await conn.execute(
            """
            UPDATE contexts
            SET document_count = (SELECT COUNT(*) FROM documents WHERE context_id = $1),
                total_chunks = (SELECT COUNT(*) FROM chunks WHERE context_id = $1),
                updated_at = NOW()
            WHERE id = $1
            """,
            uuid.UUID(context_id),
        )

    async def multi_vector_search(
        self,
        context_id: str,
        query_embeddings: Dict[str, List[float]],
        limit: int = 10,
        min_similarity: float = 0.7,
        model_weights: Dict[str, float] = None,
    ) -> List[Dict]:
        """Perform search using multiple embedding models with weighted scoring."""

        if not query_embeddings:
            return []

        # Default equal weights if not specified
        if model_weights is None:
            model_weights = {model: 1.0 for model in query_embeddings.keys()}

        async with self.pool.acquire() as conn:
            results_by_model = {}

            # Search with each embedding model
            for model_name, query_embedding in query_embeddings.items():
                if model_name not in model_weights:
                    continue

                # Pad or truncate query embedding to fit vector(4096)
                if len(query_embedding) > 4096:
                    query_embedding = query_embedding[:4096]
                elif len(query_embedding) < 4096:
                    query_embedding = query_embedding + [0.0] * (
                        4096 - len(query_embedding)
                    )

                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                try:
                    rows = await conn.fetch(
                        """
                        SELECT
                            c.id, c.content, d.title, d.url, d.metadata as doc_metadata,
                            c.metadata as chunk_metadata, c.chunk_index, d.id as document_id,
                            c.start_line, c.end_line, c.char_start, c.char_end,
                            1 - (ce.embedding <=> $2::vector) as similarity,
                            ce.model_name, ce.dimension
                        FROM chunks c
                        JOIN chunk_embeddings ce ON c.id = ce.chunk_id
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.context_id = $1
                            AND ce.model_name = $3
                            AND 1 - (ce.embedding <=> $2::vector) > $4
                        ORDER BY ce.embedding <=> $2::vector
                        LIMIT $5
                        """,
                        uuid.UUID(context_id),
                        embedding_str,
                        model_name,
                        min_similarity,
                        limit,
                    )

                    results_by_model[model_name] = rows

                except Exception as e:
                    logger.warning(f"Search with {model_name} failed: {e}")
                    results_by_model[model_name] = []

            # Combine and weight results
            chunk_scores = {}
            for model_name, rows in results_by_model.items():
                weight = model_weights[model_name]

                for row in rows:
                    chunk_id = str(row["id"])
                    weighted_score = float(row["similarity"]) * weight

                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = {
                            "chunk_data": row,
                            "total_score": weighted_score,
                            "model_scores": {model_name: float(row["similarity"])},
                        }
                    else:
                        chunk_scores[chunk_id]["total_score"] += weighted_score
                        chunk_scores[chunk_id]["model_scores"][model_name] = float(
                            row["similarity"]
                        )

            # Sort by combined score and format results
            sorted_chunks = sorted(
                chunk_scores.items(), key=lambda x: x[1]["total_score"], reverse=True
            )[:limit]

            results = []
            for chunk_id, chunk_info in sorted_chunks:
                row = chunk_info["chunk_data"]

                result = {
                    "id": chunk_id,
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": chunk_info["total_score"],
                    "model_scores": chunk_info["model_scores"],
                    "metadata": {
                        **(
                            json.loads(row["doc_metadata"])
                            if row["doc_metadata"]
                            else {}
                        ),
                        **(
                            json.loads(row["chunk_metadata"])
                            if row["chunk_metadata"]
                            else {}
                        ),
                    },
                    "chunk_index": row["chunk_index"],
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
                }
                results.append(result)

            return results

    async def content_type_search(
        self, context_id: str, content_types: List[str], limit: int = 50
    ) -> List[Dict]:
        """Search documents by content type."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT
                    d.id, d.title, d.url, d.metadata, ca.content_type,
                    ca.primary_language, ca.summary, ca.code_percentage
                FROM documents d
                JOIN content_analyses ca ON d.id = ca.document_id
                WHERE d.context_id = $1 AND ca.content_type = ANY($2)
                ORDER BY d.indexed_at DESC
                LIMIT $3
                """,
                uuid.UUID(context_id),
                content_types,
                limit,
            )

            return [
                {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "url": row["url"],
                    "content_type": row["content_type"],
                    "primary_language": row["primary_language"],
                    "summary": row["summary"],
                    "code_percentage": float(row["code_percentage"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    async def get_context_analytics(self, context_id: str) -> Dict:
        """Get analytics for a context including content analysis breakdowns."""
        async with self.pool.acquire() as conn:
            # Basic context info
            context_info = await conn.fetchrow(
                "SELECT * FROM contexts WHERE id = $1", uuid.UUID(context_id)
            )

            if not context_info:
                return {}

            # Content type distribution
            content_types = await conn.fetch(
                """
                SELECT content_type, COUNT(*) as count, AVG(code_percentage) as avg_code_percentage
                FROM content_analyses ca
                JOIN documents d ON ca.document_id = d.id
                WHERE d.context_id = $1
                GROUP BY content_type
                ORDER BY count DESC
                """,
                uuid.UUID(context_id),
            )

            # Language distribution
            languages = await conn.fetch(
                """
                SELECT primary_language, COUNT(*) as count
                FROM content_analyses ca
                JOIN documents d ON ca.document_id = d.id
                WHERE d.context_id = $1 AND primary_language IS NOT NULL
                GROUP BY primary_language
                ORDER BY count DESC
                """,
                uuid.UUID(context_id),
            )

            # Embedding model usage
            embedding_models = await conn.fetch(
                """
                SELECT model_name, COUNT(DISTINCT chunk_id) as chunk_count
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                WHERE c.context_id = $1
                GROUP BY model_name
                ORDER BY chunk_count DESC
                """,
                uuid.UUID(context_id),
            )

            return {
                "context": {
                    "id": str(context_info["id"]),
                    "name": context_info["name"],
                    "description": context_info["description"],
                    "document_count": context_info["document_count"],
                    "total_chunks": context_info["total_chunks"],
                    "embedding_strategy": context_info.get(
                        "embedding_strategy", "single"
                    ),
                    "processing_stats": json.loads(
                        context_info.get("processing_stats", "{}")
                    ),
                },
                "content_distribution": [
                    {
                        "content_type": row["content_type"],
                        "count": row["count"],
                        "avg_code_percentage": (
                            float(row["avg_code_percentage"])
                            if row["avg_code_percentage"]
                            else 0.0
                        ),
                    }
                    for row in content_types
                ],
                "language_distribution": [
                    {"language": row["primary_language"], "count": row["count"]}
                    for row in languages
                ],
                "embedding_model_usage": [
                    {"model": row["model_name"], "chunk_count": row["chunk_count"]}
                    for row in embedding_models
                ],
            }

    async def search_similar_content(
        self, embedding: List[float], limit: int = 10, filters: Dict[str, Any] = None
    ) -> List[Dict]:
        """Search for similar content using single embedding."""

        # Default context_id if not in filters (for testing)
        context_id = filters.get("context_id") if filters else None
        if not context_id:
            # Get first available context for testing
            async with self.pool.acquire() as conn:
                context_row = await conn.fetchrow("SELECT id FROM contexts LIMIT 1")
                if not context_row:
                    return []
                context_id = str(context_row["id"])

        # Pad or truncate embedding to fit vector(4096)
        if len(embedding) > 4096:
            embedding = embedding[:4096]
        elif len(embedding) < 4096:
            embedding = embedding + [0.0] * (4096 - len(embedding))

        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        async with self.pool.acquire() as conn:
            # Build WHERE clause based on filters
            where_conditions = ["c.context_id = $1"]
            params = [uuid.UUID(context_id), embedding_str]
            param_count = 2

            if filters:
                if "content_type" in filters:
                    param_count += 1
                    where_conditions.append(f"ca.content_type = ${param_count}")
                    params.append(filters["content_type"])

                if "primary_language" in filters:
                    param_count += 1
                    where_conditions.append(f"ca.primary_language = ${param_count}")
                    params.append(filters["primary_language"])

                if "min_code_percentage" in filters:
                    param_count += 1
                    where_conditions.append(f"ca.code_percentage >= ${param_count}")
                    params.append(filters["min_code_percentage"])

            where_clause = " AND ".join(where_conditions)

            query = f"""
                SELECT
                    c.id, c.content, d.title, d.url, d.content,
                    ca.content_type, ca.primary_language, ca.summary, ca.code_percentage,
                    1 - (ce.embedding <=> $2::vector) as similarity,
                    d.metadata, c.metadata as chunk_metadata
                FROM chunks c
                JOIN chunk_embeddings ce ON c.id = ce.chunk_id
                JOIN documents d ON c.document_id = d.id
                LEFT JOIN content_analyses ca ON d.id = ca.document_id
                WHERE {where_clause}
                ORDER BY ce.embedding <=> $2::vector
                LIMIT {limit}
            """

            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                result = {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "title": row["title"],
                    "url": row["url"],
                    "similarity": float(row["similarity"]),
                    "content_type": row["content_type"] or "general",
                    "primary_language": row["primary_language"],
                    "summary": row["summary"] or "",
                    "quality_score": 0.8,  # Default quality score
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                results.append(result)

            return results

    async def search_by_keywords(
        self, keywords: List[str], limit: int = 10, filters: Dict[str, Any] = None
    ) -> List[Dict]:
        """Search content by keywords using full-text search."""

        if not keywords:
            return []

        # Default context_id if not in filters (for testing)
        context_id = filters.get("context_id") if filters else None
        if not context_id:
            # Get first available context for testing
            async with self.pool.acquire() as conn:
                context_row = await conn.fetchrow("SELECT id FROM contexts LIMIT 1")
                if not context_row:
                    return []
                context_id = str(context_row["id"])

        # Create search query
        search_query = " & ".join(keywords)

        async with self.pool.acquire() as conn:
            # Build WHERE clause based on filters
            where_conditions = ["c.context_id = $1"]
            params = [uuid.UUID(context_id), search_query]
            param_count = 2

            if filters:
                if "content_type" in filters:
                    param_count += 1
                    where_conditions.append(f"ca.content_type = ${param_count}")
                    params.append(filters["content_type"])

                if "primary_language" in filters:
                    param_count += 1
                    where_conditions.append(f"ca.primary_language = ${param_count}")
                    params.append(filters["primary_language"])

            where_clause = " AND ".join(where_conditions)

            query = f"""
                SELECT
                    c.id, c.content, d.title, d.url, d.content,
                    ca.content_type, ca.primary_language, ca.summary, ca.code_percentage,
                    ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', $2)) as rank,
                    d.metadata, c.metadata as chunk_metadata
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                LEFT JOIN content_analyses ca ON d.id = ca.document_id
                WHERE {where_clause}
                    AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $2)
                ORDER BY rank DESC
                LIMIT {limit}
            """

            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                result = {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "title": row["title"],
                    "url": row["url"],
                    "similarity": float(row["rank"]),
                    "content_type": row["content_type"] or "general",
                    "primary_language": row["primary_language"],
                    "summary": row["summary"] or "",
                    "quality_score": 0.7,  # Default quality score for keyword search
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                results.append(result)

            return results

    # Knowledge Graph Storage Methods

    async def store_content_relationship(
        self,
        source_url: str,
        target_url: str,
        relationship_type: str,
        strength: float,
        confidence: float,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Store a content relationship in the knowledge graph."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO content_relationships (
                        source_url, target_url, relationship_type, strength, confidence, metadata, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                    ON CONFLICT (source_url, target_url, relationship_type) DO UPDATE SET
                        strength = EXCLUDED.strength,
                        confidence = EXCLUDED.confidence,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    source_url,
                    target_url,
                    relationship_type,
                    strength,
                    confidence,
                    json.dumps(metadata or {}),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store content relationship: {e}")
            return False

    async def store_topic_cluster(
        self,
        cluster_id: str,
        name: str,
        description: str,
        content_urls: List[str],
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Store a topic cluster in the knowledge graph."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO topic_clusters (
                        cluster_id, name, description, content_urls, metadata, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (cluster_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        content_urls = EXCLUDED.content_urls,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    cluster_id,
                    name,
                    description,
                    json.dumps(content_urls),
                    json.dumps(metadata or {}),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store topic cluster: {e}")
            return False

    async def store_graph_statistics(
        self,
        total_nodes: int,
        total_edges: int,
        cluster_count: int,
        average_node_degree: float,
        graph_density: float,
        modularity_score: float,
        coverage_ratio: float,
    ) -> bool:
        """Store knowledge graph statistics."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO graph_statistics (
                        total_nodes, total_edges, cluster_count, average_node_degree,
                        graph_density, modularity_score, coverage_ratio, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """,
                    total_nodes,
                    total_edges,
                    cluster_count,
                    average_node_degree,
                    graph_density,
                    modularity_score,
                    coverage_ratio,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store graph statistics: {e}")
            return False

    async def load_content_relationships(self) -> List[Any]:
        """Load all content relationships from the knowledge graph."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT source_url, target_url, relationship_type, strength, confidence, metadata
                    FROM content_relationships
                    ORDER BY created_at DESC
                    """
                )

                relationships = []
                for row in rows:
                    # Create ContentRelationship objects
                    import asyncio

                    from .relationship_mapping import (
                        ContentRelationship,
                        RelationshipType,
                    )

                    relationship = ContentRelationship(
                        source_url=row["source_url"],
                        target_url=row["target_url"],
                        relationship_type=RelationshipType(row["relationship_type"]),
                        strength=row["strength"],
                        confidence=row["confidence"],
                        discovered_method=json.loads(row["metadata"] or "{}").get(
                            "discovered_method", "unknown"
                        ),
                        supporting_evidence=json.loads(row["metadata"] or "{}").get(
                            "supporting_evidence", []
                        ),
                        context_elements=json.loads(row["metadata"] or "{}").get(
                            "context_elements", []
                        ),
                        discovery_timestamp=asyncio.get_event_loop().time(),
                    )
                    relationships.append(relationship)

                return relationships

        except Exception as e:
            logger.error(f"Failed to load content relationships: {e}")
            return []

    async def load_topic_clusters(self) -> List[Any]:
        """Load all topic clusters from the knowledge graph."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT cluster_id, name, description, content_urls, metadata
                    FROM topic_clusters
                    ORDER BY created_at DESC
                    """
                )

                clusters = []
                for row in rows:
                    # Create TopicCluster objects
                    from .relationship_mapping import TopicCluster

                    metadata = json.loads(row["metadata"] or "{}")

                    cluster = TopicCluster(
                        cluster_id=row["cluster_id"],
                        name=row["name"],
                        description=row["description"],
                        content_urls=json.loads(row["content_urls"]),
                        topic_keywords=metadata.get("topic_keywords", []),
                        programming_languages=metadata.get("programming_languages", []),
                        content_types=metadata.get("content_types", []),
                        difficulty_level=metadata.get("difficulty_level"),
                        coherence_score=metadata.get("coherence_score", 0.0),
                        coverage_score=metadata.get("coverage_score", 0.0),
                        quality_score=metadata.get("quality_score", 0.0),
                        related_clusters=metadata.get("related_clusters", []),
                        parent_cluster=metadata.get("parent_cluster"),
                        child_clusters=metadata.get("child_clusters", []),
                    )
                    clusters.append(cluster)

                return clusters

        except Exception as e:
            logger.error(f"Failed to load topic clusters: {e}")
            return []

    async def load_graph_statistics(self) -> Optional[Dict[str, Any]]:
        """Load the latest knowledge graph statistics."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT total_nodes, total_edges, cluster_count, average_node_degree,
                           graph_density, modularity_score, coverage_ratio
                    FROM graph_statistics
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )

                if row:
                    return {
                        "total_nodes": row["total_nodes"],
                        "total_edges": row["total_edges"],
                        "cluster_count": row["cluster_count"],
                        "average_node_degree": row["average_node_degree"],
                        "graph_density": row["graph_density"],
                        "modularity_score": row["modularity_score"],
                        "coverage_ratio": row["coverage_ratio"],
                    }

                return None

        except Exception as e:
            logger.error(f"Failed to load graph statistics: {e}")
            return None

    # Additional search methods for LLM endpoints

    async def keyword_search(
        self, query: str, context_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search content by keyword using full-text search."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                query_sql = """
                    SELECT
                        c.id, c.content, d.title, d.url, d.content as full_content,
                        ca.content_type, ca.primary_language, ca.summary, ca.code_percentage,
                        ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', $2)) as score,
                        d.metadata, c.metadata as chunk_metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    LEFT JOIN content_analyses ca ON d.id = ca.document_id
                    WHERE c.context_id = $1
                        AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $2)
                    ORDER BY score DESC
                    LIMIT $3
                """

                rows = await conn.fetch(query_sql, uuid.UUID(context_id), query, limit)

                results = []
                for row in rows:
                    result = {
                        "id": str(row["id"]),
                        "content": row["content"],
                        "title": row["title"],
                        "url": (
                            row["url"] or row["source_url"]
                            if "source_url" in row
                            else row["url"]
                        ),
                        "source_url": row["url"],
                        "score": float(row["score"]),
                        "similarity": float(row["score"]),  # Alias for compatibility
                        "content_type": row["content_type"] or "general",
                        "primary_language": row["primary_language"],
                        "summary": row["summary"] or "",
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Failed to perform keyword search: {e}")
            return []

    async def search_by_content_type(
        self, content_types: List[str], context_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search content by content type."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        try:
            async with self.pool.acquire() as conn:
                query_sql = """
                    SELECT
                        c.id, c.content, d.title, d.url, d.content as full_content,
                        ca.content_type, ca.primary_language, ca.summary, ca.code_percentage,
                        0.8 as score,  -- Default score for type-based search
                        d.metadata, c.metadata as chunk_metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    LEFT JOIN content_analyses ca ON d.id = ca.document_id
                    WHERE c.context_id = $1
                        AND ca.content_type = ANY($2)
                    ORDER BY d.indexed_at DESC
                    LIMIT $3
                """

                rows = await conn.fetch(
                    query_sql, uuid.UUID(context_id), content_types, limit
                )

                results = []
                for row in rows:
                    result = {
                        "id": str(row["id"]),
                        "content": row["content"],
                        "title": row["title"],
                        "url": (
                            row["url"] or row["source_url"]
                            if "source_url" in row
                            else row["url"]
                        ),
                        "source_url": row["url"],
                        "score": float(row["score"]),
                        "similarity": float(row["score"]),  # Alias for compatibility
                        "content_type": row["content_type"] or "general",
                        "primary_language": row["primary_language"],
                        "summary": row["summary"] or "",
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Failed to search by content type: {e}")
            return []
