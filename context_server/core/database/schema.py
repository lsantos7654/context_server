"""Database schema creation and migrations."""

import logging

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages database schema creation and migrations."""

    async def create_base_schema(self, pool):
        """Create base database schema."""
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create contexts table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-large',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0
                )
            """
            )

            # Create documents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    chunk_count INTEGER DEFAULT 0,
                    source_type VARCHAR(20) NOT NULL,
                    UNIQUE(context_id, url)
                )
            """
            )

            # Create chunks table with vector embeddings
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    summary TEXT,  -- LLM-generated summary for compact responses
                    summary_model VARCHAR(50),  -- Model used to generate summary
                    text_embedding halfvec(3072),  -- text-embedding-3-large dimension
                    code_embedding halfvec(2048),  -- voyage-code-3 dimension
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

            # Create code_snippets table with vector embeddings
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    language VARCHAR(50) NOT NULL DEFAULT 'text',
                    embedding halfvec(2048),  -- voyage-code-3 dimension
                    metadata JSONB DEFAULT '{}',
                    start_line INTEGER,
                    end_line INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    snippet_type VARCHAR(20) DEFAULT 'code_block',  -- 'code_block' or 'inline_code'
                    summary TEXT,
                    summary_model VARCHAR(50),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Create jobs table for tracking document processing jobs
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(100) PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    context_id UUID REFERENCES contexts(id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    progress REAL NOT NULL DEFAULT 0.0 CHECK (progress >= 0.0 AND progress <= 1.0),
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}',
                    error_message TEXT,
                    result_data JSONB DEFAULT '{}'
                )
            """
            )

            # Add line tracking columns to existing tables (if they don't exist)
            await conn.execute(
                """
                DO $$ BEGIN
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_line INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_line INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_start INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_end INTEGER;
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END $$;
                """
            )

            # Create indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_text_embedding ON chunks USING ivfflat (text_embedding halfvec_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_code_embedding ON chunks USING ivfflat (code_embedding halfvec_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_context_id ON chunks(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_context_id ON documents(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)"
            )

            # Code snippets indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_embedding ON code_snippets USING ivfflat (embedding halfvec_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_context_id ON code_snippets(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_language ON code_snippets(language)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_lines ON chunks(document_id, start_line, end_line)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_chars ON chunks(document_id, char_start, char_end)"
            )

            # Create jobs indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_context_id ON jobs(context_id)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)"
            )

            # Create full-text search indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks USING gin(to_tsvector('english', content))"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', content))"
            )

            logger.info("Database schema created successfully")


__all__ = ["SchemaManager"]