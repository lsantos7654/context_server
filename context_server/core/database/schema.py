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

            # Create documents table (stores cleaned content for chunks/embeddings)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,  -- Cleaned content with CODE_SNIPPET placeholders
                    metadata JSONB DEFAULT '{}',
                    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    chunk_count INTEGER DEFAULT 0,
                    source_type VARCHAR(20) NOT NULL,
                    UNIQUE(context_id, url)
                )
            """
            )

            # Create raw_documents table (stores original unprocessed content)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_documents (
                    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    raw_content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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
                    title TEXT,  -- LLM-generated title for chunk identification
                    summary TEXT,  -- LLM-generated summary for compact responses
                    summary_model VARCHAR(50),  -- Model used to generate summary
                    text_embedding halfvec(3072),  -- text-embedding-3-large dimension
                    chunk_index INTEGER NOT NULL,
                    code_snippet_ids UUID[],  -- Direct array of related code snippet UUIDs
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
                    embedding halfvec(2048),  -- voyage-code-3 dimension
                    metadata JSONB DEFAULT '{}',
                    snippet_type VARCHAR(20) DEFAULT 'code_block',  -- 'code_block' or 'inline_code'
                    line_count INTEGER NOT NULL DEFAULT 0,  -- Number of lines in code snippet
                    char_count INTEGER NOT NULL DEFAULT 0,  -- Number of characters in code snippet
                    preview TEXT,  -- Meaningful code preview for display
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

            # Migration: Update code_snippets table structure if needed
            await conn.execute("""
                ALTER TABLE code_snippets 
                ADD COLUMN IF NOT EXISTS line_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS char_count INTEGER DEFAULT 0
            """)
            
            # Migration: Remove unused columns if they exist
            try:
                await conn.execute("ALTER TABLE code_snippets DROP COLUMN IF EXISTS start_line")
                await conn.execute("ALTER TABLE code_snippets DROP COLUMN IF EXISTS end_line") 
                await conn.execute("ALTER TABLE code_snippets DROP COLUMN IF EXISTS char_start")
                await conn.execute("ALTER TABLE code_snippets DROP COLUMN IF EXISTS char_end")
            except Exception as e:
                logger.warning(f"Migration warning (expected): {e}")

            # Create indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_text_embedding ON chunks USING ivfflat (text_embedding halfvec_cosine_ops)"
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
