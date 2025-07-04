-- Initialize database with pgvector extension
-- This runs automatically when the container starts

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE context_server TO context_user;
GRANT ALL ON SCHEMA public TO context_user;

-- Set up proper collation for full-text search
CREATE COLLATION IF NOT EXISTS english_ci (provider = icu, locale = 'en-US-u-ks-level1');

-- Create any additional indexes or configurations here
