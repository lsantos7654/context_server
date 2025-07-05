-- Hierarchical Knowledge Graph Schema Design
-- This file defines the enhanced schema for document → chunk → code block hierarchy

-- Enhanced Node Types for Hierarchical Graph
CREATE TABLE IF NOT EXISTS graph_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type VARCHAR(20) NOT NULL CHECK (node_type IN ('document', 'chunk', 'code_block')),
    
    -- Hierarchical references
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Node content and metadata
    title TEXT,
    content TEXT,
    summary TEXT,
    
    -- Node characteristics
    primary_language VARCHAR(50),
    content_type VARCHAR(50),
    complexity_score FLOAT DEFAULT 0.0,
    quality_score FLOAT DEFAULT 0.0,
    
    -- Hierarchical metadata
    depth_level INTEGER NOT NULL DEFAULT 0, -- 0=document, 1=chunk, 2=code_block
    parent_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
    child_count INTEGER DEFAULT 0,
    
    -- Code-specific metadata (for code_block nodes)
    code_metadata JSONB DEFAULT '{}', -- functions, classes, imports, etc.
    
    -- Semantic metadata
    key_concepts JSONB DEFAULT '[]',
    extracted_keywords JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(node_type, document_id, chunk_id),
    
    -- Ensure hierarchical consistency
    CONSTRAINT valid_hierarchy CHECK (
        (node_type = 'document' AND chunk_id IS NULL AND parent_node_id IS NULL) OR
        (node_type = 'chunk' AND chunk_id IS NOT NULL AND parent_node_id IS NOT NULL) OR
        (node_type = 'code_block' AND parent_node_id IS NOT NULL)
    )
);

-- Enhanced Relationships with Hierarchical Types
CREATE TABLE IF NOT EXISTS graph_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Source and target nodes
    source_node_id UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    
    -- Relationship classification
    relationship_type VARCHAR(30) NOT NULL CHECK (relationship_type IN (
        'CONTAINS',           -- Hierarchical containment (document → chunk → code_block)
        'REFERENCES',         -- Cross-references between content
        'SIMILAR_TO',         -- Semantic similarity
        'IMPLEMENTS',         -- Code implementation relationships
        'DEPENDS_ON',         -- Code dependencies
        'EXPLAINS',           -- Tutorial explains code
        'EXEMPLIFIES',        -- Code exemplifies concept
        'DERIVED_FROM',       -- Content derived from other content
        'RELATED_TO'          -- General relatedness
    )),
    
    -- Relationship strength and confidence
    strength FLOAT NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Discovery metadata
    discovery_method VARCHAR(50) NOT NULL, -- 'embedding_similarity', 'text_analysis', 'llm_extraction', 'structural'
    supporting_evidence JSONB DEFAULT '[]',
    extraction_metadata JSONB DEFAULT '{}',
    
    -- Validation and quality
    validated BOOLEAN DEFAULT FALSE,
    validation_score FLOAT,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    
    -- Temporal information
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent duplicate relationships
    UNIQUE(source_node_id, target_node_id, relationship_type),
    
    -- Prevent self-references
    CONSTRAINT no_self_reference CHECK (source_node_id != target_node_id)
);

-- Code Block Extraction Details
CREATE TABLE IF NOT EXISTS code_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Code content
    code_content TEXT NOT NULL,
    language VARCHAR(50),
    
    -- Code structure
    functions JSONB DEFAULT '[]',
    classes JSONB DEFAULT '[]',
    imports JSONB DEFAULT '[]',
    variables JSONB DEFAULT '[]',
    
    -- Position in chunk
    start_line INTEGER,
    end_line INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    
    -- Code characteristics
    complexity_score FLOAT DEFAULT 0.0,
    documentation_score FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(node_id),
    UNIQUE(chunk_id, start_line, end_line)
);

-- Relationship Types with Semantic Weights
CREATE TABLE IF NOT EXISTS relationship_types (
    type_name VARCHAR(30) PRIMARY KEY,
    description TEXT,
    default_weight FLOAT DEFAULT 1.0,
    directional BOOLEAN DEFAULT TRUE,
    hierarchical BOOLEAN DEFAULT FALSE,
    
    -- Search and traversal hints
    traversal_priority INTEGER DEFAULT 1,
    search_boost FLOAT DEFAULT 1.0
);

-- Insert relationship type configurations
INSERT INTO relationship_types (type_name, description, default_weight, directional, hierarchical, traversal_priority, search_boost)
VALUES 
    ('CONTAINS', 'Hierarchical containment relationship', 1.0, TRUE, TRUE, 1, 1.2),
    ('REFERENCES', 'Cross-reference between content pieces', 0.8, TRUE, FALSE, 2, 1.1),
    ('SIMILAR_TO', 'Semantic similarity relationship', 0.7, FALSE, FALSE, 3, 1.0),
    ('IMPLEMENTS', 'Code implementation of concept', 0.9, TRUE, FALSE, 2, 1.3),
    ('DEPENDS_ON', 'Code dependency relationship', 0.8, TRUE, FALSE, 2, 1.1),
    ('EXPLAINS', 'Tutorial explains code/concept', 0.9, TRUE, FALSE, 1, 1.4),
    ('EXEMPLIFIES', 'Code example of concept', 0.8, TRUE, FALSE, 2, 1.2),
    ('DERIVED_FROM', 'Content derived from source', 0.7, TRUE, FALSE, 3, 1.0),
    ('RELATED_TO', 'General relatedness', 0.5, FALSE, FALSE, 4, 0.9)
ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    default_weight = EXCLUDED.default_weight,
    search_boost = EXCLUDED.search_boost;

-- Enhanced Graph Statistics
CREATE TABLE IF NOT EXISTS hierarchical_graph_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Node counts by type
    document_nodes INTEGER DEFAULT 0,
    chunk_nodes INTEGER DEFAULT 0,
    code_block_nodes INTEGER DEFAULT 0,
    total_nodes INTEGER DEFAULT 0,
    
    -- Relationship counts by type
    containment_relationships INTEGER DEFAULT 0,
    semantic_relationships INTEGER DEFAULT 0,
    implementation_relationships INTEGER DEFAULT 0,
    reference_relationships INTEGER DEFAULT 0,
    total_relationships INTEGER DEFAULT 0,
    
    -- Graph quality metrics
    hierarchy_completeness FLOAT DEFAULT 0.0, -- % of documents with full hierarchy
    semantic_connectivity FLOAT DEFAULT 0.0,  -- Average semantic connections per node
    code_coverage FLOAT DEFAULT 0.0,          -- % of chunks with extracted code blocks
    cross_reference_density FLOAT DEFAULT 0.0, -- Cross-references per node
    
    -- Performance metrics
    average_path_length FLOAT DEFAULT 0.0,
    clustering_coefficient FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for Performance

-- Node indexes
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_parent ON graph_nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_document ON graph_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_chunk ON graph_nodes(chunk_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_depth ON graph_nodes(depth_level);

-- Relationship indexes
CREATE INDEX IF NOT EXISTS idx_graph_relationships_source ON graph_relationships(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_target ON graph_relationships(target_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_type ON graph_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_strength ON graph_relationships(strength DESC);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_confidence ON graph_relationships(confidence DESC);

-- Code block indexes
CREATE INDEX IF NOT EXISTS idx_code_blocks_node ON code_blocks(node_id);
CREATE INDEX IF NOT EXISTS idx_code_blocks_chunk ON code_blocks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_code_blocks_language ON code_blocks(language);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_nodes_type_parent ON graph_nodes(node_type, parent_node_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type_strength ON graph_relationships(relationship_type, strength DESC);

-- Functions for Hierarchy Management

-- Function to get all children of a node
CREATE OR REPLACE FUNCTION get_node_children(parent_id UUID)
RETURNS TABLE(
    node_id UUID,
    node_type VARCHAR(20),
    title TEXT,
    depth_level INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE node_hierarchy AS (
        -- Base case: direct children
        SELECT n.id, n.node_type, n.title, n.depth_level
        FROM graph_nodes n
        WHERE n.parent_node_id = parent_id
        
        UNION ALL
        
        -- Recursive case: children of children
        SELECT n.id, n.node_type, n.title, n.depth_level
        FROM graph_nodes n
        INNER JOIN node_hierarchy nh ON n.parent_node_id = nh.node_id
    )
    SELECT * FROM node_hierarchy
    ORDER BY depth_level, title;
END;
$$ LANGUAGE plpgsql;

-- Function to get node ancestry path
CREATE OR REPLACE FUNCTION get_node_ancestry(node_id UUID)
RETURNS TABLE(
    ancestor_id UUID,
    ancestor_type VARCHAR(20),
    ancestor_title TEXT,
    depth_level INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE ancestry AS (
        -- Base case: the node itself
        SELECT n.id, n.node_type, n.title, n.depth_level
        FROM graph_nodes n
        WHERE n.id = node_id
        
        UNION ALL
        
        -- Recursive case: parents
        SELECT n.id, n.node_type, n.title, n.depth_level
        FROM graph_nodes n
        INNER JOIN ancestry a ON n.id = a.parent_node_id
    )
    SELECT * FROM ancestry
    ORDER BY depth_level;
END;
$$ LANGUAGE plpgsql;

-- Function to update graph statistics
CREATE OR REPLACE FUNCTION update_hierarchical_graph_stats()
RETURNS VOID AS $$
DECLARE
    doc_count INTEGER;
    chunk_count INTEGER;
    code_count INTEGER;
    total_nodes INTEGER;
    contain_rels INTEGER;
    semantic_rels INTEGER;
    impl_rels INTEGER;
    ref_rels INTEGER;
    total_rels INTEGER;
BEGIN
    -- Count nodes by type
    SELECT COUNT(*) INTO doc_count FROM graph_nodes WHERE node_type = 'document';
    SELECT COUNT(*) INTO chunk_count FROM graph_nodes WHERE node_type = 'chunk';
    SELECT COUNT(*) INTO code_count FROM graph_nodes WHERE node_type = 'code_block';
    total_nodes := doc_count + chunk_count + code_count;
    
    -- Count relationships by type
    SELECT COUNT(*) INTO contain_rels FROM graph_relationships WHERE relationship_type = 'CONTAINS';
    SELECT COUNT(*) INTO semantic_rels FROM graph_relationships WHERE relationship_type = 'SIMILAR_TO';
    SELECT COUNT(*) INTO impl_rels FROM graph_relationships WHERE relationship_type IN ('IMPLEMENTS', 'EXEMPLIFIES');
    SELECT COUNT(*) INTO ref_rels FROM graph_relationships WHERE relationship_type = 'REFERENCES';
    total_rels := contain_rels + semantic_rels + impl_rels + ref_rels;
    
    -- Insert new statistics record
    INSERT INTO hierarchical_graph_stats (
        document_nodes, chunk_nodes, code_block_nodes, total_nodes,
        containment_relationships, semantic_relationships, 
        implementation_relationships, reference_relationships, total_relationships
    ) VALUES (
        doc_count, chunk_count, code_count, total_nodes,
        contain_rels, semantic_rels, impl_rels, ref_rels, total_rels
    );
END;
$$ LANGUAGE plpgsql;

-- Views for Common Queries

-- View for complete document hierarchy
CREATE OR REPLACE VIEW document_hierarchy AS
SELECT 
    d.id AS document_node_id,
    d.title AS document_title,
    c.id AS chunk_node_id,
    c.title AS chunk_title,
    cb.id AS code_block_node_id,
    cb.title AS code_block_title,
    code.language AS code_language,
    d.depth_level AS doc_depth,
    c.depth_level AS chunk_depth,
    cb.depth_level AS code_depth
FROM graph_nodes d
LEFT JOIN graph_nodes c ON c.parent_node_id = d.id AND c.node_type = 'chunk'
LEFT JOIN graph_nodes cb ON cb.parent_node_id = c.id AND cb.node_type = 'code_block'
LEFT JOIN code_blocks code ON code.node_id = cb.id
WHERE d.node_type = 'document';

-- View for relationship summary by type
CREATE OR REPLACE VIEW relationship_summary AS
SELECT 
    rt.type_name,
    rt.description,
    COUNT(gr.id) AS relationship_count,
    AVG(gr.strength) AS avg_strength,
    AVG(gr.confidence) AS avg_confidence,
    rt.search_boost
FROM relationship_types rt
LEFT JOIN graph_relationships gr ON rt.type_name = gr.relationship_type
GROUP BY rt.type_name, rt.description, rt.search_boost
ORDER BY relationship_count DESC;