-- PostgreSQL + Apache AGE + pgvector Schema for Clinical Trial Knowledge Graph
-- This schema supports MedGraphRAG-inspired triple extraction and Meta-Graph construction

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE and set search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create the clinical knowledge graph
SELECT create_graph('clinical_kg');

-- Core metadata tables for document tracking
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id VARCHAR(20),
    document_type VARCHAR(50), -- protocol, csr, sap, etc.
    title TEXT,
    upload_timestamp TIMESTAMP DEFAULT NOW(),
    checksum VARCHAR(64),
    file_path TEXT,
    processing_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_nct_id ON documents(nct_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);

-- Chunks table for storing parsed document sections
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    section_title TEXT,
    content TEXT,
    page_start INTEGER,
    page_end INTEGER,
    embedding vector(768), -- BiomedCLIP embeddings
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Entities extracted from chunks
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type VARCHAR(50), -- medication, condition, procedure, etc.
    start_char INTEGER,
    end_char INTEGER,
    confidence DECIMAL(3,2),
    normalized_id VARCHAR(100), -- UMLS CUI, RxNorm, SNOMED, etc.
    normalized_source VARCHAR(20), -- umls, rxnorm, snomed
    context_flags JSONB, -- negation, uncertainty, temporality from medspaCy
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_chunk_id ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_id, normalized_source);

-- Relations between entities (triples)
CREATE TABLE IF NOT EXISTS relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    subject_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100), -- causes, treats, prevents, etc.
    object_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    confidence DECIMAL(3,2),
    evidence_span TEXT, -- supporting text from the chunk
    evidence_start_char INTEGER,
    evidence_end_char INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relations_chunk_id ON relations(chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);

-- Repository nodes for external vocabularies (UMLS, RxNorm, SNOMED)
CREATE TABLE IF NOT EXISTS repo_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(100) NOT NULL, -- CUI, RxCUI, SCTID
    source VARCHAR(20) NOT NULL, -- umls, rxnorm, snomed
    preferred_term TEXT,
    definition TEXT,
    semantic_types TEXT[], -- for UMLS
    source_version VARCHAR(50),
    ingested_at TIMESTAMP DEFAULT NOW(),
    valid_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_repo_nodes_external ON repo_nodes(external_id, source);
CREATE INDEX IF NOT EXISTS idx_repo_nodes_source ON repo_nodes(source);
CREATE INDEX IF NOT EXISTS idx_repo_nodes_term ON repo_nodes USING gin(to_tsvector('english', preferred_term));

-- Repository edges for vocabulary relationships
CREATE TABLE IF NOT EXISTS repo_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id UUID REFERENCES repo_nodes(id),
    target_node_id UUID REFERENCES repo_nodes(id),
    relationship_type VARCHAR(50), -- isa, broader_than, maps_to, etc.
    source VARCHAR(20), -- vocabulary source
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repo_edges_source ON repo_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_repo_edges_target ON repo_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_repo_edges_type ON repo_edges(relationship_type);

-- Meta-graphs: collections of entities and relations per chunk
CREATE TABLE IF NOT EXISTS meta_graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    entity_count INTEGER,
    relation_count INTEGER,
    processing_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meta_graphs_chunk ON meta_graphs(chunk_id);

-- Tag summaries for hierarchical navigation (U-Retrieval support)
CREATE TABLE IF NOT EXISTS tag_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meta_graph_id UUID REFERENCES meta_graphs(id) ON DELETE CASCADE,
    layer INTEGER, -- 1=domain, 2=subtopic, 3=focus
    tag_id VARCHAR(100),
    tag_label TEXT,
    confidence DECIMAL(3,2),
    embedding vector(768),
    evidence_span_refs TEXT[], -- references to supporting evidence
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tag_summaries_meta_graph ON tag_summaries(meta_graph_id);
CREATE INDEX IF NOT EXISTS idx_tag_summaries_layer ON tag_summaries(layer);
CREATE INDEX IF NOT EXISTS idx_tag_summaries_embedding ON tag_summaries USING hnsw (embedding vector_cosine_ops);

-- Processing logs for audit trail
CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    stage VARCHAR(50), -- parsing, extraction, graph_construction
    status VARCHAR(20), -- started, completed, failed
    message TEXT,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processing_logs_document ON processing_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_logs_stage ON processing_logs(stage);
CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp);

-- AGE graph views for openCypher queries
-- Create AGE vertex labels
SELECT create_vlabel('clinical_kg', 'Entity');
SELECT create_vlabel('clinical_kg', 'RepoNode');

-- Create AGE edge labels  
SELECT create_elabel('clinical_kg', 'RELATES_TO');
SELECT create_elabel('clinical_kg', 'MAPPED_TO');
SELECT create_elabel('clinical_kg', 'ISA');

-- Configuration table for retrieval parameters
CREATE TABLE IF NOT EXISTS retrieval_config (
    key VARCHAR(50) PRIMARY KEY,
    value JSONB,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Default retrieval configuration
INSERT INTO retrieval_config (key, value, description) VALUES
('default_hop_limit', '2', 'Default maximum hops for graph traversal'),
('max_hop_limit', '4', 'Maximum allowed hops for graph traversal'),
('min_entity_confidence', '0.75', 'Minimum confidence threshold for entities'),
('repository_freshness_days', '180', 'Maximum age in days for repository data'),
('tag_counts_per_layer', '{"L1": 8, "L2": 16, "L3": 24}', 'Number of tags to select per hierarchy layer'),
('vector_neighbors_k', '40', 'Number of vector neighbors to retrieve')
ON CONFLICT (key) DO NOTHING;