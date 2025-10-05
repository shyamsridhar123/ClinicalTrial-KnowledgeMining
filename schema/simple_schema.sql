-- Simple working schema for knowledge graph
-- Drop existing tables if they exist
DROP TABLE IF EXISTS ag_catalog.relations CASCADE;
DROP TABLE IF EXISTS ag_catalog.entities CASCADE;
DROP TABLE IF EXISTS ag_catalog.chunks CASCADE;
DROP TABLE IF EXISTS ag_catalog.documents CASCADE;
DROP TABLE IF EXISTS ag_catalog.meta_graphs CASCADE;
DROP TABLE IF EXISTS ag_catalog.processing_logs CASCADE;

-- Create the clinical_kg graph in AGE
SELECT ag_catalog.create_graph('clinical_kg');

-- Documents table
CREATE TABLE ag_catalog.documents (
    id SERIAL PRIMARY KEY,
    nct_id VARCHAR(20) UNIQUE NOT NULL,
    document_type VARCHAR(50),
    title TEXT,
    phase VARCHAR(20),
    therapeutic_area VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Chunks table
CREATE TABLE ag_catalog.chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES ag_catalog.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Entities table
CREATE TABLE ag_catalog.entities (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES ag_catalog.chunks(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    confidence FLOAT DEFAULT 0.0,
    normalized_id VARCHAR(100),
    normalized_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relations table
CREATE TABLE ag_catalog.relations (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES ag_catalog.chunks(id) ON DELETE CASCADE,
    subject_entity_id INTEGER REFERENCES ag_catalog.entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100) NOT NULL,
    object_entity_id INTEGER REFERENCES ag_catalog.entities(id) ON DELETE CASCADE,
    confidence FLOAT DEFAULT 0.0,
    evidence_span TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meta graphs table for hierarchical organization
CREATE TABLE ag_catalog.meta_graphs (
    id SERIAL PRIMARY KEY,
    graph_type VARCHAR(50),
    source_data JSONB,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing logs
CREATE TABLE ag_catalog.processing_logs (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(50),
    status VARCHAR(20),
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_documents_nct_id ON ag_catalog.documents(nct_id);
CREATE INDEX idx_chunks_document_id ON ag_catalog.chunks(document_id);
CREATE INDEX idx_entities_chunk_id ON ag_catalog.entities(chunk_id);
CREATE INDEX idx_entities_type ON ag_catalog.entities(entity_type);
CREATE INDEX idx_relations_chunk_id ON ag_catalog.relations(chunk_id);
CREATE INDEX idx_relations_subject ON ag_catalog.relations(subject_entity_id);
CREATE INDEX idx_relations_object ON ag_catalog.relations(object_entity_id);
CREATE INDEX idx_relations_predicate ON ag_catalog.relations(predicate);