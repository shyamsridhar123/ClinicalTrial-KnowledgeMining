-- Indexes to speed up entity normalization queries
-- Run with: pixi run -- psql $DOCINTEL_VECTOR_DB_DSN -f scripts/create_normalization_indexes.sql

\timing on
\echo '🔧 Creating indexes for repo_nodes normalization...'

-- 1. Index for exact match lookups (case-insensitive)
CREATE INDEX IF NOT EXISTS idx_repo_nodes_display_name_lower 
ON docintel.repo_nodes (vocabulary, LOWER(display_name))
WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_repo_nodes_code_lower 
ON docintel.repo_nodes (vocabulary, LOWER(code))
WHERE is_active = true;

-- 2. Text search index for fuzzy matching (GIN index for fast prefix/contains searches)
CREATE INDEX IF NOT EXISTS idx_repo_nodes_display_name_trgm 
ON docintel.repo_nodes USING gin (display_name gin_trgm_ops)
WHERE is_active = true;

-- 3. Composite index for vocabulary + text search
CREATE INDEX IF NOT EXISTS idx_repo_nodes_vocab_display 
ON docintel.repo_nodes (vocabulary, display_name)
WHERE is_active = true;

-- 4. Index for active records only
CREATE INDEX IF NOT EXISTS idx_repo_nodes_active 
ON docintel.repo_nodes (is_active)
WHERE is_active = true;

-- Enable pg_trgm extension for trigram similarity (better fuzzy matching)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\echo '✅ Indexes created successfully!'
\echo ''
\echo '📊 Index sizes:'
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE schemaname = 'docintel' 
  AND tablename = 'repo_nodes'
ORDER BY pg_relation_size(indexrelid) DESC;
