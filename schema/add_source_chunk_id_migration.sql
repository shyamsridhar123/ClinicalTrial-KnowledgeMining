-- Migration: Add source_chunk_id column to docintel.entities
-- Purpose: Store text-based chunk IDs (e.g., "NCT02597946-chunk-0030") to enable
--          direct linking between embeddings and entities for GraphRAG traversal.
--
-- Context: Embeddings use text chunk_ids, entities use UUID chunk_ids.
--          This column bridges the gap.

BEGIN;

-- Add the new column (nullable initially for existing rows)
ALTER TABLE docintel.entities 
ADD COLUMN IF NOT EXISTS source_chunk_id TEXT;

-- Create index for efficient lookups during entity-embedding linkage
CREATE INDEX IF NOT EXISTS idx_entities_source_chunk_id 
    ON docintel.entities(source_chunk_id);

-- Add comment for documentation
COMMENT ON COLUMN docintel.entities.source_chunk_id IS 
    'Text-based chunk identifier matching embeddings.chunk_id format (e.g., NCT02597946-chunk-0030)';

COMMIT;

-- Verification query (run after migration):
-- SELECT COUNT(*), COUNT(source_chunk_id) 
-- FROM docintel.entities;
