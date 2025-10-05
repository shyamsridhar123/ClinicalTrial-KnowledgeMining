#!/usr/bin/env python3
"""
Extract relations from pre-chunked embeddings data.

This script processes chunks from docintel.embeddings table (~944 chars avg),
extracts entities that already exist, and identifies relationships between them.

Usage:
    # Dry run on 10 chunks
    pixi run -- python scripts/extract_relations_from_chunks.py --limit 10 --dry-run
    
    # Process specific trial
    pixi run -- python scripts/extract_relations_from_chunks.py --nct-id NCT03799627
    
    # Process all chunks
    pixi run -- python scripts/extract_relations_from_chunks.py --all
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

sys.path.insert(0, 'src')

import psycopg
from psycopg.rows import dict_row
from openai import AzureOpenAI

from docintel.config import VectorDatabaseSettings, AzureOpenAISettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/relation_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChunkRelationExtractor:
    """Extract relations from pre-chunked embedding data."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        
        # Load config from .env
        vector_db_settings = VectorDatabaseSettings()
        azure_settings = AzureOpenAISettings()
        
        self.db_dsn = str(vector_db_settings.dsn)
        
        # Setup Azure OpenAI
        api_key = azure_settings.api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        
        self.llm_client = AzureOpenAI(
            api_key=api_key,
            api_version=azure_settings.api_version,
            azure_endpoint=str(azure_settings.endpoint)
        )
        self.deployment_name = azure_settings.deployment_name
        
        logger.info(f"Initialized (dry_run={dry_run})")
    
    def get_chunks_to_process(self, nct_id: str = None, limit: int = None) -> List[Dict]:
        """Get chunks from embeddings table."""
        conn = psycopg.connect(self.db_dsn)
        
        query = """
            SELECT 
                e.chunk_id,
                e.nct_id,
                e.document_name,
                e.section,
                e.char_count,
                e.source_path
            FROM docintel.embeddings e
            WHERE e.artefact_type = 'chunk'
            AND e.char_count > 100  -- Skip tiny chunks
        """
        
        params = []
        if nct_id:
            query += " AND e.nct_id = %s"
            params.append(nct_id)
        
        query += " ORDER BY e.nct_id, e.chunk_id"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            chunks = cur.fetchall()
        
        conn.close()
        
        logger.info(f"Found {len(chunks)} chunks to process")
        return [dict(c) for c in chunks]
    
    def get_entities_for_chunk(self, chunk_id: str) -> List[Dict]:
        """Get entities for a chunk from database."""
        conn = psycopg.connect(self.db_dsn)
        
        with conn.cursor(row_factory=dict_row) as cur:
            # Get entities for this chunk
            cur.execute("""
                SELECT 
                    entity_id,
                    entity_text,
                    entity_type,
                    start_char,
                    end_char,
                    confidence,
                    normalized_id,
                    normalized_source
                FROM docintel.entities
                WHERE source_chunk_id = %s
                ORDER BY start_char
            """, (chunk_id,))
            
            entities = [dict(e) for e in cur.fetchall()]
        
        conn.close()
        return entities
    
    def extract_relations(self, chunk_id: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities using GPT-4.1."""
        
        if len(entities) < 2:
            logger.info(f"  Skipping {chunk_id}: only {len(entities)} entities")
            return []
        
        # If too many entities, take top N by confidence
        if len(entities) > 40:
            entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)[:40]
            logger.info(f"  Limited to top 40 entities by confidence")
        
        # Build entity list for prompt with context
        entity_list = []
        for i, ent in enumerate(entities):
            entity_list.append(
                f"E{i}: {ent['entity_text']} ({ent['entity_type']})"
            )
        
        prompt = f"""You are a clinical relationship extraction system. Extract relationships between these co-occurring clinical entities.

Entities from the same document chunk:
{chr(10).join(entity_list)}

RELATIONSHIP TYPES:
- treats: medication/procedure treats condition
- causes: medication/condition causes adverse_event
- measured_by: condition measured by procedure/test
- administered_with: medications given together
- associated_with: entities statistically/clinically associated
- evaluated_in: drug evaluated in population/condition
- administered_to: drug/procedure administered to population
- has_symptom: condition has symptom
- contraindicated_with: entities should not be combined

Extract ALL plausible relationships based on clinical knowledge and the fact these entities co-occur in the same context.

For each relationship provide:
- subject_id: entity ID (e.g., "E0")
- predicate: relationship type from list above
- object_id: entity ID (e.g., "E1")
- confidence: 0.7-1.0 (0.7=possible, 0.8=likely, 0.9=certain)
- evidence: brief clinical reasoning

Response (JSON array only):
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise clinical relationship extraction system. Always respond with valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            # Try to fix common JSON issues
            try:
                relations_raw = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error, attempting to fix: {e}")
                # Try to extract just the array part
                if '[' in content and ']' in content:
                    start = content.index('[')
                    end = content.rindex(']') + 1
                    content = content[start:end]
                    relations_raw = json.loads(content)
                else:
                    raise
            
            # Convert E0, E1 references to actual entity IDs
            relations = []
            for rel in relations_raw:
                try:
                    subj_idx = int(rel['subject_id'].replace('E', ''))
                    obj_idx = int(rel['object_id'].replace('E', ''))
                    
                    if subj_idx >= len(entities) or obj_idx >= len(entities):
                        continue
                    
                    relations.append({
                        'subject_entity_id': entities[subj_idx]['entity_id'],
                        'object_entity_id': entities[obj_idx]['entity_id'],
                        'predicate': rel['predicate'],
                        'confidence': rel.get('confidence', 0.8),
                        'evidence_span': rel.get('evidence', '')[:500],
                        'chunk_id': chunk_id
                    })
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed relation: {rel} - {e}")
                    continue
            
            logger.info(f"  ✅ Extracted {len(relations)} relations")
            return relations
            
        except Exception as e:
            logger.error(f"  ❌ Relation extraction failed: {e}")
            return []
    
    def store_relations(self, relations: List[Dict]) -> int:
        """Store relations in database."""
        if not relations:
            return 0
        
        if self.dry_run:
            logger.info(f"  [DRY RUN] Would store {len(relations)} relations")
            return len(relations)
        
        conn = psycopg.connect(self.db_dsn)
        stored = 0
        
        with conn.cursor() as cur:
            for rel in relations:
                try:
                    cur.execute("""
                        INSERT INTO docintel.relations 
                        (relation_id, subject_entity_id, predicate, object_entity_id, confidence, evidence_span)
                        VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        rel['subject_entity_id'],
                        rel['predicate'],
                        rel['object_entity_id'],
                        rel['confidence'],
                        rel['evidence_span']
                    ))
                    stored += 1
                except Exception as e:
                    logger.warning(f"Failed to store relation: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"  💾 Stored {stored} relations to database")
        return stored
    
    def process(self, nct_id: str = None, limit: int = None):
        """Main processing loop."""
        chunks = self.get_chunks_to_process(nct_id=nct_id, limit=limit)
        
        total_relations = 0
        processed = 0
        skipped = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk['chunk_id']
            logger.info(f"[{i}/{len(chunks)}] Processing {chunk_id} (NCT: {chunk['nct_id']})")
            
            # Get entities for this chunk
            entities = self.get_entities_for_chunk(chunk_id)
            
            if len(entities) < 2:
                skipped += 1
                logger.info(f"  ⏭️  Skipping: only {len(entities)} entities")
                continue
            
            logger.info(f"  � {len(entities)} entities found")
            
            # Extract relations between co-occurring entities
            relations = self.extract_relations(chunk_id, entities)
            
            # Store relations
            stored = self.store_relations(relations)
            total_relations += stored
            processed += 1
        
        logger.info(f"""
        
=== EXTRACTION COMPLETE ===
Processed: {processed} chunks
Skipped: {skipped} chunks
Total relations extracted: {total_relations}
        """)


def main():
    parser = argparse.ArgumentParser(description='Extract relations from pre-chunked embeddings')
    parser.add_argument('--nct-id', help='Process specific NCT ID')
    parser.add_argument('--limit', type=int, help='Limit number of chunks to process')
    parser.add_argument('--all', action='store_true', help='Process all chunks')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no database writes)')
    
    args = parser.parse_args()
    
    if not args.all and not args.nct_id and not args.limit:
        print("Error: Must specify --nct-id, --limit, or --all")
        sys.exit(1)
    
    extractor = ChunkRelationExtractor(dry_run=args.dry_run)
    extractor.process(nct_id=args.nct_id, limit=args.limit)


if __name__ == '__main__':
    main()
