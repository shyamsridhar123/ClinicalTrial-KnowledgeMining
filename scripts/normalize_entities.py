#!/usr/bin/env python3
"""Normalize all entities in the knowledge graph against clinical vocabularies.

This script:
1. Fetches all entities from docintel.entities
2. Runs normalization against UMLS, RxNorm, SNOMED, ICD-10, LOINC
3. Updates normalized_id, normalized_source, and normalization_data fields
4. Reports progress and statistics

Usage:
    pixi run -- python scripts/normalize_entities.py [--batch-size 100] [--limit 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.config import get_config
from docintel.knowledge_graph.entity_normalization import (
    ClinicalEntityNormalizer,
    EntityNormalizationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EntityNormalizationJob:
    """Batch normalize entities from the knowledge graph."""

    def __init__(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        cache_dir: str = "./data/vocabulary_cache",
    ):
        self.batch_size = batch_size
        self.limit = limit
        self.config = get_config()
        self.normalizer = ClinicalEntityNormalizer(
            cache_dir=cache_dir,
            enable_scispacy=True,
            max_candidates=5,
            db_dsn=self.config.docintel_dsn,  # Pass database connection for repo_nodes
        )
        
        # Statistics
        self.total_processed = 0
        self.total_normalized = 0
        self.total_failed = 0
        self.stats_by_type: Dict[str, Dict[str, int]] = {}
        self.start_time = datetime.now()

    async def fetch_entities_batch(
        self, 
        conn: psycopg.AsyncConnection, 
        offset: int
    ) -> List[Dict[str, Any]]:
        """Fetch a batch of entities that need normalization."""
        query = """
            SELECT 
                entity_id,
                entity_text,
                entity_type,
                confidence
            FROM docintel.entities
            WHERE normalized_id IS NULL
            ORDER BY entity_id
            LIMIT %s OFFSET %s
        """
        
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, (self.batch_size, offset))
            return await cur.fetchall()

    async def update_entity_normalization(
        self,
        conn: psycopg.AsyncConnection,
        entity_id: str,
        result: EntityNormalizationResult,
    ) -> None:
        """Update entity with normalization results."""
        if result.best_match:
            # Extract normalized data
            normalized_id = result.best_match.concept_id
            normalized_source = result.best_match.vocabulary.value
            
            # Build complete normalization_data JSON
            normalization_data = {
                "best_match": {
                    "concept_id": result.best_match.concept_id,
                    "concept_name": result.best_match.concept_name,
                    "vocabulary": result.best_match.vocabulary.value,
                    "confidence_score": result.best_match.confidence_score,
                    "semantic_type": result.best_match.semantic_type,
                    "definition": result.best_match.definition,
                    "synonyms": result.best_match.synonyms[:10],  # Limit synonyms
                },
                "all_matches": [
                    {
                        "concept_id": norm.concept_id,
                        "concept_name": norm.concept_name,
                        "vocabulary": norm.vocabulary.value,
                        "confidence_score": norm.confidence_score,
                        "semantic_type": norm.semantic_type,
                    }
                    for norm in result.normalizations[:5]  # Top 5 alternatives
                ],
                "metadata": result.processing_metadata,
                "normalized_at": datetime.now().isoformat(),
            }
            
            update_query = """
                UPDATE docintel.entities
                SET 
                    normalized_id = %s,
                    normalized_source = %s,
                    normalization_data = %s,
                    updated_at = NOW()
                WHERE entity_id = %s
            """
            
            async with conn.cursor() as cur:
                await cur.execute(
                    update_query,
                    (normalized_id, normalized_source, json.dumps(normalization_data), entity_id),
                )
        else:
            # Mark as attempted but failed
            normalization_data = {
                "best_match": None,
                "all_matches": [],
                "metadata": result.processing_metadata,
                "normalized_at": datetime.now().isoformat(),
                "status": "no_match_found",
            }
            
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE docintel.entities
                    SET 
                        normalization_data = %s,
                        updated_at = NOW()
                    WHERE entity_id = %s
                    """,
                    (json.dumps(normalization_data), entity_id),
                )

    async def normalize_batch(
        self,
        conn: psycopg.AsyncConnection,
        entities: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Normalize a batch of entities using BULK database query."""
        logger.info(f"🔥 BULK normalizing {len(entities)} entities...")
        
        # Strategy: ONE mega-query for ALL entities instead of individual queries
        # This is 50-100x faster than per-entity queries
        
        # Group entities by type for vocabulary selection
        entities_by_type: Dict[str, List[Dict]] = {}
        for entity in entities:
            entity_type = entity["entity_type"]
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Build ONE massive query for all entities
        all_matches: Dict[str, List[Dict]] = {}  # entity_id -> matches
        
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get ALL unique entity texts
            unique_texts = {entity["entity_text"].lower().strip() for entity in entities}
            unique_texts = {t for t in unique_texts if t and len(t) >= 2}
            
            if not unique_texts:
                return 0, len(entities)
            
            # BULK EXACT MATCH QUERY (super fast with indexes)
            text_placeholders = ','.join(['%s'] * len(unique_texts))
            await cur.execute(
                f"""
                SELECT LOWER(display_name) as match_key, vocabulary, code, 
                       display_name, description, metadata
                FROM docintel.repo_nodes
                WHERE LOWER(display_name) IN ({text_placeholders})
                  AND is_active = true
                UNION ALL
                SELECT LOWER(code) as match_key, vocabulary, code,
                       display_name, description, metadata
                FROM docintel.repo_nodes
                WHERE LOWER(code) IN ({text_placeholders})
                  AND is_active = true
                """,
                (*unique_texts, *unique_texts)
            )
            exact_matches = await cur.fetchall()
            
            # Index exact matches by text
            matches_by_text: Dict[str, List[Dict]] = {}
            for match in exact_matches:
                key = match['match_key']
                if key not in matches_by_text:
                    matches_by_text[key] = []
                matches_by_text[key].append(match)
        
        # Process results and update database
        normalized_count = 0
        failed_count = 0
        
        for entity in entities:
            entity_text = entity["entity_text"]
            entity_type = entity["entity_type"]
            entity_id = entity["entity_id"]
            key = entity_text.lower().strip()
            
            # Track statistics
            if entity_type not in self.stats_by_type:
                self.stats_by_type[entity_type] = {"total": 0, "normalized": 0, "failed": 0}
            self.stats_by_type[entity_type]["total"] += 1
            
            # Check if we have matches
            matches = matches_by_text.get(key, [])
            
            if matches:
                # Pick best match (prefer rxnorm for drugs, umls otherwise)
                best_match = None
                if entity_type.lower() in ['medication', 'drug', 'dosage']:
                    best_match = next((m for m in matches if m['vocabulary'] == 'rxnorm'), matches[0])
                elif entity_type.lower() in ['condition', 'disease', 'adverse_event']:
                    best_match = next((m for m in matches if m['vocabulary'] == 'snomed'), 
                                    next((m for m in matches if m['vocabulary'] == 'umls'), matches[0]))
                else:
                    best_match = matches[0]
                
                # Update database with match
                metadata = best_match.get('metadata') or {}
                normalization_data = {
                    "best_match": {
                        "concept_id": f"{best_match['vocabulary'].upper()}:{best_match['code']}",
                        "concept_name": best_match['display_name'] or best_match['code'],
                        "vocabulary": best_match['vocabulary'],
                        "confidence_score": 1.0,
                        "semantic_type": metadata.get('semantic_type'),
                    },
                    "all_matches": [
                        {"vocabulary": m['vocabulary'], "code": m['code'], "display_name": m['display_name']}
                        for m in matches[:5]
                    ],
                    "normalized_at": datetime.now().isoformat(),
                }
                
                async with conn.cursor() as update_cur:
                    await update_cur.execute(
                        """
                        UPDATE docintel.entities
                        SET normalized_id = %s,
                            normalized_source = %s,
                            normalization_data = %s,
                            updated_at = NOW()
                        WHERE entity_id = %s
                        """,
                        (
                            f"{best_match['vocabulary'].upper()}:{best_match['code']}",
                            best_match['vocabulary'],
                            json.dumps(normalization_data),
                            entity_id
                        )
                    )
                
                normalized_count += 1
                self.stats_by_type[entity_type]["normalized"] += 1
            else:
                failed_count += 1
                self.stats_by_type[entity_type]["failed"] += 1
        
        await conn.commit()
        logger.info(f"   ✅ Batch complete: {normalized_count} normalized, {failed_count} failed")
        return normalized_count, failed_count

    async def run(self) -> None:
        """Execute the normalization job."""
        logger.info("🚀 Starting entity normalization job")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Limit: {self.limit or 'ALL'}")
        logger.info(f"   Database: {self.config.docintel_dsn.split('@')[-1]}")
        
        # Connect to database
        conn = await psycopg.AsyncConnection.connect(
            self.config.docintel_dsn,
            row_factory=dict_row,
        )
        
        try:
            # Get total count
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) as count FROM docintel.entities WHERE normalized_id IS NULL"
                )
                total_to_process = (await cur.fetchone())["count"]
                
                if self.limit and self.limit < total_to_process:
                    total_to_process = self.limit
                
                logger.info(f"📊 Total entities to normalize: {total_to_process:,}")
            
            # Process in batches
            offset = 0
            while True:
                if self.limit and self.total_processed >= self.limit:
                    logger.info(f"⏹️  Reached limit of {self.limit} entities")
                    break
                
                # Fetch batch
                entities = await self.fetch_entities_batch(conn, offset)
                if not entities:
                    logger.info("✅ No more entities to process")
                    break
                
                # Normalize batch
                normalized, failed = await self.normalize_batch(conn, entities)
                
                # Update stats
                self.total_processed += len(entities)
                self.total_normalized += normalized
                self.total_failed += failed
                
                # Progress report
                elapsed = (datetime.now() - self.start_time).total_seconds()
                rate = self.total_processed / elapsed if elapsed > 0 else 0
                progress = (self.total_processed / total_to_process * 100) if total_to_process > 0 else 0
                
                logger.info(
                    f"📈 Progress: {self.total_processed:,}/{total_to_process:,} ({progress:.1f}%) | "
                    f"Normalized: {self.total_normalized:,} ({self.total_normalized/self.total_processed*100:.1f}%) | "
                    f"Rate: {rate:.1f} entities/sec"
                )
                
                offset += self.batch_size
            
            # Final statistics
            await self.print_final_stats(conn)
            
        finally:
            await conn.close()

    async def print_final_stats(self, conn: psycopg.AsyncConnection) -> None:
        """Print comprehensive statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("📊 NORMALIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"⏱️  Total time: {elapsed:.1f} seconds")
        logger.info(f"📝 Total processed: {self.total_processed:,}")
        logger.info(f"✅ Successfully normalized: {self.total_normalized:,} ({self.total_normalized/self.total_processed*100:.1f}%)")
        logger.info(f"❌ Failed to normalize: {self.total_failed:,} ({self.total_failed/self.total_processed*100:.1f}%)")
        logger.info(f"⚡ Average rate: {self.total_processed/elapsed:.1f} entities/sec")
        
        # Print stats by entity type (top 15)
        logger.info("\n📋 NORMALIZATION BY ENTITY TYPE (Top 15):")
        sorted_types = sorted(
            self.stats_by_type.items(),
            key=lambda x: x[1]["total"],
            reverse=True,
        )[:15]
        
        for entity_type, stats in sorted_types:
            success_rate = stats["normalized"] / stats["total"] * 100 if stats["total"] > 0 else 0
            logger.info(
                f"   {entity_type:20} | Total: {stats['total']:5,} | "
                f"Normalized: {stats['normalized']:5,} ({success_rate:5.1f}%) | "
                f"Failed: {stats['failed']:4,}"
            )
        
        # Query final database state
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(normalized_id) as normalized,
                    COUNT(*) - COUNT(normalized_id) as not_normalized
                FROM docintel.entities
            """)
            row = await cur.fetchone()
            
            await cur.execute("""
                SELECT normalized_source, COUNT(*) as count
                FROM docintel.entities
                WHERE normalized_id IS NOT NULL
                GROUP BY normalized_source
                ORDER BY count DESC
            """)
            vocab_stats = await cur.fetchall()
        
        logger.info("\n🗄️  DATABASE STATE:")
        logger.info(f"   Total entities: {row['total']:,}")
        logger.info(f"   Normalized: {row['normalized']:,} ({row['normalized']/row['total']*100:.1f}%)")
        logger.info(f"   Not normalized: {row['not_normalized']:,} ({row['not_normalized']/row['total']*100:.1f}%)")
        
        if vocab_stats:
            logger.info("\n📚 BY VOCABULARY SOURCE:")
            for vocab_row in vocab_stats:
                logger.info(f"   {vocab_row['normalized_source']:10} : {vocab_row['count']:,}")
        
        # Get cache stats
        cache_stats = await self.normalizer.get_normalization_stats()
        logger.info("\n💾 CACHE STATISTICS:")
        logger.info(f"   Total cached: {cache_stats['cache_stats']['total_cached']:,}")
        logger.info(f"   By vocabulary: {cache_stats['cache_stats'].get('by_vocabulary', {})}")
        
        logger.info("="*80 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point."""
    job = EntityNormalizationJob(
        batch_size=args.batch_size,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    await job.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Normalize entities in the knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize all entities
  pixi run -- python scripts/normalize_entities.py
  
  # Normalize in smaller batches
  pixi run -- python scripts/normalize_entities.py --batch-size 50
  
  # Test with first 1000 entities
  pixi run -- python scripts/normalize_entities.py --limit 1000
  
  # Use custom cache directory
  pixi run -- python scripts/normalize_entities.py --cache-dir /tmp/vocab_cache
        """,
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of entities to process in each batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entities to process (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/vocabulary_cache",
        help="Directory for vocabulary cache (default: ./data/vocabulary_cache)",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("\n⚠️  Normalization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Normalization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
