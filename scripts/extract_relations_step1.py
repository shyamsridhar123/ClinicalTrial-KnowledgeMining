#!/usr/bin/env python3
"""
STEP 1: Extract Relations from Clinical Text

This script re-processes existing text chunks to extract entity relationships.
Unlike the initial run (which used --skip-relations), this will populate the
docintel.relations table with connections like:
- Drug TREATS Disease
- Drug CAUSES AdverseEvent
- Disease HAS_SYMPTOM Symptom

Usage:
    # Process one specific trial for testing
    pixi run -- python scripts/extract_relations_step1.py --nct-id NCT03799627 --limit 5
    
    # Process all trials (WARNING: takes 1-2 hours, costs Azure OpenAI tokens)
    pixi run -- python scripts/extract_relations_step1.py --all
    
Prerequisites:
    - Entities already extracted (37,657 entities exist in docintel.entities)
    - Text chunks available in data/processing/text/
    - Azure OpenAI GPT-4.1 API key configured in .env
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import psycopg
from psycopg.rows import dict_row

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.knowledge_graph.triple_extraction import ClinicalTripleExtractor, TripleExtractionResult
from docintel.config import get_config

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extracts and persists relations from existing clinical text chunks."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        
        # Load DSN from .env via VectorDatabaseSettings
        from docintel.config import VectorDatabaseSettings
        vector_db_settings = VectorDatabaseSettings()
        self.db_dsn = str(vector_db_settings.dsn)
        
        # Initialize triple extractor WITHOUT skip_relations flag
        self.extractor = ClinicalTripleExtractor(
            fast_mode=False,
            skip_relations=False  # THIS IS THE KEY - we want relations now
        )
        
        self.processed_chunks = 0
        self.extracted_relations = 0
        self.errors = []
        
    async def extract_relations_for_trial(
        self, 
        nct_id: str, 
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract relations for a specific clinical trial.
        
        Args:
            nct_id: NCT identifier (e.g., NCT03799627)
            limit: Maximum number of chunks to process (for testing)
            
        Returns:
            Extraction summary report
        """
        logger.info(f"🔬 Starting relation extraction for {nct_id}")
        
        # Find text chunks for this trial
        text_dir = Path(f"data/processing/text/{nct_id}")
        if not text_dir.exists():
            logger.error(f"No text directory found: {text_dir}")
            return {"status": "error", "message": "NCT directory not found"}
        
        chunk_files = list(text_dir.glob("*.txt"))
        if not chunk_files:
            logger.warning(f"No .txt files found in {text_dir}")
            return {"status": "no_data", "chunks_processed": 0}
        
        if limit:
            chunk_files = chunk_files[:limit]
        
        logger.info(f"📄 Found {len(chunk_files)} chunks to process")
        
        # Process each chunk
        for chunk_file in chunk_files:
            await self._process_chunk_file(nct_id, chunk_file)
        
        return {
            "status": "completed",
            "nct_id": nct_id,
            "chunks_processed": self.processed_chunks,
            "relations_extracted": self.extracted_relations,
            "errors": len(self.errors)
        }
    
    async def extract_relations_all_trials(
        self, 
        limit_per_trial: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract relations for ALL clinical trials.
        
        WARNING: This is expensive and time-consuming!
        Recommend testing with one trial first.
        """
        logger.warning("⚠️  Processing ALL trials - this will take 1-2 hours and cost Azure API credits")
        
        text_root = Path("data/processing/text")
        nct_dirs = [d for d in text_root.iterdir() if d.is_dir() and d.name.startswith("NCT")]
        
        logger.info(f"📚 Found {len(nct_dirs)} trials to process")
        
        all_reports = []
        for nct_dir in nct_dirs:
            nct_id = nct_dir.name
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {nct_id}")
            logger.info(f"{'='*60}")
            
            report = await self.extract_relations_for_trial(nct_id, limit_per_trial)
            all_reports.append(report)
            
            # Small delay to respect API rate limits
            await asyncio.sleep(2)
        
        total_chunks = sum(r.get("chunks_processed", 0) for r in all_reports)
        total_relations = sum(r.get("relations_extracted", 0) for r in all_reports)
        
        return {
            "status": "completed",
            "trials_processed": len(nct_dirs),
            "chunks_processed": total_chunks,
            "relations_extracted": total_relations,
            "trial_reports": all_reports
        }
    
    async def _process_chunk_file(self, nct_id: str, chunk_file: Path) -> None:
        """Process a single text chunk file."""
        try:
            # Read chunk content
            content = chunk_file.read_text(encoding="utf-8")
            
            if len(content.strip()) < 100:
                logger.debug(f"Skipping short chunk: {chunk_file.name}")
                return
            
            logger.info(f"  🔍 Processing {chunk_file.name} ({len(content)} chars)")
            
            # Look for existing chunk_id from embeddings table
            chunk_id = await self._find_chunk_id_for_file(nct_id, chunk_file.stem)
            
            if not chunk_id:
                logger.warning(f"  ⚠️  No chunk_id found for {chunk_file.name}, skipping")
                return
            
            # Extract entities AND relations using GPT-4.1
            # (Previously entities were extracted, now we get relations too)
            logger.info(f"  🤖 Calling Azure OpenAI GPT-4.1 for triple extraction...")
            extraction_result = self.extractor.extract_triples(content, chunk_id)
            
            entity_count = len(extraction_result.entities)
            relation_count = len(extraction_result.relations)
            
            logger.info(f"  ✅ Extracted: {entity_count} entities, {relation_count} relations")
            
            if not self.dry_run and relation_count > 0:
                # Persist relations to database
                await self._persist_relations(chunk_id, extraction_result.relations)
                logger.info(f"  💾 Stored {relation_count} relations to database")
            
            self.processed_chunks += 1
            self.extracted_relations += relation_count
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {chunk_file.name}: {e}")
            self.errors.append({
                "file": str(chunk_file),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _find_chunk_id_for_file(self, nct_id: str, file_stem: str) -> Optional[str]:
        """Find the chunk_id from embeddings table matching this file."""
        conn = await asyncio.to_thread(
            psycopg.connect, self.db_dsn, row_factory=dict_row
        )
        
        try:
            cur = conn.cursor()
            await asyncio.to_thread(
                cur.execute,
                """
                SELECT chunk_id 
                FROM docintel.embeddings 
                WHERE nct_id = %s 
                AND source_path LIKE %s
                LIMIT 1
                """,
                (nct_id, f"%{file_stem}%")
            )
            
            row = await asyncio.to_thread(cur.fetchone)
            if row:
                return row['chunk_id']
            
            # Fallback: try to find by document name
            await asyncio.to_thread(
                cur.execute,
                """
                SELECT chunk_id 
                FROM docintel.embeddings 
                WHERE nct_id = %s 
                LIMIT 1
                """,
                (nct_id,)
            )
            row = await asyncio.to_thread(cur.fetchone)
            return row['chunk_id'] if row else None
            
        finally:
            await asyncio.to_thread(conn.close)
    
    async def _persist_relations(self, chunk_id: str, relations: List[Any]) -> None:
        """Persist extracted relations to docintel.relations table."""
        if not relations:
            return
        
        conn = await asyncio.to_thread(psycopg.connect, self.db_dsn)
        
        try:
            cur = conn.cursor()
            
            # Get entity_ids for subject and object entities
            # This requires matching entity text back to docintel.entities
            
            for relation in relations:
                # Find subject entity_id
                await asyncio.to_thread(
                    cur.execute,
                    """
                    SELECT entity_id FROM docintel.entities 
                    WHERE source_chunk_id = %s 
                    AND entity_text = %s
                    AND entity_type = %s
                    LIMIT 1
                    """,
                    (chunk_id, relation.subject_entity.text, relation.subject_entity.entity_type)
                )
                subject_row = await asyncio.to_thread(cur.fetchone)
                
                # Find object entity_id
                await asyncio.to_thread(
                    cur.execute,
                    """
                    SELECT entity_id FROM docintel.entities 
                    WHERE source_chunk_id = %s 
                    AND entity_text = %s
                    AND entity_type = %s
                    LIMIT 1
                    """,
                    (chunk_id, relation.object_entity.text, relation.object_entity.entity_type)
                )
                object_row = await asyncio.to_thread(cur.fetchone)
                
                if not subject_row or not object_row:
                    logger.debug(f"    Skipping relation (entity not found): {relation.subject_entity.text} -> {relation.object_entity.text}")
                    continue
                
                subject_id = subject_row[0]
                object_id = object_row[0]
                
                # Insert relation with required fields (meta_graph_id, chunk_id)
                import uuid
                relation_id = str(uuid.uuid4())
                meta_graph_id = str(uuid.uuid4())  # TODO: Get actual meta_graph_id from KnowledgeGraphBuilder
                
                await asyncio.to_thread(
                    cur.execute,
                    """
                    INSERT INTO docintel.relations (
                        relation_id, meta_graph_id, chunk_id,
                        subject_entity_id, predicate, object_entity_id,
                        confidence, evidence_span
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (relation_id) DO NOTHING
                    """,
                    (
                        relation_id,
                        meta_graph_id,
                        chunk_id,
                        subject_id,
                        relation.predicate,
                        object_id,
                        relation.confidence,
                        relation.evidence_span if hasattr(relation, 'evidence_span') else None
                    )
                )
            
            await asyncio.to_thread(conn.commit)
            logger.debug(f"    Committed {len(relations)} relations to database")
            
        except Exception as e:
            await asyncio.to_thread(conn.rollback)
            logger.error(f"    Failed to persist relations: {e}")
            raise
        finally:
            await asyncio.to_thread(conn.close)


async def main():
    """Main entry point for relation extraction."""
    parser = argparse.ArgumentParser(
        description="Extract relations from clinical text (Step 1 of graph construction)"
    )
    parser.add_argument(
        "--nct-id",
        help="Process specific NCT ID (e.g., NCT03799627)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL trials (WARNING: expensive and time-consuming!)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks per trial (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract but don't persist to database"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if not args.nct_id and not args.all:
        parser.error("Must specify either --nct-id or --all")
    
    try:
        extractor = RelationExtractor(dry_run=args.dry_run)
        
        if args.all:
            report = await extractor.extract_relations_all_trials(limit_per_trial=args.limit)
        else:
            report = await extractor.extract_relations_for_trial(args.nct_id, limit=args.limit)
        
        # Save report
        report_file = Path("logs/relation_extraction_step1_report.json")
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2))
        
        print("\n" + "="*60)
        print("🎯 STEP 1 COMPLETE: RELATION EXTRACTION")
        print("="*60)
        print(f"Status: {report['status']}")
        if args.all:
            print(f"Trials processed: {report.get('trials_processed', 0)}")
        print(f"Chunks processed: {report.get('chunks_processed', 0)}")
        print(f"Relations extracted: {report.get('relations_extracted', 0)}")
        print(f"Errors: {report.get('errors', 0)}")
        print(f"\nReport saved: {report_file}")
        
        if report['status'] == 'completed' and report.get('relations_extracted', 0) > 0:
            print("\n✅ SUCCESS! Relations are now in docintel.relations table")
            print("\nNext step:")
            print("  pixi run -- python scripts/sync_relations_to_age_step2.py")
        elif report.get('relations_extracted', 0) == 0:
            print("\n⚠️  WARNING: No relations extracted!")
            print("   Check that:")
            print("   1. Azure OpenAI API key is configured")
            print("   2. Text chunks contain extractable relationships")
            print("   3. GPT-4.1 deployment is accessible")
        
    except Exception as e:
        logger.error(f"Relation extraction failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
