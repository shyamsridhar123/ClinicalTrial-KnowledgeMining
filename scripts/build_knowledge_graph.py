#!/usr/bin/env python3
"""
Knowledge Graph Builder Script

Processes all parsed clinical trial documents through the complete pipeline:
1. Load chunks from data/processing/
2. Extract entities and relations using Azure OpenAI GPT-4.1 + medspaCy
3. Construct knowledge graph with AGE + pgvector
4. Store in PostgreSQL database

Usage:
    pixi run -- python scripts/build_knowledge_graph.py
    pixi run -- python scripts/build_knowledge_graph.py --nct-id NCT03981107
    pixi run -- python scripts/build_knowledge_graph.py --limit 5
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.config import get_config
from docintel.knowledge_graph.triple_extraction import ClinicalTripleExtractor
from docintel.knowledge_graph.graph_construction import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class KnowledgeGraphBuildJob:
    """Manages the complete knowledge graph construction process."""
    
    def __init__(self):
        self.config = get_config()
        self.extractor = ClinicalTripleExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.processed_count = 0
        self.error_count = 0
        
    async def build_complete_graph(
        self, 
        nct_id: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from all available clinical trial documents.
        
        Args:
            nct_id: Process only specific NCT ID (optional)
            limit: Maximum number of documents to process (optional)
            
        Returns:
            Build summary report
        """
        logger.info("🧬 Starting knowledge graph construction...")
        
        # Find all available chunks
        processing_root = Path("data/processing/text")
        chunks = self._discover_chunks(processing_root, nct_id, limit)
        
        if not chunks:
            logger.warning("No chunks found to process")
            return {"status": "no_data", "processed": 0, "errors": 0}
        
        logger.info(f"📄 Found {len(chunks)} chunks to process")
        
        # Process chunks in batches to avoid overwhelming the API
        batch_size = 5
        total_entities = 0
        total_relations = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            batch_entities, batch_relations = await self._process_chunk_batch(batch)
            total_entities += batch_entities
            total_relations += batch_relations
            
            # Small delay to respect API rate limits
            await asyncio.sleep(1)
        
        # Generate final report
        report = {
            "status": "completed",
            "documents_processed": len(set(chunk["nct_id"] for chunk in chunks)),
            "chunks_processed": self.processed_count,
            "chunks_failed": self.error_count,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "database_status": "populated"
        }
        
        logger.info("🎉 Knowledge graph construction completed!")
        logger.info(f"📊 Summary: {report}")
        
        return report
    
    def _discover_chunks(
        self, 
        processing_root: Path, 
        nct_id: Optional[str], 
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Discover all available text chunks for processing."""
        chunks = []
        
        # Find all NCT directories
        nct_dirs = [d for d in processing_root.iterdir() if d.is_dir() and d.name.startswith("NCT")]
        
        if nct_id:
            nct_dirs = [d for d in nct_dirs if d.name == nct_id]
        
        for nct_dir in nct_dirs:
            # Find all chunk files directly in NCT directory
            found_chunks = False
            for chunk_file in nct_dir.glob("*.txt"):
                found_chunks = True
                try:
                        # Read chunk content
                        content = chunk_file.read_text(encoding="utf-8")
                        
                        # Look for corresponding metadata
                        metadata_file = chunk_file.with_suffix(".json")
                        metadata = {}
                        if metadata_file.exists():
                            metadata = json.loads(metadata_file.read_text())
                        
                        chunks.append({
                            "chunk_id": str(uuid4()),
                            "nct_id": nct_dir.name,
                            "file_path": str(chunk_file),
                            "content": content,
                            "metadata": metadata,
                            "size": len(content)
                        })
                        
                except Exception as e:
                    logger.error(f"Error reading chunk {chunk_file}: {e}")
                    continue
            
            if not found_chunks:
                logger.warning(f"No text files found for {nct_dir.name}")
        
        # Apply limit if specified
        if limit:
            chunks = chunks[:limit]
        
        logger.info(f"Discovered {len(chunks)} chunks from {len(set(c['nct_id'] for c in chunks))} documents")
        return chunks
    
    async def _process_chunk_batch(self, chunks: List[Dict[str, Any]]) -> tuple[int, int]:
        """Process a batch of chunks through the extraction pipeline."""
        batch_entities = 0
        batch_relations = 0
        
        for chunk in chunks:
            try:
                chunk_id = chunk["chunk_id"]
                content = chunk["content"]
                
                if len(content.strip()) < 50:  # Skip very short chunks
                    logger.debug(f"Skipping short chunk {chunk_id}")
                    continue
                
                logger.info(f"🔬 Processing chunk {chunk_id[:8]}... from {chunk['nct_id']}")
                
                # Extract triples using GPT-4.1 + medspaCy
                extraction_result = self.extractor.extract_triples(chunk_id, content)
                
                if extraction_result.entities or extraction_result.relations:
                    # Store in knowledge graph
                    meta_graph_id = await self.graph_builder.create_meta_graph(
                        chunk_id, extraction_result
                    )
                    
                    batch_entities += len(extraction_result.entities)
                    batch_relations += len(extraction_result.relations)
                    
                    logger.info(f"✅ Stored {len(extraction_result.entities)} entities, "
                              f"{len(extraction_result.relations)} relations for chunk {chunk_id[:8]}...")
                else:
                    logger.info(f"⚠️  No entities/relations found in chunk {chunk_id[:8]}...")
                
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                self.error_count += 1
                continue
        
        return batch_entities, batch_relations


async def main():
    """Main entry point for knowledge graph construction."""
    parser = argparse.ArgumentParser(
        description="Build clinical knowledge graph from parsed documents"
    )
    parser.add_argument(
        "--nct-id", 
        help="Process only specific NCT ID (e.g., NCT03981107)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Maximum number of chunks to process"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    try:
        # Build knowledge graph
        builder = KnowledgeGraphBuildJob()
        report = await builder.build_complete_graph(
            nct_id=args.nct_id,
            limit=args.limit
        )
        
        # Save report
        report_file = Path("logs/knowledge_graph_build_report.json")
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2))
        
        print("\n🎯 KNOWLEDGE GRAPH BUILD SUMMARY:")
        print(f"   • Status: {report['status']}")
        print(f"   • Documents processed: {report.get('documents_processed', 0)}")
        print(f"   • Chunks processed: {report.get('chunks_processed', 0)}")
        print(f"   • Total entities: {report.get('total_entities', 0)}")
        print(f"   • Total relations: {report.get('total_relations', 0)}")
        print(f"   • Report saved: {report_file}")
        
        if report["status"] == "completed":
            print("\n🚀 Knowledge graph ready for querying!")
            print("   Next steps:")
            print("   1. Query entities: SELECT * FROM ag_catalog.entities LIMIT 10;")
            print("   2. Query relations: SELECT * FROM ag_catalog.relations LIMIT 10;")
            print("   3. Graph traversal: SELECT * FROM cypher('{builder.config.age_graph.graph_name}', $$ MATCH (n) RETURN n LIMIT 5 $$);")
        
    except Exception as e:
        logger.error(f"Knowledge graph construction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))