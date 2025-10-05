"""CLI entrypoint for knowledge graph construction and management."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .config import get_config
from .knowledge_graph.age_utils import configure_age_session_async
from .knowledge_graph.graph_construction import KnowledgeGraphBuilder


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class GraphConstructionJob:
    """Manages knowledge graph construction and population."""
    
    def __init__(self):
        self.config = get_config()
        self.age_settings = self.config.age_graph
        self.graph_builder = KnowledgeGraphBuilder()
    
    async def build_from_extractions(
        self,
        extraction_file: str,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from extraction results file.
        
        Args:
            extraction_file: JSON file with extraction results
            clear_existing: Clear existing graph data first
            
        Returns:
            Construction summary report
        """
        logging.info("🏗️  Starting knowledge graph construction...")
        
        # Load extraction results
        extraction_path = Path(extraction_file)
        if not extraction_path.exists():
            raise FileNotFoundError(f"Extraction file not found: {extraction_file}")
        
        extraction_data = json.loads(extraction_path.read_text())
        extractions = extraction_data.get("extractions", [])
        
        if not extractions:
            return {"status": "no_data", "message": "No extractions found in file"}
        
        logging.info(f"📊 Found {len(extractions)} extractions to process")
        
        # Clear existing data if requested
        if clear_existing:
            await self._clear_graph_data()
        
        # Process extractions
        meta_graphs_created = 0
        entities_stored = 0
        relations_stored = 0
        errors = 0
        
        # Initialize asset integrator
        from .knowledge_graph.asset_integration import AssetIntegrator
        processing_root = Path(self.config.parsing.processed_storage_root)
        asset_integrator = AssetIntegrator(processing_root)
        
        for extraction in extractions:
            try:
                # Use chunk_uuid which is proper UUID format, not chunk_id (which is a string)
                chunk_id = UUID(extraction["chunk_uuid"])
                
                # Convert extraction data back to result objects
                from .knowledge_graph.triple_extraction import (
                    ClinicalEntity, ClinicalRelation, TripleExtractionResult
                )
                
                entities = []
                for e_data in extraction["entities"]:
                    entity = ClinicalEntity(
                        text=e_data["text"],
                        entity_type=e_data["type"],
                        start_char=e_data["start_char"],
                        end_char=e_data["end_char"],
                        confidence=e_data["confidence"],
                        normalized_id=e_data.get("normalized_id"),
                        normalized_source=e_data.get("normalized_source"),
                        context_flags=e_data.get("context_flags")
                    )
                    entities.append(entity)
                
                # Extract metadata for asset processing
                chunk_metadata = extraction.get("chunk_metadata", {})
                nct_id = chunk_metadata.get("nct_id", "")
                document_id = chunk_metadata.get("document_id", "")
                
                # Process tables and figures if present (document_id optional)
                if nct_id:
                    asset_result = asset_integrator.process_chunk_assets(
                        nct_id=nct_id,
                        document_id=document_id,
                        chunk_metadata=chunk_metadata,
                        existing_entities=entities
                    )
                    
                    # Add table and figure entities to the main entities list
                    entities.extend(asset_result["table_entities"])
                    entities.extend(asset_result["figure_entities"])
                
                relations = []
                for r_data in extraction["relations"]:
                    # Find subject and object entities
                    subject_entity = next(
                        (e for e in entities if e.text == r_data["subject"]), 
                        None
                    )
                    object_entity = next(
                        (e for e in entities if e.text == r_data["object"]), 
                        None
                    )
                    
                    if subject_entity and object_entity:
                        relation = ClinicalRelation(
                            subject_entity=subject_entity,
                            predicate=r_data["predicate"],
                            object_entity=object_entity,
                            confidence=r_data["confidence"],
                            evidence_span=r_data.get("evidence", ""),
                            evidence_start_char=0,  # Would need to recalculate
                            evidence_end_char=0
                        )
                        relations.append(relation)
                
                # Create extraction result
                result = TripleExtractionResult(
                    entities=entities,
                    relations=relations,
                    processing_metadata=extraction.get("metadata", {})
                )
                
                # Extract source_chunk_id from extraction metadata
                source_chunk_id = extraction.get("chunk_id")  # Text-based chunk ID
                
                # Store in knowledge graph
                try:
                    meta_graph_id = await self.graph_builder.create_meta_graph(chunk_id, result, source_chunk_id)
                    
                    meta_graphs_created += 1
                    entities_stored += len(entities)
                    relations_stored += len(relations)
                    
                    logging.info(f"✅ Created meta-graph {meta_graph_id} with "
                               f"{len(entities)} entities, {len(relations)} relations")
                except Exception as db_error:
                    # Check if this is a duplicate key error
                    if "duplicate key value violates unique constraint" in str(db_error):
                        logging.warning(f"⚠️  Skipping duplicate chunk {chunk_id} - already exists in graph")
                        errors += 1
                        continue
                    else:
                        # Re-raise non-duplicate errors
                        raise
                
            except Exception as e:
                logging.error(f"❌ Error processing extraction for chunk "
                            f"{extraction.get('chunk_id', 'unknown')}: {e}")
                errors += 1
                continue
        
        # Generate report
        report = {
            "status": "completed",
            "meta_graphs_created": meta_graphs_created,
            "entities_stored": entities_stored,
            "relations_stored": relations_stored,
            "processing_errors": errors,
            "source_file": extraction_file
        }
        
        logging.info("🎉 Knowledge graph construction completed!")
        return report
    
    async def _clear_graph_data(self) -> None:
        """Clear existing knowledge graph data."""
        logging.info("🗑️  Clearing existing graph data...")
        
        try:
            import psycopg  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
            
            conn = await psycopg.AsyncConnection.connect(
                self.config.docintel_dsn, 
                row_factory=dict_row
            )
            async with conn:
                async with conn.cursor() as cur:
                    await configure_age_session_async(cur, self.age_settings)
 
                     # Clear docintel schema tables in dependency order
                    await cur.execute("DELETE FROM docintel.relations")
                    await cur.execute("DELETE FROM docintel.entities")
                    await cur.execute("DELETE FROM docintel.meta_graphs")
 
                     # Clear AGE graph using proper literal format
                    await cur.execute(
                        f"SELECT * FROM cypher('{self.age_settings.graph_name}', $$ MATCH (n) DETACH DELETE n $$) AS (result agtype)"
                    )
 
                    await conn.commit()
                    logging.info("✅ Graph data cleared")
                    
        except Exception as e:
            logging.error(f"Error clearing graph data: {e}")
            raise
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get current knowledge graph statistics."""
        try:
            import psycopg  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
            
            conn = await psycopg.AsyncConnection.connect(
                self.config.docintel_dsn,
                row_factory=dict_row
            )
            async with conn:
                async with conn.cursor() as cur:
                    await configure_age_session_async(cur, self.age_settings)
 
                     # Get entity counts by type
                    await cur.execute("""
                        SELECT entity_type, COUNT(*) as count
                        FROM ag_catalog.entities
                        GROUP BY entity_type
                        ORDER BY count DESC
                    """)
                    entity_stats = await cur.fetchall()
                    
                    # Get relation counts by predicate
                    await cur.execute("""
                        SELECT predicate, COUNT(*) as count
                        FROM ag_catalog.relations
                        GROUP BY predicate
                        ORDER BY count DESC
                    """)
                    relation_stats = await cur.fetchall()
                    
                    # Get total counts
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.entities")
                    total_entities = (await cur.fetchone())["count"]
                    
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.relations")
                    total_relations = (await cur.fetchone())["count"]
                    
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.meta_graphs")
                    total_meta_graphs = (await cur.fetchone())["count"]
                    
                    return {
                        "total_entities": total_entities,
                        "total_relations": total_relations,
                        "total_meta_graphs": total_meta_graphs,
                        "entity_types": [dict(row) for row in entity_stats],
                        "relation_types": [dict(row) for row in relation_stats]
                    }
                    
        except Exception as e:
            logging.error(f"Error getting graph statistics: {e}")
            return {}


async def _run_async(
    command: str,
    extraction_file: Optional[str] = None,
    clear_existing: bool = False
) -> Dict[str, Any]:
    """Async graph construction runner."""
    job = GraphConstructionJob()
    
    if command == "build":
        if not extraction_file:
            raise ValueError("--extraction-file required for build command")
        return await job.build_from_extractions(extraction_file, clear_existing)
    
    elif command == "stats":
        return await job.get_graph_statistics()
    
    elif command == "clear":
        await job._clear_graph_data()
        return {"status": "cleared", "message": "Graph data cleared"}
    
    else:
        raise ValueError(f"Unknown command: {command}")


def run(
    command: str,
    extraction_file: Optional[str] = None,
    clear_existing: bool = False
) -> Dict[str, Any]:
    """Run knowledge graph construction."""
    return asyncio.run(_run_async(command, extraction_file, clear_existing))


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for graph construction."""
    parser = argparse.ArgumentParser(
        description="Construct and manage clinical knowledge graphs"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge graph from extractions")
    build_parser.add_argument(
        "--extraction-file",
        required=True,
        help="JSON file with extraction results"
    )
    build_parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing graph data before building"
    )
    
    # Stats command
    subparsers.add_parser("stats", help="Show knowledge graph statistics")
    
    # Clear command  
    subparsers.add_parser("clear", help="Clear all graph data")
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        result = run(
            command=args.command,
            extraction_file=getattr(args, 'extraction_file', None),
            clear_existing=getattr(args, 'clear_existing', False)
        )
        
        if args.command == "build":
            print("\n🏗️  GRAPH CONSTRUCTION SUMMARY:")
            print(f"   • Status: {result['status']}")
            print(f"   • Meta-graphs created: {result.get('meta_graphs_created', 0)}")
            print(f"   • Entities stored: {result.get('entities_stored', 0)}")
            print(f"   • Relations stored: {result.get('relations_stored', 0)}")
            
            if result['status'] == 'completed':
                print("\n✅ Knowledge graph ready for querying!")
                print("   Next: pixi run -- python -m docintel.query stats")
        
        elif args.command == "stats":
            print("\n📊 KNOWLEDGE GRAPH STATISTICS:")
            print(f"   • Total entities: {result.get('total_entities', 0)}")
            print(f"   • Total relations: {result.get('total_relations', 0)}")
            print(f"   • Total meta-graphs: {result.get('total_meta_graphs', 0)}")
            
            entity_types = result.get('entity_types', [])
            if entity_types:
                print("\n   📝 Entity Types:")
                for et in entity_types[:10]:  # Top 10
                    print(f"      • {et['entity_type']}: {et['count']}")
            
            relation_types = result.get('relation_types', [])
            if relation_types:
                print("\n   🔗 Relation Types:")
                for rt in relation_types[:10]:  # Top 10
                    print(f"      • {rt['predicate']}: {rt['count']}")
        
        elif args.command == "clear":
            print(f"\n🗑️  {result['message']}")
        
    except Exception as e:
        logging.error(f"Graph construction failed: {e}")
        raise


if __name__ == "__main__":
    main()