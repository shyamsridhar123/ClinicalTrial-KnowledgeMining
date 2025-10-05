"""CLI entrypoint for knowledge graph querying and analysis."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .knowledge_graph.age_utils import configure_age_session_async
from .knowledge_graph.graph_construction import GraphQueryService


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class QueryJob:
    """Manages knowledge graph queries and analysis."""
    
    def __init__(self):
        self.config = get_config()
        self.age_settings = self.config.age_graph
        self.query_service = GraphQueryService()
    
    async def run_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics."""
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
                    stats = {}
                    
                    # Basic counts
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.documents")
                    stats["documents"] = (await cur.fetchone())["count"]
                    
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.chunks")
                    stats["chunks"] = (await cur.fetchone())["count"]
                    
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.entities")
                    stats["entities"] = (await cur.fetchone())["count"]
                    
                    await cur.execute("SELECT COUNT(*) as count FROM ag_catalog.relations")
                    stats["relations"] = (await cur.fetchone())["count"]
                    
                    # Entity distribution
                    await cur.execute("""
                        SELECT entity_type, COUNT(*) as count
                        FROM ag_catalog.entities
                        GROUP BY entity_type
                        ORDER BY count DESC
                        LIMIT 20
                    """)
                    stats["entity_types"] = [dict(row) for row in await cur.fetchall()]
                    
                    # Relation distribution
                    await cur.execute("""
                        SELECT predicate, COUNT(*) as count
                        FROM ag_catalog.relations
                        GROUP BY predicate
                        ORDER BY count DESC
                        LIMIT 20
                    """)
                    stats["relation_types"] = [dict(row) for row in await cur.fetchall()]
                    
                    # Top entities by frequency
                    await cur.execute("""
                        SELECT entity_text, entity_type, COUNT(*) as frequency
                        FROM ag_catalog.entities
                        GROUP BY entity_text, entity_type
                        ORDER BY frequency DESC
                        LIMIT 50
                    """)
                    stats["top_entities"] = [dict(row) for row in await cur.fetchall()]
                    
                    return stats
                    
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            return {}
    
    async def search_entities(
        self,
        search_term: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for entities by text or type."""
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
                    query = """
                        SELECT 
                            e.entity_text,
                            e.entity_type,
                            e.confidence,
                            e.normalized_id,
                            e.normalized_source,
                            d.nct_id,
                            d.document_type
                        FROM ag_catalog.entities e
                        JOIN ag_catalog.chunks c ON e.chunk_id = c.id
                        JOIN ag_catalog.documents d ON c.document_id = d.id
                        WHERE e.entity_text ILIKE %s
                    """
                    
                    params = [f"%{search_term}%"]
                    
                    if entity_type:
                        query += " AND e.entity_type = %s"
                        params.append(entity_type)
                    
                    query += " ORDER BY e.confidence DESC LIMIT %s"
                    params.append(limit)
                    
                    await cur.execute(query, params)
                    return [dict(row) for row in await cur.fetchall()]
                    
        except Exception as e:
            logging.error(f"Error searching entities: {e}")
            return []
    
    async def find_relations(
        self,
        entity_text: Optional[str] = None,
        predicate: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find relations by entity or predicate."""
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
                    query = """
                        SELECT 
                            e1.entity_text as subject,
                            e1.entity_type as subject_type,
                            r.predicate,
                            e2.entity_text as object,
                            e2.entity_type as object_type,
                            r.confidence,
                            r.evidence_span,
                            d.nct_id
                        FROM ag_catalog.relations r
                        JOIN ag_catalog.entities e1 ON r.subject_entity_id = e1.id
                        JOIN ag_catalog.entities e2 ON r.object_entity_id = e2.id
                        JOIN ag_catalog.chunks c ON r.chunk_id = c.id
                        JOIN ag_catalog.documents d ON c.document_id = d.id
                        WHERE 1=1
                    """
                    
                    params = []
                    
                    if entity_text:
                        query += " AND (e1.entity_text ILIKE %s OR e2.entity_text ILIKE %s)"
                        params.extend([f"%{entity_text}%", f"%{entity_text}%"])
                    
                    if predicate:
                        query += " AND r.predicate ILIKE %s"
                        params.append(f"%{predicate}%")
                    
                    query += " ORDER BY r.confidence DESC LIMIT %s"
                    params.append(limit)
                    
                    await cur.execute(query, params)
                    return [dict(row) for row in await cur.fetchall()]
                    
        except Exception as e:
            logging.error(f"Error finding relations: {e}")
            return []
    
    async def run_cypher_query(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the AGE graph."""
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
                    await cur.execute(
                        "SELECT * FROM ag_catalog.cypher(%s, %s) AS (result ag_catalog.agtype)",
                        (self.age_settings.graph_name, cypher),
                    )
                    results = await cur.fetchall()

                    # Convert agtype results to regular dict
                    return [dict(row) for row in results]

        except Exception as e:
            logging.error(f"Error executing Cypher query: {e}")
            return []
    
    async def analyze_document(self, nct_id: str) -> Dict[str, Any]:
        """Analyze entities and relations for a specific document."""
        return await self.query_service.get_document_summary(nct_id)


async def _run_async(
    command: str,
    search_term: Optional[str] = None,
    entity_type: Optional[str] = None,
    predicate: Optional[str] = None,
    nct_id: Optional[str] = None,
    cypher: Optional[str] = None,
    limit: int = 20,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Async query runner."""
    job = QueryJob()
    
    if command == "stats":
        result = await job.run_statistics()
        
    elif command == "search":
        if not search_term:
            raise ValueError("--search-term required for search command")
        result = await job.search_entities(search_term, entity_type, limit)
        
    elif command == "relations":
        result = await job.find_relations(search_term, predicate, limit)
        
    elif command == "cypher":
        if not cypher:
            raise ValueError("--cypher required for cypher command")
        result = await job.run_cypher_query(cypher)
        
    elif command == "analyze":
        if not nct_id:
            raise ValueError("--nct-id required for analyze command")
        result = await job.analyze_document(nct_id)
        
    else:
        raise ValueError(f"Unknown command: {command}")
    
    # Save output if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str))
        logging.info(f"💾 Results saved to {output_path}")
    
    return result


def run(
    command: str,
    search_term: Optional[str] = None,
    entity_type: Optional[str] = None,
    predicate: Optional[str] = None,
    nct_id: Optional[str] = None,
    cypher: Optional[str] = None,
    limit: int = 20,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run knowledge graph query."""
    return asyncio.run(_run_async(
        command, search_term, entity_type, predicate, 
        nct_id, cypher, limit, output_file
    ))


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for graph querying."""
    parser = argparse.ArgumentParser(
        description="Query and analyze clinical knowledge graphs"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show knowledge graph statistics")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for entities")
    search_parser.add_argument("--search-term", required=True, help="Text to search for")
    search_parser.add_argument("--entity-type", help="Filter by entity type")
    search_parser.add_argument("--limit", type=int, default=20, help="Maximum results")
    
    # Relations command
    relations_parser = subparsers.add_parser("relations", help="Find relations")
    relations_parser.add_argument("--search-term", help="Entity text to search for")
    relations_parser.add_argument("--predicate", help="Relation predicate to filter by")
    relations_parser.add_argument("--limit", type=int, default=20, help="Maximum results")
    
    # Cypher command
    cypher_parser = subparsers.add_parser("cypher", help="Execute Cypher query")
    cypher_parser.add_argument("--cypher", required=True, help="Cypher query to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific document")
    analyze_parser.add_argument("--nct-id", required=True, help="NCT ID to analyze")
    
    # Global options
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        result = run(
            command=args.command,
            search_term=getattr(args, 'search_term', None),
            entity_type=getattr(args, 'entity_type', None),
            predicate=getattr(args, 'predicate', None),
            nct_id=getattr(args, 'nct_id', None),
            cypher=getattr(args, 'cypher', None),
            limit=getattr(args, 'limit', 20),
            output_file=getattr(args, 'output', None)
        )
        
        if args.command == "stats":
            print("\n📊 KNOWLEDGE GRAPH STATISTICS:")
            print(f"   • Documents: {result.get('documents', 0)}")
            print(f"   • Chunks: {result.get('chunks', 0)}")
            print(f"   • Entities: {result.get('entities', 0)}")
            print(f"   • Relations: {result.get('relations', 0)}")
            
            entity_types = result.get('entity_types', [])[:10]
            if entity_types:
                print("\n   📝 Top Entity Types:")
                for et in entity_types:
                    print(f"      • {et['entity_type']}: {et['count']}")
            
            relation_types = result.get('relation_types', [])[:10]
            if relation_types:
                print("\n   🔗 Top Relation Types:")
                for rt in relation_types:
                    print(f"      • {rt['predicate']}: {rt['count']}")
        
        elif args.command in ["search", "relations"]:
            if isinstance(result, list):
                print(f"\n🔍 Found {len(result)} results:")
                for i, item in enumerate(result[:10], 1):  # Show first 10
                    if args.command == "search":
                        print(f"   {i}. \"{item['entity_text']}\" ({item['entity_type']})")
                        print(f"      NCT: {item['nct_id']}, Confidence: {item['confidence']:.3f}")
                    else:  # relations
                        print(f"   {i}. {item['subject']} --[{item['predicate']}]--> {item['object']}")
                        print(f"      NCT: {item['nct_id']}, Confidence: {item['confidence']:.3f}")
        
        elif args.command == "cypher":
            print(f"\n⚡ Cypher Results ({len(result)} rows):")
            for i, row in enumerate(result[:10], 1):
                print(f"   {i}. {row}")
        
        elif args.command == "analyze":
            print(f"\n📋 Analysis for {args.nct_id}:")
            entity_counts = result.get('entity_counts', [])
            if entity_counts:
                print("   📝 Entity Counts:")
                for ec in entity_counts:
                    print(f"      • {ec['entity_type']}: {ec['count']}")
            
            relation_counts = result.get('relation_counts', [])
            if relation_counts:
                print("   🔗 Relation Counts:")
                for rc in relation_counts:
                    print(f"      • {rc['predicate']}: {rc['count']}")
        
        if args.output:
            print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise


if __name__ == "__main__":
    main()