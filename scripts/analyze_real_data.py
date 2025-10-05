#!/usr/bin/env python3
"""
Script to analyze the REAL clinical trial data in the database.
This script connects to PostgreSQL and reports on the clinical data currently stored in the system.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    print("❌ psycopg not available - run: pixi add psycopg[binary]")

try:
    from docintel.config import get_config as _get_config
except ImportError:  # pragma: no cover - standalone script execution
    _get_config = None


def _resolve_connection_string(explicit: Optional[str] = None) -> str:
    """Resolve PostgreSQL connection string from explicit value, environment, or config."""
    if explicit:
        return explicit
    env_dsn = os.getenv("DOCINTEL_DSN") or os.getenv("DOCINTEL_VECTOR_DB_DSN")
    if env_dsn:
        return env_dsn
    if _get_config:
        return _get_config().docintel_dsn
    raise RuntimeError("No PostgreSQL connection string configured.")


async def analyze_embeddings_data() -> Dict[str, Any]:
    """Analyze the 2,290 BiomedCLIP embeddings to see what real clinical data exists."""
    if not PSYCOPG_AVAILABLE:
        return {"error": "psycopg not available"}
    
    conn = await psycopg.AsyncConnection.connect(_resolve_connection_string())
    
    try:
        print("🔍 ANALYZING REAL EMBEDDINGS DATA...")
        print("=" * 80)
        
        # Get embeddings summary
        cursor = await conn.execute("""
            SELECT 
                COUNT(*) as total_embeddings,
                COUNT(DISTINCT nct_id) as unique_trials,
                embedding_model,
                quantization_encoding,
                ROUND(AVG(token_count)) as avg_tokens,
                ROUND(AVG(char_count)) as avg_chars
            FROM docintel.embeddings 
            GROUP BY embedding_model, quantization_encoding
        """)
        embeddings_summary = await cursor.fetchone()
        
        print(f"📊 EMBEDDINGS SUMMARY:")
        print(f"  Total embeddings: {embeddings_summary[0]:,}")
        print(f"  Unique clinical trials: {embeddings_summary[1]}")
        print(f"  Model: {embeddings_summary[2]}")
        print(f"  Quantization: {embeddings_summary[3]}")
        print(f"  Average tokens per chunk: {embeddings_summary[4]}")
        print(f"  Average characters per chunk: {embeddings_summary[5]}")
        
        # Get trial breakdown
        cursor = await conn.execute("""
            SELECT 
                nct_id,
                COUNT(*) as embedding_count,
                COUNT(DISTINCT document_name) as document_count,
                STRING_AGG(DISTINCT document_name, ', ') as documents
            FROM docintel.embeddings 
            GROUP BY nct_id
            ORDER BY embedding_count DESC
        """)
        trials = await cursor.fetchall()
        
        print(f"\n📋 CLINICAL TRIALS PROCESSED:")
        for trial in trials:
            print(f"  {trial[0]}: {trial[1]:,} embeddings, {trial[2]} documents ({trial[3]})")
        
        # Sample embedding content
        cursor = await conn.execute("""
            SELECT nct_id, document_name, chunk_id, token_count, char_count
            FROM docintel.embeddings 
            ORDER BY char_count DESC
            LIMIT 5
        """)
        sample_chunks = await cursor.fetchall()
        
        print(f"\n📝 LARGEST CHUNKS (by character count):")
        for chunk in sample_chunks:
            print(f"  {chunk[0]}/{chunk[1]} [{chunk[2]}]: {chunk[3]} tokens, {chunk[4]} chars")
        
        return {
            "total_embeddings": embeddings_summary[0],
            "unique_trials": embeddings_summary[1],
            "model": embeddings_summary[2],
            "quantization": embeddings_summary[3],
            "trials": [{"nct_id": t[0], "embeddings": t[1], "documents": t[2]} for t in trials]
        }
        
    finally:
        await conn.close()


async def analyze_knowledge_graph() -> Dict[str, Any]:
    """Analyze the knowledge graph entities and relations."""
    if not PSYCOPG_AVAILABLE:
        return {"error": "psycopg not available"}
    
    conn = await psycopg.AsyncConnection.connect(_resolve_connection_string())
    
    try:
        print("\n🕸️ ANALYZING KNOWLEDGE GRAPH...")
        print("=" * 80)
        
        # Get entities by type
        cursor = await conn.execute("""
            SELECT 
                entity_type,
                COUNT(*) as count
            FROM ag_catalog.entities 
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        entity_types = await cursor.fetchall()
        
        print(f"👥 ENTITY TYPES:")
        total_entities = sum(et[1] for et in entity_types)
        for entity_type, count in entity_types:
            percentage = (count / total_entities) * 100
            print(f"  {entity_type}: {count} ({percentage:.1f}%)")
        
        # Get relations by predicate
        cursor = await conn.execute("""
            SELECT 
                predicate,
                COUNT(*) as count
            FROM ag_catalog.relations 
            GROUP BY predicate
            ORDER BY count DESC
        """)
        relation_types = await cursor.fetchall()
        
        print(f"\n🔗 RELATION TYPES:")
        total_relations = sum(rt[1] for rt in relation_types)
        for predicate, count in relation_types:
            percentage = (count / total_relations) * 100
            print(f"  {predicate}: {count} ({percentage:.1f}%)")
        
        # Get communities info
        cursor = await conn.execute("""
            SELECT 
                COUNT(*) as total_communities,
                AVG(occurrence) as avg_occurrence,
                MAX(occurrence) as max_occurrence,
                MIN(occurrence) as min_occurrence
            FROM ag_catalog.communities
        """)
        communities_info = await cursor.fetchone()
        
        print(f"\n🏘️ COMMUNITIES:")
        print(f"  Total communities: {communities_info[0]}")
        print(f"  Average occurrence: {communities_info[1]:.3f}")
        print(f"  Occurrence range: {communities_info[3]:.3f} - {communities_info[2]:.3f}")
        
        # Sample entities
        cursor = await conn.execute("""
            SELECT entity_type, entity_text, confidence
            FROM ag_catalog.entities
            WHERE confidence > 0.8
            ORDER BY confidence DESC
            LIMIT 10
        """)
        high_confidence_entities = await cursor.fetchall()
        
        print(f"\n⭐ HIGH CONFIDENCE ENTITIES:")
        for entity_type, text, confidence in high_confidence_entities:
            print(f"  {entity_type}: '{text}' (confidence: {confidence:.3f})")
        
        return {
            "total_entities": total_entities,
            "entity_types": dict(entity_types),
            "total_relations": total_relations,
            "relation_types": dict(relation_types),
            "communities": communities_info[0]
        }
        
    finally:
        await conn.close()


async def check_chunk_files() -> Dict[str, Any]:
    """Check what chunk files exist on disk vs what's in the database."""
    print("\n📁 ANALYZING CHUNK FILES ON DISK...")
    print("=" * 80)
    
    chunk_base_path = Path("data/processing/chunks")
    if not chunk_base_path.exists():
        print(f"❌ Chunk directory not found: {chunk_base_path}")
        return {"error": f"Directory not found: {chunk_base_path}"}
    
    chunk_files = list(chunk_base_path.glob("**/*.json"))
    print(f"📁 Found {len(chunk_files)} chunk files on disk")
    
    # Analyze a few files
    file_analysis = []
    for i, chunk_file in enumerate(chunk_files[:5]):  # First 5 files
        try:
            with chunk_file.open('r') as f:
                data = json.load(f)
            
            chunks = data.get("chunks", []) if isinstance(data, dict) else data
            chunk_count = len(chunks)
            
            # Get sample text from first chunk
            sample_text = ""
            if chunks and isinstance(chunks[0], dict):
                sample_text = chunks[0].get("text", "")[:200] + "..."
            
            file_info = {
                "file": str(chunk_file.relative_to(chunk_base_path)),
                "chunks": chunk_count,
                "sample_text": sample_text
            }
            file_analysis.append(file_info)
            
            print(f"  {chunk_file.name}: {chunk_count} chunks")
            if sample_text:
                print(f"    Sample: {sample_text[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error reading {chunk_file.name}: {e}")
    
    return {
        "total_files": len(chunk_files),
        "sample_files": file_analysis
    }


async def main():
    """Run complete analysis of real clinical trial data."""
    print("🚀 CLINICAL TRIAL DATA ANALYSIS")
    print("🔍 Checking what's currently stored in the database against production expectations")
    print("="*80)
    
    try:
        # Analyze embeddings
        embeddings_result = await analyze_embeddings_data()
        
        # Analyze knowledge graph
        kg_result = await analyze_knowledge_graph()
        
        # Check disk files
        files_result = await check_chunk_files()
        
        print("\n" + "="*80)
        print("📊 ANALYSIS SUMMARY:")
        print("="*80)
        
        if "error" not in embeddings_result:
            print(f"✅ EMBEDDINGS: {embeddings_result['total_embeddings']:,} BiomedCLIP embeddings from {embeddings_result['unique_trials']} clinical trials")
        else:
            print(f"❌ EMBEDDINGS: {embeddings_result['error']}")
        
        if "error" not in kg_result:
            print(f"✅ KNOWLEDGE GRAPH: {kg_result['total_entities']} entities, {kg_result['total_relations']} relations, {kg_result['communities']} communities")
        else:
            print(f"❌ KNOWLEDGE GRAPH: {kg_result['error']}")
        
        if "error" not in files_result:
            print(f"✅ CHUNK FILES: {files_result['total_files']} JSON files on disk")
        else:
            print(f"❌ CHUNK FILES: {files_result['error']}")
        
        print("\n🎯 CONCLUSION:")
        if ("error" not in embeddings_result and 
            "error" not in kg_result and 
            embeddings_result['total_embeddings'] > 0):
            print("✅ Clinical trial data is processed and stored")
            print("✅ System has both vector embeddings and a populated knowledge graph")
            print("ℹ️ Use knowledge_graph_cli.py to orchestrate end-to-end processing when new data arrives")
        else:
            print("❌ System appears to be missing real data or has connection issues")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not PSYCOPG_AVAILABLE:
        print("⚠️  Install psycopg first: pixi add 'psycopg[binary]'")
    else:
        asyncio.run(main())