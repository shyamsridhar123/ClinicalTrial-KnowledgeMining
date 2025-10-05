#!/usr/bin/env python3
"""Check if tables and images have embeddings."""

import psycopg
from pgvector.psycopg import register_vector

# Direct connection
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
register_vector(conn)
cur = conn.cursor()

print("=" * 60)
print("EMBEDDINGS BREAKDOWN BY ARTEFACT TYPE")
print("=" * 60)

# Count embeddings by artefact_type
cur.execute("""
    SELECT 
        COALESCE(artefact_type, 'NULL') as type,
        COUNT(*) as count,
        COUNT(DISTINCT nct_id) as trials
    FROM docintel.embeddings
    GROUP BY artefact_type
    ORDER BY count DESC;
""")

results = cur.fetchall()
total = sum(r[1] for r in results)

print(f"\n📊 Total embeddings: {total:,}\n")
for artefact_type, count, trials in results:
    pct = (count / total * 100) if total > 0 else 0
    print(f"   {artefact_type:20} : {count:5,} embeddings ({pct:5.1f}%) | {trials:2} trials")

# Check for table and image specific data
print("\n" + "=" * 60)
print("TABLE & IMAGE CONTENT CHECK")
print("=" * 60)

# Check metadata for table/figure indicators
cur.execute("""
    SELECT 
        chunk_id,
        artefact_type,
        metadata::text LIKE '%table%' as has_table_ref,
        metadata::text LIKE '%figure%' as has_figure_ref,
        metadata::text LIKE '%image%' as has_image_ref
    FROM docintel.embeddings
    WHERE metadata::text ILIKE '%table%' 
       OR metadata::text ILIKE '%figure%'
       OR metadata::text ILIKE '%image%'
    LIMIT 10;
""")

metadata_results = cur.fetchall()
if metadata_results:
    print(f"\n✅ Found {len(metadata_results)} embeddings with table/figure/image references:")
    for chunk_id, art_type, has_table, has_fig, has_img in metadata_results[:5]:
        refs = []
        if has_table: refs.append("table")
        if has_fig: refs.append("figure")
        if has_img: refs.append("image")
        print(f"   • {chunk_id[:40]:40} | type={art_type:10} | refs={','.join(refs)}")
else:
    print("\n⚠️  No embeddings with table/figure/image metadata references found")

# Check knowledge graph for table/figure entities
print("\n" + "=" * 60)
print("KNOWLEDGE GRAPH: TABLES & FIGURES")
print("=" * 60)

cur.execute("""
    SELECT 
        entity_type,
        COUNT(*) as count
    FROM docintel.entities
    WHERE entity_type IN ('table_caption', 'figure', 'table_data', 'measurement', 'timepoint', 'TABLE', 'FIGURE')
    GROUP BY entity_type
    ORDER BY count DESC;
""")

kg_results = cur.fetchall()
if kg_results:
    print("\n📈 Entities in knowledge graph:")
    total_kg = sum(r[1] for r in kg_results)
    for entity_type, count in kg_results:
        print(f"   • {entity_type:20} : {count:6,} entities")
    print(f"   {'TOTAL':20} : {total_kg:6,} table/figure entities")
else:
    print("\n⚠️  No table/figure entities found in knowledge graph")

# Check if table/figure entities have embeddings
print("\n" + "=" * 60)
print("DO TABLE/FIGURE ENTITIES HAVE EMBEDDINGS?")
print("=" * 60)

cur.execute("""
    SELECT 
        e.entity_type,
        COUNT(DISTINCT e.entity_id) as total_entities,
        COUNT(DISTINCT CASE WHEN emb.chunk_id IS NOT NULL THEN e.entity_id END) as entities_with_embeddings
    FROM docintel.entities e
    LEFT JOIN docintel.embeddings emb ON e.chunk_id::text = emb.chunk_id
    WHERE e.entity_type IN ('table_caption', 'figure', 'table_data', 'measurement', 'timepoint', 'TABLE', 'FIGURE')
    GROUP BY e.entity_type
    ORDER BY total_entities DESC;
""")

entity_emb_results = cur.fetchall()
if entity_emb_results:
    print("\n🔗 Entity → Embedding linkage:")
    for entity_type, total, with_emb in entity_emb_results:
        pct = (with_emb / total * 100) if total > 0 else 0
        status = "✅" if pct > 0 else "❌"
        print(f"   {status} {entity_type:20} : {with_emb:6,}/{total:6,} ({pct:5.1f}%) have embeddings")

# Check chunks directory for table/figure content
print("\n" + "=" * 60)
print("FILE SYSTEM: TABLE/FIGURE CHUNKS")
print("=" * 60)

from pathlib import Path
import json

chunks_dir = Path("data/processing/chunks")
if chunks_dir.exists():
    table_fig_chunks = []
    for chunk_file in chunks_dir.rglob("*.json"):
        try:
            with open(chunk_file) as f:
                data = json.load(f)
                text = data.get('text', '').lower()
                metadata = data.get('metadata', {})
                
                if ('table' in text or 'figure' in text or 
                    'table' in str(metadata).lower() or 'figure' in str(metadata).lower()):
                    table_fig_chunks.append({
                        'file': chunk_file.name,
                        'chunk_id': data.get('chunk_id', 'unknown'),
                        'has_table': 'table' in text,
                        'has_figure': 'figure' in text
                    })
        except:
            continue
    
    if table_fig_chunks:
        print(f"\n✅ Found {len(table_fig_chunks)} chunk files with table/figure content:")
        for item in table_fig_chunks[:10]:
            refs = []
            if item['has_table']: refs.append('table')
            if item['has_figure']: refs.append('figure')
            print(f"   • {item['file']:30} | {item['chunk_id'][:30]:30} | {','.join(refs)}")
        if len(table_fig_chunks) > 10:
            print(f"   ... and {len(table_fig_chunks) - 10} more")
    else:
        print("\n⚠️  No chunk files with explicit table/figure content found")
else:
    print("\n❌ Chunks directory does not exist")

conn.close()
print("\n" + "=" * 60)
