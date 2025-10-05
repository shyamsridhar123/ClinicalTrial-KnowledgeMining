#!/usr/bin/env python3
"""
Diagnose and fix entity-embedding linkage issue.
Problem: chunk_id format differs between embeddings and entities.
"""

import psycopg
import json
from pathlib import Path

conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()

print("=" * 80)
print("ENTITY-EMBEDDING LINKAGE DIAGNOSIS")
print("=" * 80)

# 1. Compare chunk_id formats
print("\n" + "=" * 80)
print("CHUNK_ID FORMAT COMPARISON")
print("=" * 80)

print("\n📌 EMBEDDINGS chunk_id samples:")
cur.execute("""
    SELECT chunk_id, artefact_type, nct_id
    FROM docintel.embeddings
    WHERE artefact_type = 'chunk'
    LIMIT 5;
""")
for chunk_id, atype, nct in cur.fetchall():
    print(f"   {chunk_id} | {atype} | {nct}")

print("\n📌 ENTITIES chunk_id samples:")
cur.execute("""
    SELECT DISTINCT chunk_id::text, entity_type
    FROM docintel.entities
    LIMIT 10;
""")
entity_chunks = cur.fetchall()
for chunk_id, etype in entity_chunks[:5]:
    print(f"   {chunk_id} | {etype}")

# 2. Check if chunk_id is UUID in entities
print("\n" + "=" * 80)
print("ENTITIES CHUNK_ID TYPE CHECK")
print("=" * 80)

cur.execute("""
    SELECT 
        pg_typeof(chunk_id) as chunk_id_type,
        COUNT(*) as count
    FROM docintel.entities
    GROUP BY pg_typeof(chunk_id);
""")
type_info = cur.fetchall()
print(f"\nchunk_id type in entities table:")
for typ, count in type_info:
    print(f"   {typ}: {count:,} rows")

# 3. Find what fields entities DO have that could link
print("\n" + "=" * 80)
print("ENTITY FIELDS THAT COULD LINK TO EMBEDDINGS")
print("=" * 80)

cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'docintel' AND table_name = 'entities'
    ORDER BY ordinal_position;
""")
entity_columns = cur.fetchall()
print("\nEntity table columns:")
for col, dtype in entity_columns:
    print(f"   {col:25} : {dtype}")

# 4. Sample entity with all fields
print("\n" + "=" * 80)
print("SAMPLE ENTITY (Full Record)")
print("=" * 80)

cur.execute("""
    SELECT 
        entity_id,
        chunk_id::text,
        entity_text,
        entity_type,
        asset_kind,
        asset_ref,
        provenance::text
    FROM docintel.entities
    WHERE entity_type = 'medication'
    LIMIT 1;
""")

sample = cur.fetchone()
if sample:
    entity_id, chunk_id, text, etype, asset_kind, asset_ref, prov = sample
    print(f"\nEntity ID: {entity_id}")
    print(f"chunk_id: {chunk_id}")
    print(f"entity_text: {text}")
    print(f"entity_type: {etype}")
    print(f"asset_kind: {asset_kind}")
    print(f"asset_ref: {asset_ref}")
    print(f"provenance: {prov[:200] if prov else 'None'}...")

# 5. Check provenance structure - might have linking info
print("\n" + "=" * 80)
print("PROVENANCE STRUCTURE")
print("=" * 80)

cur.execute("""
    SELECT provenance
    FROM docintel.entities
    WHERE provenance IS NOT NULL
    LIMIT 1;
""")

prov_sample = cur.fetchone()
if prov_sample and prov_sample[0]:
    print("\nProvenance JSON structure:")
    prov_data = prov_sample[0]
    if isinstance(prov_data, dict):
        print(f"Keys: {list(prov_data.keys())}")
        print(f"Content: {json.dumps(prov_data, indent=2)[:500]}...")

# 6. Check extraction JSON files for linking info
print("\n" + "=" * 80)
print("EXTRACTION FILES (Source of entities)")
print("=" * 80)

extraction_files = list(Path("data/processing/structured").rglob("*.json"))
print(f"\nFound {len(extraction_files)} extraction files")

if extraction_files:
    sample_file = extraction_files[0]
    print(f"\nSample extraction file: {sample_file}")
    with open(sample_file) as f:
        data = json.load(f)
        if isinstance(data, dict):
            print(f"  Top-level keys: {list(data.keys())[:10]}")
            if 'entities' in data:
                entities = data['entities']
                if entities and len(entities) > 0:
                    sample_entity = entities[0]
                    print(f"  Sample entity keys: {list(sample_entity.keys())}")
                    print(f"  Sample entity: {json.dumps(sample_entity, indent=2)[:300]}...")

# 7. Propose linking strategies
print("\n" + "=" * 80)
print("POSSIBLE LINKING STRATEGIES")
print("=" * 80)

print("""
Based on the data, here are potential ways to link embeddings ↔ entities:

Strategy 1: By NCT ID + Document
   - Embeddings have: nct_id, document_name
   - Entities have: asset_ref might contain document info
   - Link: Match all entities from same document
   - Granularity: Document-level (loose)

Strategy 2: By chunk_id modification
   - Extract NCT from embedding chunk_id: "NCT12345-chunk-0001"
   - Match entities where chunk_id contains that NCT
   - Requires: Understanding entity chunk_id format

Strategy 3: By provenance metadata
   - Check if entity.provenance contains chunk reference
   - May have page numbers, positions that map to chunk ranges

Strategy 4: Add linking table
   - Create new table: chunk_entity_links (chunk_id_text, entity_id)
   - Populate during extraction or as post-process
   - Most reliable but requires re-processing

Strategy 5: By asset_ref mapping
   - Parse asset_ref to extract document/page info
   - Match to embedding metadata (page_reference, source_path)
   - Requires: Understanding asset_ref format
""")

# 8. Test Strategy 1 (NCT + Document level)
print("\n" + "=" * 80)
print("TESTING STRATEGY 1: NCT + Document Matching")
print("=" * 80)

# Get an embedding
cur.execute("""
    SELECT chunk_id, nct_id, document_name
    FROM docintel.embeddings
    WHERE artefact_type = 'chunk'
    LIMIT 1;
""")
emb_sample = cur.fetchone()

if emb_sample:
    emb_chunk, emb_nct, emb_doc = emb_sample
    print(f"\nTest embedding:")
    print(f"  chunk_id: {emb_chunk}")
    print(f"  NCT: {emb_nct}")
    print(f"  Document: {emb_doc}")
    
    # Try to find entities from same NCT
    cur.execute("""
        SELECT entity_type, entity_text, COUNT(*) as count
        FROM docintel.entities
        WHERE asset_ref LIKE %s
        GROUP BY entity_type, entity_text
        ORDER BY count DESC
        LIMIT 5;
    """, (f"%{emb_nct}%",))
    
    entities_by_nct = cur.fetchall()
    if entities_by_nct:
        print(f"\n✅ Found entities by NCT match:")
        for etype, etext, count in entities_by_nct:
            print(f"   • {etype:20}: {etext[:40]:40} ({count}x)")
    else:
        print("\n❌ No entities found by NCT match")

conn.close()

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
Based on the diagnosis, recommend:

1. IMMEDIATE (Document-level linking):
   - Use asset_ref to match entities to embeddings by NCT + document
   - Good enough for most GraphRAG queries
   - Implementation: Add JOIN on NCT pattern matching

2. BETTER (Chunk-level linking):
   - Parse provenance to get page/position info
   - Match to embedding page_reference and char ranges
   - More precise entity context

3. BEST (Rebuild with proper linking):
   - Modify extraction to store embedding chunk_id directly
   - Or create linking table during embedding phase
   - Enables exact chunk → entity traversal

Choose based on urgency vs. precision needs.
""")
