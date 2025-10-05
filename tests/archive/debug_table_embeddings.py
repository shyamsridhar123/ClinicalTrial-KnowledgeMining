#!/usr/bin/env python3
"""Debug table embeddings to understand why search returns 0 results."""

import psycopg
from pgvector.psycopg import register_vector
import json

conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
register_vector(conn)
cur = conn.cursor()

print("=" * 80)
print("TABLE EMBEDDING DIAGNOSTICS")
print("=" * 80)

# 1. Count table embeddings
cur.execute("SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'table';")
table_count = cur.fetchone()[0]
print(f"\n📊 Total table embeddings: {table_count}")

# 2. Sample table embeddings
print("\n" + "=" * 80)
print("SAMPLE TABLE EMBEDDINGS")
print("=" * 80)

cur.execute("""
    SELECT chunk_id, nct_id, document_name, metadata
    FROM docintel.embeddings
    WHERE artefact_type = 'table'
    LIMIT 5;
""")

samples = cur.fetchall()
if samples:
    for i, (chunk_id, nct_id, doc_name, metadata) in enumerate(samples, 1):
        print(f"\n{i}. Chunk ID: {chunk_id}")
        print(f"   NCT: {nct_id}")
        print(f"   Document: {doc_name}")
        print(f"   Metadata keys: {list(metadata.keys())[:10]}")
        
        # Check if there's source text or content
        table_id = metadata.get('table_id', 'N/A')
        source_path = metadata.get('source_path', 'N/A')
        print(f"   Table ID: {table_id}")
        print(f"   Source: {source_path}")
else:
    print("❌ No table embeddings found!")

# 3. Check vector files for table content
print("\n" + "=" * 80)
print("CHECKING VECTOR FILES FOR TABLE TEXT")
print("=" * 80)

from pathlib import Path

vector_files = list(Path("data/processing/embeddings/vectors").rglob("*.jsonl"))
table_embeddings_in_files = []

for vfile in vector_files[:5]:  # Check first 5 files
    with open(vfile) as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('metadata', {}).get('artefact_type') == 'table':
                    table_embeddings_in_files.append({
                        'file': vfile.name,
                        'chunk_id': data.get('chunk_id'),
                        'has_embedding': 'embedding' in data,
                        'embedding_len': len(data.get('embedding', [])),
                        'metadata': data.get('metadata', {})
                    })
                    if len(table_embeddings_in_files) >= 3:
                        break
            except:
                continue
    if len(table_embeddings_in_files) >= 3:
        break

if table_embeddings_in_files:
    print(f"\n✅ Found {len(table_embeddings_in_files)} table embeddings in vector files:")
    for item in table_embeddings_in_files:
        print(f"\n  File: {item['file']}")
        print(f"  Chunk: {item['chunk_id']}")
        print(f"  Has embedding: {item['has_embedding']}")
        print(f"  Embedding length: {item['embedding_len']}")
        print(f"  Source: {item['metadata'].get('source_path', 'N/A')}")
else:
    print("\n⚠️  No table embeddings found in vector files")

# 4. Check actual table JSON files to see source content
print("\n" + "=" * 80)
print("CHECKING TABLE SOURCE FILES")
print("=" * 80)

table_files = list(Path("data/processing/tables").rglob("*.json"))
print(f"\nFound {len(table_files)} table JSON files")

if table_files:
    # Check a few samples
    for tfile in table_files[:3]:
        print(f"\n📄 {tfile}")
        try:
            with open(tfile) as f:
                content = f.read()
                data = json.loads(content) if content.strip() else []
                
                if isinstance(data, list):
                    print(f"   Tables in file: {len(data)}")
                    if data:
                        sample = data[0]
                        print(f"   Sample keys: {list(sample.keys())[:10]}")
                        
                        # Check for text content
                        text_fields = ['text', 'caption', 'content', 'cells', 'data']
                        for field in text_fields:
                            if field in sample:
                                value = str(sample[field])[:100]
                                print(f"   {field}: {value}...")
                    else:
                        print(f"   ⚠️  EMPTY ARRAY")
                elif content.strip() == '[]':
                    print(f"   ⚠️  EMPTY ARRAY")
                else:
                    print(f"   Type: {type(data)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

# 5. Test similarity search directly
print("\n" + "=" * 80)
print("TESTING DIRECT SIMILARITY SEARCH")
print("=" * 80)

# Generate a simple embedding for testing (all zeros)
test_embedding = [0.0] * 512

cur.execute("""
    SELECT chunk_id, nct_id, 
           1 - (embedding <=> %s::vector) as similarity
    FROM docintel.embeddings
    WHERE artefact_type = 'table'
    ORDER BY embedding <=> %s::vector
    LIMIT 3;
""", (test_embedding, test_embedding))

results = cur.fetchall()
if results:
    print(f"\n✅ Similarity search returns {len(results)} results with zero vector:")
    for chunk_id, nct_id, sim in results:
        print(f"   {chunk_id}: similarity={sim:.4f}")
else:
    print("\n❌ Similarity search returns 0 results even with zero vector")

# 6. Check if embeddings are all zeros or NaN
print("\n" + "=" * 80)
print("CHECKING EMBEDDING QUALITY")
print("=" * 80)

cur.execute("""
    SELECT chunk_id,
           (SELECT SUM(val) FROM UNNEST(embedding) AS val) as embedding_sum,
           (SELECT AVG(val) FROM UNNEST(embedding) AS val) as embedding_avg
    FROM docintel.embeddings
    WHERE artefact_type = 'table'
    LIMIT 5;
""")

quality_results = cur.fetchall()
if quality_results:
    print("\nEmbedding statistics:")
    for chunk_id, emb_sum, emb_avg in quality_results:
        if emb_sum == 0:
            status = "❌ ALL ZEROS"
        elif emb_sum is None or emb_avg is None:
            status = "❌ NULL/NaN"
        else:
            status = f"✅ sum={emb_sum:.4f}, avg={emb_avg:.4f}"
        print(f"   {chunk_id[:50]:50} | {status}")

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print("""
Possible issues:
1. Table source files are empty ([]) → No text to embed
2. Embeddings are all zeros → Model failed to generate
3. Embeddings are NaN/NULL → Database write error
4. Source path mismatch → Can't find table content

Next steps:
- If source files are empty: Re-run table extraction from documents
- If embeddings are zeros: Re-run embedding with proper table text
- If embeddings are good: Query syntax issue (unlikely)
""")
