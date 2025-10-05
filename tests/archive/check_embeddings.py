#!/usr/bin/env python3
"""Quick check of embedding status."""

import psycopg
from pathlib import Path

# Direct connection
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()

# Check pgvector extension
print("=" * 60)
print("PGVECTOR EXTENSION")
print("=" * 60)
cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
result = cur.fetchall()
if result:
    print(f"✅ pgvector installed: version {result[0][1]}")
else:
    print("❌ pgvector NOT installed")

# Check embeddings table
print("\n" + "=" * 60)
print("EMBEDDINGS TABLE")
print("=" * 60)
cur.execute("""
    SELECT COUNT(*) 
    FROM information_schema.tables 
    WHERE table_schema = 'docintel' AND table_name = 'embeddings';
""")
table_exists = cur.fetchone()[0] > 0

if table_exists:
    print("✅ docintel.embeddings table exists")
    
    # Get table structure
    cur.execute("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns 
        WHERE table_schema = 'docintel' AND table_name = 'embeddings'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()
    print(f"\n📋 Table has {len(columns)} columns:")
    for col, dtype, udt in columns[:10]:  # Show first 10
        print(f"   • {col}: {dtype} ({udt})")
    if len(columns) > 10:
        print(f"   ... and {len(columns) - 10} more columns")
    
    # Count embeddings
    cur.execute("SELECT COUNT(*) as total, COUNT(DISTINCT nct_id) as trials FROM docintel.embeddings;")
    total, trials = cur.fetchone()
    print(f"\n📊 Embeddings in database:")
    print(f"   • Total embeddings: {total:,}")
    print(f"   • Unique trials: {trials}")
    
    if total > 0:
        # Sample embedding
        from pgvector.psycopg import register_vector
        register_vector(conn)
        
        cur.execute("""
            SELECT nct_id, document_name, chunk_id, 
                   embedding_model,
                   token_count
            FROM docintel.embeddings 
            LIMIT 1;
        """)
        sample = cur.fetchone()
        print(f"\n🔍 Sample embedding:")
        print(f"   • NCT ID: {sample[0]}")
        print(f"   • Document: {sample[1]}")
        print(f"   • Chunk: {sample[2]}")
        print(f"   • Model: {sample[3]}")
        print(f"   • Tokens: {sample[4]}")
else:
    print("❌ docintel.embeddings table DOES NOT EXIST")

# Check file system embeddings
print("\n" + "=" * 60)
print("FILE SYSTEM EMBEDDINGS")
print("=" * 60)
vectors_dir = Path("data/processing/embeddings/vectors")
if vectors_dir.exists():
    trial_dirs = [d for d in vectors_dir.iterdir() if d.is_dir()]
    total_files = sum(len(list(d.glob("*.jsonl"))) for d in trial_dirs)
    print(f"✅ Embeddings directory exists")
    print(f"   • Trial directories: {len(trial_dirs)}")
    print(f"   • Total JSONL files: {total_files}")
    
    # Sample file
    if total_files > 0:
        sample_file = next(vectors_dir.rglob("*.jsonl"))
        import json
        with open(sample_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_data = json.loads(line)
                    break
            else:
                sample_data = None
        
        if sample_data:
            print(f"\n🔍 Sample file: {sample_file.name}")
            print(f"   • Chunk ID: {sample_data['chunk_id']}")
            print(f"   • Model: {sample_data['metadata'].get('model', 'N/A')}")
            print(f"   • Embedding dimensions: {len(sample_data.get('embedding', []))}")
else:
    print("❌ Embeddings directory does not exist")

# Check BiomedCLIP model
print("\n" + "=" * 60)
print("BIOMEDCLIP MODEL")
print("=" * 60)
model_dir = Path("models/biomedclip")
if model_dir.exists():
    files = list(model_dir.rglob("*"))
    print(f"✅ BiomedCLIP model directory exists")
    print(f"   • Total files: {len(files)}")
    print(f"   • Location: {model_dir}")
else:
    print("❌ BiomedCLIP model directory does not exist")

conn.close()
print("\n" + "=" * 60)
