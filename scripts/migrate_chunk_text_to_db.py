"""
Migration script to populate embeddings.chunk_text from JSON files.

This script:
1. Reads chunk text from data/processing/chunks/**/*.json
2. Reads table markdown from data/processing/tables/**/*.json
3. Matches by chunk_id and updates embeddings.chunk_text
4. Provides progress tracking and error handling
5. Supports rollback on errors

Usage:
    pixi run -- python scripts/migrate_chunk_text_to_db.py

Expected time: 5-10 minutes for current data (456 chunks + 353 tables)
"""

import json
import psycopg
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import sys

load_dotenv()


def load_chunks_from_files(chunks_dir: Path) -> Dict[str, str]:
    """Load chunk text from JSON files, keyed by chunk_id."""
    chunk_texts = {}
    
    json_files = list(chunks_dir.rglob('*.json'))
    print(f'📁 Found {len(json_files)} chunk JSON files')
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                
            if not isinstance(chunks, list):
                chunks = [chunks]
            
            for chunk in chunks:
                chunk_id = chunk.get('id')
                text = chunk.get('text', '')
                
                if chunk_id and text:
                    chunk_texts[chunk_id] = text
                    
        except Exception as e:
            print(f'⚠️  Error reading {json_file}: {e}')
            continue
    
    print(f'✅ Loaded {len(chunk_texts)} text chunks from files')
    return chunk_texts


def load_tables_from_files(tables_dir: Path) -> Dict[str, str]:
    """Load table markdown from JSON files, keyed by table ID."""
    table_texts = {}
    
    json_files = list(tables_dir.rglob('*.json'))
    print(f'📁 Found {len(json_files)} table JSON files')
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                tables = json.load(f)
            
            if not isinstance(tables, list):
                tables = [tables]
            
            for table in tables:
                table_id = table.get('id')
                markdown = table.get('markdown', '')
                caption = table.get('caption', '')
                
                if table_id:
                    # Combine caption and markdown
                    full_text = f"{caption}\n\n{markdown}" if caption else markdown
                    if full_text.strip():
                        table_texts[table_id] = full_text.strip()
                        
        except Exception as e:
            print(f'⚠️  Error reading {json_file}: {e}')
            continue
    
    print(f'✅ Loaded {len(table_texts)} tables from files')
    return table_texts


def get_embeddings_to_update(conn) -> List[Tuple[str, str, str]]:
    """Get list of (chunk_id, artefact_type, metadata) that need text."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT chunk_id, artefact_type, metadata
        FROM docintel.embeddings
        WHERE artefact_type IN ('chunk', 'table', 'figure_caption')
          AND (chunk_text IS NULL OR chunk_text = '')
    """)
    
    results = cur.fetchall()
    print(f'📊 Found {len(results)} embeddings needing text')
    return results


def update_chunk_text(conn, chunk_id: str, text: str) -> bool:
    """Update chunk_text for a single embedding."""
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE docintel.embeddings
            SET chunk_text = %s
            WHERE chunk_id = %s
        """, (text, chunk_id))
        return True
    except Exception as e:
        print(f'❌ Error updating {chunk_id}: {e}')
        return False


def main():
    dsn = os.getenv('DOCINTEL_VECTOR_DB_DSN')
    if not dsn:
        print('❌ DOCINTEL_VECTOR_DB_DSN not set')
        sys.exit(1)
    
    print('=' * 80)
    print('CHUNK TEXT MIGRATION TO DATABASE')
    print('=' * 80)
    
    # Load text from files
    chunks_dir = Path('data/processing/chunks')
    tables_dir = Path('data/processing/tables')
    
    if not chunks_dir.exists():
        print(f'❌ Chunks directory not found: {chunks_dir}')
        sys.exit(1)
    
    if not tables_dir.exists():
        print(f'⚠️  Tables directory not found: {tables_dir}')
        table_texts = {}
    else:
        table_texts = load_tables_from_files(tables_dir)
    
    chunk_texts = load_chunks_from_files(chunks_dir)
    
    # Combine into single lookup
    all_texts = {**chunk_texts, **table_texts}
    print(f'📦 Total text items: {len(all_texts)}')
    
    # Connect to database
    conn = psycopg.connect(dsn)
    
    # Get embeddings that need text
    embeddings = get_embeddings_to_update(conn)
    
    if not embeddings:
        print('✅ No embeddings need updating')
        conn.close()
        return
    
    # Update embeddings
    print(f'\n🔄 Starting migration...')
    updated = 0
    skipped = 0
    errors = 0
    
    for i, (chunk_id, artefact_type, metadata) in enumerate(embeddings, 1):
        if i % 50 == 0:
            print(f'  Progress: {i}/{len(embeddings)} ({i*100//len(embeddings)}%)')
        
        # Try direct chunk_id match first
        text = all_texts.get(chunk_id)
        
        # If no match, try extracting from metadata
        if not text and metadata:
            # For tables, try matching by figure_id or other IDs
            if artefact_type == 'table':
                table_id = metadata.get('table_id') or metadata.get('id')
                if table_id:
                    text = all_texts.get(table_id)
        
        if text:
            if update_chunk_text(conn, chunk_id, text):
                updated += 1
            else:
                errors += 1
        else:
            skipped += 1
            if skipped <= 5:  # Only show first few
                print(f'⚠️  No text found for {chunk_id} ({artefact_type})')
    
    # Commit changes
    try:
        conn.commit()
        print(f'\n✅ Migration complete!')
        print(f'   Updated: {updated}')
        print(f'   Skipped: {skipped}')
        print(f'   Errors: {errors}')
    except Exception as e:
        print(f'\n❌ Error committing: {e}')
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()
    
    # Verify results
    conn = psycopg.connect(dsn)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(chunk_text) as with_text,
            SUM(LENGTH(chunk_text)) as total_chars
        FROM docintel.embeddings
        WHERE artefact_type IN ('chunk', 'table', 'figure_caption')
    """)
    
    total, with_text, total_chars = cur.fetchone()
    print(f'\n📊 Final Statistics:')
    print(f'   Total embeddings: {total}')
    print(f'   With text: {with_text} ({with_text*100//total}%)')
    print(f'   Total characters: {total_chars:,}')
    print(f'   Average chars per chunk: {total_chars // with_text if with_text else 0:,}')
    
    conn.close()
    
    print('\n' + '=' * 80)
    print('Migration complete! Text is now stored in database.')
    print('=' * 80)


if __name__ == '__main__':
    main()
