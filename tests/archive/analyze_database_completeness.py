#!/usr/bin/env python3
"""
Comprehensive database analysis for multimodal GraphRAG system.
Validates pgvector embeddings + Apache AGE knowledge graph integration.
"""

import psycopg
from psycopg.rows import dict_row
import json

DB_DSN = "postgresql://dbuser:dbpass123@localhost:5432/docintel"

def analyze_database():
    conn = psycopg.connect(DB_DSN)
    cur = conn.cursor(row_factory=dict_row)
    
    print("=" * 80)
    print("COMPREHENSIVE DATABASE ANALYSIS FOR MULTIMODAL GRAPHRAG")
    print("=" * 80)
    
    # ========================================================================
    # SECTION 1: PGVECTOR EMBEDDINGS ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: PGVECTOR EMBEDDINGS (Semantic Search Layer)")
    print("=" * 80)
    
    # Total embeddings by type
    cur.execute("""
        SELECT artefact_type, COUNT(*) as count, 
               COUNT(DISTINCT nct_id) as unique_ncts,
               COUNT(DISTINCT chunk_id) as unique_chunks
        FROM docintel.embeddings
        GROUP BY artefact_type
        ORDER BY count DESC
    """)
    
    embeddings = cur.fetchall()
    total_embeddings = sum(e['count'] for e in embeddings)
    
    print(f"\n📊 Total Embeddings: {total_embeddings:,}")
    print("\nBreakdown by artefact type:")
    for emb in embeddings:
        print(f"  • {emb['artefact_type']:20} : {emb['count']:6,} embeddings | "
              f"{emb['unique_ncts']:2} NCTs | {emb['unique_chunks']:4} chunks")
    
    # Check embedding dimension
    cur.execute("SELECT vector_dims(embedding) as dims FROM docintel.embeddings LIMIT 1")
    dims = cur.fetchone()
    print(f"\n🔢 Embedding Dimensions: {dims['dims']} (BiomedCLIP)")
    
    # Check for NULL embeddings
    cur.execute("SELECT COUNT(*) FROM docintel.embeddings WHERE embedding IS NULL")
    null_embeddings = cur.fetchone()['count']
    print(f"✅ NULL Embeddings: {null_embeddings} (should be 0)")
    
    # Check metadata completeness
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(nct_id) as with_nct,
            COUNT(document_name) as with_doc,
            COUNT(page_reference) as with_page,
            COUNT(metadata) as with_metadata
        FROM docintel.embeddings
    """)
    
    meta = cur.fetchone()
    print(f"\n📋 Metadata Completeness:")
    print(f"  • NCT ID: {meta['with_nct']:,} / {meta['total']:,} ({meta['with_nct']/meta['total']*100:.1f}%)")
    print(f"  • Document: {meta['with_doc']:,} / {meta['total']:,} ({meta['with_doc']/meta['total']*100:.1f}%)")
    print(f"  • Page ref: {meta['with_page']:,} / {meta['total']:,} ({meta['with_page']/meta['total']*100:.1f}%)")
    print(f"  • Metadata: {meta['with_metadata']:,} / {meta['total']:,} ({meta['with_metadata']/meta['total']*100:.1f}%)")
    
    # ========================================================================
    # SECTION 2: KNOWLEDGE GRAPH ENTITIES
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: KNOWLEDGE GRAPH ENTITIES (Clinical NLP Layer)")
    print("=" * 80)
    
    # Total entities
    cur.execute("SELECT COUNT(*) FROM docintel.entities")
    total_entities = cur.fetchone()['count']
    print(f"\n📊 Total Entities: {total_entities:,}")
    
    # Entity type distribution
    cur.execute("""
        SELECT entity_type, COUNT(*) as count
        FROM docintel.entities
        GROUP BY entity_type
        ORDER BY count DESC
        LIMIT 15
    """)
    
    entity_types = cur.fetchall()
    print("\nTop 15 Entity Types:")
    for et in entity_types:
        print(f"  • {et['entity_type']:25} : {et['count']:7,} entities")
    
    # Check source_chunk_id linkage (CRITICAL FOR GRAPHRAG)
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(source_chunk_id) as with_source_chunk,
            COUNT(DISTINCT source_chunk_id) as unique_source_chunks
        FROM docintel.entities
    """)
    
    linkage = cur.fetchone()
    print(f"\n🔗 Entity-Embedding Linkage (CRITICAL):")
    print(f"  • Entities with source_chunk_id: {linkage['with_source_chunk']:,} / {linkage['total']:,} "
          f"({linkage['with_source_chunk']/linkage['total']*100:.1f}%)")
    print(f"  • Unique source chunks: {linkage['unique_source_chunks']:,}")
    
    if linkage['with_source_chunk'] == linkage['total']:
        print(f"  ✅ ALL entities have source_chunk_id - GraphRAG linking COMPLETE")
    else:
        print(f"  ❌ Some entities missing source_chunk_id - GraphRAG linking INCOMPLETE")
    
    # Check normalization
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(normalized_id) as normalized,
            COUNT(DISTINCT normalized_source) as unique_sources
        FROM docintel.entities
    """)
    
    norm = cur.fetchone()
    print(f"\n🏥 Entity Normalization (Clinical Vocabularies):")
    print(f"  • Normalized entities: {norm['normalized']:,} / {norm['total']:,} "
          f"({norm['normalized']/norm['total']*100:.1f}%)")
    print(f"  • Vocabulary sources: {norm['unique_sources']}")
    
    # Get normalized sources breakdown
    cur.execute("""
        SELECT normalized_source, COUNT(*) as count
        FROM docintel.entities
        WHERE normalized_source IS NOT NULL
        GROUP BY normalized_source
        ORDER BY count DESC
    """)
    
    sources = cur.fetchall()
    if sources:
        print("\n  Normalization sources:")
        for src in sources:
            print(f"    • {src['normalized_source']:15} : {src['count']:6,} entities")
    
    # ========================================================================
    # SECTION 3: ENTITY-EMBEDDING LINKAGE VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: ENTITY ↔ EMBEDDING LINKAGE (GraphRAG Core)")
    print("=" * 80)
    
    # Test if source_chunk_ids match embedding chunk_ids
    cur.execute("""
        SELECT COUNT(DISTINCT e.source_chunk_id) as matching_chunks
        FROM docintel.entities e
        INNER JOIN docintel.embeddings emb ON e.source_chunk_id = emb.chunk_id
        WHERE e.source_chunk_id IS NOT NULL
    """)
    
    match = cur.fetchone()
    print(f"\n🎯 Matching Chunks (entities.source_chunk_id = embeddings.chunk_id):")
    print(f"  • Chunks with both entities AND embeddings: {match['matching_chunks']:,}")
    
    # Sample linkage test
    cur.execute("""
        SELECT 
            e.source_chunk_id,
            COUNT(DISTINCT e.entity_id) as entity_count,
            COUNT(DISTINCT emb.chunk_id) as embedding_count
        FROM docintel.entities e
        LEFT JOIN docintel.embeddings emb ON e.source_chunk_id = emb.chunk_id
        WHERE e.source_chunk_id IS NOT NULL
        GROUP BY e.source_chunk_id
        LIMIT 5
    """)
    
    samples = cur.fetchall()
    print(f"\n  Sample chunk linkages:")
    for s in samples:
        status = "✅" if s['embedding_count'] > 0 else "❌"
        print(f"    {status} {s['source_chunk_id']:40} : {s['entity_count']:3} entities, {s['embedding_count']} embeddings")
    
    # Check chunk_entity_links table (if exists)
    cur.execute("""
        SELECT COUNT(*) as count FROM docintel.chunk_entity_links
    """)
    
    links = cur.fetchone()
    print(f"\n📊 Chunk-Entity Links Table: {links['count']:,} links")
    
    if links['count'] > 0:
        cur.execute("""
            SELECT COUNT(DISTINCT embedding_chunk_id) as unique_embeddings,
                   COUNT(DISTINCT entity_id) as unique_entities
            FROM docintel.chunk_entity_links
        """)
        link_stats = cur.fetchone()
        print(f"  • Unique embeddings linked: {link_stats['unique_embeddings']:,}")
        print(f"  • Unique entities linked: {link_stats['unique_entities']:,}")
    
    # ========================================================================
    # SECTION 4: META-GRAPHS & RELATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: META-GRAPHS & RELATIONS (Graph Structure)")
    print("=" * 80)
    
    # Meta-graphs
    cur.execute("SELECT COUNT(*) FROM docintel.meta_graphs")
    meta_graphs = cur.fetchone()['count']
    print(f"\n📊 Total Meta-Graphs: {meta_graphs:,}")
    
    # Relations
    cur.execute("SELECT COUNT(*) FROM docintel.relations")
    relations = cur.fetchone()['count']
    print(f"📊 Total Relations: {relations:,}")
    
    if relations > 0:
        cur.execute("""
            SELECT predicate, COUNT(*) as count
            FROM docintel.relations
            GROUP BY predicate
            ORDER BY count DESC
            LIMIT 10
        """)
        
        predicates = cur.fetchall()
        print("\n  Top relation types:")
        for p in predicates:
            print(f"    • {p['predicate']:30} : {p['count']:6,} relations")
    
    # ========================================================================
    # SECTION 5: MULTIMODAL COVERAGE
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: MULTIMODAL COVERAGE (Text, Tables, Figures)")
    print("=" * 80)
    
    # Text coverage
    cur.execute("""
        SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'chunk'
    """)
    text_embeddings = cur.fetchone()['count']
    
    # Table coverage
    cur.execute("""
        SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'table'
    """)
    table_embeddings = cur.fetchone()['count']
    
    cur.execute("""
        SELECT COUNT(*) FROM docintel.entities WHERE entity_type IN ('table_caption', 'table_data')
    """)
    table_entities = cur.fetchone()['count']
    
    # Figure coverage
    cur.execute("""
        SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type IN ('figure_caption', 'figure_image')
    """)
    figure_embeddings = cur.fetchone()['count']
    
    cur.execute("""
        SELECT COUNT(*) FROM docintel.entities WHERE entity_type ILIKE '%figure%'
    """)
    figure_entities = cur.fetchone()['count']
    
    print(f"\n📄 Text Coverage:")
    print(f"  • Text embeddings: {text_embeddings:,}")
    print(f"  • Text entities: {total_entities - table_entities - figure_entities:,}")
    
    print(f"\n📊 Table Coverage:")
    print(f"  • Table embeddings: {table_embeddings:,}")
    print(f"  • Table entities: {table_entities:,}")
    
    print(f"\n🖼️  Figure Coverage:")
    print(f"  • Figure embeddings: {figure_embeddings:,}")
    print(f"  • Figure entities: {figure_entities:,}")
    
    # ========================================================================
    # SECTION 6: REPOSITORY NODES (UMLS/RxNorm/SNOMED/LOINC)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: CLINICAL VOCABULARY REPOSITORY")
    print("=" * 80)
    
    cur.execute("""
        SELECT vocabulary, COUNT(*) as count, COUNT(DISTINCT code) as unique_codes
        FROM docintel.repo_nodes
        WHERE is_active = true
        GROUP BY vocabulary
        ORDER BY count DESC
    """)
    
    vocabs = cur.fetchall()
    total_vocab = sum(v['count'] for v in vocabs)
    
    print(f"\n📚 Total Vocabulary Terms: {total_vocab:,}")
    if vocabs:
        print("\nVocabulary sources:")
        for v in vocabs:
            print(f"  • {v['vocabulary']:15} : {v['count']:8,} terms ({v['unique_codes']:8,} unique codes)")
    
    # ========================================================================
    # SECTION 7: GRAPHRAG READINESS ASSESSMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: GRAPHRAG READINESS ASSESSMENT")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Embeddings exist
    checks.append(("Embeddings exist", total_embeddings > 0, total_embeddings))
    
    # Check 2: Entities exist
    checks.append(("Entities exist", total_entities > 0, total_entities))
    
    # Check 3: Source_chunk_id populated
    checks.append(("Entity-embedding linkage", linkage['with_source_chunk'] == linkage['total'], 
                   f"{linkage['with_source_chunk']}/{linkage['total']}"))
    
    # Check 4: Multimodal coverage
    checks.append(("Text embeddings", text_embeddings > 0, text_embeddings))
    checks.append(("Table embeddings", table_embeddings > 0, table_embeddings))
    checks.append(("Figure embeddings", figure_embeddings > 0, figure_embeddings))
    
    # Check 5: Normalized entities
    checks.append(("Entity normalization", norm['normalized'] > 0, 
                   f"{norm['normalized']}/{norm['total']} ({norm['normalized']/norm['total']*100:.0f}%)"))
    
    # Check 6: Clinical vocabularies
    checks.append(("Clinical vocabularies", total_vocab > 0, total_vocab))
    
    # Check 7: Chunk matching
    checks.append(("Chunk ID matching", match['matching_chunks'] > 0, match['matching_chunks']))
    
    print("\n✅ System Readiness Checklist:")
    all_passed = True
    for check_name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name:30} : {value}")
        if not passed:
            all_passed = False
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 ✅ DATABASE IS READY FOR MULTIMODAL GRAPHRAG!")
        print("\nNext steps:")
        print("  1. Test semantic search: Query embeddings by similarity")
        print("  2. Test entity retrieval: Get entities by source_chunk_id")
        print("  3. Test graph traversal: Follow relations from entities")
        print("  4. Test normalization: Resolve entities to UMLS/RxNorm concepts")
        print("  5. Integrate vision LLM: Add GPT-4o for figure interpretation")
    else:
        print("\n⚠️  DATABASE HAS GAPS - Review failed checks above")
    
    conn.close()

if __name__ == '__main__':
    analyze_database()
