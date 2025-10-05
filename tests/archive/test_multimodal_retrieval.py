#!/usr/bin/env python3
"""
Test multimodal retrieval across text, tables, and images.
Validates that semantic search works for all content types.
"""

import asyncio
import psycopg
from pgvector.psycopg import register_vector
from pathlib import Path
import json

# BiomedCLIP client for generating query embeddings
from src.docintel.embeddings.client import EmbeddingClient
from src.docintel.config import get_embedding_settings

class MultimodalRetrieval:
    """Test semantic search across text, tables, and figure images."""
    
    def __init__(self):
        self.conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
        register_vector(self.conn)
        self.cur = self.conn.cursor()
        
        # Load embedding client
        self.embedding_settings = get_embedding_settings()
        self.client = EmbeddingClient(self.embedding_settings)
    
    async def search(self, query_text: str, limit: int = 5, artefact_filter=None):
        """
        Semantic search using cosine similarity.
        
        Args:
            query_text: Natural language query
            limit: Number of results to return
            artefact_filter: Optional list of artefact types to filter (e.g., ['figure_image', 'table'])
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 QUERY: {query_text}")
        print(f"{'=' * 80}")
        
        # Generate embedding for query
        print(f"⏳ Generating query embedding...")
        embeddings = await self.client.embed_texts([query_text])
        query_embedding = embeddings[0].embedding
        
        # Build SQL query
        filter_clause = ""
        if artefact_filter:
            placeholders = ', '.join(['%s'] * len(artefact_filter))
            filter_clause = f"AND artefact_type IN ({placeholders})"
        
        sql = f"""
            SELECT 
                chunk_id,
                artefact_type,
                nct_id,
                document_name,
                metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM docintel.embeddings
            WHERE 1=1 {filter_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        params = [query_embedding]
        if artefact_filter:
            params.extend(artefact_filter)
        params.extend([query_embedding, limit])
        
        self.cur.execute(sql, params)
        results = self.cur.fetchall()
        
        print(f"\n✅ Found {len(results)} results:\n")
        
        for i, (chunk_id, artefact_type, nct_id, doc_name, metadata, similarity) in enumerate(results, 1):
            print(f"  {i}. [{artefact_type:15}] Similarity: {similarity:.4f}")
            print(f"     NCT: {nct_id}")
            print(f"     Chunk: {chunk_id[:60]}")
            
            # Show relevant metadata based on type
            if artefact_type == 'figure_image':
                image_path = metadata.get('image_path', 'N/A')
                page = metadata.get('page_reference', 'N/A')
                print(f"     Image: {image_path}")
                print(f"     Page: {page}")
                
                # Check if image file exists
                if image_path != 'N/A':
                    full_path = Path('data/processing') / image_path
                    exists = "✅ EXISTS" if full_path.exists() else "❌ MISSING"
                    print(f"     File: {exists}")
            
            elif artefact_type == 'table':
                table_id = metadata.get('table_id', 'N/A')
                print(f"     Table ID: {table_id}")
            
            elif artefact_type == 'chunk':
                token_count = metadata.get('token_count', 'N/A')
                section = metadata.get('section', 'N/A')
                print(f"     Tokens: {token_count}, Section: {section}")
            
            print()
        
        return results
    
    async def test_text_search(self):
        """Test 1: Text chunk retrieval"""
        print("\n" + "=" * 80)
        print("TEST 1: TEXT SEARCH")
        print("=" * 80)
        
        results = await self.search(
            "pembrolizumab adverse events side effects",
            limit=5,
            artefact_filter=['chunk']
        )
        
        if results:
            print("✅ TEXT SEARCH WORKS")
            
            # Try to find related entities
            chunk_id = results[0][0]
            nct_id = results[0][2]
            
            print(f"\n📊 Checking for entities in top result...")
            self.cur.execute("""
                SELECT entity_type, entity_text, normalized_id, normalized_source
                FROM docintel.entities
                WHERE chunk_id::text LIKE %s
                LIMIT 10;
            """, (f"%{nct_id}%",))
            
            entities = self.cur.fetchall()
            if entities:
                print(f"   Found {len(entities)} entities in this trial:")
                for etype, etext, norm_id, norm_src in entities[:5]:
                    print(f"      • {etype:20}: {etext[:40]:40} | {norm_src or 'not normalized'}")
            else:
                print("   ⚠️  No entities found (checking by NCT pattern)")
        else:
            print("❌ TEXT SEARCH FAILED")
    
    async def test_table_search(self):
        """Test 2: Table retrieval"""
        print("\n" + "=" * 80)
        print("TEST 2: TABLE SEARCH")
        print("=" * 80)
        
        results = await self.search(
            "adverse events grade 3 grade 4 toxicity rates",
            limit=5,
            artefact_filter=['table']
        )
        
        if results:
            print("✅ TABLE SEARCH WORKS")
            
            # Check for table_data entities
            nct_id = results[0][2]
            
            print(f"\n📊 Checking for table_data entities...")
            self.cur.execute("""
                SELECT entity_text, COUNT(*) as count
                FROM docintel.entities
                WHERE entity_type = 'table_data'
                  AND chunk_id::text LIKE %s
                GROUP BY entity_text
                LIMIT 5;
            """, (f"%{nct_id}%",))
            
            table_data = self.cur.fetchall()
            if table_data:
                print(f"   Found {len(table_data)} table_data entries for this trial:")
                for text, count in table_data:
                    print(f"      • {text[:60]:60} ({count}x)")
            else:
                print("   ⚠️  No table_data entities found")
        else:
            print("❌ TABLE SEARCH FAILED")
    
    async def test_image_search(self):
        """Test 3: Figure image retrieval"""
        print("\n" + "=" * 80)
        print("TEST 3: IMAGE SEARCH (FIGURES & CHARTS)")
        print("=" * 80)
        
        results = await self.search(
            "Kaplan-Meier survival curve progression free survival overall survival",
            limit=5,
            artefact_filter=['figure_image']
        )
        
        if results:
            print("✅ IMAGE SEARCH WORKS")
            
            # Verify we can actually load the images
            print(f"\n🖼️  Verifying image files...")
            for chunk_id, artefact_type, nct_id, doc_name, metadata, similarity in results[:3]:
                image_path = metadata.get('image_path')
                if image_path:
                    full_path = Path('data/processing') / image_path
                    if full_path.exists():
                        file_size = full_path.stat().st_size / 1024  # KB
                        print(f"   ✅ {image_path}: {file_size:.1f} KB")
                    else:
                        print(f"   ❌ {image_path}: FILE NOT FOUND")
        else:
            print("❌ IMAGE SEARCH FAILED - No figure_image embeddings found")
    
    async def test_mixed_search(self):
        """Test 4: Search across all types"""
        print("\n" + "=" * 80)
        print("TEST 4: MIXED SEARCH (ALL TYPES)")
        print("=" * 80)
        
        results = await self.search(
            "clinical efficacy outcomes response rate",
            limit=10,
            artefact_filter=None  # Search all types
        )
        
        if results:
            # Count by type
            type_counts = {}
            for _, artefact_type, _, _, _, _ in results:
                type_counts[artefact_type] = type_counts.get(artefact_type, 0) + 1
            
            print("✅ MIXED SEARCH WORKS")
            print(f"\n📊 Result distribution:")
            for atype, count in sorted(type_counts.items()):
                print(f"   {atype:20}: {count} results")
        else:
            print("❌ MIXED SEARCH FAILED")
    
    async def test_entity_retrieval(self):
        """Test 5: Can we go from embedding → entities?"""
        print("\n" + "=" * 80)
        print("TEST 5: EMBEDDING → ENTITY LINKAGE")
        print("=" * 80)
        
        # Find a text embedding
        results = await self.search("pembrolizumab treatment", limit=1, artefact_filter=['chunk'])
        
        if not results:
            print("❌ No text chunks found")
            return
        
        chunk_id, _, nct_id, _, _, _ = results[0]
        
        print(f"\n🔗 Testing linkage for: {chunk_id}")
        print(f"   NCT ID: {nct_id}")
        
        # Try multiple strategies to find entities
        strategies = [
            ("Exact chunk_id match", "chunk_id::text = %s", [chunk_id]),
            ("NCT pattern match", "chunk_id::text LIKE %s", [f"%{nct_id}%"]),
            ("Same trial (any entity)", "chunk_id::text LIKE %s", [f"{nct_id}%"]),
        ]
        
        found_any = False
        for strategy_name, where_clause, params in strategies:
            self.cur.execute(f"""
                SELECT entity_type, entity_text, normalized_id, normalized_source
                FROM docintel.entities
                WHERE {where_clause}
                LIMIT 5;
            """, params)
            
            entities = self.cur.fetchall()
            if entities:
                print(f"\n   ✅ Strategy '{strategy_name}' found {len(entities)} entities:")
                for etype, etext, norm_id, norm_src in entities[:3]:
                    status = "✅ normalized" if norm_id else "⚠️  not normalized"
                    print(f"      • {etype:20}: {etext[:40]:40} | {status}")
                found_any = True
                break
        
        if not found_any:
            print("\n   ❌ No entities found with any strategy")
    
    async def run_all_tests(self):
        """Run complete validation suite"""
        print("\n" + "=" * 80)
        print("🚀 MULTIMODAL RETRIEVAL VALIDATION")
        print("=" * 80)
        print(f"\nTesting semantic search across:")
        print(f"  • Text chunks (3,214 embeddings)")
        print(f"  • Tables (284 embeddings)")
        print(f"  • Figure images (212 embeddings)")
        print(f"  • Knowledge graph (161,902 entities, 19,360 normalized)")
        
        await self.test_text_search()
        await self.test_table_search()
        await self.test_image_search()
        await self.test_mixed_search()
        await self.test_entity_retrieval()
        
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        print("""
✅ If all tests passed, you're ready for:
   1. Vision LLM integration (GPT-4o/GPT-4V)
   2. U-Retrieval hierarchical query layer
   3. Full GraphRAG application

⚠️  If any tests failed:
   - Text/Table failures: Check embeddings generation
   - Image failures: Verify figure PNG files exist
   - Entity failures: Check chunk_id linking strategy
        """)
    
    def close(self):
        self.conn.close()


async def main():
    """Run validation tests"""
    retrieval = MultimodalRetrieval()
    
    try:
        await retrieval.run_all_tests()
    finally:
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
