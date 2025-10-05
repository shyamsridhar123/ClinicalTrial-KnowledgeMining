#!/usr/bin/env python3
"""
Clinical Trial Question Answering System
Uses GraphRAG (semantic search + entities + LLM) to answer clinical questions.

Usage:
    pixi run -- python query_clinical_trials.py "What are adverse events for pembrolizumab?"
"""

import sys
import asyncio
import json
from pathlib import Path
sys.path.insert(0, 'src')

import psycopg
from psycopg.rows import dict_row
from docintel.embeddings.client import EmbeddingClient
from docintel.config import EmbeddingSettings
from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope
from docintel.query import QueryRewriter
from docintel.query.model_router import get_router, ModelChoice
from docintel.query.llm_client import get_llm_client


class ClinicalTrialQA:
    """Question answering system over clinical trial knowledge graph."""
    
    def __init__(self):
        self.db_dsn = "postgresql://dbuser:dbpass123@localhost:5432/docintel"
        self.embedding_client = None
        self.llm_client = None
        self.router = None
        self.query_rewriter = QueryRewriter(enable_rewriting=True)
        
    async def initialize(self):
        """Initialize embedding, router, and LLM clients."""
        print("🔧 Initializing system...")
        
        # Load embedding client
        embedding_settings = EmbeddingSettings()
        self.embedding_client = EmbeddingClient(embedding_settings)
        print("  ✅ BiomedCLIP embedding model loaded")
        
        # Load model router and LLM client
        # FEATURE FLAG: disable_routing=True to always use GPT-4.1 (fixes adverse events extraction)
        self.router = get_router(disable_routing=True)
        self.llm_client = get_llm_client()
        print("  ✅ Model router initialized (GPT-4.1 only - routing disabled)")
        print()
    
    async def retrieve_context(self, query: str, top_k: int = 50, use_graph_expansion: bool = True) -> dict:
        """
        Retrieve relevant context using U-Retrieval (hierarchical graph-aware search).
        
        Args:
            query: Clinical question
            top_k: Maximum entities to retrieve (default 50 for graph expansion)
            use_graph_expansion: Enable multi-hop graph traversal (default True)
        
        Returns:
            {
                'chunks': [chunk_data with text, metadata, entities],
                'entities': [SearchResult objects from U-Retrieval],
                'total_entities': int,
                'unique_ncts': [str],
                'graph_expanded_count': int,
                'processing_time_ms': float,
                'original_query': str,
                'rewritten_query': str (if rewritten)
            }
        """
        # Apply query rewriting for better semantic search
        original_query = query
        rewritten_query = self.query_rewriter.rewrite(query)
        
        # Show rewriting explanation if query was changed
        explanation = self.query_rewriter.explain_rewrite(original_query, rewritten_query)
        if explanation:
            print(explanation)
            print()
        
        print(f"🔍 Query: {rewritten_query if rewritten_query != original_query else query}")
        print(f"📊 Using U-Retrieval with graph expansion (max_results={top_k})...\n")
        
        # Use U-Retrieval for hierarchical entity search with graph expansion
        # Pass embedding_client to enable semantic vector search fallback
        u_retrieval = ClinicalURetrieval(self.db_dsn, embedding_client=self.embedding_client)
        
        query_type = QueryType.HYBRID_SEARCH if use_graph_expansion else QueryType.ENTITY_SEARCH
        
        # Use rewritten query for actual search
        u_result = await u_retrieval.u_retrieval_search(
            query=rewritten_query,
            query_type=query_type,
            search_scope=SearchScope.GLOBAL,
            max_results=top_k
        )
        
        await u_retrieval.close()
        
        # Extract entities and their source chunks
        entities = u_result.results
        graph_expanded_count = sum(1 for e in entities if e.metadata.get('relation_type') == 'graph_expansion')
        
        print(f"  ✅ Found {len(entities)} entities ({graph_expanded_count} via graph expansion)")
        print(f"  ⏱️  Processing time: {u_result.processing_time_ms:.1f}ms\n")
        
        # Group entities by their source chunks to build context
        chunk_entities_map = {}
        for entity in entities:
            # Use source_chunk_id from metadata (e.g., "NCT02467621-chunk-0051")
            chunk_id = entity.metadata.get('source_chunk_id', '')
            if chunk_id:
                if chunk_id not in chunk_entities_map:
                    chunk_entities_map[chunk_id] = []
                chunk_entities_map[chunk_id].append(entity)
        
        # Retrieve chunk metadata and text for top chunks (by entity count)
        top_chunks = sorted(chunk_entities_map.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        conn = psycopg.connect(self.db_dsn)
        chunks = []
        
        with conn.cursor(row_factory=dict_row) as cur:
            for i, (chunk_id, chunk_entities) in enumerate(top_chunks, 1):
                # Get chunk metadata and text from embeddings table
                cur.execute("""
                    SELECT 
                        chunk_id,
                        nct_id,
                        document_name,
                        artefact_type,
                        section,
                        token_count,
                        source_path,
                        chunk_text
                    FROM docintel.embeddings
                    WHERE chunk_id = %s
                    LIMIT 1
                """, (chunk_id,))
                
                chunk_meta = cur.fetchone()
                if not chunk_meta:
                    continue
                
                chunk_data = dict(chunk_meta)
                chunk_data['similarity'] = chunk_entities[0].relevance_score  # Use top entity's score
                
                print(f"  [{i}] NCT: {chunk_meta['nct_id']} | Type: {chunk_meta['artefact_type']} | "
                      f"Entities: {len(chunk_entities)} | Relevance: {chunk_data['similarity']:.3f}")
                
                # Attach entities to chunk WITH context flags
                chunk_data['entities'] = [
                    {
                        'entity_text': e.entity_text,
                        'entity_type': e.entity_type,
                        'normalized_id': e.normalized_concept_id,
                        'normalized_source': e.normalized_vocabulary,
                        'confidence': e.confidence,
                        'relevance_score': e.relevance_score,
                        'explanation': e.explanation,
                        'graph_expanded': e.metadata.get('relation_type') == 'graph_expansion',
                        'hop_distance': e.metadata.get('hop_distance'),
                        'context_flags': e.metadata.get('context_flags', {})  # Clinical context
                    }
                    for e in chunk_entities
                ]
                
                # Get chunk text directly from database (no file I/O!)
                chunk_data['text'] = chunk_meta.get('chunk_text', '')
                
                chunks.append(chunk_data)
                
                # Show graph-expanded entities
                graph_expanded_entities = [e for e in chunk_entities if e.metadata.get('relation_type') == 'graph_expansion']
                if graph_expanded_entities:
                    print(f"      └─ {len(graph_expanded_entities)} graph-expanded entities")
        
        conn.close()
        
        unique_ncts = list(set(c['nct_id'] for c in chunks if 'nct_id' in c))
        
        print(f"\n✅ Retrieved {len(chunks)} chunks from {len(unique_ncts)} NCTs")
        print(f"   Total entities: {len(entities)} ({graph_expanded_count} expanded via graph)\n")
        
        return {
            'chunks': chunks,
            'entities': entities,
            'total_entities': len(entities),
            'unique_ncts': unique_ncts,
            'graph_expanded_count': graph_expanded_count,
            'processing_time_ms': u_result.processing_time_ms,
            'original_query': original_query,
            'rewritten_query': rewritten_query if rewritten_query != original_query else None
        }
    
    def _load_chunk_text(self, source_path: str, chunk_id: str) -> str:
        """Load actual chunk text from storage."""
        if not source_path:
            return ""
        
        # Extract NCT and document name from source_path: "chunks/NCT02467621/Prot_SAP_000.json"
        try:
            parts = source_path.split('/')
            if len(parts) >= 3:
                nct_id = parts[1]
                doc_name = parts[2].replace('.json', '.txt')
                
                # Try to load full document text
                text_file = Path(f"data/processing/text/{nct_id}/{doc_name}")
                if text_file.exists():
                    full_text = text_file.read_text(encoding='utf-8')
                    # Return truncated version (in production, extract specific chunk region)
                    return full_text[:3000] if len(full_text) > 3000 else full_text
        except Exception:
            pass
        
        return ""
    
    async def answer_question(self, query: str, top_k: int = 50) -> dict:
        """
        Answer a clinical question using U-Retrieval GraphRAG pipeline.
        
        Args:
            query: Clinical question
            top_k: Maximum entities to retrieve (default 50 for graph expansion)
        
        Returns:
            {
                'question': str,
                'answer': str,
                'sources': [{'nct_id': str, 'similarity': float}],
                'entities_found': int,
                'graph_expanded_count': int,
                'ncts_searched': [str],
                'processing_time_ms': float
            }
        """
        # Step 1: Retrieve relevant context with graph expansion
        context = await self.retrieve_context(query, top_k=top_k, use_graph_expansion=True)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(query, context)
        
        # Step 3: Route query to appropriate model
        routing_decision = self.router.route(
            query_text=query,
            images=None,  # No image input for this query type
            context_docs=[c['text'] for c in context['chunks']]
        )
        
        model_name = "GPT-5-mini" if routing_decision.model == ModelChoice.GPT_5_MINI else "GPT-4.1"
        print(f"🤖 Generating answer with {model_name}...")
        print(f"   Routing reason: {routing_decision.reason}\n")
        
        # Step 4: Call LLM with routed model
        system_message = """You are a clinical research assistant analyzing clinical trial data.
Answer questions accurately based on the provided context.
Always cite specific NCT IDs when making claims.
If information is not in the context, say so clearly.

IMPORTANT - Clinical Context Flags:
- ❌NEGATED: Absence/denial (e.g., "no evidence of toxicity"). DO NOT report as observed event.
- 📅HISTORICAL: Past medical history, not current study finding.
- 🤔HYPOTHETICAL: Protocol instructions ("if X occurs"), not observed events.
- ❓UNCERTAIN: Possible/suspected findings, not confirmed.
- 👨‍👩‍👧FAMILY: Family medical history, not about the patient.

When entities are marked with these flags, clearly indicate the context in your answer."""
        
        # GPT-5-mini needs more tokens because reasoning tokens count toward max_tokens
        # For reasoning models: max_tokens = reasoning_tokens + output_tokens
        max_tokens = 8000 if routing_decision.model == ModelChoice.GPT_5_MINI else 1000
        
        response = self.llm_client.query(
            model=routing_decision.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        answer = response.content
        
        # Step 5: Build response
        sources = [
            {
                'nct_id': c['nct_id'],
                'document': c['document_name'],
                'similarity': round(c['similarity'], 3),
                'entities': len(c['entities'])
            }
            for c in context['chunks']
        ]
        
        return {
            'question': query,
            'answer': answer,
            'sources': sources,
            'entities_found': context['total_entities'],
            'graph_expanded_count': context.get('graph_expanded_count', 0),
            'ncts_searched': context['unique_ncts'],
            'processing_time_ms': context.get('processing_time_ms', 0),
            'model_used': routing_decision.model.value,
            'routing_reason': routing_decision.reason
        }
    
    def _build_prompt(self, query: str, context: dict) -> str:
        """Build LLM prompt with retrieved context and graph expansion info."""
        prompt_parts = [
            f"Question: {query}",
            "",
            "Context from clinical trials (with graph-expanded entities):",
            ""
        ]
        
        # Add graph expansion summary if applicable
        if context.get('graph_expanded_count', 0) > 0:
            prompt_parts.append(
                f"ℹ️  {context['graph_expanded_count']} additional entities retrieved via knowledge graph traversal\n"
            )
        
        for i, chunk in enumerate(context['chunks'], 1):
            prompt_parts.append(f"--- Source {i}: NCT {chunk['nct_id']} ---")
            
            if chunk['text']:
                # Truncate long text
                text = chunk['text'][:1500]
                prompt_parts.append(f"Text: {text}")
            
            if chunk['entities']:
                # Separate direct matches from graph-expanded entities
                direct_entities = [e for e in chunk['entities'] if not e.get('graph_expanded')]
                graph_entities = [e for e in chunk['entities'] if e.get('graph_expanded')]
                
                # Helper to format entity with context flags
                def format_entity(e):
                    name = f"{e['entity_text']} ({e['entity_type']})"
                    flags = e.get('context_flags', {})
                    annotations = []
                    
                    if flags.get('is_negated'):
                        annotations.append('❌NEGATED-no evidence')
                    if flags.get('is_historical'):
                        annotations.append('📅HISTORICAL-past event')
                    if flags.get('is_hypothetical'):
                        annotations.append('🤔HYPOTHETICAL-if scenario')
                    if flags.get('is_uncertain'):
                        annotations.append('❓UNCERTAIN-possible')
                    if flags.get('is_family'):
                        annotations.append('👨‍👩‍👧FAMILY-not patient')
                    
                    if annotations:
                        return f"{name} [{', '.join(annotations)}]"
                    return name
                
                # Add direct match entities with context
                if direct_entities:
                    entities_str = ", ".join(
                        format_entity(e)
                        for e in direct_entities[:10]
                    )
                    prompt_parts.append(f"Key entities: {entities_str}")
                
                # Add graph-expanded entities with hop information
                if graph_entities:
                    graph_str = ", ".join(
                        f"{format_entity(e)}, {e.get('hop_distance', '?')}-hop"
                        for e in graph_entities[:5]
                    )
                    prompt_parts.append(f"Related entities (via graph): {graph_str}")
            
            prompt_parts.append("")
        
        prompt_parts.append(
            "Based on the above context, answer the question. "
            "Cite NCT IDs when possible. "
            "Note: Some entities were found via knowledge graph relations to enhance context."
        )
        
        return "\n".join(prompt_parts)
    
    def print_result(self, result: dict):
        """Pretty print the QA result with graph expansion info."""
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(f"\n{result['answer']}\n")
        
        print("=" * 80)
        print("SOURCES")
        print("=" * 80)
        for src in result['sources']:
            print(f"  • NCT {src['nct_id']} (similarity: {src['similarity']}, {src['entities']} entities)")
            print(f"    {src['document']}")
        
        print(f"\n📊 Searched {len(result['ncts_searched'])} NCTs")
        print(f"   Found {result['entities_found']} entities ({result.get('graph_expanded_count', 0)} via graph expansion)")
        print(f"   🤖 Model: {result.get('model_used', 'unknown')} ({result.get('routing_reason', 'unknown')})")
        print(f"   ⏱️  Processing time: {result.get('processing_time_ms', 0):.1f}ms")
        print("=" * 80)


async def main():
    """Run interactive or command-line query."""
    if len(sys.argv) < 2:
        print("Usage: pixi run -- python query_clinical_trials.py \"Your question here\"")
        print("\nExample questions:")
        print('  - "What are the most common adverse events?"')
        print('  - "What are the inclusion criteria for patients?"')
        print('  - "What is the primary endpoint?"')
        print('  - "What statistical methods were used?"')
        return
    
    query = " ".join(sys.argv[1:])
    
    # Initialize QA system
    qa_system = ClinicalTrialQA()
    await qa_system.initialize()
    
    # Answer question (use default top_k=50 for graph expansion)
    result = await qa_system.answer_question(query)
    
    # Display result
    qa_system.print_result(result)
    
    # Optionally save to file
    output_file = Path("query_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full result saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
