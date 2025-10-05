"""
U-Retrieval System for Clinical Knowledge Graphs

Implements hierarchical retrieval system that leverages community structure 
for more precise and context-aware query results per Medical-Graph-RAG patterns.

Features:
- Global-to-local search strategy
- Community-aware ranking
- Multi-level context aggregation
- Clinical vocabulary integration
- Semantic similarity scoring
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

import psycopg
from ..config import get_config

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of clinical queries supported"""
    ENTITY_SEARCH = "entity_search"
    RELATION_SEARCH = "relation_search"
    COMMUNITY_SEARCH = "community_search"
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID_SEARCH = "hybrid_search"


class SearchScope(Enum):
    """Search scope levels"""
    GLOBAL = "global"           # Search across all communities
    COMMUNITY = "community"     # Search within specific communities
    LOCAL = "local"            # Search within specific documents/chunks


@dataclass
class QueryContext:
    """Context information for queries"""
    clinical_area: Optional[str] = None      # e.g., "oncology", "cardiology"
    study_phase: Optional[str] = None        # e.g., "phase_1", "phase_2"
    entity_types: List[str] = None          # Filter by entity types
    vocabularies: List[str] = None          # Filter by vocabularies (UMLS, SNOMED, etc.)
    confidence_threshold: float = 0.0       # Minimum confidence threshold
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = []
        if self.vocabularies is None:
            self.vocabularies = []


@dataclass
class SearchResult:
    """Individual search result with context"""
    entity_id: str
    entity_text: str
    entity_type: str
    normalized_concept_id: Optional[str]
    normalized_vocabulary: Optional[str]
    confidence: float
    community_id: Optional[str]
    community_title: Optional[str]
    chunk_id: str
    document_context: str
    relevance_score: float
    explanation: str
    metadata: Dict[str, Any]


@dataclass
class URetrievalResult:
    """Complete U-Retrieval result with hierarchical context"""
    query: str
    query_type: QueryType
    search_scope: SearchScope
    results: List[SearchResult]
    community_aggregation: Dict[str, Any]
    global_context: Dict[str, Any]
    processing_stats: Dict[str, Any]
    total_results: int
    processing_time_ms: float


class ClinicalURetrieval:
    """
    U-Retrieval system for clinical knowledge graphs.
    
    Implements hierarchical retrieval leveraging:
    - Community structure for context-aware search
    - Normalized entities for vocabulary-aware matching
    - Multi-level aggregation for comprehensive results
    - Clinical domain expertise for relevance scoring
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        
        # Clinical relevance weights for different entity types
        self.entity_type_weights = {
            'drug': 1.0,
            'medication': 1.0,
            'disease': 0.9,
            'condition': 0.9,
            'symptom': 0.8,
            'adverse_event': 0.8,
            'procedure': 0.7,
            'measurement': 0.6,
            'population': 0.5,
            'temporal': 0.4
        }
        
        # Vocabulary authority weights
        self.vocabulary_weights = {
            'rxnorm': 1.0,      # Authoritative for medications
            'snomed': 0.9,      # Comprehensive clinical terminology
            'umls': 0.8,        # Broad medical coverage
            'icd10': 0.7,       # Diagnostic codes
            'loinc': 0.6        # Laboratory terms
        }
    
    async def connect(self):
        """Establish database connection"""
        self.conn = await psycopg.AsyncConnection.connect(self.connection_string)
        await self.conn.execute("LOAD 'age'")
        await self.conn.execute("SET search_path = ag_catalog, '$user', public")
        logger.info("Connected to database and loaded AGE extension")
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
    
    async def u_retrieval_search(
        self,
        query: str,
        query_type: QueryType = QueryType.HYBRID_SEARCH,
        search_scope: SearchScope = SearchScope.GLOBAL,
        context: Optional[QueryContext] = None,
        max_results: int = 50
    ) -> URetrievalResult:
        """
        Perform U-Retrieval search with hierarchical community-aware ranking.
        
        Args:
            query: Search query text
            query_type: Type of search to perform
            search_scope: Scope of search (global, community, local)
            context: Additional query context
            max_results: Maximum number of results to return
            
        Returns:
            URetrievalResult with hierarchical search results
        """
        start_time = asyncio.get_event_loop().time()
        
        if context is None:
            context = QueryContext()
        
        logger.info(f"Starting U-Retrieval search: '{query}' (type: {query_type.value}, scope: {search_scope.value})")
        
        try:
            await self.connect()
            
            # Step 1: Global search - find relevant communities
            relevant_communities = await self._find_relevant_communities(query, context)
            logger.info(f"Found {len(relevant_communities)} relevant communities")
            
            # Step 2: Community-aware entity search
            entity_results = await self._community_aware_entity_search(
                query, relevant_communities, context, max_results
            )
            logger.info(f"Found {len(entity_results)} entity matches")
            
            # Step 3: Relation-aware expansion (if hybrid search)
            if query_type in [QueryType.RELATION_SEARCH, QueryType.HYBRID_SEARCH]:
                relation_results = await self._relation_aware_expansion(
                    query, entity_results, context
                )
                entity_results.extend(relation_results)
                logger.info(f"Expanded to {len(entity_results)} results with relations")
            
            # Step 4: Community aggregation and ranking
            ranked_results = await self._community_aware_ranking(
                entity_results, relevant_communities, query
            )
            
            # Step 5: Global context aggregation
            global_context = await self._aggregate_global_context(
                ranked_results, relevant_communities
            )
            
            # Step 6: Community-level aggregation
            community_aggregation = await self._aggregate_community_context(
                ranked_results, relevant_communities
            )
            
            end_time = asyncio.get_event_loop().time()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Create final result
            result = URetrievalResult(
                query=query,
                query_type=query_type,
                search_scope=search_scope,
                results=ranked_results[:max_results],
                community_aggregation=community_aggregation,
                global_context=global_context,
                processing_stats={
                    "communities_searched": len(relevant_communities),
                    "raw_results": len(entity_results),
                    "final_results": min(len(ranked_results), max_results),
                    "processing_steps": ["community_discovery", "entity_search", "relation_expansion", "ranking", "aggregation"]
                },
                total_results=len(ranked_results),
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"U-Retrieval search completed in {processing_time_ms:.1f}ms")
            return result
            
        finally:
            await self.close()
    
    async def _find_relevant_communities(
        self, 
        query: str, 
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Find communities relevant to the query"""
        
        # Get top communities by occurrence (limit to avoid performance issues with 33K communities)
        community_query = """
            SELECT 
                cluster_key,
                level,
                title,
                nodes,
                edges,
                chunk_ids,
                occurrence,
                created_at
            FROM ag_catalog.communities
            ORDER BY occurrence DESC, cluster_key
            LIMIT 100
        """
        
        result = await self.conn.execute(community_query)
        communities = await result.fetchall()
        
        if not communities:
            logger.warning("No communities found in database")
            return []
        
        relevant_communities = []
        query_lower = query.lower()
        
        # Filter out common words that don't add meaning
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'is', 'are', 'how', 'when', 'where', 'why'}
        
        # Strip punctuation from query terms (CRITICAL FIX: removes '?' from 'niraparib?')
        import re
        meaningful_terms = [
            re.sub(r'[^\w\s]', '', term) 
            for term in query_lower.split() 
            if term not in stop_words and len(term) > 2
        ]
        # Remove any empty strings after punctuation removal
        meaningful_terms = [term for term in meaningful_terms if term]
        
        logger.info(f"Meaningful terms after punctuation removal: {meaningful_terms}")
        
        for community in communities:
            cluster_key, level, title, nodes, edges, chunk_ids, occurrence, created_at = community
            
            # Parse JSON fields
            nodes_list = json.loads(nodes) if isinstance(nodes, str) else nodes
            edges_list = json.loads(edges) if isinstance(edges, str) else edges
            chunk_ids_list = json.loads(chunk_ids) if isinstance(chunk_ids, str) else chunk_ids
            
            # Calculate relevance score for this community
            relevance_score = 0.0
            
            # Check title relevance with meaningful terms
            if title and any(term in title.lower() for term in meaningful_terms):
                relevance_score += 0.3
            
            # Check if community contains entities matching meaningful query terms
            # CHANGED: Pass nodes_list (meta_graph_id UUIDs) instead of chunk_ids_list
            entity_match_score = await self._calculate_community_entity_relevance(
                nodes_list, query_lower, meaningful_terms
            )
            relevance_score += entity_match_score * 0.7
            
            # Boost relevance for safety/clinical terms
            clinical_terms = {'adverse', 'safety', 'monitor', 'effect', 'event', 'risk'}
            if any(term in meaningful_terms for term in clinical_terms):
                relevance_score *= 1.5
            
            # Don't weight by occurrence to avoid missing relevant but smaller communities
            
            # Accept communities with any relevance
            if relevance_score > 0:
                relevant_communities.append({
                    'cluster_key': cluster_key,
                    'level': level,
                    'title': title,
                    'nodes': nodes_list,
                    'edges': edges_list,
                    'chunk_ids': chunk_ids_list,
                    'occurrence': occurrence,
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score
        relevant_communities.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        if len(relevant_communities) > 0:
            logger.info(f"Found {len(relevant_communities)} communities with relevance > 0.001")
            top_scores = ', '.join([f"{c['title']}: {c['relevance_score']:.4f}" for c in relevant_communities[:3]])
            logger.info(f"Top 3 scores: {top_scores}")
            return relevant_communities[:10]  # Top 10 most relevant
        
        logger.warning("No relevant communities found - query may not match any content")
        return []
    
    async def _calculate_community_entity_relevance(
        self, 
        meta_graph_ids: List[str], 
        query_lower: str,
        meaningful_terms: List[str] = None
    ) -> float:
        """
        Calculate how well community entities match the query.
        
        SIMPLIFIED: Communities now contain meta_graph_id UUIDs (not entity UUIDs).
        This eliminates the 2-query lookup (entity_ids → meta_graphs → entities)
        and replaces it with 1 query (meta_graph_ids → entities).
        
        Args:
            meta_graph_ids: Meta_graph UUIDs from the community (e.g., ['babc2bbe-794c-...'])
            query_lower: lowercase query string
            meaningful_terms: extracted query terms
        """
        if not meta_graph_ids:
            return 0.0
        
        # SINGLE QUERY: Get entities from these meta_graphs that match query terms
        # Build LIKE patterns from meaningful_terms
        like_patterns = []
        params = list(meta_graph_ids)
        
        if meaningful_terms:
            # Add OR clauses for each term
            like_clauses = []
            for term in meaningful_terms:
                if len(term) > 2:
                    like_clauses.append("LOWER(entity_text) LIKE %s")
                    params.append(f"%{term}%")
            
            like_filter = f" AND ({' OR '.join(like_clauses)})" if like_clauses else ""
        else:
            like_filter = ""
        
        mg_placeholders = ','.join(['%s'] * len(meta_graph_ids))
        entity_query = f"""
            SELECT entity_text, entity_type, normalized_id
            FROM docintel.entities 
            WHERE meta_graph_id::text IN ({mg_placeholders})
            {like_filter}
            LIMIT 500
        """
        
        result = await self.conn.execute(entity_query, params)
        entities = await result.fetchall()
        
        if not entities:
            return 0.0
        
        relevance_score = 0.0
        search_terms = meaningful_terms if meaningful_terms else query_lower.split()
        match_count = 0
        
        
        for entity_text, entity_type, normalized_id in entities:
            entity_lower = entity_text.lower()
            entity_match = False
            
            # Exact match - very high weight (case-insensitive)
            for term in search_terms:
                if len(term) > 2 and term.lower() == entity_lower:
                    relevance_score += 2.0
                    match_count += 1
                    entity_match = True
                    break
            
            if not entity_match:
                # Substring match - good weight
                for term in search_terms:
                    if len(term) > 3 and term in entity_lower:
                        relevance_score += 1.0
                        match_count += 1
                        entity_match = True
                        break
            
            # Boost for high-value entity types
            if entity_match:
                if entity_type in {'drug', 'medication', 'disease', 'condition', 'adverse_event'}:
                    relevance_score += 0.5
        
        # Return score combining density and absolute matches
        # Weight more heavily by absolute match count to prioritize communities with ANY matches
        if len(entities) > 0 and match_count > 0:
            density_score = relevance_score / len(entities)
            match_bonus = match_count * 0.01  # Boost by 0.01 per matching entity
            return density_score + match_bonus
        return 0.0
    
    async def _community_aware_entity_search(
        self,
        query: str,
        relevant_communities: List[Dict[str, Any]],
        context: QueryContext,
        max_results: int
    ) -> List[SearchResult]:
        """
        Search for entities with community awareness.
        
        CRITICAL ARCHITECTURE UNDERSTANDING:
        - AGE graph nodes represent meta_graphs (chunks), NOT individual entities
        - AGE node ID '50' = the 50th meta_graph in creation order
        - Communities cluster meta_graphs, then we get all entities from those meta_graphs
        - This preserves graph structure while working with UUID entity schema
        """
        
        search_results = []
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        # Get meta_graph IDs from relevant communities
        # CHANGED: Communities now contain meta_graph UUIDs (not entity UUIDs)
        all_meta_graph_ids = set()
        meta_graph_to_community = {}  # meta_graph_id -> community info
        
        for community in relevant_communities:
            for meta_graph_id in community['nodes']:  # Meta_graph UUIDs
                all_meta_graph_ids.add(meta_graph_id)
                meta_graph_to_community[meta_graph_id] = community
        
        if not all_meta_graph_ids:
            logger.warning("No meta_graph IDs found in relevant communities")
            return []
        
        # Get all entities from these meta_graphs
        meta_graph_list = list(all_meta_graph_ids)
        mg_placeholders = ','.join(['%s'] * len(meta_graph_list))
        
        result = await self.conn.execute(f"""
            SELECT entity_id, meta_graph_id, entity_text, entity_type
            FROM docintel.entities 
            WHERE meta_graph_id::text IN ({mg_placeholders})
        """, meta_graph_list)
        rows = await result.fetchall()
        
        # Build mappings
        meta_graph_ids = list(all_meta_graph_ids)
        entity_to_meta_graph = {}  # For reverse lookup
        
        for entity_id_val, meta_graph_id_val, entity_text, entity_type in rows:
            entity_id = str(entity_id_val)
            meta_graph_id = str(meta_graph_id_val)
            entity_to_meta_graph[entity_id] = meta_graph_id
        
        if not meta_graph_ids:
            logger.warning(f"No meta_graphs found in communities")
            return []
        
        logger.info(f"Found {len(rows)} entities across {len(meta_graph_ids)} meta_graphs")
        
        # Search entities from these meta_graphs
        placeholders = ','.join(['%s'] * len(meta_graph_ids))
        
        # Build more precise entity filter based on query context
        entity_type_filter = ""
        if context.entity_types:
            type_placeholders = ','.join(['%s'] * len(context.entity_types))
            entity_type_filter = f" AND e.entity_type IN ({type_placeholders})"
        
        # CRITICAL FIX: Prioritize entities that have relations for graph expansion
        # Left join with relations to count connections, then sort by relation_count DESC
        search_query = f"""
            SELECT 
                e.entity_id,
                e.entity_text,
                e.entity_type,
                e.confidence,
                e.normalized_id,
                e.normalized_source,
                e.context_flags,
                e.chunk_id,
                e.source_chunk_id,
                e.meta_graph_id,
                COUNT(r.relation_id) as relation_count
            FROM docintel.entities e
            LEFT JOIN docintel.relations r ON (
                e.entity_id = r.subject_entity_id OR e.entity_id = r.object_entity_id
            )
            WHERE e.meta_graph_id::text IN ({placeholders})
            AND (
                LOWER(e.entity_text) LIKE %s
                OR LOWER(e.entity_text) LIKE %s
                OR e.normalized_id IS NOT NULL
            )
            {entity_type_filter}
            GROUP BY e.entity_id, e.entity_text, e.entity_type, e.confidence, 
                     e.normalized_id, e.normalized_source, e.context_flags, 
                     e.chunk_id, e.source_chunk_id, e.meta_graph_id
            ORDER BY 
                COUNT(r.relation_id) DESC,  -- Prioritize entities with relations
                CASE 
                    WHEN e.entity_type IN ('drug', 'medication', 'adverse_event', 'disease') THEN 1
                    WHEN e.entity_type IN ('procedure', 'measurement') THEN 2
                    ELSE 3
                END,
                e.confidence DESC
            LIMIT 200
        """
        
        # Create search patterns - try exact terms first
        search_patterns = []
        for term in query_terms:
            if len(term) > 2:
                search_patterns.append(f"%{term}%")
        
        # Use first two most meaningful patterns
        if not search_patterns:
            search_patterns = [f"%{query_lower}%", f"%{query_lower}%"]
        elif len(search_patterns) == 1:
            search_patterns.append(search_patterns[0])
        
        params = meta_graph_ids + search_patterns[:2]
        if context.entity_types:
            params.extend(context.entity_types)
        
        result = await self.conn.execute(search_query, params)
        entities = await result.fetchall()
        
        # Log statistics about relations
        entities_with_relations = sum(1 for e in entities if e[10] > 0)  # relation_count is column 10
        logger.info(f"Found {len(entities)} entities from {len(meta_graph_ids)} meta_graphs ({entities_with_relations} have relations)")
        
        # Build results with community context
        for entity in entities:
            (entity_id, entity_text, entity_type, confidence, normalized_id, 
             normalized_source, context_flags, chunk_id, source_chunk_id, meta_graph_id, relation_count) = entity
            
            # Calculate relevance score
            relevance_score = self._calculate_entity_relevance_score(
                entity_text, entity_type, query_terms, confidence, normalized_source
            )
            
            # Find which community this entity belongs to
            # Get community info from meta_graph
            meta_graph_id_str = entity_to_meta_graph.get(str(entity_id))
            community_info = meta_graph_to_community.get(meta_graph_id_str, {}) if meta_graph_id_str else {}
            
            # Create search result
            search_result = SearchResult(
                entity_id=str(entity_id),
                entity_text=entity_text,
                entity_type=entity_type,
                normalized_concept_id=normalized_id,
                normalized_vocabulary=normalized_source,
                confidence=confidence or 0.0,
                community_id=community_info.get('cluster_key'),
                community_title=community_info.get('title'),
                chunk_id=str(chunk_id) if chunk_id else "",
                document_context="",  # Will be filled later if needed
                relevance_score=relevance_score,
                explanation=f"Found in community '{community_info.get('title', 'Unknown')}' with relevance {relevance_score:.3f}",
                metadata={
                    'community_occurrence': community_info.get('occurrence', 0.0),
                    'community_level': community_info.get('level', 0),
                    'context_flags': context_flags if isinstance(context_flags, dict) else {},
                    'normalized_id': normalized_id,
                    'normalized_source': normalized_source,
                    'source_chunk_id': source_chunk_id,
                    'meta_graph_id': str(meta_graph_id)
                }
            )
            
            search_results.append(search_result)
        
        logger.info(f"Returning {len(search_results)} entities matching query")
        return search_results
    
    def _calculate_entity_relevance_score(
        self,
        entity_text: str,
        entity_type: str,
        query_terms: List[str],
        confidence: float,
        normalized_source: Optional[str]
    ) -> float:
        """Calculate relevance score for an entity"""
        
        score = 0.0
        entity_lower = entity_text.lower()
        
        # Text matching score
        for term in query_terms:
            if len(term) < 3:
                continue
            if term == entity_lower:
                score += 2.0  # Exact match - very high
            elif entity_lower.startswith(term) or entity_lower.endswith(term):
                score += 1.5  # Strong match
            elif term in entity_lower:
                score += 1.0  # Contains term
        
        # If no matches at all, return 0
        if score == 0.0:
            return 0.0
        
        # Entity type weighting - prioritize clinical entities
        type_weight = self.entity_type_weights.get(entity_type, 0.3)
        score *= type_weight
        
        # Confidence weighting
        conf_weight = confidence if confidence else 0.5
        score *= conf_weight
        
        # Vocabulary authority weighting - boost for normalized entities
        if normalized_source:
            vocab_weight = self.vocabulary_weights.get(normalized_source.lower(), 0.5)
            score *= (1.0 + vocab_weight)  # Additive boost instead of multiplicative
        
        return score
    
    async def _relation_aware_expansion(
        self,
        query: str,
        entity_results: List[SearchResult],
        context: QueryContext,
        max_hops: int = 2
    ) -> List[SearchResult]:
        """
        Expand results using AGE graph multi-hop traversal.
        
        Args:
            query: Search query
            entity_results: Initial entity results to expand from
            context: Query context
            max_hops: Maximum relationship hops (1-3 recommended, default 2)
        
        Returns:
            List of expanded SearchResult objects found via graph traversal
        """
        if not entity_results:
            return []
        
        expanded_results = []
        entity_ids = [result.entity_id for result in entity_results[:20]]  # Limit seed entities for performance
        
        if not entity_ids:
            return []
        
        logger.info(f"Performing {max_hops}-hop AGE graph expansion from {len(entity_ids)} seed entities")
        logger.debug(f"Seed entity IDs: {entity_ids[:3]}...")  # Show first 3 IDs
        
        # Use AGE Cypher for multi-hop traversal
        # Build entity_id filter for WHERE clause
        entity_id_list = "', '".join(entity_ids)
        
        # AGE Cypher syntax - no list comprehension support, use relationships() directly
        cypher_query = f"""
            MATCH path = (start:Entity)-[r:RELATES_TO*1..{max_hops}]->(target:Entity)
            WHERE start.entity_id IN ['{entity_id_list}']
            RETURN 
                target.entity_id as entity_id,
                target.entity_text as entity_text,
                target.entity_type as entity_type,
                target.normalized_id as normalized_id,
                length(path) as hop_distance,
                relationships(path) as path_rels
            LIMIT 100
        """
        
        try:
            # Execute Cypher directly (AGE doesn't support parameterized Cypher queries)
            result = await self.conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('clinical_graph', $$
                {cypher_query}
                $$) as (
                    entity_id agtype, entity_text agtype, entity_type agtype, 
                    normalized_id agtype, hop_distance agtype, path_rels agtype
                )
            """)
            
            graph_results = await result.fetchall()
            logger.info(f"AGE graph traversal returned {len(graph_results)} entities")
            
            query_lower = query.lower()
            seen_entities = {r.entity_id for r in entity_results}  # Don't duplicate initial results
            logger.debug(f"Deduplication: tracking {len(seen_entities)} existing entity IDs")
            
            skipped_duplicates = 0
            for row in graph_results:
                entity_id_raw, entity_text_raw, entity_type_raw, normalized_id_raw, hop_dist_raw, path_rels_raw = row
                
                # Parse agtype values (strip quotes and extract JSON)
                entity_id = str(entity_id_raw).strip('"')
                entity_text = str(entity_text_raw).strip('"')
                entity_type = str(entity_type_raw).strip('"') if entity_type_raw else 'unknown'
                normalized_id = str(normalized_id_raw).strip('"') if normalized_id_raw and str(normalized_id_raw) != 'null' else None
                hop_distance = int(str(hop_dist_raw)) if hop_dist_raw else 1
                
                # Skip if already in results
                if entity_id in seen_entities:
                    skipped_duplicates += 1
                    continue
                seen_entities.add(entity_id)
                
                # Parse relationships from path (AGE returns array of edge objects)
                try:
                    import re
                    # Extract predicates and confidences from relationship array
                    path_rels_str = str(path_rels_raw)
                    predicates = re.findall(r'predicate["\s:]+([^"]+)"', path_rels_str)
                    confidences = re.findall(r'confidence["\s:]+(\d+\.?\d*)', path_rels_str)
                    
                    predicate_path = ' → '.join(predicates) if predicates else 'related'
                    avg_confidence = sum(float(c) for c in confidences) / len(confidences) if confidences else 0.5
                except Exception as e:
                    logger.debug(f"Failed to parse path relationships: {e}")
                    predicate_path = 'related'
                    avg_confidence = 0.5
                
                # Penalize distant hops: 1-hop = 0.4, 2-hop = 0.25, 3-hop = 0.15
                hop_penalty = max(0.1, 0.5 / hop_distance)
                relevance_score = hop_penalty * avg_confidence
                
                # Extract evidence text from relationships
                try:
                    evidences = re.findall(r'evidence["\s:]+([^"]+)"', path_rels_str)
                    evidence_text = ' | '.join(evidences[:2]) if evidences else ""  # First 2 evidences
                except:
                    evidence_text = ""
                
                expanded_result = SearchResult(
                    entity_id=entity_id,
                    entity_text=entity_text,
                    entity_type=entity_type,
                    normalized_concept_id=normalized_id,
                    normalized_vocabulary=normalized_id.split(':')[0] if normalized_id and ':' in normalized_id else None,
                    confidence=avg_confidence,
                    community_id=None,
                    community_title=None,
                    chunk_id="",
                    document_context=evidence_text[:200],
                    relevance_score=relevance_score,
                    explanation=f"Found via {hop_distance}-hop graph traversal: {predicate_path}",
                    metadata={
                        'relation_type': 'graph_expansion',
                        'hop_distance': hop_distance,
                        'predicate_path': predicate_path,
                        'avg_confidence': avg_confidence,
                        'source_chunk_id': None  # Will be populated in post-processing
                    }
                )
                
                expanded_results.append(expanded_result)
            
            # Batch lookup source_chunk_id for all graph-expanded entities
            if expanded_results:
                entity_uuid_list = [r.entity_id for r in expanded_results]
                placeholders = ','.join(['%s'] * len(entity_uuid_list))
                
                chunk_lookup_result = await self.conn.execute(f"""
                    SELECT entity_id, source_chunk_id
                    FROM docintel.entities
                    WHERE entity_id::text IN ({placeholders})
                """, entity_uuid_list)
                
                chunk_lookup_rows = await chunk_lookup_result.fetchall()
                entity_to_chunk = {str(row[0]): row[1] for row in chunk_lookup_rows if row[1]}
                
                # Populate source_chunk_id in metadata
                for result in expanded_results:
                    if result.entity_id in entity_to_chunk:
                        result.metadata['source_chunk_id'] = entity_to_chunk[result.entity_id]
                
                logger.debug(f"Populated source_chunk_id for {len(entity_to_chunk)}/{len(expanded_results)} graph-expanded entities")
        
        except Exception as e:
            logger.error(f"AGE graph traversal failed: {e}", exc_info=True)
            # Fallback: return empty list, don't crash
            return []
        
        logger.info(f"Graph expansion: {len(graph_results)} retrieved, {skipped_duplicates} skipped as duplicates, {len(expanded_results)} new entities added")
        return expanded_results
    
    async def _community_aware_ranking(
        self,
        results: List[SearchResult],
        relevant_communities: List[Dict[str, Any]],
        query: str
    ) -> List[SearchResult]:
        """Rank results using community context"""
        
        # Create community relevance lookup
        community_relevance = {
            comm['cluster_key']: comm['relevance_score'] 
            for comm in relevant_communities
        }
        
        # Enhance ranking with community context
        for result in results:
            # Base relevance score
            final_score = result.relevance_score
            
            # Community boost
            if result.community_id and result.community_id in community_relevance:
                community_boost = community_relevance[result.community_id] * 0.3
                final_score += community_boost
                
                # Update explanation
                result.explanation += f" (community boost: +{community_boost:.3f})"
            
            # Update final relevance score
            result.relevance_score = final_score
        
        # Sort by enhanced relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    async def _aggregate_global_context(
        self,
        results: List[SearchResult],
        relevant_communities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate global context from search results"""
        
        # Entity type distribution
        entity_types = Counter(result.entity_type for result in results)
        
        # Vocabulary distribution
        vocabularies = Counter(
            result.normalized_vocabulary for result in results 
            if result.normalized_vocabulary
        )
        
        # Community distribution
        communities = Counter(
            result.community_title for result in results 
            if result.community_title
        )
        
        # Confidence statistics
        confidences = [result.confidence for result in results if result.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_entities': len(results),
            'unique_entity_types': len(entity_types),
            'entity_type_distribution': dict(entity_types.most_common(10)),
            'vocabulary_distribution': dict(vocabularies.most_common()),
            'community_distribution': dict(communities.most_common()),
            'average_confidence': float(avg_confidence),
            'communities_involved': len(relevant_communities),
            'search_coverage': {
                'entities_with_normalization': sum(1 for r in results if r.normalized_concept_id),
                'entities_with_communities': sum(1 for r in results if r.community_id),
                'entities_with_relations': sum(1 for r in results if r.metadata.get('relation_type'))
            }
        }
    
    async def _aggregate_community_context(
        self,
        results: List[SearchResult],
        relevant_communities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate community-level context"""
        
        community_results = defaultdict(list)
        
        # Group results by community
        for result in results:
            if result.community_id:
                community_results[result.community_id].append(result)
        
        community_summaries = {}
        
        for community_id, community_results_list in community_results.items():
            # Find community info
            community_info = next(
                (c for c in relevant_communities if c['cluster_key'] == community_id), 
                {}
            )
            
            # Calculate community-specific statistics
            entity_types = Counter(r.entity_type for r in community_results_list)
            avg_relevance = np.mean([r.relevance_score for r in community_results_list])
            
            community_summaries[community_id] = {
                'title': community_info.get('title', f'Community {community_id}'),
                'total_results': len(community_results_list),
                'average_relevance': float(avg_relevance),
                'entity_types': dict(entity_types),
                'occurrence': community_info.get('occurrence', 0.0),
                'level': community_info.get('level', 0)
            }
       