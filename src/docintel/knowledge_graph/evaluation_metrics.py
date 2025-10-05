"""
Clinical Knowledge Graph Evaluation Metrics

Comprehensive evaluation system for assessing quality of:
- Entity extraction (precision, recall, F1)
- Relation extraction (accuracy, coverage)
- Community detection (modularity, silhouette)
- Clinical relevance (domain-specific scoring)
- U-Retrieval performance (relevance ranking)

Supports Medical-Graph-RAG compliance assessment.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, silhouette_score
import psycopg

logger = logging.getLogger(__name__)


@dataclass
class EntityEvaluationMetrics:
    """Entity extraction evaluation results"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_extracted: int
    total_expected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    entity_type_breakdown: Dict[str, Dict[str, float]]
    coverage_rate: float
    clinical_relevance_score: float


@dataclass
class RelationEvaluationMetrics:
    """Relation extraction evaluation results"""
    precision: float
    recall: float
    f1_score: float
    total_extracted: int
    total_expected: int
    relation_type_breakdown: Dict[str, Dict[str, float]]
    graph_connectivity: float
    semantic_coherence: float


@dataclass
class CommunityEvaluationMetrics:
    """Community detection evaluation results"""
    modularity: float
    silhouette_score: float
    num_communities: int
    average_community_size: float
    community_coherence: float
    coverage: float
    clinical_clustering_quality: float


@dataclass
class RetrievalEvaluationMetrics:
    """U-Retrieval system evaluation results"""
    mean_reciprocal_rank: float
    ndcg_at_k: Dict[int, float]  # NDCG@1, @5, @10
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    average_query_time: float
    clinical_relevance_correlation: float


@dataclass
class ComprehensiveEvaluationReport:
    """Complete evaluation report"""
    entity_metrics: EntityEvaluationMetrics
    relation_metrics: RelationEvaluationMetrics
    community_metrics: CommunityEvaluationMetrics
    retrieval_metrics: RetrievalEvaluationMetrics
    overall_clinical_relevance: float
    medical_graph_rag_compliance: float
    timestamp: str
    dataset_info: Dict[str, Any]


class ClinicalEvaluationFramework:
    """
    Comprehensive evaluation framework for clinical knowledge graphs.
    
    Implements evaluation metrics for Medical-Graph-RAG compliance:
    - Entity extraction quality assessment
    - Relation extraction validation
    - Community detection evaluation
    - Retrieval system performance
    - Clinical domain relevance scoring
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        
        # Clinical domain weights for relevance scoring
        self.clinical_entity_weights = {
            'medication': 1.0,
            'drug': 1.0,
            'adverse_event': 0.95,
            'condition': 0.9,
            'disease': 0.9,
            'procedure': 0.85,
            'measurement': 0.8,
            'organization': 0.7,
            'population': 0.75,
            'endpoint': 0.8,
            'temporal': 0.6
        }
        
        # High-value clinical relation types
        self.clinical_relation_weights = {
            'treats': 1.0,
            'causes': 0.95,
            'prevents': 0.9,
            'interacts_with': 0.85,
            'measured_by': 0.8,
            'associated_with': 0.75,
            'occurs_in': 0.7
        }
        
        # Gold standard clinical vocabularies for validation
        self.authoritative_vocabularies = {
            'rxnorm', 'snomed', 'umls', 'icd10', 'loinc'
        }
    
    async def connect(self):
        """Establish database connection"""
        self.conn = await psycopg.AsyncConnection.connect(self.connection_string)
        await self.conn.execute("LOAD 'age';")
        await self.conn.execute('SET search_path = ag_catalog, public;')
        logger.info("Connected to database for evaluation")
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
    
    async def evaluate_entity_extraction(
        self, 
        ground_truth_entities: Optional[List[Dict[str, Any]]] = None
    ) -> EntityEvaluationMetrics:
        """
        Evaluate entity extraction quality using various metrics.
        
        Args:
            ground_truth_entities: Optional gold standard entities for validation
            
        Returns:
            EntityEvaluationMetrics with comprehensive assessment
        """
        logger.info("🔍 Evaluating entity extraction quality...")
        
        # Get extracted entities from database
        result = await self.conn.execute("""
            SELECT 
                id, entity_text, entity_type, confidence, 
                normalized_id, normalized_source,
                context_flags, chunk_id
            FROM entities
            ORDER BY confidence DESC
        """)
        extracted_entities = await result.fetchall()
        
        if not extracted_entities:
            logger.warning("No entities found for evaluation")
            return self._empty_entity_metrics()
        
        logger.info(f"Evaluating {len(extracted_entities)} extracted entities")
        
        # Calculate basic statistics
        total_extracted = len(extracted_entities)
        entity_types = Counter([entity[2] for entity in extracted_entities])
        
        # Calculate coverage rate (entities per document/chunk)
        result = await self.conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM entities")
        unique_chunks = (await result.fetchone())[0]
        coverage_rate = total_extracted / max(unique_chunks, 1)
        
        # Evaluate clinical relevance
        clinical_relevance_score = await self._calculate_clinical_relevance(extracted_entities)
        
        # Entity type breakdown analysis
        entity_type_breakdown = {}
        for entity_type, count in entity_types.items():
            # Calculate confidence statistics for each type
            type_entities = [e for e in extracted_entities if e[2] == entity_type]
            confidences = [e[3] for e in type_entities if e[3] is not None]
            
            entity_type_breakdown[entity_type] = {
                'count': count,
                'percentage': count / total_extracted * 100,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'normalization_rate': sum(1 for e in type_entities if e[4] is not None) / count,
                'clinical_weight': self.clinical_entity_weights.get(entity_type, 0.5)
            }
        
        # If ground truth is available, calculate precision/recall
        if ground_truth_entities:
            precision, recall, f1, accuracy, tp, fp, fn = await self._calculate_entity_prf(
                extracted_entities, ground_truth_entities
            )
        else:
            # Use heuristic evaluation based on clinical criteria
            precision, recall, f1 = await self._heuristic_entity_evaluation(extracted_entities)
            accuracy = (precision + recall) / 2
            tp = int(total_extracted * precision)
            fp = total_extracted - tp
            fn = int(tp / recall) - tp if recall > 0 else 0
        
        return EntityEvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            total_extracted=total_extracted,
            total_expected=tp + fn,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            entity_type_breakdown=entity_type_breakdown,
            coverage_rate=coverage_rate,
            clinical_relevance_score=clinical_relevance_score
        )
    
    async def evaluate_relation_extraction(
        self, 
        ground_truth_relations: Optional[List[Dict[str, Any]]] = None
    ) -> RelationEvaluationMetrics:
        """
        Evaluate relation extraction quality and graph connectivity.
        
        Args:
            ground_truth_relations: Optional gold standard relations
            
        Returns:
            RelationEvaluationMetrics with comprehensive assessment
        """
        logger.info("🔗 Evaluating relation extraction quality...")
        
        # Get extracted relations from database
        result = await self.conn.execute("""
            SELECT 
                id, subject_entity_id, object_entity_id, 
                predicate, confidence, evidence_span, chunk_id
            FROM relations
            ORDER BY confidence DESC
        """)
        extracted_relations = await result.fetchall()
        
        if not extracted_relations:
            logger.warning("No relations found for evaluation")
            return self._empty_relation_metrics()
        
        total_extracted = len(extracted_relations)
        relation_types = Counter([rel[3] for rel in extracted_relations])
        
        logger.info(f"Evaluating {total_extracted} extracted relations")
        
        # Calculate graph connectivity metrics
        graph_connectivity = await self._calculate_graph_connectivity(extracted_relations)
        
        # Calculate semantic coherence
        semantic_coherence = await self._calculate_semantic_coherence(extracted_relations)
        
        # Relation type breakdown
        relation_type_breakdown = {}
        for rel_type, count in relation_types.items():
            type_relations = [r for r in extracted_relations if r[3] == rel_type]
            confidences = [r[4] for r in type_relations if r[4] is not None]
            
            relation_type_breakdown[rel_type] = {
                'count': count,
                'percentage': count / total_extracted * 100,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'clinical_weight': self.clinical_relation_weights.get(rel_type, 0.5)
            }
        
        # Calculate precision, recall, F1
        if ground_truth_relations:
            precision, recall, f1 = await self._calculate_relation_prf(
                extracted_relations, ground_truth_relations
            )
        else:
            precision, recall, f1 = await self._heuristic_relation_evaluation(extracted_relations)
        
        return RelationEvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_extracted=total_extracted,
            total_expected=int(total_extracted / precision) if precision > 0 else total_extracted,
            relation_type_breakdown=relation_type_breakdown,
            graph_connectivity=graph_connectivity,
            semantic_coherence=semantic_coherence
        )
    
    async def evaluate_community_detection(self) -> CommunityEvaluationMetrics:
        """
        Evaluate community detection quality using clustering metrics.
        
        Returns:
            CommunityEvaluationMetrics with clustering assessment
        """
        logger.info("🏘️ Evaluating community detection quality...")
        
        # Get communities and their entities
        result = await self.conn.execute("""
            SELECT cluster_key, title, nodes, occurrence, level
            FROM communities
            ORDER BY occurrence DESC
        """)
        communities = await result.fetchall()
        
        if not communities:
            logger.warning("No communities found for evaluation")
            return self._empty_community_metrics()
        
        num_communities = len(communities)
        
        # Calculate basic community statistics
        community_sizes = []
        total_nodes = 0
        
        for cluster_key, title, nodes_json, occurrence, level in communities:
            nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            community_sizes.append(len(nodes))
            total_nodes += len(nodes)
        
        average_community_size = np.mean(community_sizes)
        
        # Calculate modularity and silhouette score
        modularity = await self._calculate_modularity(communities)
        silhouette = await self._calculate_community_silhouette(communities)
        
        # Calculate community coherence (clinical relevance within communities)
        coherence = await self._calculate_community_coherence(communities)
        
        # Calculate coverage (how many entities are in communities)
        result = await self.conn.execute("SELECT COUNT(*) FROM entities")
        total_entities = (await result.fetchone())[0]
        coverage = total_nodes / max(total_entities, 1)
        
        # Clinical clustering quality assessment
        clinical_quality = await self._assess_clinical_clustering_quality(communities)
        
        logger.info(f"Evaluated {num_communities} communities with avg size {average_community_size:.1f}")
        
        return CommunityEvaluationMetrics(
            modularity=modularity,
            silhouette_score=silhouette,
            num_communities=num_communities,
            average_community_size=average_community_size,
            community_coherence=coherence,
            coverage=coverage,
            clinical_clustering_quality=clinical_quality
        )
    
    async def evaluate_retrieval_system(
        self, 
        test_queries: List[Dict[str, Any]] = None
    ) -> RetrievalEvaluationMetrics:
        """
        Evaluate U-Retrieval system performance using ranking metrics.
        
        Args:
            test_queries: List of test queries with expected results
            
        Returns:
            RetrievalEvaluationMetrics with retrieval assessment
        """
        logger.info("🔍 Evaluating U-Retrieval system performance...")
        
        from .u_retrieval import ClinicalURetrieval, QueryType, SearchScope
        
        # Use default test queries if none provided
        if test_queries is None:
            test_queries = await self._generate_test_queries()
        
        retrieval_system = ClinicalURetrieval(self.connection_string)
        
        mrr_scores = []
        ndcg_scores = {1: [], 5: [], 10: []}
        precision_scores = {1: [], 5: [], 10: []}
        recall_scores = {1: [], 5: [], 10: []}
        query_times = []
        
        for query_data in test_queries:
            query = query_data['query']
            expected_entities = set(query_data.get('expected_entities', []))
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await retrieval_system.u_retrieval_search(
                    query=query,
                    query_type=QueryType.HYBRID_SEARCH,
                    search_scope=SearchScope.GLOBAL,
                    max_results=10
                )
                
                end_time = asyncio.get_event_loop().time()
                query_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate ranking metrics
                retrieved_entities = [r.entity_text for r in result.results]
                
                if expected_entities and retrieved_entities:
                    # MRR calculation
                    mrr_score = self._calculate_mrr(retrieved_entities, expected_entities)
                    mrr_scores.append(mrr_score)
                    
                    # NDCG and Precision@K
                    for k in [1, 5, 10]:
                        ndcg_k = self._calculate_ndcg_at_k(retrieved_entities, expected_entities, k)
                        prec_k = self._calculate_precision_at_k(retrieved_entities, expected_entities, k)
                        recall_k = self._calculate_recall_at_k(retrieved_entities, expected_entities, k)
                        
                        ndcg_scores[k].append(ndcg_k)
                        precision_scores[k].append(prec_k)
                        recall_scores[k].append(recall_k)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                continue
        
        await retrieval_system.close()
        
        # Calculate average metrics
        mean_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        avg_ndcg = {k: np.mean(scores) if scores else 0.0 for k, scores in ndcg_scores.items()}
        avg_precision = {k: np.mean(scores) if scores else 0.0 for k, scores in precision_scores.items()}
        avg_recall = {k: np.mean(scores) if scores else 0.0 for k, scores in recall_scores.items()}
        avg_query_time = np.mean(query_times) if query_times else 0.0
        
        # Clinical relevance correlation (heuristic)
        clinical_correlation = await self._calculate_clinical_relevance_correlation()
        
        logger.info(f"Evaluated {len(test_queries)} queries with avg response time {avg_query_time:.1f}ms")
        
        return RetrievalEvaluationMetrics(
            mean_reciprocal_rank=mean_mrr,
            ndcg_at_k=avg_ndcg,
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            average_query_time=avg_query_time,
            clinical_relevance_correlation=clinical_correlation
        )
    
    async def comprehensive_evaluation(
        self,
        ground_truth_entities: Optional[List[Dict[str, Any]]] = None,
        ground_truth_relations: Optional[List[Dict[str, Any]]] = None,
        test_queries: Optional[List[Dict[str, Any]]] = None
    ) -> ComprehensiveEvaluationReport:
        """
        Run complete evaluation across all system components.
        
        Returns:
            ComprehensiveEvaluationReport with full assessment
        """
        logger.info("🚀 Starting comprehensive evaluation...")
        
        await self.connect()
        
        try:
            # Run all evaluations
            entity_metrics = await self.evaluate_entity_extraction(ground_truth_entities)
            relation_metrics = await self.evaluate_relation_extraction(ground_truth_relations)
            community_metrics = await self.evaluate_community_detection()
            retrieval_metrics = await self.evaluate_retrieval_system(test_queries)
            
            # Calculate overall clinical relevance
            overall_clinical_relevance = self._calculate_overall_clinical_relevance(
                entity_metrics, relation_metrics, community_metrics
            )
            
            # Calculate Medical-Graph-RAG compliance score
            medical_graph_rag_compliance = self._calculate_medical_graph_rag_compliance(
                entity_metrics, relation_metrics, community_metrics, retrieval_metrics
            )
            
            # Get dataset information
            dataset_info = await self._get_dataset_info()
            
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            logger.info("✅ Comprehensive evaluation completed")
            
            return ComprehensiveEvaluationReport(
                entity_metrics=entity_metrics,
                relation_metrics=relation_metrics,
                community_metrics=community_metrics,
                retrieval_metrics=retrieval_metrics,
                overall_clinical_relevance=overall_clinical_relevance,
                medical_graph_rag_compliance=medical_graph_rag_compliance,
                timestamp=timestamp,
                dataset_info=dataset_info
            )
            
        finally:
            await self.close()
    
    # Helper methods for metric calculations
    
    async def _calculate_clinical_relevance(self, entities: List[Tuple]) -> float:
        """Calculate clinical relevance score for entities"""
        if not entities:
            return 0.0
        
        relevance_scores = []
        for entity in entities:
            entity_type = entity[2]
            confidence = entity[3] or 0.0
            normalized_id = entity[4]
            normalized_source = entity[5]
            
            # Base relevance from clinical weights
            base_relevance = self.clinical_entity_weights.get(entity_type, 0.3)
            
            # Boost for normalization to authoritative vocabularies
            normalization_boost = 0.2 if normalized_source in self.authoritative_vocabularies else 0.0
            
            # Weight by confidence
            weighted_relevance = (base_relevance + normalization_boost) * confidence
            relevance_scores.append(weighted_relevance)
        
        return np.mean(relevance_scores)
    
    async def _heuristic_entity_evaluation(self, entities: List[Tuple]) -> Tuple[float, float, float]:
        """Heuristic evaluation of entity quality without ground truth"""
        total_entities = len(entities)
        if total_entities == 0:
            return 0.0, 0.0, 0.0
        
        # Estimate precision based on confidence and normalization
        high_confidence_entities = sum(1 for e in entities if (e[3] or 0) > 0.7)
        normalized_entities = sum(1 for e in entities if e[4] is not None)
        clinical_entities = sum(1 for e in entities if e[2] in self.clinical_entity_weights)
        
        # Heuristic precision estimation
        precision = (
            (high_confidence_entities / total_entities) * 0.4 +
            (normalized_entities / total_entities) * 0.3 +
            (clinical_entities / total_entities) * 0.3
        )
        
        # Estimate recall based on coverage and entity density
        result = await self.conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM entities")
        unique_chunks = (await result.fetchone())[0]
        entities_per_chunk = total_entities / max(unique_chunks, 1)
        
        # Heuristic recall estimation (more entities per chunk suggests better recall)
        recall = min(1.0, entities_per_chunk / 10.0)  # Assuming ~10 entities per chunk is good
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    async def _calculate_graph_connectivity(self, relations: List[Tuple]) -> float:
        """Calculate graph connectivity metrics"""
        if not relations:
            return 0.0
        
        # Build adjacency information
        nodes = set()
        for rel in relations:
            nodes.add(rel[1])  # subject_entity_id
            nodes.add(rel[2])  # object_entity_id
        
        total_nodes = len(nodes)
        total_edges = len(relations)
        
        if total_nodes < 2:
            return 0.0
        
        # Calculate density (actual edges / possible edges)
        max_possible_edges = total_nodes * (total_nodes - 1)  # Directed graph
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        return min(1.0, density * 10)  # Scale to reasonable range
    
    async def _calculate_semantic_coherence(self, relations: List[Tuple]) -> float:
        """Calculate semantic coherence of relations"""
        if not relations:
            return 0.0
        
        # Calculate coherence based on relation type distribution and confidence
        relation_types = Counter([rel[3] for rel in relations])
        confidences = [rel[4] for rel in relations if rel[4] is not None]
        
        # Diversity penalty (too many relation types might indicate noise)
        type_diversity = len(relation_types) / len(relations)
        diversity_score = 1.0 - min(type_diversity, 0.5)  # Penalize if > 50% are different types
        
        # Confidence score
        confidence_score = np.mean(confidences) if confidences else 0.0
        
        # Clinical relevance of relation types
        clinical_score = np.mean([
            self.clinical_relation_weights.get(rel_type, 0.3) 
            for rel_type in relation_types.keys()
        ])
        
        return (diversity_score * 0.3 + confidence_score * 0.4 + clinical_score * 0.3)
    
    async def _heuristic_relation_evaluation(self, relations: List[Tuple]) -> Tuple[float, float, float]:
        """Heuristic evaluation of relation quality"""
        if not relations:
            return 0.0, 0.0, 0.0
        
        # Estimate precision based on confidence and clinical relevance
        confidences = [rel[4] for rel in relations if rel[4] is not None]
        clinical_relations = sum(1 for rel in relations if rel[3] in self.clinical_relation_weights)
        
        precision = (
            (np.mean(confidences) if confidences else 0.0) * 0.6 +
            (clinical_relations / len(relations)) * 0.4
        )
        
        # Estimate recall based on entity connectivity
        connectivity = await self._calculate_graph_connectivity(relations)
        recall = min(1.0, connectivity)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    async def _calculate_modularity(self, communities: List[Tuple]) -> float:
        """Calculate modularity of community structure"""
        # Simplified modularity calculation
        # In a real implementation, this would use the full graph structure
        if not communities:
            return 0.0
        
        total_edges = 0
        intra_community_edges = 0
        
        for cluster_key, title, nodes_json, occurrence, level in communities:
            nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            community_size = len(nodes)
            
            # Estimate edges within community (heuristic)
            estimated_intra_edges = community_size * (community_size - 1) / 2 * occurrence
            intra_community_edges += estimated_intra_edges
            total_edges += estimated_intra_edges
        
        # Rough modularity estimate
        if total_edges == 0:
            return 0.0
        
        return min(1.0, intra_community_edges / total_edges)
    
    async def _calculate_community_silhouette(self, communities: List[Tuple]) -> float:
        """Calculate silhouette score for communities"""
        # Simplified silhouette calculation
        # Would need full embedding space for proper calculation
        community_sizes = []
        occurrences = []
        
        for cluster_key, title, nodes_json, occurrence, level in communities:
            nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            community_sizes.append(len(nodes))
            occurrences.append(occurrence)
        
        if len(community_sizes) < 2:
            return 0.0
        
        # Heuristic based on size distribution and occurrence scores
        size_variance = np.var(community_sizes)
        occurrence_mean = np.mean(occurrences)
        
        # Good communities have balanced sizes and high occurrence
        size_score = 1.0 / (1.0 + size_variance / np.mean(community_sizes))
        occurrence_score = occurrence_mean
        
        return (size_score + occurrence_score) / 2
    
    async def _calculate_community_coherence(self, communities: List[Tuple]) -> float:
        """Calculate clinical coherence within communities"""
        coherence_scores = []
        
        for cluster_key, title, nodes_json, occurrence, level in communities:
            nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            
            if not nodes:
                continue
            
            # Get entity types in this community
            placeholders = ','.join(['%s'] * len(nodes))
            result = await self.conn.execute(f"""
                SELECT entity_type FROM entities 
                WHERE id IN ({placeholders})
            """, [int(node_id) for node_id in nodes])
            
            entity_types = [row[0] for row in await result.fetchall()]
            
            if not entity_types:
                continue
            
            # Calculate type diversity (lower is more coherent)
            type_counts = Counter(entity_types)
            dominant_type_ratio = max(type_counts.values()) / len(entity_types)
            
            # Weight by clinical relevance
            clinical_weights = [self.clinical_entity_weights.get(et, 0.3) for et in entity_types]
            avg_clinical_weight = np.mean(clinical_weights)
            
            community_coherence = dominant_type_ratio * 0.6 + avg_clinical_weight * 0.4
            coherence_scores.append(community_coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    async def _assess_clinical_clustering_quality(self, communities: List[Tuple]) -> float:
        """Assess how well communities represent clinical concepts"""
        clinical_quality_scores = []
        
        for cluster_key, title, nodes_json, occurrence, level in communities:
            nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            
            if not nodes:
                continue
            
            # Get entities in this community
            placeholders = ','.join(['%s'] * len(nodes))
            result = await self.conn.execute(f"""
                SELECT entity_text, entity_type, normalized_source 
                FROM entities 
                WHERE id IN ({placeholders})
            """, [int(node_id) for node_id in nodes])
            
            entities = await result.fetchall()
            
            if not entities:
                continue
            
            # Assess clinical clustering quality
            entity_types = [e[1] for e in entities]
            normalized_count = sum(1 for e in entities if e[2] in self.authoritative_vocabularies)
            
            # Type coherence
            type_coherence = len(set(entity_types)) / len(entity_types)  # Lower is better
            type_coherence_score = 1.0 - min(type_coherence, 0.8)
            
            # Normalization quality
            normalization_score = normalized_count / len(entities)
            
            # Clinical relevance
            clinical_types = sum(1 for et in entity_types if et in self.clinical_entity_weights)
            clinical_score = clinical_types / len(entity_types)
            
            community_quality = (
                type_coherence_score * 0.4 + 
                normalization_score * 0.3 + 
                clinical_score * 0.3
            )
            clinical_quality_scores.append(community_quality)
        
        return np.mean(clinical_quality_scores) if clinical_quality_scores else 0.0
    
    async def _fetch_top_entities_by_types(self, entity_types: List[str], limit: int = 5) -> List[str]:
        """Fetch top entity texts for the given entity types ordered by confidence."""
        if limit <= 0:
            return []
        if not entity_types:
            result = await self.conn.execute(
                """
                SELECT entity_text
                FROM entities
                WHERE entity_text IS NOT NULL AND entity_text <> ''
                ORDER BY confidence DESC, id ASC
                LIMIT %s
                """,
                (limit,)
            )
        else:
            placeholders = ','.join(['%s'] * len(entity_types))
            query = f"""
                SELECT entity_text
                FROM entities
                WHERE entity_type IN ({placeholders})
                  AND entity_text IS NOT NULL AND entity_text <> ''
                ORDER BY confidence DESC, id ASC
                LIMIT %s
            """
            params = tuple(entity_types) + (limit,)
            result = await self.conn.execute(query, params)
        rows = await result.fetchall()
        seen: Set[str] = set()
        entities: List[str] = []
        for row in rows:
            text = row[0]
            if text and text not in seen:
                entities.append(text)
                seen.add(text)
        return entities

    async def _generate_test_queries(self) -> List[Dict[str, Any]]:
        """Generate test queries for retrieval evaluation"""
        query_definitions = [
            {"query": "adverse events", "entity_types": ["adverse_event"]},
            {"query": "study medication", "entity_types": ["medication", "drug"]},
            {"query": "trial endpoints", "entity_types": ["endpoint"]},
            {"query": "study population", "entity_types": ["population"]},
            {"query": "trial locations", "entity_types": ["location"]},
        ]
        test_queries: List[Dict[str, Any]] = []
        for definition in query_definitions:
            expected_entities = await self._fetch_top_entities_by_types(definition["entity_types"], limit=5)
            if not expected_entities:
                continue
            test_queries.append({
                "query": definition["query"],
                "expected_entities": expected_entities,
                "entity_types": definition["entity_types"]
            })
        if not test_queries:
            fallback_entities = await self._fetch_top_entities_by_types([], limit=5)
            test_queries.append({
                "query": "clinical knowledge",
                "expected_entities": fallback_entities,
                "entity_types": []
            })
        return test_queries
    
    def _calculate_mrr(self, retrieved: List[str], expected: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for rank, item in enumerate(retrieved, 1):
            if item in expected:
                return 1.0 / rank
        return 0.0
    
    def _calculate_ndcg_at_k(self, retrieved: List[str], expected: Set[str], k: int) -> float:
        """Calculate NDCG@k"""
        retrieved_k = retrieved[:k]
        relevance_scores = [1.0 if item in expected else 0.0 for item in retrieved_k]
        
        if not any(relevance_scores):
            return 0.0
        
        # DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # IDCG (perfect ranking)
        ideal_scores = sorted([1.0] * len(expected) + [0.0] * (k - len(expected)), reverse=True)[:k]
        idcg = ideal_scores[0] if ideal_scores else 0.0
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_precision_at_k(self, retrieved: List[str], expected: Set[str], k: int) -> float:
        """Calculate Precision@k"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in expected)
        return relevant_retrieved / len(retrieved_k) if retrieved_k else 0.0
    
    def _calculate_recall_at_k(self, retrieved: List[str], expected: Set[str], k: int) -> float:
        """Calculate Recall@k"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in expected)
        return relevant_retrieved / len(expected) if expected else 0.0
    
    async def _calculate_clinical_relevance_correlation(self) -> float:
        """Calculate correlation between retrieval results and clinical relevance"""
        # Heuristic: higher-weighted entity types should appear more frequently in results
        result = await self.conn.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        entity_type_counts = await result.fetchall()
        
        if not entity_type_counts:
            return 0.0
        
        # Calculate correlation between frequency and clinical weights
        frequencies = []
        weights = []
        
        for entity_type, count in entity_type_counts:
            frequencies.append(count)
            weights.append(self.clinical_entity_weights.get(entity_type, 0.3))
        
        if len(frequencies) < 2:
            return 0.0
        
        # Simple correlation calculation
        correlation = np.corrcoef(frequencies, weights)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_overall_clinical_relevance(
        self, 
        entity_metrics: EntityEvaluationMetrics,
        relation_metrics: RelationEvaluationMetrics,
        community_metrics: CommunityEvaluationMetrics
    ) -> float:
        """Calculate overall clinical relevance score"""
        return (
            entity_metrics.clinical_relevance_score * 0.4 +
            relation_metrics.semantic_coherence * 0.3 +
            community_metrics.clinical_clustering_quality * 0.3
        )
    
    def _calculate_medical_graph_rag_compliance(
        self,
        entity_metrics: EntityEvaluationMetrics,
        relation_metrics: RelationEvaluationMetrics,
        community_metrics: CommunityEvaluationMetrics,
        retrieval_metrics: RetrievalEvaluationMetrics
    ) -> float:
        """Calculate Medical-Graph-RAG compliance score"""
        # Weight different aspects of Medical-Graph-RAG compliance
        entity_score = (entity_metrics.f1_score * 0.5 + entity_metrics.clinical_relevance_score * 0.5)
        relation_score = (relation_metrics.f1_score * 0.5 + relation_metrics.semantic_coherence * 0.5)
        community_score = (community_metrics.modularity * 0.3 + 
                          community_metrics.clinical_clustering_quality * 0.7)
        retrieval_score = (retrieval_metrics.mean_reciprocal_rank * 0.4 + 
                          retrieval_metrics.ndcg_at_k.get(5, 0.0) * 0.6)
        
        return (
            entity_score * 0.25 +
            relation_score * 0.25 +
            community_score * 0.25 +
            retrieval_score * 0.25
        )
    
    async def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information for evaluation report"""
        # Get basic statistics
        result = await self.conn.execute("SELECT COUNT(*) FROM entities")
        total_entities = (await result.fetchone())[0]
        
        result = await self.conn.execute("SELECT COUNT(*) FROM relations")
        total_relations = (await result.fetchone())[0]
        
        result = await self.conn.execute("SELECT COUNT(*) FROM communities")
        total_communities = (await result.fetchone())[0]
        
        result = await self.conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM entities")
        total_chunks = (await result.fetchone())[0]
        
        return {
            'total_entities': total_entities,
            'total_relations': total_relations,
            'total_communities': total_communities,
            'total_chunks': total_chunks,
            'entities_per_chunk': total_entities / max(total_chunks, 1)
        }
    
    # Empty metric objects for error cases
    
    def _empty_entity_metrics(self) -> EntityEvaluationMetrics:
        return EntityEvaluationMetrics(
            precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
            total_extracted=0, total_expected=0, true_positives=0,
            false_positives=0, false_negatives=0, entity_type_breakdown={},
            coverage_rate=0.0, clinical_relevance_score=0.0
        )
    
    def _empty_relation_metrics(self) -> RelationEvaluationMetrics:
        return RelationEvaluationMetrics(
            precision=0.0, recall=0.0, f1_score=0.0, total_extracted=0,
            total_expected=0, relation_type_breakdown={}, graph_connectivity=0.0,
            semantic_coherence=0.0
        )
    
    def _empty_community_metrics(self) -> CommunityEvaluationMetrics:
        return CommunityEvaluationMetrics(
            modularity=0.0, silhouette_score=0.0, num_communities=0,
            average_community_size=0.0, community_coherence=0.0,
            coverage=0.0, clinical_clustering_quality=0.0
        )


# Convenience function for easy evaluation
async def evaluate_clinical_knowledge_graph(
    connection_string: str,
    output_file: Optional[str] = None
) -> ComprehensiveEvaluationReport:
    """
    Convenience function to run complete evaluation.
    
    Args:
        connection_string: Database connection string
        output_file: Optional file to save evaluation report
        
    Returns:
        ComprehensiveEvaluationReport
    """
    evaluator = ClinicalEvaluationFramework(connection_string)
    report = await evaluator.comprehensive_evaluation()
    
    if output_file:
        # Save report to file
        import json
        from dataclasses import asdict
        
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_file}")
    
    return report