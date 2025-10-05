"""
Enhanced Clinical Entity Extraction with Normalization

Integrates entity extraction with UMLS/SNOMED/RxNorm normalization
for standardized clinical knowledge graph construction.
"""

import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from uuid import UUID, uuid5

from docintel.repository.constants import NODE_NAMESPACE

from .triple_extraction import ClinicalTripleExtractor, ClinicalEntity, ClinicalRelation, TripleExtractionResult
from .entity_normalization import (
    ClinicalEntityNormalizer,
    EntityNormalizationResult,
    NormalizedEntity,
    ClinicalVocabulary,
)

logger = logging.getLogger(__name__)

def _normalized_entity_to_dict(normalized: NormalizedEntity) -> Dict[str, Any]:
    """Convert NormalizedEntity into JSON-serialisable dict."""
    payload = asdict(normalized)
    vocab = normalized.vocabulary.value if isinstance(normalized.vocabulary, ClinicalVocabulary) else normalized.vocabulary
    payload["vocabulary"] = vocab
    return payload


@dataclass
class EnhancedClinicalEntity(ClinicalEntity):
    """Clinical entity enriched with normalization and repository data."""
    normalization_data: Optional[Dict[str, Any]] = None
    repository_node_id: Optional[str] = None
    repository_vocabulary: Optional[str] = None
    repository_code: Optional[str] = None

    @classmethod
    def from_clinical_entity(
        cls,
        entity: ClinicalEntity,
        normalization_result: Optional[EntityNormalizationResult] = None,
    ):
        """Create enhanced entity from base entity and normalization result."""
        normalization_data = None
        normalized_id = entity.normalized_id
        normalized_source = entity.normalized_source

        if normalization_result and normalization_result.best_match:
            best_match = normalization_result.best_match
            normalized_id = best_match.concept_id
            normalized_source = best_match.vocabulary.value
            normalization_data = {
                "best_match": _normalized_entity_to_dict(best_match),
                "all_matches": [_normalized_entity_to_dict(norm) for norm in normalization_result.normalizations],
                "metadata": normalization_result.processing_metadata,
            }

        repository_node_id, repository_code = _derive_repository_link(normalized_source, normalized_id)
        if normalization_data is not None:
            normalization_data.setdefault("repository", {})
            normalization_data["repository"].update(
                {
                    "repository_node_id": repository_node_id,
                    "vocabulary": normalized_source,
                    "code": repository_code,
                }
            )

        # Propagate normalization context back to the original entity so downstream
        # builders (e.g., KnowledgeGraphBuilder) can persist the metadata without
        # depending on the enhanced wrapper type.
        entity.normalized_id = normalized_id
        entity.normalized_source = normalized_source
        entity.normalization_data = normalization_data
        if repository_node_id:
            entity.repository_node_id = repository_node_id
            entity.repository_vocabulary = normalized_source
            entity.repository_code = repository_code

        return cls(
            text=entity.text,
            entity_type=entity.entity_type,
            start_char=entity.start_char,
            end_char=entity.end_char,
            confidence=entity.confidence,
            normalized_id=normalized_id,
            normalized_source=normalized_source,
            context_flags=entity.context_flags,
            normalization_data=normalization_data,
            repository_node_id=repository_node_id,
            repository_vocabulary=normalized_source,
            repository_code=repository_code,
        )


@dataclass
class EnhancedTripleExtractionResult(TripleExtractionResult):
    """Triple extraction result with normalized entities"""
    normalized_entities: List[EnhancedClinicalEntity]
    normalization_stats: Dict[str, Any]
    
    @classmethod
    def from_base_result(cls, base_result: TripleExtractionResult, normalized_entities: List[EnhancedClinicalEntity], normalization_stats: Dict[str, Any]):
        """Create enhanced result from base result"""
        return cls(
            entities=base_result.entities,
            relations=base_result.relations,
            processing_metadata=base_result.processing_metadata,
            normalized_entities=normalized_entities,
            normalization_stats=normalization_stats
        )


class EnhancedClinicalTripleExtractor:
    """
    Enhanced clinical triple extractor with integrated entity normalization.
    
    Combines entity/relation extraction with UMLS/SNOMED/RxNorm normalization
    for standardized clinical knowledge graph construction.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/vocabulary_cache",
        *,
        enable_scispacy: bool = True,
        max_candidates: int = 5,
        fast_mode: bool = False,
        skip_relations: bool = False,
    ):
        self.base_extractor = ClinicalTripleExtractor(
            fast_mode=fast_mode,
            skip_relations=skip_relations
        )
        self.normalizer = ClinicalEntityNormalizer(
            cache_dir,
            enable_scispacy=enable_scispacy,
            max_candidates=max_candidates,
        )
        
    async def extract_and_normalize_triples(self, text: str, chunk_id: UUID) -> EnhancedTripleExtractionResult:
        """
        Extract entities/relations and normalize entities to clinical vocabularies.
        
        Args:
            text: Input clinical text
            chunk_id: UUID of the source chunk
            
        Returns:
            EnhancedTripleExtractionResult with normalized entities
        """
        logger.info(f"Extracting and normalizing entities from chunk {chunk_id}")
        
        # Step 1: Extract base entities and relations
        base_result = self.base_extractor.extract_triples(text, chunk_id)
        logger.info(f"Extracted {len(base_result.entities)} entities and {len(base_result.relations)} relations")
        
        # Step 2: Normalize entities
        if base_result.entities:
            normalized_entities, normalization_stats = await self._normalize_entities(base_result.entities)
        else:
            normalized_entities = []
            normalization_stats = {"total_entities": 0, "normalized_entities": 0, "normalization_rate": 0.0}
        
        logger.info(f"Normalized {normalization_stats.get('normalized_entities', 0)}/{len(base_result.entities)} entities")
        
        # Step 3: Create enhanced result
        enhanced_result = EnhancedTripleExtractionResult.from_base_result(
            base_result, normalized_entities, normalization_stats
        )
        
        return enhanced_result
    
    async def _normalize_entities(self, entities: List[ClinicalEntity]) -> Tuple[List[EnhancedClinicalEntity], Dict[str, Any]]:
        """Normalize extracted entities to clinical vocabularies"""
        logger.info(f"Normalizing {len(entities)} entities...")
        
        # Prepare entities for normalization
        entity_tuples = [(entity.text, entity.entity_type) for entity in entities]
        
        # Batch normalize entities
        normalization_results = await self.normalizer.normalize_entities_batch(entity_tuples)
        
        # Create enhanced entities
        normalized_entities = []
        normalization_success_count = 0
        
        for i, entity in enumerate(entities):
            normalization_result = normalization_results[i] if i < len(normalization_results) else None
            
            enhanced_entity = EnhancedClinicalEntity.from_clinical_entity(entity, normalization_result)
            normalized_entities.append(enhanced_entity)
            
            if normalization_result and normalization_result.best_match:
                normalization_success_count += 1
        
        # Calculate normalization statistics
        normalization_stats = {
            "total_entities": len(entities),
            "normalized_entities": normalization_success_count,
            "normalization_rate": normalization_success_count / len(entities) if entities else 0.0,
            "vocabularies_used": self._get_vocabulary_usage_stats(normalization_results),
            "scispacy_enabled": getattr(self.normalizer, "scispacy_enabled", False),
        }
        
        return normalized_entities, normalization_stats
    
    def _get_vocabulary_usage_stats(self, normalization_results: List[EntityNormalizationResult]) -> Dict[str, int]:
        """Get statistics about which vocabularies were used"""
        vocab_usage = {}
        
        for result in normalization_results:
            if result.best_match:
                vocab = result.best_match.vocabulary.value
                vocab_usage[vocab] = vocab_usage.get(vocab, 0) + 1
        
        return vocab_usage
    
    async def process_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[EnhancedTripleExtractionResult]:
        """
        Process multiple document chunks with entity extraction and normalization.
        
        Args:
            chunks: List of document chunks with 'text' and 'id' keys
            
        Returns:
            List of EnhancedTripleExtractionResult
        """
        logger.info(f"Processing {len(chunks)} document chunks with normalization...")
        
        results = []
        total_entities = 0
        total_normalized = 0
        
        for chunk in chunks:
            try:
                chunk_id = UUID(chunk['id']) if isinstance(chunk['id'], str) else chunk['id']
                result = await self.extract_and_normalize_triples(chunk['text'], chunk_id)
                results.append(result)
                
                total_entities += result.normalization_stats['total_entities']
                total_normalized += result.normalization_stats['normalized_entities']
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
                continue
        
        overall_normalization_rate = total_normalized / total_entities if total_entities > 0 else 0.0
        
        logger.info(f"Processed {len(results)} chunks successfully")
        logger.info(f"Overall normalization rate: {overall_normalization_rate:.1%} ({total_normalized}/{total_entities})")
        
        return results

def _derive_repository_link(vocabulary: Optional[str], concept_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return repository node identifier and normalized code for a concept."""
    if not vocabulary or not concept_id:
        return None, None
    vocab_slug = vocabulary.strip().lower()
    code = concept_id.strip()
    if not vocab_slug or not code:
        return None, None
    prefix = f"{vocab_slug}:"
    if code.lower().startswith(prefix):
        code = code[len(prefix):]
    code = code.strip()
    if not code:
        return None, None
    code_normalized = code.upper() if any(ch.isalpha() for ch in code) else code
    repo_uuid = uuid5(NODE_NAMESPACE, f"{vocab_slug}:{code_normalized}")
    return str(repo_uuid), code_normalized


async def extract_and_normalize_clinical_data(text: str, chunk_id: UUID, cache_dir: str = "./data/vocabulary_cache") -> EnhancedTripleExtractionResult:
    """
    Convenience function for extracting and normalizing clinical data.
    
    Args:
        text: Clinical text to process
        chunk_id: UUID of the source chunk
        cache_dir: Directory for vocabulary cache
        
    Returns:
        EnhancedTripleExtractionResult with normalized entities
    """
    extractor = EnhancedClinicalTripleExtractor(cache_dir)
    return await extractor.extract_and_normalize_triples(text, chunk_id)


# Demo/test function removed - use knowledge_graph_cli.py for real clinical data processing