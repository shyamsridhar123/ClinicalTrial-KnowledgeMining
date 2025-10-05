"""
Clinical Triple Extraction Module

Extracts entities and relationships from clinical text using Azure OpenAI GPT-4.1
and persists them as knowledge graph triples in PostgreSQL + AGE.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

from openai import AzureOpenAI
import spacy
from spacy import displacy

from docintel.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class ClinicalEntity:
    """Represents a clinical entity extracted from text."""
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_id: Optional[str] = None
    normalized_source: Optional[str] = None
    context_flags: Optional[Dict[str, Any]] = None

@dataclass 
class ClinicalRelation:
    """Represents a relationship between two clinical entities."""
    subject_entity: ClinicalEntity
    predicate: str
    object_entity: ClinicalEntity
    confidence: float
    evidence_span: str
    evidence_start_char: int
    evidence_end_char: int

@dataclass
class TripleExtractionResult:
    """Results from triple extraction on a text chunk."""
    entities: List[ClinicalEntity]
    relations: List[ClinicalRelation]
    processing_metadata: Dict[str, Any]

class ClinicalTripleExtractor:
    """Extracts clinical entities and relationships using Azure OpenAI GPT-4.1."""
    
    def __init__(self, fast_mode: bool = False, skip_relations: bool = False):
        config = get_config()
        
        self.fast_mode = fast_mode
        self.skip_relations = skip_relations
        
        # Initialize Azure OpenAI client (skip if fast mode)
        if not fast_mode:
            self.client = AzureOpenAI(
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                azure_endpoint=config.azure_openai_endpoint
            )
            self.deployment_name = config.azure_openai_deployment_name
        else:
            self.client = None
            self.deployment_name = None
            logger.info("⚡ Fast mode enabled - skipping Azure OpenAI initialization")
        
        # Load spaCy model for basic NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy en_core_web_sm model not found. Some features may be limited.")
            self.nlp = None
    
    def extract_triples(self, text: str, chunk_id: UUID) -> TripleExtractionResult:
        """
        Extract clinical entities and relationships from text.
        
        Args:
            text: Input clinical text
            chunk_id: UUID of the source chunk
            
        Returns:
            TripleExtractionResult containing entities and relations
        """
        logger.info(f"Extracting triples from chunk {chunk_id} (length: {len(text)})")
        
        try:
            # Handle very large texts by splitting into manageable sections
            if len(text) > 50000:  # Split documents larger than 50k chars
                logger.info(f"Large document ({len(text)} chars), processing in sections")
                return self._extract_from_large_text(text, chunk_id)
            
            # FAST MODE: Use only medspaCy/spaCy without GPT
            if self.fast_mode:
                logger.info("⚡ Using fast extraction (medspaCy only)")
                entities = self._extract_entities_fast(text)
                logger.info(f"Extracted {len(entities)} entities (fast mode)")
                relations = []  # No relations in fast mode
            else:
                # Step 1: Extract entities using GPT-4.1
                entities = self._extract_entities(text)
                logger.info(f"Extracted {len(entities)} entities")
                
                # Step 2: Extract clinical context using medspaCy BEFORE relations
                # This allows relation extraction to be context-aware
                context_extractor = MedSpaCyContextExtractor()
                entities = context_extractor.extract_context(text, entities)
                logger.info(f"Applied clinical context to {len(entities)} entities")
                
                # Step 3: Extract relationships using GPT-4.1 with context-aware entities (unless skipped)
                if self.skip_relations:
                    logger.info("Skipping relation extraction (--skip-relations)")
                    relations = []
                else:
                    relations = self._extract_relations(text, entities)
                    logger.info(f"Extracted {len(relations)} relations")
            
            # Step 4: Apply post-processing and validation
            entities, relations = self._post_process_extractions(text, entities, relations)
            
            processing_metadata = {
                "chunk_id": str(chunk_id),
                "text_length": len(text),
                "extraction_model": self.deployment_name,
                "entity_count": len(entities),
                "relation_count": len(relations)
            }
            
            return TripleExtractionResult(
                entities=entities,
                relations=relations,
                processing_metadata=processing_metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting triples from chunk {chunk_id}: {e}")
            return TripleExtractionResult(
                entities=[],
                relations=[],
                processing_metadata={"error": str(e), "chunk_id": str(chunk_id)}
            )
    
    def _extract_from_large_text(self, text: str, chunk_id: UUID) -> TripleExtractionResult:
        """Process large texts in smaller sections to avoid GPT-4.1 hanging."""
        max_section_size = 40000  # ~40k chars per section
        overlap = 1000  # Overlap to catch cross-boundary entities
        
        all_entities = []
        all_relations = []
        section_count = 0
        
        start = 0
        while start < len(text):
            end = min(start + max_section_size, len(text))
            
            # Find good break point (sentence boundary)
            if end < len(text):
                for break_char in ['. ', '.\n', '!\n', '?\n']:
                    break_pos = text.rfind(break_char, start + max_section_size - 2000, end)
                    if break_pos != -1:
                        end = break_pos + len(break_char)
                        break
            
            section_text = text[start:end].strip()
            if len(section_text) < 100:  # Skip tiny sections
                start = end
                continue
                
            section_count += 1
            logger.info(f"Processing section {section_count}: chars {start}-{end} ({len(section_text)} chars)")
            
            try:
                # Extract from this section
                entities = self._extract_entities(section_text)
                relations = self._extract_relations(section_text, entities)
                
                # Adjust entity positions to match original text
                for entity in entities:
                    entity.start_char += start
                    entity.end_char += start
                
                # Adjust relation evidence positions  
                for relation in relations:
                    relation.evidence_start_char += start
                    relation.evidence_end_char += start
                
                all_entities.extend(entities)
                all_relations.extend(relations)
                
                logger.info(f"Section {section_count}: {len(entities)} entities, {len(relations)} relations")
                
            except Exception as e:
                logger.warning(f"Error in section {section_count}: {e}")
                continue
            
            # Move to next section with overlap
            start = max(end - overlap, start + 1)
        
        # Remove duplicate entities
        unique_entities = []
        seen = set()
        for entity in all_entities:
            key = (entity.text.lower().strip(), entity.entity_type, entity.start_char)
            if key not in seen:
                unique_entities.append(entity)
                seen.add(key)
        
        # Apply post-processing
        entities, relations = self._post_process_extractions(text, unique_entities, all_relations)
        
        processing_metadata = {
            "chunk_id": str(chunk_id),
            "text_length": len(text),
            "extraction_model": self.deployment_name,
            "sections_processed": section_count,
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
        
        return TripleExtractionResult(
            entities=entities,
            relations=relations,
            processing_metadata=processing_metadata
        )
    
    def _extract_entities(self, text: str) -> List[ClinicalEntity]:
        """Extract clinical entities using GPT-4.1."""
        
        entity_prompt = """
You are an advanced clinical trial information extraction system. Extract ALL relevant clinical entities from the following text with comprehensive coverage.

ENTITY TYPES (extract ALL applicable):

MEDICAL ENTITIES:
- condition: diseases, symptoms, disorders, syndromes, medical conditions
- medication: DRUGS AND PHARMACEUTICALS ONLY. Examples: niraparib, pembrolizumab, aspirin, chemotherapy agents, PARP inhibitors, immunotherapy drugs, olaparib, rucaparib, bevacizumab, carboplatin, cisplatin, paclitaxel. Common suffixes: -ib, -mab, -tinib, -parin, -nazole, -statin, -prazole, -olol, -dipine, -mycin, -cillin, -vir. DO NOT confuse drug names with people receiving medication.
- procedure: surgeries, interventions, diagnostic tests, medical procedures, examinations
- device: medical devices, equipment, instruments, implants, diagnostic tools
- measurement: lab values, vital signs, scores, biomarkers, clinical measurements
- adverse_event: side effects, adverse reactions, safety events, complications

CLINICAL TRIAL ENTITIES:
- population: patient groups, demographics, inclusion criteria, study participants
- endpoint: primary outcomes, secondary outcomes, efficacy measures, safety measures
- dosage: drug doses, administration routes, frequencies, treatment regimens
- timepoint: study durations, follow-up periods, treatment schedules, visit windows
- protocol: study protocols, procedures, methodologies, trial phases
- location: study sites, geographical regions, institutions, clinical centers

ORGANIZATIONAL ENTITIES:
- organization: pharmaceutical companies, research institutions, regulatory bodies, sponsors
- person: HUMAN INDIVIDUALS ONLY. Examples: Dr. Smith, Principal Investigator Jane Doe, investigators with credentials (MD, PhD), named study personnel. DO NOT classify drug names, treatments, or therapies as persons.
- publication: journal articles, guidelines, references, regulatory documents
- identifier: NCT numbers, protocol IDs, study codes, regulatory identifiers

STATISTICAL/ANALYTICAL:
- statistic: p-values, confidence intervals, statistical tests, effect sizes
- sample_size: participant numbers, enrollment targets, statistical power calculations

REGULATORY/ADMINISTRATIVE:
- regulation: FDA approvals, regulatory requirements, compliance standards
- ethics: IRB approvals, informed consent, ethical considerations

DISAMBIGUATION RULES (CRITICAL):
1. If entity is being "administered", "prescribed", "dosed", "given" → medication
2. If entity has drug name patterns (ends in -ib, -mab, -tinib, -parin, etc.) → medication
3. If entity is doing an action or has credentials (Dr., PhD, MD, investigator) → person
4. If entity is a patient receiving treatment → population (NOT person)
5. Generic drug names (aspirin, ibuprofen) AND brand names (Keytruda, Lynparza) → medication
6. If uncertain between medication and person, default to medication for pharmaceutical terms

Extract entities comprehensively to achieve 15-20% coverage of all clinical concepts in the text.

For each entity, provide:
- text: exact text span (be precise, avoid over-broad spans)
- entity_type: one of the types listed above
- start_char: character start position  
- end_char: character end position
- confidence: confidence score (0.0-1.0)

IMPORTANT: Be thorough but precise. Extract specific medical terms, drug names, condition names, measurements, and clinical concepts. Avoid generic words unless they represent specific clinical entities.

Text to analyze:
{text}

Response (JSON array only):
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise clinical entity extraction system. Always respond with valid JSON."},
                    {"role": "user", "content": entity_prompt.format(text=text)}
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            logger.debug(f"Raw GPT-4.1 response: {content[:500]}...")
            
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            # Clean up common JSON issues
            content = content.replace('\n', ' ').replace('\r', ' ')
            
            try:
                entities_data = json.loads(content)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Problematic content: {content[:1000]}")
                return []
            
            # Convert to ClinicalEntity objects
            entities = []
            for entity_data in entities_data:
                entity = ClinicalEntity(
                    text=entity_data["text"],
                    entity_type=entity_data["entity_type"],
                    start_char=entity_data["start_char"],
                    end_char=entity_data["end_char"],
                    confidence=entity_data["confidence"]
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_entities_fast(self, text: str) -> List[ClinicalEntity]:
        """Fast entity extraction using only spaCy/medspaCy (no GPT calls)."""
        entities = []
        
        if not self.nlp:
            logger.warning("spaCy not available for fast extraction")
            return entities
        
        try:
            # Use spaCy's built-in NER
            doc = self.nlp(text)
            
            # Extract entities from spaCy
            for ent in doc.ents:
                # Map spaCy entity types to our clinical types
                entity_type_mapping = {
                    "DISEASE": "condition",
                    "SYMPTOM": "condition", 
                    "CHEMICAL": "medication",
                    "DRUG": "medication",
                    "MEDICATION": "medication",  # medspaCy label
                    "TREATMENT": "procedure",
                    "TEST": "procedure",
                    "ORG": "organization",
                    "PERSON": "other",  # Changed: Don't auto-trust PERSON from general spaCy
                    "DATE": "timepoint",
                    "CARDINAL": "measurement",
                    "QUANTITY": "measurement",
                    "PERCENT": "statistic",
                }
                
                entity_type = entity_type_mapping.get(ent.label_, "other")
                
                # Defensive check: If spaCy labeled as PERSON but has drug suffix, override
                if ent.label_ == "PERSON" and self._is_likely_drug_name(ent.text):
                    entity_type = "medication"
                    logger.debug(f"Override: '{ent.text}' PERSON → medication (drug pattern detected)")
                
                entities.append(ClinicalEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.7,  # Fixed confidence for spaCy entities
                    normalized_id=None,
                    normalized_source=None,
                    context_flags=None
                ))
            
            logger.info(f"⚡ Fast extraction: extracted {len(entities)} entities from spaCy")
            
        except Exception as e:
            logger.error(f"Error in fast entity extraction: {e}")
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalRelation]:
        """Extract relationships between entities using GPT-4.1."""
        
        if len(entities) < 2:
            return []
        
        # Create entity reference for the prompt with context flags
        entity_refs = []
        for i, entity in enumerate(entities):
            # Add context flags to help GPT avoid inappropriate relations
            context_note = ""
            if entity.context_flags:
                flags = []
                if entity.context_flags.get('is_negated'):
                    flags.append("NEGATED")
                if entity.context_flags.get('is_historical'):
                    flags.append("HISTORICAL")
                if entity.context_flags.get('is_hypothetical'):
                    flags.append("HYPOTHETICAL")
                if entity.context_flags.get('is_uncertain'):
                    flags.append("UNCERTAIN")
                if entity.context_flags.get('is_family'):
                    flags.append("FAMILY_HISTORY")
                if flags:
                    context_note = f" [{', '.join(flags)}]"
            
            entity_refs.append(f"E{i}: {entity.text} ({entity.entity_type}){context_note}")
        
        relation_prompt = """
You are an advanced clinical trial relationship extraction system. Given the text and entities, identify ALL meaningful relationships to achieve comprehensive knowledge graph coverage.

IMPORTANT CONTEXT RULES:
- NEGATED entities: absence/denial (e.g., "no evidence of toxicity"). Do NOT create causal relationships with negated findings.
- HISTORICAL entities: past events (e.g., "history of cancer"). May create context relations but not current causal relations.
- HYPOTHETICAL entities: conditional scenarios (e.g., "if toxicity occurs"). Do NOT create definite causal relations.
- UNCERTAIN entities: possible/suspected findings (e.g., "possible progression"). Use lower confidence or avoid strong causal claims.
- FAMILY_HISTORY entities: about family members, not the patient. Do NOT create patient-specific relations.

Entities:
{entities}

RELATIONSHIP TYPES (extract ALL applicable):

TREATMENT RELATIONSHIPS:
- treats: medication/procedure treats condition
- prevents: intervention prevents condition/adverse_event
- causes: condition/medication causes adverse_event/symptom
- administered_with: medications given together
- contraindicated_with: treatments that should not be combined

MEASUREMENT RELATIONSHIPS:
- measured_by: condition/endpoint measured by procedure/device
- indicates: measurement/test indicates condition/outcome
- monitors: device/procedure monitors condition/parameter
- assesses: endpoint/procedure assesses condition/outcome

STUDY RELATIONSHIPS:
- enrolls: study/protocol enrolls population
- randomized_to: population randomized to medication/procedure
- compared_with: treatment compared with control/placebo
- evaluated_in: medication/device evaluated in population/condition
- conducted_at: study conducted at organization/location

TEMPORAL RELATIONSHIPS:
- followed_by: procedure followed by measurement/outcome
- occurs_during: event occurs during timepoint/procedure
- administered_at: medication given at timepoint/dosage

CAUSAL RELATIONSHIPS:
- associated_with: entity associated with outcome/condition
- correlates_with: measurement correlates with outcome
- predicts: biomarker/test predicts outcome/condition

REGULATORY RELATIONSHIPS:
- approved_for: medication approved for condition
- sponsored_by: study sponsored by organization
- conducted_by: study conducted by person/organization
- published_in: study published in publication

For each relationship, provide:
- subject_id: entity ID (e.g., "E0")
- predicate: relationship type from above list
- object_id: entity ID (e.g., "E1") 
- confidence: confidence score (0.7-1.0 for clear relationships)
- evidence_span: supporting text (keep concise but specific)
- evidence_start_char: start position in original text
- evidence_end_char: end position in original text

**CRITICAL EXTRACTION REQUIREMENTS:**
1. Extract relationships AGGRESSIVELY - aim for AT LEAST 1 relationship per 3-4 entities
2. Look for IMPLICIT relationships, not just explicit statements
3. Connect entities across semantic distance (e.g., drug mentioned in methods → outcome in results)
4. Include temporal sequences, study design relationships, measurement chains
5. A well-connected knowledge graph should have 25-40% relationship density

**QUALITY TARGETS:**
- For 50 entities → expect 12-20 relationships minimum
- For 100 entities → expect 25-40 relationships minimum
- Prioritize high-confidence (0.8+) but don't skip medium-confidence (0.7+) relationships

Extract relationships comprehensively to build a rich, highly-connected clinical knowledge graph.

Text:
{text}

Response (JSON only):
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise clinical relationship extraction system. Always respond with valid JSON."},
                    {"role": "user", "content": relation_prompt.format(
                        entities="\n".join(entity_refs),
                        text=text
                    )}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
                
            relations_data = json.loads(content)
            
            # Convert to ClinicalRelation objects
            relations = []
            for rel_data in relations_data:
                try:
                    subject_idx = int(rel_data["subject_id"][1:])  # Remove 'E' prefix
                    object_idx = int(rel_data["object_id"][1:])
                    
                    if subject_idx < len(entities) and object_idx < len(entities):
                        relation = ClinicalRelation(
                            subject_entity=entities[subject_idx],
                            predicate=rel_data["predicate"],
                            object_entity=entities[object_idx],
                            confidence=rel_data["confidence"],
                            evidence_span=rel_data["evidence_span"],
                            evidence_start_char=rel_data["evidence_start_char"],
                            evidence_end_char=rel_data["evidence_end_char"]
                        )
                        relations.append(relation)
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed relation: {e}")
                    continue
            
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def _post_process_extractions(self, text: str, entities: List[ClinicalEntity], 
                                 relations: List[ClinicalRelation]) -> Tuple[List[ClinicalEntity], List[ClinicalRelation]]:
        """Post-process and validate extractions."""
        
        # Validate entity positions with fuzzy matching for GPT-4.1 alignment issues
        validated_entities = []
        for entity in entities:
            # Try exact position first
            if (0 <= entity.start_char < entity.end_char <= len(text) and
                text[entity.start_char:entity.end_char].strip() == entity.text.strip()):
                validated_entities.append(entity)
            else:
                # Try aggressive fuzzy matching for GPT-4.1 position alignment issues
                entity_found = False
                entity_text_clean = entity.text.strip()
                
                # Try multiple search strategies
                search_strategies = [
                    # Strategy 1: Small window around expected position
                    (max(0, entity.start_char - 10), min(len(text), entity.end_char + 10)),
                    # Strategy 2: Larger window 
                    (max(0, entity.start_char - 50), min(len(text), entity.end_char + 50)),
                    # Strategy 3: Global search if entity is short enough
                    (0, len(text)) if len(entity_text_clean) < 100 else (entity.start_char, entity.end_char)
                ]
                
                for search_start, search_end in search_strategies:
                    if entity_found:
                        break
                        
                    # Try exact match first
                    correct_start = text.find(entity_text_clean, search_start)
                    if correct_start != -1 and correct_start < search_end:
                        correct_end = correct_start + len(entity_text_clean)
                        entity.start_char = correct_start
                        entity.end_char = correct_end
                        validated_entities.append(entity)
                        entity_found = True
                        logger.debug(f"Fixed position for entity: {entity.text} -> {correct_start}-{correct_end}")
                        break
                    
                    # Try case-insensitive match
                    text_lower = text[search_start:search_end].lower()
                    entity_lower = entity_text_clean.lower()
                    relative_pos = text_lower.find(entity_lower)
                    if relative_pos != -1:
                        correct_start = search_start + relative_pos
                        correct_end = correct_start + len(entity_text_clean)
                        entity.start_char = correct_start
                        entity.end_char = correct_end
                        validated_entities.append(entity)
                        entity_found = True
                        logger.debug(f"Fixed position (case-insensitive) for entity: {entity.text} -> {correct_start}-{correct_end}")
                        break
                
                if not entity_found:
                    logger.warning(f"Invalid entity position: {entity.text} at {entity.start_char}-{entity.end_char}")
        
        # Validate relations based on validated entities
        validated_relations = []
        for relation in relations:
            if (relation.subject_entity in validated_entities and 
                relation.object_entity in validated_entities and
                0 <= relation.evidence_start_char < relation.evidence_end_char <= len(text)):
                validated_relations.append(relation)
            else:
                logger.warning(f"Invalid relation: {relation.predicate}")
        
        return validated_entities, validated_relations
    
    def extract_triples_batch(self, chunks: List[Dict[str, Any]]) -> List[TripleExtractionResult]:
        """
        Extract entities and relations from multiple chunks in a single batched GPT call.
        
        Args:
            chunks: List of dicts with 'text', 'chunk_id', 'chunk_uuid' keys
            
        Returns:
            List of TripleExtractionResult, one per chunk
        """
        if not chunks:
            return []
        
        if self.fast_mode:
            # Fast mode: process individually (no batching benefit for local spaCy)
            logger.info(f"⚡ Fast mode: processing {len(chunks)} chunks individually")
            return [self.extract_triples(chunk['text'], chunk['chunk_uuid']) for chunk in chunks]
        
        logger.info(f"🚀 Batch extracting from {len(chunks)} chunks")
        
        # Prepare batch input
        batch_input = []
        for i, chunk in enumerate(chunks):
            batch_input.append({
                "chunk_index": i,
                "chunk_id": str(chunk['chunk_uuid']),
                "text": chunk['text'][:10000]  # Limit each chunk to 10k chars for batching
            })
        
        # Step 1: Batch entity extraction
        all_entities = self._extract_entities_batch(batch_input)
        
        # Step 2: Batch relation extraction (if not skipped)
        if self.skip_relations:
            logger.info("Skipping batch relation extraction (--skip-relations)")
            all_relations = [[] for _ in chunks]
        else:
            all_relations = self._extract_relations_batch(batch_input, all_entities)
        
        # Step 3: Post-process and add context for each chunk
        results = []
        context_extractor = MedSpaCyContextExtractor()
        
        for i, chunk in enumerate(chunks):
            entities = all_entities[i]
            relations = all_relations[i]
            
            # Post-process
            entities, relations = self._post_process_extractions(chunk['text'], entities, relations)
            
            # Add clinical context
            entities = context_extractor.extract_context(chunk['text'], entities)
            
            processing_metadata = {
                "chunk_id": str(chunk['chunk_uuid']),
                "text_length": len(chunk['text']),
                "extraction_model": self.deployment_name,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "batched": True
            }
            
            results.append(TripleExtractionResult(
                entities=entities,
                relations=relations,
                processing_metadata=processing_metadata
            ))
        
        logger.info(f"✅ Batch extracted {sum(len(r.entities) for r in results)} entities, "
                   f"{sum(len(r.relations) for r in results)} relations from {len(chunks)} chunks")
        
        return results
    
    def _extract_entities_batch(self, batch_input: List[Dict[str, Any]]) -> List[List[ClinicalEntity]]:
        """Extract entities from multiple chunks in one GPT call."""
        
        batch_prompt = """
You are extracting clinical entities from multiple text chunks. For each chunk, extract ALL relevant entities.

ENTITY TYPES: condition, medication, procedure, device, measurement, adverse_event, population, endpoint, dosage, timepoint, protocol, location, organization, person, publication, identifier, statistic, sample_size, regulation, ethics

"""
        
        # Add each chunk to the prompt
        for item in batch_input:
            batch_prompt += f"\n\n=== CHUNK {item['chunk_index']} (ID: {item['chunk_id']}) ===\n{item['text']}\n"
        
        batch_prompt += """

Respond with a JSON object where keys are chunk indices (0, 1, 2...) and values are arrays of entities:

{
  "0": [{"text": "...", "entity_type": "...", "start_char": 0, "end_char": 10, "confidence": 0.95}, ...],
  "1": [{"text": "...", "entity_type": "...", "start_char": 0, "end_char": 10, "confidence": 0.95}, ...],
  ...
}

Response (JSON object only):
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise clinical entity extraction system. Always respond with valid JSON."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Batch entity response: {content[:500]}...")
            
            # Clean JSON
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            batch_data = json.loads(content)
            
            # Convert to ClinicalEntity objects per chunk
            all_entities = []
            for i in range(len(batch_input)):
                chunk_entities = []
                entities_data = batch_data.get(str(i), [])
                
                for entity_data in entities_data:
                    try:
                        chunk_entities.append(ClinicalEntity(
                            text=entity_data["text"],
                            entity_type=entity_data["entity_type"],
                            start_char=entity_data.get("start_char", 0),
                            end_char=entity_data.get("end_char", len(entity_data["text"])),
                            confidence=entity_data.get("confidence", 0.8)
                        ))
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid entity in chunk {i}: {e}")
                        continue
                
                all_entities.append(chunk_entities)
                logger.info(f"Chunk {i}: extracted {len(chunk_entities)} entities")
            
            return all_entities
            
        except Exception as e:
            logger.error(f"Batch entity extraction failed: {e}")
            # Fallback: return empty lists for each chunk
            return [[] for _ in batch_input]
    
    def _extract_relations_batch(self, batch_input: List[Dict[str, Any]], 
                                 all_entities: List[List[ClinicalEntity]]) -> List[List[ClinicalRelation]]:
        """Extract relations from multiple chunks in one GPT call."""
        
        batch_prompt = """
You are extracting relationships between clinical entities in multiple text chunks.

"""
        
        # Add each chunk with its entities
        for i, item in enumerate(batch_input):
            entities = all_entities[i]
            entity_list = ", ".join([f'"{e.text}" ({e.entity_type})' for e in entities[:50]])  # Limit to first 50
            
            batch_prompt += f"\n\n=== CHUNK {i} (ID: {item['chunk_id']}) ===\n"
            batch_prompt += f"TEXT: {item['text']}\n"
            batch_prompt += f"ENTITIES: {entity_list}\n"
        
        batch_prompt += """

For each chunk, extract relationships between entities. Use these predicates:
TREATS, CAUSES, PREVENTS, DIAGNOSES, ADMINISTERED_TO, HAS_DOSAGE, HAS_ADVERSE_EVENT, MEASURED_BY, DEFINED_BY, CONDUCTED_AT, INVESTIGATES, HAS_OUTCOME, REQUIRES, EXCLUDES

Respond with JSON object where keys are chunk indices:

{
  "0": [{"subject": "entity text", "predicate": "TREATS", "object": "entity text", "confidence": 0.9, "evidence": "text span"}, ...],
  "1": [...],
  ...
}

Response (JSON object only):
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise clinical relationship extraction system. Always respond with valid JSON."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Batch relation response: {content[:500]}...")
            
            # Clean JSON
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            batch_data = json.loads(content)
            
            # Convert to ClinicalRelation objects per chunk
            all_relations = []
            for i in range(len(batch_input)):
                chunk_relations = []
                relations_data = batch_data.get(str(i), [])
                entities = all_entities[i]
                
                for rel_data in relations_data:
                    try:
                        # Find matching entities
                        subject = next((e for e in entities if e.text.lower() == rel_data["subject"].lower()), None)
                        obj = next((e for e in entities if e.text.lower() == rel_data["object"].lower()), None)
                        
                        if subject and obj:
                            chunk_relations.append(ClinicalRelation(
                                subject_entity=subject,
                                predicate=rel_data["predicate"],
                                object_entity=obj,
                                confidence=rel_data.get("confidence", 0.8),
                                evidence_span=rel_data.get("evidence", ""),
                                evidence_start_char=0,
                                evidence_end_char=len(rel_data.get("evidence", ""))
                            ))
                    except (KeyError, StopIteration) as e:
                        logger.warning(f"Skipping invalid relation in chunk {i}: {e}")
                        continue
                
                all_relations.append(chunk_relations)
                logger.info(f"Chunk {i}: extracted {len(chunk_relations)} relations")
            
            return all_relations
            
        except Exception as e:
            logger.error(f"Batch relation extraction failed: {e}")
            # Fallback: return empty lists for each chunk
            return [[] for _ in batch_input]

class MedSpaCyContextExtractor:
    """Extracts clinical context using medspaCy (negation, uncertainty, temporality)."""
    
    def __init__(self):
        try:
            import medspacy
            self.nlp = medspacy.load()
            self.available = True
            logger.info("medspaCy loaded successfully")
        except ImportError:
            logger.warning("medspaCy not available. Context extraction will be limited.")
            self.available = False
            self.nlp = None
    
    def extract_context(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """
        Extract clinical context flags for entities.
        
        Args:
            text: Source text
            entities: List of entities to analyze
            
        Returns:
            Updated entities with context flags
        """
        if not self.available:
            return entities
        
        try:
            doc = self.nlp(text)
            
            # Entity types that should NOT receive clinical context flags
            # These are subjects/actors, not clinical concepts
            CONTEXT_IMMUNE_TYPES = {'population', 'person', 'organization', 'location'}
            
            # Update entities with context information
            updated_entities = []
            for entity in entities:
                context_flags = None
                
                # Skip context extraction for subject entities (patients, investigators, sites)
                # These are actors in the trial, not clinical findings that can be negated/historical
                if entity.entity_type in CONTEXT_IMMUNE_TYPES:
                    logger.debug(f"Skipping context for subject entity: '{entity.text}' ({entity.entity_type})")
                    updated_entities.append(entity)
                    continue
                
                # Try to find overlapping medspaCy entity with stricter matching
                best_match = None
                best_overlap = 0
                best_overlap_ratio = 0.0
                
                for ent in doc.ents:
                    # Calculate character overlap
                    overlap_start = max(entity.start_char, ent.start_char)
                    overlap_end = min(entity.end_char, ent.end_char)
                    overlap_size = max(0, overlap_end - overlap_start)
                    
                    # Calculate overlap ratio (what % of the entity overlaps)
                    entity_length = entity.end_char - entity.start_char
                    overlap_ratio = overlap_size / entity_length if entity_length > 0 else 0
                    
                    # Require at least 50% overlap AND text match for better precision
                    if overlap_ratio >= 0.5 and entity.text.lower() in ent.text.lower():
                        if overlap_size > best_overlap:
                            best_overlap = overlap_size
                            best_overlap_ratio = overlap_ratio
                            best_match = ent
                
                if best_match and best_overlap_ratio >= 0.5:
                    # Extract context modifiers from medspaCy
                    context_flags = {
                        "is_negated": best_match._.is_negated,
                        "is_uncertain": best_match._.is_uncertain,
                        "is_historical": best_match._.is_historical,
                        "is_hypothetical": best_match._.is_hypothetical,
                        "is_family": best_match._.is_family
                    }
                    logger.debug(f"Applied medspaCy context to: {entity.text} (matched with '{best_match.text}') - {context_flags}")
                else:
                    # Enhanced rule-based context detection with better clinical patterns
                    entity_start = max(0, entity.start_char - 30)
                    entity_end = min(len(text), entity.end_char + 30)
                    entity_context = text[entity_start:entity_end].lower()
                    
                    context_flags = {
                        "is_negated": any(neg in entity_context for neg in [
                            'denies', 'no ', 'not ', 'negative', 'absent', 'without', 'ruled out',
                            'no evidence of', 'no signs of', 'no symptoms of', 'unremarkable'
                        ]),
                        "is_uncertain": any(unc in entity_context for unc in [
                            'possible', 'likely', 'probable', 'maybe', 'uncertain', 'questionable',
                            'suggestive of', 'consistent with', 'suspicious for', 'rule out'
                        ]),
                        "is_historical": any(hist in entity_context for hist in [
                            'history', 'previous', 'prior', 'past', 'formerly', 'previously',
                            'h/o', 'hx of', 'old ', 'chronic'
                        ]),
                        "is_hypothetical": any(hyp in entity_context for hyp in [
                            'if ', 'should', 'would', 'could', 'may ', 'might', 'potential',
                            'risk of', 'risk for', 'prophylaxis'
                        ]),
                        "is_family": any(fam in entity_context for fam in [
                            'family', 'familial', 'hereditary', 'genetic', 'mother', 'father',
                            'sibling', 'parent', 'relative'
                        ])
                    }
                    
                    if any(context_flags.values()):
                        logger.debug(f"Applied enhanced fallback context to: {entity.text} - {context_flags}")
                    else:
                        # Default to no special context
                        context_flags = {
                            "is_negated": False,
                            "is_uncertain": False,
                            "is_historical": False,
                            "is_hypothetical": False,
                            "is_family": False
                        }
                
                # Create updated entity
                updated_entity = ClinicalEntity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=entity.confidence,
                    normalized_id=entity.normalized_id,
                    normalized_source=entity.normalized_source,
                    context_flags=context_flags
                )
                updated_entities.append(updated_entity)
            
            return updated_entities
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return entities




def extract_clinical_triples(text: str, chunk_id: UUID) -> TripleExtractionResult:
    """
    Main entry point for clinical triple extraction.
    
    Args:
        text: Clinical text to process
        chunk_id: UUID of the source chunk
        
    Returns:
        TripleExtractionResult with entities and relations
    """
    # Initialize extractors
    triple_extractor = ClinicalTripleExtractor()
    context_extractor = MedSpaCyContextExtractor()
    
    # Extract triples
    result = triple_extractor.extract_triples(text, chunk_id)
    
    # Add clinical context
    result.entities = context_extractor.extract_context(text, result.entities)
    
    return result