"""
Enhanced Semantic Chunking for Clinical Documents

Implements clinical-aware semantic boundary detection to replace simple token-based chunking.
Based on Medical-Graph-RAG semantic chunking principles with clinical domain optimization.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClinicalSectionType(Enum):
    """Clinical document section types for semantic boundary detection"""
    TITLE = "title"
    ABSTRACT = "abstract"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    INCLUSION_CRITERIA = "inclusion_criteria"
    EXCLUSION_CRITERIA = "exclusion_criteria"
    PRIMARY_ENDPOINT = "primary_endpoint"
    SECONDARY_ENDPOINT = "secondary_endpoint"
    ADVERSE_EVENTS = "adverse_events"
    DEMOGRAPHICS = "demographics"
    PROTOCOL = "protocol"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    UNKNOWN = "unknown"


@dataclass
class SemanticChunk:
    """Enhanced chunk with semantic metadata"""
    id: str
    text: str
    token_count: int
    char_count: int
    section_type: ClinicalSectionType
    section_header: Optional[str]
    sentence_count: int
    start_char_index: int
    end_char_index: int
    contains_tables: bool
    contains_figures: bool
    clinical_entities_hint: List[str]  # Predicted entity types
    semantic_coherence_score: float  # 0-1 coherence measure


class ClinicalSemanticChunker:
    """
    Advanced semantic chunking for clinical trial documents.
    
    Features:
    - Clinical section header detection
    - Sentence boundary preservation  
    - Topic coherence analysis
    - Table/figure boundary awareness
    - Clinical entity preservation
    """
    
    def __init__(self, 
                 target_token_size: int = 1200,
                 overlap_tokens: int = 100,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 2000):
        self.target_token_size = target_token_size
        self.overlap_tokens = overlap_tokens
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Clinical section patterns (case-insensitive)
        self.section_patterns = {
            ClinicalSectionType.ABSTRACT: [
                r'^abstract\s*:?\s*$',
                r'^summary\s*:?\s*$',
                r'^\d+\.\s*abstract\s*$'
            ],
            ClinicalSectionType.BACKGROUND: [
                r'^background\s*:?\s*$',
                r'^introduction\s*:?\s*$',
                r'^\d+\.\s*background\s*$',
                r'^\d+\.\s*introduction\s*$'
            ],
            ClinicalSectionType.METHODS: [
                r'^methods?\s*:?\s*$',
                r'^methodology\s*:?\s*$',
                r'^study\s+design\s*:?\s*$',
                r'^\d+\.\s*methods?\s*$'
            ],
            ClinicalSectionType.RESULTS: [
                r'^results?\s*:?\s*$',
                r'^findings?\s*:?\s*$',
                r'^\d+\.\s*results?\s*$'
            ],
            ClinicalSectionType.DISCUSSION: [
                r'^discussion\s*:?\s*$',
                r'^\d+\.\s*discussion\s*$'
            ],
            ClinicalSectionType.CONCLUSION: [
                r'^conclusions?\s*:?\s*$',
                r'^summary\s+and\s+conclusions?\s*:?\s*$',
                r'^\d+\.\s*conclusions?\s*$'
            ],
            ClinicalSectionType.INCLUSION_CRITERIA: [
                r'^inclusion\s+criteria\s*:?\s*$',
                r'^participant\s+inclusion\s*:?\s*$',
                r'^\d+\.\d+\s*inclusion\s+criteria\s*$'
            ],
            ClinicalSectionType.EXCLUSION_CRITERIA: [
                r'^exclusion\s+criteria\s*:?\s*$',
                r'^participant\s+exclusion\s*:?\s*$',
                r'^\d+\.\d+\s*exclusion\s+criteria\s*$'
            ],
            ClinicalSectionType.PRIMARY_ENDPOINT: [
                r'^primary\s+endpoint\s*:?\s*$',
                r'^primary\s+outcome\s*:?\s*$',
                r'^\d+\.\d+\s*primary\s+endpoint\s*$'
            ],
            ClinicalSectionType.SECONDARY_ENDPOINT: [
                r'^secondary\s+endpoint\s*:?\s*$',
                r'^secondary\s+outcome\s*:?\s*$',
                r'^\d+\.\d+\s*secondary\s+endpoint\s*$'
            ],
            ClinicalSectionType.ADVERSE_EVENTS: [
                r'^adverse\s+events?\s*:?\s*$',
                r'^safety\s+events?\s*:?\s*$',
                r'^side\s+effects?\s*:?\s*$',
                r'^\d+\.\d+\s*adverse\s+events?\s*$'
            ],
            ClinicalSectionType.DEMOGRAPHICS: [
                r'^demographics?\s*:?\s*$',
                r'^baseline\s+characteristics?\s*:?\s*$',
                r'^patient\s+characteristics?\s*:?\s*$'
            ],
            ClinicalSectionType.STATISTICAL_ANALYSIS: [
                r'^statistical\s+analysis\s*:?\s*$',
                r'^statistical\s+methods?\s*:?\s*$',
                r'^data\s+analysis\s*:?\s*$'
            ]
        }
        
        # Clinical entity hint patterns
        self.entity_patterns = {
            'drug': r'\b(?:mg|mcg|units?|tablets?|capsules?|injection|infusion)\b',
            'disease': r'\b(?:cancer|diabetes|hypertension|infection|syndrome|disorder)\b',
            'symptom': r'\b(?:pain|nausea|fatigue|fever|headache|dizziness)\b',
            'measurement': r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|%|mmHg)\b',
            'temporal': r'\b(?:daily|weekly|monthly|baseline|week\s+\d+|day\s+\d+)\b'
        }
    
    def _normalize_document_id(self, document_id: str) -> str:
        """Create a filesystem and ID-safe document identifier."""
        sanitized = re.sub(r'[^A-Za-z0-9]+', '-', document_id.strip())
        sanitized = sanitized.strip('-')
        return sanitized or "document"

    def _format_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Return chunk IDs that always contain the chunk- token for compatibility."""
        normalized = self._normalize_document_id(document_id)
        return f"{normalized}-chunk-{chunk_index:04d}"

    def _extract_chunk_prefix(self, chunk_id: str) -> str:
        """Extract the document prefix portion from an existing chunk ID."""
        parts = re.split(r'(?:-chunk-|_chunk_)', chunk_id, maxsplit=1)
        if parts and parts[0]:
            return parts[0]
        return self._normalize_document_id(chunk_id)
    
    def chunk_document(self, text: str, document_id: str = "unknown") -> List[SemanticChunk]:
        """
        Chunk document using clinical semantic boundaries.
        
        Args:
            text: Full document text
            document_id: Document identifier for chunk IDs
            
        Returns:
            List of semantic chunks with clinical metadata
        """
        logger.info(f"Starting semantic chunking for document {document_id} ({len(text)} chars)")
        
        if not text or not text.strip():
            return []
        
        normalized_document_id = self._normalize_document_id(document_id)
        
        # Step 1: Detect clinical sections
        sections = self._detect_clinical_sections(text)
        logger.info(f"Detected {len(sections)} clinical sections")
        
        # Step 2: Split into sentences with section awareness
        sentences = self._split_into_sentences(text, sections)
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Step 3: Create semantic chunks
        chunks = self._create_semantic_chunks(sentences, sections, normalized_document_id)
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        # Step 4: Validate and optimize chunks
        optimized_chunks = self._optimize_chunks(chunks)
        logger.info(f"Optimized to {len(optimized_chunks)} final chunks")
        
        return optimized_chunks
    
    def _detect_clinical_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect clinical document sections and boundaries"""
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for section headers
            section_type = self._classify_section_header(line_stripped)
            if section_type != ClinicalSectionType.UNKNOWN:
                sections.append({
                    'type': section_type,
                    'header': line_stripped,
                    'line_number': i,
                    'char_start': sum(len(lines[j]) + 1 for j in range(i)),
                    'char_end': None  # Will be set when next section found
                })
        
        # Set section end boundaries
        for i in range(len(sections)):
            if i < len(sections) - 1:
                sections[i]['char_end'] = sections[i + 1]['char_start']
            else:
                sections[i]['char_end'] = len(text)
        
        # Add default section if no sections detected
        if not sections:
            sections.append({
                'type': ClinicalSectionType.UNKNOWN,
                'header': None,
                'line_number': 0,
                'char_start': 0,
                'char_end': len(text)
            })
        
        return sections
    
    def _classify_section_header(self, line: str) -> ClinicalSectionType:
        """Classify a line as a clinical section header"""
        line_lower = line.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower):
                    return section_type
        
        return ClinicalSectionType.UNKNOWN
    
    def _split_into_sentences(self, text: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text into sentences with section context"""
        sentences = []
        
        # Enhanced sentence boundary detection
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        for section in sections:
            section_text = text[section['char_start']:section['char_end']]
            section_sentences = re.split(sentence_pattern, section_text.strip())
            
            char_offset = section['char_start']
            for sentence_text in section_sentences:
                if sentence_text.strip():
                    sentences.append({
                        'text': sentence_text.strip(),
                        'section_type': section['type'],
                        'section_header': section['header'],
                        'char_start': char_offset,
                        'char_end': char_offset + len(sentence_text),
                        'token_count': len(sentence_text.split()),
                        'contains_table': self._contains_table(sentence_text),
                        'contains_figure': self._contains_figure(sentence_text)
                    })
                    char_offset += len(sentence_text) + 1  # +1 for space
        
        return sentences
    
    def _contains_table(self, text: str) -> bool:
        """Detect if text contains table references or data"""
        table_indicators = [
            r'\btable\s+\d+\b',
            r'\|.*\|.*\|',  # Simple table format
            r'\t.*\t.*\t',  # Tab-separated
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in table_indicators)
    
    def _contains_figure(self, text: str) -> bool:
        """Detect if text contains figure references"""
        figure_indicators = [
            r'\bfigure\s+\d+\b',
            r'\bfig\.?\s+\d+\b',
            r'\bchart\s+\d+\b'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in figure_indicators)
    
    def _create_semantic_chunks(self, 
                              sentences: List[Dict[str, Any]], 
                              sections: List[Dict[str, Any]],
                              document_id: str) -> List[SemanticChunk]:
        """Create semantic chunks from sentences with clinical awareness"""
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_token_count = sentence['token_count']
            
            # Check if adding this sentence would exceed target size
            if (current_chunk_sentences and 
                current_token_count + sentence_token_count > self.target_token_size):
                
                # Finalize current chunk
                chunk = self._finalize_chunk(
                    current_chunk_sentences, 
                    chunk_index, 
                    document_id
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap if needed
                if self.overlap_tokens > 0:
                    # Find overlap sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk_sentences, 
                        self.overlap_tokens
                    )
                    current_chunk_sentences = overlap_sentences
                    current_token_count = sum(s['token_count'] for s in overlap_sentences)
                else:
                    current_chunk_sentences = []
                    current_token_count = 0
                
                chunk_index += 1
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_token_count
        
        # Add final chunk
        if current_chunk_sentences:
            chunk = self._finalize_chunk(current_chunk_sentences, chunk_index, document_id)
            chunks.append(chunk)
        
        return chunks
    
    def _finalize_chunk(self, 
                       sentences: List[Dict[str, Any]], 
                       chunk_index: int, 
                       document_id: str) -> SemanticChunk:
        """Create final semantic chunk from sentences"""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentences")
        
        # Combine sentence texts
        chunk_text = " ".join(s['text'] for s in sentences)
        
        # Determine dominant section type
        section_types = [s['section_type'] for s in sentences]
        dominant_section = max(set(section_types), key=section_types.count)
        
        # Get section header (first non-None header)
        section_header = None
        for s in sentences:
            if s['section_header']:
                section_header = s['section_header']
                break
        
        # Calculate positions
        start_char = sentences[0]['char_start']
        end_char = sentences[-1]['char_end']
        
        # Detect clinical entities
        entity_hints = self._get_entity_hints(chunk_text)
        
        # Calculate semantic coherence
        coherence_score = self._calculate_coherence(sentences)
        
        return SemanticChunk(
            id=self._format_chunk_id(document_id, chunk_index),
            text=chunk_text,
            token_count=sum(s['token_count'] for s in sentences),
            char_count=len(chunk_text),
            section_type=dominant_section,
            section_header=section_header,
            sentence_count=len(sentences),
            start_char_index=start_char,
            end_char_index=end_char,
            contains_tables=any(s['contains_table'] for s in sentences),
            contains_figures=any(s['contains_figure'] for s in sentences),
            clinical_entities_hint=entity_hints,
            semantic_coherence_score=coherence_score
        )
    
    def _get_overlap_sentences(self, 
                              sentences: List[Dict[str, Any]], 
                              target_overlap_tokens: int) -> List[Dict[str, Any]]:
        """Get sentences for chunk overlap"""
        if not sentences or target_overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Work backwards from end of chunk
        for sentence in reversed(sentences):
            if overlap_tokens + sentence['token_count'] <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence['token_count']
            else:
                break
        
        return overlap_sentences
    
    def _get_entity_hints(self, text: str) -> List[str]:
        """Get hints about clinical entities in the text"""
        entity_hints = []
        text_lower = text.lower()
        
        for entity_type, pattern in self.entity_patterns.items():
            if re.search(pattern, text_lower):
                entity_hints.append(entity_type)
        
        return entity_hints
    
    def _calculate_coherence(self, sentences: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence score for chunk"""
        if len(sentences) <= 1:
            return 1.0
        
        # Simple coherence based on section consistency
        section_types = [s['section_type'] for s in sentences]
        unique_sections = set(section_types)
        
        # Higher coherence for sentences from same section
        coherence = 1.0 - (len(unique_sections) - 1) * 0.2
        return max(0.0, min(1.0, coherence))
    
    def _optimize_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Optimize chunks by merging small ones or splitting large ones"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            # If chunk is too small, try to merge with next
            if (chunk.token_count < self.min_chunk_size and 
                i + 1 < len(chunks) and
                chunks[i + 1].token_count + chunk.token_count <= self.max_chunk_size):
                
                merged_chunk = self._merge_chunks(chunk, chunks[i + 1])
                optimized.append(merged_chunk)
                i += 2  # Skip both chunks
            
            # If chunk is too large, split it
            elif chunk.token_count > self.max_chunk_size:
                split_chunks = self._split_chunk(chunk)
                optimized.extend(split_chunks)
                i += 1
            
            else:
                optimized.append(chunk)
                i += 1
        
        # Update chunk IDs to be sequential
        for idx, chunk in enumerate(optimized):
            prefix = self._extract_chunk_prefix(chunk.id)
            chunk.id = self._format_chunk_id(prefix, idx)
        
        return optimized
    
    def _merge_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> SemanticChunk:
        """Merge two adjacent chunks"""
        return SemanticChunk(
            id=chunk1.id,  # Keep first chunk's ID
            text=chunk1.text + " " + chunk2.text,
            token_count=chunk1.token_count + chunk2.token_count,
            char_count=chunk1.char_count + chunk2.char_count + 1,
            section_type=chunk1.section_type,  # Keep first chunk's section
            section_header=chunk1.section_header,
            sentence_count=chunk1.sentence_count + chunk2.sentence_count,
            start_char_index=chunk1.start_char_index,
            end_char_index=chunk2.end_char_index,
            contains_tables=chunk1.contains_tables or chunk2.contains_tables,
            contains_figures=chunk1.contains_figures or chunk2.contains_figures,
            clinical_entities_hint=list(set(chunk1.clinical_entities_hint + chunk2.clinical_entities_hint)),
            semantic_coherence_score=(chunk1.semantic_coherence_score + chunk2.semantic_coherence_score) / 2
        )
    
    def _split_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a large chunk into smaller ones"""
        # Simple split by sentences for now
        # In production, this could be more sophisticated
        text_parts = chunk.text.split('. ')
        if len(text_parts) < 2:
            return [chunk]  # Can't split
        
        mid_point = len(text_parts) // 2
        first_half = '. '.join(text_parts[:mid_point]) + '.'
        second_half = '. '.join(text_parts[mid_point:])
        
        return [
            SemanticChunk(
                id=f"{chunk.id}_a",
                text=first_half,
                token_count=len(first_half.split()),
                char_count=len(first_half),
                section_type=chunk.section_type,
                section_header=chunk.section_header,
                sentence_count=mid_point,
                start_char_index=chunk.start_char_index,
                end_char_index=chunk.start_char_index + len(first_half),
                contains_tables=chunk.contains_tables,
                contains_figures=chunk.contains_figures,
                clinical_entities_hint=chunk.clinical_entities_hint,
                semantic_coherence_score=chunk.semantic_coherence_score
            ),
            SemanticChunk(
                id=f"{chunk.id}_b",
                text=second_half,
                token_count=len(second_half.split()),
                char_count=len(second_half),
                section_type=chunk.section_type,
                section_header=chunk.section_header,
                sentence_count=len(text_parts) - mid_point,
                start_char_index=chunk.start_char_index + len(first_half) + 1,
                end_char_index=chunk.end_char_index,
                contains_tables=chunk.contains_tables,
                contains_figures=chunk.contains_figures,
                clinical_entities_hint=chunk.clinical_entities_hint,
                semantic_coherence_score=chunk.semantic_coherence_score
            )
        ]


def create_semantic_chunks(text: str, 
                          document_id: str = "unknown",
                          target_token_size: int = 1200,
                          overlap_tokens: int = 100) -> List[Dict[str, Any]]:
    """
    Convenience function to create semantic chunks compatible with existing pipeline.
    
    Args:
        text: Document text to chunk
        document_id: Document identifier
        target_token_size: Target tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of chunk dictionaries compatible with existing format
    """
    chunker = ClinicalSemanticChunker(
        target_token_size=target_token_size,
        overlap_tokens=overlap_tokens
    )
    
    semantic_chunks = chunker.chunk_document(text, document_id)
    
    # Convert to compatible format
    compatible_chunks = []
    for chunk in semantic_chunks:
        compatible_chunks.append({
            "id": chunk.id,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "section_type": chunk.section_type.value,
            "section_header": chunk.section_header,
            "sentence_count": chunk.sentence_count,
            "start_char_index": chunk.start_char_index,
            "end_char_index": chunk.end_char_index,
            "contains_tables": chunk.contains_tables,
            "contains_figures": chunk.contains_figures,
            "clinical_entities_hint": chunk.clinical_entities_hint,
            "semantic_coherence_score": chunk.semantic_coherence_score
        })
    
    return compatible_chunks