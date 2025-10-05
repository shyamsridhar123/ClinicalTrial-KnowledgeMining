"""Integrate tables and figures into knowledge graph entities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from .triple_extraction import ClinicalEntity


logger = logging.getLogger(__name__)


class AssetIntegrator:
    """Extract entities from tables and figures and link to knowledge graph."""
    
    def __init__(self, processing_root: Path):
        """
        Initialize asset integrator.
        
        Args:
            processing_root: Root directory containing tables/ and figures/ subdirs
        """
        self.tables_dir = processing_root / "tables"
        self.figures_dir = processing_root / "figures"
    
    def process_chunk_assets(
        self,
        nct_id: str,
        document_id: str,
        chunk_metadata: Dict[str, Any],
        existing_entities: List[ClinicalEntity]
    ) -> Dict[str, Any]:
        """
        Process tables and figures for a chunk.
        
        Args:
            nct_id: NCT identifier
            document_id: Document identifier (may be empty)
            chunk_metadata: Chunk metadata with contains_tables/contains_figures flags
            existing_entities: Existing text entities from this chunk
            
        Returns:
            Dict with:
                - table_entities: List of entities extracted from tables
                - figure_entities: List of entities from figure captions
                - asset_metadata: Asset file references
        """
        result = {
            "table_entities": [],
            "figure_entities": [],
            "asset_metadata": {}
        }
        
        if not nct_id:
            return result
        
        contains_tables = chunk_metadata.get("contains_tables", False)
        contains_figures = chunk_metadata.get("contains_figures", False)
        
        # If document_id is missing, find all document IDs for this NCT
        if contains_tables:
            table_entities = self._extract_all_tables_for_nct(nct_id, document_id)
            result["table_entities"] = table_entities
            result["asset_metadata"]["tables"] = f"data/processing/tables/{nct_id}/"
            logger.info(f"Extracted {len(table_entities)} entities from tables in {nct_id}")
        
        if contains_figures:
            figure_entities = self._extract_all_figures_for_nct(nct_id, document_id)
            result["figure_entities"] = figure_entities
            result["asset_metadata"]["figures"] = f"data/processing/figures/{nct_id}/"
            logger.info(f"Extracted {len(figure_entities)} entities from figures in {nct_id}")
        
        return result
    
    def _extract_all_tables_for_nct(self, nct_id: str, document_id: str = "") -> List[ClinicalEntity]:
        """
        Extract entities from all table files for an NCT study.
        
        Args:
            nct_id: NCT identifier
            document_id: Optional specific document ID to filter
            
        Returns:
            List of ClinicalEntity objects from table data
        """
        entities = []
        nct_tables_dir = self.tables_dir / nct_id
        
        if not nct_tables_dir.exists():
            logger.debug(f"Tables directory not found: {nct_tables_dir}")
            return entities
        
        # Find all table JSON files (exclude _figures.json files)
        table_files = [f for f in nct_tables_dir.glob("*.json") if not f.name.endswith("_figures.json")]
        
        for table_file in table_files:
            # If document_id is specified, only process that specific file
            if document_id and not table_file.name.startswith(document_id):
                continue
            
            file_entities = self._extract_table_entities_from_file(table_file)
            entities.extend(file_entities)
        
        return entities
    
    def _extract_table_entities_from_file(self, table_file: Path) -> List[ClinicalEntity]:
        """
        Extract entities from a single table JSON file.
        
        Args:
            table_file: Path to table JSON file
            
        Returns:
            List of ClinicalEntity objects from table data
        """
        entities = []
        
        if not table_file.exists():
            logger.debug(f"Table file not found: {table_file}")
            return entities
        
        try:
            with open(table_file, 'r') as f:
                tables = json.load(f)
            
            for table_idx, table in enumerate(tables):
                # Parse HTML table
                html = table.get("html", "")
                caption = table.get("caption", "")
                
                if not html:
                    continue
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text from all cells
                cells = soup.find_all(['td', 'th'])
                for cell_idx, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)
                    
                    # Skip empty cells and very short text
                    if len(cell_text) < 3:
                        continue
                    
                    # Create entity for cell content
                    # Type inference based on content patterns
                    entity_type = self._infer_cell_entity_type(cell_text)
                    
                    entity = ClinicalEntity(
                        text=cell_text,
                        entity_type=entity_type,
                        start_char=0,  # Relative to table
                        end_char=len(cell_text),
                        confidence=0.85,  # Table extraction confidence
                        normalized_id=None,
                        normalized_source=None,
                        context_flags={"from_table": True, "table_index": table_idx, "cell_index": cell_idx}
                    )
                    entities.append(entity)
                
                # Extract entities from caption if present
                if caption:
                    caption_entity = ClinicalEntity(
                        text=caption,
                        entity_type="table_caption",
                        start_char=0,
                        end_char=len(caption),
                        confidence=0.9,
                        normalized_id=None,
                        normalized_source=None,
                        context_flags={"is_caption": True, "table_index": table_idx}
                    )
                    entities.append(caption_entity)
        
        except Exception as e:
            logger.error(f"Error extracting table entities from {table_file}: {e}")
        
        return entities
    
    def _extract_all_figures_for_nct(self, nct_id: str, document_id: str = "") -> List[ClinicalEntity]:
        """
        Extract entities from all figure metadata files for an NCT study.
        
        Args:
            nct_id: NCT identifier
            document_id: Optional specific document ID to filter
            
        Returns:
            List of ClinicalEntity objects from figure captions
        """
        entities = []
        nct_tables_dir = self.tables_dir / nct_id
        
        if not nct_tables_dir.exists():
            logger.debug(f"Tables directory not found: {nct_tables_dir}")
            return entities
        
        # Find all *_figures.json files
        figure_meta_files = list(nct_tables_dir.glob("*_figures.json"))
        
        for figures_meta_file in figure_meta_files:
            # If document_id is specified, only process that specific file
            if document_id and not figures_meta_file.name.startswith(document_id):
                continue
            
            file_entities = self._extract_figure_entities_from_file(figures_meta_file)
            entities.extend(file_entities)
        
        return entities
    
    def _extract_figure_entities_from_file(self, figures_meta_file: Path) -> List[ClinicalEntity]:
        """
        Extract entities from a single figure metadata JSON file.
        
        Args:
            figures_meta_file: Path to figures metadata JSON file
            
        Returns:
            List of ClinicalEntity objects from figure captions
        """
        entities = []
        
        if not figures_meta_file.exists():
            logger.debug(f"Figures metadata not found: {figures_meta_file}")
            return entities
        
        try:
            with open(figures_meta_file, 'r') as f:
                figures = json.load(f)
            
            for fig_idx, figure in enumerate(figures):
                caption = figure.get("caption", "")
                fig_id = figure.get("id", "")
                
                if not caption:
                    continue
                
                # Create entity for figure caption
                entity = ClinicalEntity(
                    text=caption,
                    entity_type="figure_caption",
                    start_char=0,
                    end_char=len(caption),
                    confidence=0.9,
                    normalized_id=None,
                    normalized_source=None,
                    context_flags={"is_caption": True, "figure_index": fig_idx, "figure_id": fig_id}
                )
                entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error extracting figure entities from {figures_meta_file}: {e}")
        
        return entities
    
    def _infer_cell_entity_type(self, cell_text: str) -> str:
        """
        Infer entity type from table cell content.
        
        Args:
            cell_text: Cell text content
            
        Returns:
            Inferred entity type
        """
        text_lower = cell_text.lower()
        
        # Measurement patterns
        if any(unit in text_lower for unit in ['mg', 'ml', 'kg', '%', 'mcg', 'iu', 'mmol']):
            return "measurement"
        
        # Timepoint patterns
        if any(time in text_lower for time in ['day', 'week', 'month', 'year', 'hour', 'minute']):
            return "timepoint"
        
        # Number patterns (likely measurements)
        if any(char.isdigit() for char in cell_text):
            return "measurement"
        
        # Protocol/study identifiers
        if cell_text.startswith(('NCT', 'PROTO', 'STUDY')):
            return "protocol"
        
        # Default to generic table_data
        return "table_data"
    
    def get_asset_provenance(
        self,
        nct_id: str,
        document_id: str,
        asset_type: str,
        asset_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate provenance metadata for an asset.
        
        Args:
            nct_id: NCT identifier
            document_id: Document identifier
            asset_type: 'table' or 'figure'
            asset_index: Index of asset within document
            
        Returns:
            Provenance metadata dict
        """
        if asset_type == "table":
            return {
                "asset_kind": "table",
                "asset_ref": f"data/processing/tables/{nct_id}/{document_id}.json",
                "asset_index": asset_index,
                "nct_id": nct_id,
                "document_id": document_id
            }
        elif asset_type == "figure":
            return {
                "asset_kind": "figure",
                "asset_ref": f"data/processing/figures/{nct_id}/{document_id}/",
                "asset_index": asset_index,
                "nct_id": nct_id,
                "document_id": document_id
            }
        else:
            return {
                "asset_kind": "text",
                "nct_id": nct_id,
                "document_id": document_id
            }
