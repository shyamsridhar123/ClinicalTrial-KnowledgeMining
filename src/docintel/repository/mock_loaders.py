"""Mock vocabulary loaders for development and testing when licensed vocabularies are unavailable."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterator, List

from .ingestion import RepositoryNode, RepositoryEdge, VocabularyLoader


class MockUMLSLoader(VocabularyLoader):
    """Mock UMLS loader with synthetic medical concepts for development."""
    
    vocabulary = "umls"
    
    def __init__(self, root: Path = None):
        self.root = root or Path("mock")
        self._mock_concepts = self._generate_mock_concepts()
    
    def describe(self) -> str:
        return f"Mock UMLS (development stub)"
    
    def source_version(self) -> str:
        return "2024AA-MOCK"
    
    def iter_nodes(self) -> Iterator[RepositoryNode]:
        """Generate mock UMLS concepts."""
        for concept in self._mock_concepts:
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=concept["cui"],
                display_name=concept["preferred_name"],
                canonical_uri=f"https://uts.nlm.nih.gov/umls/concept/{concept['cui']}",
                description=concept.get("definition"),
                metadata={
                    "synonyms": concept.get("synonyms", []),
                    "semantic_types": concept.get("semantic_types", []),
                    "source_codes": concept.get("source_codes", {}),
                    "mock_data": True,
                },
                source_version=self.source_version(),
            )
    
    def iter_edges(self) -> Iterator[RepositoryEdge]:
        """Generate mock UMLS relationships."""
        concepts = list(self._mock_concepts)
        for i, source in enumerate(concepts[:50]):  # Limit edges for development
            for j, target in enumerate(concepts[i+1:i+3], i+1):  # Each concept relates to 2 others
                if j < len(concepts):
                    yield RepositoryEdge(
                        vocabulary=self.vocabulary,
                        source_code=source["cui"],
                        target_code=target["cui"],
                        predicate="broader_than",
                        metadata={"rel": "RB", "rela": "broader_than", "sab": "MOCK", "mock_data": True}
                    )
    
    def _generate_mock_concepts(self) -> List[Dict]:
        """Generate synthetic medical concepts for development."""
        base_concepts = [
            {"name": "Heart Disease", "semantic": "Disease or Syndrome", "synonyms": ["Cardiac Disease", "Cardiovascular Disease"]},
            {"name": "Diabetes Mellitus", "semantic": "Disease or Syndrome", "synonyms": ["Diabetes", "DM"]},
            {"name": "Hypertension", "semantic": "Disease or Syndrome", "synonyms": ["High Blood Pressure", "HTN"]},
            {"name": "Myocardial Infarction", "semantic": "Disease or Syndrome", "synonyms": ["Heart Attack", "MI"]},
            {"name": "Pneumonia", "semantic": "Disease or Syndrome", "synonyms": ["Lung Infection"]},
            {"name": "Aspirin", "semantic": "Pharmacologic Substance", "synonyms": ["ASA", "Acetylsalicylic Acid"]},
            {"name": "Metformin", "semantic": "Pharmacologic Substance", "synonyms": ["Glucophage"]},
            {"name": "Lisinopril", "semantic": "Pharmacologic Substance", "synonyms": ["ACE Inhibitor"]},
            {"name": "Chest Pain", "semantic": "Sign or Symptom", "synonyms": ["Thoracic Pain"]},
            {"name": "Shortness of Breath", "semantic": "Sign or Symptom", "synonyms": ["Dyspnea", "SOB"]},
        ]
        
        concepts = []
        for i, concept in enumerate(base_concepts):
            cui = f"C{str(i).zfill(7)}"  # Mock CUI format
            concepts.append({
                "cui": cui,
                "preferred_name": concept["name"],
                "synonyms": concept.get("synonyms", []),
                "semantic_types": [concept["semantic"]],
                "definition": f"Mock definition for {concept['name']}",
                "source_codes": {"MOCK": [f"MOCK_{i}"]}
            })
        
        return concepts


class MockRxNormLoader(VocabularyLoader):
    """Mock RxNorm loader with synthetic medication concepts."""
    
    vocabulary = "rxnorm"
    
    def __init__(self, root: Path = None):
        self.root = root or Path("mock")
        self._mock_medications = self._generate_mock_medications()
    
    def describe(self) -> str:
        return f"Mock RxNorm (development stub)"
    
    def source_version(self) -> str:
        return "20240201-MOCK"
    
    def iter_nodes(self) -> Iterator[RepositoryNode]:
        """Generate mock RxNorm medication concepts."""
        for med in self._mock_medications:
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=med["rxcui"],
                display_name=med["name"],
                canonical_uri=f"https://rxnav.nlm.nih.gov/REST/rxcui/{med['rxcui']}",
                description=med.get("description"),
                metadata={
                    "synonyms": med.get("synonyms", []),
                    "mock_data": True,
                },
                source_version=self.source_version(),
            )
    
    def iter_edges(self) -> Iterator[RepositoryEdge]:
        """Generate mock RxNorm relationships."""
        meds = list(self._mock_medications)
        for i, source in enumerate(meds[:20]):  # Limit for development
            for j, target in enumerate(meds[i+1:i+2], i+1):
                if j < len(meds):
                    yield RepositoryEdge(
                        vocabulary=self.vocabulary,
                        source_code=source["rxcui"],
                        target_code=target["rxcui"],
                        predicate="ingredient_of",
                        metadata={"rel": "ingredient_of", "mock_data": True}
                    )
    
    def _generate_mock_medications(self) -> List[Dict]:
        """Generate synthetic medication concepts."""
        medications = [
            {"name": "Aspirin 81 MG Oral Tablet", "synonyms": ["Low-dose Aspirin", "Baby Aspirin"]},
            {"name": "Metformin 500 MG Oral Tablet", "synonyms": ["Glucophage"]},
            {"name": "Lisinopril 10 MG Oral Tablet", "synonyms": ["Prinivil", "Zestril"]},
            {"name": "Simvastatin 20 MG Oral Tablet", "synonyms": ["Zocor"]},
            {"name": "Omeprazole 20 MG Oral Capsule", "synonyms": ["Prilosec"]},
            {"name": "Acetaminophen 500 MG Oral Tablet", "synonyms": ["Tylenol", "Paracetamol"]},
            {"name": "Ibuprofen 200 MG Oral Tablet", "synonyms": ["Advil", "Motrin"]},
            {"name": "Amlodipine 5 MG Oral Tablet", "synonyms": ["Norvasc"]},
        ]
        
        mock_meds = []
        for i, med in enumerate(medications):
            rxcui = str(100000 + i)  # Mock RXCUI
            mock_meds.append({
                "rxcui": rxcui,
                "name": med["name"],
                "synonyms": med.get("synonyms", []),
                "description": f"Mock medication: {med['name']}"
            })
        
        return mock_meds


class MockSnomedLoader(VocabularyLoader):
    """Mock SNOMED CT loader with synthetic clinical concepts."""
    
    vocabulary = "snomed"
    
    def __init__(self, root: Path = None):
        self.root = root or Path("mock")
        self._mock_concepts = self._generate_mock_concepts()
    
    def describe(self) -> str:
        return f"Mock SNOMED CT (development stub)"
    
    def source_version(self) -> str:
        return "20240201-MOCK"
    
    def iter_nodes(self) -> Iterator[RepositoryNode]:
        """Generate mock SNOMED CT concepts."""
        for concept in self._mock_concepts:
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=concept["sctid"],
                display_name=concept["fsn"],
                canonical_uri=f"https://snomedbrowser.com/Codes/Details/{concept['sctid']}",
                description=concept.get("description"),
                metadata={
                    "synonyms": concept.get("synonyms", []),
                    "module_id": concept.get("module_id", "449081005"),
                    "mock_data": True,
                },
                source_version=self.source_version(),
                is_active=True,
            )
    
    def iter_edges(self) -> Iterator[RepositoryEdge]:
        """Generate mock SNOMED CT relationships."""
        concepts = list(self._mock_concepts)
        for i, source in enumerate(concepts[:30]):
            for j, target in enumerate(concepts[i+1:i+2], i+1):
                if j < len(concepts):
                    yield RepositoryEdge(
                        vocabulary=self.vocabulary,
                        source_code=source["sctid"],
                        target_code=target["sctid"],
                        predicate="116680003",  # "Is a" relationship
                        metadata={
                            "characteristicTypeId": "900000000000011006",
                            "modifierId": "900000000000451002",
                            "mock_data": True
                        }
                    )
    
    def _generate_mock_concepts(self) -> List[Dict]:
        """Generate synthetic SNOMED CT concepts."""
        concepts = [
            {"fsn": "Heart disease (disorder)", "synonyms": ["Cardiac disorder", "Heart condition"]},
            {"fsn": "Diabetes mellitus (disorder)", "synonyms": ["Diabetes", "DM"]},
            {"fsn": "Hypertensive disorder (disorder)", "synonyms": ["High blood pressure", "Hypertension"]},
            {"fsn": "Myocardial infarction (disorder)", "synonyms": ["Heart attack", "MI"]},
            {"fsn": "Pneumonia (disorder)", "synonyms": ["Lung infection"]},
            {"fsn": "Chest pain (finding)", "synonyms": ["Thoracic pain", "Chest discomfort"]},
            {"fsn": "Dyspnea (finding)", "synonyms": ["Shortness of breath", "Breathing difficulty"]},
            {"fsn": "Aspirin (substance)", "synonyms": ["Acetylsalicylic acid", "ASA"]},
            {"fsn": "Administration of drug (procedure)", "synonyms": ["Drug administration", "Medication given"]},
            {"fsn": "Blood pressure taking (procedure)", "synonyms": ["BP measurement", "Vital signs"]},
        ]
        
        mock_concepts = []
        for i, concept in enumerate(concepts):
            sctid = str(1000000 + i)  # Mock SNOMED ID
            mock_concepts.append({
                "sctid": sctid,
                "fsn": concept["fsn"],
                "synonyms": concept.get("synonyms", []),
                "description": f"Mock SNOMED concept: {concept['fsn']}",
                "module_id": "449081005"  # Mock module ID
            })
        
        return mock_concepts