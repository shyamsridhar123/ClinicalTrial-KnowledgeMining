"""Repository ingestion utilities for clinical vocabularies."""

from .ingestion import (
    RepositoryNode,
    RepositoryEdge,
    RepositoryIngestionResult,
    VocabularyLoader,
    UMLSLoader,
    RxNormLoader,
    SnomedLoader,
    RepositoryIngestionOrchestrator,
)

# Optional medspaCy-based loaders
try:
    from .medspacy_loaders import (
        MedSpaCyUMLSLoader,
        MedSpaCyEnhancedRxNormLoader,
        MedSpaCyClinicalSnomedLoader,
        build_medspacy_loaders,
    )
    __all__ = [
        "RepositoryNode",
        "RepositoryEdge",
        "RepositoryIngestionResult",
        "VocabularyLoader",
        "UMLSLoader",
        "RxNormLoader",
        "SnomedLoader",
        "RepositoryIngestionOrchestrator",
        "MedSpaCyUMLSLoader",
        "MedSpaCyEnhancedRxNormLoader", 
        "MedSpaCyClinicalSnomedLoader",
        "build_medspacy_loaders",
    ]
except ImportError:
    __all__ = [
        "RepositoryNode",
        "RepositoryEdge",
        "RepositoryIngestionResult",
        "VocabularyLoader",
        "UMLSLoader",
        "RxNormLoader",
        "SnomedLoader",
        "RepositoryIngestionOrchestrator",
    ]

# Optional mock loaders for development
try:
    from .mock_loaders import MockUMLSLoader, MockRxNormLoader, MockSnomedLoader
    __all__.extend(["MockUMLSLoader", "MockRxNormLoader", "MockSnomedLoader"])
except ImportError:
    pass
