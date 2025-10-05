from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional


@dataclass(slots=True)
class RepoNodeRecord:
    """Normalized representation of a repository vocabulary node."""

    vocabulary: str
    code: str
    display_name: str
    canonical_uri: Optional[str]
    description: Optional[str]
    metadata: Dict[str, object]
    source_version: str
    checksum: str
    is_active: bool = True


@dataclass(slots=True)
class RepoEdgeRecord:
    """Normalized representation of a directed repository relationship."""

    vocabulary: str
    predicate: str
    source_code: str
    target_code: str
    metadata: Dict[str, object]


@dataclass(slots=True)
class ReleaseMetadata:
    """High-level metadata captured for each ingested vocabulary release."""

    vocabulary: str
    version: str
    release_checksum: str
    file_checksums: Mapping[str, str]
    ingested_at: datetime
    total_concepts: int
    total_relationships: int

    @classmethod
    def create(
        cls,
        *,
        vocabulary: str,
        version: str,
        release_checksum: str,
        file_checksums: Mapping[str, str],
        total_concepts: int,
        total_relationships: int,
        ingested_at: Optional[datetime] = None,
    ) -> "ReleaseMetadata":
        return cls(
            vocabulary=vocabulary,
            version=version,
            release_checksum=release_checksum,
            file_checksums=dict(file_checksums),
            total_concepts=total_concepts,
            total_relationships=total_relationships,
            ingested_at=ingested_at or datetime.now(timezone.utc),
        )
