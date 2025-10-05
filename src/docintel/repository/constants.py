"""Shared constants for repository graph identifiers."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

# Deterministic namespaces used to derive node and edge identifiers across the
# repository ingestion and linking pipelines. Keeping them in one module avoids
# subtle mismatches that would break referential integrity.
NODE_NAMESPACE = uuid5(NAMESPACE_URL, "docintel/repository/node")
EDGE_NAMESPACE = uuid5(NAMESPACE_URL, "docintel/repository/edge")
