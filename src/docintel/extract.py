"""CLI entrypoint for clinical entity and relation extraction."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .config import get_config
from .knowledge_graph.enhanced_extraction import EnhancedClinicalTripleExtractor


CHUNK_NAMESPACE = uuid5(NAMESPACE_URL, "docintel/chunks")
# Minimum character length before we consider a chunk worth sending to the extractor.
MIN_CHUNK_DEFAULT = 50


def _stable_chunk_uuid(chunk_id: str) -> UUID:
    """Return a UUID for the provided chunk identifier, preserving valid UUID strings."""

    if not chunk_id:
        return uuid4()

    try:
        return UUID(str(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return uuid5(CHUNK_NAMESPACE, str(chunk_id))


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class ExtractionJob:
    """Manages clinical entity and relation extraction from parsed documents."""

    def __init__(
        self,
        extractor: Optional[EnhancedClinicalTripleExtractor] = None,
        *,
        chunk_root: Optional[Path] = None,
        max_concurrency: Optional[int] = None,
        min_chunk_characters: int = MIN_CHUNK_DEFAULT,
        fast_mode: bool = False,
        skip_relations: bool = False,
        batch_size: int = 20,
    ) -> None:
        self.config = get_config()
        self.extractor = extractor or EnhancedClinicalTripleExtractor(
            fast_mode=fast_mode,
            skip_relations=skip_relations
        )
        self.batch_size = max(0, batch_size)  # 0 = disable batching
        parsing_dirs = self.config.parsing.processing_directories()
        self.chunk_root = Path(chunk_root) if chunk_root is not None else parsing_dirs["chunks"]
        self.chunk_root = self.chunk_root.expanduser().resolve()
        self.results: List[Dict[str, Any]] = []
        self.min_chunk_characters = max(0, min_chunk_characters)

        env_concurrency = os.getenv("DOCINTEL_EXTRACTION_MAX_CONCURRENCY")
        parsed_env_concurrency: Optional[int] = None
        if env_concurrency:
            try:
                parsed_env_concurrency = int(env_concurrency)
            except ValueError:
                logging.warning(
                    "Invalid DOCINTEL_EXTRACTION_MAX_CONCURRENCY value '%s'; falling back to default", env_concurrency
                )

        if max_concurrency is not None:
            parsed_concurrency = max_concurrency
        elif parsed_env_concurrency is not None:
            parsed_concurrency = parsed_env_concurrency
        else:
            cpu_count = os.cpu_count() or 1
            # Increased from 4 to 10 - Azure OpenAI can handle higher concurrency
            parsed_concurrency = min(10, max(1, cpu_count))

        self.max_concurrency = max(1, parsed_concurrency)

        rate_limit_env = os.getenv("DOCINTEL_EXTRACTION_RATE_LIMIT_SECONDS", "0")
        try:
            self.rate_limit_delay = max(0.0, float(rate_limit_env))
        except ValueError:
            logging.warning(
                "Invalid DOCINTEL_EXTRACTION_RATE_LIMIT_SECONDS value '%s'; disabling client-side rate limiting",
                rate_limit_env,
            )
            self.rate_limit_delay = 0.0
    
    async def extract_from_chunks(
        self,
        nct_id: Optional[str] = None,
        limit: Optional[int] = None,
        output_file: Optional[str] = None,
        fast_mode: bool = False,
        skip_relations: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract clinical entities and relations from document chunks.
        
        Args:
            nct_id: Process only specific NCT ID
            limit: Maximum number of chunks to process
            output_file: Save results to JSON file
            fast_mode: Use only medspaCy/QuickUMLS (skip GPT)
            skip_relations: Skip relation extraction
            
        Returns:
            Extraction summary report
        """
        # Use instance batch_size unless overridden
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        if fast_mode:
            logging.info("🔬 Starting FAST extraction (medspaCy only, no GPT)...")
        elif effective_batch_size > 0:
            logging.info(f"🔬 Starting BATCHED extraction (batch_size={effective_batch_size})...")
        else:
            logging.info("🔬 Starting clinical entity/relation extraction...")
        
        self.fast_mode = fast_mode
        self.skip_relations = skip_relations
        # Ensure fresh result buffer per run
        self.results = []

        chunks = self._discover_chunks(nct_id, limit)
        if not chunks:
            return {"status": "no_data", "message": "No chunks found to process"}

        logging.info(
            "📄 Found %s chunks to process (nct filter=%s, concurrency=%s)",
            len(chunks),
            nct_id or "all",
            self.max_concurrency,
        )

        total_entities = 0
        total_relations = 0
        total_normalized = 0
        processed = 0
        skipped = 0
        errors = 0

        # Process chunks in batches if batching is enabled
        if effective_batch_size > 0 and not fast_mode:
            # Batched processing
            for batch_start in range(0, len(chunks), effective_batch_size):
                batch = chunks[batch_start:batch_start + effective_batch_size]
                logging.info(f"📦 Processing batch {batch_start//effective_batch_size + 1}: chunks {batch_start}-{batch_start+len(batch)-1}")
                
                outcomes = await self._process_batch(batch)
                
                for outcome in outcomes:
                    status = outcome.get("status")
                    if status == "processed":
                        processed += 1
                        total_entities += outcome.get("entity_count", 0)
                        total_relations += outcome.get("relation_count", 0)
                        total_normalized += outcome.get("normalized_count", 0)
                        self.results.append(outcome["extraction"])
                    elif status == "skipped":
                        skipped += 1
                    elif status == "error":
                        errors += 1
        else:
            # Individual processing (fast mode or batching disabled)
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _bounded_process(chunk: Dict[str, Any]) -> Dict[str, Any]:
                return await self._process_chunk(chunk, semaphore)

            tasks = [asyncio.create_task(_bounded_process(chunk)) for chunk in chunks]

            for task in asyncio.as_completed(tasks):
                outcome = await task
                status = outcome.get("status")

                if status == "processed":
                    processed += 1
                    total_entities += outcome.get("entity_count", 0)
                    total_relations += outcome.get("relation_count", 0)
                    total_normalized += outcome.get("normalized_count", 0)
                    self.results.append(outcome["extraction"])
                elif status == "skipped":
                    skipped += 1
                elif status == "error":
                    errors += 1

        # Re-establish deterministic ordering for downstream consumers
        self.results.sort(
            key=lambda item: item.get("chunk_metadata", {}).get("sequence_index", 0)
        )

        report = {
            "status": "completed",
            "chunks_processed": processed,
            "chunks_discovered": len(chunks),
            "chunks_failed": errors,
            "chunks_skipped": skipped,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "normalized_entities": total_normalized,
            "results_count": len(self.results)
        }
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "report": report,
                "extractions": self.results
            }
            
            output_path.write_text(json.dumps(output_data, indent=2))
            logging.info(f"💾 Results saved to {output_path}")
            report["output_file"] = str(output_path)
        
        # If no output file specified, store directly in database
        if not output_file:
            logging.info("💾 No output file specified - storing entities directly in database...")
            try:
                from .knowledge_graph.graph_construction import KnowledgeGraphBuilder
                
                graph_builder = KnowledgeGraphBuilder()
                entities_stored = 0
                relations_stored = 0
                
                for extraction in self.results:
                    # Import here to avoid circular dependency
                    from .knowledge_graph.triple_extraction import ClinicalEntity, ClinicalRelation, TripleExtractionResult
                    
                    # Reconstruct entities from serialized data
                    entities = []
                    for e_data in extraction["entities"]:
                        entity = ClinicalEntity(
                            text=e_data["text"],
                            entity_type=e_data["type"],
                            start_char=e_data["start_char"],
                            end_char=e_data["end_char"],
                            confidence=e_data["confidence"],
                            normalized_id=e_data.get("normalized_id"),
                            normalized_source=e_data.get("normalized_source"),
                            context_flags=e_data.get("context_flags")
                        )
                        entities.append(entity)
                    
                    # Reconstruct relations
                    relations = []
                    for r_data in extraction["relations"]:
                        subject_entity = next((e for e in entities if e.text == r_data["subject"]), None)
                        object_entity = next((e for e in entities if e.text == r_data["object"]), None)
                        
                        if subject_entity and object_entity:
                            relation = ClinicalRelation(
                                subject_entity=subject_entity,
                                predicate=r_data["predicate"],
                                object_entity=object_entity,
                                confidence=r_data["confidence"],
                                evidence_span=r_data.get("evidence", ""),
                                evidence_start_char=0,
                                evidence_end_char=0
                            )
                            relations.append(relation)
                    
                    # Create extraction result
                    result = TripleExtractionResult(
                        entities=entities,
                        relations=relations,
                        processing_metadata=extraction.get("metadata", {})
                    )
                    
                    # Get chunk IDs
                    chunk_uuid = _stable_chunk_uuid(extraction["chunk_uuid"])
                    source_chunk_id = extraction["chunk_id"]
                    
                    # Store in database with source_chunk_id
                    try:
                        await graph_builder.create_meta_graph(chunk_uuid, result, source_chunk_id)
                        entities_stored += len(entities)
                        relations_stored += len(relations)
                    except Exception as e:
                        if "duplicate key" in str(e).lower():
                            logging.warning(f"⚠️  Skipping duplicate chunk {source_chunk_id}")
                        else:
                            raise
                
                logging.info(f"✅ Stored {entities_stored:,} entities and {relations_stored:,} relations in database")
                report["database_storage"] = {
                    "entities_stored": entities_stored,
                    "relations_stored": relations_stored
                }
            except Exception as e:
                logging.error(f"❌ Failed to store in database: {e}")
                raise
        
        logging.info("🎉 Extraction completed!")
        return report
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of chunks using the extractor's batch processing capability.
        
        Args:
            batch: List of chunk dictionaries
            
        Returns:
            List of outcome dictionaries, one per chunk
        """
        start_time = time.perf_counter()
        
        # Filter out short chunks
        valid_chunks = []
        outcomes = []
        
        for chunk in batch:
            if len(chunk["content"].strip()) < self.min_chunk_characters:
                logging.debug("Skipping short chunk %s", chunk["chunk_id"])
                outcomes.append({"status": "skipped", "chunk_id": chunk["chunk_id"]})
            else:
                valid_chunks.append(chunk)
        
        if not valid_chunks:
            return outcomes
        
        # Prepare batch input for extractor
        batch_input = [
            {
                "text": chunk["content"],
                "chunk_id": chunk["chunk_id"],
                "chunk_uuid": chunk["chunk_uuid"]
            }
            for chunk in valid_chunks
        ]
        
        try:
            # Call batch extraction (extractor is EnhancedClinicalTripleExtractor, it has base_extractor)
            results = self.extractor.base_extractor.extract_triples_batch(batch_input)
            
            # Process each result
            for i, (chunk, result) in enumerate(zip(valid_chunks, results)):
                chunk_id = chunk["chunk_id"]
                chunk_uuid = chunk["chunk_uuid"]
                nct_id = chunk["nct_id"]
                
                entities = result.entities
                relations = result.relations
                normalization_stats = getattr(result, "normalization_stats", {}) or {}
                
                processing_metadata = dict(result.processing_metadata or {})
                processing_metadata.setdefault("chunk_id", str(chunk_uuid))
                processing_metadata.setdefault("source_chunk_id", chunk_id)
                processing_metadata.setdefault("nct_id", nct_id)
                
                chunk_metadata = dict(chunk["metadata"])
                chunk_metadata.setdefault("sequence_index", chunk["sequence_index"])
                chunk_metadata.setdefault("chunk_uuid", str(chunk_uuid))
                
                extraction_data = {
                    "chunk_id": chunk_id,
                    "chunk_uuid": str(chunk_uuid),
                    "nct_id": nct_id,
                    "document_id": chunk["document_id"],
                    "entities": self._serialize_entities(result),
                    "relations": [
                        {
                            "subject": relation.subject_entity.text,
                            "subject_type": relation.subject_entity.entity_type,
                            "predicate": relation.predicate,
                            "object": relation.object_entity.text,
                            "object_type": relation.object_entity.entity_type,
                            "confidence": relation.confidence,
                            "evidence": relation.evidence_span,
                        }
                        for relation in relations
                    ],
                    "metadata": processing_metadata,
                    "chunk_metadata": chunk_metadata,
                }
                
                normalized_count = sum(
                    1 for entity in entities 
                    if entity.normalized_id or getattr(entity, "normalized_id", None)
                )
                
                outcomes.append({
                    "status": "processed",
                    "chunk_id": chunk_id,
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "normalized_count": normalized_count,
                    "extraction": extraction_data
                })
                
                logging.info(f"✅ Extracted {len(entities)} entities and {len(relations)} relations from chunk {chunk_id}")
            
            elapsed = time.perf_counter() - start_time
            logging.info(f"📦 Batch of {len(valid_chunks)} chunks completed in {elapsed:.2f}s")
            
        except Exception as exc:
            logging.error(f"❌ Batch processing failed: {exc}")
            for chunk in valid_chunks:
                outcomes.append({
                    "status": "error",
                    "chunk_id": chunk["chunk_id"],
                    "exception": str(exc)
                })
        
        return outcomes
    
    async def _process_chunk(
        self,
        chunk: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Process a single chunk respecting the shared concurrency semaphore."""

        async with semaphore:
            chunk_id = chunk["chunk_id"]
            chunk_uuid = chunk["chunk_uuid"]
            nct_id = chunk["nct_id"]
            content = chunk["content"]

            if len(content.strip()) < self.min_chunk_characters:
                logging.debug("Skipping short chunk %s (nct=%s)", chunk_id, nct_id)
                return {"status": "skipped", "chunk_id": chunk_id}

            logging.info("Processing chunk %s from %s", chunk_id, nct_id)

            start_time = time.perf_counter()
            try:
                result = await self.extractor.extract_and_normalize_triples(content, chunk_uuid)
            except Exception as exc:  # pragma: no cover - safety net for runtime issues
                logging.error("❌ Error processing chunk %s: %s", chunk_id, exc)
                return {"status": "error", "chunk_id": chunk_id, "exception": str(exc)}
            finally:
                elapsed = time.perf_counter() - start_time

            entities = result.entities
            relations = result.relations
            normalization_stats = getattr(result, "normalization_stats", {}) or {}

            processing_metadata = dict(result.processing_metadata or {})
            processing_metadata.setdefault("chunk_id", str(chunk_uuid))
            processing_metadata.setdefault("source_chunk_id", chunk_id)
            processing_metadata.setdefault("nct_id", nct_id)
            processing_metadata.setdefault("processing_time_seconds", elapsed)

            chunk_metadata = dict(chunk["metadata"])
            chunk_metadata.setdefault("sequence_index", chunk["sequence_index"])
            chunk_metadata.setdefault("chunk_uuid", str(chunk_uuid))

            extraction_data = {
                "chunk_id": chunk_id,
                "chunk_uuid": str(chunk_uuid),
                "nct_id": nct_id,
                "document_id": chunk["document_id"],
                "entities": self._serialize_entities(result),
                "relations": [
                    {
                        "subject": relation.subject_entity.text,
                        "subject_type": relation.subject_entity.entity_type,
                        "predicate": relation.predicate,
                        "object": relation.object_entity.text,
                        "object_type": relation.object_entity.entity_type,
                        "confidence": relation.confidence,
                        "evidence": relation.evidence_span,
                    }
                    for relation in relations
                ],
                "metadata": processing_metadata,
                "chunk_metadata": chunk_metadata,
            }

            if normalization_stats:
                extraction_data["normalization_stats"] = normalization_stats

            if self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)

            logging.info(
                "✅ Extracted %s entities and %s relations from chunk %s",
                len(entities),
                len(relations),
                chunk_id,
            )

            return {
                "status": "processed",
                "extraction": extraction_data,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "normalized_count": normalization_stats.get("normalized_entities", 0),
            }

    def _discover_chunks(self, nct_id: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
        """Discover available parsed chunks for processing."""

        chunks: List[Dict[str, Any]] = []

        if not self.chunk_root.exists():
            logging.warning("Chunk directory not found at %s", self.chunk_root)
            return chunks

        nct_directories = [path for path in self.chunk_root.iterdir() if path.is_dir() and path.name.startswith("NCT")]

        if nct_id:
            nct_directories = [path for path in nct_directories if path.name == nct_id]

        nct_directories.sort(key=lambda path: path.name)

        for nct_directory in nct_directories:
            chunk_files = sorted(nct_directory.glob("*.json"))
            for chunk_file in chunk_files:
                try:
                    payload = json.loads(chunk_file.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - log unexpected artefact format
                    logging.error("Failed to read chunk file %s: %s", chunk_file, exc)
                    continue

                if not isinstance(payload, list):
                    logging.warning("Chunk file %s did not contain a list payload", chunk_file)
                    continue

                for entry_index, entry in enumerate(payload):
                    if limit is not None and len(chunks) >= limit:
                        return chunks

                    if not isinstance(entry, dict):
                        logging.debug("Skipping malformed chunk entry %s in %s", entry_index, chunk_file)
                        continue

                    text = entry.get("text") or ""
                    chunk_id = entry.get("id") or entry.get("chunk_id")

                    if not chunk_id:
                        chunk_id = f"{nct_directory.name}-{chunk_file.stem}-{entry_index}"

                    chunk_uuid = _stable_chunk_uuid(chunk_id)

                    sequence_index = len(chunks)
                    chunk_metadata = {key: value for key, value in entry.items() if key != "text"}
                    chunk_metadata.setdefault("id", chunk_id)
                    chunk_metadata.setdefault("sequence_index", sequence_index)
                    chunk_metadata.setdefault("chunk_index", entry_index)
                    chunk_metadata.setdefault("source_document", chunk_file.stem)
                    chunk_metadata.setdefault("source_path", str(chunk_file))
                    chunk_metadata.setdefault("nct_id", nct_directory.name)

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_uuid": chunk_uuid,
                            "nct_id": nct_directory.name,
                            "document_id": chunk_file.stem,
                            "content": text,
                            "metadata": chunk_metadata,
                            "sequence_index": sequence_index,
                        }
                    )

        return chunks

    def _serialize_entities(self, result: Any) -> List[Dict[str, Any]]:
        """Convert extractor entities to JSON-serialisable dictionaries."""
        serialized: List[Dict[str, Any]] = []
        normalized_entities = getattr(result, "normalized_entities", []) or []

        if normalized_entities:
            for base_entity, enhanced_entity in zip(result.entities, normalized_entities):
                normalization_payload = getattr(enhanced_entity, "normalization_data", None)
                serialized.append(
                    {
                        "text": base_entity.text,
                        "type": base_entity.entity_type,
                        "start_char": base_entity.start_char,
                        "end_char": base_entity.end_char,
                        "confidence": base_entity.confidence,
                        "normalized_id": getattr(enhanced_entity, "normalized_id", None),
                        "normalized_source": getattr(enhanced_entity, "normalized_source", None),
                        "context_flags": getattr(enhanced_entity, "context_flags", None),
                        "normalization": normalization_payload,
                    }
                )
        else:
            for entity in result.entities:
                serialized.append(
                    {
                        "text": entity.text,
                        "type": entity.entity_type,
                        "start_char": entity.start_char,
                        "end_char": entity.end_char,
                        "confidence": entity.confidence,
                        "normalized_id": entity.normalized_id,
                        "normalized_source": entity.normalized_source,
                        "context_flags": entity.context_flags,
                        "normalization": None,
                    }
                )

        return serialized
    
async def _run_async(
    nct_id: Optional[str],
    limit: Optional[int],
    output_file: Optional[str],
    fast_mode: bool = False,
    skip_relations: bool = False,
    max_concurrency: Optional[int] = None,
    batch_size: int = 20
) -> Dict[str, Any]:
    """Async extraction runner."""
    job = ExtractionJob(
        max_concurrency=max_concurrency,
        fast_mode=fast_mode,
        skip_relations=skip_relations,
        batch_size=batch_size
    )
    return await job.extract_from_chunks(
        nct_id=nct_id,
        limit=limit,
        output_file=output_file,
        fast_mode=fast_mode,
        skip_relations=skip_relations,
        batch_size=batch_size
    )


def run(
    nct_id: Optional[str] = None,
    limit: Optional[int] = None,
    output_file: Optional[str] = None,
    fast_mode: bool = False,
    skip_relations: bool = False,
    max_concurrency: Optional[int] = None,
    batch_size: int = 20
) -> Dict[str, Any]:
    """Run clinical entity/relation extraction."""
    return asyncio.run(_run_async(nct_id, limit, output_file, fast_mode, skip_relations, max_concurrency, batch_size))


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for extraction."""
    parser = argparse.ArgumentParser(
        description="Extract clinical entities and relations from parsed documents"
    )
    parser.add_argument(
        "--nct-id",
        help="Process only specific NCT ID (e.g., NCT03981107)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of chunks to process"
    )
    parser.add_argument(
        "--output",
        help="Save extraction results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: use only medspaCy/QuickUMLS, skip slow GPT-4.1 API calls"
    )
    parser.add_argument(
        "--skip-relations",
        action="store_true",
        help="Skip relation extraction (only extract entities)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent extractions (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of chunks to batch per GPT call (0=disable batching, default: 20)"
    )
    
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    
    try:
        report = run(
            nct_id=args.nct_id,
            limit=args.limit,
            output_file=args.output,
            fast_mode=args.fast,
            skip_relations=args.skip_relations,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size
        )
        
        print("\n🔬 EXTRACTION SUMMARY:")
        print(f"   • Status: {report['status']}")
        print(f"   • Chunks processed: {report.get('chunks_processed', 0)}")
        print(f"   • Total entities: {report.get('total_entities', 0)}")
        print(f"   • Total relations: {report.get('total_relations', 0)}")
        if report.get("normalized_entities") is not None:
            print(f"   • Normalized entities: {report.get('normalized_entities', 0)}")
        
        if report.get('output_file'):
            print(f"   • Results saved: {report['output_file']}")
        
        if report['status'] == 'completed':
            print("\n✅ Ready for knowledge graph construction!")
            print("   Next: pixi run -- python -m docintel.graph")
        
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()