#!/usr/bin/env python3
"""Simple test to process DV007 chunks with direct embeddings."""

import asyncio
import json
import logging
from pathlib import Path

# Set up the module path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.config import get_embedding_settings
from docintel.embeddings.client import EmbeddingClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def process_dv007_chunks():
    """Process DV007 chunks with direct embeddings."""
    
    # Load the chunk file
    chunk_file = Path("data/processing/chunks/DV07_test/DV07.json")
    if not chunk_file.exists():
        logger.error(f"Chunk file not found: {chunk_file}")
        return
        
    logger.info(f"Loading chunks from: {chunk_file}")
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
    
    # The chunk file is a JSON array of chunks
    if not isinstance(chunk_data, list):
        logger.error("Expected chunk data to be a list")
        return
        
    chunks = chunk_data
    logger.info(f"Found {len(chunks)} chunks")
    
    # Extract text content from chunks
    texts = []
    chunk_metadata = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if text:
            texts.append(text)
            chunk_metadata.append({
                "chunk_index": i,
                "page": chunk.get("page", None),
                "section": chunk.get("section", None),
                "element_type": chunk.get("element_type", None),
                "char_count": len(text),
                "chunk_id": chunk.get("id", f"chunk-{i:04d}"),
                "nct_id": "DV07_test",
                "document_name": chunk_file.name,
            })
    
    logger.info(f"Extracted {len(texts)} non-empty text chunks")
    
    # Initialize embeddings
    settings = get_embedding_settings()
    client = EmbeddingClient(settings)
    
    try:
        # Process in batches
        batch_size = settings.embedding_batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadata = chunk_metadata[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} texts)")
            
            # Generate embeddings
            responses = await client.embed_texts(batch_texts)

            # Combine with metadata
            for j, response in enumerate(responses):
                metadata = dict(batch_metadata[j])
                metadata.setdefault("chunk_id", metadata.get("chunk_id") or f"chunk-{i+j:04d}")
                metadata["model"] = response.model
                metadata["dimension"] = len(response.embedding)
                all_embeddings.append({
                    "chunk_id": metadata["chunk_id"],
                    "embedding": response.embedding,
                    "metadata": metadata,
                })
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        
        # Save results as JSONL to match pipeline output
        output_dir = Path("data/processing/embeddings/vectors/DV07_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "DV07.jsonl"
        with output_file.open('w', encoding='utf-8') as f:
            for record in all_embeddings:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

        logger.info(f"Saved embeddings to: {output_file}")
        
        # Print summary
        if all_embeddings:
            dimensions = set(e["metadata"].get("dimension") for e in all_embeddings)
            char_counts = [e["metadata"].get("char_count", 0) for e in all_embeddings]
            
            logger.info(f"Summary:")
            logger.info(f"  Total embeddings: {len(all_embeddings)}")
            logger.info(f"  Embedding dimension: {dimensions}")
            logger.info(f"  Text length range: {min(char_counts)} - {max(char_counts)} chars")
            logger.info(f"  Average text length: {sum(char_counts) / len(char_counts):.1f} chars")
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(process_dv007_chunks())