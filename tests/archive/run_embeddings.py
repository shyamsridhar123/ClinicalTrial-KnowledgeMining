#!/usr/bin/env python3
"""CLI to run embedding pipeline on parsed documents."""

import asyncio
import logging
import argparse
from pathlib import Path

# Set up the module path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.config import get_embedding_settings
from docintel.embeddings.phase import EmbeddingPhase
from docintel.pipeline import PipelineContext

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_embeddings(document_id: str, max_workers: int = 1):
    """Run embedding pipeline for a specific document."""
    
    logger.info(f"Starting embedding pipeline for document: {document_id}")
    
    # Create pipeline context
    context = PipelineContext(
        document_id=document_id,
        max_workers=max_workers,
        metadata={"source": "cli"},
    )
    
    # Initialize embedding phase
    settings = get_embedding_settings()
    phase = EmbeddingPhase(settings)
    
    try:
        # Run the embedding phase
        result = await phase.run(context)
        
        logger.info(f"Embedding pipeline completed successfully")
        logger.info(f"Status: {result.status}")
        logger.info(f"Message: {result.message}")
        
        if result.metadata:
            logger.info(f"Metadata: {result.metadata}")
            
        return result
        
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run embedding pipeline on parsed documents")
    parser.add_argument("--document-id", required=True, help="Document ID to process")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of workers")
    
    args = parser.parse_args()
    
    # Run the embedding pipeline
    asyncio.run(run_embeddings(args.document_id, args.max_workers))

if __name__ == "__main__":
    main()