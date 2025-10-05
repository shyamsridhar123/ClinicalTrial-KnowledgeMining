#!/usr/bin/env python3
"""Quick test of direct BiomedCLIP embeddings."""

import asyncio
import logging
from pathlib import Path
import sys

# Set up the module path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.config import get_embedding_settings
from docintel.embeddings.client import EmbeddingClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_embeddings():
    """Test embedding generation with sample clinical text."""

    # Sample clinical trial texts
    sample_texts = [
        "This randomized controlled trial evaluated the efficacy of the investigational drug in patients with advanced cancer.",
        "Primary endpoint was overall survival measured from randomization to death from any cause.",
        "Adverse events were graded according to Common Terminology Criteria for Adverse Events version 5.0.",
        "The study population included adult patients aged 18 years or older with histologically confirmed diagnosis.",
    ]

    # Initialize settings and client
    settings = get_embedding_settings()
    logger.info("Using model: %s", settings.embedding_model_name)
    logger.info("Max tokens: %s", settings.embedding_max_tokens)

    client = EmbeddingClient(settings)

    try:
        # Generate embeddings
        logger.info("Generating embeddings for %d texts...", len(sample_texts))
        responses = await client.embed_texts(sample_texts)

        # Log results
        logger.info("Generated %d embeddings", len(responses))
        if responses:
            first_embedding = responses[0]
            logger.info(
                "First embedding: dimension=%d, model=%s", len(first_embedding.embedding), first_embedding.model
            )
            logger.info("Sample values: %s", first_embedding.embedding[:5])

        for i, response in enumerate(responses):
            logger.info(
                "Text %d: %d chars -> %d dims", i, len(sample_texts[i]), len(response.embedding)
            )

        return responses

    except Exception as exc:  # noqa: BLE001 - demo script
        logger.error("Embedding failed: %s", exc)
        raise
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(test_embeddings())
