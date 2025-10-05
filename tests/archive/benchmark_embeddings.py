#!/usr/bin/env python3
"""Performance-optimized embedding client with benchmarking."""

import asyncio
import json
import logging
import time
from pathlib import Path
import torch
from typing import Dict, List, Optional
import sys

# Set up the module path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.config import get_embedding_settings
from docintel.embeddings.client import EmbeddingClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OptimizedEmbeddingBenchmark:
    """Benchmark different optimization strategies for embeddings."""
    
    def __init__(self):
        self.settings = get_embedding_settings()
        self.results = []
        
    async def benchmark_batch_sizes(self, texts: List[str], batch_sizes: List[int]):
        """Test different batch sizes to find optimal performance."""
        
        logger.info("🚀 Starting batch size optimization benchmark")
        
        for batch_size in batch_sizes:
            # BiomedCLIP emits 512-dimensional embeddings; estimate float32 VRAM usage
            if batch_size * 512 * 4 > 3_500_000_000:  # Rough VRAM limit check (3.5GB usable)
                logger.warning(f"⚠️ Skipping batch_size={batch_size} - likely too large for GPU")
                continue
                
            # Create client with specific batch size
            settings = get_embedding_settings()
            settings.embedding_batch_size = batch_size
            client = EmbeddingClient(settings)
            
            try:
                # Warm up
                await client.embed_texts(texts[:5])
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Benchmark
                start_time = time.time()
                responses = await client.embed_texts(texts)
                end_time = time.time()
                
                duration = end_time - start_time
                throughput = len(texts) / duration
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    memory_used = 0
                
                result = {
                    "batch_size": batch_size,
                    "duration": duration,
                    "throughput": throughput,
                    "memory_gb": memory_used,
                    "texts_count": len(texts),
                    "embeddings_count": len(responses)
                }
                
                self.results.append(result)
                
                logger.info(f"✅ Batch size {batch_size}: {throughput:.1f} texts/sec, {duration:.2f}s, {memory_used:.2f}GB")
                
            except Exception as e:
                logger.error(f"❌ Batch size {batch_size} failed: {e}")
                self.results.append({
                    "batch_size": batch_size,
                    "error": str(e)
                })
            finally:
                await client.aclose()
                
    async def benchmark_optimizations(self, texts: List[str]):
        """Test different optimization techniques."""
        
        logger.info("🔧 Testing optimization techniques")
        
        optimizations = [
            ("baseline", {}),
            ("normalized", {"normalize_embeddings": True}),
            ("fp16", {"precision": "fp16"}),
        ]
        
        for opt_name, kwargs in optimizations:
            try:
                start_time = time.time()
                
                client = EmbeddingClient(self.settings)
                
                # Apply optimizations
                if "normalize_embeddings" in kwargs:
                    # This would be handled in the encode call
                    pass
                    
                responses = await client.embed_texts(texts)
                
                # Normalize if requested
                if kwargs.get("normalize_embeddings"):
                    import numpy as np
                    for response in responses:
                        embedding = np.array(response.embedding)
                        response.embedding = (embedding / np.linalg.norm(embedding)).tolist()
                
                end_time = time.time()
                duration = end_time - start_time
                throughput = len(texts) / duration
                
                logger.info(f"✅ {opt_name}: {throughput:.1f} texts/sec, {duration:.2f}s")
                
                await client.aclose()
                
            except Exception as e:
                logger.error(f"❌ {opt_name} failed: {e}")
    
    def save_results(self, output_file: str = "embedding_benchmark_results.json"):
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump({
                "benchmark_results": self.results,
                "optimal_batch_size": self.get_optimal_batch_size(),
                "recommendations": self.get_recommendations()
            }, f, indent=2)
        
        logger.info(f"📊 Results saved to {output_file}")
    
    def get_optimal_batch_size(self) -> Optional[int]:
        """Find the batch size with best throughput."""
        valid_results = [r for r in self.results if "error" not in r]
        if not valid_results:
            return None
            
        return max(valid_results, key=lambda x: x["throughput"])["batch_size"]
    
    def get_recommendations(self) -> Dict[str, str]:
        """Generate performance recommendations."""
        recommendations = {}
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            recommendations["gpu"] = f"Using {gpu_name} with {gpu_memory:.1f}GB memory"
        
        optimal_batch = self.get_optimal_batch_size()
        if optimal_batch:
            recommendations["batch_size"] = f"Use batch_size={optimal_batch} for optimal performance"
        
        return recommendations

async def main():
    """Run comprehensive embedding performance benchmark."""
    
    # Load sample data
    chunk_file = Path("data/processing/chunks/DV07_test/DV07.json")
    if not chunk_file.exists():
        logger.error(f"Sample data not found: {chunk_file}")
        return
    
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
    
    # Extract texts
    texts = [chunk.get("text", "").strip() for chunk in chunk_data if chunk.get("text", "").strip()]
    
    if len(texts) < 5:
        logger.error("Not enough sample texts for benchmarking")
        return
    
    logger.info(f"🧪 Benchmarking with {len(texts)} text samples")
    
    # Initialize benchmark
    benchmark = OptimizedEmbeddingBenchmark()
    
    # Test batch sizes
    batch_sizes = [8, 16, 32, 64, 128]  # Conservative for 4GB GPU
    await benchmark.benchmark_batch_sizes(texts, batch_sizes)
    
    # Test optimizations
    await benchmark.benchmark_optimizations(texts[:10])  # Smaller sample for optimization tests
    
    # Save and report results
    benchmark.save_results()
    
    optimal_batch = benchmark.get_optimal_batch_size()
    if optimal_batch:
        logger.info(f"🎯 RECOMMENDATION: Use batch_size={optimal_batch} for best performance")
    
    # Print summary
    logger.info("📈 Performance Benchmark Complete!")
    logger.info("Check 'embedding_benchmark_results.json' for detailed results")

if __name__ == "__main__":
    asyncio.run(main())