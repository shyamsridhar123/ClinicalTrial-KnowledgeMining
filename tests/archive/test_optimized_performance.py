#!/usr/bin/env python3
"""Advanced performance monitoring and optimization for embedding pipeline."""

import asyncio
import json
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Set up the module path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system resources and embedding performance."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = []
        self.gpu_available = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = []
        
        if self.gpu_available:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
                logger.info("🔍 GPU monitoring started")
            except ImportError:
                pass
    
    def record_metric(self, operation: str, duration: float, items_processed: int):
        """Record a performance metric."""
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metric = {
            "timestamp": time.time(),
            "operation": operation,
            "duration": duration,
            "items_processed": items_processed,
            "throughput": items_processed / duration if duration > 0 else 0,
            "cpu_percent": cpu_percent,
            "memory_used_gb": memory_info.used / (1024**3),
            "memory_percent": memory_info.percent,
        }
        
        if self.gpu_available:
            try:
                import torch
                metric.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "gpu_max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
                })
            except ImportError:
                pass
        
        self.metrics.append(metric)
        logger.info(f"📊 {operation}: {metric['throughput']:.1f} items/sec, CPU: {cpu_percent:.1f}%")
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        if not self.metrics or not self.start_time:
            return {}
        
        total_duration = time.time() - self.start_time
        total_items = sum(m["items_processed"] for m in self.metrics)
        avg_throughput = total_items / total_duration if total_duration > 0 else 0
        
        summary = {
            "total_duration": total_duration,
            "total_items_processed": total_items,
            "average_throughput": avg_throughput,
            "peak_cpu_percent": max(m["cpu_percent"] for m in self.metrics),
            "peak_memory_gb": max(m["memory_used_gb"] for m in self.metrics),
        }
        
        if self.gpu_available and any("gpu_max_memory_gb" in m for m in self.metrics):
            summary.update({
                "peak_gpu_memory_gb": max(m.get("gpu_max_memory_gb", 0) for m in self.metrics),
                "gpu_utilization": "Active" if summary.get("peak_gpu_memory_gb", 0) > 0 else "Inactive"
            })
        
        return summary

async def run_optimized_embedding_test():
    """Run optimized embedding pipeline with performance monitoring."""
    
    logger.info("🚀 Starting optimized embedding performance test")
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Import after path setup
    from docintel.embed import run
    
    try:
        # Run optimized embedding pipeline
        start_time = time.time()
        
        logger.info("⚡ Running embedding pipeline with all optimizations:")
        logger.info("  - Model persistence (shared instance)")
        logger.info("  - PyTorch 2.0 compilation")
        logger.info("  - Mixed precision (FP16/TF32)")
        logger.info("  - Optimal batch size (8)")
        logger.info("  - Parallel document processing")
        logger.info("  - GPU memory optimization")
        logger.info("  - Embedding quantization")
        
        # Run the optimized embedding pipeline
        result = run(force_reembed=True, batch_size=8)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate performance metrics
        total_embeddings = result.get("total_embeddings", 0)
        documents_processed = result.get("documents_processed", 0)
        
        monitor.record_metric("full_pipeline", duration, total_embeddings)
        
        # Get performance summary
        summary = monitor.get_summary()
        
        # Enhanced results
        enhanced_results = {
            "optimization_results": {
                "total_duration_seconds": duration,
                "total_embeddings_generated": total_embeddings,
                "documents_processed": documents_processed,
                "throughput_embeddings_per_second": total_embeddings / duration if duration > 0 else 0,
                "throughput_documents_per_second": documents_processed / duration if duration > 0 else 0,
            },
            "system_performance": summary,
            "optimizations_applied": [
                "Model persistence (no reloading)",
                "PyTorch 2.0 compilation (max-autotune)",
                "Mixed precision (FP16/TF32)",
                "Optimal batch size (8)",
                "Parallel document processing",
                "GPU memory management",
                "L2 normalization",
                "Embedding quantization (int8)"
            ],
            "raw_metrics": monitor.metrics
        }
        
        # Save detailed results
        results_file = "optimized_embedding_performance.json"
        with open(results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        # Display results
        logger.info("🎉 Optimized Embedding Pipeline Results:")
        logger.info(f"  ⏱️  Total time: {duration:.1f} seconds")
        logger.info(f"  📊 Total embeddings: {total_embeddings}")
        logger.info(f"  📄 Documents processed: {documents_processed}")
        logger.info(f"  🚀 Throughput: {total_embeddings/duration:.1f} embeddings/sec")
        logger.info(f"  💾 Peak memory: {summary.get('peak_memory_gb', 0):.1f} GB")
        
        if summary.get("peak_gpu_memory_gb"):
            logger.info(f"  🎮 Peak GPU memory: {summary['peak_gpu_memory_gb']:.1f} GB")
        
        logger.info(f"  📁 Detailed results saved to: {results_file}")
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"❌ Optimization test failed: {e}")
        raise

def compare_performance_improvements():
    """Compare performance before and after optimizations."""
    
    logger.info("📈 Performance Comparison Analysis")
    
    # Load benchmark results if available
    benchmark_file = "embedding_benchmark_results.json"
    if Path(benchmark_file).exists():
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        
        logger.info("📊 Baseline Performance (from benchmark):")
        optimal_batch = benchmark_data.get("optimal_batch_size", 8)
        logger.info(f"  - Optimal batch size identified: {optimal_batch}")
        
        # Find best throughput from benchmark
        best_result = None
        for result in benchmark_data.get("benchmark_results", []):
            if result.get("batch_size") == optimal_batch:
                best_result = result
                break
        
        if best_result:
            logger.info(f"  - Best throughput: {best_result['throughput']:.1f} texts/sec")
            logger.info(f"  - Memory usage: {best_result['memory_gb']:.1f} GB")
    
    # Performance targets from TRD
    logger.info("🎯 Performance Targets:")
    logger.info("  - 3000-page document processing: <10 minutes")
    logger.info("  - Concurrent documents: ≥150")
    logger.info("  - GPU utilization: 85-95%")
    logger.info("  - Semantic query latency: <1.2s")

async def main():
    """Run comprehensive performance optimization test."""
    
    logger.info("🔬 Advanced Embedding Performance Optimization Test")
    
    try:
        # Run optimized embedding test
        results = await run_optimized_embedding_test()
        
        # Performance comparison
        compare_performance_improvements()
        
        # Recommendations
        logger.info("💡 Next Optimization Opportunities:")
        logger.info("  1. Vector database integration (ChromaDB/Qdrant)")
        logger.info("  2. Semantic search optimization")
        logger.info("  3. Distributed processing across GPUs")
        logger.info("  4. ONNX model export for production")
        logger.info("  5. Embedding index optimization")
        
        logger.info("✅ Performance optimization test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())