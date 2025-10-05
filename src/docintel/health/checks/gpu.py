"""GPU health check."""

from ..base import HealthCheck, HealthStatus


class GPUHealthCheck(HealthCheck):
    """Check GPU availability and status."""
    
    name = "GPU (CUDA)"
    timeout_seconds = 2.0
    
    async def _perform_check(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return (
                    HealthStatus.HEALTHY,
                    f"{gpu_name} ({gpu_memory:.1f}GB)",
                    {
                        "name": gpu_name,
                        "memory_gb": round(gpu_memory, 1),
                        "cuda_version": torch.version.cuda
                    }
                )
            else:
                return (
                    HealthStatus.WARNING,
                    "CUDA not available - using CPU",
                    {"cuda_available": False}
                )
        except ImportError:
            return (
                HealthStatus.WARNING,
                "PyTorch not installed",
                {"torch_available": False}
            )
