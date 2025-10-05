"""Model cache health check."""

from pathlib import Path
from ..base import HealthCheck, HealthStatus


class ModelCacheHealthCheck(HealthCheck):
    """Check local model cache status."""
    
    name = "Model Cache"
    timeout_seconds = 2.0
    
    async def _perform_check(self):
        models_dir = Path("models")
        
        expected_models = {
            "granite-docling": "models--ibm-granite--granite-docling-258M",
            "biomedclip": "biomedclip",
            "embedding": "models--sentence-transformers--embeddinggemma-300m-medical"
        }
        
        found = {}
        missing = []
        sizes = {}
        
        for name, path in expected_models.items():
            model_path = models_dir / path
            if model_path.exists():
                found[name] = True
                # Calculate size
                size_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                size_mb = size_bytes / (1024**2)
                sizes[name] = f"{size_mb:.1f}MB" if size_mb < 1024 else f"{size_mb/1024:.2f}GB"
            else:
                found[name] = False
                missing.append(name)
        
        if not missing:
            return (
                HealthStatus.HEALTHY,
                f"All {len(expected_models)} models cached",
                {"models": found, "sizes": sizes, "cache_dir": str(models_dir)}
            )
        elif len(missing) < len(expected_models):
            return (
                HealthStatus.WARNING,
                f"Missing: {', '.join(missing)}",
                {"found": found, "sizes": sizes, "missing": missing}
            )
        else:
            return (
                HealthStatus.ERROR,
                "No models cached - run warm-cache",
                {"models": found, "cache_dir": str(models_dir)}
            )
