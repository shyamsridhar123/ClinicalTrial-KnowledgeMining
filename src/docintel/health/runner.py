"""Health check runner and orchestration."""

import asyncio
from typing import List
from .base import HealthCheckResult, HealthStatus
from .checks import (
    GPUHealthCheck,
    AzureOpenAIHealthCheck,
    MaxServerHealthCheck,
    DatabaseHealthCheck,
    ModelCacheHealthCheck,
    DataHealthCheck,
    SystemResourcesHealthCheck,
)


class HealthCheckRunner:
    """Orchestrate all health checks."""
    
    def __init__(self):
        self.checks = [
            GPUHealthCheck(),
            AzureOpenAIHealthCheck(),
            # MaxServerHealthCheck(),  # Disabled - not using Modular MAX/Mojo
            DatabaseHealthCheck(),
            ModelCacheHealthCheck(),
            DataHealthCheck(),
            SystemResourcesHealthCheck(),
        ]
    
    async def run_all(self) -> List[HealthCheckResult]:
        """Run all health checks in parallel."""
        tasks = [check.check() for check in self.checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(HealthCheckResult(
                    name=self.checks[i].name,
                    status=HealthStatus.ERROR,
                    message=f"Check crashed: {str(result)[:100]}",
                    details={},
                    duration_ms=0,
                    timestamp=0,
                    error=result
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def summarize(self, results: List[HealthCheckResult]) -> dict:
        """Generate summary statistics."""
        healthy = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        warning = sum(1 for r in results if r.status == HealthStatus.WARNING)
        error = sum(1 for r in results if r.status == HealthStatus.ERROR)
        
        overall = HealthStatus.HEALTHY
        if error > 0:
            overall = HealthStatus.ERROR
        elif warning > 0:
            overall = HealthStatus.WARNING
        
        return {
            "total": len(results),
            "healthy": healthy,
            "warning": warning,
            "error": error,
            "overall_status": overall
        }
