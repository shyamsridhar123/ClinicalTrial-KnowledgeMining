"""System resources health check."""

import shutil
from pathlib import Path
from ..base import HealthCheck, HealthStatus


class SystemResourcesHealthCheck(HealthCheck):
    """Check disk space and system resources."""
    
    name = "System Resources"
    timeout_seconds = 2.0
    
    async def _perform_check(self):
        try:
            from docintel.config import get_settings
            settings = get_settings()
            
            # Check disk space
            data_root = settings.storage_root
            usage = shutil.disk_usage(data_root)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_pct = (usage.used / usage.total) * 100
            
            # Check memory if psutil available
            mem_info = {}
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_info = {
                    "total_gb": round(mem.total / (1024**3), 1),
                    "available_gb": round(mem.available / (1024**3), 1),
                    "used_percent": round(mem.percent, 1)
                }
            except ImportError:
                mem_info = {"note": "Install psutil for memory stats"}
            
            details = {
                "disk": {
                    "path": str(data_root),
                    "free_gb": round(free_gb, 1),
                    "total_gb": round(total_gb, 1),
                    "used_percent": round(used_pct, 1)
                },
                "memory": mem_info
            }
            
            if free_gb < 5:
                return (
                    HealthStatus.ERROR,
                    f"Critical: {free_gb:.1f}GB free",
                    details
                )
            elif free_gb < 20:
                return (
                    HealthStatus.WARNING,
                    f"Low disk space: {free_gb:.1f}GB free",
                    details
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"Sufficient: {free_gb:.1f}GB free",
                    details
                )
        except Exception as e:
            return (
                HealthStatus.ERROR,
                f"Check failed: {str(e)[:100]}",
                {"error": str(e)[:100]}
            )
