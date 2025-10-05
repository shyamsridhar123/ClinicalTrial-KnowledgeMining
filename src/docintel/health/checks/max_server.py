"""Modular MAX server health check."""

import aiohttp
from ..base import HealthCheck, HealthStatus


class MaxServerHealthCheck(HealthCheck):
    """Check Modular MAX server status."""
    
    name = "Modular MAX (Granite Docling)"
    timeout_seconds = 5.0
    
    async def _perform_check(self):
        try:
            from docintel.config import get_parsing_settings
            settings = get_parsing_settings()
            base_url = str(settings.docling_max_base_url).rstrip("/")
            expected_model = settings.docling_model_name
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["id"] for m in data.get("data", [])]
                        
                        if expected_model in models:
                            return (
                                HealthStatus.HEALTHY,
                                f"Running with {len(models)} model(s)",
                                {"models": models, "endpoint": base_url}
                            )
                        else:
                            return (
                                HealthStatus.WARNING,
                                f"Running but {expected_model} not loaded",
                                {"models": models, "expected": expected_model}
                            )
                    else:
                        return (
                            HealthStatus.ERROR,
                            f"Server returned HTTP {resp.status}",
                            {"status_code": resp.status, "endpoint": base_url}
                        )
        except aiohttp.ClientError as e:
            return (
                HealthStatus.ERROR,
                f"Cannot connect: {str(e)[:80]}",
                {"error": str(e)[:80]}
            )
        except Exception as e:
            return (
                HealthStatus.ERROR,
                f"Check failed: {str(e)[:80]}",
                {"error": str(e)[:80]}
            )
