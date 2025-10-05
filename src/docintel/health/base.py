"""Base classes for health checks."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = 0.0
    error: Optional[Exception] = None


class HealthCheck:
    """Base class for all health checks."""
    
    name: str = "Base Health Check"
    timeout_seconds: float = 5.0
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check and return result."""
        start = time.time()
        try:
            status, message, details = await self._perform_check()
            duration = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.ERROR,
                message=f"Check failed: {str(e)}",
                details={},
                duration_ms=duration,
                timestamp=time.time(),
                error=e
            )
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this in subclasses."""
        raise NotImplementedError
