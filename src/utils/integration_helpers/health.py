"""Health monitoring utilities for integration layer.

Provides health checks for Neo4j and Qdrant components,
and overall integration health status.
"""

import logging
import time
from typing import Any

from src.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class IntegrationHealthMonitor:
    """Monitor the health of Neo4j-Qdrant integration components."""

    def __init__(self) -> None:
        """Initialize health monitor."""
        self.health_checks: dict[str, Any] = {}
        self.last_check_time: dict[str, float] = {}

    async def check_neo4j_health(self, neo4j_driver: Any) -> dict[str, Any]:
        """Check Neo4j connection health.

        Args:
            neo4j_driver: Neo4j driver instance

        Returns:
            Health status dictionary
        """
        try:
            if not neo4j_driver:
                return {"status": "unavailable", "reason": "No driver provided"}

            session = neo4j_driver.session()
            try:
                # Simple health query
                result = await session.run("RETURN 1 as health_check")
                record = await result.single()

                if record and record["health_check"] == 1:
                    return {
                        "status": "healthy",
                        "latency_ms": 0,
                    }  # Could measure actual latency
                return {"status": "unhealthy", "reason": "Unexpected query result"}

            finally:
                await session.close()

        except DatabaseError as e:
            return {"status": "error", "reason": f"Database error: {e!s}"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def check_qdrant_health(self, qdrant_client: Any) -> dict[str, Any]:
        """Check Qdrant connection health.

        Args:
            qdrant_client: Qdrant client instance

        Returns:
            Health status dictionary
        """
        try:
            if not qdrant_client:
                return {"status": "unavailable", "reason": "No client provided"}

            # Try to get collection info
            collections = await qdrant_client.get_collections()

            if collections is not None:
                return {
                    "status": "healthy",
                    "collections_count": len(collections)
                    if hasattr(collections, "__len__")
                    else 0,
                }
            return {"status": "unhealthy", "reason": "Could not retrieve collections"}

        except DatabaseError as e:
            return {"status": "error", "reason": f"Database error: {e!s}"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def get_integration_health(
        self,
        database_client: Any = None,
        neo4j_driver: Any = None,
    ) -> dict[str, Any]:
        """Get overall integration health status.

        Args:
            database_client: Qdrant database client
            neo4j_driver: Neo4j driver

        Returns:
            Comprehensive health status report
        """
        health_status: dict[str, Any] = {
            "overall_status": "unknown",
            "timestamp": time.time(),
            "components": {},
        }

        # Check Qdrant health
        qdrant_health = await self.check_qdrant_health(database_client)
        health_status["components"]["qdrant"] = qdrant_health

        # Check Neo4j health
        neo4j_health = await self.check_neo4j_health(neo4j_driver)
        health_status["components"]["neo4j"] = neo4j_health

        # Determine overall status
        qdrant_ok = qdrant_health["status"] in ["healthy", "unavailable"]
        neo4j_ok = neo4j_health["status"] in ["healthy", "unavailable"]

        if qdrant_ok and neo4j_ok:
            # Both components are working or gracefully unavailable
            if (
                qdrant_health["status"] == "healthy"
                and neo4j_health["status"] == "healthy"
            ):
                health_status["overall_status"] = "fully_operational"
            elif (
                qdrant_health["status"] == "healthy"
                or neo4j_health["status"] == "healthy"
            ):
                health_status["overall_status"] = "partially_operational"
            else:
                health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "error"

        return health_status


async def validate_integration_health(
    database_client: Any = None, neo4j_driver: Any = None,
) -> dict[str, Any]:
    """Quick health check for the integration layer.

    Args:
        database_client: Qdrant database client
        neo4j_driver: Neo4j driver

    Returns:
        Health status report
    """
    from .performance import get_performance_optimizer

    optimizer = get_performance_optimizer()
    return await optimizer.health_monitor.get_integration_health(
        database_client=database_client,
        neo4j_driver=neo4j_driver,
    )
