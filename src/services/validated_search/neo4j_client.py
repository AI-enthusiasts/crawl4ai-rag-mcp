"""Neo4j client for structural validation queries.

This module handles Neo4j connection management and executes queries
to validate code structure against the knowledge graph.
"""

import logging
import os
from typing import Any

from neo4j import AsyncGraphDatabase

from src.core.exceptions import QueryError

logger = logging.getLogger(__name__)


class Neo4jValidationClient:
    """Client for Neo4j structural validation queries."""

    def __init__(self, neo4j_driver: Any | None = None) -> None:
        """Initialize Neo4j validation client.

        Args:
            neo4j_driver: Optional pre-configured Neo4j driver
        """
        self.neo4j_driver = neo4j_driver
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_enabled = bool(self.neo4j_uri and self.neo4j_password)

    async def get_session(self) -> Any | None:
        """Get or create Neo4j session.

        Returns:
            Neo4j session or None if Neo4j is not configured
        """
        if not self.neo4j_enabled:
            return None

        if not self.neo4j_driver:
            # Import notification suppression (available in neo4j>=5.21.0)
            try:
                from neo4j import NotificationMinimumSeverity

                # Create Neo4j driver with notification suppression
                self.neo4j_driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    warn_notification_severity=NotificationMinimumSeverity.OFF,
                )
            except (ImportError, AttributeError):
                # Fallback for older versions - use logging suppression
                import logging

                logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
                self.neo4j_driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )

        return self.neo4j_driver.session()

    async def check_repository_exists(self, session: Any, source_id: str) -> bool:
        """Check if repository exists in Neo4j.

        Args:
            session: Neo4j session
            source_id: Repository identifier

        Returns:
            True if repository exists, False otherwise
        """
        if not source_id:
            return False

        try:
            query = """
            MATCH (r:Repository {name: $repo_name})
            RETURN count(r) > 0 as exists
            """
            result = await session.run(query, repo_name=source_id)
            record = await result.single()
            return record["exists"] if record else False
        except QueryError as e:
            logger.warning(f"Query error checking repository existence: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking repository existence: {e}")
            return False

    async def check_class_exists(
        self, session: Any, class_name: str, source_id: str,
    ) -> bool:
        """Check if class exists in the repository.

        Args:
            session: Neo4j session
            class_name: Name of the class
            source_id: Repository identifier

        Returns:
            True if class exists, False otherwise
        """
        if not class_name:
            return False

        try:
            query = """
            MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
            WHERE c.name = $class_name OR c.full_name = $class_name
            RETURN count(c) > 0 as exists
            """
            result = await session.run(
                query, repo_name=source_id, class_name=class_name,
            )
            record = await result.single()
            return record["exists"] if record else False
        except QueryError as e:
            logger.warning(f"Query error checking class existence: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking class existence: {e}")
            return False

    async def check_method_exists(
        self,
        session: Any,
        method_name: str,
        class_name: str,
        source_id: str,
    ) -> bool:
        """Check if method exists in the specified class.

        Args:
            session: Neo4j session
            method_name: Name of the method
            class_name: Name of the class (empty string for any class)
            source_id: Repository identifier

        Returns:
            True if method exists, False otherwise
        """
        if not method_name:
            return False

        try:
            if class_name:
                query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE (c.name = $class_name OR c.full_name = $class_name) AND m.name = $method_name
                RETURN count(m) > 0 as exists
                """
                result = await session.run(
                    query,
                    repo_name=source_id,
                    class_name=class_name,
                    method_name=method_name,
                )
            else:
                # Search for method across all classes
                query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE m.name = $method_name
                RETURN count(m) > 0 as exists
                """
                result = await session.run(
                    query, repo_name=source_id, method_name=method_name,
                )

            record = await result.single()
            return record["exists"] if record else False
        except QueryError as e:
            logger.warning(f"Query error checking method existence: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking method existence: {e}")
            return False

    async def check_function_exists(
        self, session: Any, function_name: str, source_id: str,
    ) -> bool:
        """Check if standalone function exists in the repository.

        Args:
            session: Neo4j session
            function_name: Name of the function
            source_id: Repository identifier

        Returns:
            True if function exists, False otherwise
        """
        if not function_name:
            return False

        try:
            query = """
            MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
            WHERE func.name = $function_name
            RETURN count(func) > 0 as exists
            """
            result = await session.run(
                query, repo_name=source_id, function_name=function_name,
            )
            record = await result.single()
            return record["exists"] if record else False
        except QueryError as e:
            logger.warning(f"Query error checking function existence: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking function existence: {e}")
            return False

    async def validate_class_structure(
        self,
        session: Any,
        class_name: str,
        metadata: dict[str, Any],
        source_id: str,
    ) -> bool:
        """Validate class structure against metadata.

        Args:
            session: Neo4j session
            class_name: Name of the class
            metadata: Code metadata
            source_id: Repository identifier

        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # This is a placeholder for more sophisticated structure validation
            # Could check method counts, attribute presence, etc.
            return True
        except Exception as e:
            logger.warning(f"Unexpected error validating class structure: {e}")
            return False

    async def validate_method_signature(
        self,
        session: Any,
        method_name: str,
        class_name: str,
        metadata: dict[str, Any],
        source_id: str,
    ) -> bool:
        """Validate method signature against metadata.

        Args:
            session: Neo4j session
            method_name: Name of the method
            class_name: Name of the class
            metadata: Code metadata
            source_id: Repository identifier

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # This is a placeholder for method signature validation
            # Could check parameter counts, types, return types, etc.
            return True
        except Exception as e:
            logger.warning(f"Unexpected error validating method signature: {e}")
            return False
