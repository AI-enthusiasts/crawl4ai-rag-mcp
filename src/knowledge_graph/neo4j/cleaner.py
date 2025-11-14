"""Neo4j repository cleanup operations"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def clear_repository_data(driver: Any, repo_name: str) -> None:
    """Clear all data for a specific repository with production-ready error handling and transaction management.

    This method uses a single Neo4j transaction to ensure atomicity - either all cleanup operations
    succeed or none are applied. The deletion order follows dependency hierarchy to prevent constraint violations:
    1. Methods and Attributes (depend on Classes)
    2. Functions (depend on Files)
    3. Classes (depend on Files)
    4. Files (depend on Repository)
    5. Branches and Commits (depend on Repository)
    6. Repository (root node)

    Args:
        driver: Neo4j driver instance
        repo_name: Name of the repository to clear

    Raises:
        Exception: If repository validation fails or Neo4j operations encounter errors
    """
    logger.info(f"Starting cleanup for repository: {repo_name}")

    # Validate that repository exists before attempting cleanup
    async with driver.session() as session:
        try:
            result = await session.run(
                "MATCH (r:Repository {name: $repo_name}) RETURN count(r) as repo_count",
                repo_name=repo_name,
            )
            record = await result.single()
            if not record or record["repo_count"] == 0:
                logger.warning(f"Repository '{repo_name}' not found in database - nothing to clean")
                return

            logger.info(f"Confirmed repository '{repo_name}' exists, proceeding with cleanup")
        except Exception as e:
            logger.exception(f"Failed to validate repository '{repo_name}': {e}")
            msg = f"Repository validation failed: {e}"
            raise Exception(msg)

    # Track cleanup statistics for logging
    cleanup_stats = {
        "methods": 0,
        "attributes": 0,
        "functions": 0,
        "classes": 0,
        "files": 0,
        "branches": 0,
        "commits": 0,
        "repository": 0,
    }

    # Execute all cleanup operations within a single transaction
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            logger.info("Starting transactional cleanup operations...")

            # Step 1: Delete methods and attributes (they depend on classes)
            logger.debug("Deleting methods...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
                OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
                DETACH DELETE m
                RETURN count(m) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["methods"] = record["deleted_count"] if record else 0

            logger.debug("Deleting attributes...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
                OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
                DETACH DELETE a
                RETURN count(a) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["attributes"] = record["deleted_count"] if record else 0

            # Step 2: Delete functions (they depend on files)
            logger.debug("Deleting functions...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
                OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
                DETACH DELETE func
                RETURN count(func) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["functions"] = record["deleted_count"] if record else 0

            # Step 3: Delete classes (they depend on files)
            logger.debug("Deleting classes...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
                OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
                DETACH DELETE c
                RETURN count(c) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["classes"] = record["deleted_count"] if record else 0

            # Step 4: Delete files (they depend on repository)
            logger.debug("Deleting files...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})
                OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
                DETACH DELETE f
                RETURN count(f) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["files"] = record["deleted_count"] if record else 0

            # Step 5: Delete branches and commits (they depend on repository)
            # This is the key fix for HAS_COMMIT relationship warnings
            logger.debug("Deleting branches...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})
                OPTIONAL MATCH (r)-[:HAS_BRANCH]->(b:Branch)
                DETACH DELETE b
                RETURN count(b) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["branches"] = record["deleted_count"] if record else 0

            logger.debug("Deleting commits...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})
                OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
                DETACH DELETE c
                RETURN count(c) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["commits"] = record["deleted_count"] if record else 0

            # Step 6: Finally delete the repository
            logger.debug("Deleting repository...")
            result = await tx.run("""
                MATCH (r:Repository {name: $repo_name})
                DETACH DELETE r
                RETURN count(r) as deleted_count
            """, repo_name=repo_name)
            record = await result.single()
            cleanup_stats["repository"] = record["deleted_count"] if record else 0

            # Commit the transaction
            await tx.commit()
            logger.info("Transaction committed successfully")

        except Exception as e:
            # Rollback the transaction on any error
            logger.exception(f"Error during cleanup transaction, rolling back: {e}")
            await tx.rollback()
            msg = f"Repository cleanup failed and was rolled back: {e}"
            raise Exception(msg)

    # Log cleanup statistics
    total_deleted = sum(cleanup_stats.values())
    logger.info(f"Successfully cleared repository '{repo_name}' - {total_deleted} total nodes deleted:")
    for entity_type, count in cleanup_stats.items():
        if count > 0:
            logger.info(f"  - {entity_type}: {count}")

    if total_deleted == 0:
        logger.info("Repository was already empty or contained no data to clean")
