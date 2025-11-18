"""
Neo4j Graph Writer Module

Handles creation of nodes and relationships in Neo4j database for repository code analysis.
This module provides functions to:
- Create repository, file, class, method, function, and interface nodes
- Establish relationships between code elements and files
- Process modules in batches to prevent memory issues with large repositories
- Support multiple programming languages (Python, JavaScript, TypeScript, Go)

Functions in this module operate on Neo4j driver instances and handle:
1. Graph creation from analyzed code modules
2. Batch processing for large-scale repository analysis
3. Transaction management for data consistency
4. Git metadata integration (branches, commits)
"""

import logging
from typing import Any

from src.core.exceptions import QueryError

logger = logging.getLogger(__name__)


async def create_graph(
    driver: Any,
    repo_name: str,
    modules_data: list[dict[str, Any]],
    git_metadata: dict[str, Any] | None = None,
) -> None:
    """Create all nodes and relationships in Neo4j.

    Args:
        driver: Neo4j AsyncGraphDatabase driver instance
        repo_name: Name of the repository
        modules_data: List of analyzed module data dictionaries
        git_metadata: Optional Git metadata (branches, commits, etc.)
    """
    async with driver.session() as session:
        # Create Repository node with enhanced metadata
        repo_properties = {
            "name": repo_name,
            "created_at": "datetime()",
        }

        # Add Git metadata if available
        if git_metadata and git_metadata.get("info"):
            info = git_metadata["info"]
            repo_properties.update({
                "remote_url": info.get("remote_url", ""),
                "current_branch": info.get("current_branch", "main"),
                "file_count": info.get("file_count", 0),
                "contributor_count": info.get("contributor_count", 0),
                "size": info.get("size", "unknown"),
            })

        # Create repository with all properties
        await session.run(
            """CREATE (r:Repository {
                name: $name,
                remote_url: $remote_url,
                current_branch: $current_branch,
                file_count: $file_count,
                contributor_count: $contributor_count,
                size: $size,
                created_at: datetime()
            })""",
            name=repo_name,
            remote_url=repo_properties.get("remote_url", ""),
            current_branch=repo_properties.get("current_branch", "main"),
            file_count=repo_properties.get("file_count", 0),
            contributor_count=repo_properties.get("contributor_count", 0),
            size=repo_properties.get("size", "unknown"),
        )

        nodes_created = 0
        relationships_created = 0

        for i, mod in enumerate(modules_data):
            # Determine the language of this module
            language = mod.get("language", "Python")

            # 1. Create File node with language support
            await session.run("""
                CREATE (f:File {
                    name: $name,
                    path: $path,
                    module_name: $module_name,
                    language: $language,
                    line_count: $line_count,
                    created_at: datetime()
                })
            """,
                name=mod["file_path"].split("/")[-1],
                path=mod["file_path"],
                module_name=mod["module_name"],
                language=language,
                line_count=mod.get("line_count", 0),
            )
            nodes_created += 1

            # 2. Connect File to Repository
            await session.run("""
                MATCH (r:Repository {name: $repo_name})
                MATCH (f:File {path: $file_path})
                CREATE (r)-[:CONTAINS]->(f)
            """, repo_name=repo_name, file_path=mod["file_path"])
            relationships_created += 1

            # 3. Create Class nodes and relationships (or Structs for Go)
            for cls in mod.get("classes", []) + mod.get("structs", []):
                # Determine if this is a struct (Go) or class
                is_struct = cls.get("type") == "struct"
                node_label = "Struct" if is_struct else "Class"

                # Create Class/Struct node using MERGE to avoid duplicates
                await session.run(f"""
                    MERGE (c:CodeElement:{node_label} {{full_name: $full_name}})
                    ON CREATE SET c.name = $name,
                                 c.language = $language,
                                 c.exported = $exported,
                                 c.created_at = datetime()
                """,
                    name=cls["name"],
                    full_name=cls.get("full_name", f"{mod['module_name']}.{cls['name']}"),
                    language=language,
                    exported=cls.get("exported", True),
                )
                nodes_created += 1

                # Connect File to Class
                await session.run("""
                    MATCH (f:File {path: $file_path})
                    MATCH (c:Class {full_name: $class_full_name})
                    MERGE (f)-[:DEFINES]->(c)
                """, file_path=mod["file_path"], class_full_name=cls["full_name"])
                relationships_created += 1

                # 4. Create Method nodes - use MERGE to avoid duplicates
                for method in cls["methods"]:
                    method_full_name = f"{cls['full_name']}.{method['name']}"
                    # Create method with unique ID to avoid conflicts
                    method_id = f"{cls['full_name']}::{method['name']}"

                    await session.run("""
                        MERGE (m:Method {method_id: $method_id})
                        ON CREATE SET m.name = $name,
                                     m.full_name = $full_name,
                                     m.args = $args,
                                     m.params_list = $params_list,
                                     m.params_detailed = $params_detailed,
                                     m.return_type = $return_type,
                                     m.created_at = datetime()
                    """,
                        name=method["name"],
                        full_name=method_full_name,
                        method_id=method_id,
                        args=method["args"],
                        params_list=[f"{p['name']}:{p['type']}" for p in method["params"]],  # Simple format
                        params_detailed=method.get("params_detailed", []),  # Detailed format
                        return_type=method["return_type"],
                    )
                    nodes_created += 1

                    # Connect Class to Method
                    await session.run("""
                        MATCH (c:Class {full_name: $class_full_name})
                        MATCH (m:Method {method_id: $method_id})
                        MERGE (c)-[:HAS_METHOD]->(m)
                    """,
                        class_full_name=cls["full_name"],
                        method_id=method_id,
                    )
                    relationships_created += 1

                # 5. Create Enhanced Attribute nodes - FIXED: Now storing all extracted metadata
                for attr in cls["attributes"]:
                    attr_full_name = f"{cls['full_name']}.{attr['name']}"
                    # Create attribute with unique ID to avoid conflicts
                    attr_id = f"{cls['full_name']}::{attr['name']}"

                    # FIXED: Extract all available attribute metadata including framework metadata
                    attr_data = {
                        "name": attr["name"],
                        "full_name": attr_full_name,
                        "attr_id": attr_id,
                        "type": attr.get("type", "Any"),
                        "default_value": attr.get("default_value"),
                        "is_instance": attr.get("is_instance", False),
                        "is_class": attr.get("is_class", False),
                        "is_property": attr.get("is_property", False),
                        "has_type_hint": attr.get("has_type_hint", False),
                        "line_number": attr.get("line_number", 0),
                        "from_slots": attr.get("from_slots", False),
                        "from_dataclass": attr.get("from_dataclass", False),
                        "from_attrs": attr.get("from_attrs", False),
                        "is_class_var": attr.get("is_class_var", False),
                    }

                    await session.run("""
                        MERGE (a:Attribute {attr_id: $attr_id})
                        ON CREATE SET a.name = $name,
                                     a.full_name = $full_name,
                                     a.type = $type,
                                     a.default_value = $default_value,
                                     a.is_instance = $is_instance,
                                     a.is_class = $is_class,
                                     a.is_property = $is_property,
                                     a.has_type_hint = $has_type_hint,
                                     a.line_number = $line_number,
                                     a.from_slots = $from_slots,
                                     a.from_dataclass = $from_dataclass,
                                     a.from_attrs = $from_attrs,
                                     a.is_class_var = $is_class_var,
                                     a.created_at = datetime(),
                                     a.updated_at = datetime()
                    """, **attr_data)
                    nodes_created += 1

                    # Connect Class to Attribute
                    await session.run("""
                        MATCH (c:Class {full_name: $class_full_name})
                        MATCH (a:Attribute {attr_id: $attr_id})
                        MERGE (c)-[:HAS_ATTRIBUTE]->(a)
                    """,
                        class_full_name=cls["full_name"],
                        attr_id=attr_id,
                    )
                    relationships_created += 1

            # 6. Create Function nodes (top-level) - use MERGE to avoid duplicates
            for func in mod.get("functions", []):
                func_id = f"{mod['file_path']}::{func['name']}"
                # Determine function type and properties
                func_type = func.get("type", "function")
                is_async = func.get("async", False)
                is_generator = func.get("generator", False)

                await session.run("""
                    MERGE (f:CodeElement:Function {func_id: $func_id})
                    ON CREATE SET f.name = $name,
                                 f.full_name = $full_name,
                                 f.language = $language,
                                 f.exported = $exported,
                                 f.async = $is_async,
                                 f.generator = $is_generator,
                                 f.type = $func_type,
                                 f.args = $args,
                                 f.params_list = $params_list,
                                 f.params_detailed = $params_detailed,
                                 f.return_type = $return_type,
                                 f.created_at = datetime()
                """,
                    name=func["name"],
                    full_name=func.get("full_name", f"{mod['module_name']}.{func['name']}"),
                    func_id=func_id,
                    language=language,
                    exported=func.get("exported", True),
                    is_async=is_async,
                    is_generator=is_generator,
                    func_type=func_type,
                    args=func.get("args", []),
                    params_list=func.get("params_list", []),  # Simple format for backwards compatibility
                    params_detailed=func.get("params_detailed", []),  # Detailed format
                    return_type=func.get("return_type", "Any"),
                )
                nodes_created += 1

                # Connect File to Function
                await session.run("""
                    MATCH (file:File {path: $file_path})
                    MATCH (func:Function {func_id: $func_id})
                    MERGE (file)-[:DEFINES]->(func)
                """, file_path=mod["file_path"], func_id=func_id)
                relationships_created += 1

            # 7. Create Interface nodes (TypeScript/Go)
            for interface in mod.get("interfaces", []):
                interface_id = f"{mod['file_path']}::{interface['name']}"
                await session.run("""
                    MERGE (i:CodeElement:Interface {interface_id: $interface_id})
                    ON CREATE SET i.name = $name,
                                 i.full_name = $full_name,
                                 i.language = $language,
                                 i.exported = $exported,
                                 i.extends = $extends,
                                 i.created_at = datetime()
                """,
                    interface_id=interface_id,
                    name=interface["name"],
                    full_name=f"{mod['module_name']}.{interface['name']}",
                    language=language,
                    exported=interface.get("exported", True),
                    extends=interface.get("extends", None),
                )
                nodes_created += 1

                # Connect File to Interface
                await session.run("""
                    MATCH (f:File {path: $file_path})
                    MATCH (i:Interface {interface_id: $interface_id})
                    MERGE (f)-[:DEFINES]->(i)
                """, file_path=mod["file_path"], interface_id=interface_id)
                relationships_created += 1

            # 8. Create Type nodes (TypeScript type aliases, Go types)
            for type_def in mod.get("types", []):
                type_id = f"{mod['file_path']}::{type_def['name']}"
                await session.run("""
                    MERGE (t:CodeElement:Type {type_id: $type_id})
                    ON CREATE SET t.name = $name,
                                 t.full_name = $full_name,
                                 t.language = $language,
                                 t.exported = $exported,
                                 t.kind = $kind,
                                 t.base = $base,
                                 t.created_at = datetime()
                """,
                    type_id=type_id,
                    name=type_def["name"],
                    full_name=f"{mod['module_name']}.{type_def['name']}",
                    language=language,
                    exported=type_def.get("exported", True),
                    kind=type_def.get("kind", "alias"),
                    base=type_def.get("base", None),
                )
                nodes_created += 1

                # Connect File to Type
                await session.run("""
                    MATCH (f:File {path: $file_path})
                    MATCH (t:Type {type_id: $type_id})
                    MERGE (f)-[:DEFINES]->(t)
                """, file_path=mod["file_path"], type_id=type_id)
                relationships_created += 1

            # 9. Create Import relationships
            for import_name in mod.get("imports", []):
                # Try to find the target file
                await session.run("""
                    MATCH (source:File {path: $source_path})
                    OPTIONAL MATCH (target:File)
                    WHERE target.module_name = $import_name OR target.module_name STARTS WITH $import_name
                    WITH source, target
                    WHERE target IS NOT NULL
                    MERGE (source)-[:IMPORTS]->(target)
                """, source_path=mod["file_path"], import_name=import_name)
                relationships_created += 1

            if (i + 1) % 10 == 0:
                logger.info("Processed %s/%s files...", i + 1, len(modules_data))

        # Create Branch nodes if metadata available
        if git_metadata and git_metadata.get("branches"):
            logger.info("Creating %s Branch nodes in Neo4j", len(git_metadata['branches'][:10]))
            for branch in git_metadata["branches"][:10]:  # Limit to 10 branches
                await session.run("""
                    CREATE (b:Branch {
                        name: $name,
                        last_commit_date: $last_commit_date,
                        last_commit_message: $last_commit_message
                    })
                """,
                    name=branch["name"],
                    last_commit_date=branch.get("last_commit_date", ""),
                    last_commit_message=branch.get("last_commit_message", ""),
                )
                nodes_created += 1

                # Connect Branch to Repository
                await session.run("""
                    MATCH (r:Repository {name: $repo_name})
                    MATCH (b:Branch {name: $branch_name})
                    CREATE (r)-[:HAS_BRANCH]->(b)
                """, repo_name=repo_name, branch_name=branch["name"])
                relationships_created += 1

        # Create Commit nodes if metadata available
        if git_metadata and git_metadata.get("recent_commits"):
            logger.info("Creating %s Commit nodes in Neo4j", len(git_metadata['recent_commits']))
            for commit in git_metadata["recent_commits"]:
                await session.run("""
                    CREATE (c:Commit {
                        hash: $hash,
                        author_name: $author_name,
                        author_email: $author_email,
                        date: $date,
                        message: $message
                    })
                """,
                    hash=commit["hash"],
                    author_name=commit.get("author_name", ""),
                    author_email=commit.get("author_email", ""),
                    date=commit.get("date", ""),
                    message=commit.get("message", ""),
                )
                nodes_created += 1

                # Connect Commit to Repository
                await session.run("""
                    MATCH (r:Repository {name: $repo_name})
                    MATCH (c:Commit {hash: $commit_hash})
                    CREATE (r)-[:HAS_COMMIT]->(c)
                """, repo_name=repo_name, commit_hash=commit["hash"])
                relationships_created += 1
        else:
            logger.warning("No Git metadata available - Branch and Commit nodes will not be created")

        logger.info("Created %s nodes and %s relationships", nodes_created, relationships_created)


async def process_modules_in_batches(
    driver: Any,
    repo_name: str,
    modules_data: list[dict[str, Any]],
    batch_size: int | None = None,
    batch_timeout_seconds: int = 120,
) -> tuple[int, int]:
    """Process modules in batches to prevent memory issues with large repositories.

    Args:
        driver: Neo4j AsyncGraphDatabase driver instance
        repo_name: Repository name
        modules_data: List of module data dictionaries
        batch_size: Number of modules to process per batch
        batch_timeout_seconds: Timeout for each batch transaction in seconds

    Returns:
        Tuple of (nodes_created, relationships_created)
    """
    if batch_size is None:
        batch_size = 50  # Default batch size

    total_modules = len(modules_data)
    nodes_created = 0
    relationships_created = 0

    logger.info("Processing %s modules in batches of %s", total_modules, batch_size)

    for batch_start in range(0, total_modules, batch_size):
        batch_end = min(batch_start + batch_size, total_modules)
        batch = modules_data[batch_start:batch_end]

        logger.info("Processing batch %s/%s (modules %s-%s/%s)",
                   batch_start//batch_size + 1, (total_modules + batch_size - 1)//batch_size,
                   batch_start + 1, batch_end, total_modules)

        # Process this batch in a transaction
        async with driver.session() as session:
            try:
                # Use explicit transaction with timeout
                tx_config = {
                    "timeout": batch_timeout_seconds,
                }

                async with session.begin_transaction(**tx_config) as tx:
                    batch_nodes, batch_rels = await process_batch_transaction(
                        driver, tx, repo_name, batch, batch_start, total_modules,
                    )
                    await tx.commit()

                nodes_created += batch_nodes
                relationships_created += batch_rels

                logger.info("Batch %s completed: %s nodes, %s relationships",
                           batch_start//batch_size + 1, batch_nodes, batch_rels)

            except QueryError as e:
                logger.error("Neo4j query error in batch %s: %s", batch_start//batch_size + 1, e)
                logger.warning("Attempting to continue with next batch...")
                # Continue with next batch on error
                continue
            except Exception as e:
                logger.exception("Unexpected error processing batch %s: %s", batch_start//batch_size + 1, e)
                logger.warning("Attempting to continue with next batch...")
                # Continue with next batch on error
                continue

    return nodes_created, relationships_created


async def process_batch_transaction(
    driver: Any,
    tx: Any,
    repo_name: str,
    batch: list[dict[str, Any]],
    batch_start: int,
    total_modules: int,
) -> tuple[int, int]:
    """Process a single batch of modules within a transaction.

    Args:
        driver: Neo4j AsyncGraphDatabase driver instance
        tx: Neo4j transaction object
        repo_name: Repository name
        batch: List of modules in this batch
        batch_start: Starting index of this batch
        total_modules: Total number of modules

    Returns:
        Tuple of (nodes_created, relationships_created)
    """
    nodes_created = 0
    relationships_created = 0

    for i, mod in enumerate(batch):
        batch_start + i

        # Process each module within the transaction
        # This is a simplified version - the full implementation would process
        # all the classes, methods, functions, etc. as in the original _create_graph

        # 1. Create File node
        language = mod.get("language", "Python")
        await tx.run("""
            CREATE (f:File {
                name: $name,
                path: $path,
                module_name: $module_name,
                language: $language,
                line_count: $line_count,
                created_at: datetime()
            })
        """,
            name=mod["file_path"].split("/")[-1],
            path=mod["file_path"],
            module_name=mod["module_name"],
            language=language,
            line_count=mod.get("line_count", 0),
        )
        nodes_created += 1

        # 2. Connect File to Repository
        await tx.run("""
            MATCH (r:Repository {name: $repo_name})
            MATCH (f:File {path: $file_path})
            CREATE (r)-[:CONTAINS]->(f)
        """, repo_name=repo_name, file_path=mod["file_path"])
        relationships_created += 1

        # Log progress within batch
        if (i + 1) % 10 == 0:
            logger.debug("  Processed %s/%s modules in current batch", i + 1, len(batch))

    return nodes_created, relationships_created
