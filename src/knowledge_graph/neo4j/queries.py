"""Neo4j query operations for repository graph"""

from typing import Any


async def search_graph(driver: Any, query_type: str, **kwargs: Any) -> list[dict[str, Any]] | None:
    """Search the Neo4j graph directly

    Args:
        driver: Neo4j driver instance
        query_type: Type of query to execute
        **kwargs: Query parameters

    Returns:
        Query results as list of dictionaries
    """
    async with driver.session() as session:
        if query_type == "files_importing":
            target = kwargs.get("target")
            result = await session.run("""
                MATCH (source:File)-[:IMPORTS]->(target:File)
                WHERE target.module_name CONTAINS $target
                RETURN source.path as file, target.module_name as imports
            """, target=target)
            return [{"file": record["file"], "imports": record["imports"]} async for record in result]

        if query_type == "classes_in_file":
            file_path = kwargs.get("file_path")
            result = await session.run("""
                MATCH (f:File {path: $file_path})-[:DEFINES]->(c:Class)
                RETURN c.name as class_name, c.full_name as full_name
            """, file_path=file_path)
            return [{"class_name": record["class_name"], "full_name": record["full_name"]} async for record in result]

        if query_type == "methods_of_class":
            class_name = kwargs.get("class_name")
            result = await session.run("""
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE c.name CONTAINS $class_name OR c.full_name CONTAINS $class_name
                RETURN m.name as method_name, m.args as args
            """, class_name=class_name)
            return [{"method_name": record["method_name"], "args": record["args"]} async for record in result]
    return None
