"""
Neo4j Query Functions

Standalone async functions for querying the Neo4j knowledge graph.
These functions were extracted from KnowledgeGraphValidator to enable
reusability and testability across different validation contexts.
"""

from typing import Any


async def find_modules(driver: Any, module_name: str) -> list[str]:
    """Find repository matching the module name, then return its files"""
    async with driver.session() as session:
        # First, try to find files with module names that match or start with the search term
        module_query = """
        MATCH (r:Repository)-[:CONTAINS]->(f:File)
        WHERE f.module_name = $module_name
           OR f.module_name STARTS WITH $module_name + '.'
           OR split(f.module_name, '.')[0] = $module_name
        RETURN DISTINCT r.name as repo_name, count(f) as file_count
        ORDER BY file_count DESC
        LIMIT 5
        """

        result = await session.run(module_query, module_name=module_name)
        repos_from_modules = []
        async for record in result:
            repos_from_modules.append(record["repo_name"])

        # Also try repository name matching as fallback
        repo_query = """
        MATCH (r:Repository)
        WHERE toLower(r.name) = toLower($module_name)
           OR toLower(replace(r.name, '-', '_')) = toLower($module_name)
           OR toLower(replace(r.name, '_', '-')) = toLower($module_name)
        RETURN r.name as repo_name
        ORDER BY
            CASE
                WHEN toLower(r.name) = toLower($module_name) THEN 1
                WHEN toLower(replace(r.name, '-', '_')) = toLower($module_name) THEN 2
                WHEN toLower(replace(r.name, '_', '-')) = toLower($module_name) THEN 3
            END
        LIMIT 5
        """

        result = await session.run(repo_query, module_name=module_name)
        repos_from_names = []
        async for record in result:
            repos_from_names.append(record["repo_name"])

        # Combine results, prioritizing module-based matches
        all_repos = repos_from_modules + [r for r in repos_from_names if r not in repos_from_modules]

        if not all_repos:
            return []

        # Get files from the best matching repository
        best_repo = all_repos[0]
        files_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
        RETURN f.path, f.module_name
        LIMIT 50
        """

        result = await session.run(files_query, repo_name=best_repo)
        files = []
        async for record in result:
            files.append(record["f.path"])

        return files


async def get_module_contents(driver: Any, module_name: str) -> tuple[list[str], list[str]]:
    """Get classes and functions available in a repository matching the module name"""
    async with driver.session() as session:
        # First, try to find repository by module names in files
        module_query = """
        MATCH (r:Repository)-[:CONTAINS]->(f:File)
        WHERE f.module_name = $module_name
           OR f.module_name STARTS WITH $module_name + '.'
           OR split(f.module_name, '.')[0] = $module_name
        RETURN DISTINCT r.name as repo_name, count(f) as file_count
        ORDER BY file_count DESC
        LIMIT 1
        """

        result = await session.run(module_query, module_name=module_name)
        record = await result.single()

        if record:
            repo_name = record["repo_name"]
        else:
            # Fallback to repository name matching
            repo_query = """
            MATCH (r:Repository)
            WHERE toLower(r.name) = toLower($module_name)
               OR toLower(replace(r.name, '-', '_')) = toLower($module_name)
               OR toLower(replace(r.name, '_', '-')) = toLower($module_name)
            RETURN r.name as repo_name
            ORDER BY
                CASE
                    WHEN toLower(r.name) = toLower($module_name) THEN 1
                    WHEN toLower(replace(r.name, '-', '_')) = toLower($module_name) THEN 2
                    WHEN toLower(replace(r.name, '_', '-')) = toLower($module_name) THEN 3
                END
            LIMIT 1
            """

            result = await session.run(repo_query, module_name=module_name)
            record = await result.single()

            if not record:
                return [], []

            repo_name = record["repo_name"]

        # Get classes from this repository
        class_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN DISTINCT c.name as class_name
        """

        result = await session.run(class_query, repo_name=repo_name)
        classes = []
        async for record in result:
            classes.append(record["class_name"])

        # Get functions from this repository
        func_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
        RETURN DISTINCT func.name as function_name
        """

        result = await session.run(func_query, repo_name=repo_name)
        functions = []
        async for record in result:
            functions.append(record["function_name"])

        return classes, functions


async def find_repository_for_module(
    driver: Any, module_name: str, repo_cache: dict[str, str | None]
) -> str | None:
    """Find the repository name that matches a module name"""
    if module_name in repo_cache:
        return repo_cache[module_name]

    async with driver.session() as session:
        # First, try to find repository by module names in files
        module_query = """
        MATCH (r:Repository)-[:CONTAINS]->(f:File)
        WHERE f.module_name = $module_name
           OR f.module_name STARTS WITH $module_name + '.'
           OR split(f.module_name, '.')[0] = $module_name
        RETURN DISTINCT r.name as repo_name, count(f) as file_count
        ORDER BY file_count DESC
        LIMIT 1
        """

        result = await session.run(module_query, module_name=module_name)
        record = await result.single()

        if record:
            repo_name = record["repo_name"]
        else:
            # Fallback to repository name matching
            query = """
            MATCH (r:Repository)
            WHERE toLower(r.name) = toLower($module_name)
               OR toLower(replace(r.name, '-', '_')) = toLower($module_name)
               OR toLower(replace(r.name, '_', '-')) = toLower($module_name)
               OR toLower(r.name) CONTAINS toLower($module_name)
               OR toLower($module_name) CONTAINS toLower(replace(r.name, '-', '_'))
            RETURN r.name as repo_name
            ORDER BY
                CASE
                    WHEN toLower(r.name) = toLower($module_name) THEN 1
                    WHEN toLower(replace(r.name, '-', '_')) = toLower($module_name) THEN 2
                    ELSE 3
                END
            LIMIT 1
            """

            result = await session.run(query, module_name=module_name)
            record = await result.single()

            repo_name = record["repo_name"] if record else None

        repo_cache[module_name] = repo_name
        return repo_name


async def find_class(driver: Any, class_name: str, repo_cache: dict[str, str | None]) -> dict[str, Any] | None:
    """Find class information in knowledge graph"""
    async with driver.session() as session:
        # First try exact match
        query = """
        MATCH (c:Class)
        WHERE c.name = $class_name OR c.full_name = $class_name
        RETURN c.name as name, c.full_name as full_name
        LIMIT 1
        """

        result = await session.run(query, class_name=class_name)
        record = await result.single()

        if record:
            return {
                "name": record["name"],
                "full_name": record["full_name"],
            }

        # If no exact match and class_name has dots, try repository-based search
        if "." in class_name:
            parts = class_name.split(".")
            module_part = ".".join(parts[:-1])  # e.g., "pydantic_ai"
            class_part = parts[-1]  # e.g., "Agent"

            # Find repository for the module
            repo_name = await find_repository_for_module(driver, module_part, repo_cache)

            if repo_name:
                # Search for class within this repository
                repo_query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
                WHERE c.name = $class_name
                RETURN c.name as name, c.full_name as full_name
                LIMIT 1
                """

                result = await session.run(repo_query, repo_name=repo_name, class_name=class_part)
                record = await result.single()

                if record:
                    return {
                        "name": record["name"],
                        "full_name": record["full_name"],
                    }

        return None


async def find_method(
    driver: Any,
    class_name: str,
    method_name: str,
    method_cache: dict[str, list[dict[str, Any]]],
    repo_cache: dict[str, str | None],
) -> dict[str, Any] | None:
    """Find method information for a class"""
    cache_key = f"{class_name}.{method_name}"
    if cache_key in method_cache:
        methods = method_cache[cache_key]
        return methods[0] if methods else None

    async with driver.session() as session:
        # First try exact match
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed,
               m.return_type as return_type, m.args as args
        LIMIT 1
        """

        result = await session.run(query, class_name=class_name, method_name=method_name)
        record = await result.single()

        if record:
            # Use detailed params if available, fall back to simple params
            params_to_use = record["params_detailed"] or record["params_list"] or []

            method_info = {
                "name": record["name"],
                "params_list": params_to_use,
                "return_type": record["return_type"],
                "args": record["args"] or [],
            }
            method_cache[cache_key] = [method_info]
            return method_info

        # If no exact match and class_name has dots, try repository-based search
        if "." in class_name:
            parts = class_name.split(".")
            module_part = ".".join(parts[:-1])  # e.g., "pydantic_ai"
            class_part = parts[-1]  # e.g., "Agent"

            # Find repository for the module
            repo_name = await find_repository_for_module(driver, module_part, repo_cache)

            if repo_name:
                # Search for method within this repository's classes
                repo_query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE c.name = $class_name AND m.name = $method_name
                RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed,
                       m.return_type as return_type, m.args as args
                LIMIT 1
                """

                result = await session.run(repo_query, repo_name=repo_name, class_name=class_part, method_name=method_name)
                record = await result.single()

                if record:
                    # Use detailed params if available, fall back to simple params
                    params_to_use = record["params_detailed"] or record["params_list"] or []

                    method_info = {
                        "name": record["name"],
                        "params_list": params_to_use,
                        "return_type": record["return_type"],
                        "args": record["args"] or [],
                    }
                    method_cache[cache_key] = [method_info]
                    return method_info

        method_cache[cache_key] = []
        return None


async def find_attribute(
    driver: Any, class_name: str, attr_name: str, repo_cache: dict[str, str | None]
) -> dict[str, Any] | None:
    """Find attribute information for a class"""
    async with driver.session() as session:
        # First try exact match
        query = """
        MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND a.name = $attr_name
        RETURN a.name as name, a.type as type
        LIMIT 1
        """

        result = await session.run(query, class_name=class_name, attr_name=attr_name)
        record = await result.single()

        if record:
            return {
                "name": record["name"],
                "type": record["type"],
            }

        # If no exact match and class_name has dots, try repository-based search
        if "." in class_name:
            parts = class_name.split(".")
            module_part = ".".join(parts[:-1])  # e.g., "pydantic_ai"
            class_part = parts[-1]  # e.g., "Agent"

            # Find repository for the module
            repo_name = await find_repository_for_module(driver, module_part, repo_cache)

            if repo_name:
                # Search for attribute within this repository's classes
                repo_query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
                WHERE c.name = $class_name AND a.name = $attr_name
                RETURN a.name as name, a.type as type
                LIMIT 1
                """

                result = await session.run(repo_query, repo_name=repo_name, class_name=class_part, attr_name=attr_name)
                record = await result.single()

                if record:
                    return {
                        "name": record["name"],
                        "type": record["type"],
                    }

        return None


async def find_function(
    driver: Any, func_name: str, repo_cache: dict[str, str | None]
) -> dict[str, Any] | None:
    """Find function information"""
    async with driver.session() as session:
        # First try exact match
        query = """
        MATCH (f:Function)
        WHERE f.name = $func_name OR f.full_name = $func_name
        RETURN f.name as name, f.params_list as params_list, f.params_detailed as params_detailed,
               f.return_type as return_type, f.args as args
        LIMIT 1
        """

        result = await session.run(query, func_name=func_name)
        record = await result.single()

        if record:
            # Use detailed params if available, fall back to simple params
            params_to_use = record["params_detailed"] or record["params_list"] or []

            return {
                "name": record["name"],
                "params_list": params_to_use,
                "return_type": record["return_type"],
                "args": record["args"] or [],
            }

        # If no exact match and func_name has dots, try repository-based search
        if "." in func_name:
            parts = func_name.split(".")
            module_part = ".".join(parts[:-1])  # e.g., "pydantic_ai"
            func_part = parts[-1]  # e.g., "some_function"

            # Find repository for the module
            repo_name = await find_repository_for_module(driver, module_part, repo_cache)

            if repo_name:
                # Search for function within this repository
                repo_query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
                WHERE func.name = $func_name
                RETURN func.name as name, func.params_list as params_list, func.params_detailed as params_detailed,
                       func.return_type as return_type, func.args as args
                LIMIT 1
                """

                result = await session.run(repo_query, repo_name=repo_name, func_name=func_part)
                record = await result.single()

                if record:
                    # Use detailed params if available, fall back to simple params
                    params_to_use = record["params_detailed"] or record["params_list"] or []

                    return {
                        "name": record["name"],
                        "params_list": params_to_use,
                        "return_type": record["return_type"],
                        "args": record["args"] or [],
                    }

        return None


async def find_pydantic_ai_result_method(driver: Any, method_name: str) -> dict[str, Any] | None:
    """Find method information for pydantic_ai result objects"""
    # Look for methods on pydantic_ai classes that could be result objects
    async with driver.session() as session:
        # Search for common result methods in pydantic_ai repository
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
          AND (c.name CONTAINS 'Result' OR c.name CONTAINS 'Stream' OR c.name CONTAINS 'Run')
        RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed,
               m.return_type as return_type, m.args as args, c.name as class_name
        LIMIT 1
        """

        result = await session.run(query, repo_name="pydantic_ai", method_name=method_name)
        record = await result.single()

        if record:
            # Use detailed params if available, fall back to simple params
            params_to_use = record["params_detailed"] or record["params_list"] or []

            return {
                "name": record["name"],
                "params_list": params_to_use,
                "return_type": record["return_type"],
                "args": record["args"] or [],
                "source_class": record["class_name"],
            }

        return None


async def find_similar_modules(driver: Any, module_name: str) -> list[str]:
    """Find similar repository names for suggestions"""
    async with driver.session() as session:
        query = """
        MATCH (r:Repository)
        WHERE toLower(r.name) CONTAINS toLower($partial_name)
           OR toLower(replace(r.name, '-', '_')) CONTAINS toLower($partial_name)
           OR toLower(replace(r.name, '_', '-')) CONTAINS toLower($partial_name)
        RETURN r.name
        LIMIT 5
        """

        result = await session.run(query, partial_name=module_name[:3])
        suggestions = []
        async for record in result:
            suggestions.append(record["name"])

        return suggestions


async def find_similar_methods(
    driver: Any, class_name: str, method_name: str, repo_cache: dict[str, str | None]
) -> list[str]:
    """Find similar method names for suggestions"""
    async with driver.session() as session:
        # First try exact class match
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name CONTAINS $partial_name
        RETURN m.name as name
        LIMIT 5
        """

        result = await session.run(query, class_name=class_name, partial_name=method_name[:3])
        suggestions = []
        async for record in result:
            suggestions.append(record["name"])

        # If no suggestions and class_name has dots, try repository-based search
        if not suggestions and "." in class_name:
            parts = class_name.split(".")
            module_part = ".".join(parts[:-1])  # e.g., "pydantic_ai"
            class_part = parts[-1]  # e.g., "Agent"

            # Find repository for the module
            repo_name = await find_repository_for_module(driver, module_part, repo_cache)

            if repo_name:
                repo_query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE c.name = $class_name AND m.name CONTAINS $partial_name
                RETURN m.name as name
                LIMIT 5
                """

                result = await session.run(repo_query, repo_name=repo_name, class_name=class_part, partial_name=method_name[:3])
                async for record in result:
                    suggestions.append(record["name"])

        return suggestions
