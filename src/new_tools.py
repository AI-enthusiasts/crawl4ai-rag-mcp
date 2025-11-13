from src.core import track_request
from src.main import mcp


@mcp.tool()
@track_request("parse_local_repository")
async def parse_local_repository(
    ctx: Context,
    local_path: str,
) -> str:
    """
    Parse a local Git repository into the Neo4j knowledge graph.

    This tool parses a local Git repository directly without cloning, useful for:
    - Analyzing repositories already present on the system
    - Parsing private repositories not accessible via URL
    - Working with repositories that have been modified locally
    - Faster parsing of repositories you already have locally

    The tool analyzes multiple programming languages including:
    - Python (.py files)
    - JavaScript/TypeScript (.js, .ts, .jsx, .tsx files)
    - Go (.go files)
    - And more based on the multi-language analyzer factory

    Args:
        local_path: Absolute path to the local Git repository directory

    Returns:
        JSON string with parsing results, statistics, and repository information
    """
    import json
    import os

    try:
        # Get the app context
        app_ctx = get_app_context()

        if not app_ctx:
            return json.dumps(
                {
                    "success": False,
                    "error": "Application context not available",
                },
                indent=2,
            )

        # Check if repository extractor is available
        repo_extractor = getattr(app_ctx, "repo_extractor", None)
        if not repo_extractor:
            return json.dumps(
                {
                    "success": False,
                    "error": "Repository extractor not available. Neo4j may not be configured or USE_KNOWLEDGE_GRAPH may be false.",
                },
                indent=2,
            )

        # Validate local path
        if not os.path.exists(local_path):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Local path does not exist: {local_path}",
                },
                indent=2,
            )

        if not os.path.isdir(local_path):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Path is not a directory: {local_path}",
                },
                indent=2,
            )

        # Check if it's a Git repository
        git_dir = os.path.join(local_path, ".git")
        if not os.path.exists(git_dir):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Not a Git repository (no .git directory found): {local_path}",
                },
                indent=2,
            )

        # Extract repository name from path
        repo_name = os.path.basename(os.path.abspath(local_path))

        logger.info(f"Parsing local repository: {repo_name} at {local_path}")

        # Use a custom method to analyze local repository
        await repo_extractor.analyze_local_repository(local_path, repo_name)

        # Query Neo4j to get statistics about what was stored
        stats_query = """
        MATCH (r:Repository {name: $repo_name})
        OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
        OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
        OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
        OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
        WITH r,
             COLLECT(DISTINCT f) as files,
             COLLECT(DISTINCT c) as classes,
             COLLECT(DISTINCT m) as methods,
             COLLECT(DISTINCT func) as functions
        RETURN
            SIZE([f IN files WHERE f IS NOT NULL]) as file_count,
            SIZE([c IN classes WHERE c IS NOT NULL]) as class_count,
            SIZE([m IN methods WHERE m IS NOT NULL]) as method_count,
            SIZE([func IN functions WHERE func IS NOT NULL]) as function_count
        """

        async with repo_extractor.driver.session() as session:
            stats_result = await session.run(stats_query, repo_name=repo_name)
            stats = await stats_result.single()

        return json.dumps(
            {
                "success": True,
                "local_path": local_path,
                "repository_name": repo_name,
                "statistics": {
                    "files_processed": stats["file_count"] if stats else 0,
                    "classes_created": stats["class_count"] if stats else 0,
                    "methods_created": stats["method_count"] if stats else 0,
                    "functions_created": stats["function_count"] if stats else 0,
                },
                "message": f"Successfully parsed local repository '{repo_name}' into the knowledge graph",
                "next_steps": [
                    "Use 'query_knowledge_graph' tool with 'explore <repo_name>' to see detailed statistics",
                    "Use 'check_ai_script_hallucinations' tool to validate AI-generated code against this repository",
                ],
            },
            indent=2,
        )

    except Exception as e:
        logger.exception(f"Error parsing local repository {local_path}: {e}")
        return json.dumps(
            {
                "success": False,
                "local_path": local_path,
                "error": f"Local repository parsing failed: {e!s}",
            },
            indent=2,
        )

@mcp.tool()
@track_request("analyze_code_cross_language")
async def analyze_code_cross_language(
    ctx: Context,
    query: str,
    languages: list[str] | None = None,
    match_count: int = 10,
    source_filter: str | None = None,
    include_file_context: bool = True,
) -> str:
    """
    Cross-language code analysis using semantic search across multiple programming languages.

    This tool performs advanced code analysis by searching across multiple programming languages
    simultaneously, enabling developers to:
    - Find similar patterns across different languages (e.g., authentication logic in Python, JS, Go)
    - Compare implementation approaches between languages
    - Discover code reuse opportunities
    - Understand how concepts are implemented across your stack

    Supported languages include Python, JavaScript, TypeScript, Go, and more based on
    the parsed repositories in your knowledge graph.

    Args:
        query: Search query for finding code patterns across languages
        languages: Optional list of languages to search (e.g., ['python', 'javascript', 'go']). If None, searches all languages
        match_count: Maximum number of results to return per language (default: 10)
        source_filter: Optional repository filter (e.g., 'repo-name')
        include_file_context: Whether to include file path and language context (default: True)

    Returns:
        JSON string with cross-language search results, organized by language and confidence scores
    """
    import json

    try:
        # Get the app context
        app_ctx = get_app_context()

        if not app_ctx:
            return json.dumps(
                {
                    "success": False,
                    "error": "Application context not available",
                },
                indent=2,
            )

        # Check database client availability
        database_client = getattr(app_ctx, "database_client", None)
        if not database_client:
            return json.dumps(
                {
                    "success": False,
                    "error": "Database client not available",
                },
                indent=2,
            )

        # Handle languages parameter (from JSON if needed)
        parsed_languages = None
        if languages is not None:
            if isinstance(languages, str):
                if languages.strip().startswith("[") and languages.strip().endswith("]"):
                    try:
                        parsed_languages = json.loads(languages)
                    except json.JSONDecodeError:
                        parsed_languages = [languages]
                else:
                    parsed_languages = [languages]
            else:
                parsed_languages = languages

        logger.info(f"Performing cross-language code analysis for query: {query}")
        if parsed_languages:
            logger.info(f"Filtering by languages: {parsed_languages}")

        # Get all available sources first to understand what repositories we have
        sources_result = await get_available_sources(database_client)
        sources_data = json.loads(sources_result)

        if not sources_data.get("success", False):
            return json.dumps(
                {
                    "success": False,
                    "error": "Could not retrieve available sources for analysis",
                    "details": sources_data,
                },
                indent=2,
            )

        # Perform semantic search
        rag_result = await perform_rag_query(
            database_client,
            query=query,
            source=source_filter,
            match_count=match_count * 3,  # Get more results to filter by language
        )

        rag_data = json.loads(rag_result)

        if not rag_data.get("success", False):
            return json.dumps(
                {
                    "success": False,
                    "error": "Semantic search failed",
                    "details": rag_data,
                },
                indent=2,
            )

        # Organize results by language
        results_by_language = {}

        for result in rag_data.get("results", []):
            # Extract language information from metadata or URL
            language = "unknown"
            metadata = result.get("metadata", {})
            url = result.get("url", "")

            # Try to determine language from metadata
            if "language" in metadata:
                language = metadata["language"].lower()
            elif "file_extension" in metadata:
                ext = metadata["file_extension"].lower()
                language_map = {
                    "py": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "jsx": "javascript",
                    "tsx": "typescript",
                    "go": "go",
                    "java": "java",
                    "cpp": "c++",
                    "c": "c",
                    "rs": "rust",
                    "php": "php",
                    "rb": "ruby",
                    "swift": "swift",
                    "kt": "kotlin",
                    "cs": "csharp",
                }
                language = language_map.get(ext, ext)
            elif url:
                # Try to extract from URL/filename
                for ext, lang in {
                    ".py": "python",
                    ".js": "javascript",
                    ".ts": "typescript",
                    ".jsx": "javascript",
                    ".tsx": "typescript",
                    ".go": "go",
                    ".java": "java",
                    ".cpp": "c++",
                    ".c": "c",
                    ".rs": "rust",
                    ".php": "php",
                    ".rb": "ruby",
                    ".swift": "swift",
                    ".kt": "kotlin",
                    ".cs": "csharp",
                }.items():
                    if ext in url.lower():
                        language = lang
                        break

            # Filter by languages if specified
            if parsed_languages and language not in [lang.lower() for lang in parsed_languages]:
                continue

            # Initialize language group if needed
            if language not in results_by_language:
                results_by_language[language] = []

            # Add file context if requested
            result_item = {
                "content": result.get("content", ""),
                "similarity_score": result.get("similarity_score", 0),
                "source": result.get("source", "unknown"),
            }

            if include_file_context:
                result_item["file_context"] = {
                    "url": url,
                    "metadata": metadata,
                    "language": language,
                }

            results_by_language[language].append(result_item)

        # Limit results per language and sort by similarity
        for language in results_by_language:
            results_by_language[language] = sorted(
                results_by_language[language],
                key=lambda x: x.get("similarity_score", 0),
                reverse=True,
            )[:match_count]

        # Calculate summary statistics
        total_results = sum(len(results) for results in results_by_language.values())
        languages_found = list(results_by_language.keys())

        return json.dumps(
            {
                "success": True,
                "query": query,
                "languages_requested": parsed_languages or "all",
                "languages_found": languages_found,
                "total_results": total_results,
                "results_by_language": results_by_language,
                "analysis_summary": {
                    "most_relevant_language": max(results_by_language.keys(), key=lambda k: len(results_by_language[k])) if results_by_language else None,
                    "coverage": f"{len(languages_found)} languages analyzed",
                    "avg_similarity_per_language": {
                        lang: round(
                            sum(r.get("similarity_score", 0) for r in results) / len(results), 3,
                        ) if results else 0
                        for lang, results in results_by_language.items()
                    },
                },
                "message": f"Found {total_results} code examples across {len(languages_found)} languages",
            },
            indent=2,
        )

    except Exception as e:
        logger.exception(f"Error in cross-language code analysis: {e}")
        return json.dumps(
            {
                "success": False,
                "query": query,
                "error": f"Cross-language analysis failed: {e!s}",
            },
            indent=2,
        )
