"""Smart crawling service with intelligent URL type detection."""

import json
import logging

import httpx
from fastmcp import Context

from src.core import MCPToolError
from src.core.context import get_app_context
from src.core.exceptions import CrawlError, DatabaseError, FetchError

# Import will be done in the function to avoid circular imports
from src.utils import is_sitemap, is_txt, normalize_url, parse_sitemap_content

from .crawling import (
    crawl_markdown_file,
    crawl_recursive_internal_links,
    process_urls_for_mcp,
)

logger = logging.getLogger(__name__)


async def _perform_rag_query_with_context(
    ctx: Context,
    query: str,
    source: str | None = None,
    match_count: int = 5,
) -> str:
    """
    Helper function to properly extract database_client from context and call perform_rag_query.
    """
    import json

    # Get the app context that was stored during lifespan
    app_ctx = get_app_context()

    if (
        not app_ctx
        or not hasattr(app_ctx, "database_client")
        or not app_ctx.database_client
    ):
        return json.dumps(
            {
                "success": False,
                "error": "Database client not available",
            },
            indent=2,
        )

    from src.database.rag_queries import perform_rag_query

    return await perform_rag_query(
        app_ctx.database_client,
        query=query,
        source=source,
        match_count=match_count,
    )


async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    chunk_size: int = 5000,
    return_raw_markdown: bool = False,
    query: list[str] | None = None,
) -> str:
    """
    Intelligently crawl a URL based on its type.

    Detects URL type and applies appropriate crawling strategy:
    - Sitemaps: Extract and crawl all URLs
    - Text files: Direct retrieval
    - Regular pages: Recursive crawling

    Args:
        ctx: FastMCP context
        url: URL to crawl
        max_depth: Max recursion depth for regular URLs
        chunk_size: Chunk size for content
        return_raw_markdown: Return raw markdown
        query: Optional RAG queries to run

    Returns:
        JSON string with results
    """
    try:
        normalized_url = normalize_url(url)

        # Detect URL type and crawl accordingly
        if is_sitemap(normalized_url):
            logger.info(f"Detected sitemap: {normalized_url}")
            return await _crawl_sitemap(
                ctx,
                normalized_url,
                chunk_size,
                return_raw_markdown,
                query,
            )
        if is_txt(normalized_url):
            logger.info(f"Detected text file: {normalized_url}")
            return await _crawl_text_file(
                ctx,
                normalized_url,
                chunk_size,
                return_raw_markdown,
            )
        logger.info(f"Regular URL, crawling recursively: {normalized_url}")
        return await _crawl_recursive(
            ctx,
            normalized_url,
            max_depth,
            chunk_size,
            return_raw_markdown,
            query,
        )

    except (CrawlError, FetchError, DatabaseError) as e:
        logger.error(f"Smart crawl operation failed: {e}")
        msg = f"Smart crawl failed: {e!s}"
        raise MCPToolError(msg)
    except Exception as e:
        logger.exception(f"Unexpected error in smart_crawl_url: {e}")
        msg = f"Smart crawl failed: {e!s}"
        raise MCPToolError(msg)


async def _crawl_sitemap(
    ctx: Context,
    url: str,
    chunk_size: int,
    return_raw_markdown: bool,
    query: list[str] | None,
) -> str:
    """Crawl a sitemap URL."""
    try:
        # Fetch and parse sitemap
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                msg = f"Failed to fetch sitemap: HTTP {response.status_code}"
                raise MCPToolError(
                    msg,
                )
            content = response.text

        # Parse sitemap URLs
        urls = parse_sitemap_content(content)
        if not urls:
            return json.dumps(
                {
                    "success": False,
                    "type": "sitemap",
                    "message": "No URLs found in sitemap",
                    "url": url,
                },
            )

        logger.info(f"Found {len(urls)} URLs in sitemap")

        # Crawl all URLs
        result = await process_urls_for_mcp(
            ctx=ctx,
            urls=urls,
            batch_size=20,
            return_raw_markdown=return_raw_markdown,
        )

        # Parse result and add metadata
        data = json.loads(result)
        data["type"] = "sitemap"
        data["sitemap_url"] = url
        data["total_urls"] = len(urls)

        # Run RAG queries if requested
        if query and not return_raw_markdown and data.get("success"):
            data["query_results"] = {}
            for q in query:
                try:
                    rag_result = await _perform_rag_query_with_context(
                        ctx,
                        q,
                        source=None,
                        match_count=5,
                    )
                    data["query_results"][q] = json.loads(rag_result)
                except DatabaseError as e:
                    logger.error(f"Database error during RAG query '{q}': {e}")
                    data["query_results"][q] = {"error": str(e)}
                except Exception as e:
                    logger.exception(f"RAG query failed for '{q}': {e}")
                    data["query_results"][q] = {"error": str(e)}

        return json.dumps(data)

    except FetchError as e:
        logger.error(f"Failed to fetch sitemap: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "sitemap",
                "error": str(e),
                "url": url,
            },
        )
    except CrawlError as e:
        logger.error(f"Sitemap crawl error: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "sitemap",
                "error": str(e),
                "url": url,
            },
        )
    except Exception as e:
        logger.exception(f"Unexpected error in sitemap crawl: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "sitemap",
                "error": str(e),
                "url": url,
            },
        )


async def _crawl_text_file(
    ctx: Context,
    url: str,
    chunk_size: int,
    return_raw_markdown: bool,
) -> str:
    """Crawl a text file directly."""
    try:
        # Get the app context to access the browser config
        from src.core.context import get_app_context

        app_ctx = get_app_context()

        if not app_ctx or not hasattr(app_ctx, "browser_config"):
            return json.dumps(
                {
                    "success": False,
                    "type": "text_file",
                    "error": "Browser config not available in application context",
                    "url": url,
                },
            )

        # Call low-level crawl_markdown_file with correct parameters
        crawl_results = await crawl_markdown_file(
            browser_config=app_ctx.browser_config,
            url=url,
        )

        if not crawl_results:
            return json.dumps(
                {
                    "success": False,
                    "type": "text_file",
                    "error": "Failed to crawl file",
                    "url": url,
                },
            )

        if return_raw_markdown:
            # Return raw markdown content directly
            return json.dumps(
                {
                    "success": True,
                    "type": "text_file",
                    "url": url,
                    "markdown": crawl_results[0]["markdown"],
                },
            )

        # Store in database
        if not app_ctx.database_client:
            return json.dumps(
                {
                    "success": False,
                    "type": "text_file",
                    "error": "Database client not available",
                    "url": url,
                },
            )

        from src.utils import add_documents_to_database
        from src.utils.text_processing import smart_chunk_markdown
        from src.utils.url_helpers import extract_domain_from_url

        result = crawl_results[0]
        chunks = smart_chunk_markdown(result["markdown"], chunk_size=chunk_size)

        if not chunks:
            return json.dumps(
                {
                    "success": False,
                    "type": "text_file",
                    "error": "No content to store after chunking",
                    "url": url,
                },
            )

        source_id = extract_domain_from_url(result["url"])
        urls = [result["url"]] * len(chunks)
        chunk_numbers = list(range(len(chunks)))
        contents = chunks
        metadatas = [{"url": result["url"], "chunk": i} for i in range(len(chunks))]
        url_to_full_document = {result["url"]: result["markdown"]}
        source_ids = [source_id] * len(chunks) if source_id else None

        await add_documents_to_database(
            database=app_ctx.database_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=20,
            source_ids=source_ids,
        )

        return json.dumps(
            {
                "success": True,
                "type": "text_file",
                "url": url,
                "chunks_stored": len(chunks),
                "source_id": source_id,
            },
        )

    except CrawlError as e:
        logger.error(f"Text file crawl error: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "text_file",
                "error": str(e),
                "url": url,
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error storing text file: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "text_file",
                "error": str(e),
                "url": url,
            },
        )
    except Exception as e:
        logger.exception(f"Unexpected error in text file crawl: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "text_file",
                "error": str(e),
                "url": url,
            },
        )


async def _crawl_recursive(
    ctx: Context,
    url: str,
    max_depth: int,
    chunk_size: int,
    return_raw_markdown: bool,
    query: list[str] | None,
) -> str:
    """Crawl a regular URL recursively."""
    try:
        # Get the app context to access the browser config
        from src.core.context import get_app_context

        app_ctx = get_app_context()

        if not app_ctx or not hasattr(app_ctx, "browser_config"):
            msg = "Browser config not available in application context"
            raise MCPToolError(msg)

        # Call crawl_recursive_internal_links with correct parameters
        crawl_results = await crawl_recursive_internal_links(
            browser_config=app_ctx.browser_config,
            start_urls=[url],  # Note: expects a list
            dispatcher=app_ctx.dispatcher,
            max_depth=max_depth,
        )

        # Process results - it returns a list of dicts
        if return_raw_markdown:
            # Return raw markdown from all crawled pages
            markdown_content = "\n\n---\n\n".join(
                [
                    f"# {result.get('url', 'Unknown URL')}\n\n{result.get('markdown', '')}"
                    for result in crawl_results
                ],
            )
            return json.dumps(
                {
                    "success": True,
                    "type": "recursive",
                    "raw_markdown": markdown_content,
                    "urls_crawled": len(crawl_results),
                },
            )

        # Store results in database if not returning raw
        app_ctx = get_app_context()
        if not app_ctx or not app_ctx.database_client:
            msg = "Database client not available in application context"
            raise MCPToolError(msg)
        db_client = app_ctx.database_client

        stored_count = 0
        for result in crawl_results:
            if result.get("success") and result.get("markdown"):
                try:
                    await db_client.store_crawled_page(
                        url=result["url"],
                        content=result["markdown"],
                        chunk_size=chunk_size,
                    )
                    stored_count += 1
                except DatabaseError as e:
                    logger.error(f"Database error storing {result['url']}: {e}")
                except Exception as e:
                    logger.exception(f"Failed to store {result['url']}: {e}")

        # Create response data
        data = {
            "success": True,
            "type": "recursive",
            "urls_crawled": len(crawl_results),
            "urls_stored": stored_count,
            "max_depth": max_depth,
        }

        # Run RAG queries if requested
        if query and not return_raw_markdown and data.get("success"):
            data["query_results"] = {}
            for q in query:
                try:
                    rag_result = await _perform_rag_query_with_context(
                        ctx,
                        q,
                        source=None,
                        match_count=5,
                    )
                    data["query_results"][q] = json.loads(rag_result)
                except DatabaseError as e:
                    logger.error(f"Database error during RAG query '{q}': {e}")
                    data["query_results"][q] = {"error": str(e)}
                except Exception as e:
                    logger.exception(f"RAG query failed for '{q}': {e}")
                    data["query_results"][q] = {"error": str(e)}

        return json.dumps(data)

    except CrawlError as e:
        logger.error(f"Recursive crawl error: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "recursive",
                "error": str(e),
                "url": url,
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error in recursive crawl: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "recursive",
                "error": str(e),
                "url": url,
            },
        )
    except Exception as e:
        logger.exception(f"Unexpected error in recursive crawl: {e}")
        return json.dumps(
            {
                "success": False,
                "type": "recursive",
                "error": str(e),
                "url": url,
            },
        )
