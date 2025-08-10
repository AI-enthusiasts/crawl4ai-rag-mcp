# MCP Tools Production-Grade Testing Results - 2025-08-10

**Date**: 2025-08-10
**Time**: 18:21
**Environment**: Production-grade (stdio mode with Docker services)
**Testing Tool**: Claude Code with MCP connection

## Production Configuration

- OPENAI_API_KEY: ✓ Valid production key
- USE_CONTEXTUAL_EMBEDDINGS: true
- USE_HYBRID_SEARCH: true  
- USE_AGENTIC_RAG: true
- USE_RERANKING: true
- USE_KNOWLEDGE_GRAPH: true
- VECTOR_DATABASE: qdrant
- TRANSPORT: stdio

## Service Health Check

Checking Docker services status...

### Test Summary

| Tool | Test Case | Status | Time | Notes |
|------|-----------|--------|------|-------|
| get_available_sources | List sources | ⏳ | - | Testing... |
| scrape_urls | Single URL | ⏳ | - | Pending |
| scrape_urls | Multiple URLs | ⏳ | - | Pending |
| search | Search and scrape | ⏳ | - | Pending |
| smart_crawl_url | Regular website (small) | ⏳ | - | Pending |
| smart_crawl_url | Regular website (large) | ⏳ | - | Pending |
| smart_crawl_url | Sitemap | ⏳ | - | Pending |
| perform_rag_query | Basic query | ⏳ | - | Pending |
| perform_rag_query | Filtered query | ⏳ | - | Pending |
| search_code_examples | Code search | ⏳ | - | Pending |
| parse_github_repository | Basic parsing | ⏳ | - | Pending |
| parse_repository_branch | Branch parsing | ⏳ | - | Pending |
| get_repository_info | Metadata retrieval | ⏳ | - | Pending |
| update_parsed_repository | Repository update | ⏳ | - | Pending |
| extract_and_index_repository_code | Neo4j-Qdrant bridge | ⏳ | - | Pending |
| smart_code_search | Fast mode | ⏳ | - | Pending |
| smart_code_search | Balanced mode | ⏳ | - | Pending |
| smart_code_search | Thorough mode | ⏳ | - | Pending |
| check_ai_script_hallucinations_enhanced | Enhanced detection | ⏳ | - | Pending |
| query_knowledge_graph | Graph queries | ⏳ | - | Pending |
| check_ai_script_hallucinations | Basic detection | ⏳ | - | Pending |

## Detailed Test Results

---
