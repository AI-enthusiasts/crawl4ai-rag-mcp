# MCP Tools Production-Grade Testing Results - 2025-08-10 18:20:16 BST

**Date**: 2025-08-10
**Time**: 18:20:16 BST
**Environment**: Production-grade MCP server connection
**Testing Tool**: Claude Code with MCP connection
**Tester**: QA Agent

## Production Configuration

- OPENAI_API_KEY: ✓ Production key configured
- USE_CONTEXTUAL_EMBEDDINGS: true
- USE_HYBRID_SEARCH: true  
- USE_AGENTIC_RAG: true
- USE_RERANKING: true
- USE_KNOWLEDGE_GRAPH: true (Neo4j)
- VECTOR_DATABASE: qdrant

## Environment Status

**Date/Time**: Sun Aug 10 18:20:16 BST 2025
**Docker Services**:

```
NAMES     STATUS                 PORTS
valkey    Up 5 hours (healthy)   0.0.0.0:6379->6379/tcp, [::]:6379->6379/tcp
```

## Test Execution Log

Starting systematic execution of MCP Tools Testing Plan...

### Test 1.1: get_available_sources

**DateTime**: Sun Aug 10 18:20:27 BST 2025
**Input(s)**: (none)
**Steps Taken**: Executing mcp__crawl4ai-docker__get_available_sources
