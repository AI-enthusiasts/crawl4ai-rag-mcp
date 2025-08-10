# 🐳 Crawl4AI+SearXNG MCP Server

<em>Web Crawling, Search and RAG Capabilities for AI Agents and AI Coding Assistants</em>

[![CI/CD Pipeline](https://github.com/krashnicov/crawl4aimcp/workflows/CI%2FCD%20Pipeline%20-%20Test%20%26%20Coverage/badge.svg)](https://github.com/krashnicov/crawl4aimcp/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **(FORKED FROM <https://github.com/coleam00/mcp-crawl4ai-rag>). Added SearXNG integration and batch scrape and processing capabilities.**

A **self-contained Docker solution** that combines the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), [Crawl4AI](https://crawl4ai.com), [SearXNG](https://github.com/searxng/searxng), and [Supabase](https://supabase.com/) to provide AI agents and coding assistants with complete web **search, crawling, and RAG capabilities**.

**🚀 Complete Stack in One Command**: Deploy everything with `make prod` - no Python setup, no dependencies, no external services required.

## 🎯 Smart RAG vs Traditional Scraping

Unlike traditional scraping (such as [Firecrawl](https://github.com/mendableai/firecrawl-mcp-server)) that dumps raw content and overwhelms LLM context windows, this solution uses **intelligent RAG (Retrieval Augmented Generation)** to:

- **🔍 Extract only relevant content** using semantic similarity search
- **⚡ Prevent context overflow** by returning focused, pertinent information
- **🧠 Enhance AI responses** with precisely targeted knowledge
- **📊 Maintain context efficiency** for better LLM performance

**Flexible Output Options:**

- **RAG Mode** (default): Returns semantically relevant chunks with similarity scores
- **Raw Markdown Mode**: Full content extraction when complete context is needed
- **Hybrid Search**: Combines semantic and keyword search for comprehensive results

## 💡 Key Benefits

- **🔧 Zero Configuration**: Pre-configured SearXNG instance included
- **🐳 Docker-Only**: No Python environment setup required
- **🔍 Integrated Search**: Built-in SearXNG for private, fast search
- **⚡ Production Ready**: HTTPS, security, and monitoring included
- **🎯 AI-Optimized**: RAG strategies built for coding assistants

## Overview

This Docker-based MCP server provides a complete web intelligence stack that enables AI agents to:

- **Search the web** using the integrated SearXNG instance
- **Crawl and scrape** websites with advanced content extraction
- **Store content** in vector databases with intelligent chunking
- **Perform RAG queries** with multiple enhancement strategies

**Advanced RAG Strategies Available:**

- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models
- **Knowledge Graph** for AI hallucination detection and repository code analysis

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Features

- **Contextual Embeddings**: Enhanced RAG with LLM-generated context for each chunk, improving search accuracy by 20-30% ([Learn more](docs/CONTEXTUAL_EMBEDDINGS.md))
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

## Tools

The server provides essential web crawling and search tools:

### Core Tools (Always Available)

1. **`scrape_urls`**: Scrape one or more URLs and store their content in the vector database. Supports both single URLs and lists of URLs for batch processing.
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering
5. **NEW!** **`search`**: Comprehensive web search tool that integrates SearXNG search with automated scraping and RAG processing. Performs a complete workflow: (1) searches SearXNG with the provided query, (2) extracts URLs from search results, (3) automatically scrapes all found URLs using existing scraping infrastructure, (4) stores content in vector database, and (5) returns either RAG-processed results organized by URL or raw markdown content. Key parameters: `query` (search terms), `return_raw_markdown` (bypasses RAG for raw content), `num_results` (search result limit), `batch_size` (database operation batching), `max_concurrent` (parallel scraping sessions). Ideal for research workflows, competitive analysis, and content discovery with built-in intelligence.

### Conditional Tools

6. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

### Knowledge Graph Tools (requires `USE_KNOWLEDGE_GRAPH=true`, see below)

**🚀 NEW: Multi-Language Repository Parsing** - The system now supports comprehensive analysis of repositories containing Python, JavaScript, TypeScript, Go, and other languages. See [Multi-Language Parsing Documentation](docs/MULTI_LANGUAGE_PARSING.md) for complete details.

7. **`parse_github_repository`**: Parse a GitHub repository into a Neo4j knowledge graph, extracting classes, methods, functions, and their relationships across **multiple programming languages** (Python, JavaScript, TypeScript, Go, etc.)
8. **`parse_local_repository`**: Parse local Git repositories directly without cloning, supporting **multi-language codebases**
9. **`parse_repository_branch`**: Parse specific branches of repositories for version-specific analysis
10. **`analyze_code_cross_language`**: **NEW!** Perform semantic search across multiple programming languages to find similar patterns (e.g., "authentication logic" across Python, JavaScript, and Go)
11. **`check_ai_script_hallucinations`**: Analyze Python scripts for AI hallucinations by validating imports, method calls, and class usage against the knowledge graph
12. **`query_knowledge_graph`**: Explore and query the Neo4j knowledge graph with commands like `repos`, `classes`, `methods`, and custom Cypher queries
13. **`get_script_analysis_info`**: Get information about script analysis setup, available paths, and usage instructions for hallucination detection tools

## 🔍 Code Search and Validation

**Advanced Neo4j-Qdrant Integration for Reliable AI Code Generation**

The system provides sophisticated code search and validation capabilities by combining:

- **Qdrant**: Semantic vector search for finding relevant code examples
- **Neo4j**: Structural validation against parsed repository knowledge graphs
- **AI Hallucination Detection**: Prevents AI from generating non-existent methods or incorrect usage patterns

### When to Use Neo4j vs Qdrant

| Use Case | Neo4j (Knowledge Graph) | Qdrant (Vector Search) | Combined Approach |
|----------|------------------------|----------------------|-------------------|
| **Exact Structure Validation** | ✅ Perfect - validates class/method existence | ❌ Cannot verify structure | 🏆 Best - structure + semantics |
| **Semantic Code Search** | ❌ Limited - no semantic understanding | ✅ Perfect - finds similar patterns | 🏆 Best - validated similarity |
| **Hallucination Detection** | ✅ Good - catches structural errors | ❌ Cannot detect fake methods | 🏆 Best - comprehensive validation |
| **Code Discovery** | ❌ Requires exact names | ✅ Perfect - fuzzy semantic search | 🏆 Best - discovered + validated |
| **Performance** | ⚡ Fast for exact queries | ⚡ Fast for semantic search | ⚖️ Balanced - parallel validation |

### Enhanced Tools for Code Search and Validation

#### 14. **`smart_code_search`** (requires both `USE_KNOWLEDGE_GRAPH=true` and `USE_AGENTIC_RAG=true`)

Intelligent code search that combines Qdrant semantic search with Neo4j structural validation:

- **Semantic Discovery**: Find code patterns using natural language queries
- **Structural Validation**: Verify all code examples against real repository structure
- **Confidence Scoring**: Get reliability scores for each result (0.0-1.0)
- **Validation Modes**: Choose between "fast", "balanced", or "thorough" validation
- **Intelligent Fallback**: Works even when one system is unavailable

#### 15. **`extract_and_index_repository_code`** (requires both systems)

Bridge Neo4j knowledge graph data into Qdrant for searchable code examples:

- **Knowledge Graph Extraction**: Pull structured code from Neo4j
- **Semantic Indexing**: Generate embeddings and store in Qdrant
- **Rich Metadata**: Preserve class/method relationships and context
- **Batch Processing**: Efficient indexing of large repositories

#### 16. **`check_ai_script_hallucinations_enhanced`** (requires both systems)

Advanced hallucination detection using dual validation:

- **Neo4j Structural Check**: Validate against actual repository structure
- **Qdrant Semantic Check**: Find similar real code examples
- **Combined Confidence**: Merge validation results for higher accuracy
- **Code Suggestions**: Provide corrections from real code examples

### Basic Workflow

1. **Index Repository Structure**:

   ```
   parse_github_repository("https://github.com/pydantic/pydantic-ai.git")
   ```

2. **Extract and Index Code Examples**:

   ```
   extract_and_index_repository_code("pydantic-ai")
   ```

3. **Search with Validation**:

   ```
   smart_code_search(
     query="async function with error handling",
     source_filter="pydantic-ai",
     min_confidence=0.7,
     validation_mode="balanced"
   )
   ```

4. **Validate AI Code**:

   ```
   check_ai_script_hallucinations_enhanced("/path/to/ai_script.py")
   ```

### 📁 Using Hallucination Detection Tools

The hallucination detection tools require access to Python scripts. The Docker container includes volume mounts for convenient script analysis:

**Script Locations:**

- **`./analysis_scripts/user_scripts/`** - Place your Python scripts here (recommended)
- **`./analysis_scripts/test_scripts/`** - For test scripts
- **`./analysis_scripts/validation_results/`** - Results are automatically saved here

**Quick Start:**

1. Create a script: `echo "import pandas as pd" > ./analysis_scripts/user_scripts/test.py`
2. Run validation: Use the `check_ai_script_hallucinations` tool with `script_path="test.py"`
3. Check results: View detailed analysis in `./analysis_scripts/validation_results/`

**Path Translation:** The system automatically translates relative paths to container paths, making it convenient to reference scripts by filename.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Make (optional, for convenience commands)
- 8GB+ available RAM for all services

### 1. Start the Stack

**Production deployment:**

```bash
git clone https://github.com/krashnicov/crawl4aimcp.git
cd crawl4aimcp
make prod  # Starts all services in production mode
```

**Development deployment:**

```bash
make dev   # Starts services with hot reloading and debug logging
```

### 2. Configure Claude Desktop (or other MCP client)

Add the MCP server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "crawl4ai-mcp": {
      "command": "docker",
      "args": [
        "exec", "-i", "crawl4aimcp-mcp-1",
        "uv", "run", "python", "src/main.py"
      ],
      "env": {
        "USE_KNOWLEDGE_GRAPH": "true"
      }
    }
  }
}
```

### 3. Test the Connection

Try these commands in Claude to verify everything works:

```
Use the search tool to find information about "FastAPI authentication"
```

```
Use the scrape_urls tool to scrape https://fastapi.tiangolo.com/tutorial/security/
```

```
Parse this GitHub repository: https://github.com/fastapi/fastapi
```

### 4. Multi-Language Repository Analysis

Test the new multi-language capabilities:

```
Parse a multi-language repository: https://github.com/microsoft/vscode
```

```
Search for authentication patterns across Python, JavaScript, and TypeScript
```

## 🏗️ Architecture

The system consists of several Docker services working together:

### Core Services

- **MCP Server**: FastMCP-based server exposing all tools
- **Crawl4AI**: Advanced web crawling and content extraction
- **SearXNG**: Privacy-focused search engine (no external API keys)
- **Supabase**: PostgreSQL + pgvector for embeddings and RAG
- **Neo4j**: (Optional) Knowledge graph for code structure and hallucination detection
- **Qdrant**: (Optional) Alternative vector database with advanced features

### Data Flow

```
Search Query → SearXNG → URL Extraction → Crawl4AI → Content Processing → Vector Storage → RAG Query → Results
```

```
Repository → Multi-Language Parser → Neo4j Knowledge Graph → Code Validation → Hallucination Detection
```

## Configuration

The system supports extensive configuration through environment variables:

### Core Configuration

```bash
# Basic Configuration
USE_SUPABASE=true                    # Enable Supabase for vector storage
USE_QDRANT=false                     # Use Qdrant instead of Supabase (optional)
USE_KNOWLEDGE_GRAPH=true             # Enable Neo4j for code analysis
USE_AGENTIC_RAG=true                 # Enable advanced RAG features

# Search Configuration  
SEARXNG_URL=http://searxng:8080      # Internal SearXNG URL
CRAWL4AI_URL=http://crawl4ai:8000    # Internal Crawl4AI URL

# Multi-Language Repository Parsing
NEO4J_BATCH_SIZE=50                  # Batch size for large repository processing
NEO4J_BATCH_TIMEOUT=120              # Timeout for batch operations
REPO_MAX_SIZE_MB=500                 # Maximum repository size
REPO_MAX_FILE_COUNT=10000            # Maximum number of files
```

### Advanced RAG Configuration

```bash
# Contextual Embeddings (improves search accuracy by 20-30%)
USE_CONTEXTUAL_EMBEDDINGS=false      # Requires OpenAI API or compatible LLM
LLM_PROVIDER=openai                  # openai, anthropic, groq, etc.
OPENAI_API_KEY=your_key_here         # Required for contextual embeddings

# Hybrid Search (combines vector + keyword search)
USE_HYBRID_SEARCH=false              # Requires PostgreSQL full-text search

# Cross-encoder Reranking (improves result relevance)
USE_RERANKING=false                  # Uses sentence-transformers reranking
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

## Multi-Language Repository Support

The system now provides comprehensive support for multi-language repositories:

### Supported Languages

- **Python** (.py) - Classes, functions, methods, imports, docstrings
- **JavaScript** (.js, .jsx, .mjs, .cjs) - ES6+ features, React components
- **TypeScript** (.ts, .tsx) - Interfaces, types, enums, generics  
- **Go** (.go) - Structs, interfaces, methods, packages

### Key Features

- **Unified Knowledge Graph**: All languages stored in single Neo4j instance
- **Cross-Language Search**: Find similar patterns across different languages
- **Language-Aware Analysis**: Respects language-specific syntax and conventions
- **Repository Size Safety**: Built-in validation prevents resource exhaustion
- **Batch Processing**: Optimized for large multi-language repositories

### Example Multi-Language Workflow

```bash
# Parse a full-stack repository
parse_github_repository("https://github.com/microsoft/vscode")

# Search across languages
analyze_code_cross_language(
  query="authentication middleware",
  languages=["python", "javascript", "typescript", "go"]
)

# Explore repository structure
query_knowledge_graph("explore vscode")
```

For complete documentation, see [Multi-Language Parsing Guide](docs/MULTI_LANGUAGE_PARSING.md).

## Docker Services Detail

### Service URLs (Development)

- **MCP Server**: Internal only (accessed via Docker exec)
- **SearXNG**: <http://localhost:4040>
- **Crawl4AI**: <http://localhost:8000>
- **Supabase Studio**: <http://localhost:54323>
- **Neo4j Browser**: <http://localhost:7474>
- **Qdrant Dashboard**: <http://localhost:6333/dashboard> (if enabled)

### Volume Mounts

```
./analysis_scripts/          → /app/analysis_scripts/
./data/supabase/             → /var/lib/postgresql/data
./data/neo4j/                → /data
./data/qdrant/               → /qdrant/storage
```

## Performance and Scaling

### Resource Requirements

**Minimum (Development):**

- 4GB RAM
- 10GB disk space
- 2 CPU cores

**Recommended (Production):**

- 8GB+ RAM  
- 50GB+ disk space
- 4+ CPU cores

### Optimization Settings

```bash
# Large Repository Processing
export NEO4J_BATCH_SIZE=100
export NEO4J_BATCH_TIMEOUT=300
export REPO_MAX_SIZE_MB=1000

# High-Volume Crawling
export CRAWL4AI_MAX_CONCURRENT=20
export SUPABASE_MAX_CONNECTIONS=20
```

## Troubleshooting

### Common Issues

**1. Services not starting:**

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs mcp
docker-compose logs searxng
```

**2. MCP connection issues:**

```bash
# Test MCP server directly
docker exec -it crawl4aimcp-mcp-1 uv run python src/main.py

# Check Claude Desktop logs
tail -f ~/Library/Logs/Claude/mcp*.log
```

**3. Multi-language parsing issues:**

```bash
# Check Neo4j connection
docker-compose logs neo4j

# Verify language analyzers
docker exec crawl4aimcp-mcp-1 python -c "from src.knowledge_graph.analyzers.factory import AnalyzerFactory; print(AnalyzerFactory().get_supported_languages())"
```

**4. Repository too large:**

```bash
# Increase limits
export REPO_MAX_SIZE_MB=1000
export REPO_MAX_FILE_COUNT=15000
```

### Getting Help

- **Documentation**: Check the `/docs` directory for detailed guides
- **Issues**: Report bugs on GitHub Issues
- **Logs**: All services log to Docker, accessible via `docker-compose logs [service]`

## Development

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request

### Adding Language Support

To add support for new programming languages:

1. Create analyzer in `src/knowledge_graph/analyzers/`
2. Extend `AnalyzerFactory` to recognize file extensions
3. Add language-specific patterns and parsing logic
4. Update documentation and tests

See the [Language Analyzer Development Guide](docs/QA/LANGUAGE_ANALYZER_DEVELOPMENT.md) for details.

### Testing

```bash
# Run unit tests
make test

# Run specific language analyzer tests  
make test-analyzers

# Run integration tests
make test-integration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **Original MCP Crawl4AI RAG**: [coleam00/mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag)
- **Crawl4AI**: [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)  
- **SearXNG**: [searxng/searxng](https://github.com/searxng/searxng)
- **FastMCP**: [jlowin/fastmcp](https://github.com/jlowin/fastmcp)
