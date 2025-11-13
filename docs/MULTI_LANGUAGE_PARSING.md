# Multi-Language Repository Parsing

The Crawl4AI MCP server now supports comprehensive multi-language repository parsing, enabling developers to analyze codebases written in multiple programming languages simultaneously. This feature provides intelligent code structure extraction, cross-language code search, and enhanced AI hallucination detection across diverse technology stacks.

## Overview

The multi-language parsing system extends beyond Python to support modern polyglot development environments. It can analyze repositories containing code in Python, JavaScript, TypeScript, Go, and other languages, creating a unified knowledge graph that enables powerful cross-language analysis and validation.

### Key Benefits

- **Polyglot Analysis**: Parse repositories containing multiple programming languages
- **Unified Knowledge Graph**: Store code structure from all languages in a single Neo4j graph
- **Cross-Language Code Search**: Find similar patterns across different languages
- **Enhanced AI Validation**: Detect hallucinations in multi-language AI-generated code
- **Repository Size Safety**: Built-in validation to prevent resource exhaustion
- **Performance Optimization**: Batched processing for large repositories

## Supported Languages

### Currently Supported

| Language | File Extensions | Features |
|----------|----------------|----------|
| **Python** | `.py` | Classes, functions, methods, imports, docstrings |
| **JavaScript** | `.js`, `.jsx`, `.mjs`, `.cjs` | Classes, functions, ES6+ features, imports/exports |
| **TypeScript** | `.ts`, `.tsx` | Interfaces, types, enums, generics |
| **Go** | `.go` | Structs, interfaces, functions, methods, packages |

### Language-Specific Features

#### Python

- Class definitions with inheritance
- Function and method signatures
- Import statements and dependencies
- Docstring extraction
- Decorator support

#### JavaScript/TypeScript

- ES6 classes and methods
- Arrow functions and generators
- Import/export statements (ES6 and CommonJS)
- TypeScript interfaces and type definitions
- React component detection
- JSDoc comment extraction

#### Go

- Struct definitions and fields
- Interface specifications
- Methods with receivers
- Package management
- Exported symbol detection

## Tools and Commands

### Repository Parsing Tools

#### `parse_github_repository`

Parse a remote GitHub repository with multi-language support.

```json
{
  "tool": "parse_github_repository",
  "arguments": {
    "repo_url": "https://github.com/username/multi-lang-project"
  }
}
```

**Features:**

- Automatic language detection based on file extensions
- Repository size validation (default 500MB limit)
- Batch processing for large repositories
- Git metadata extraction (branches, tags, commits)

#### `parse_local_repository`

Parse a local Git repository directly from the filesystem.

```json
{
  "tool": "parse_local_repository", 
  "arguments": {
    "local_path": "/home/user/projects/my-repo"
  }
}
```

**Security Features:**

- Path validation restricted to safe directories
- Git repository verification
- Sandboxed execution

#### `parse_repository_branch`

Parse a specific branch of a repository for version-specific analysis.

```json
{
  "tool": "parse_repository_branch",
  "arguments": {
    "repo_url": "https://github.com/username/project",
    "branch": "feature/new-api"
  }
}
```

### Analysis and Search Tools

#### `analyze_code_cross_language`

Perform semantic search across multiple programming languages.

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "authentication middleware",
    "languages": ["python", "javascript", "go"],
    "match_count": 10
  }
}
```

**Use Cases:**

- Find similar patterns across languages
- Compare implementation approaches
- Discover code reuse opportunities
- Understand architectural patterns

#### `query_knowledge_graph`

Explore the multi-language knowledge graph with Cypher queries.

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "classes python-api"
  }
}
```

**Available Commands:**

- `repos` - List all parsed repositories
- `classes <repo_name>` - List classes in a repository
- `method <method_name>` - Search for methods across languages
- `query <cypher>` - Execute custom Cypher queries

## Configuration Options

### Repository Size Limits

Control resource usage with configurable size limits:

```bash
# Maximum repository size in MB (default: 500)
export REPO_MAX_SIZE_MB=1000

# Maximum number of files (default: 10000)
export REPO_MAX_FILE_COUNT=15000

# Minimum free disk space in GB (default: 1.0)
export REPO_MIN_FREE_SPACE_GB=2.0
```

### Neo4j Batch Processing

Optimize performance for large repositories:

```bash
# Batch size for Neo4j operations (default: 50)
export NEO4J_BATCH_SIZE=100

# Batch timeout in seconds (default: 120)
export NEO4J_BATCH_TIMEOUT=180
```

### Language Detection

The system automatically detects languages based on file extensions:

```python
# Language mapping (internal configuration)
LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript", 
    ".ts": "TypeScript",
    ".jsx": "JavaScript",
    ".tsx": "TypeScript", 
    ".go": "Go"
}
```

## Usage Examples

### Example 1: Analyzing a Full-Stack Repository

Parse a repository containing frontend JavaScript, backend Python, and microservices in Go:

```json
{
  "tool": "parse_github_repository",
  "arguments": {
    "repo_url": "https://github.com/company/full-stack-app"
  }
}
```

Expected output structure:

```json
{
  "success": true,
  "repository_name": "full-stack-app",
  "languages_detected": ["Python", "JavaScript", "TypeScript", "Go"],
  "statistics": {
    "total_files": 247,
    "python_files": 89,
    "javascript_files": 134,
    "go_files": 24,
    "classes_created": 45,
    "methods_created": 312,
    "functions_created": 128
  },
  "processing_summary": {
    "batch_count": 5,
    "processing_time_seconds": 42,
    "memory_usage_mb": 156
  }
}
```

### Example 2: Cross-Language Code Search

Find authentication patterns across your entire stack:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "JWT token validation middleware",
    "languages": ["python", "javascript", "go"],
    "match_count": 5,
    "include_file_context": true
  }
}
```

Expected response:

```json
{
  "success": true,
  "query": "JWT token validation middleware",
  "results_by_language": {
    "python": [
      {
        "content": "def verify_jwt_token(token):\n    try:\n        payload = jwt.decode(token, SECRET_KEY)\n        return payload\n    except jwt.ExpiredSignatureError:\n        raise AuthenticationError('Token expired')",
        "similarity_score": 0.89,
        "source": "auth-service",
        "file_context": {
          "url": "neo4j://repository/auth-service/function/verify_jwt_token",
          "metadata": {"language": "Python", "file_path": "auth/validators.py"},
          "language": "python"
        }
      }
    ],
    "javascript": [
      {
        "content": "const validateJWT = (token) => {\n  try {\n    const decoded = jwt.verify(token, process.env.JWT_SECRET);\n    return decoded;\n  } catch (error) {\n    throw new Error('Invalid token');\n  }\n}",
        "similarity_score": 0.85,
        "source": "frontend-app",
        "file_context": {
          "url": "neo4j://repository/frontend-app/function/validateJWT",
          "metadata": {"language": "JavaScript", "file_path": "middleware/auth.js"},
          "language": "javascript"
        }
      }
    ],
    "go": [
      {
        "content": "func ValidateJWT(tokenString string) (*Claims, error) {\n    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {\n        return []byte(secret), nil\n    })\n    if err != nil {\n        return nil, err\n    }\n    return token.Claims.(*Claims), nil\n}",
        "similarity_score": 0.82,
        "source": "api-gateway",
        "file_context": {
          "url": "neo4j://repository/api-gateway/function/ValidateJWT", 
          "metadata": {"language": "Go", "file_path": "auth/middleware.go"},
          "language": "go"
        }
      }
    ]
  },
  "summary": {
    "total_results": 3,
    "languages_found": ["python", "javascript", "go"],
    "average_similarity": 0.853
  }
}
```

### Example 3: Repository Exploration

Explore the structure of a parsed multi-language repository:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "explore my-project"
  }
}
```

Response includes:

- File count by language
- Class and function distribution
- Import/dependency analysis
- Code complexity metrics

## Performance Considerations

### Repository Size Management

Large repositories are automatically validated before processing:

1. **Size Check**: Repository size estimated before cloning
2. **File Count**: Prevents processing repositories with excessive files
3. **Disk Space**: Ensures sufficient free space (2x repository size)
4. **Memory Usage**: Batch processing prevents memory exhaustion

### Processing Optimization

Multi-language parsing is optimized for performance:

```bash
# Recommended settings for large repositories
export NEO4J_BATCH_SIZE=100
export NEO4J_BATCH_TIMEOUT=300
export REPO_MAX_SIZE_MB=1000
```

### Concurrent Processing

The system uses concurrent processing where possible:

- File analysis runs in parallel
- Database operations are batched
- Network requests are throttled

## Troubleshooting Guide

### Common Issues

#### 1. Repository Too Large

**Error**: `Repository too large: 750.2MB exceeds limit of 500MB`

**Solution**:

```bash
export REPO_MAX_SIZE_MB=1000
# Restart the MCP server
```

#### 2. Insufficient Disk Space

**Error**: `Insufficient disk space: 0.8GB available, 2.0GB required`

**Solution**: Free up disk space or increase available storage

#### 3. Language Not Detected

**Issue**: Files not being analyzed

**Solution**: Check if file extensions are supported:

- Verify file extensions in repository
- Check analyzer factory configuration
- Submit feature request for new languages

#### 4. Neo4j Connection Issues

**Error**: `Repository extractor not available`

**Solution**:

```bash
# Ensure Neo4j is enabled
export USE_KNOWLEDGE_GRAPH=true
# Verify Neo4j connection settings
export NEO4J_URI=bolt://localhost:7687
```

#### 5. Batch Processing Timeouts

**Error**: `Batch processing timeout after 120 seconds`

**Solution**:

```bash
# Increase timeout for large repositories
export NEO4J_BATCH_TIMEOUT=300
```

### Performance Tuning

For optimal performance with multi-language repositories:

1. **Batch Size**: Increase for faster processing of large repos
2. **Concurrent Sessions**: Adjust based on system resources
3. **Memory Limits**: Monitor and adjust Docker memory limits
4. **Disk I/O**: Use SSD storage for better performance

### Debugging Tools

Use these tools to debug parsing issues:

```bash
# Check repository statistics
curl -X POST http://localhost:3000/tools/query_knowledge_graph \
  -d '{"command": "repos"}'

# Analyze specific repository
curl -X POST http://localhost:3000/tools/get_repository_info \
  -d '{"repo_name": "my-project"}'

# Test cross-language search
curl -X POST http://localhost:3000/tools/analyze_code_cross_language \
  -d '{"query": "test query", "languages": ["python", "javascript"]}'
```

## Advanced Features

### Custom Cypher Queries

Execute advanced queries across the multi-language knowledge graph:

```cypher
// Find all methods that call external APIs across languages
MATCH (m:Method)-[:CALLS]->(api:ExternalAPI)
RETURN m.name, m.language, api.endpoint
ORDER BY m.language, m.name
```

### Code Pattern Analysis

Identify common patterns across languages:

```cypher
// Find similar class structures across languages
MATCH (c1:Class), (c2:Class)
WHERE c1.language <> c2.language
  AND c1.name = c2.name
RETURN c1.name, c1.language, c2.language
```

### Dependency Mapping

Analyze dependencies across the entire stack:

```cypher
// Map dependencies across languages
MATCH (r:Repository)-[:CONTAINS]->(f:File)-[:IMPORTS]->(dep:Dependency)
RETURN r.name, f.language, dep.name
ORDER BY r.name, f.language
```

## Future Language Support

The system is designed for extensibility. Future language support includes:

- **Java** - Classes, interfaces, annotations
- **C#** - Classes, properties, LINQ
- **Rust** - Structs, traits, macros
- **C++** - Classes, templates, namespaces
- **PHP** - Classes, traits, namespaces
- **Ruby** - Classes, modules, gems
- **Swift** - Classes, protocols, extensions
- **Kotlin** - Classes, data classes, coroutines

To request support for additional languages, please submit an issue with:

- Language name and common file extensions
- Key language constructs to analyze
- Sample repository for testing
- Use case description

## Best Practices

### Repository Selection

Choose repositories that benefit from multi-language analysis:

- Full-stack applications
- Microservice architectures  
- Monorepos with multiple languages
- API implementations across languages

### Query Optimization

Structure cross-language queries effectively:

- Use specific language filters when possible
- Limit match count for faster responses
- Include file context for better understanding
- Use semantic queries rather than exact matches

### Integration Workflow

Integrate multi-language parsing into your development workflow:

1. **Parse repositories** after major releases
2. **Update local repos** when switching branches
3. **Cross-language search** during code reviews
4. **Validate AI code** before committing
5. **Explore patterns** during architecture decisions

## Conclusion

Multi-language repository parsing enables comprehensive analysis of modern polyglot codebases. By combining structural analysis with semantic search, developers can better understand, validate, and improve their multi-language applications.

The system scales from small projects to enterprise repositories while maintaining performance and accuracy through intelligent batching, size validation, and concurrent processing.
