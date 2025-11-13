# Cross-Language Code Search Examples

This example demonstrates how to search for similar code patterns across multiple programming languages using the `analyze_code_cross_language` tool.

## Overview

Cross-language code search enables you to find similar implementation patterns across different programming languages in your parsed repositories. This is particularly useful for:

- **Architecture Consistency**: Ensure similar patterns across microservices
- **Code Reuse Discovery**: Find existing implementations in different languages
- **Learning and Training**: Understand how concepts are implemented across languages
- **Migration Planning**: Find equivalent patterns when migrating between languages

## Basic Cross-Language Search

### Example 1: Authentication Patterns

Search for authentication logic across Python, JavaScript, and Go:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "JWT token authentication",
    "languages": ["python", "javascript", "go"],
    "match_count": 5,
    "include_file_context": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "JWT token authentication",
  "results_by_language": {
    "python": [
      {
        "content": "def verify_jwt_token(token: str) -> dict:\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n        return payload\n    except jwt.ExpiredSignatureError:\n        raise HTTPException(status_code=401, detail='Token expired')\n    except jwt.InvalidTokenError:\n        raise HTTPException(status_code=401, detail='Invalid token')",
        "similarity_score": 0.92,
        "source": "api-backend",
        "file_context": {
          "url": "neo4j://repository/api-backend/function/verify_jwt_token",
          "metadata": {
            "language": "Python",
            "file_path": "auth/jwt_utils.py",
            "function_name": "verify_jwt_token"
          }
        }
      }
    ],
    "javascript": [
      {
        "content": "const verifyJWT = (token) => {\n  try {\n    const decoded = jwt.verify(token, process.env.JWT_SECRET);\n    return decoded;\n  } catch (error) {\n    if (error.name === 'TokenExpiredError') {\n      throw new Error('Token expired');\n    }\n    throw new Error('Invalid token');\n  }\n};",
        "similarity_score": 0.89,
        "source": "frontend-api",
        "file_context": {
          "url": "neo4j://repository/frontend-api/function/verifyJWT", 
          "metadata": {
            "language": "JavaScript",
            "file_path": "middleware/auth.js",
            "function_name": "verifyJWT"
          }
        }
      }
    ],
    "go": [
      {
        "content": "func VerifyJWTToken(tokenString string) (*Claims, error) {\n    claims := &Claims{}\n    token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {\n        return []byte(jwtSecret), nil\n    })\n    \n    if err != nil {\n        if ve, ok := err.(*jwt.ValidationError); ok {\n            if ve.Errors&jwt.ValidationErrorExpired != 0 {\n                return nil, errors.New(\"token expired\")\n            }\n        }\n        return nil, errors.New(\"invalid token\")\n    }\n    \n    if !token.Valid {\n        return nil, errors.New(\"invalid token\")\n    }\n    \n    return claims, nil\n}",
        "similarity_score": 0.87,
        "source": "auth-service",
        "file_context": {
          "url": "neo4j://repository/auth-service/function/VerifyJWTToken",
          "metadata": {
            "language": "Go", 
            "file_path": "internal/auth/jwt.go",
            "function_name": "VerifyJWTToken"
          }
        }
      }
    ]
  },
  "summary": {
    "total_results": 3,
    "languages_found": ["python", "javascript", "go"],
    "average_similarity": 0.893,
    "pattern_insights": {
      "common_concepts": ["token_verification", "error_handling", "expiration_check"],
      "language_differences": {
        "python": "Uses exceptions for error handling",
        "javascript": "Uses try/catch with custom Error objects",
        "go": "Uses explicit error returns with custom error types"
      }
    }
  }
}
```

### Example 2: Database Connection Patterns

Find database connection and query patterns across languages:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "database connection pooling",
    "languages": ["python", "javascript", "go"],
    "match_count": 3,
    "source_filter": null
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "database connection pooling",
  "results_by_language": {
    "python": [
      {
        "content": "from sqlalchemy import create_engine\nfrom sqlalchemy.pool import QueuePool\n\nclass DatabaseManager:\n    def __init__(self, database_url: str, pool_size: int = 10):\n        self.engine = create_engine(\n            database_url,\n            poolclass=QueuePool,\n            pool_size=pool_size,\n            max_overflow=20,\n            pool_pre_ping=True\n        )\n    \n    def get_connection(self):\n        return self.engine.connect()\n    \n    def execute_query(self, query: str, params: dict = None):\n        with self.get_connection() as conn:\n            return conn.execute(query, params or {})",
        "similarity_score": 0.91,
        "source": "user-service-py",
        "file_context": {
          "language": "Python",
          "file_path": "database/manager.py",
          "class_name": "DatabaseManager"
        }
      }
    ],
    "javascript": [
      {
        "content": "const { Pool } = require('pg');\n\nclass DatabasePool {\n  constructor(config) {\n    this.pool = new Pool({\n      host: config.host,\n      port: config.port,\n      database: config.database,\n      user: config.user,\n      password: config.password,\n      max: config.maxConnections || 10,\n      idleTimeoutMillis: 30000,\n      connectionTimeoutMillis: 2000,\n    });\n  }\n  \n  async query(text, params) {\n    const client = await this.pool.connect();\n    try {\n      const result = await client.query(text, params);\n      return result.rows;\n    } finally {\n      client.release();\n    }\n  }\n  \n  async close() {\n    await this.pool.end();\n  }\n}",
        "similarity_score": 0.88,
        "source": "api-gateway-js",
        "file_context": {
          "language": "JavaScript",
          "file_path": "database/pool.js",
          "class_name": "DatabasePool"
        }
      }
    ],
    "go": [
      {
        "content": "package database\n\nimport (\n    \"database/sql\"\n    \"time\"\n    _ \"github.com/lib/pq\"\n)\n\ntype Pool struct {\n    db *sql.DB\n}\n\nfunc NewPool(dsn string) (*Pool, error) {\n    db, err := sql.Open(\"postgres\", dsn)\n    if err != nil {\n        return nil, err\n    }\n    \n    db.SetMaxOpenConns(25)\n    db.SetMaxIdleConns(5)\n    db.SetConnMaxLifetime(5 * time.Minute)\n    \n    if err = db.Ping(); err != nil {\n        return nil, err\n    }\n    \n    return &Pool{db: db}, nil\n}\n\nfunc (p *Pool) Query(query string, args ...interface{}) (*sql.Rows, error) {\n    return p.db.Query(query, args...)\n}\n\nfunc (p *Pool) Close() error {\n    return p.db.Close()\n}",
        "similarity_score": 0.85,
        "source": "payment-service-go",
        "file_context": {
          "language": "Go",
          "file_path": "internal/database/pool.go",
          "struct_name": "Pool"
        }
      }
    ]
  },
  "summary": {
    "total_results": 3,
    "languages_found": ["python", "javascript", "go"],
    "average_similarity": 0.88,
    "architectural_patterns": {
      "common_features": ["connection_pooling", "resource_management", "error_handling"],
      "implementation_differences": {
        "python": "SQLAlchemy with declarative pool configuration",
        "javascript": "pg library with Promise-based API",
        "go": "Standard library with explicit resource management"
      }
    }
  }
}
```

## Advanced Cross-Language Analysis

### Example 3: Error Handling Patterns

Compare error handling approaches across languages:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "retry mechanism with exponential backoff",
    "languages": ["python", "javascript", "typescript", "go"],
    "match_count": 4
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "retry mechanism with exponential backoff", 
  "results_by_language": {
    "python": [
      {
        "content": "import time\nimport random\nfrom typing import Callable, Any\n\ndef retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Any:\n    for attempt in range(max_retries + 1):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_retries:\n                raise e\n            \n            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)\n            time.sleep(delay)\n            print(f\"Retry {attempt + 1}/{max_retries} after {delay:.2f}s\")",
        "similarity_score": 0.93,
        "source": "data-processor",
        "file_context": {
          "language": "Python",
          "function_name": "retry_with_backoff"
        }
      }
    ],
    "javascript": [
      {
        "content": "async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {\n  for (let attempt = 0; attempt <= maxRetries; attempt++) {\n    try {\n      return await fn();\n    } catch (error) {\n      if (attempt === maxRetries) {\n        throw error;\n      }\n      \n      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;\n      console.log(`Retry ${attempt + 1}/${maxRetries} after ${delay}ms`);\n      await new Promise(resolve => setTimeout(resolve, delay));\n    }\n  }\n}",
        "similarity_score": 0.91,
        "source": "notification-service",
        "file_context": {
          "language": "JavaScript",
          "function_name": "retryWithBackoff"
        }
      }
    ],
    "typescript": [
      {
        "content": "interface RetryOptions {\n  maxRetries: number;\n  baseDelay: number;\n  maxDelay?: number;\n}\n\nasync function retryWithBackoff<T>(\n  operation: () => Promise<T>,\n  options: RetryOptions = { maxRetries: 3, baseDelay: 1000 }\n): Promise<T> {\n  const { maxRetries, baseDelay, maxDelay = 30000 } = options;\n  \n  for (let attempt = 0; attempt <= maxRetries; attempt++) {\n    try {\n      return await operation();\n    } catch (error) {\n      if (attempt === maxRetries) {\n        throw error;\n      }\n      \n      const exponentialDelay = baseDelay * Math.pow(2, attempt);\n      const jitter = Math.random() * baseDelay;\n      const delay = Math.min(exponentialDelay + jitter, maxDelay);\n      \n      console.warn(`Attempt ${attempt + 1} failed, retrying in ${delay}ms`, error);\n      await new Promise(resolve => setTimeout(resolve, delay));\n    }\n  }\n}",
        "similarity_score": 0.94,
        "source": "api-client-ts",
        "file_context": {
          "language": "TypeScript",
          "function_name": "retryWithBackoff"
        }
      }
    ],
    "go": [
      {
        "content": "package retry\n\nimport (\n    \"context\"\n    \"fmt\"\n    \"math\"\n    \"math/rand\"\n    \"time\"\n)\n\ntype Config struct {\n    MaxRetries int\n    BaseDelay  time.Duration\n    MaxDelay   time.Duration\n}\n\nfunc WithBackoff(ctx context.Context, config Config, operation func() error) error {\n    for attempt := 0; attempt <= config.MaxRetries; attempt++ {\n        err := operation()\n        if err == nil {\n            return nil\n        }\n        \n        if attempt == config.MaxRetries {\n            return fmt.Errorf(\"operation failed after %d attempts: %w\", config.MaxRetries+1, err)\n        }\n        \n        exponentialDelay := time.Duration(float64(config.BaseDelay) * math.Pow(2, float64(attempt)))\n        jitter := time.Duration(rand.Float64() * float64(config.BaseDelay))\n        delay := exponentialDelay + jitter\n        \n        if delay > config.MaxDelay {\n            delay = config.MaxDelay\n        }\n        \n        select {\n        case <-ctx.Done():\n            return ctx.Err()\n        case <-time.After(delay):\n        }\n    }\n    \n    return nil\n}",
        "similarity_score": 0.89,
        "source": "order-service",
        "file_context": {
          "language": "Go",
          "file_path": "pkg/retry/backoff.go",
          "function_name": "WithBackoff"
        }
      }
    ]
  },
  "summary": {
    "total_results": 4,
    "languages_found": ["python", "javascript", "typescript", "go"],
    "average_similarity": 0.917,
    "pattern_analysis": {
      "common_elements": [
        "exponential_delay_calculation",
        "jitter_for_thundering_herd",
        "max_retry_limit",
        "configurable_base_delay"
      ],
      "language_specific_features": {
        "python": "Exception handling with type hints",
        "javascript": "Promise-based with setTimeout",
        "typescript": "Generic types with interface configuration",
        "go": "Context cancellation with struct configuration"
      },
      "complexity_progression": {
        "python": "Simple and readable",
        "javascript": "Promise-based async handling", 
        "typescript": "Type-safe with advanced configuration",
        "go": "Context-aware with structured error handling"
      }
    }
  }
}
```

### Example 4: Caching Strategies

Compare caching implementations across different languages:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "LRU cache implementation",
    "languages": ["python", "javascript", "go"],
    "match_count": 2,
    "source_filter": null
  }
}
```

## Language-Specific Search Examples

### Example 5: Python-Only Patterns

Search for Python-specific patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "decorator with functools wraps",
    "languages": ["python"],
    "match_count": 5
  }
}
```

### Example 6: JavaScript/TypeScript Modern Patterns

Search for modern JS/TS patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "async generator function",
    "languages": ["javascript", "typescript"],
    "match_count": 3
  }
}
```

### Example 7: Go Concurrency Patterns

Search for Go-specific concurrency patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "worker pool with channels",
    "languages": ["go"],
    "match_count": 4
  }
}
```

## Comparative Analysis Examples

### Example 8: API Client Implementations

Compare how different languages implement HTTP API clients:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "REST API client with authentication headers",
    "languages": ["python", "javascript", "go"],
    "match_count": 3,
    "include_file_context": true
  }
}
```

This search will return implementations showing:

- **Python**: Using `requests` library with session management
- **JavaScript**: Using `fetch` API or `axios` with interceptors  
- **Go**: Using `net/http` with custom Transport and middleware

### Example 9: Configuration Management

Compare configuration loading patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "environment variable configuration loading",
    "languages": ["python", "javascript", "go"],
    "match_count": 3
  }
}
```

### Example 10: Testing Patterns

Find testing patterns across languages:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "mock HTTP requests in unit tests",
    "languages": ["python", "javascript", "go"],
    "match_count": 3
  }
}
```

## Advanced Query Techniques

### Using Specific Technical Terms

Search for specific architectural patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "observer pattern event subscription",
    "languages": ["python", "javascript", "typescript", "go"],
    "match_count": 4
  }
}
```

### Framework-Specific Searches

Find framework-specific implementations:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "middleware pipeline request processing",
    "languages": ["python", "javascript", "go"],
    "match_count": 3
  }
}
```

### Performance-Focused Searches

Search for performance optimization patterns:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "connection pooling resource management",
    "languages": ["python", "javascript", "go"],
    "match_count": 3
  }
}
```

## Use Cases for Cross-Language Search

### 1. Architecture Standardization

- Ensure consistent patterns across microservices
- Standardize error handling approaches
- Align authentication and authorization methods

### 2. Technology Migration

- Find equivalent patterns when migrating from one language to another
- Understand how to implement existing logic in a new language
- Compare performance characteristics across implementations

### 3. Code Review and Learning

- Learn best practices from existing implementations
- Identify inconsistencies in coding patterns
- Share knowledge across language-specific teams

### 4. API Design Consistency

- Ensure consistent API patterns across different service languages
- Standardize request/response handling
- Align error codes and message formats

### 5. Security Pattern Validation

- Ensure consistent security implementations
- Validate authentication/authorization patterns
- Check for security best practices across languages

## Best Practices for Cross-Language Searches

### 1. Use Semantic Queries

Instead of searching for exact code, search for concepts:

- ✅ "JWT token validation"
- ❌ "jwt.decode(token)"

### 2. Include Context

Enable `include_file_context` to understand implementation context:

```json
{
  "include_file_context": true
}
```

### 3. Adjust Match Count

Use appropriate match counts based on your needs:

- **Exploration**: 5-10 matches per language
- **Specific Research**: 2-3 matches per language
- **Comprehensive Analysis**: 10+ matches per language

### 4. Filter by Source

Use `source_filter` when comparing specific repositories:

```json
{
  "source_filter": "my-company-repositories"
}
```

### 5. Analyze Similarity Scores

Pay attention to similarity scores to understand result relevance:

- **0.9+**: Highly relevant matches
- **0.8-0.9**: Good matches with similar concepts
- **0.7-0.8**: Loosely related patterns
- **<0.7**: May not be directly relevant

## Interpreting Results

### Understanding the Response Structure

Each cross-language search returns:

1. **Results by Language**: Code examples grouped by programming language
2. **Similarity Scores**: How closely each result matches your query (0.0-1.0)
3. **File Context**: Location and metadata about each code example
4. **Summary Statistics**: Overall search statistics and insights

### Analyzing Pattern Differences

Look for:

- **Common Concepts**: Shared ideas across languages
- **Implementation Differences**: How each language handles the same concept
- **Best Practices**: Which implementations follow language-specific conventions
- **Performance Implications**: Different approaches to efficiency and resource usage

### Making Architectural Decisions

Use the results to:

- Choose the best pattern for your specific use case
- Understand trade-offs between different implementations
- Ensure consistency across your multi-language architecture
- Learn from proven patterns in production codebases
