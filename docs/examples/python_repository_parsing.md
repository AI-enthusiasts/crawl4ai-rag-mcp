# Python Repository Parsing Example

This example demonstrates parsing a Python repository and exploring the extracted code structure.

## Example Repository: FastAPI

FastAPI is a modern Python web framework that provides excellent examples of Python classes, functions, and type hints.

### Step 1: Parse the Repository

```json
{
  "tool": "parse_github_repository",
  "arguments": {
    "repo_url": "https://github.com/tiangolo/fastapi"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository_name": "fastapi",
  "clone_path": "/tmp/fastapi_20231201_123456",
  "statistics": {
    "files_processed": 147,
    "classes_created": 45,
    "methods_created": 312,
    "functions_created": 128
  },
  "languages_detected": ["Python"],
  "processing_summary": {
    "batch_count": 3,
    "processing_time_seconds": 28,
    "total_python_files": 147
  }
}
```

### Step 2: Explore Repository Structure

```json
{
  "tool": "query_knowledge_graph", 
  "arguments": {
    "command": "explore fastapi"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository": "fastapi",
  "overview": {
    "total_files": 147,
    "total_classes": 45,
    "total_methods": 312,
    "total_functions": 128,
    "languages": ["Python"],
    "main_packages": [
      "fastapi.routing",
      "fastapi.security",
      "fastapi.dependencies",
      "fastapi.middleware"
    ]
  },
  "top_classes": [
    {
      "name": "FastAPI",
      "module": "fastapi.applications",
      "method_count": 15,
      "description": "Main FastAPI application class"
    },
    {
      "name": "APIRouter",
      "module": "fastapi.routing",
      "method_count": 12,
      "description": "Router class for organizing endpoints"
    }
  ]
}
```

### Step 3: Search for Specific Classes

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "class FastAPI"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "class_details": {
    "name": "FastAPI",
    "full_name": "fastapi.applications.FastAPI",
    "file_path": "fastapi/applications.py",
    "module": "fastapi.applications",
    "methods": [
      {
        "name": "__init__",
        "params": ["self", "debug", "routes", "title", "description"],
        "return_type": "None",
        "line_number": 45
      },
      {
        "name": "add_api_route",
        "params": ["self", "path", "endpoint", "methods"],
        "return_type": "None", 
        "line_number": 89
      },
      {
        "name": "get",
        "params": ["self", "path"],
        "return_type": "Callable",
        "line_number": 156
      }
    ],
    "attributes": [
      {
        "name": "router",
        "type": "APIRouter"
      },
      {
        "name": "middleware_stack", 
        "type": "ASGIApp"
      }
    ]
  }
}
```

### Step 4: Find Methods Across Classes

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "method get"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "method_search": {
    "query": "get",
    "results": [
      {
        "class": "FastAPI",
        "method": "get",
        "module": "fastapi.applications",
        "params": ["self", "path"],
        "return_type": "Callable",
        "description": "Decorator for GET endpoints"
      },
      {
        "class": "APIRouter", 
        "method": "get",
        "module": "fastapi.routing",
        "params": ["self", "path"],
        "return_type": "Callable",
        "description": "Router GET decorator"
      },
      {
        "class": "Request",
        "method": "get_header",
        "module": "fastapi.requests",
        "params": ["self", "key", "default"],
        "return_type": "str | None"
      }
    ],
    "total_matches": 3
  }
}
```

### Step 5: Extract and Index Code Examples

```json
{
  "tool": "extract_and_index_repository_code",
  "arguments": {
    "repo_name": "fastapi"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository_name": "fastapi",
  "indexed_count": 485,
  "extraction_summary": {
    "classes": 45,
    "methods": 312, 
    "functions": 128
  },
  "storage_summary": {
    "embeddings_generated": 485,
    "examples_stored": 485,
    "total_code_words": 15420
  }
}
```

### Step 6: Smart Code Search

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "dependency injection with async function",
    "source_filter": "fastapi",
    "match_count": 3,
    "validation_mode": "balanced"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "dependency injection with async function",
  "results": [
    {
      "code_example": "async def get_current_user(\n    token: str = Depends(oauth2_scheme),\n    db: Session = Depends(get_database)\n) -> User:\n    credentials_exception = HTTPException(\n        status_code=status.HTTP_401_UNAUTHORIZED,\n        detail=\"Could not validate credentials\"\n    )\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        username: str = payload.get(\"sub\")\n        if username is None:\n            raise credentials_exception\n    except JWTError:\n        raise credentials_exception\n    \n    user = db.query(User).filter(User.username == username).first()\n    if user is None:\n        raise credentials_exception\n    return user",
      "similarity_score": 0.91,
      "confidence_score": 0.87,
      "validation_status": "validated",
      "source": "fastapi",
      "metadata": {
        "class_name": null,
        "function_name": "get_current_user",
        "module": "fastapi.security.utils",
        "file_path": "fastapi/security/utils.py",
        "language": "Python",
        "code_type": "function"
      }
    },
    {
      "code_example": "@app.get(\"/users/me\", response_model=User)\nasync def read_users_me(\n    current_user: User = Depends(get_current_active_user)\n):\n    return current_user",
      "similarity_score": 0.84,
      "confidence_score": 0.89,
      "validation_status": "validated",
      "source": "fastapi",
      "metadata": {
        "function_name": "read_users_me",
        "module": "example_dependencies",
        "language": "Python",
        "code_type": "function"
      }
    }
  ],
  "search_summary": {
    "total_results": 2,
    "average_similarity": 0.875,
    "average_confidence": 0.88,
    "validation_success_rate": 1.0
  }
}
```

### Step 7: Validate AI-Generated Code

Create a test script:

```python
# ./analysis_scripts/user_scripts/test_fastapi.py
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # This should validate against FastAPI's actual patterns
    return {"username": "testuser"}

@app.get("/protected")
async def protected_route(user = Depends(get_current_user)):
    return user
```

Validate the script:

```json
{
  "tool": "check_ai_script_hallucinations_enhanced",
  "arguments": {
    "script_path": "test_fastapi.py",
    "include_code_suggestions": true,
    "detailed_analysis": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "script_path": "test_fastapi.py",
  "validation_summary": {
    "total_imports": 3,
    "valid_imports": 3,
    "invalid_imports": 0,
    "total_function_calls": 4,
    "valid_function_calls": 4,
    "confidence_score": 0.95
  },
  "detailed_analysis": {
    "imports": [
      {
        "import": "from fastapi import FastAPI, Depends",
        "status": "valid",
        "confidence": 1.0,
        "found_in_repository": "fastapi"
      },
      {
        "import": "from fastapi.security import OAuth2PasswordBearer", 
        "status": "valid",
        "confidence": 1.0,
        "found_in_repository": "fastapi"
      }
    ],
    "function_calls": [
      {
        "function": "FastAPI()",
        "status": "valid",
        "confidence": 1.0,
        "validation_source": "neo4j_structural"
      },
      {
        "function": "OAuth2PasswordBearer(tokenUrl='token')",
        "status": "valid", 
        "confidence": 0.92,
        "validation_source": "semantic_similarity"
      }
    ]
  },
  "suggestions": [
    {
      "type": "best_practice",
      "message": "Consider adding response models for better API documentation",
      "example_code": "@app.get('/protected', response_model=UserResponse)\nasync def protected_route(user: User = Depends(get_current_user)):\n    return user"
    }
  ],
  "overall_assessment": "HIGH_CONFIDENCE - Code follows FastAPI patterns and uses valid imports/functions"
}
```

## Advanced Python Analysis Examples

### Custom Cypher Queries

Find all classes that inherit from a base class:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (base:Class {name: 'BaseModel'})<-[:INHERITS_FROM]-(child:Class) RETURN child.name, child.module LIMIT 10"
  }
}
```

Find all async methods:

```json
{
  "tool": "query_knowledge_graph", 
  "arguments": {
    "command": "query MATCH (m:Method) WHERE m.is_async = true RETURN m.name, m.class_name, m.module LIMIT 10"
  }
}
```

### Cross-Repository Analysis

Compare authentication patterns across multiple Python repositories:

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "JWT token authentication decorator",
    "match_count": 5,
    "validation_mode": "thorough"
  }
}
```

This will search across all indexed Python repositories and return validated examples of JWT authentication implementations.

## Benefits of Python Repository Analysis

1. **Import Validation**: Verify that AI-generated imports are real and correctly used
2. **Method Signature Validation**: Ensure function calls match actual method signatures
3. **Pattern Discovery**: Find similar implementations across different Python projects  
4. **Code Suggestions**: Get real examples from parsed repositories
5. **Architecture Understanding**: Explore how large Python codebases are structured

## Common Python Patterns Detected

- **Class Inheritance**: `Class A` inherits from `Class B`
- **Decorator Usage**: `@app.route`, `@property`, etc.
- **Dependency Injection**: FastAPI/Flask dependency patterns
- **Async/Await Patterns**: Async function definitions and calls
- **Type Hints**: Modern Python typing annotations
- **Context Managers**: `with` statement usage patterns
- **Exception Handling**: Try/except patterns and custom exceptions
