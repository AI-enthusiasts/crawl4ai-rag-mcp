# Go Repository Parsing Example

This example demonstrates parsing a Go repository and exploring the extracted code structure, including structs, interfaces, methods, and packages.

## Example Repository: Kubernetes

Kubernetes is a large Go project that showcases excellent Go patterns including interfaces, structs, methods, and package organization.

### Step 1: Parse the Repository

```json
{
  "tool": "parse_github_repository",
  "arguments": {
    "repo_url": "https://github.com/kubernetes/kubernetes"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository_name": "kubernetes",
  "clone_path": "/tmp/kubernetes_20231201_123456",
  "statistics": {
    "files_processed": 4523,
    "structs_created": 892,
    "interfaces_created": 234,
    "functions_created": 1567,
    "methods_created": 2103,
    "types_created": 445
  },
  "languages_detected": ["Go"],
  "processing_summary": {
    "batch_count": 91,
    "processing_time_seconds": 287,
    "memory_usage_mb": 456,
    "go_specific": {
      "packages_analyzed": 234,
      "exported_functions": 1203,
      "exported_types": 567
    }
  }
}
```

### Step 2: Explore Go Package Structure

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "explore kubernetes"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository": "kubernetes",
  "overview": {
    "total_files": 4523,
    "language": "Go",
    "go_summary": {
      "packages": 234,
      "structs": 892,
      "interfaces": 234,
      "functions": 1567,
      "methods": 2103,
      "exported_symbols": 1770
    },
    "main_packages": [
      "k8s.io/kubernetes/pkg/kubelet",
      "k8s.io/kubernetes/pkg/controller",
      "k8s.io/kubernetes/pkg/scheduler",
      "k8s.io/kubernetes/pkg/proxy"
    ],
    "top_structs": [
      {
        "name": "Pod",
        "package": "k8s.io/api/core/v1",
        "field_count": 15,
        "exported": true
      },
      {
        "name": "Deployment",
        "package": "k8s.io/api/apps/v1", 
        "field_count": 8,
        "exported": true
      }
    ]
  }
}
```

### Step 3: Search for Go Interfaces

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (i:Interface) WHERE i.language = 'Go' AND i.exported = true RETURN i.name, i.package, SIZE(i.methods) as method_count ORDER BY method_count DESC LIMIT 10"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query_result": {
    "query": "Go exported interfaces",
    "results": [
      {
        "name": "Interface",
        "package": "k8s.io/apimachinery/pkg/runtime",
        "method_count": 8,
        "methods": [
          "GetObjectKind",
          "DeepCopyObject", 
          "GetNamespace",
          "GetName",
          "GetUID",
          "GetResourceVersion"
        ]
      },
      {
        "name": "Client",
        "package": "k8s.io/client-go/rest",
        "method_count": 6,
        "methods": [
          "Get",
          "Post", 
          "Put",
          "Delete",
          "Patch"
        ]
      },
      {
        "name": "Manager",
        "package": "k8s.io/kubernetes/pkg/controller",
        "method_count": 5,
        "methods": [
          "Start",
          "Stop",
          "Add",
          "Update",
          "Delete"
        ]
      }
    ]
  }
}
```

### Step 4: Analyze Go Struct Definitions

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (s:Struct) WHERE s.name = 'Pod' RETURN s.name, s.package, s.fields, s.exported"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query_result": {
    "struct_details": {
      "name": "Pod",
      "package": "k8s.io/api/core/v1",
      "exported": true,
      "file_path": "staging/src/k8s.io/api/core/v1/types.go",
      "fields": [
        {
          "name": "TypeMeta",
          "type": "metav1.TypeMeta",
          "exported": true,
          "json_tag": "`json:\",inline\"`"
        },
        {
          "name": "ObjectMeta",
          "type": "metav1.ObjectMeta", 
          "exported": true,
          "json_tag": "`json:\"metadata,omitempty\"`"
        },
        {
          "name": "Spec",
          "type": "PodSpec",
          "exported": true,
          "json_tag": "`json:\"spec,omitempty\"`"
        },
        {
          "name": "Status", 
          "type": "PodStatus",
          "exported": true,
          "json_tag": "`json:\"status,omitempty\"`"
        }
      ]
    }
  }
}
```

### Step 5: Find Methods with Receivers

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (m:Method) WHERE m.receiver_type = 'Pod' RETURN m.name, m.receiver_name, m.exported, m.package LIMIT 10"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query_result": {
    "methods": [
      {
        "name": "DeepCopy",
        "receiver_name": "p",
        "receiver_type": "*Pod",
        "exported": true,
        "package": "k8s.io/api/core/v1",
        "return_type": "*Pod"
      },
      {
        "name": "GetName",
        "receiver_name": "p", 
        "receiver_type": "*Pod",
        "exported": true,
        "package": "k8s.io/api/core/v1",
        "return_type": "string"
      },
      {
        "name": "GetNamespace",
        "receiver_name": "p",
        "receiver_type": "*Pod", 
        "exported": true,
        "package": "k8s.io/api/core/v1",
        "return_type": "string"
      }
    ]
  }
}
```

### Step 6: Cross-Language Comparison

Compare similar patterns across Go, Python, and JavaScript:

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "HTTP client with retry logic",
    "languages": ["go", "python", "javascript"],
    "match_count": 3,
    "include_file_context": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "HTTP client with retry logic",
  "results_by_language": {
    "go": [
      {
        "content": "type RetryableClient struct {\n    client     *http.Client\n    maxRetries int\n    backoff    time.Duration\n}\n\nfunc (c *RetryableClient) Do(req *http.Request) (*http.Response, error) {\n    var resp *http.Response\n    var err error\n    \n    for i := 0; i <= c.maxRetries; i++ {\n        resp, err = c.client.Do(req)\n        if err == nil && resp.StatusCode < 500 {\n            return resp, nil\n        }\n        \n        if i < c.maxRetries {\n            time.Sleep(c.backoff * time.Duration(i+1))\n        }\n    }\n    \n    return resp, err\n}",
        "similarity_score": 0.91,
        "source": "kubernetes",
        "file_context": {
          "url": "neo4j://repository/kubernetes/struct/RetryableClient",
          "metadata": {
            "language": "Go",
            "file_path": "pkg/util/http/client.go",
            "struct_name": "RetryableClient",
            "package": "k8s.io/kubernetes/pkg/util/http"
          }
        }
      }
    ],
    "python": [
      {
        "content": "import requests\nimport time\n\nclass RetryableHttpClient:\n    def __init__(self, max_retries=3, backoff_factor=1.0):\n        self.session = requests.Session()\n        self.max_retries = max_retries\n        self.backoff_factor = backoff_factor\n    \n    def request(self, method, url, **kwargs):\n        for attempt in range(self.max_retries + 1):\n            try:\n                response = self.session.request(method, url, **kwargs)\n                if response.status_code < 500:\n                    return response\n            except requests.exceptions.RequestException as e:\n                if attempt == self.max_retries:\n                    raise e\n            \n            time.sleep(self.backoff_factor * (2 ** attempt))",
        "similarity_score": 0.87,
        "source": "api-client-py",
        "file_context": {
          "language": "Python",
          "file_path": "client/retry.py"
        }
      }
    ],
    "javascript": [
      {
        "content": "class RetryableHttpClient {\n    constructor(maxRetries = 3, backoffMs = 1000) {\n        this.maxRetries = maxRetries;\n        this.backoffMs = backoffMs;\n    }\n    \n    async request(url, options = {}) {\n        let lastError;\n        \n        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {\n            try {\n                const response = await fetch(url, options);\n                if (response.ok || response.status < 500) {\n                    return response;\n                }\n            } catch (error) {\n                lastError = error;\n            }\n            \n            if (attempt < this.maxRetries) {\n                await this.sleep(this.backoffMs * Math.pow(2, attempt));\n            }\n        }\n        \n        throw lastError;\n    }\n    \n    sleep(ms) {\n        return new Promise(resolve => setTimeout(resolve, ms));\n    }\n}",
        "similarity_score": 0.84,
        "source": "frontend-utils",
        "file_context": {
          "language": "JavaScript",
          "file_path": "src/utils/http.js"
        }
      }
    ]
  },
  "summary": {
    "total_results": 3,
    "languages_found": ["go", "python", "javascript"],
    "average_similarity": 0.873,
    "pattern_comparison": {
      "go": "Uses struct with methods, explicit error handling",
      "python": "Uses class with exception handling, requests library",
      "javascript": "Uses class with async/await, Promise-based"
    }
  }
}
```

### Step 7: Go-Specific Pattern Search

Search for common Go patterns:

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "context cancellation pattern",
    "source_filter": "kubernetes",
    "match_count": 3,
    "validation_mode": "balanced"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "context cancellation pattern",
  "results": [
    {
      "code_example": "func (c *Controller) Run(ctx context.Context) error {\n    defer c.queue.ShutDown()\n    \n    klog.Info(\"Starting controller\")\n    defer klog.Info(\"Shutting down controller\")\n    \n    if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced) {\n        return fmt.Errorf(\"failed to wait for cache sync\")\n    }\n    \n    for i := 0; i < c.workers; i++ {\n        go wait.UntilWithContext(ctx, c.worker, time.Second)\n    }\n    \n    <-ctx.Done()\n    return ctx.Err()\n}",
      "similarity_score": 0.93,
      "confidence_score": 0.91,
      "validation_status": "validated",
      "source": "kubernetes",
      "metadata": {
        "struct_name": "Controller",
        "method_name": "Run", 
        "package": "k8s.io/kubernetes/pkg/controller",
        "file_path": "pkg/controller/controller.go",
        "language": "Go",
        "go_patterns": ["context_cancellation", "goroutines", "channels"]
      }
    },
    {
      "code_example": "func processItem(ctx context.Context, item workqueue.RateLimitingInterface) error {\n    select {\n    case <-ctx.Done():\n        return ctx.Err()\n    default:\n    }\n    \n    obj, shutdown := item.Get()\n    if shutdown {\n        return fmt.Errorf(\"queue shutdown\")\n    }\n    defer item.Done(obj)\n    \n    return c.handleObject(ctx, obj)\n}",
      "similarity_score": 0.88,
      "confidence_score": 0.87,
      "source": "kubernetes",
      "metadata": {
        "function_name": "processItem",
        "go_patterns": ["context_select", "workqueue", "defer"]
      }
    }
  ]
}
```

### Step 8: Validate Go Code Patterns

Create a test Go file:

```go
// ./analysis_scripts/user_scripts/test_k8s_patterns.go
package main

import (
    "context"
    "fmt"
    "time"
    
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/api/core/v1"
    "k8s.io/client-go/kubernetes"
)

type PodManager struct {
    client kubernetes.Interface
}

func (pm *PodManager) CreatePod(ctx context.Context, pod *v1.Pod) (*v1.Pod, error) {
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }
    
    return pm.client.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

func (pm *PodManager) WatchPods(ctx context.Context, namespace string) error {
    watcher, err := pm.client.CoreV1().Pods(namespace).Watch(ctx, metav1.ListOptions{})
    if err != nil {
        return err
    }
    defer watcher.Stop()
    
    for {
        select {
        case event := <-watcher.ResultChan():
            if event.Type == "" {
                return fmt.Errorf("watch closed")
            }
            fmt.Printf("Pod event: %v\n", event.Type)
        case <-ctx.Done():
            return ctx.Err()
        }
    }
}
```

Validate the Go code:

```json
{
  "tool": "check_ai_script_hallucinations_enhanced",
  "arguments": {
    "script_path": "test_k8s_patterns.go", 
    "include_code_suggestions": true,
    "detailed_analysis": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "script_path": "test_k8s_patterns.go",
  "language": "Go",
  "validation_summary": {
    "total_imports": 6,
    "valid_imports": 6,
    "invalid_imports": 0,
    "total_struct_usage": 1,
    "valid_struct_usage": 1,
    "total_interface_usage": 1,
    "confidence_score": 0.94
  },
  "detailed_analysis": {
    "imports": [
      {
        "import": "k8s.io/apimachinery/pkg/apis/meta/v1",
        "alias": "metav1",
        "status": "valid",
        "confidence": 0.98,
        "found_in_repository": "kubernetes"
      },
      {
        "import": "k8s.io/api/core/v1",
        "status": "valid", 
        "confidence": 0.96,
        "found_in_repository": "kubernetes"
      },
      {
        "import": "k8s.io/client-go/kubernetes",
        "status": "valid",
        "confidence": 0.95,
        "found_in_repository": "kubernetes"
      }
    ],
    "struct_usage": [
      {
        "struct": "v1.Pod",
        "usage_context": "method parameter",
        "status": "valid",
        "confidence": 0.97,
        "validation_source": "neo4j_structural"
      }
    ],
    "interface_usage": [
      {
        "interface": "kubernetes.Interface",
        "field": "client",
        "status": "valid",
        "confidence": 0.93,
        "available_methods": ["CoreV1", "AppsV1", "NetworkingV1"]
      }
    ],
    "go_patterns": {
      "context_usage": ["context.Context parameter", "ctx.Done() select"],
      "error_handling": ["explicit error returns", "defer cleanup"],
      "channel_operations": ["select statements", "watcher.ResultChan()"]
    }
  },
  "suggestions": [
    {
      "type": "best_practice",
      "message": "Consider using typed clients for better performance",
      "example_code": "import clientset \"k8s.io/client-go/kubernetes/typed/core/v1\""
    },
    {
      "type": "error_handling",
      "message": "Add timeout to context for long-running operations",
      "example_code": "ctx, cancel := context.WithTimeout(parentCtx, 30*time.Second)\ndefer cancel()"
    }
  ],
  "overall_assessment": "HIGH_CONFIDENCE - Code follows Kubernetes client-go patterns and uses valid APIs"
}
```

## Advanced Go Analysis Features

### Package Dependency Analysis

Find package dependencies:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (f:File)-[:IMPORTS]->(dep:Dependency) WHERE f.language = 'Go' RETURN dep.name, COUNT(*) as usage_count ORDER BY usage_count DESC LIMIT 10"
  }
}
```

### Interface Implementation Analysis

Find all structs that implement an interface:

```json
{
  "tool": "query_knowledge_graph", 
  "arguments": {
    "command": "query MATCH (s:Struct)-[:IMPLEMENTS]->(i:Interface {name: 'Client'}) RETURN s.name, s.package"
  }
}
```

### Method Signature Analysis

Find methods with specific signatures:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (m:Method) WHERE m.return_type CONTAINS 'error' AND SIZE(m.params) > 2 RETURN m.name, m.struct_name, m.params LIMIT 10"
  }
}
```

## Go-Specific Benefits

### 1. Interface Compliance Validation

- **Implementation Checking**: Verify structs implement required interface methods
- **Method Signature Validation**: Ensure method signatures match interface definitions
- **Exported Symbol Detection**: Validate public API usage

### 2. Package Structure Analysis

- **Import Path Validation**: Verify correct package import paths
- **Dependency Tracking**: Map package dependencies and usage
- **Circular Dependency Detection**: Identify problematic import cycles

### 3. Go Idiom Verification

- **Error Handling Patterns**: Validate proper error handling
- **Context Usage**: Check context.Context propagation patterns
- **Channel Operations**: Verify proper channel usage and select statements

### 4. Performance Pattern Analysis

- **Goroutine Usage**: Identify goroutine creation patterns
- **Memory Management**: Find potential memory leaks or inefficiencies
- **Synchronization**: Analyze mutex and channel synchronization

## Common Go Patterns Detected

- **Struct Definitions**: With fields, tags, and embedded types
- **Interface Definitions**: Method signatures and implementations
- **Method Receivers**: Pointer vs value receivers
- **Package Organization**: Internal vs external packages
- **Error Handling**: Explicit error returns and checking
- **Context Patterns**: Context propagation and cancellation
- **Channel Operations**: Select statements, goroutine communication
- **Build Tags**: Conditional compilation directives

## Go Development Workflow Integration

1. **API Validation**: Check if AI-generated Go code uses real Kubernetes APIs
2. **Pattern Compliance**: Ensure code follows established Go and Kubernetes patterns
3. **Interface Verification**: Validate that custom types implement required interfaces
4. **Import Optimization**: Find unused imports and suggest optimizations
5. **Documentation Generation**: Extract struct and interface documentation
6. **Migration Support**: Help migrate between API versions
