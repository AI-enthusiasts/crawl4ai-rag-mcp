# JavaScript/TypeScript Repository Parsing Example

This example demonstrates parsing a JavaScript/TypeScript repository and exploring the extracted code structure across both languages.

## Example Repository: Microsoft VSCode

VSCode contains both JavaScript and TypeScript code, making it an excellent example for multi-language analysis.

### Step 1: Parse the Repository

```json
{
  "tool": "parse_github_repository",
  "arguments": {
    "repo_url": "https://github.com/microsoft/vscode"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository_name": "vscode",
  "clone_path": "/tmp/vscode_20231201_123456",
  "statistics": {
    "files_processed": 2847,
    "classes_created": 156,
    "methods_created": 1247,
    "functions_created": 892,
    "interfaces_created": 234,
    "types_created": 89
  },
  "languages_detected": ["JavaScript", "TypeScript"],
  "language_breakdown": {
    "typescript_files": 2103,
    "javascript_files": 744,
    "tsx_files": 45,
    "jsx_files": 12
  },
  "processing_summary": {
    "batch_count": 57,
    "processing_time_seconds": 156,
    "memory_usage_mb": 234
  }
}
```

### Step 2: Explore Multi-Language Structure

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "explore vscode"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "repository": "vscode",
  "overview": {
    "total_files": 2847,
    "languages": ["JavaScript", "TypeScript"],
    "typescript_summary": {
      "classes": 98,
      "interfaces": 234,
      "types": 89,
      "functions": 567,
      "methods": 823
    },
    "javascript_summary": {
      "classes": 58,
      "functions": 325,
      "methods": 424,
      "es6_features": ["arrow_functions", "destructuring", "classes"]
    },
    "main_modules": [
      "vs.editor",
      "vs.workbench",
      "vs.platform",
      "vs.base"
    ]
  }
}
```

### Step 3: Search TypeScript Interfaces

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (i:Interface) WHERE i.language = 'TypeScript' RETURN i.name, i.module LIMIT 10"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query_result": {
    "query": "TypeScript interfaces",
    "results": [
      {
        "name": "ICodeEditor",
        "module": "vs.editor.common",
        "file_path": "src/vs/editor/common/editorCommon.ts",
        "methods": ["getModel", "setModel", "focus", "hasTextFocus"]
      },
      {
        "name": "ITextModel",
        "module": "vs.editor.common.model",
        "file_path": "src/vs/editor/common/model/textModel.ts", 
        "methods": ["getValue", "setValue", "getLineCount"]
      },
      {
        "name": "IWorkbenchContribution",
        "module": "vs.workbench.common",
        "file_path": "src/vs/workbench/common/contributions.ts",
        "methods": ["startup", "shutdown"]
      }
    ]
  }
}
```

### Step 4: Cross-Language Code Search

```json
{
  "tool": "analyze_code_cross_language",
  "arguments": {
    "query": "event listener registration",
    "languages": ["javascript", "typescript"],
    "match_count": 5,
    "include_file_context": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "event listener registration", 
  "results_by_language": {
    "typescript": [
      {
        "content": "class EventEmitter<T> {\n    private _listeners: Map<string, Function[]> = new Map();\n    \n    on(event: keyof T, listener: Function): void {\n        if (!this._listeners.has(event as string)) {\n            this._listeners.set(event as string, []);\n        }\n        this._listeners.get(event as string)!.push(listener);\n    }\n    \n    emit(event: keyof T, ...args: any[]): void {\n        const listeners = this._listeners.get(event as string);\n        if (listeners) {\n            listeners.forEach(listener => listener(...args));\n        }\n    }\n}",
        "similarity_score": 0.89,
        "source": "vscode",
        "file_context": {
          "url": "neo4j://repository/vscode/class/EventEmitter",
          "metadata": {
            "language": "TypeScript",
            "file_path": "src/vs/base/common/event.ts",
            "class_name": "EventEmitter"
          }
        }
      }
    ],
    "javascript": [
      {
        "content": "function addListener(element, event, handler) {\n    if (element.addEventListener) {\n        element.addEventListener(event, handler, false);\n    } else if (element.attachEvent) {\n        element.attachEvent('on' + event, handler);\n    } else {\n        element['on' + event] = handler;\n    }\n}",
        "similarity_score": 0.82,
        "source": "vscode",
        "file_context": {
          "url": "neo4j://repository/vscode/function/addListener",
          "metadata": {
            "language": "JavaScript",
            "file_path": "src/vs/base/browser/dom.js",
            "function_name": "addListener"
          }
        }
      }
    ]
  },
  "summary": {
    "total_results": 2,
    "languages_found": ["typescript", "javascript"],
    "average_similarity": 0.855
  }
}
```

### Step 5: Find React Components (JSX/TSX)

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "React functional component with hooks",
    "source_filter": "vscode",
    "match_count": 3
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query": "React functional component with hooks",
  "results": [
    {
      "code_example": "const SearchBox: React.FC<SearchBoxProps> = ({ onSearch, placeholder }) => {\n    const [searchTerm, setSearchTerm] = useState('');\n    const [isLoading, setIsLoading] = useState(false);\n    \n    const handleSearch = useCallback(async () => {\n        setIsLoading(true);\n        try {\n            await onSearch(searchTerm);\n        } finally {\n            setIsLoading(false);\n        }\n    }, [searchTerm, onSearch]);\n    \n    return (\n        <div className=\"search-box\">\n            <input \n                type=\"text\"\n                value={searchTerm}\n                onChange={(e) => setSearchTerm(e.target.value)}\n                placeholder={placeholder}\n            />\n            <button onClick={handleSearch} disabled={isLoading}>\n                {isLoading ? 'Searching...' : 'Search'}\n            </button>\n        </div>\n    );\n};",
      "similarity_score": 0.91,
      "confidence_score": 0.88,
      "validation_status": "validated",
      "source": "vscode",
      "metadata": {
        "component_name": "SearchBox",
        "file_path": "src/vs/workbench/contrib/search/browser/searchBox.tsx",
        "language": "TypeScript",
        "code_type": "react_component",
        "hooks_used": ["useState", "useCallback"]
      }
    }
  ]
}
```

### Step 6: Analyze Import Patterns

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (f:File)-[:HAS_IMPORT]->(i:Import) WHERE f.language IN ['JavaScript', 'TypeScript'] RETURN i.source, COUNT(*) as usage_count ORDER BY usage_count DESC LIMIT 10"
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "query_result": {
    "query": "Most imported modules",
    "results": [
      {
        "module": "vs/base/common/event",
        "usage_count": 234,
        "type": "internal"
      },
      {
        "module": "vs/platform/registry/common/platform",
        "usage_count": 187,
        "type": "internal"
      },
      {
        "module": "react",
        "usage_count": 67,
        "type": "external"
      },
      {
        "module": "vs/base/common/lifecycle",
        "usage_count": 156,
        "type": "internal"
      }
    ]
  }
}
```

### Step 7: Validate JavaScript/TypeScript Code

Create a test script:

```typescript
// ./analysis_scripts/user_scripts/test_vscode_patterns.ts
import { EventEmitter } from 'vs/base/common/event';
import { ICodeEditor } from 'vs/editor/common/editorCommon';
import * as React from 'react';

interface EditorConfig {
    theme: string;
    fontSize: number;
}

class MyEditor implements ICodeEditor {
    private _eventEmitter = new EventEmitter<EditorEvents>();
    
    getModel(): ITextModel | null {
        return this._model;
    }
    
    setModel(model: ITextModel): void {
        this._model = model;
        this._eventEmitter.emit('modelChanged', model);
    }
    
    focus(): void {
        this._domNode.focus();
    }
}

// React component using VSCode patterns
const EditorComponent: React.FC<{config: EditorConfig}> = ({ config }) => {
    const [editor, setEditor] = React.useState<MyEditor | null>(null);
    
    React.useEffect(() => {
        const newEditor = new MyEditor();
        setEditor(newEditor);
        return () => newEditor.dispose();
    }, []);
    
    return <div className="editor-container" />;
};
```

Validate the TypeScript code:

```json
{
  "tool": "check_ai_script_hallucinations_enhanced",
  "arguments": {
    "script_path": "test_vscode_patterns.ts",
    "include_code_suggestions": true,
    "detailed_analysis": true
  }
}
```

**Expected Response:**

```json
{
  "success": true,
  "script_path": "test_vscode_patterns.ts",
  "language": "TypeScript",
  "validation_summary": {
    "total_imports": 3,
    "valid_imports": 3,
    "invalid_imports": 0,
    "total_interface_usage": 2,
    "valid_interface_usage": 2,
    "total_class_definitions": 1,
    "confidence_score": 0.92
  },
  "detailed_analysis": {
    "imports": [
      {
        "import": "import { EventEmitter } from 'vs/base/common/event'",
        "status": "valid",
        "confidence": 0.98,
        "found_in_repository": "vscode",
        "validation_source": "neo4j_structural"
      },
      {
        "import": "import { ICodeEditor } from 'vs/editor/common/editorCommon'",
        "status": "valid",
        "confidence": 0.95,
        "found_in_repository": "vscode"
      }
    ],
    "interfaces": [
      {
        "interface": "ICodeEditor",
        "usage": "class MyEditor implements ICodeEditor",
        "status": "valid",
        "confidence": 0.94,
        "required_methods": ["getModel", "setModel", "focus", "hasTextFocus"],
        "implemented_methods": ["getModel", "setModel", "focus"],
        "missing_methods": ["hasTextFocus"]
      }
    ],
    "typescript_features": {
      "generics": ["EventEmitter<EditorEvents>"],
      "interfaces": ["EditorConfig", "ICodeEditor"], 
      "react_patterns": ["functional_component", "hooks"]
    }
  },
  "suggestions": [
    {
      "type": "missing_method",
      "message": "ICodeEditor interface requires hasTextFocus() method",
      "fix": "Add hasTextFocus(): boolean method to MyEditor class"
    },
    {
      "type": "improvement",
      "message": "Consider using VSCode's lifecycle patterns",
      "example_code": "import { Disposable } from 'vs/base/common/lifecycle';\n\nclass MyEditor extends Disposable implements ICodeEditor {"
    }
  ]
}
```

## Advanced JavaScript/TypeScript Features

### ES6+ Feature Detection

Find all arrow functions:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (f:Function) WHERE f.type = 'arrow_function' RETURN f.name, f.module, f.language LIMIT 10"
  }
}
```

Find async/await patterns:

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "async await error handling pattern",
    "languages": ["javascript", "typescript"],
    "match_count": 5
  }
}
```

### React Component Analysis

Find all React components:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (f:Function) WHERE f.type = 'react_component' RETURN f.name, f.file_path LIMIT 10"
  }
}
```

Search for specific React patterns:

```json
{
  "tool": "smart_code_search",
  "arguments": {
    "query": "useEffect cleanup function",
    "languages": ["typescript", "javascript"],
    "source_filter": "vscode"
  }
}
```

### TypeScript-Specific Analysis

Find generic types:

```json
{
  "tool": "query_knowledge_graph", 
  "arguments": {
    "command": "query MATCH (t:Type) WHERE t.has_generics = true RETURN t.name, t.module LIMIT 10"
  }
}
```

Find interface inheritance:

```json
{
  "tool": "query_knowledge_graph",
  "arguments": {
    "command": "query MATCH (i1:Interface)-[:EXTENDS]->(i2:Interface) RETURN i1.name, i2.name, i1.module LIMIT 10"
  }
}
```

## JavaScript/TypeScript Analysis Benefits

### 1. Modern JavaScript Validation

- **ES6+ Features**: Arrow functions, destructuring, template literals
- **Module Systems**: ES6 imports/exports, CommonJS require
- **Async Patterns**: Promise chains, async/await, generator functions

### 2. TypeScript Type Safety

- **Interface Compliance**: Verify implementations match interfaces
- **Type Annotations**: Validate parameter and return types
- **Generic Usage**: Ensure proper generic type usage

### 3. React Development Support

- **Component Patterns**: Functional vs class components
- **Hook Usage**: useState, useEffect, custom hooks
- **JSX Validation**: Proper JSX syntax and patterns

### 4. Code Quality Insights

- **Import Organization**: Find circular dependencies
- **Dead Code Detection**: Unused exports and imports
- **Pattern Consistency**: Consistent coding patterns across files

## Common JavaScript/TypeScript Patterns Detected

- **Class Definitions**: ES6 classes with methods and properties
- **Function Types**: Regular functions, arrow functions, async functions
- **Import/Export**: ES6 modules, CommonJS, dynamic imports
- **TypeScript Features**: Interfaces, types, enums, generics
- **React Patterns**: Components, hooks, JSX, props
- **Event Handling**: Event listeners, emitters, callbacks
- **Error Handling**: Try/catch, error boundaries, promise rejection
- **Design Patterns**: Observer, factory, singleton implementations

## Integration with Development Workflow

1. **Code Review**: Validate new JavaScript/TypeScript code against existing patterns
2. **Refactoring**: Find all usages of deprecated patterns
3. **Documentation**: Generate API documentation from parsed interfaces
4. **Migration**: Identify JavaScript files that could benefit from TypeScript
5. **Training**: Help team members learn existing codebase patterns
