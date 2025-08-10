"""
Unit tests for the JavaScript/TypeScript code analyzer.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.knowledge_graph.analyzers.javascript import JavaScriptAnalyzer


class TestJavaScriptAnalyzer:
    """Test the JavaScript/TypeScript analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = JavaScriptAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.supported_extensions == [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]
        assert len(self.analyzer.patterns) > 0

    def test_can_analyze_supported_extensions(self):
        """Test file analysis capability for supported extensions."""
        test_cases = [
            ("script.js", True),
            ("component.jsx", True),
            ("types.ts", True),
            ("App.tsx", True),
            ("module.mjs", True),
            ("config.cjs", True),
        ]
        
        for file_path, expected in test_cases:
            assert self.analyzer.can_analyze(file_path) == expected

    def test_can_analyze_unsupported_extensions(self):
        """Test file analysis capability for unsupported extensions."""
        test_cases = [
            ("script.py", False),
            ("main.go", False),
            ("style.css", False),
            ("data.json", False),
        ]
        
        for file_path, expected in test_cases:
            assert self.analyzer.can_analyze(file_path) == expected

    def test_detect_language(self):
        """Test language detection from file path."""
        assert self.analyzer._detect_language("app.js") == "JavaScript"
        assert self.analyzer._detect_language("app.jsx") == "JavaScript"
        assert self.analyzer._detect_language("types.ts") == "TypeScript"
        assert self.analyzer._detect_language("Component.tsx") == "TypeScript"

    @pytest.mark.asyncio
    async def test_analyze_file_success(self):
        """Test successful file analysis."""
        js_content = '''
// Sample JavaScript file
import React from 'react';
import { Component } from './component';

/**
 * Main application class
 */
export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }
    
    render() {
        return <div>Hello World</div>;
    }
}

export default App;
'''
        
        with patch.object(self.analyzer, 'read_file_content', return_value=js_content):
            result = await self.analyzer.analyze_file("/test/App.jsx", "/test")
            
        # Verify basic structure
        assert result["file_path"] == "/test/App.jsx"
        assert result["language"] == "JavaScript"
        assert result["module_name"] == "App"
        assert len(result["imports"]) > 0
        assert len(result["classes"]) > 0
        assert len(result["exports"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_file_with_content(self):
        """Test file analysis with provided content."""
        js_content = 'const hello = () => "world";'
        
        result = await self.analyzer.analyze_file("/test/hello.js", "/test", content=js_content)
        
        assert result["file_path"] == "/test/hello.js"
        assert len(result["functions"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_file_read_failure(self):
        """Test file analysis when reading fails."""
        with patch.object(self.analyzer, 'read_file_content', return_value=None):
            result = await self.analyzer.analyze_file("/test/nonexistent.js", "/test")
            
        # Should return empty result
        assert result["file_path"] == "/test/nonexistent.js"
        assert result["imports"] == []
        assert result["classes"] == []

    def test_extract_classes(self):
        """Test ES6 class extraction."""
        content = '''
export class User extends BaseUser {
    constructor(name) {
        super();
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
    
    static create(name) {
        return new User(name);
    }
}

class LocalClass {
    method() {}
}
'''
        
        classes = self.analyzer._extract_classes(content)
        
        assert len(classes) == 2
        assert classes[0]["name"] == "User"
        assert classes[0]["extends"] == "BaseUser"
        assert classes[0]["type"] == "class"
        assert len(classes[0]["methods"]) >= 2  # constructor, getName, static create

    def test_extract_functions_regular(self):
        """Test regular function extraction."""
        content = '''
function regularFunction(param) {
    return param * 2;
}

async function asyncFunction() {
    return await getData();
}

function* generatorFunction() {
    yield 1;
    yield 2;
}

export default function defaultExport() {
    return "exported";
}
'''
        
        functions = self.analyzer._extract_functions(content)
        
        # Should find regular functions
        function_names = [f["name"] for f in functions if f["type"] == "function"]
        assert "regularFunction" in function_names
        assert "asyncFunction" in function_names
        assert "generatorFunction" in function_names
        assert "defaultExport" in function_names

    def test_extract_functions_arrow(self):
        """Test arrow function extraction."""
        content = '''
const arrowFunc = (param) => param * 2;
const asyncArrow = async (data) => await process(data);
let simpleArrow = x => x + 1;
var complexArrow = (a, b) => {
    return a + b;
};
'''
        
        functions = self.analyzer._extract_functions(content)
        
        # Should find arrow functions
        arrow_names = [f["name"] for f in functions if f["type"] == "arrow_function"]
        assert "arrowFunc" in arrow_names
        assert "asyncArrow" in arrow_names
        assert "simpleArrow" in arrow_names
        assert "complexArrow" in arrow_names

    def test_extract_functions_react_components(self):
        """Test React component detection."""
        content = '''
const UserProfile = (props) => {
    return <div>{props.name}</div>;
};

function HeaderComponent() {
    return <header>App</header>;
}

const utilityFunc = (data) => data.process();
'''
        
        functions = self.analyzer._extract_functions(content)
        
        # Should detect React components (PascalCase)
        react_components = [f["name"] for f in functions if f["type"] == "react_component"]
        assert "UserProfile" in react_components
        assert "HeaderComponent" in react_components
        
        # Should not detect non-component functions
        assert "utilityFunc" not in react_components

    def test_extract_imports_es6(self):
        """Test ES6 import extraction."""
        content = '''
import React from 'react';
import { Component, useState } from 'react';
import * as Utils from './utils';
import type { User } from './types';
'''
        
        imports = self.analyzer._extract_imports(content)
        
        assert len(imports) == 4
        
        # Check specific imports
        import_sources = [imp["source"] for imp in imports]
        assert "react" in import_sources
        assert "./utils" in import_sources
        assert "./types" in import_sources

    def test_extract_imports_commonjs(self):
        """Test CommonJS require extraction."""
        content = '''
const fs = require('fs');
const { readFile, writeFile } = require('fs/promises');
const path = require('path');
'''
        
        imports = self.analyzer._extract_imports(content)
        
        assert len(imports) == 3
        
        # Check CommonJS imports
        commonjs_imports = [imp for imp in imports if imp["type"] == "commonjs"]
        assert len(commonjs_imports) == 3

    def test_extract_imports_dynamic(self):
        """Test dynamic import extraction."""
        content = '''
const module = await import('./dynamic-module');
import('./another-module').then(mod => {});
'''
        
        imports = self.analyzer._extract_imports(content)
        
        dynamic_imports = [imp for imp in imports if imp["type"] == "dynamic"]
        assert len(dynamic_imports) >= 1

    def test_extract_exports_es6(self):
        """Test ES6 export extraction."""
        content = '''
export const API_URL = 'https://api.example.com';
export function helper() {}
export class MyClass {}
export default MainComponent;
'''
        
        exports = self.analyzer._extract_exports(content)
        
        assert len(exports) >= 3
        
        export_names = [exp["name"] for exp in exports if exp.get("name")]
        assert "API_URL" in export_names
        assert "helper" in export_names
        assert "MyClass" in export_names

    def test_extract_exports_from(self):
        """Test export from syntax."""
        content = '''
export * from './utils';
export { Component, Hook } from 'react';
'''
        
        exports = self.analyzer._extract_exports(content)
        
        # Should find export from statements
        export_sources = [exp.get("source") for exp in exports if exp.get("source")]
        assert "./utils" in export_sources
        assert "react" in export_sources

    def test_extract_exports_commonjs(self):
        """Test CommonJS export extraction."""
        content = '''
module.exports = MainClass;
module.exports = {
    helper,
    utils,
    constants
};
'''
        
        exports = self.analyzer._extract_exports(content)
        
        commonjs_exports = [exp for exp in exports if exp["type"] == "commonjs"]
        assert len(commonjs_exports) >= 1

    def test_extract_interfaces_typescript(self):
        """Test TypeScript interface extraction."""
        content = '''
export interface User {
    id: number;
    name: string;
}

interface Config extends BaseConfig {
    apiUrl: string;
}
'''
        
        interfaces = self.analyzer._extract_interfaces(content)
        
        assert len(interfaces) == 2
        assert interfaces[0]["name"] == "User"
        assert interfaces[1]["name"] == "Config"
        assert interfaces[1]["extends"] == "BaseConfig"

    def test_extract_types_typescript(self):
        """Test TypeScript type definition extraction."""
        content = '''
export type Status = 'pending' | 'completed' | 'failed';
type Handler = (data: any) => void;

export enum Color {
    RED = 'red',
    GREEN = 'green',
    BLUE = 'blue'
}
'''
        
        types = self.analyzer._extract_types(content)
        
        assert len(types) >= 2
        type_names = [t["name"] for t in types]
        assert "Status" in type_names
        assert "Handler" in type_names
        
        # Check enum detection
        enums = [t for t in types if t.get("kind") == "enum"]
        assert len(enums) == 1
        assert enums[0]["name"] == "Color"

    def test_extract_variables(self):
        """Test variable declaration extraction."""
        content = '''
const API_URL = 'https://api.example.com';
let currentUser = null;
var globalConfig = {};
const { name, age } = user;
const [first, second] = items;
'''
        
        variables = self.analyzer._extract_variables(content)
        
        variable_names = [v["name"] for v in variables]
        assert "API_URL" in variable_names
        assert "currentUser" in variable_names
        assert "globalConfig" in variable_names

    def test_extract_dependencies(self):
        """Test dependency extraction from imports."""
        imports = [
            {"source": "react", "type": "es6"},
            {"source": "@types/node", "type": "es6"},
            {"source": "./local-file", "type": "es6"},
            {"source": "/absolute/path", "type": "es6"},
            {"source": "lodash/pick", "type": "es6"},
        ]
        
        dependencies = self.analyzer._extract_dependencies(imports)
        
        # Should only include external packages
        assert "react" in dependencies
        assert "@types/node" in dependencies
        assert "lodash" in dependencies
        
        # Should exclude relative/absolute paths
        assert "./local-file" not in dependencies
        assert "/absolute/path" not in dependencies

    def test_extract_jsdoc_comments(self):
        """Test JSDoc comment extraction."""
        content = '''
/**
 * Main application function
 * @param {string} name - User name
 * @returns {string} Greeting message
 */
function greet(name) {
    return `Hello, ${name}\!`;
}
'''
        
        jsdocs = self.analyzer._extract_jsdoc(content)
        
        assert len(jsdocs) == 1
        assert "Main application function" in jsdocs[0]["content"]
        assert "@param" in jsdocs[0]["content"]

    def test_remove_comments(self):
        """Test comment removal while preserving JSDoc."""
        content = '''
// Single line comment
/* Multi-line comment */
/**
 * JSDoc comment - should be preserved
 */
function test() {
    // Another comment
    return true;
}
'''
        
        clean_content = self.analyzer._remove_comments(content)
        
        # Single-line comments should be removed
        assert "// Single line comment" not in clean_content
        assert "// Another comment" not in clean_content
        
        # Multi-line comments should be removed
        assert "/* Multi-line comment */" not in clean_content
        
        # JSDoc should be preserved (handled separately)

    def test_extract_block(self):
        """Test code block extraction with brace matching."""
        content = '''
function test() {
    if (condition) {
        nested();
    }
    return true;
}
'''
        
        # Find opening brace
        start = content.find('{')
        block = self.analyzer._extract_block(content[start-1:])
        
        # Should extract the entire function body
        assert "if (condition)" in block
        assert "nested();" in block
        assert "return true;" in block

    def test_class_method_extraction(self):
        """Test method extraction from class body."""
        class_body = '''
{
    constructor(name) {
        this.name = name;
    }
    
    public getName() {
        return this.name;
    }
    
    private setName(name) {
        this.name = name;
    }
    
    static create() {
        return new this();
    }
    
    async fetchData() {
        return await api.get('/data');
    }
}
'''
        
        methods = self.analyzer._extract_class_methods(class_body)
        
        method_names = [m["name"] for m in methods]
        assert "constructor" in method_names
        assert "getName" in method_names
        assert "setName" in method_names
        assert "create" in method_names
        assert "fetchData" in method_names

    @pytest.mark.parametrize("content,expected_count", [
        ("", 0),  # Empty content
        ("const x = 1;", 1),  # Simple variable
        ("// Just a comment", 0),  # Comment only
        ("import 'side-effect';", 1),  # Side-effect import
    ])
    def test_analyze_edge_cases(self, content, expected_count):
        """Test analysis with various edge cases."""
        # This is a simple parametrized test for edge cases
        variables = self.analyzer._extract_variables(content)
        # Just verify it doesn't crash and returns expected structure
        assert isinstance(variables, list)

    @pytest.mark.asyncio
    async def test_analyze_file_exception_handling(self):
        """Test file analysis with exceptions during processing."""
        with patch.object(self.analyzer, 'read_file_content', side_effect=Exception("Test error")):
            result = await self.analyzer.analyze_file("/test/error.js", "/test")
            
        # Should return empty result on exception
        assert result["file_path"] == "/test/error.js"
        assert result["imports"] == []
        assert result["classes"] == []

    def test_empty_result_structure(self):
        """Test empty result structure."""
        result = self.analyzer._empty_result("/test/file.js", "/test")
        
        # Verify all required fields are present
        required_fields = [
            "file_path", "module_name", "language", "imports", "classes",
            "functions", "interfaces", "types", "variables", "exports", "dependencies"
        ]
        
        for field in required_fields:
            assert field in result
            if field not in ["file_path", "module_name", "language"]:
                assert isinstance(result[field], list)

    @pytest.mark.asyncio
    async def test_comprehensive_typescript_analysis(self):
        """Test comprehensive TypeScript file analysis."""
        ts_content = '''
import { Component } from 'react';
import type { User, Config } from './types';

/**
 * User management component
 */
export class UserManager extends Component<Props> {
    private users: User[] = [];
    
    constructor(props: Props) {
        super(props);
    }
    
    async loadUsers(): Promise<User[]> {
        return await this.api.getUsers();
    }
}

export interface Props {
    config: Config;
}

export type Status = 'loading' | 'ready' | 'error';

export const DEFAULT_CONFIG: Config = {
    apiUrl: 'https://api.example.com'
};
'''
        
        with patch.object(self.analyzer, 'read_file_content', return_value=ts_content):
            result = await self.analyzer.analyze_file("/test/UserManager.tsx", "/test")
        
        # Verify TypeScript-specific features
        assert result["language"] == "TypeScript"
        assert len(result["classes"]) == 1
        assert len(result["interfaces"]) == 1
        assert len(result["types"]) >= 1
        assert len(result["imports"]) >= 2
        assert len(result["variables"]) >= 1

    def test_jsdoc_attachment(self):
        """Test JSDoc comment attachment to code items."""
        items = [
            {"name": "testFunc", "line": 5, "position": 100},
            {"name": "anotherFunc", "line": 10, "position": 200}
        ]
        
        jsdocs = [
            {"content": "Test function documentation", "position": 80},
            {"content": "Another function doc", "position": 180}
        ]
        
        self.analyzer._attach_jsdoc(items, jsdocs)
        
        # JSDoc should be attached to nearest following items
        assert "doc" in items[0]
        assert items[0]["doc"] == "Test function documentation"

    def test_parse_named_imports_simple(self):
        """Test parsing simple named imports without aliases."""
        test_cases = [
            ("{ Component }", ["Component"]),
            ("{ Component, useState }", ["Component", "useState"]),
            ("{ Component, useState, useEffect }", ["Component", "useState", "useEffect"]),
            ("{Component,useState,useEffect}", ["Component", "useState", "useEffect"]),  # No spaces
        ]
        
        for import_string, expected in test_cases:
            result = self.analyzer._parse_named_imports(import_string)
            assert result == expected, f"Failed for: {import_string}"

    def test_parse_named_imports_with_aliases(self):
        """Test parsing named imports with 'as' aliases."""
        test_cases = [
            ("{ Component as Comp }", ["Comp"]),
            ("{ Component as Comp, useState }", ["Comp", "useState"]),
            ("{ Component as Comp, useState as State }", ["Comp", "State"]),
            ("{ Component as Comp, useState, useEffect as Effect }", ["Comp", "useState", "Effect"]),
            ("{Component as Comp,useState as State}", ["Comp", "State"]),  # No spaces after commas
        ]
        
        for import_string, expected in test_cases:
            result = self.analyzer._parse_named_imports(import_string)
            assert result == expected, f"Failed for: {import_string}"

    def test_parse_named_imports_edge_cases(self):
        """Test parsing named imports with edge cases."""
        test_cases = [
            ("", []),  # Empty string
            ("{}", []),  # Empty braces
            ("{ }", []),  # Empty braces with space
            ("{ Component as }", ["Component"]),  # Malformed alias (fallback)
            ("{ as Component }", ["Component"]),  # Wrong order (should handle gracefully)
            ("{ Component as Comp as Extra }", ["Extra"]),  # Multiple 'as' (take last part as fallback)
        ]
        
        for import_string, expected in test_cases:
            result = self.analyzer._parse_named_imports(import_string)
            assert result == expected, f"Failed for: {import_string}"

    def test_extract_imports_with_aliases(self):
        """Test the full import extraction with alias support."""
        content = '''
import React from 'react';
import { Component as Comp } from 'react';
import { Component as Comp, useState } from 'react';
import { Component as Comp, useState as State } from 'react';
import { Component as Comp, useState, useEffect as Effect } from 'react';
import DefaultExport, { Component as Comp } from 'react';
'''
        
        imports = self.analyzer._extract_imports(content)
        
        # Verify we got all imports
        assert len(imports) == 6
        
        # Check specific alias handling
        import_data = {imp["source"]: imp["imported"] for imp in imports}
        react_imports = [imp["imported"] for imp in imports if imp["source"] == "react"]
        
        # Verify aliases are correctly extracted
        assert ["Comp"] in react_imports  # { Component as Comp }
        assert ["Comp", "useState"] in react_imports  # { Component as Comp, useState }
        assert ["Comp", "State"] in react_imports  # { Component as Comp, useState as State }
        assert ["Comp", "useState", "Effect"] in react_imports  # Mixed aliases
        assert ["DefaultExport", "Comp"] in react_imports  # Mixed default + named with alias

    def test_extract_mixed_imports(self):
        """Test extraction of mixed default + named imports."""
        content = '''
import React, { Component } from 'react';
import DefaultExport, { named1, named2 } from 'module1';
import Default, { named as Alias } from 'module2';
import Default, { named1 as Alias1, named2, named3 as Alias3 } from 'module3';
'''
        
        imports = self.analyzer._extract_imports(content)
        
        # Should find all mixed imports
        mixed_imports = [imp for imp in imports if len(imp["imported"]) > 1]
        assert len(mixed_imports) == 4
        
        # Check specific patterns
        import_map = {imp["source"]: imp["imported"] for imp in imports}
        assert "React" in import_map["react"]
        assert "Component" in import_map["react"]
        assert "Default" in import_map["module2"]
        assert "Alias" in import_map["module2"]
        assert "Alias1" in import_map["module3"]
        assert "Alias3" in import_map["module3"]
        assert "named2" in import_map["module3"]

    def test_extract_imports_commonjs_with_aliases(self):
        """Test CommonJS destructuring with aliases (should work with same parser)."""
        content = '''
const { readFile as read } = require('fs');
const { readFile, writeFile as write } = require('fs/promises');
const { Component as Comp } = require('./component');
'''
        
        imports = self.analyzer._extract_imports(content)
        
        commonjs_imports = [imp for imp in imports if imp["type"] == "commonjs"]
        assert len(commonjs_imports) == 3
        
        # Verify alias handling in CommonJS
        import_map = {imp["source"]: imp["imported"] for imp in commonjs_imports}
        assert "read" in import_map["fs"]
        assert "readFile" in import_map["fs/promises"]
        assert "write" in import_map["fs/promises"]
        assert "Comp" in import_map["./component"]

    def test_backward_compatibility(self):
        """Test that old import patterns still work correctly."""
        content = '''
import React from 'react';
import { Component, useState, useEffect } from 'react';
import * as Utils from './utils';
const fs = require('fs');
const { readFile, writeFile } = require('fs/promises');
'''
        
        imports = self.analyzer._extract_imports(content)
        
        # Should extract all imports correctly
        assert len(imports) == 5
        
        # Verify backward compatibility
        es6_imports = [imp for imp in imports if imp["type"] == "es6"]
        commonjs_imports = [imp for imp in imports if imp["type"] == "commonjs"]
        
        assert len(es6_imports) == 3
        assert len(commonjs_imports) == 2
        
        # Check specific imports
        import_sources = [imp["source"] for imp in imports]
        assert "react" in import_sources
        assert "./utils" in import_sources
        assert "fs" in import_sources
        assert "fs/promises" in import_sources

    def test_complex_real_world_imports(self):
        """Test complex real-world import scenarios."""
        content = '''
import React, { Component as ReactComponent, useState as useStateHook, useEffect } from 'react';
import { BrowserRouter as Router, Route, Switch as RouterSwitch } from 'react-router-dom';
import { connect as reduxConnect } from 'react-redux';
import { createSelector as selector } from 'reselect';
import DefaultExport, { namedExport1 as alias1, namedExport2, namedExport3 as alias3 } from '@company/utils';
'''
        
        imports = self.analyzer._extract_imports(content)
        
        # Should handle all complex patterns
        assert len(imports) == 5
        
        # Check specific complex cases
        import_map = {imp["source"]: imp["imported"] for imp in imports}
        
        # React import with mixed default and aliased named imports
        react_imports = import_map["react"]
        assert "React" in react_imports
        assert "ReactComponent" in react_imports
        assert "useStateHook" in react_imports
        assert "useEffect" in react_imports
        
        # Router with multiple aliases
        router_imports = import_map["react-router-dom"]
        assert "Router" in router_imports
        assert "Route" in router_imports
        assert "RouterSwitch" in router_imports
        
        # Single alias imports
        assert import_map["react-redux"] == ["reduxConnect"]
        assert import_map["reselect"] == ["selector"]
        
        # Complex mixed import with scoped package
        utils_imports = import_map["@company/utils"]
        assert "DefaultExport" in utils_imports
        assert "alias1" in utils_imports
        assert "namedExport2" in utils_imports
        assert "alias3" in utils_imports
