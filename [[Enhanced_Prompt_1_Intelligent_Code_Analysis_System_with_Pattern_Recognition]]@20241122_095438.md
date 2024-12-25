---
epoch: 1732290878328
modelKey: anthropic/claude-3.5-sonnet:beta|openrouterai
tags:
  - copilot-conversation
---

# [Enhanced_Prompt_1_Intelligent_Code_Analysis_System_with_Pattern_Recognition](Enhanced_Prompt_1_Intelligent_Code_Analysis_System_with_Pattern_Recognition.md)@20241122_095438

**user**: [Enhanced Prompt 1 Intelligent Code Analysis System with Pattern Recognition](./Enhanced%20Prompt%201%20Intelligent%20Code%20Analysis%20System%20with%20Pattern%20Recognition.md) [Extract AST](./Extract%20AST.md)  
[Timestamp: 2024/11/22 09:54:38]

**ai**: Based on the provided context, I'll help create a comprehensive implementation that combines the AST extraction capabilities with the hierarchical storage system. Here's a complete solution:

```python
import ast
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pymongo import MongoClient
from datetime import datetime

@dataclass
class CodeAnalysisConfig:
    """Configuration for code analysis"""
    max_complexity: int = 10
    min_docstring_length: int = 10
    track_imports: bool = True
    track_comments: bool = True
    track_types: bool = True

class CodeAnalyzer:
    def __init__(self, config: CodeAnalysisConfig):
        self.config = config
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['code_analysis']
        self.collection = self.db['files']

    def analyze_file(self, file_path: str, source_code: str) -> Dict[str, Any]:
        """Analyze a single file and return structured data"""
        try:
            tree = ast.parse(source_code)
            return {
                "file_name": file_path,
                "timestamp": datetime.utcnow(),
                "classes": self._analyze_classes(tree),
                "functions": self._analyze_functions(tree),
                "imports": self._analyze_imports(tree) if self.config.track_imports else [],
                "comments": self._extract_comments(source_code) if self.config.track_comments else [],
                "metrics": self._calculate_metrics(tree)
            }
        except SyntaxError as e:
            return {
                "file_name": file_path,
                "timestamp": datetime.utcnow(),
                "error": f"Syntax error: {str(e)}"
            }

    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class information from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "methods": self._analyze_methods(node),
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                    "docstring": ast.get_docstring(node),
                    "complexity": self._calculate_complexity(node)
                }
                if self.config.track_types:
                    class_info["type_annotations"] = self._extract_type_annotations(node)
                classes.append(class_info)
        return classes

    def _analyze_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract method information from a class"""
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "line_number": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "complexity": self._calculate_complexity(node)
                }
                if self.config.track_types:
                    method_info["type_annotations"] = self._extract_type_annotations(node)
                methods.append(method_info)
        return methods

    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function information from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node):
                function_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "line_number": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "complexity": self._calculate_complexity(node)
                }
                if self.config.track_types:
                    function_info["type_annotations"] = self._extract_type_annotations(node)
                functions.append(function_info)
        return functions

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
        return complexity

    def _analyze_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import information"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "alias": alias.asname})
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname
                    })
        return imports

    def _extract_comments(self, source_code: str) -> List[str]:
        """Extract comments from source code"""
        comments = []
        for line in source_code.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                comments.append(line[1:].strip())
        return comments

    def _extract_type_annotations(self, node: ast.AST) -> Dict[str, str]:
        """Extract type annotations from a node"""
        annotations = {}
        if isinstance(node, ast.FunctionDef):
            if node.returns:
                annotations["return"] = self._get_type_string(node.returns)
            for arg in node.args.args:
                if arg.annotation:
                    annotations[arg.arg] = self._get_type_string(arg.annotation)
        return annotations

    def _get_type_string(self, node: ast.AST) -> str:
        """Convert type annotation AST node to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{self._get_type_string(node.value)}[{self._get_type_string(node.slice)}]"
        return "unknown"

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function definition is a method"""
        return isinstance(node.parent, ast.ClassDef)

    def _calculate_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate various metrics for the code"""
        return {
            "total_lines": len(ast.dump(tree).split('\n')),
            "total_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            "total_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "total_imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        }

    def store_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Store analysis results in MongoDB"""
        result = self.collection.insert_one(analysis_data)
        return str(result.inserted_id)

    def get_analysis(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from MongoDB"""
        return self.collection.find_one({"file_name": file_name})

# Example usage
if __name__ == "__main__":
    config = CodeAnalysisConfig(
        max_complexity=10,
        min_docstring_length=10,
        track_imports=True,
        track_comments=True,
        track_types=True
    )
    
    analyzer = CodeAnalyzer(config)
    
    # Example source code
    source_code = """
    class ExampleClass:
        def example_method(self, x: int) -> int:
            '''Example method docstring'''
            return x * 2
    
    def standalone_function(y: str) -> str:
        # This is a comment
        return y.upper()
    """
    
    # Analyze and store
    analysis = analyzer.analyze_file("example.py", source_code)
    doc_id = analyzer.store_analysis(analysis)
    print(f"Analysis stored with ID: {doc_id}")
    
    # Retrieve analysis
    retrieved = analyzer.get_analysis("example.py")
    print(json.dumps(retrieved, indent=2, default=str))
```

This implementation:

1. Uses dataclasses for configuration management
2. Implements hierarchical AST analysis
3. Extracts comprehensive code information
4. Stores results in MongoDB
5. Supports type annotations
6. Tracks imports and comments
7. Calculates code metrics
8. Provides easy retrieval of stored analyses

The code is modular, well-documented, and follows Python best practices. It can be extended further based on specific needs, such as adding more metrics or analysis features.  
[Timestamp: 2024/11/22 09:55:15]
