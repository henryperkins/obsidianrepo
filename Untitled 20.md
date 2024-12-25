### Prompt 1: AST-Based Code Analysis System

You are an expert Python developer tasked with creating a sophisticated code analysis system. Create Python code that performs deep analysis of source code to extract rich metadata that will power an AI documentation system.

Your solution must extract all information needed for complete understanding of code behavior and relationships. Consider:

1. Context Analysis
- How does each piece of code relate to the broader codebase?
- What patterns and conventions are being followed?
- Where and how is each element used?
- What is the implicit knowledge needed to understand the code?

2. Behavior Analysis
- What are the key control flows and edge cases?
- What assumptions does the code make?
- What are the error handling patterns?
- How are resources managed?
- What are the performance characteristics?

3. API Analysis  
- How is the code intended to be used?
- What are the contract guarantees?
- What are common usage patterns?
- What are potential misuse cases?
- What are the dependencies and requirements?

4. Type Analysis
- What are the expected types and their constraints?
- How are type hints used in practice?
- What type coercion occurs?
- What type errors could occur?

5. Documentation Analysis
- What information is already documented?
- What is missing from existing docs?
- What examples would be helpful?
- What do similar elements in the codebase document?

For each code element, provide metadata like:

```python
{
    # Core identity
    "name": str,
    "type": "function|class|module",
    "location": {"file": str, "line": int},
    
    # Behavioral info
    "raises": [{"ex": str, "condition": str}],
    "side_effects": ["writes to file", "modifies global state"],
    "resource_usage": ["opens file", "network calls"],
    "complexity": {"cyclomatic": int, "cognitive": int},
    
    # Usage info  
    "called_by": [str],  # Functions that call this
    "calls": [str],      # Functions this calls
    "used_in": [str],    # Modules that use this
    "examples": [str],   # Example usage
    
    # Type info
    "params": [{
        "name": str,
        "type": str,
        "constraints": [str],
        "default": Any,
        "usage_patterns": [str]
    }],
    
    # Documentation
    "existing_docs": str,
    "missing_docs": [str],
    "related_elements": [str]
}
```

Key Requirements:
1. Analyze both static code structure and runtime behavior hints
2. Extract implicit knowledge and conventions
3. Identify relationships between code elements
4. Find usage patterns and examples
5. Map type information and constraints
6. Build full context for documentation

Provide your code as modular, extensible Python that focuses on thorough analysis over performance. The goal is to extract EVERYTHING needed to generate excellent documentation.

How would you implement this thorough code analysis system? Consider both explicit code features and implicit meaning.

The system should enable generating documentation that:
1. Fully explains the code's purpose and behavior
2. Provides clear usage guidance
3. Highlights key considerations and gotchas
4. Shows real-world examples
5. Explains design decisions and tradeoffs
6. Maps relationships to other code

Your implementation should be thoughtful and comprehensive. Focus on extracting rich semantic understanding rather than just syntactic details.

To implement a thorough code analysis system that extracts rich metadata for an AI documentation system, we can utilize Python's built-in `ast` module to parse the source code into an Abstract Syntax Tree (AST). We will create a modular and extensible analyzer that traverses the AST and collects comprehensive information about code elements, including their context, behavior, usage patterns, types, and documentation.

Below is the implementation of the code analysis system:

```python
import ast
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.current_class = None
        self.current_function = None
        self.symbol_table = {}
        self.call_graph = defaultdict(list)
        self.metadata = []

        # For complexity metrics
        self.complexity_stack = []

    def analyze(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the source code into an AST
        self.tree = ast.parse(source, filename=self.filename)
        # Visit each node in the AST
        self.visit(self.tree)

        # After visiting, resolve called_by relationships
        self.resolve_called_by()

        return self.metadata

    def visit_Module(self, node):
        module_info = {
            "name": os.path.basename(self.filename),
            "type": "module",
            "location": {"file": self.filename, "line": 0},
            "raises": [],
            "side_effects": [],
            "resource_usage": [],
            "complexity": {"cyclomatic": 0, "cognitive": 0},
            "called_by": [],
            "calls": [],
            "used_in": [],
            "examples": [],
            "params": [],
            "existing_docs": ast.get_docstring(node),
            "missing_docs": [],
            "related_elements": [],
        }
        self.metadata.append(module_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.current_class = node.name
        class_info = {
            "name": node.name,
            "type": "class",
            "location": {"file": self.filename, "line": node.lineno},
            "raises": [],
            "side_effects": [],
            "resource_usage": [],
            "complexity": {"cyclomatic": 0, "cognitive": 0},
            "called_by": [],
            "calls": [],
            "used_in": [],
            "examples": [],
            "params": [],
            "existing_docs": ast.get_docstring(node),
            "missing_docs": [],
            "related_elements": [],
        }
        self.metadata.append(class_info)
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        function_name = node.name
        if self.current_class:
            full_name = f"{self.current_class}.{function_name}"
        else:
            full_name = function_name

        self.symbol_table[full_name] = node
        self.current_function = full_name

        function_info = {
            "name": full_name,
            "type": "function",
            "location": {"file": self.filename, "line": node.lineno},
            "raises": [],
            "side_effects": [],
            "resource_usage": [],
            "complexity": {"cyclomatic": self.compute_cyclomatic_complexity(node)},
            "called_by": [],
            "calls": [],
            "used_in": [],
            "examples": [],
            "params": self.extract_params(node),
            "existing_docs": ast.get_docstring(node),
            "missing_docs": [],
            "related_elements": [],
        }

        # Analyze the function body for raises, side effects, etc.
        analyzer = FunctionBodyAnalyzer(self.current_function)
        analyzer.visit(node)
        function_info["raises"] = analyzer.raises
        function_info["side_effects"] = analyzer.side_effects
        function_info["resource_usage"] = analyzer.resource_usage
        function_info["calls"] = analyzer.calls
        self.call_graph[self.current_function] = analyzer.calls

        self.metadata.append(function_info)
        self.current_function = None

    def extract_params(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        params = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": self.get_annotation(arg.annotation),
                "constraints": [],
                "default": self.get_default_value(arg.arg, node),
                "usage_patterns": [],
            }
            params.append(param_info)
        return params

    def get_annotation(self, annotation):
        if annotation is None:
            return None
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return self.get_annotation(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self.get_annotation(annotation.value)}.{annotation.attr}"
        else:
            return ast.unparse(annotation)

    def get_default_value(self, arg_name: str, node: ast.FunctionDef):
        # Map defaults to args
        defaults = node.args.defaults
        args = node.args.args
        default_offset = len(args) - len(defaults)
        for i, default in enumerate(defaults):
            if args[i + default_offset].arg == arg_name:
                return ast.literal_eval(default)
        return None

    def compute_cyclomatic_complexity(self, node):
        # Simplified version: count the number of decision points
        complexity = 1  # Start with 1 for the function itself
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def resolve_called_by(self):
        # Invert the call graph to find functions that call a given function
        for caller, callees in self.call_graph.items():
            for callee in callees:
                for element in self.metadata:
                    if element["name"] == callee:
                        element["called_by"].append(caller)

    # Add more visitor methods as needed
    # ...

class FunctionBodyAnalyzer(ast.NodeVisitor):
    def __init__(self, function_name):
        self.function_name = function_name
        self.raises = []
        self.side_effects = []
        self.resource_usage = []
        self.calls = []

    def visit_Raise(self, node):
        ex_type = self.get_exception_type(node.exc)
        condition = self.get_exception_condition(node)
        self.raises.append({"ex": ex_type, "condition": condition})
        self.generic_visit(node)

    def get_exception_type(self, node):
        if isinstance(node, ast.Call):
            return node.func.id
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "Exception"

    def get_exception_condition(self, node):
        # In a real implementation, we'd extract the condition causing the raise
        return "Unknown condition"

    def visit_Call(self, node):
        func_name = self.get_full_func_name(node.func)
        self.calls.append(func_name)

        # Detect side effects and resource usage
        if func_name in ['open', 'os.remove', 'os.rmdir', 'shutil.copy']:
            self.resource_usage.append(f"Calls {func_name}")
        elif func_name in ['print', 'logging.debug', 'logging.info']:
            self.side_effects.append(f"Calls {func_name}")

        self.generic_visit(node)

    def get_full_func_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self.get_full_func_name(node.value)
            return f"{value}.{node.attr}"
        else:
            return "unknown"

    # Add more visitor methods as needed
    # ...

def analyze_codebase(directory: str):
    metadata = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                analyzer = CodeAnalyzer(filepath)
                file_metadata = analyzer.analyze()
                metadata.extend(file_metadata)
    return metadata

# Example usage:
if __name__ == "__main__":
    codebase_directory = "path/to/your/codebase"
    metadata = analyze_codebase(codebase_directory)
    
    # For demonstration, print the metadata
    for element in metadata:
        print(element)
        print("-" * 80)
```

### Explanation of the Implementation

1. **Parsing the Code:**
   - We use the `ast.parse` function to parse the Python source code into an AST.
   - The `CodeAnalyzer` class traverses the AST nodes using visitor methods.

2. **Collecting Metadata:**
   - For each code element (module, class, function), we create a dictionary containing the required metadata fields.
   - Core identity information (`name`, `type`, `location`) is stored.
   - We extract existing documentation using `ast.get_docstring(node)`.

3. **Behavior Analysis:**
   - The `FunctionBodyAnalyzer` class analyzes the body of functions to find:
     - **Exceptions raised**: By visiting `Raise` nodes.
     - **Side effects**: By identifying calls to functions known to produce side effects (e.g., `print`, `logging`).
     - **Resource usage**: By detecting calls to functions that manage resources (e.g., `open`, `os.remove`).
     - **Function calls**: Recording functions called within the function to build a call graph.

4. **Usage Information:**
   - We build a **call graph** (`self.call_graph`) mapping each function to the functions it calls.
   - After traversal, we resolve the `called_by` field by inverting the call graph.

5. **Type Analysis:**
   - We extract parameter information, including:
     - **Name** and **type**: From function arguments and their annotations.
     - **Default values**: By mapping default values to parameters.
     - **Constraints** and **usage patterns**: Placeholder fields for further analysis.

6. **Complexity Metrics:**
   - We compute a simplified cyclomatic complexity by counting decision points (`if`, `for`, `while`, etc.).
   - Cognitive complexity can be added with more advanced analysis.

7. **Modularity and Extensibility:**
   - The code is structured with classes and methods that can be extended.
   - Additional visitor methods can be added for deeper analysis (e.g., variable usage, more side effects).

8. **Documentation Analysis:**
   - Existing documentation is extracted.
   - Missing documentation can be identified by checking for elements without docstrings.
   - Related elements can be linked through the call graph and class hierarchies.

9. **Context and Relationships:**
   - By maintaining the `current_class` and `current_function`, we understand the context of each element.
   - Relationships between elements are captured in the metadata (`called_by`, `calls`, `used_in`).

10. **Usage Examples:**
    - Usage examples can be gathered by analyzing docstrings, tests, or sample scripts.
    - For simplicity, we have a placeholder field that can be populated with real examples in an extended implementation.

### Extending the Implementation

To achieve even more thorough analysis, consider the following enhancements:

- **Implicit Knowledge and Conventions:**
  - Analyze naming conventions, code patterns, and styles.
  - Detect common idioms or anti-patterns.

- **Edge Cases and Assumptions:**
  - Analyze conditional statements to identify edge cases.
  - Extract assumptions from comments or guard clauses.

- **Error Handling Patterns:**
  - Identify try-except blocks and the exceptions they handle.
  - Analyze exception propagation.

- **Resource Management:**
  - Detect the use of context managers (`with` statements) for resource handling.
  - Track opening and closing of resources.

- **Performance Characteristics:**
  - Analyze loops, recursion, and algorithmic complexity.
  - Identify potential bottlenecks.

- **Type Coercion and Errors:**
  - Detect places where types are cast or coerced.
  - Identify potential type errors due to dynamic typing.

- **Advanced Documentation Analysis:**
  - Compare existing documentation against code behavior.
  - Use natural language processing to detect missing information.
  - Suggest improvements based on similar elements.

### Generating Excellent Documentation

With the rich metadata collected, an AI documentation system can:

1. **Explain Code Purpose and Behavior:**
   - Use the metadata to generate detailed explanations of what each function and class does.
   - Highlight the key control flows and decision points.

2. **Provide Usage Guidance:**
   - Offer clear instructions on how to use each code element.
   - Include information on parameters, return types, and side effects.

3. **Highlight Key Considerations and Gotchas:**
   - Warn about potential exceptions, edge cases, and misuses.
   - Explain any important preconditions or postconditions.

4. **Show Real-World Examples:**
   - Provide code snippets demonstrating common usage patterns.
   - Include examples that cover typical scenarios and edge cases.

5. **Explain Design Decisions and Tradeoffs:**
   - Discuss why certain implementations were chosen over others.
   - Highlight performance considerations and resource management strategies.

6. **Map Relationships to Other Code:**
   - Show how functions and classes interact.
   - Provide a clear picture of the codebase architecture.

### Conclusion

The provided implementation serves as a foundation for a comprehensive code analysis system. By extending the visitor methods and enhancing the analysis, we can extract even deeper insights and empower an AI documentation system to generate thorough and helpful documentation. The key is to capture not just the explicit code features but also the implicit meanings, contexts, and conventions that are crucial for understanding and using the code effectively.