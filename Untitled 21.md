### Prompt 1: Deep Code Analysis System

You are an expert Python developer creating a system that analyzes code to extract rich semantic information for AI documentation generation. Your task is to implement AST-based analysis with the ast module to capture both explicit and implicit code understanding.

For each code element, your analyzer should extract:

1. **Core Structure**
- Complete function/class signatures 
- Parameter details including types, defaults
- Return type information
- Control flow and complexity metrics

2. **Behavioral Understanding**
- What exceptions can be raised and why
- Side effects and resource usage 
- Prerequisites and postconditions
- Performance characteristics and bottlenecks

3. **Usage Context** 
- How the code is meant to be used
- Common usage patterns and anti-patterns
- Related functions/classes
- Example usage scenarios

4. **Developer Intent**
- Purpose and motivation
- Design decisions and tradeoffs
- Implicit assumptions
- Edge cases and error conditions

Create a CodeAnalyzer class that:
1. Parses Python code into an AST
2. Extracts comprehensive metadata
3. Builds relationships between elements
4. Identifies patterns and conventions
5. Captures semantic understanding

The goal is to enable generating documentation that completely explains how to use the code effectively.

How would you implement this deep code analysis capability? Focus on extracting both explicit syntax and implicit meaning.

To implement a deep code analysis capability that extracts both explicit syntax and implicit meaning from Python code, you'll need to build a sophisticated `CodeAnalyzer` class that leverages Python's Abstract Syntax Tree (AST) module. The goal is to parse the code, traverse the AST, and extract rich semantic information that can be used to generate comprehensive documentation.

Below is a step-by-step guide on how you might implement this:

## 1. Parsing Python Code into an AST

The first step is to parse the source code into an abstract syntax tree using Python's built-in `ast` module.

```python
import ast

class CodeAnalyzer:
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)
```

By parsing the code, you get a tree representation of the source code, which you can traverse to extract information.

## 2. Extracting Comprehensive Metadata

### Function and Class Definitions

Traverse the AST to find all function and class definitions.

```python
    def extract_definitions(self):
        self.functions = []
        self.classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.functions.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes.append(node)
```

### Complete Signatures

Extract function and class signatures, including parameter details, defaults, and return types.

```python
    def get_function_signature(self, func_node):
        func_name = func_node.name
        args = func_node.args
        params = []
        for arg in args.args:
            param = {'name': arg.arg}
            # Handle type annotations (Python 3.5+)
            if arg.annotation:
                param['type'] = ast.get_source_segment(self.code, arg.annotation)
            params.append(param)
        # Handle default values
        defaults = [ast.literal_eval(d) for d in args.defaults]
        for i, default in enumerate(defaults):
            params[len(params) - len(defaults) + i]['default'] = default
        # Return type
        return_type = None
        if func_node.returns:
            return_type = ast.get_source_segment(self.code, func_node.returns)
        signature = {
            'name': func_name,
            'parameters': params,
            'return_type': return_type,
        }
        return signature
```

### Control Flow and Complexity Metrics

Analyze the control flow to compute complexity metrics like Cyclomatic Complexity.

```python
    def compute_cyclomatic_complexity(self, func_node):
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                 ast.And, ast.Or, ast.ExceptHandler)):
                complexity += 1
        return complexity
```

## 3. Behavioral Understanding

### Exceptions Raised

Identify all exceptions that can be raised within functions.

```python
    def extract_exceptions(self, func_node):
        exceptions = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Raise):
                if node.exc:
                    exception_type = ast.get_source_segment(self.code, node.exc)
                    exceptions.append(exception_type)
        return exceptions
```

### Side Effects and Resource Usage

Detect side effects such as modifying global variables, I/O operations, or database access.

```python
    def detect_side_effects(self, func_node):
        side_effects = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Global):
                side_effects.append('Modifies global variables')
            elif isinstance(node, (ast.Print, ast.Call)):
                # You can expand this to detect specific function calls like file operations
                side_effects.append('Performs I/O operations')
        return side_effects
```

### Prerequisites and Postconditions

Analyze for prerequisites (e.g., required state before function execution) and postconditions (state after execution).

This often involves deeper analysis and may require natural language processing of comments or docstrings.

```python
    def extract_prerequisites_postconditions(self, func_node):
        prerequisites = []
        postconditions = []
        if ast.get_docstring(func_node):
            docstring = ast.get_docstring(func_node)
            # Use NLP to parse docstring for keywords like "Requires", "Ensures"
            # This is a placeholder for actual NLP implementation
            if 'Requires' in docstring:
                prerequisites.append('...')  # Extract the actual prerequisite
            if 'Ensures' in docstring:
                postconditions.append('...')  # Extract the actual postcondition
        return prerequisites, postconditions
```

### Performance Characteristics and Bottlenecks

Estimate performance by analyzing loops, recursion, and algorithm complexity.

```python
    def estimate_performance(self, func_node):
        loops = 0
        recursive = False
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                loops += 1
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == func_node.name:
                recursive = True
        performance = {
            'loops': loops,
            'recursion': recursive,
        }
        return performance
```

## 4. Usage Context

### How the Code is Meant to be Used

Extract usage examples from docstrings or comments.

```python
    def extract_usage_examples(self, func_node):
        usage_examples = []
        if ast.get_docstring(func_node):
            docstring = ast.get_docstring(func_node)
            # Use regular expressions to find code examples in docstrings
            examples = re.findall(r'```python(.*?)```', docstring, re.DOTALL)
            usage_examples.extend(examples)
        return usage_examples
```

### Common Usage Patterns and Anti-patterns

Identify patterns like Singleton, Factory, etc., and anti-patterns like God Object or Spaghetti Code.

This may involve pattern recognition techniques.

```python
    def identify_patterns(self):
        # Placeholder for pattern recognition logic
        patterns = []
        # Analyze class inheritance, method structure, etc.
        return patterns
```

### Related Functions/Classes

Build relationships between different code elements.

```python
    def build_relationships(self):
        relationships = {}
        # For each function, find calls to other functions/classes
        for func in self.functions:
            called_functions = []
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_functions.append(node.func.id)
            relationships[func.name] = called_functions
        return relationships
```

## 5. Developer Intent

### Purpose and Motivation

Attempt to infer the purpose by analyzing docstrings, comments, and naming conventions.

```python
    def infer_purpose(self, func_node):
        purpose = ''
        if ast.get_docstring(func_node):
            docstring = ast.get_docstring(func_node)
            # First line often contains the summary
            purpose = docstring.strip().split('\n')[0]
        else:
            # Fallback to function name if no docstring
            purpose = func_node.name.replace('_', ' ').capitalize()
        return purpose
```

### Design Decisions and Trade-offs

Look for comments or docstrings that explain why certain approaches were taken.

```python
    def extract_design_decisions(self, func_node):
        design_notes = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                # This is a string expression, could be a comment or note
                comment = node.value.s
                if 'Note:' in comment or 'TODO:' in comment:
                    design_notes.append(comment)
        return design_notes
```

### Implicit Assumptions and Edge Cases

Identify any assumptions made in the code.

```python
    def identify_assumptions(self, func_node):
        assumptions = []
        # Analyze for type checks, assertions, or lack thereof
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                assumptions.append(ast.get_source_segment(self.code, node))
        return assumptions
```

## Integrating All Components

Putting it all together within the `CodeAnalyzer` class.

```python
import ast
import re

class CodeAnalyzer:
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)
        self.functions = []
        self.classes = []
        self.extract_definitions()
        self.analysis_results = []

    def extract_definitions(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.functions.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes.append(node)

    def analyze(self):
        for func in self.functions:
            analysis = {}
            signature = self.get_function_signature(func)
            complexity = self.compute_cyclomatic_complexity(func)
            exceptions = self.extract_exceptions(func)
            side_effects = self.detect_side_effects(func)
            prerequisites, postconditions = self.extract_prerequisites_postconditions(func)
            performance = self.estimate_performance(func)
            usage_examples = self.extract_usage_examples(func)
            purpose = self.infer_purpose(func)
            design_notes = self.extract_design_decisions(func)
            assumptions = self.identify_assumptions(func)
            analysis.update({
                'signature': signature,
                'complexity': complexity,
                'exceptions': exceptions,
                'side_effects': side_effects,
                'prerequisites': prerequisites,
                'postconditions': postconditions,
                'performance': performance,
                'usage_examples': usage_examples,
                'purpose': purpose,
                'design_notes': design_notes,
                'assumptions': assumptions,
            })
            self.analysis_results.append(analysis)

    # Include the methods defined above (get_function_signature, compute_cyclomatic_complexity, etc.)
```

## Capturing Semantic Understanding

To capture deeper semantic understanding beyond syntax, you may need to incorporate additional techniques:

- **Natural Language Processing (NLP):** Use NLP to process docstrings and comments to extract meaningful information.
  
  ```python
  import nltk
  from nltk.tokenize import sent_tokenize

  def process_docstring(self, docstring):
      sentences = sent_tokenize(docstring)
      # Further NLP processing to extract key information
  ```

- **Static Analysis Tools:** Integrate with tools like `pylint`, `mypy`, or `bandit` to get more insights.

  ```python
  def integrate_static_analysis(self):
      # Run external tools and parse their outputs
      pass
  ```

- **Pattern Recognition:** Implement algorithms to detect design patterns or code smells.

  ```python
  def detect_design_patterns(self):
      # Implement logic to detect Singleton, Factory patterns, etc.
      pass
  ```

- **Data Flow Analysis:** Analyze how data moves through the code to identify dependencies and potential side effects.

  ```python
  def data_flow_analysis(self):
      # Use control flow graphs or other techniques
      pass
  ```

## Generating Documentation

With all the extracted information, you can format it into comprehensive documentation.

```python
    def generate_documentation(self):
        docs = []
        for analysis in self.analysis_results:
            doc = f"### Function `{analysis['signature']['name']}`\n"
            doc += f"**Purpose:** {analysis['purpose']}\n\n"
            doc += f"**Signature:** `{analysis['signature']}`\n\n"
            doc += f"**Parameters:**\n"
            for param in analysis['signature']['parameters']:
                doc += f"- `{param['name']}`"
                if 'type' in param:
                    doc += f": `{param['type']}`"
                if 'default' in param:
                    doc += f" (default `{param['default']}`)"
                doc += "\n"
            doc += f"\n**Returns:** `{analysis['signature']['return_type']}`\n\n"
            doc += f"**Complexity:** Cyclomatic Complexity = {analysis['complexity']}\n\n"
            if analysis['exceptions']:
                doc += f"**Exceptions Raised:** {', '.join(analysis['exceptions'])}\n\n"
            if analysis['side_effects']:
                doc += f"**Side Effects:** {', '.join(analysis['side_effects'])}\n\n"
            if analysis['prerequisites']:
                doc += f"**Prerequisites:** {', '.join(analysis['prerequisites'])}\n\n"
            if analysis['postconditions']:
                doc += f"**Postconditions:** {', '.join(analysis['postconditions'])}\n\n"
            if analysis['usage_examples']:
                doc += f"**Usage Examples:**\n"
                for example in analysis['usage_examples']:
                    doc += f"```python\n{example}\n```\n"
            if analysis['design_notes']:
                doc += f"**Design Notes:**\n{analysis['design_notes']}\n\n"
            if analysis['assumptions']:
                doc += f"**Assumptions:**\n{analysis['assumptions']}\n\n"
            docs.append(doc)
        return "\n".join(docs)
```

## Handling Implicit Meanings

Extracting implicit meanings such as developer intent, performance bottlenecks, or implicit assumptions often requires heuristics or machine learning techniques.

- **Heuristics:** Define rule-based systems that look for certain code patterns.

- **Machine Learning:** Use trained models to predict code characteristics based on labeled datasets.

  ```python
  def predict_implications(self, func_node):
      # Feature extraction from code
      features = self.extract_features(func_node)
      # Use a pre-trained ML model to predict
      prediction = self.ml_model.predict(features)
      return prediction
  ```

## Summary

Implementing deep code analysis involves:

- **AST Parsing:** Using the `ast` module to parse and traverse the code.

- **Feature Extraction:** Gathering explicit details like function signatures, parameters, and control flow.

- **Behavioral Analysis:** Identifying exceptions, side effects, and performance aspects.

- **Context Understanding:** Extracting usage patterns, related elements, and examples.

- **Semantic Interpretation:** Inferring developer intent, assumptions, and design decisions using docstrings, comments, and naming conventions.

- **Integration:** Combining these analyses to generate rich, informative documentation.

Remember that while AST can provide structural insights, capturing semantic meaning often requires going beyond syntax analysis, potentially incorporating NLP and machine learning.

## Tools and Libraries to Consider

- **`ast` Module:** Built-in module for parsing Python code.

- **`astor` Library:** Helps with reconstructing code from ASTs.

- **`inspect` Module:** For live objects, can provide additional information.

- **Natural Language Toolkit (NLTK):** For processing docstrings and comments.

- **Static Analysis Tools:** Like `pylint`, `mypy`, or `radon` for complexity analysis.

## Example Usage

```python
code = '''
def add(a, b=0):
    """
    Adds two numbers together.

    Parameters:
    - a (int): The first number.
    - b (int, optional): The second number. Defaults to 0.

    Returns:
    int: The result of adding a and b.

    Example:
    ```python
    result = add(5, 3)
    ```
    """
    return a + b
'''

analyzer = CodeAnalyzer(code)
analyzer.analyze()
documentation = analyzer.generate_documentation()
print(documentation)
```

This should output a detailed documentation for the `add` function, including signature, purpose, parameters, return type, and usage examples.

---

By carefully designing your `CodeAnalyzer` class and leveraging the power of AST and other analysis techniques, you can create a system that not only understands the explicit structure of the code but also captures the implicit meanings, leading to comprehensive and insightful documentation.