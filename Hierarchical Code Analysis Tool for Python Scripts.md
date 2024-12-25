```python
import ast
import os
import re
from datetime import datetime
from collections import defaultdict
import statistics
import json
import jsonschema
import subprocess
import networkx as nx

# Load the JSON schema
def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as schema_file:
        return json.load(schema_file)

# Validate the analysis output against the schema
def validate_output(analysis_result, schema):
    try:
        jsonschema.validate(instance=analysis_result, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e}")
        return False

# Parse docstrings into summary and details
def parse_docstring(docstring):
    if not docstring:
        return {"summary": "", "details": None}

    # Split into summary and details
    parts = docstring.strip().split('\n\n', 1)
    summary = parts[0].strip()
    details = parts[1].strip() if len(parts) > 1 else None

    return {
        "summary": summary,
        "details": details
    }

# Validate function and class names
def validate_name(name, type_="function"):
    if type_ == "function":
        pattern = r'^[a-z_][a-z0-9_]*$'
    else:  # class
        pattern = r'^[A-Z][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))

# Validate decorators
def validate_decorator(decorator):
    pattern = r'^@?[A-Za-z_][A-Za-z0-9_]*$'
    return bool(re.match(pattern, decorator))

# Extract decorators from a decorator list
def extract_decorators(decorator_list):
    decorators = []
    for decorator in decorator_list:
        if isinstance(decorator, ast.Name) and validate_decorator(decorator.id):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            try:
                decorators.append(ast.unparse(decorator))
            except:
                pass  # Skip if unable to unparse
    return decorators

# Extract type annotations for functions
def extract_type_annotations(node):
    try:
        annotations = {}
        # Arguments
        for arg in node.args.args:
            if arg.annotation:
                try:
                    annotations[arg.arg] = ast.unparse(arg.annotation)
                except:
                    annotations[arg.arg] = None
            else:
                annotations[arg.arg] = None

        # Return type
        if node.returns:
            try:
                annotations['return'] = ast.unparse(node.returns)
            except:
                annotations['return'] = None
        else:
            annotations['return'] = None

        return annotations
    except:
        return {"return": None}

# Calculate complexity with dynamic thresholds
def calculate_complexity(node):
    complexity = 1  # Start with 1 for the function itself

    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFunctionDef, ast.FunctionDef)):
            complexity += 1
        elif isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            complexity += len(n.values) - 1
        elif isinstance(n, ast.ExceptHandler):
            complexity += 1
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1
        elif isinstance(n, ast.Lambda):
            complexity += 1

    # Dynamic threshold logic
    length_threshold = 20  # Base threshold for function length
    complexity_threshold = 10  # Base threshold for complexity

    # Adjust thresholds based on function size
    if node.body:
        function_length = node.body[-1].lineno - node.lineno + 1
        if function_length > length_threshold:
            complexity_threshold += (function_length - length_threshold) // 10

    is_complex = complexity > complexity_threshold

    return complexity, is_complex

# Analyze function for improvements based on complexity
def analyze_function_for_improvements(func):
    explanation = None
    improvements = None
    complexity, is_complex = calculate_complexity(func['node'])
    if is_complex:
        explanation = "High complexity due to multiple control structures or constructs."
        improvements = "Consider refactoring to simplify logic or break down the function."
    return explanation, improvements

# Extract arguments from function arguments
def extract_arguments(args):
    return [arg.arg for arg in args if validate_argument(arg.arg)]

# Validate function arguments
def validate_argument(arg):
    pattern = r'^[a-z_][A-Za-z0-9_]*$'
    return bool(re.match(pattern, arg))

# Extract functions from AST
def extract_functions(tree):
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not validate_name(node.name, type_="function"):
                continue
            explanation, improvements = analyze_function_for_improvements({
                'node': node
            })
            functions.append({
                "name": node.name,
                "args": extract_arguments(node.args.args),
                "docstring": parse_docstring(ast.get_docstring(node)),
                "line_number": node.lineno,
                "complexity": calculate_complexity(node)[0],
                "decorators": extract_decorators(node.decorator_list),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "explanation": explanation,
                "improvements": improvements,
                "nested_functions": extract_nested_functions(node)
            })
    return functions

# Extract nested functions within a function
def extract_nested_functions(function_node):
    nested_functions = []
    for node in ast.walk(function_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node != function_node:
            if not validate_name(node.name, type_="function"):
                continue
            explanation, improvements = analyze_function_for_improvements({
                'node': node
            })
            nested_functions.append({
                "name": node.name,
                "args": extract_arguments(node.args.args),
                "docstring": parse_docstring(ast.get_docstring(node)),
                "line_number": node.lineno,
                "complexity": calculate_complexity(node)[0],
                "decorators": extract_decorators(node.decorator_list),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "explanation": explanation,
                "improvements": improvements
            })
    return nested_functions

# Extract classes from AST
def extract_classes(tree):
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not validate_name(node.name, type_="class"):
                continue
            classes.append({
                "name": node.name,
                "line_number": node.lineno,
                "decorators": extract_decorators(node.decorator_list),
                "docstring": parse_docstring(ast.get_docstring(node)),
                "methods": extract_methods(node)
            })
    return classes

# Extract methods from a class
def extract_methods(class_node):
    methods = []
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not validate_name(node.name, type_="function"):
                continue
            explanation, improvements = analyze_function_for_improvements({
                'node': node
            })
            methods.append({
                "name": node.name,
                "args": extract_arguments(node.args.args),
                "docstring": parse_docstring(ast.get_docstring(node)),
                "line_number": node.lineno,
                "complexity": calculate_complexity(node)[0],
                "decorators": extract_decorators(node.decorator_list),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "explanation": explanation,
                "improvements": improvements
            })
    return methods

# Extract imports from AST
def extract_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "alias": alias.asname,
                    "line_number": node.lineno,
                    "import_type": "import"
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                full_module = f"{node.module}.{alias.name}" if node.module else alias.name
                imports.append({
                    "module": full_module,
                    "alias": alias.asname,
                    "line_number": node.lineno,
                    "import_type": "from_import"
                })
    # Remove duplicates based on unique items
    unique_imports = {json.dumps(import_item, sort_keys=True): import_item for import_item in imports}
    return list(unique_imports.values())

# Extract comments from source code
def extract_comments(source_code):
    comments = []
    for lineno, line in enumerate(source_code.splitlines(), start=1):
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            comments.append({
                "content": stripped_line,
                "line_number": lineno
            })
    return comments

# Extract docstrings from AST
def extract_docstrings(tree):
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)
    return docstrings

# Extract comprehensions from AST
def extract_comprehensions(tree):
    comprehensions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            comp_type = "list"
        elif isinstance(node, ast.SetComp):
            comp_type = "set"
        elif isinstance(node, ast.DictComp):
            comp_type = "dict"
        elif isinstance(node, ast.GeneratorExp):
            comp_type = "generator"
        else:
            continue

        complexity = calculate_complexity(node)[0]

        comprehensions.append({
            "type": comp_type,
            "line_number": node.lineno,
            "complexity": complexity
        })
    return comprehensions

# Extract context managers from AST
def extract_context_managers(tree):
    context_managers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                try:
                    context_expr = ast.unparse(item.context_expr)
                except:
                    context_expr = None
                try:
                    optional_vars = ast.unparse(item.optional_vars) if item.optional_vars else None
                except:
                    optional_vars = None
                if context_expr:
                    context_managers.append({
                        "context_expr": context_expr,
                        "optional_vars": optional_vars,
                        "line_number": node.lineno
                    })
    return context_managers

# Extract try-except blocks from AST
def extract_try_blocks(tree):
    try_blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            handlers = []
            for handler in node.handlers:
                handler_type = ast.unparse(handler.type) if handler.type else None
                handler_name = handler.name
                body = [{"type": type(stmt).__name__, "line_number": stmt.lineno} for stmt in handler.body]
                handlers.append({
                    "type": handler_type,
                    "name": handler_name,
                    "body": body
                })
            else_block = [{"type": type(stmt).__name__, "line_number": stmt.lineno} for stmt in node.orelse]
            finally_block = [{"type": type(stmt).__name__, "line_number": stmt.lineno} for stmt in node.finalbody]
            try_blocks.append({
                "line_number": node.lineno,
                "handlers": handlers,
                "else": else_block,
                "finally": finally_block
            })
    return try_blocks

# Extract variable usage from AST
def extract_variable_usage(tree):
    variables = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables[target.id].append(node.lineno)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            variables[node.id].append(node.lineno)
    return dict(variables)

# Detect patterns in AST
def detect_patterns(tree):
    patterns = []
    line_threshold = 20  # Example threshold for long methods
    nesting_threshold = 3  # Example threshold for deep nesting

    for node in ast.walk(tree):
        # Long Methods
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body:
                function_length = node.body[-1].lineno - node.lineno + 1
                if function_length > line_threshold:
                    patterns.append({
                        'pattern': 'Long method',
                        'function_name': node.name,
                        'line_number': node.lineno,
                        'argument_count': len(node.args.args)
                        # 'length': function_length  # Optional: Add if needed
                    })

        # Too Many Arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.args.args) > 5:
                patterns.append({
                    'pattern': 'Too many arguments',
                    'function_name': node.name,
                    'line_number': node.lineno,
                    'argument_count': len(node.args.args)
                })

        # Deeply Nested If Statements
        if isinstance(node, ast.If):
            depth = calculate_nesting_depth(node)
            if depth > nesting_threshold:
                patterns.append({
                    'pattern': 'Deeply nested if statement',
                    'line_number': node.lineno,
                    'nesting_depth': depth
                })

        # Magic Numbers
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in (0, 1):  # Allow common values like 0 and 1
                patterns.append({
                    'pattern': 'Magic number',
                    'line_number': node.lineno,
                    'value': node.value
                })

        # Complex Conditionals
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            if len(node.values) > 2:  # More than two conditions combined
                patterns.append({
                    'pattern': 'Complex conditional',
                    'line_number': node.lineno,
                    'condition_count': len(node.values)
                })

    # Unused Variables
    variable_usage = extract_variable_usage(tree)
    for var, lines in variable_usage.items():
        if len(lines) == 1:  # Declared but not used
            patterns.append({
                'pattern': 'Unused variable',
                'variable_name': var,
                'line_number': lines[0]
            })

    # Code Smell Detection
    patterns.extend(detect_code_smells(tree))

    return patterns

# Calculate nesting depth for a node
def calculate_nesting_depth(node, current_depth=0):
    if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
        current_depth += 1
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        max_depth = max(max_depth, calculate_nesting_depth(child, current_depth))
    return max_depth

# Code Smell Detection
def detect_code_smells(tree):
    code_smells = []
    # Example Code Smells: Large Classes, Duplicated Code, Long Parameter Lists

    # Large Classes
    class_threshold = 10  # Example threshold for number of methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            if method_count > class_threshold:
                code_smells.append({
                    'pattern': 'Large class',
                    'class_name': node.name,
                    'line_number': node.lineno,
                    'method_count': method_count
                })

    # Long Parameter Lists
    param_threshold = 5
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.args.args) > param_threshold:
                code_smells.append({
                    'pattern': 'Long parameter list',
                    'function_name': node.name,
                    'line_number': node.lineno,
                    'parameter_count': len(node.args.args)
                })

    # Duplicated Code Detection Placeholder
    # Implementing actual code clone detection would require more complex logic or external tools
    # Here, we can flag functions with identical bodies as potential duplicates
    function_bodies = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = ast.dump(ast.fix_missing_locations(ast.Module(body=node.body)))
            function_bodies[body].append(node.name)

    for body, functions in function_bodies.items():
        if len(functions) > 1:
            code_smells.append({
                'pattern': 'Duplicated code',
                'function_names': functions,
                'line_number': 'Multiple lines'
            })

    return code_smells

# Analyze relationships in AST
def analyze_relationships(tree):
    relationships = {
        'function_calls': [],
        'class_inheritance': [],
        'module_dependencies': []
    }
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    relationships['function_calls'].append({
                        'function_name': node.func.id,
                        'line_number': node.lineno
                    })
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        relationships['function_calls'].append({
                            'function_name': f"{node.func.value.id}.{node.func.attr}",
                            'line_number': node.lineno
                        })
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        relationships['class_inheritance'].append({
                            'class_name': node.name,
                            'base_class': base.id,
                            'line_number': node.lineno
                        })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        relationships['module_dependencies'].append(alias.name)
                else:
                    for alias in node.names:
                        full_module = f"{node.module}.{alias.name}" if node.module else alias.name
                        relationships['module_dependencies'].append(full_module)
    except:
        pass
    return relationships

# Extract metadata from source code and AST
def extract_metadata(source_code, tree):
    lines_of_code = len(source_code.splitlines())
    comments = sum(1 for line in source_code.splitlines() if line.strip().startswith('#'))
    docstrings = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and ast.get_docstring(node))

    metadata = {
        'lines_of_code': lines_of_code,
        'comments': comments,
        'docstrings': docstrings,
        'functions': len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]),
        'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    }
    return metadata

# Perform statistical analysis on functions
def perform_statistical_analysis(tree):
    function_details = extract_functions(tree)
    if not function_details:
        return {
            'average_lines_per_function': 0,
            'average_arguments_per_function': 0,
            'complexity_distribution': {
                'mean': 0,
                'median': 0,
                'stdev': 0
            }
        }

    # Calculate lines per function
    function_lines = []
    for func in function_details:
        # Assuming body is parsed, but since we have line_number, estimate lines
        # Here, we'll need actual line counts which are not directly available
        # So, this is a placeholder assuming function length as complexity
        function_lines.append(func['complexity'])  # Using complexity as a proxy

    # Arguments per function
    function_args = [len(f['args']) for f in function_details]

    # Complexity scores
    complexity_scores = [f['complexity'] for f in function_details]

    stats = {
        'average_lines_per_function': statistics.mean(function_lines) if function_lines else 0,
        'average_arguments_per_function': statistics.mean(function_args) if function_args else 0,
        'complexity_distribution': {
            'mean': statistics.mean(complexity_scores) if complexity_scores else 0,
            'median': statistics.median(complexity_scores) if complexity_scores else 0,
            'stdev': statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0
        }
    }
    return stats

# Analyze lambdas and comprehensions
def analyze_lambdas_and_comprehensions(tree):
    lambdas = []
    comprehensions = defaultdict(int)
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambdas.append({
                'line_number': node.lineno,
                'arguments': [arg.arg for arg in node.args.args]
            })
        elif isinstance(node, ast.ListComp):
            comprehensions['list_comprehensions'] += 1
        elif isinstance(node, ast.DictComp):
            comprehensions['dict_comprehensions'] += 1
        elif isinstance(node, ast.SetComp):
            comprehensions['set_comprehensions'] += 1
        elif isinstance(node, ast.GeneratorExp):
            comprehensions['generator_expressions'] += 1
    return {
        'lambda_functions': lambdas,
        'comprehensions': dict(comprehensions)
    }

# Analyze type annotations
def analyze_type_annotations(tree):
    type_annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = extract_type_annotations(node)
            type_annotations.append({
                'function_name': node.name,
                'arguments': {k: v for k, v in annotations.items() if k != 'return'},
                'return_annotation': annotations.get('return'),
                'line_number': node.lineno
            })
    return type_annotations

# Analyze data flow and dependencies
def analyze_data_flow_and_dependencies(tree):
    data_flow = defaultdict(list)
    dependencies = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    data_flow[target.id].append(node.lineno)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                dependencies[node.func.id].append(node.lineno)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    dependencies[f"{node.func.value.id}.{node.func.attr}"].append(node.lineno)
    return {
        'data_flow': dict(data_flow),
        'dependencies': dict(dependencies)
    }

# Analyze context managers and exceptions
def analyze_context_managers_and_exceptions(tree):
    context_managers = []
    exception_handling = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            try:
                context_exprs = [ast.unparse(item.context_expr) for item in node.items]
            except:
                context_exprs = [None for _ in node.items]
            context_managers.append({
                'line_number': node.lineno,
                'context_exprs': context_exprs
            })
        elif isinstance(node, ast.Try):
            handlers = [{'type': ast.unparse(handler.type) if handler.type else 'Any', 'line_number': handler.lineno}
                        for handler in node.handlers]
            exception_handling.append({
                'line_number': node.lineno,
                'handlers': handlers
            })
    return {
        'context_managers': context_managers,
        'exception_handling': exception_handling
    }

# Analyze async and await constructs
def analyze_async_await_constructs(tree):
    async_functions = []
    await_expressions = []
    async_for_statements = []
    async_with_statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            async_functions.append({
                'name': node.name,
                'line_number': node.lineno
            })
        elif isinstance(node, ast.Await):
            await_expressions.append({
                'line_number': node.lineno
            })
        elif isinstance(node, ast.AsyncFor):
            async_for_statements.append({
                'line_number': node.lineno
            })
        elif isinstance(node, ast.AsyncWith):
            async_with_statements.append({
                'line_number': node.lineno
            })
    return {
        'async_functions': async_functions,
        'await_expressions': await_expressions,
        'async_for_statements': async_for_statements,
        'async_with_statements': async_with_statements
    }

# Generate control flow graph
def generate_control_flow_graph(tree):
    cfgs = defaultdict(lambda: defaultdict(list))
    current_function = None

    class CFGVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            nonlocal current_function
            current_function = node.name
            self.generic_visit(node)
            current_function = None

        def visit_AsyncFunctionDef(self, node):
            nonlocal current_function
            current_function = node.name
            self.generic_visit(node)
            current_function = None

        def visit_If(self, node):
            if current_function:
                if node.body:
                    try:
                        cfgs[current_function][node.lineno].append(node.body[0].lineno)
                    except:
                        pass
                if node.orelse:
                    try:
                        cfgs[current_function][node.lineno].append(node.orelse[0].lineno)
                    except:
                        pass
            self.generic_visit(node)

        def visit_For(self, node):
            if current_function:
                if node.body:
                    try:
                        cfgs[current_function][node.lineno].append(node.body[0].lineno)
                    except:
                        pass
            self.generic_visit(node)

        def visit_While(self, node):
            if current_function:
                if node.body:
                    try:
                        cfgs[current_function][node.lineno].append(node.body[0].lineno)
                    except:
                        pass
            self.generic_visit(node)

        def visit_Try(self, node):
            if current_function:
                for handler in node.handlers:
                    if handler.body:
                        try:
                            cfgs[current_function][node.lineno].append(handler.body[0].lineno)
                        except:
                            pass
            self.generic_visit(node)

    CFGVisitor().visit(tree)
    return {func: dict(edges) for func, edges in cfgs.items()}

# Generate call graph
def generate_call_graph(tree):
    call_graph = defaultdict(list)
    current_function = None

    class CallGraphVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            nonlocal current_function
            current_function = node.name
            self.generic_visit(node)
            current_function = None

        def visit_AsyncFunctionDef(self, node):
            nonlocal current_function
            current_function = node.name
            self.generic_visit(node)
            current_function = None

        def visit_Call(self, node):
            if current_function:
                if isinstance(node.func, ast.Name):
                    call_graph[current_function].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        call_graph[current_function].append(f"{node.func.value.id}.{node.func.attr}")
            self.generic_visit(node)

    CallGraphVisitor().visit(tree)
    return dict(call_graph)

# Analyze memory and performance patterns
def analyze_memory_and_performance_patterns(tree):
    issues = []

    class MemoryPerformanceVisitor(ast.NodeVisitor):
        def visit_For(self, node):
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                if len(node.iter.args) >= 1:
                    first_arg = node.iter.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, int) and first_arg.value > 1000:
                        issues.append({
                            'issue': 'Large range in loop',
                            'line_number': node.lineno
                        })
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Detect recursion
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == node.name:
                        issues.append({
                            'issue': 'Recursive function',
                            'function_name': node.name,
                            'line_number': child.lineno
                        })
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            # Detect recursion in async functions
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == node.name:
                        issues.append({
                            'issue': 'Recursive async function',
                            'function_name': node.name,
                            'line_number': child.lineno
                        })
            self.generic_visit(node)

        def visit_Global(self, node):
            if len(node.names) > 5:
                issues.append({
                    'issue': 'Excessive global variables',
                    'line_number': node.lineno
                })
            self.generic_visit(node)

    MemoryPerformanceVisitor().visit(tree)
    return issues

# Analyze symbol table and scope
def analyze_symbol_table_and_scope(tree):
    symbol_table = defaultdict(lambda: defaultdict(list))
    current_scope = 'global'

    class SymbolTableVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            nonlocal current_scope
            current_scope = node.name
            self.generic_visit(node)
            current_scope = 'global'

        def visit_AsyncFunctionDef(self, node):
            nonlocal current_scope
            current_scope = node.name
            self.generic_visit(node)
            current_scope = 'global'

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbol_table[current_scope][target.id].append(node.lineno)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                symbol_table[current_scope][node.id].append(node.lineno)
            self.generic_visit(node)

    SymbolTableVisitor().visit(tree)
    return {scope: dict(vars_) for scope, vars_ in symbol_table.items()}

# Analyze parallelization and concurrency opportunities
def analyze_parallelization_and_concurrency_opportunities(tree):
    opportunities = []

    class ParallelizationVisitor(ast.NodeVisitor):
        def visit_For(self, node):
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                opportunities.append({
                    'opportunity': 'Parallelizable loop',
                    'line_number': node.lineno
                })
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            if any(isinstance(n, ast.Await) for n in ast.walk(node)):
                opportunities.append({
                    'opportunity': 'Concurrency improvement in async function',
                    'function_name': node.name,
                    'line_number': node.lineno
                })
            self.generic_visit(node)

    ParallelizationVisitor().visit(tree)
    return opportunities

# Detect Code Smells
def detect_code_smells(tree):
    code_smells = []
    # Example Code Smells: Large Classes, Duplicated Code, Long Parameter Lists

    # Large Classes
    class_threshold = 10  # Example threshold for number of methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            if method_count > class_threshold:
                code_smells.append({
                    'pattern': 'Large class',
                    'class_name': node.name,
                    'line_number': node.lineno,
                    'method_count': method_count
                })

    # Long Parameter Lists
    param_threshold = 5
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.args.args) > param_threshold:
                code_smells.append({
                    'pattern': 'Long parameter list',
                    'function_name': node.name,
                    'line_number': node.lineno,
                    'parameter_count': len(node.args.args)
                })

    # Duplicated Code Detection Placeholder
    # Implementing actual code clone detection would require more complex logic or external tools
    # Here, we can flag functions with identical bodies as potential duplicates
    function_bodies = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = ast.dump(ast.fix_missing_locations(ast.Module(body=node.body)))
            function_bodies[body].append(node.name)

    for body, functions in function_bodies.items():
        if len(functions) > 1:
            code_smells.append({
                'pattern': 'Duplicated code',
                'function_names': functions,
                'line_number': 'Multiple lines'
            })

    return code_smells

# License and Compliance Checks
def check_license_compliance(file_path):
    licenses_found = []
    license_patterns = {
        'MIT License': r'MIT License',
        'Apache License 2.0': r'Apache License, Version 2\.0',
        'GNU General Public License v3.0': r'GNU GENERAL PUBLIC LICENSE Version 3',
        # Add more licenses as needed
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(4096)  # Read first 4KB for license info
            for license_name, pattern in license_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    licenses_found.append(license_name)
    except:
        pass

    return licenses_found

# Code Style and Linting
def perform_code_style_check(file_path):
    try:
        # Using flake8 for code style checking
        result = subprocess.run(['flake8', file_path], capture_output=True, text=True)
        if result.stdout:
            # Parse flake8 output
            issues = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':')
                    issues.append({
                        'line_number': int(parts[1]),
                        'column_number': int(parts[2]),
                        'code': parts[3].split(' ')[0],
                        'message': ':'.join(parts[3].split(':')[1:]).strip()
                    })
            return issues
        else:
            return []
    except FileNotFoundError:
        print("flake8 is not installed. Install it using 'pip install flake8'")
        return []
    except Exception as e:
        print(f"Error during code style check: {e}")
        return []

# Security Vulnerability Detection using Bandit
def perform_security_analysis(file_path):
    try:
        # Using bandit for security vulnerability scanning
        result = subprocess.run(['bandit', '-r', file_path, '-f', 'json'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            security_issues = json.loads(result.stdout)
            return security_issues.get('results', [])
        else:
            return []
    except FileNotFoundError:
        print("Bandit is not installed. Install it using 'pip install bandit'")
        return []
    except Exception as e:
        print(f"Error during security analysis: {e}")
        return []

# Dependency Graph Analysis using NetworkX
def generate_dependency_graph(imports):
    G = nx.DiGraph()
    for imp in imports:
        module = imp['module']
        G.add_node(module)
    return nx.to_dict_of_lists(G)

# Code Evolution and Technical Debt Analysis
def analyze_technical_debt(repo_path):
    # Placeholder: Implementing technical debt analysis requires integrating with tools like SonarQube or Code Climate
    # Alternatively, use metrics like cyclomatic complexity to estimate technical debt
    # Here, we will calculate cyclomatic complexity using radon
    try:
        result = subprocess.run(['radon', 'cc', repo_path, '-s', '-a', '-j'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            complexity_data = json.loads(result.stdout)
            return complexity_data
        else:
            return {}
    except FileNotFoundError:
        print("Radon is not installed. Install it using 'pip install radon'")
        return {}
    except Exception as e:
        print(f"Error during technical debt analysis: {e}")
        return {}

# Test Coverage Analysis using Coverage.py
def perform_test_coverage_analysis(file_path):
    try:
        # Assuming tests are run and coverage data is available
        # This function would parse coverage reports
        # Placeholder implementation
        coverage_data = {}
        return coverage_data
    except:
        return {}

# Main analysis function
def analyze_file(file_path, schema):
    error_message = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
    except Exception as e:
        error_message = f"Error reading file: {str(e)}"
        source_code = ""

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        error_message = f"Syntax error in file: {str(e)}"
        tree = None

    file_name = os.path.basename(file_path)
    last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    # Perform License and Compliance Checks
    licenses = check_license_compliance(file_path)

    # Perform Code Style and Linting
    code_style_issues = perform_code_style_check(file_path)

    # Perform Security Vulnerability Detection
    security_issues = perform_security_analysis(file_path)

    # Generate Dependency Graph
    imports = extract_imports(tree) if tree else []
    dependency_graph = generate_dependency_graph(imports)

    # Analyze Technical Debt
    # Assuming the script is run within a Git repository
    repo_path = os.path.dirname(file_path)
    technical_debt = analyze_technical_debt(repo_path)

    analysis_result = {
        "file_name": file_name,
        "file_path": file_path,
        "last_updated": last_updated,
        "error": error_message or "",
        "classes": extract_classes(tree) if tree else [],
        "functions": extract_functions(tree) if tree else [],
        "imports": imports if tree else [],
        "comments": extract_comments(source_code) if source_code else [],
        "docstrings": [parse_docstring(ds) for ds in extract_docstrings(tree) if ds] if tree else [],
        "comprehensions": extract_comprehensions(tree) if tree else [],
        "context_managers": extract_context_managers(tree) if tree else [],
        "try_blocks": extract_try_blocks(tree) if tree else [],
        "variable_usage": extract_variable_usage(tree) if tree else {},
        "patterns": detect_patterns(tree) if tree else [],
        "relationships": analyze_relationships(tree) if tree else {
            "function_calls": [],
            "class_inheritance": [],
            "module_dependencies": []
        },
        "metadata": extract_metadata(source_code, tree) if tree else {
            "lines_of_code": 0,
            "comments": 0,
            "docstrings": 0,
            "functions": 0,
            "classes": 0
        },
        "statistics": perform_statistical_analysis(tree) if tree else {
            "average_lines_per_function": 0,
            "average_arguments_per_function": 0,
            "complexity_distribution": {
                "mean": 0,
                "median": 0,
                "stdev": 0
            }
        },
        "lambdas_and_comprehensions": analyze_lambdas_and_comprehensions(tree) if tree else {
            "lambda_functions": [],
            "comprehensions": {
                "list_comprehensions": 0,
                "dict_comprehensions": 0,
                "set_comprehensions": 0,
                "generator_expressions": 0
            }
        },
        "type_annotations": analyze_type_annotations(tree) if tree else [],
        "data_flow_and_dependencies": analyze_data_flow_and_dependencies(tree) if tree else {
            "data_flow": {},
            "dependencies": {}
        },
        "context_managers_and_exceptions": analyze_context_managers_and_exceptions(tree) if tree else {
            "context_managers": [],
            "exception_handling": []
        },
        "async_await_constructs": analyze_async_await_constructs(tree) if tree else {
            "async_functions": [],
            "await_expressions": [],
            "async_for_statements": [],
            "async_with_statements": []
        },
        "control_flow_graphs": generate_control_flow_graph(tree) if tree else {},
        "call_graph": generate_call_graph(tree) if tree else {},
        "memory_and_performance_patterns": analyze_memory_and_performance_patterns(tree) if tree else [],
        "symbol_table_and_scope": analyze_symbol_table_and_scope(tree) if tree else {},
        "parallelization_and_concurrency_opportunities": analyze_parallelization_and_concurrency_opportunities(tree) if tree else [],
        "licenses": licenses,
        "code_style_issues": code_style_issues,
        "security_issues": security_issues,
        "dependency_graph": dependency_graph,
        "technical_debt": technical_debt
    }

    # Validate the analysis result against the schema
    if validate_output({"files": [analysis_result]}, schema):
        return {"files": [analysis_result]}
    else:
        return {
            "files": [{
                "file_name": file_name,
                "file_path": file_path,
                "error": "Analysis result does not match schema"
            }]
        }

# Main entry point
def main():
    schema_path = 'schemas/hierarchical_code_analysis_schema.json'  # Replace with your schema path
    file_path = 'example.py'  # Replace with your file path

    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    if not os.path.isfile(schema_path):
        print(f"Schema file '{schema_path}' does not exist.")
        return

    schema = load_schema(schema_path)
    analysis = analyze_file(file_path, schema)

    # Print or store the analysis result
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()

```
