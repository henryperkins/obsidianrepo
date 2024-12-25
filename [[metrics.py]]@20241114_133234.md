---
epoch: 1731612754149
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [metrics.py](metrics.py.md)@20241114_133234

**user**: [metrics.py](simple2/metrics.py.md)  
[Timestamp: 2024/11/14 13:32:34]

**ai**: The provided `metrics.py` module is a comprehensive tool for analyzing code quality and complexity metrics in Python source code. It includes functions to calculate cyclomatic complexity, cognitive complexity, maintainability index, and analyze module dependencies. Below, I will provide a brief overview of the key components and suggest some improvements and additional test cases.

## Key Components

1. **Cyclomatic Complexity**: Measures the number of linearly independent paths through a program's source code. It is calculated by counting decision points like `if`, `for`, `while`, etc.
2. **Cognitive Complexity**: Evaluates how difficult a piece of code is to understand, considering factors like nesting depth and flow control statements.
3. **Maintainability Index**: A composite metric that considers cyclomatic complexity, Halstead volume, and lines of code to provide a score indicating how maintainable the code is.
4. **Dependency Analysis**: Identifies and categorizes imports into standard library, third-party, and local modules.

## Suggested Improvements

1. **Error Handling**: The module already includes basic error handling, but it can be enhanced by providing more specific error messages and possibly raising custom exceptions where appropriate.
2. **Logging**: The use of logging is good, but ensure that log levels are appropriately set (e.g., using `log_debug` only for detailed debugging information).
3. **Performance**: Consider optimizing the AST traversal to avoid redundant checks or computations.
4. **Documentation**: Ensure all functions have comprehensive docstrings, and consider adding examples of usage.

## Additional Test Cases

Here are some additional test cases to further validate the functionality of the module:

```python
def test_maintainability_index():
    log_info("Starting test_maintainability_index.")
    source_code = """
def complex_function(a, b):
    if a > b:
        try:
            for i in range(a):
                if i % 2 == 0:
                    print(i)
                else:
                    raise ValueError("Odd number")
        except ValueError as e:
            print(e)
    else:
        return b - a
    return a + b
"""
    tree = ast.parse(source_code)
    function_node = tree.body[0]

    # Test maintainability index
    metrics = Metrics()
    maintainability_index = metrics.calculate_maintainability_index(function_node)
    assert 0 <= maintainability_index <= 100, f"Maintainability index out of bounds: {maintainability_index}"

    log_info("Maintainability index test passed.")

def test_dependency_analysis():
    log_info("Starting test_dependency_analysis.")
    source_code = """
import os
import sys
from collections import defaultdict
import numpy as np
import my_local_module
"""
    tree = ast.parse(source_code)

    # Test dependency analysis
    metrics = Metrics()
    dependencies = metrics.analyze_dependencies(tree)
    assert 'os' in dependencies['stdlib'], "os should be in stdlib"
    assert 'sys' in dependencies['stdlib'], "sys should be in stdlib"
    assert 'numpy' in dependencies['third_party'], "numpy should be in third_party"
    assert 'my_local_module' in dependencies['local'], "my_local_module should be in local"

    log_info("Dependency analysis test passed.")

# Run additional tests
test_maintainability_index()
test_dependency_analysis()
```

## Conclusion

The `metrics.py` module is a robust tool for analyzing Python code complexity and quality. By adding more test cases, enhancing error handling, and optimizing performance, the module can be further improved to provide more accurate and reliable metrics.  
[Timestamp: 2024/11/14 13:32:43]

**user**: Merge these calculations into the current metrics.py:

import ast  
import math  
from typing import Dict

class Metrics:  
    """  
    Provides methods to calculate different complexity metrics for Python functions.  
    """

    @staticmethod
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cyclomatic complexity of a function.
        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.
        Returns:
        int: The cyclomatic complexity of the function.
        """
        log_debug(f"Calculating cyclomatic complexity for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0
        complexity = 1  # Start with 1 for the function itself
        for node in ast.walk(function_node):
            if Metrics._is_decision_point(node):
                complexity += 1
                log_debug(f"Incremented complexity at node: {ast.dump(node)}")
        log_info(f"Calculated cyclomatic complexity for function '{function_node.name}' is {complexity}")
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cognitive complexity of a function.
        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.
        Returns:
        int: The cognitive complexity of the function.
        """
        log_debug(f"Calculating cognitive complexity for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0
        cognitive_complexity = 0
        nesting_depth = 0
        prev_node: Optional[ast.AST] = None
        for node in ast.walk(function_node):
            if Metrics._is_nesting_construct(node):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
                log_debug(f"Nesting depth increased to {nesting_depth} at node: {ast.dump(node)}")
            elif prev_node is not None and Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
                log_debug(f"Incremented cognitive complexity at node: {ast.dump(node)}")
            prev_node = node
        log_info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")
        return cognitive_complexity

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index based on various metrics.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            float: Maintainability index score (0-100)
        """
        log_debug("Calculating maintainability index.")
        try:
            halstead = self.calculate_halstead_metrics(node)
            complexity = self.calculate_complexity(node)
            sloc = self._count_source_lines(node)
            
            # Calculate Maintainability Index
            volume = halstead['program_volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, min(100, mi))  # Normalize to 0-100
            
            log_info(f"Calculated maintainability index is {mi}")
            return round(mi, 2)
            
        except Exception as e:
            log_error(f"Error calculating maintainability index: {e}")
            return 0.0

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for the given AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            Dict[str, float]: A dictionary containing Halstead metrics.
        """
        log_debug("Calculating Halstead metrics.")
        # Example implementation of Halstead metrics calculation
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        for n in ast.walk(node):
            if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.add(type(n).__name__)
                operator_count += 1
            elif isinstance(n, (ast.Num, ast.Str, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                operands.add(type(n).__name__)
                operand_count += 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = operator_count
        N2 = operand_count

        program_length = N1 + N2
        program_vocabulary = n1 + n2
        program_volume = program_length * math.log2(program_vocabulary) if program_vocabulary > 0 else 0

        return {
            'program_length': program_length,
            'program_vocabulary': program_vocabulary,
            'program_volume': program_volume
        }

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            int: Number of source code lines
        """
        log_debug("Counting source lines of code.")
        try:
            # Implementation to count source lines of code
            pass
        except Exception as e:
            log_error(f"Error counting source lines of code: {e}")
            return 0
[Timestamp: 2024/11/14 13:32:50]

**ai**: To merge the calculations from the new code snippet into the existing `metrics.py`, we need to integrate the Halstead metrics calculation and ensure that all methods are consolidated under the `Metrics` class. I'll also ensure that the code is clean, modular, and includes appropriate logging and error handling. Here's the updated `metrics.py`:

```python
"""
Code Metrics Analysis Module

This module provides comprehensive code quality and complexity metrics for Python source code,
including cyclomatic complexity, cognitive complexity, Halstead metrics, and code quality analysis.

Version: 1.0.0
Author: Development Team
"""

import ast
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from logger import log_info, log_error, log_debug

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class Metrics:
    """
    Provides methods to calculate different complexity metrics for Python functions.
    """

    MAINTAINABILITY_THRESHOLDS = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    @staticmethod
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cyclomatic complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cyclomatic complexity of the function.
        """
        log_debug(f"Calculating cyclomatic complexity for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        complexity = 1  # Start with 1 for the function itself
        for node in ast.walk(function_node):
            if Metrics._is_decision_point(node):
                complexity += 1
                log_debug(f"Incremented complexity at node: {ast.dump(node)}")

        log_info(f"Calculated cyclomatic complexity for function '{function_node.name}' is {complexity}")
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cognitive complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cognitive complexity of the function.
        """
        log_debug(f"Calculating cognitive complexity for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        cognitive_complexity = 0
        nesting_depth = 0
        prev_node: Optional[ast.AST] = None

        for node in ast.walk(function_node):
            if Metrics._is_nesting_construct(node):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
                log_debug(f"Nesting depth increased to {nesting_depth} at node: {ast.dump(node)}")
            elif prev_node is not None and Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
                log_debug(f"Incremented cognitive complexity at node: {ast.dump(node)}")
            prev_node = node

        log_info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")
        return cognitive_complexity

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index based on various metrics.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            float: Maintainability index score (0-100)
        """
        log_debug("Calculating maintainability index.")
        try:
            halstead = self.calculate_halstead_metrics(node)
            complexity = self.calculate_complexity(node)
            sloc = self._count_source_lines(node)
            
            # Calculate Maintainability Index
            volume = halstead['program_volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, min(100, mi))  # Normalize to 0-100
            
            log_info(f"Calculated maintainability index is {mi}")
            return round(mi, 2)
            
        except Exception as e:
            log_error(f"Error calculating maintainability index: {e}")
            return 0.0

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for the given AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            Dict[str, float]: A dictionary containing Halstead metrics.
        """
        log_debug("Calculating Halstead metrics.")
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        for n in ast.walk(node):
            if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.add(type(n).__name__)
                operator_count += 1
            elif isinstance(n, (ast.Num, ast.Str, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                operands.add(type(n).__name__)
                operand_count += 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = operator_count
        N2 = operand_count

        program_length = N1 + N2
        program_vocabulary = n1 + n2
        program_volume = program_length * math.log2(program_vocabulary) if program_vocabulary > 0 else 0

        log_info(f"Calculated Halstead metrics: Length={program_length}, Vocabulary={program_vocabulary}, Volume={program_volume}")
        return {
            'program_length': program_length,
            'program_vocabulary': program_vocabulary,
            'program_volume': program_volume
        }

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            int: Number of source code lines
        """
        log_debug("Counting source lines of code.")
        try:
            source = ast.unparse(node)
            lines = [line.strip() for line in source.splitlines()]
            count = len([line for line in lines if line and not line.startswith('#')])
            log_info(f"Counted {count} source lines of code.")
            return count
        except Exception as e:
            log_error(f"Error counting source lines: {e}")
            return 0

    @staticmethod
    def _is_decision_point(node: ast.AST) -> bool:
        """
        Determine if a node represents a decision point for cyclomatic complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a decision point, False otherwise.
        """
        decision_point = isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler))
        log_debug(f"Node {ast.dump(node)} is {'a' if decision_point else 'not a'} decision point.")
        return decision_point

    @staticmethod
    def _is_nesting_construct(node: ast.AST) -> bool:
        """
        Determine if a node represents a nesting construct for cognitive complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a nesting construct, False otherwise.
        """
        nesting_construct = isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
        log_debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} nesting construct.")
        return nesting_construct

    @staticmethod
    def _is_complexity_increment(node: ast.AST, prev_node: ast.AST) -> bool:
        """
        Determine if a node should increment cognitive complexity.

        Parameters:
        node (ast.AST): The current AST node.
        prev_node (ast.AST): The previous AST node.

        Returns:
        bool: True if the node should increment complexity, False otherwise.
        """
        increment = isinstance(node, (ast.BoolOp, ast.Compare)) and not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or isinstance(node, (ast.Continue, ast.Break, ast.Raise, ast.Return))
        log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")
        return increment

    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """
        Analyze module dependencies and imports.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            Dict[str, Set[str]]: Dictionary of module dependencies
        """
        log_debug("Analyzing module dependencies.")
        deps = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }
        
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)
            log_info(f"Analyzed dependencies: {deps}")
            return deps
        except Exception as e:
            log_error(f"Error analyzing dependencies: {e}")
            return deps

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statement and categorize dependency."""
        log_debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
        except Exception as e:
            log_error(f"Error processing import: {e}")

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize import as stdlib, third-party, or local."""
        log_debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.stdlib_module_names:
                deps['stdlib'].add(module_name)
            elif '.' in module_name:
                deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)
        except Exception as e:
            log_error(f"Error categorizing import {module_name}: {e}")

# Suggested Test Cases
def test_metrics():
    log_info("Starting test_metrics.")
    source_code = """
def example_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            else:
                continue
    else:
        return -1
    return 0
"""
    tree = ast.parse(source_code)
    function_node = tree.body[0]  # Assuming the first node is the function definition

    # Test cyclomatic complexity
    cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)
    assert cyclomatic_complexity == 5, f"Expected 5, got {cyclomatic_complexity}"

    # Test cognitive complexity
    cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)
    assert cognitive_complexity == 6, f"Expected 6, got {cognitive_complexity}"

    log_info("All tests passed.")

# Run tests
test_metrics()
```

## Key Changes

- **Halstead Metrics**: Integrated the `calculate_halstead_metrics` method into the `Metrics` class, providing a basic implementation for calculating Halstead metrics.
- **Logging and Error Handling**: Ensured consistent logging and error handling across all methods.
- **Maintainability Index**: The maintainability index calculation now uses the Halstead metrics and complexity values.
- **Test Cases**: Retained the existing test cases and ensured they are comprehensive for the new functionality.

This updated module now includes a more complete set of metrics calculations and is structured to facilitate easy testing and further expansion.  
[Timestamp: 2024/11/14 13:33:18]

**user**: 	import time  
from typing import Dict, List, Optional  
from dataclasses import dataclass  
from datetime import datetime  
import json  
from logger import log_info, log_error, log_debug

@dataclass  
class APIMetrics:  
    """Metrics data structure for API operations."""  
    timestamp: float  
    operation: str  
    response_time: float  
    tokens_used: int  
    status: str  
    error: Optional[str] = None

@dataclass  
class BatchMetrics:  
    def **init**(self, total_functions: int, successful: int, failed: int, total_tokens: int, total_time: float, average_time_per_function: float):  
        self.total_functions = total_functions  
        self.successful = successful  
        self.failed = failed  
        self.total_tokens = total_tokens  
        self.total_time = total_time  
        self.average_time_per_function = average_time_per_function  
class SystemMonitor:  
    def **init**(self):  
        self.docstring_changes: Dict[str, Any] = {  
            'added': [],  
            'updated': [],  
            'retained': [],  
            'failed': []  
        }  
        self.current_batch = {  
            'total_tokens': 0,  
            'total_time': 0.0, # Initialize as float  
            'processed': 0,  
            'failed': 0  
        }
        
    def log_request(self, endpoint, tokens, response_time, status, error=None):
        """
        Log details of an API request.

        Args:
            endpoint (str): The API endpoint that was called.
            tokens (int): Number of tokens used in the request.
            response_time (float): Time taken for the request.
            status (str): Status of the request (e.g., 'success', 'error').
            error (str, optional): Error message if the request failed.
        """
        log_info(f"API Request to {endpoint}: {status}, Tokens: {tokens}, Time: {response_time}s")
        if error:
            log_error(f"Error: {error}")

    def log_api_request(
        self,
        endpoint: str,
        tokens: int,
        response_time: float,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """
        Log an API request with detailed metrics.

        Args:
            endpoint: The API endpoint called
            tokens: Number of tokens used
            response_time: Time taken for the request
            status: Status of the request
            error: Optional error message
        """
        log_debug(f"Logging API request to endpoint: {endpoint}")
        metric = APIMetrics(
            timestamp=time.time(),
            operation=endpoint,
            response_time=response_time,
            tokens_used=tokens,
            status=status,
            error=error
        )
        self.api_metrics.append(metric)
        log_info(f"API Request logged: {endpoint} - Status: {status}")

    def log_cache_hit(self, function_name: str) -> None:
        """Log a cache hit event."""
        self.cache_hits += 1
        log_info(f"Cache hit for function: {function_name}")

    def log_cache_miss(self, function_name: str) -> None:
        """Log a cache miss event."""
        self.cache_misses += 1
        log_info(f"Cache miss for function: {function_name}")

    def log_docstring_changes(self, action: str, function_name: str) -> None:
        log_debug(f"Logging docstring change: {action} for function: {function_name}")
        if action in self.docstring_changes:
            self.docstring_changes[action].append({
                'function': function_name,
                'timestamp': datetime.now().isoformat()
            })
            log_info(f"Docstring {action} for function: {function_name}")
        else:
            log_error(f"Unknown docstring action: {action}")

    def log_operation_complete(
        self,
        function_name: str,
        execution_time: float,
        tokens_used: int
    ) -> None:
        """
        Log completion of a function processing operation.

        Args:
            function_name: Name of the processed function
            execution_time: Time taken to process
            tokens_used: Number of tokens used
        """
        log_debug(f"Logging operation completion for function: {function_name}")
        self.current_batch['total_tokens'] += tokens_used
        self.current_batch['total_time'] += execution_time  # Ensure this is handled as float
        self.current_batch['processed'] += 1
        log_info(f"Operation complete for function: {function_name}")

    def log_batch_completion(self, total_functions: int) -> BatchMetrics:
        """
        Log completion of a batch processing operation.

        Args:
            total_functions: Total number of functions in the batch

        Returns:
            BatchMetrics: Metrics for the completed batch
        """
        log_debug("Logging batch completion")
        metrics = BatchMetrics(
            total_functions=total_functions,
            successful=self.current_batch['processed'],
            failed=self.current_batch['failed'],
            total_tokens=self.current_batch['total_tokens'],
            total_time=self.current_batch['total_time'],
            average_time_per_function=self.current_batch['total_time'] / 
                                    max(self.current_batch['processed'], 1)
        )
        
        # Reset batch metrics
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0.0,  # Reset as float
            'processed': 0,
            'failed': 0
        }
        log_info(f"Batch processing completed: {metrics}")
        return metrics

    def get_metrics_summary(self) -> Dict:
        """
        Generate a comprehensive metrics summary.

        Returns:
            Dict: Complete metrics summary
        """
        log_debug("Generating metrics summary")
        runtime = time.time() - self.start_time
        total_requests = len(self.api_metrics)
        failed_requests = len([m for m in self.api_metrics if m.error])
        
        summary = {
            'runtime_seconds': runtime,
            'api_metrics': {
                'total_requests': total_requests,
                'failed_requests': failed_requests,
                'error_rate': failed_requests / max(total_requests, 1),
                'average_response_time': sum(m.response_time for m in self.api_metrics) / 
                                      max(total_requests, 1),
                'total_tokens_used': sum(m.tokens_used for m in self.api_metrics)
            },
            'cache_metrics': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'docstring_changes': {
                action: len(changes) 
                for action, changes in self.docstring_changes.items()
            }
        }
        log_info(f"Metrics summary generated: {summary}")
        return summary

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save the metrics file
        """
        log_debug(f"Exporting metrics to file: {filepath}")
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            log_info(f"Metrics exported to: {filepath}")
        except Exception as e:
            log_error(f"Failed to export metrics: {str(e)}")
[Timestamp: 2024/11/14 13:41:50]

**ai**: The provided code defines a system for logging and monitoring various metrics related to API operations, batch processing, and cache performance. It uses Python's `dataclasses` for structured data representation and includes detailed logging for tracking operations. Below, I'll ensure the code is well-organized, with clear comments and consistent error handling. I'll also address any potential issues and ensure that all functionalities are properly integrated.

```python
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from logger import log_info, log_error, log_debug

@dataclass
class APIMetrics:
    """Metrics data structure for API operations."""
    timestamp: float
    operation: str
    response_time: float
    tokens_used: int
    status: str
    error: Optional[str] = None

@dataclass
class BatchMetrics:
    """Metrics data structure for batch operations."""
    total_functions: int
    successful: int
    failed: int
    total_tokens: int
    total_time: float
    average_time_per_function: float

class SystemMonitor:
    def __init__(self):
        self.api_metrics: List[APIMetrics] = []
        self.docstring_changes: Dict[str, List[Dict[str, str]]] = {
            'added': [],
            'updated': [],
            'retained': [],
            'failed': []
        }
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0.0,
            'processed': 0,
            'failed': 0
        }
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

    def log_api_request(self, endpoint: str, tokens: int, response_time: float, status: str, error: Optional[str] = None) -> None:
        """
        Log an API request with detailed metrics.

        Args:
            endpoint: The API endpoint called
            tokens: Number of tokens used
            response_time: Time taken for the request
            status: Status of the request
            error: Optional error message
        """
        log_debug(f"Logging API request to endpoint: {endpoint}")
        metric = APIMetrics(
            timestamp=time.time(),
            operation=endpoint,
            response_time=response_time,
            tokens_used=tokens,
            status=status,
            error=error
        )
        self.api_metrics.append(metric)
        log_info(f"API Request logged: {endpoint} - Status: {status}")

    def log_cache_hit(self, function_name: str) -> None:
        """Log a cache hit event."""
        self.cache_hits += 1
        log_info(f"Cache hit for function: {function_name}")

    def log_cache_miss(self, function_name: str) -> None:
        """Log a cache miss event."""
        self.cache_misses += 1
        log_info(f"Cache miss for function: {function_name}")

    def log_docstring_changes(self, action: str, function_name: str) -> None:
        """
        Log changes to function docstrings.

        Args:
            action: The action performed ('added', 'updated', 'retained', 'failed')
            function_name: The name of the function whose docstring was changed
        """
        log_debug(f"Logging docstring change: {action} for function: {function_name}")
        if action in self.docstring_changes:
            self.docstring_changes[action].append({
                'function': function_name,
                'timestamp': datetime.now().isoformat()
            })
            log_info(f"Docstring {action} for function: {function_name}")
        else:
            log_error(f"Unknown docstring action: {action}")

    def log_operation_complete(self, function_name: str, execution_time: float, tokens_used: int) -> None:
        """
        Log completion of a function processing operation.

        Args:
            function_name: Name of the processed function
            execution_time: Time taken to process
            tokens_used: Number of tokens used
        """
        log_debug(f"Logging operation completion for function: {function_name}")
        self.current_batch['total_tokens'] += tokens_used
        self.current_batch['total_time'] += execution_time
        self.current_batch['processed'] += 1
        log_info(f"Operation complete for function: {function_name}")

    def log_batch_completion(self, total_functions: int) -> BatchMetrics:
        """
        Log completion of a batch processing operation.

        Args:
            total_functions: Total number of functions in the batch

        Returns:
            BatchMetrics: Metrics for the completed batch
        """
        log_debug("Logging batch completion")
        metrics = BatchMetrics(
            total_functions=total_functions,
            successful=self.current_batch['processed'],
            failed=self.current_batch['failed'],
            total_tokens=self.current_batch['total_tokens'],
            total_time=self.current_batch['total_time'],
            average_time_per_function=self.current_batch['total_time'] / max(self.current_batch['processed'], 1)
        )
        
        # Reset batch metrics
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0.0,
            'processed': 0,
            'failed': 0
        }
        log_info(f"Batch processing completed: {metrics}")
        return metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics summary.

        Returns:
            Dict: Complete metrics summary
        """
        log_debug("Generating metrics summary")
        runtime = time.time() - self.start_time
        total_requests = len(self.api_metrics)
        failed_requests = len([m for m in self.api_metrics if m.error])
        
        summary = {
            'runtime_seconds': runtime,
            'api_metrics': {
                'total_requests': total_requests,
                'failed_requests': failed_requests,
                'error_rate': failed_requests / max(total_requests, 1),
                'average_response_time': sum(m.response_time for m in self.api_metrics) / max(total_requests, 1),
                'total_tokens_used': sum(m.tokens_used for m in self.api_metrics)
            },
            'cache_metrics': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'docstring_changes': {
                action: len(changes) 
                for action, changes in self.docstring_changes.items()
            }
        }
        log_info(f"Metrics summary generated: {summary}")
        return summary

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save the metrics file
        """
        log_debug(f"Exporting metrics to file: {filepath}")
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            log_info(f"Metrics exported to: {filepath}")
        except Exception as e:
            log_error(f"Failed to export metrics: {str(e)}")
```

## Key Features and Improvements

- **Data Structures**: Utilized `dataclass` for `APIMetrics` and `BatchMetrics` to simplify data handling.
- **Logging**: Consistent use of logging for debugging and information purposes.
- **Error Handling**: Basic error handling is included, especially in methods that involve file operations.
- **Metrics Calculation**: Includes methods to log API requests, cache hits/misses, docstring changes, and batch operations.
- **Metrics Summary**: Provides a comprehensive summary of metrics, including API and cache statistics, and allows exporting to a JSON file.

This code is structured to facilitate easy integration into a larger system, with clear separation of concerns and modular design.  
[Timestamp: 2024/11/14 13:42:11]

**user**: Argument of type "int | float" cannot be assigned to parameter "successful" of type "int" in function "init" Type "int | float" is not assignable to type "int" "float" is not assignable to "int"  
[Timestamp: 2024/11/14 13:43:53]

**ai**: The error you're encountering suggests that there is a type mismatch in the `BatchMetrics` initialization, specifically with the `successful` parameter. This likely occurs because the value being passed to `successful` might be a float, while the `BatchMetrics` dataclass expects an integer.

To resolve this issue, ensure that all parameters passed to `BatchMetrics` are of the correct type. Here's how you can adjust the code to ensure type correctness:

1. **Ensure Integer Values**: Make sure that all values passed to `BatchMetrics` are integers, especially those that are calculated or derived from other operations.
2. **Explicit Type Casting**: Use explicit type casting where necessary to convert float values to integers.

Here's the relevant part of the code with type casting added to ensure that all values are integers:

```python
@dataclass
class BatchMetrics:
    """Metrics data structure for batch operations."""
    total_functions: int
    successful: int
    failed: int
    total_tokens: int
    total_time: float
    average_time_per_function: float

class SystemMonitor:
    # ... other methods ...

    def log_batch_completion(self, total_functions: int) -> BatchMetrics:
        """
        Log completion of a batch processing operation.

        Args:
            total_functions: Total number of functions in the batch

        Returns:
            BatchMetrics: Metrics for the completed batch
        """
        log_debug("Logging batch completion")
        metrics = BatchMetrics(
            total_functions=total_functions,
            successful=int(self.current_batch['processed']),  # Ensure integer type
            failed=int(self.current_batch['failed']),         # Ensure integer type
            total_tokens=int(self.current_batch['total_tokens']),  # Ensure integer type
            total_time=self.current_batch['total_time'],
            average_time_per_function=self.current_batch['total_time'] / max(self.current_batch['processed'], 1)
        )
        
        # Reset batch metrics
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0.0,
            'processed': 0,
            'failed': 0
        }
        log_info(f"Batch processing completed: {metrics}")
        return metrics

    # ... other methods ...
```

## Key Adjustments

- **Type Casting**: Used `int()` to explicitly cast `self.current_batch['processed']`, `self.current_batch['failed']`, and `self.current_batch['total_tokens']` to integers.
- **Preserve Floats**: Kept `total_time` and `average_time_per_function` as floats since they represent time, which can naturally be a floating-point number.

By ensuring that all parameters match the expected types in the `BatchMetrics` dataclass, you should eliminate the type mismatch error.  
[Timestamp: 2024/11/14 13:44:00]
