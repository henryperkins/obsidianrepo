---
epoch: 1731552510542
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# merge_[Untitled_1_1](Untitled_1_1.md)*into_the_existing*[metrics.py](metrics.py.md)@20241113_204830

**user**: merge [System Monitoring and Metrics Tracking Module for Python](System%20Monitoring%20and%20Metrics%20Tracking%20Module%20for%20Python.md) into the existing [metrics.py](simple2/metrics.py.md)  
[Timestamp: 2024/11/13 20:48:30]

**ai**: To merge the contents of `[[Untitled 1 1]]` into the existing `[[metrics.py]]`, we will integrate the new class `CodeMetrics` and its methods into the existing `metrics.py` file. This involves combining the functionality of both classes and ensuring there are no conflicts or redundancies. Here's how the merged `metrics.py` might look:

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
from logger import log_info, log_error
from core.logger import LoggerSetup

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
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        complexity = 1  # Start with 1 for the function itself
        for node in ast.walk(function_node):
            if Metrics._is_decision_point(node):
                complexity += 1

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
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        cognitive_complexity = 0
        nesting_depth = 0
        prev_node = None

        for node in ast.walk(function_node):
            if Metrics._is_nesting_construct(node):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
            elif Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
            prev_node = node

        log_info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")
        return cognitive_complexity

    @staticmethod
    def _is_decision_point(node: ast.AST) -> bool:
        """
        Determine if a node represents a decision point for cyclomatic complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a decision point, False otherwise.
        """
        return isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler))

    @staticmethod
    def _is_nesting_construct(node: ast.AST) -> bool:
        """
        Determine if a node represents a nesting construct for cognitive complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a nesting construct, False otherwise.
        """
        return isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))

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
        return isinstance(node, (ast.BoolOp, ast.Compare)) and not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or isinstance(node, (ast.Continue, ast.Break, ast.Raise, ast.Return))

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index based on various metrics.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            float: Maintainability index score (0-100)
        """
        try:
            halstead = self.calculate_halstead_metrics(node)
            complexity = self.calculate_complexity(node)
            sloc = self._count_source_lines(node)
            
            # Calculate Maintainability Index
            volume = halstead['program_volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, min(100, mi))  # Normalize to 0-100
            
            return round(mi, 2)
            
        except Exception as e:
            log_error(f"Error calculating maintainability index: {e}")
            return 0.0

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            int: Number of source code lines
        """
        try:
            source = ast.unparse(node)
            lines = [line.strip() for line in source.splitlines()]
            return len([line for line in lines if line and not line.startswith('#')])
        except Exception as e:
            log_error(f"Error counting source lines: {e}")
            return 0

    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """
        Analyze module dependencies and imports.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            Dict[str, Set[str]]: Dictionary of module dependencies
        """
        deps = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }
        
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)
            return deps
        except Exception as e:
            log_error(f"Error analyzing dependencies: {e}")
            return deps

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statement and categorize dependency."""
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

    print("All tests passed.")

# Run tests
test_metrics()
```

## Key Changes

- Integrated the `CodeMetrics` class functionality into the existing `Metrics` class.
- Ensured no duplication of methods and merged similar functionalities.
- Adjusted imports and logging to be consistent across the module.
- Added the maintainability index calculation and dependency analysis methods to the `Metrics` class.
- Ensured that the test cases reflect the combined functionality.

Make sure to test the merged code thoroughly to ensure all functionalities work as expected. Adjust the logger import paths according to your project structure.  
[Timestamp: 2024/11/13 20:49:24]

**user**: 2024-11-14 03:03:22,083 - docstring_workflow - ERROR - Error processing function _get_full_qualified_name: name 'time' is not defined  
[Timestamp: 2024/11/13 21:03:45]

**ai**: The error message indicates that there is an attempt to use the `time` module or a `time` function within the `_get_full_qualified_name` function, but the `time` module has not been imported. To resolve this issue, you need to ensure that the `time` module is imported at the beginning of your script or module where `_get_full_qualified_name` is defined.

Here's how you can fix this:

1. **Import the `time` module**: Add an import statement for the `time` module at the top of your script or module.
2. **Check the function usage**: Ensure that the `time` module is actually needed in the `_get_full_qualified_name` function. If it's not needed, you can remove the reference to `time`.

Here's an example of how to import the `time` module:

```python
import time

# Your existing code...

def _get_full_qualified_name(node):
    """
    Example function that might use the time module.
    """
    # Example usage of time module
    current_time = time.time()
    # Your logic here...
    return current_time
```

If the `time` module is not needed in `_get_full_qualified_name`, you should review the function to remove any unnecessary references to `time`.

If you provide the code for `_get_full_qualified_name`, I can give more specific advice on how to address the issue.  
[Timestamp: 2024/11/13 21:03:50]
