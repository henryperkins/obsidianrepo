---
epoch: 1732439481846
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# Merge_[metrics.py](metrics.py.md)*and*[metrics_calculator.py](metrics_calculator.py.md)@20241124_031121

**user**: Merge [metrics.py](metrics.py.md) and [metrics_calculator.py](metrics_calculator.py.md)  
[Timestamp: 2024/11/24 03:11:21]

**ai**: To merge the contents of `metrics.py` and `metrics_calculator.py`, we need to integrate the functionalities of both scripts into a single module. Since both scripts have overlapping functionalities, we'll consolidate them into a unified class structure while ensuring that all methods are preserved and any redundant code is removed. Here's the merged version:

```python
"""
Code Metrics Analysis Module

This module provides comprehensive code quality and complexity metrics for
Python source code, including cyclomatic complexity, cognitive complexity,
Halstead metrics, and code quality analysis.

Version: 1.2.0
Author: Development Team
"""

import ast
import math
import sys
from dataclasses import dataclass
from typing import Dict, Set
from core.logger import LoggerSetup, log_debug, log_info, log_error

logger = LoggerSetup.get_logger(__name__)

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics"""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1000000.0


class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass


class Metrics:
    """
    Provides methods to calculate different complexity metrics for Python
    functions.

    This class includes methods for calculating cyclomatic complexity,
    cognitive complexity, Halstead metrics, and maintainability index. It
    also provides functionality to analyze module dependencies.
    """

    MAINTAINABILITY_THRESHOLDS = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    @staticmethod
    def calculate_cyclomatic_complexity(node: ast.AST) -> int:
        """
        Calculate the cyclomatic complexity of a function or class.

        Parameters:
            node (ast.AST): The AST node representing the function or class.

        Returns:
            int: The cyclomatic complexity of the function or class.
        """
        log_debug(f"Calculating cyclomatic complexity for node: "
                  f"{getattr(node, 'name', 'unknown')}")
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            log_error(f"Provided node is not a function or class definition: "
                      f"{ast.dump(node)}")
            return 0

        complexity = 1  # Start with 1 for the function or class itself
        decision_points = (
            ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler,
            ast.With, ast.Try, ast.BoolOp, ast.Lambda, ast.ListComp, ast.DictComp,
            ast.SetComp, ast.GeneratorExp, ast.IfExp, ast.Match  # For Python 3.10+
        )

        for subnode in ast.walk(node):
            if isinstance(subnode, decision_points):
                if isinstance(subnode, ast.BoolOp):
                    complexity += len(subnode.values) - 1
                    log_debug(f"Incremented complexity for BoolOp with "
                              f"{len(subnode.values) - 1} decision points: "
                              f"{ast.dump(subnode)}")
                elif isinstance(subnode, (ast.ListComp, ast.DictComp, ast.SetComp,
                                          ast.GeneratorExp)):
                    complexity += 1
                    log_debug(f"Incremented complexity for comprehension: "
                              f"{ast.dump(subnode)}")
                elif isinstance(subnode, ast.Match):
                    complexity += len(subnode.cases)
                    log_debug(f"Incremented complexity for Match with "
                              f"{len(subnode.cases)} cases: {ast.dump(subnode)}")
                else:
                    complexity += 1
                    log_debug(f"Incremented complexity at node: {ast.dump(subnode)}")

        log_info(f"Calculated cyclomatic complexity for node "
                 f"'{getattr(node, 'name', 'unknown')}' is {complexity}")
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(node: ast.AST) -> int:
        """
        Calculate the cognitive complexity of a function or class.

        Parameters:
            node (ast.AST): The AST node representing the function or class.

        Returns:
            int: The cognitive complexity of the function or class.
        """
        log_debug(f"Calculating cognitive complexity for node: "
                  f"{getattr(node, 'name', 'unknown')}")
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            log_error(f"Provided node is not a function or class definition: "
                      f"{ast.dump(node)}")
            return 0

        cognitive_complexity = 0
        nesting_depth = 0
        prev_node = None

        for subnode in ast.walk(node):
            if Metrics._is_nesting_construct(subnode):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
                log_debug(f"Nesting depth increased to {nesting_depth} at node: "
                          f"{ast.dump(subnode)}")
            elif prev_node is not None and Metrics._is_complexity_increment(subnode, prev_node):
                cognitive_complexity += 1
                log_debug(f"Incremented cognitive complexity at node: "
                          f"{ast.dump(subnode)}")
            prev_node = subnode

        log_info(f"Calculated cognitive complexity for node "
                 f"'{getattr(node, 'name', 'unknown')}' is {cognitive_complexity}")
        return cognitive_complexity

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate the overall complexity of the given AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The overall complexity score.
        """
        log_debug("Calculating overall complexity.")
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            log_error(f"Provided node is not a function or class definition: "
                      f"{ast.dump(node)}")
            return 0
        cyclomatic_complexity = self.calculate_cyclomatic_complexity(node)
        cognitive_complexity = self.calculate_cognitive_complexity(node)
        overall_complexity = cyclomatic_complexity + cognitive_complexity
        log_info(f"Calculated overall complexity for node '{getattr(node, 'name', 'unknown')}' is "
                 f"{overall_complexity}")
        return overall_complexity

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

        # Define operator and operand types
        operator_nodes = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                          ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
                          ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
                          ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                          ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
                          ast.Call, ast.Attribute, ast.Subscript, ast.Index,
                          ast.Slice)
        operand_nodes = (ast.Num, ast.Str, ast.Bytes, ast.Name, ast.List, ast.Tuple,
                         ast.Set, ast.Dict, ast.Constant)

        for subnode in ast.walk(node):
            if isinstance(subnode, operator_nodes):
                operators.add(type(subnode).__name__)
                operator_count += 1
            elif isinstance(subnode, operand_nodes):
                operands.add(type(subnode).__name__)
                operand_count += 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = operator_count
        N2 = operand_count

        program_length = N1 + N2
        program_vocabulary = n1 + n2
        program_volume = program_length * math.log2(program_vocabulary) if program_vocabulary > 0 else 0

        log_info(f"Calculated Halstead metrics: Length={program_length}, "
                 f"Vocabulary={program_vocabulary}, Volume={program_volume}")
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
    def _is_nesting_construct(node: ast.AST) -> bool:
        """
        Determine if a node represents a nesting construct for cognitive complexity.

        Parameters:
            node (ast.AST): The AST node to check.

        Returns:
            bool: True if the node is a nesting construct, False otherwise.
        """
        nesting_construct = isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                                              ast.ExceptHandler, ast.With,
                                              ast.Lambda, ast.ListComp, ast.DictComp,
                                              ast.SetComp, ast.GeneratorExp))
        log_debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} "
                  f"nesting construct.")
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
        increment = (isinstance(node, (ast.BoolOp, ast.Compare)) and
                     not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or
                     isinstance(node, (ast.Continue, ast.Break, ast.Raise,
                                       ast.Return)))
        log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not "
                  f"increment'} complexity.")
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


def add_parent_references(node: ast.AST) -> None:
    """Add parent references to all nodes in the AST."""
    for child in ast.walk(node):
        for child_node in ast.iter_child_nodes(child):
            child_node.parent = child


def test_metrics():
    """
    Test function for the Metrics class.

    This function tests the calculation of cyclomatic and cognitive complexity
    for a sample function defined in source_code.
    """
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
    add_parent_references(tree)
    function_node = tree.body[0]

    if isinstance(function_node, ast.FunctionDef):
        # Test cyclomatic complexity
        cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(
            function_node)
        assert cyclomatic_complexity == 4, f"Expected 4, got {cyclomatic_complexity}"

        # Test cognitive complexity
        cognitive_complexity = Metrics.calculate_cognitive_complexity(
            function_node)
        assert cognitive_complexity == 11, f"Expected 11, got {cognitive_complexity}"

        log_info("All tests passed.")
    else:
        log_error("The node is not a function definition.")


# Ensure tests run only when the script is executed directly
if __name__ == "__main__":
    test_metrics()
```

## Key Changes

- Combined the functionalities from both `metrics.py` and `metrics_calculator.py` into a single `Metrics` class.
- Removed redundant methods and ensured all necessary methods are included.
- Updated the `calculate_cyclomatic_complexity` and `calculate_cognitive_complexity` methods to handle both functions and classes.
- Unified logging and error handling across the module.
- Ensured the test function `test_metrics` is included to validate the functionalities.  
[Timestamp: 2024/11/24 03:12:09]

**user**: Traceback (most recent call last):  
  File "/home/azureuser/simple/main.py", line 15, in <module>  
    from ai_interaction import AIInteractionHandler  
  File "/home/azureuser/simple/ai_interaction.py", line 27, in <module>  
    from extract.extraction_manager import ExtractionManager  
  File "/home/azureuser/simple/extract/extraction_manager.py", line 4, in <module>  
    from core.metrics import Metrics # Assuming the metrics module is available  
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
  File "/home/azureuser/simple/core/metrics.py", line 296  
    log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not "  
                                                                      ^  
SyntaxError: unterminated string literal (detected at line 296)  
[Timestamp: 2024/11/24 03:13:09]

**ai**: The error message indicates that there's a syntax error in your `metrics.py` file, specifically an unterminated string literal in a `log_debug` statement. This usually happens when a string is not properly closed with matching quotation marks.

Here's the problematic line from the traceback:

```python
log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not "
```

The issue is that the string inside the f-string is not properly closed. Let's fix it by ensuring the string is properly terminated:

```python
log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")
```

Make sure the entire string is enclosed within a single pair of quotes. Here's the corrected section of the code:

```python
def _is_complexity_increment(node: ast.AST, prev_node: ast.AST) -> bool:
    """
    Determine if a node should increment cognitive complexity.

    Parameters:
    node (ast.AST): The current AST node.
    prev_node (ast.AST): The previous AST node.

    Returns:
    bool: True if the node should increment complexity, False otherwise.
    """
    increment = (isinstance(node, (ast.BoolOp, ast.Compare)) and
                 not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or
                 isinstance(node, (ast.Continue, ast.Break, ast.Raise,
                                   ast.Return)))
    log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")
    return increment
```

After making this change, save the file and try running your program again. This should resolve the syntax error.  
[Timestamp: 2024/11/24 03:13:13]

**user**: ERROR: Error in class metadata extraction for MetricsThresholds: Metrics._is_complexity_increment() missing 1 required positional argument: 'prev_node'  
ERROR: Error in function metadata extraction for get_metric_severity: Metrics._is_complexity_increment() missing 1 required positional argument: 'prev_node'  
ERROR: Error in function metadata extraction for format_metric_value: Metrics._is_complexity_increment() missing 1 required positional argument: 'prev_node'  
ERROR: Unexpected error: 'args'  
[Timestamp: 2024/11/24 03:15:58]

**ai**: The error messages indicate that there is an issue with how the `Metrics._is_complexity_increment` method is being called. This method requires two arguments: `node` and `prev_node`. If it is being called without providing both arguments, it will raise an error.

To resolve this, let's ensure that the method is being called correctly wherever it is used. Additionally, let's check the context in which these errors occur to provide a more comprehensive solution.

## Steps to Fix the Error

1. **Check Method Calls**:
   - Ensure that every call to `Metrics._is_complexity_increment` provides both the `node` and `prev_node` arguments.

2. **Review the Code**:
   - Look for any place in your code where `Metrics._is_complexity_increment` is called and ensure it has the correct number of arguments.

3. **Debugging Information**:
   - Add logging or print statements before the calls to `Metrics._is_complexity_increment` to verify what arguments are being passed.

4. **Check for Other Errors**:
   - The error messages also mention issues with class metadata extraction and function metadata extraction. Ensure that any functions or classes that are using `Metrics._is_complexity_increment` are correctly structured and that all required arguments are provided.

Here's a general example of how you might correct a call to `Metrics._is_complexity_increment`:

```python
# Example of a correct call to _is_complexity_increment
for node in ast.walk(some_ast_node):
    prev_node = get_previous_node(node)  # Hypothetical function to get the previous node
    if Metrics._is_complexity_increment(node, prev_node):
        # Do something with the increment
```

## Additional Debugging Steps

- **Check the Usage Context**: If the method is used in a loop or a recursive function, ensure that both the current and previous nodes are being tracked correctly.
- **Error Handling**: Add try-except blocks around the calls to capture and log any unexpected errors for further analysis.

## If the Problem Persists

- **Provide More Context**: If the problem persists, providing more context or code snippets where the method is being used would help in diagnosing the issue more accurately.
- **Review Related Functions**: Check the functions `get_metric_severity` and `format_metric_value` for any incorrect usage of the `Metrics` class methods.

By following these steps, you should be able to identify and fix the issue with the method call. If you need further assistance, feel free to provide more details or code snippets related to the error.  
[Timestamp: 2024/11/24 03:16:05]
