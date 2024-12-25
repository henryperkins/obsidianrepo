```python
"""
documentation_analyzer.py - Docstring Analysis System

This module analyzes existing docstrings to determine if they are complete and correct,
according to a predefined schema.

Classes:
    DocumentationAnalyzer: Analyzes existing docstrings to determine if they are complete and correct.
"""

import ast
from typing import Optional, List
from docstring_utils import parse_docstring, validate_docstring, analyze_code_element_docstring
from logger import log_info, log_error, log_debug

class DocumentationAnalyzer:
    """
    Analyzes existing docstrings to determine if they are complete and correct.

    Methods:
        is_docstring_incomplete(function_node: ast.FunctionDef) -> bool: Determines if the existing docstring for a function is incomplete.
        is_class_docstring_incomplete(class_node: ast.ClassDef) -> bool: Checks if a class has an incomplete docstring.
    """

    def is_docstring_incomplete(self, function_node: ast.FunctionDef) -> bool:
        """
        Determine if the existing docstring for a function is incomplete.

        Args:
            function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
            bool: True if the docstring is incomplete, False otherwise.
        """
        log_debug(f"Checking if docstring is incomplete for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Node is not a function definition, cannot evaluate docstring.")
            return True

        issues = analyze_code_element_docstring(function_node)
        if issues:
            log_info(f"Function '{function_node.name}' has an incomplete docstring: {issues}")
            return True
        else:
            log_info(f"Function '{function_node.name}' has a complete docstring.")
            return False

    def is_class_docstring_incomplete(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class has an incomplete docstring.

        Args:
            class_node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the docstring is incomplete, False otherwise.
        """
        log_debug(f"Checking if class docstring is incomplete for class: {class_node.name}")
        issues = analyze_code_element_docstring(class_node)
        if issues:
            log_info(f"Class '{class_node.name}' has an incomplete docstring: {issues}")
            return True

        log_info(f"Class '{class_node.name}' has a complete docstring.")
        return False
        
    def analyze_function_docstring(self, function_node: ast.FunctionDef) -> List[str]:
        """
        Analyze the docstring of a function for completeness and quality.

        Args:
            function_node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of issues found in the docstring.
        """
        issues = analyze_code_element_docstring(function_node)
        docstring_data = parse_docstring(ast.get_docstring(function_node) or "")

        issues.extend(check_parameter_descriptions(docstring_data, function_node))
        issues.extend(check_return_description(docstring_data, function_node))
        issues.extend(check_exception_details(docstring_data, function_node))

        return issues
```