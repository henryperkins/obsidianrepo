```python
# docstring_processor.py
"""
Docstring Processing Module

Handles parsing, validation, and formatting of docstrings with improved structure
and centralized processing.
"""

import ast
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

@dataclass
class DocstringData:
    """Structured representation of a docstring."""
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, Any]
    raises: List[Dict[str, Any]]

@dataclass
class DocstringMetrics:
    """Metrics related to a docstring."""
    length: int
    sections_count: int
    args_count: int
    cognitive_complexity: float
    completeness_score: float

@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: Optional[List['DocumentationSection']] = field(default_factory=list)
    source_code: Optional[str] = None
    tables: Optional[List[str]] = field(default_factory=list)  # Added for tables

class DocstringProcessor:
    """
    Processes docstrings by parsing, validating, and formatting them.
    """

    def __init__(self, min_length: Optional[Dict[str, int]] = None):
        """
        Initialize DocstringProcessor with optional minimum length requirements.

        Args:
            min_length (Optional[Dict[str, int]]): Minimum length requirements for docstring sections.
        """
        self.min_length = min_length or {
            'summary': 10,
            'description': 20
        }

    def extract(self, node: ast.AST) -> DocstringData:
        """
        Extract and process the docstring from an AST node.

        Args:
            node (ast.AST): The AST node to extract the docstring from.

        Returns:
            DocstringData: The extracted and processed docstring data.
        """
        try:
            raw_docstring = ast.get_docstring(node) or ""
            docstring_data = self.parse(raw_docstring)
            return docstring_data
        except Exception as e:
            logger.error(f"Error processing node: {e}")
            return DocstringData("", "", [], {}, [])

    def parse(self, docstring: str, style: str = 'google') -> DocstringData:
        """
        Parse a raw docstring into a structured format.

        Args:
            docstring (str): The raw docstring text.
            style (str): The docstring style to parse ('google', 'numpy', 'sphinx').

        Returns:
            DocstringData: The parsed docstring data.
        """
        try:
            if not docstring.strip():
                return DocstringData("", "", [], {}, [])

            # Use a third-party library like docstring_parser
            from docstring_parser import parse

            parsed = parse(docstring, style=style)

            args = [
                {
                    'name': param.arg_name,
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                }
                for param in parsed.params
            ]

            returns = {
                'type': parsed.returns.type_name if parsed.returns else 'Any',
                'description': parsed.returns.description if parsed.returns else ''
            }

            raises = [
                {
                    'exception': e.type_name or 'Exception',
                    'description': e.description or ''
                }
                for e in parsed.raises
            ] if parsed.raises else []

            return DocstringData(
                summary=parsed.short_description or '',
                description=parsed.long_description or '',
                args=args,
                returns=returns,
                raises=raises
            )
        except Exception as e:
            logger.error(f"Error parsing docstring: {e}")
            return DocstringData("", "", [], {}, [])

    def validate(self, docstring_data: DocstringData) -> Tuple[bool, List[str]]:
        """
        Validate the structured docstring data.

        Args:
            docstring_data (DocstringData): The docstring data to validate.

        Returns:
            Tuple[bool, List[str]]: Validation result and list of errors.
        """
        errors = []
        if len(docstring_data.summary) < self.min_length['summary']:
            errors.append("Summary is too short.")
        if len(docstring_data.description) < self.min_length['description']:
            errors.append("Description is too short.")
        if not docstring_data.args:
            errors.append("Arguments section is missing.")
        if not docstring_data.returns:
            errors.append("Returns section is missing.")
        is_valid = not errors
        return is_valid, errors

    def format(self, docstring_data: DocstringData) -> str:
        """
        Format structured docstring data into a docstring string.

        Args:
            docstring_data (DocstringData): The structured docstring data.

        Returns:
            str: The formatted docstring.
        """
        docstring_lines = []

        # Add summary
        if docstring_data.summary:
            docstring_lines.append(docstring_data.summary)
            docstring_lines.append("")

        # Add description
        if docstring_data.description:
            docstring_lines.append(docstring_data.description)
            docstring_lines.append("")

        # Add arguments
        if docstring_data.args:
            docstring_lines.append("Args:")
            for arg in docstring_data.args:
                docstring_lines.append(
                    f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}"
                )
            docstring_lines.append("")

        # Add returns
        if docstring_data.returns:
            docstring_lines.append("Returns:")
            docstring_lines.append(
                f"    {docstring_data.returns.get('type', 'Any')}: {docstring_data.returns.get('description', '')}"
            )
            docstring_lines.append("")

        # Add raises
        if docstring_data.raises:
            docstring_lines.append("Raises:")
            for exc in docstring_data.raises:
                docstring_lines.append(
                    f"    {exc.get('exception', 'Exception')}: {exc.get('description', '')}"
                )
            docstring_lines.append("")

        return "\n".join(docstring_lines).strip()

    def insert(self, node: ast.AST, docstring: str) -> bool:
        """
        Insert a docstring into an AST node.

        Args:
            node (ast.AST): The AST node to insert the docstring into.
            docstring (str): The docstring to insert.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        try:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                node.body.insert(0, docstring_node)  # Add docstring as the first body element
                return True
            else:
                logger.warning(f"Cannot insert docstring into node type: {type(node).__name__}")
                return False
        except Exception as e:
            logger.error(f"Error inserting docstring: {e}")
            return False

```