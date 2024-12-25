```python
# docs.py
"""
Documentation Management Module

Handles docstring operations and documentation generation with improved structure
and centralized processing.
"""

import ast
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
import asyncio

from core.logger import LoggerSetup, log_debug, log_error, log_info
from core.docstring_processor import DocstringProcessor, DocumentationSection
from markdown_generator import MarkdownGenerator, MarkdownConfig
from code_extraction import CodeExtractor, ExtractionContext
from metrics import Metrics

logger = LoggerSetup.get_logger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, errors: List[str]) -> None:
        """
        Initialize ValidationError with a message and list of errors.

        Args:
            message (str): Error message.
            errors (List[str]): List of validation errors.
        """
        super().__init__(message)
        self.errors = errors


class DocumentationError(Exception):
    """Custom exception for documentation generation errors."""

    def __init__(self, message: str, details: Dict[str, Any]) -> None:
        """
        Initialize DocumentationError with a message and error details.

        Args:
            message (str): Error message.
            details (Dict[str, Any]): Additional error details.
        """
        super().__init__(message)
        self.details = details


@dataclass
class DocumentationContext:
    """Holds context for documentation generation."""
    source_code: str
    module_path: Optional[Path] = None
    include_source: bool = True
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class DocStringManager:
    """Manages docstring operations and documentation generation."""

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): Documentation generation context.
            cache (Optional[Any]): Optional cache implementation.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator(MarkdownConfig(include_source=True))
        self.code_extractor = CodeExtractor(ExtractionContext())
        self.metrics_calculator = Metrics()

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation for the current context.

        Returns:
            str: Generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            # Prepare documentation sections
            sections: List[DocumentationSection] = []

            # Module documentation
            module_section = self._create_module_section()
            sections.append(module_section)

            # Extract code elements
            extraction_result = self.code_extractor.extract_code(self.context.source_code)

            # Classes documentation
            if extraction_result.classes:
                classes_section = await self._create_classes_section(extraction_result.classes)
                sections.append(classes_section)

                class_methods_section = await self._create_class_methods_section(extraction_result.classes)
                sections.append(class_methods_section)

            # Functions documentation
            if extraction_result.functions:
                functions_section = await self._create_functions_section(extraction_result.functions)
                sections.append(functions_section)

            # Constants and Variables
            if extraction_result.constants:
                constants_section = self._create_constants_section(extraction_result.constants)
                sections.append(constants_section)

            # Recent Changes
            recent_changes_section = self._create_recent_changes_section()
            sections.append(recent_changes_section)

            # Source Code
            if self.context.include_source:
                source_code_section = self._create_source_code_section()
                sections.append(source_code_section)

            # Generate markdown
            markdown = self.markdown_generator.generate(sections)
            return markdown

        except Exception as e:
            log_error(f"Failed to generate documentation: {e}")
            raise DocumentationError(
                "Documentation generation failed",
                {'error': str(e)}
            )

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module-level documentation section matching the template.
        """
        module_name = self.context.module_path.stem if self.context.module_path else 'Unknown Module'
        file_path = self.context.module_path or 'Unknown Path'
        description = self.context.metadata.get('description', 'No description provided.')

        content = f"**File:** `{file_path}`\n**Description:** {description}\n"

        return DocumentationSection(
            title=f"# Module: {module_name}",
            content=content
        )

    async def _create_classes_section(self, classes: List[Any]) -> DocumentationSection:
        """
        Create the classes section with a table.
        """
        table_header = "| Class | Inherits From | Complexity Score* |\n|-------|---------------|-------------------|"
        table_rows = []

        for cls in classes:
            class_name = cls.name
            base_classes = cls.bases if cls.bases else ['object']
            complexity_score = await self._get_complexity_score(cls.node)
            row = f"| `{class_name}` | `{', '.join(base_classes)}` | {complexity_score} |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Classes",
            content=content
        )

    async def _create_class_methods_section(self, classes: List[Any]) -> DocumentationSection:
        """
        Create the class methods section with a table.
        """
        table_header = "| Class | Method | Parameters | Returns | Complexity Score* |\n|-------|--------|------------|---------|-------------------|"
        table_rows = []

        for cls in classes:
            class_name = cls.name
            for method in cls.methods:
                method_name = method.name
                parameters = self._get_parameters_signature(method.node)
                return_type = self._get_return_type(method.node)
                complexity_score = await self._get_complexity_score(method.node)
                row = f"| `{class_name}` | `{method_name}` | `{parameters}` | `{return_type}` | {complexity_score} |"
                table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="### Class Methods",
            content=content
        )

    async def _create_functions_section(self, functions: List[Any]) -> DocumentationSection:
        """
        Create the functions section with a table.
        """
        table_header = "| Function | Parameters | Returns | Complexity Score* |\n|----------|------------|---------|-------------------|"
        table_rows = []

        for func in functions:
            function_name = func.name
            parameters = self._get_parameters_signature(func.node)
            return_type = self._get_return_type(func.node)
            complexity_score = await self._get_complexity_score(func.node)
            row = f"| `{function_name}` | `{parameters}` | `{return_type}` | {complexity_score} |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Functions",
            content=content
        )

    def _create_constants_section(self, constants: List[Dict[str, Any]]) -> DocumentationSection:
        """
        Create the constants and variables section with a table.
        """
        table_header = "| Name | Type | Value |\n|------|------|-------|"
        table_rows = []

        for const in constants:
            name = const['name']
            type_ = const.get('type', 'Unknown')
            value = const.get('value', 'Unknown')
            row = f"| `{name}` | `{type_}` | `{value}` |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Constants and Variables",
            content=content
        )

    def _create_recent_changes_section(self) -> DocumentationSection:
        """
        Create the recent changes section.
        """
        # Placeholder for actual change logs
        changes = self.context.metadata.get('recent_changes', [
            "- [YYYY-MM-DD] Added feature X",
            "- [YYYY-MM-DD] Modified method Y",
            "- [YYYY-MM-DD] Fixed bug Z"
        ])

        content = "\n".join(changes)

        return DocumentationSection(
            title="## Recent Changes",
            content=content
        )

    def _create_source_code_section(self) -> DocumentationSection:
        """
        Create the source code section with code blocks.
        """
        source_code = self.context.source_code
        content = "```python\n" + source_code + "\n```"

        return DocumentationSection(
            title="## Source Code",
            content=content
        )

    async def _get_complexity_score(self, node: ast.AST) -> str:
        """
        Calculate the complexity score for a given node.
        """
        if isinstance(node, ast.FunctionDef):
            complexity = self.metrics_calculator.calculate_cyclomatic_complexity(node)
        elif isinstance(node, ast.ClassDef):
            complexities = [
                self.metrics_calculator.calculate_cyclomatic_complexity(n)
                for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            complexity = sum(complexities) / len(complexities) if complexities else 0
        else:
            complexity = 0

        # Add warning symbol if complexity is high
        warning_symbol = '⚠️' if complexity > 10 else ''
        return f"{int(complexity)} {warning_symbol}"

    def _get_parameters_signature(self, node: ast.FunctionDef) -> str:
        """
        Get the function or method parameters as a string.
        """
        params = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ast.unparse(arg.annotation) if arg.annotation else 'Any'
            params.append(f"{arg_name}: {arg_type}")
        return f"({', '.join(params)})"

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """
        Get the return type of a function or method.
        """
        return_type = ast.unparse(node.returns) if node.returns else 'Any'
        return return_type

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """
        Check if a function node is a method.

        Args:
            node (ast.FunctionDef): Function node to check.

        Returns:
            bool: True if the function is a method of a class, False otherwise.
        """
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The root AST node.
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)

```