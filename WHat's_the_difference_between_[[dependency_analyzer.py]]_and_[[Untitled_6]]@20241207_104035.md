---
epoch: 1733589635310
modelKey: o1-mini|openai
tags:
  - copilot-conversation
---

**user**: WHat's the difference between [dependency_analyzer.py](dependency_analyzer.py.md) and [Untitled 6](./Untitled%206.md)
[Timestamp: 2024/12/07 10:40:35]

**ai**: The key differences between `[[dependency_analyzer.py]]` and `[[Untitled 6]]` are as follows:

1. **Logger Setup:**
    - In `[[dependency_analyzer.py]]`, the logger is set up using `LoggerSetup.get_logger(__name__)`.
    - In `[[Untitled 6]]`, the logger is enhanced using `CorrelationLoggerAdapter` for `LoggerSetup.get_logger(__name__)`.

2. **Exception Logging:**
    - `Untitled 6` uses `exc_info=True` in logging exceptions to provide detailed traceback information, which is more thorough than the logging in `dependency_analyzer.py`.

3. **Functionality:**
    - The scripts are nearly identical in terms of functionality, as they both analyze and categorize dependencies using the Abstract Syntax Tree (AST), detect circular dependencies, and can generate dependency graphs.

4. **Code Maintenance and Consistency:**
    - `[[Untitled 6]]` seems to have more robust error handling and logging practices, e.g., ensuring traceback is logged with errors, which may aid debugging and maintenance.

These differences mainly revolve around logging practices and exception handling while maintaining the same core functionality of dependency analysis.
[Timestamp: 2024/12/07 10:40:40]

**user**: Enhance docstrings and fix types (if you can) in [code_extractor.py](code_extractor.py.md) [docs.py](docs.py.md) [response_parsing.py](response_parsing.py.md) [ai_service.py](ai_service.py.md) and [main.py](main.py.md) and [docstring_processor.py](docstring_processor.py.md) - starting with dependency_analyzer.py that  Ialready shared with you
[Timestamp: 2024/12/07 10:45:28]

**ai**: Let's enhance the docstrings and correct any types that we find in the provided modules. Starting with `dependency_analyzer.py` as you suggested, here is an example of how the docstrings can be improved:

### dependency_analyzer.py Enhanced Example

```python
"""
Dependency analysis module for Python source code.

This module provides functionality to analyze and categorize dependencies
within Python source code using the Abstract Syntax Tree (AST). It identifies
imports, function calls, and attributes used in the code, detects circular
dependencies, 
and can generate a visual dependency graph.

Functions:
    extract_dependencies_from_node: Extracts dependencies from an AST node.

Classes:
    DependencyAnalyzer: Analyzes and categorizes code dependencies.

Example usage:
    analyzer = DependencyAnalyzer(context)
    dependencies = analyzer.analyze_dependencies(ast_tree)
"""

import sysconfig
import ast
import importlib.util
import os
import sys
from typing import Dict, Set, Optional, List, Tuple, Union, Any

from core.logger import LoggerSetup
from core.types import ExtractionContext
from core.utils import NodeNameVisitor
from core.metrics import Metrics

logger = LoggerSetup.get_logger(__name__)


def extract_dependencies_from_node(node: ast.AST) -> Dict[str, Set[str]]:
    """Extract dependencies from an AST node.

    This function scans through the AST node, identifying and categorizing
    imports, function calls, and attribute accesses.

    Args:
        node (ast.AST): An AST node representing a portion of Python code.

    Returns:
        Dict[str, Set[str]]: A dictionary with three keys: 'imports',
        'calls', and 'attributes'. Each key maps to a set of strings
        representing the respective dependencies found in the node.
    """
    dependencies: Dict[str, Set[str]] = {"imports": set(), "calls": set(), "attributes": set()}
    for child in ast.walk(node):
        try:
            if isinstance(child, ast.Import):
                for name in child.names:
                    dependencies["imports"].add(name.name)
            elif isinstance(child, ast.ImportFrom) and child.module:
                dependencies["imports"].add(child.module)
            elif isinstance(child, ast.Call):
                visitor = NodeNameVisitor()
                visitor.visit(child.func)
                dependencies["calls"].add(visitor.name)
            elif isinstance(child, ast.Attribute):
                visitor = NodeNameVisitor()
                visitor.visit(child)
                dependencies["attributes"].add(visitor.name)
        except Exception as e:
            logger.warning(f"Unsupported AST node encountered: {e}")
    return dependencies


class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Analyzes a parsed AST to identify standard library, third-party,
    and local module dependencies, and evaluates their impact on
    maintainability.

    Attributes:
        context (ExtractionContext): The context used to provide necessary
            information during analysis.
        module_name (Optional[str]): Name of the module being analyzed.
        _function_errors (List[str]): Records any errors encountered during
            function metadata extraction.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The context used to provide necessary
                information during analysis operations.
        """
        self.logger = logger
        self.context = context
        self.module_name = context.module_name
        self._function_errors: List[str] = []
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Perform dependency analysis on an AST node.

        Args:
            node (ast.AST): The AST representing the code to analyze.
            module_name (Optional[str]): Name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: Categorized dependencies including the maintainability impact.
        """
        self.logger.info("Starting dependency analysis")
        
        if module_name:
            self.module_name = module_name
        elif not self.module_name:
            self.logger.warning("No module name provided for dependency analysis")
            self.module_name = "<unknown_module>"

        try:
            raw_deps = extract_dependencies_from_node(node)

            self.logger.debug(f"Raw dependencies extracted: {raw_deps}")

            if not isinstance(raw_deps, dict):
                raise ValueError("raw_deps is expected to be a dictionary")
            
            for key, value in raw_deps.items():
                if not isinstance(value, set):
                    raise ValueError(f"Dependencies for {key} must be a set, got {type(value)}")

            deps = self._categorize_dependencies(raw_deps)

            self.logger.debug(f"Categorized dependencies: {deps}")

            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} categories found")

            maintainability_impact = self._calculate_maintainability_impact(deps)
            deps["maintainability_impact"] = maintainability_impact

            return deps

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {"stdlib": set(), "third_party": set(), "local": set()}

    def extract_function_metadata(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract metadata from a function node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract metadata from.

        Returns:
            Dict[str, Any]: Metadata including name, docstring, raises, args, and returns information.
        """
        try:
            metadata = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "raises": self._extract_raises(node),
                "args": self._extract_args(node),
                "returns": self._extract_returns(node)
            }
            return metadata
        except Exception as e:
            self._function_errors.append(f"Error extracting metadata for {getattr(node, 'name', 'unknown')}: {e}")
            return {}

    def generate_complexity_graph(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a visual representation of code metrics.

        Args:
            metrics (Dict[str, Any]): A dictionary of code metrics to display.

        Returns:
            str: A base64-encoded string representation of the generated graph.
        """
        try:
            # Filter and flatten metrics
            filtered_metrics: Dict[str, float] = {}
            
            for key, value in metrics.items():
                if key == "halstead_metrics" and isinstance(value, dict):
                    # Include selected Halstead metrics
                    important_halstead = ["program_volume", "difficulty", "effort"]
                    for h_key in important_halstead:
                        if h_key in value and isinstance(value[h_key], (int, float)):
                            filtered_metrics[f"halstead_{h_key}"] = float(value[h_key])
                elif key in ["cyclomatic_complexity", "cognitive_complexity", 
                            "maintainability_index", "method_count"]:
                    if isinstance(value, (int, float)):
                        filtered_metrics[key] = float(value)

            if not filtered_metrics:
                self.logger.warning("No valid metrics for graph generation")
                return ""

            # Create plot
            plt.clf()  # Clear any existing plots
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort metrics by value for better visualization
            sorted_items = sorted(filtered_metrics.items(), key=lambda x: x[1])
            labels, values = zip(*sorted_items)

            # Create horizontal bar chart
            bars = ax.barh(labels, values, color='skyblue')
            
            # Customize plot
            ax.set_xlabel('Metric Value')
            ax.set_title('Code Complexity Metrics')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', 
                    ha='left', va='center', fontsize=8)

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Cleanup
            plt.close(fig)
            buf.close()

            return img_base64

        except Exception as e:
            self.logger.error(f"Error generating complexity graph: {e}")
            return ""
        finally:
            plt.close('all')  # Ensure all figures are closed

    # Other methods like _extract_raises, _extract_args, _extract_returns, 
    # _categorize_dependencies, etc., should also include detailed docstrings
```

**Note:** Each method's docstring now explicitly describes the input parameters, return types, and general functionality. This makes it much clearer how any given method or function is expected to be used or what it returns, aligning with practices for robust Python documentation.

Now, let's proceed with enhancing the docstrings for each provided module, focusing on clarity and completeness. Since there are multiple files, here are a few enhancements:

### code_extractor.py Enhancement

```python
"""
Code extraction module for Python source code analysis.

Provides functionality to extract various code elements and their metadata from Python
source code using the Abstract Syntax Tree (AST), identifying functions, classes,
variables, constants, and dependencies, and calculating maintainability indices.

Classes:
    CodeExtractor: Extracts code elements and metadata from Python source code.

Example usage:
    extractor = CodeExtractor(context)
    result = await extractor.extract_code(source_code)
"""

# Additional imports and code...

class CodeExtractor:
    """Extracts code elements and metadata from Python source code.

    Attributes:
        context (ExtractionContext): The context through which the extraction operations are informed.
        errors (List[str]): Records any errors encountered during extraction.
        metrics_calculator (Metrics): Used to calculate various code metrics.
        function_extractor (FunctionExtractor): Responsible for extracting functions.
        class_extractor (ClassExtractor): Responsible for extracting classes.
        dependency_analyzer (DependencyAnalyzer): Responsible for analyzing code dependencies.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor.

        Args:
            context (Optional[ExtractionContext]): The context for extraction operations.
        """
        # Initialization logic...

    async def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract code elements and metadata from source code.

        Parses and analyzes the AST of Python source code to extract classes,
        functions, variables, constants, and dependencies, providing metadata
        about each.

        Args:
            source_code (str): The Python source code to analyze.
            context (Optional[ExtractionContext]): The context for extraction.

        Returns:
            Optional[ExtractionResult]: An object containing details of the extracted elements and any errors.

        Raises:
            SyntaxError: Raised if there is a syntax error in the code.
            Exception: Raised in case of any unexpected error during extraction.
        """
        # Extraction logic...
```

### Additional Enhancements

Repeat similar logical steps for `docs.py`, `response_parsing.py`, `ai_interaction.py`, `main.py`, and `docstring_processor.py`, refining each class and method with explicit types, expected functionalities, error handling, and usage examples (if appropriate).

This approach aids users in understanding how to interact with each component of your codebase, potentially reducing misinterpretations and misuse. Additional refactoring can also be done, like ensuring all attributes and methods leverage Python type hints, enhancing the overall comprehensibility of the modules. If you want to proceed with the docstring enhancement of another specific file, feel free to mention it.
[Timestamp: 2024/12/07 10:45:56]

**ai**: To enhance the docstrings and fix types, I will assess the current state of the code provided (in parts) and describe changes based on the previous example `dependency_analyzer.py`. Here's an enhanced version reflecting similar improvements for these files:

### Enhanced Docstrings and Type Annotations:

#### `code_extractor.py`

```python
"""
Code extraction module for Python source code analysis.

This module provides functionality to extract various elements and metadata
from Python source code using the Abstract Syntax Tree (AST). It identifies
functions, classes, variables, constants, and dependencies, and calculates
the maintainability index. It integrates with other extractors to gather
comprehensive details and handles errors during extraction.

Classes:
    CodeExtractor: Extracts code elements and metadata from Python source code.

Example usage:
    extractor = CodeExtractor(context)
    result = await extractor.extract_code(source_code)
"""

import ast
from typing import Any, Optional, List, Dict
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import ExtractionContext, ExtractionResult
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.utils import handle_extraction_error, get_source_segment
from core.docstringutils import DocstringUtils
from core.utils import NodeNameVisitor

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)
logger = CorrelationLoggerAdapter(base_logger)

class CodeExtractor:
    """Extracts code elements and metadata from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction operations.
        errors (List[str]): List of errors encountered during extraction.
        metrics_calculator (Metrics): The metrics calculator for evaluating code metrics.
        function_extractor (FunctionExtractor): Extractor for functions.
        class_extractor (ClassExtractor): Extractor for classes.
        dependency_analyzer (DependencyAnalyzer): Analyzer for code dependencies.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor.

        Args:
            context (Optional[ExtractionContext]): The context for extraction operations.
        """
        self.logger = logger
        self.context = context or ExtractionContext()
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.dependency_analyzer = DependencyAnalyzer(self.context)

    async def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code.

        Args:
            source_code (str): The Python source code to analyze.
            context (Optional[ExtractionContext]): The context for extraction operations.

        Returns:
            Optional[ExtractionResult]: The result of the extraction, including extracted elements and errors.

        Raises:
            SyntaxError: If there is a syntax error in the source code.
            Exception: If an error occurs during extraction.
        """
        if context:
            self.context = context
        self.context.source_code = source_code

        self.logger.info("Starting code extraction for module.")
        self.context.module_name = self._extract_module_name(source_code)

        try:
            tree = ast.parse(source_code)
            self.context.tree = tree
            self._add_parent_references(tree)
            first_node = tree.body[0] if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
            docstring_info = DocstringUtils.extract_docstring_info(first_node) if first_node else {}
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(tree)

            result = ExtractionResult(
                module_docstring=docstring_info,
                maintainability_index=maintainability_index,
                classes=[],
                functions=[],
                variables=[],
                constants=[],
                dependencies={},
                errors=[],
            )

            await self._extract_elements(tree, result)

            self.logger.info("Code extraction completed successfully.")
            return result

        except SyntaxError as e:
            self.logger.error("Syntax error in source code: %s", e, exc_info=True)
            return ExtractionResult(module_docstring={}, errors=[f"Syntax error: {str(e)}"])
        except Exception as e:
            self.logger.error("Error extracting code: %s", e, exc_info=True)
            return ExtractionResult(module_docstring={}, errors=[f"Error extracting code: {str(e)}"])

    async def _extract_elements(self, tree: ast.AST, result: ExtractionResult) -> None:
        """Extract different code elements from the AST.

        Args:
            tree (ast.AST): The AST to extract elements from.
            result (ExtractionResult): Result object to store extracted elements.
        """
        try:
            result.classes = await self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Class extraction", e)
            result.errors.extend(self.errors)

        try:
            result.functions = await self.function_extractor.extract_functions(tree)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Function extraction", e)
            result.errors.extend(self.errors)

        try:
            result.variables = self._extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Variable extraction", e)
            result.errors.extend(self.errors)

        try:
            result.constants = self._extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Constant extraction", e)
            result.errors.extend(self.errors)

        try:
            result.dependencies = self.dependency_analyzer.analyze_dependencies(tree, self.context.module_name)
            self.logger.debug(f"Extracted {len(result.dependencies)} dependencies.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Dependency extraction", e)
            result.errors.extend(self.errors)

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST.

        Args:
            tree (ast.AST): The AST to extract variables from.

        Returns:
            List[Dict[str, Any]]: A list of extracted variables.
        """
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        visitor = NodeNameVisitor()
                        visitor.visit(getattr(node, 'annotation', ast.AST()))
                        var_info = {
                            "name": target.id,
                            "type": visitor.name or "Any",
                            "value": get_source_segment(self.context.source_code or "", getattr(node, 'value', ast.AST()))
                        }
                        variables.append(var_info)
        return variables

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract constants from the AST.

        Args:
            tree (ast.AST): The AST to extract constants from.

        Returns:
            List[Dict[str, Any]]: A list of extracted constants.
        """
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        const_info = {
                            "name": target.id,
                            "value": get_source_segment(self.context.source_code or "", getattr(node, 'value', ast.AST()))
                        }
                        constants.append(const_info)
        return constants

    def _add_parent_references(self, node: ast.AST) -> None:
        """Add parent references to AST nodes for accurate context extraction.

        Args:
            node (ast.AST): The AST node to add parent references to.
        """
        for child in ast.walk(node):
            for child_node in ast.iter_child_nodes(child):
                setattr(child_node, "parent", child)
                setattr(child_node, "module", getattr(node, "name", None))

    def _extract_module_name(self, source_code: str) -> str:
        """Extract the module name from the source code.

        Args:
            source_code (str): The source code to extract the module name from.

        Returns:
            str: The extracted module name.
        """
        # Implement logic to extract module name from source code
        return "default_module_name"
```

The rest of the files can be enhanced similarly, focusing on the following:
- Adding descriptions for all parameters and return types in docstrings.
- Clarifying any steps or exceptions handled within functions.
- Ensuring type hints are accurate and complete for parameters and return values.

If you need enhancements for any specific file among the given ones, let me know, and I will provide the improvements for it.
[Timestamp: 2024/12/07 10:46:00]

**user**: Update the other files, generating complete versions of each.
[Timestamp: 2024/12/07 10:47:20]

**ai**: Sure, I'll generate complete updated versions of each specified file, enhancing docstrings and ensuring type annotations are correct.

### Enhanced `docs.py`

```python
from typing import TYPE_CHECKING, Type, Union, Any, Optional, Tuple, Dict, List
from pathlib import Path
from datetime import datetime
import json

if TYPE_CHECKING:
    from ai_interaction import AIInteractionHandler
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import DocumentationContext, ExtractionContext, ExtractionResult
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.metrics import Metrics
from core.schema_loader import load_schema
from exceptions import DocumentationError

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)
logger = CorrelationLoggerAdapter(base_logger)

class DocumentationOrchestrator:
    """Orchestrates the process of generating documentation from source code files.

    Attributes:
        ai_handler (AIInteractionHandler): Handles AI interactions for generating docstrings.
        docstring_schema (Dict[str, Any]): Schema used for validating docstring formats.
        metrics (Metrics): Assists in calculating various code metrics.
        code_extractor (CodeExtractor): Parses and extracts code elements from source code.
        docstring_processor (DocstringProcessor): Processes and validates docstrings.
        response_parser (ResponseParsingService): Parses responses from the AI model.
    """

    def __init__(
        self,
        ai_handler: Optional['AIInteractionHandler'] = None,
        docstring_processor: Optional[DocstringProcessor] = None,
        code_extractor: Optional[CodeExtractor] = None,
        metrics: Optional[Metrics] = None,
        response_parser: Optional[ResponseParsingService] = None
    ) -> None:
        """Initialize the DocumentationOrchestrator.

        Args:
            ai_handler (Optional[AIInteractionHandler]): The AI handler for generating docstrings.
            docstring_processor (Optional[DocstringProcessor]): Manager for processing docstrings.
            code_extractor (Optional[CodeExtractor]): Extractor for code elements from source.
            metrics (Optional[Metrics]): Placeholder for handling different metrics.
            response_parser (Optional[ResponseParsingService]): Parser for AI model responses.

        Raises:
            ValueError: If ai_handler is not provided.
        """
        self.logger = logger
        if ai_handler is None:
            raise ValueError("ai_handler is required for DocumentationOrchestrator")
        self.ai_handler: 'AIInteractionHandler' = ai_handler
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")
        self.metrics: Metrics = metrics or Metrics()
        self.code_extractor: CodeExtractor = code_extractor or CodeExtractor(ExtractionContext())
        self.docstring_processor: DocstringProcessor = docstring_processor or DocstringProcessor(metrics=self.metrics)
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """Generate documentation for the given source code.

        Args:
            context (DocumentationContext): Holds information regarding the source code and its environment.

        Returns:
            Tuple[str, str]: The updated source code and generated documentation.

        Raises:
            DocumentationError: If code elements extraction or AI response validation fails.
        """
        try:
            self.logger.info("Starting documentation generation process")

            extraction_result: Optional[ExtractionResult] = await self.code_extractor.extract_code(context.source_code)
            if not extraction_result:
                self.logger.error("Failed to extract code elements", exc_info=True)
                raise DocumentationError("Code elements extraction failed")
            self.logger.info("Code extraction completed successfully")

            extraction_dict = extraction_result.to_dict()
            prompt: str = await self.ai_handler.create_dynamic_prompt(extraction_dict)
            self.logger.info("Prompt generated for AI model")

            ai_response: Union[str, Dict[str, Any]] = await self.ai_handler._interact_with_ai(prompt)
            self.logger.info("AI interaction completed successfully")

            parsed_response = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )
            if not parsed_response.validation_success:
                self.logger.error("Validation failed for AI response", exc_info=True)
                raise DocumentationError("AI response validation failed")
            self.logger.info("AI response parsed and validated successfully")

            updated_code, updated_documentation = await self.ai_handler._integrate_ai_response(
                parsed_response.content, extraction_result
            )
            self.logger.info("Docstring integration completed successfully")

            self.logger.info("Documentation generation completed successfully")
            return updated_code, updated_documentation

        except DocumentationError as de:
            self.logger.error(f"Documentation error: {de}", exc_info=True)
            raise
        except Exception as e:
            error_msg = f"Unexpected error during documentation generation: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentationError(error_msg)

    async def generate_module_documentation(self, file_path: Path, output_dir: Path) -> None:
        """Generate documentation for a single module.

        Args:
            file_path (Path): Path to the module file.
            output_dir (Path): Directory where the documentation will be output.

        Raises:
            DocumentationError: If there is an error during documentation generation.
        """
        try:
            self.logger.info(f"Generating documentation for {file_path}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.with_suffix(".md").name

            source_code = file_path.read_text(encoding="utf-8")
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat(),
                },
            )

            updated_code, documentation = await self.generate_documentation(context)
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")
            self.logger.info(f"Documentation written to {output_path}")

        except Exception as e:
            error_msg = f"Module documentation generation failed for {file_path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentationError(error_msg)

    async def generate_batch_documentation(self, file_paths: List[Path], output_dir: Path) -> Dict[Path, bool]:
        """Generate documentation for multiple files.

        Args:
            file_paths (List[Path]): List of file paths for which documentation should be generated.
            output_dir (Path): Directory where documentation results will be stored.

        Returns:
            Dict[Path, bool]: A dictionary mapping each file path to a boolean indicating success.
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                self.logger.error(f"Failed to generate docs for {file_path}: {e}", exc_info=True)
                results[file_path] = False
        return results

    def _generate_markdown(self, context: Dict[str, Any]) -> str:
        """Generate markdown documentation.

        Args:
            context (Dict[str, Any]): Context used to generate documentation.

        Returns:
            str: Generated markdown documentation.
        """
        return "Generated markdown documentation"

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        """Exit the async context manager."""
        pass
```

### Enhanced `response_parsing.py`

```python
"""
Response parsing service with consistent error handling and validation.

This module provides functionality for parsing AI responses, validating 
them against specified schemas, and managing parsing statistics.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.schema_loader import load_schema
from core.docstring_processor import DocstringProcessor
from core.types import ParsedResponse
from exceptions import ValidationError as CustomValidationError

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)
logger = CorrelationLoggerAdapter(base_logger)

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses.

    Attributes:
        docstring_processor (DocstringProcessor): Processes and validates docstring content.
        docstring_schema (Dict[str, Any]): Schema for validating docstring content.
        function_schema (Dict[str, Any]): Schema for validating function structures.
        _parsing_stats (Dict[str, int]): Tracks statistics about parsing processes.
    """

    def __init__(self) -> None:
        """Initialize the response parsing service."""
        self.logger = logger
        self.docstring_processor = DocstringProcessor()
        self.docstring_schema = load_schema("docstring_schema")
        self.function_schema = load_schema("function_tools_schema")
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }

    async def _parse_docstring_response(self, response: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse a docstring response, handling both string and dictionary inputs.

        Args:
            response (Union[str, Dict[str, Any]]): The response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed response content, if successful.
        """
        try:
            if isinstance(response, dict):
                self.logger.debug("Processing response as a dictionary")
                parsed_content = self.docstring_processor.parse(response)
                return parsed_content.__dict__ if parsed_content else None

            elif isinstance(response, str):
                self.logger.debug("Processing response as a string")
                response = response.strip()
                if response.startswith('{') and response.endswith('}'):
                    try:
                        parsed_dict = json.loads(response)
                        parsed_content = self.docstring_processor.parse(parsed_dict)
                        return parsed_content.__dict__ if parsed_content else None
                    except json.JSONDecodeError as json_error:
                        self.logger.warning(f"JSON decoding failed: {json_error}")

                parsed_content = self.docstring_processor.parse(response)
                return parsed_content.__dict__ if parsed_content else None

            self.logger.error(f"Unsupported response type: {type(response)}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to parse docstring response: {e}", exc_info=True)
            return None

    async def parse_response(self, response: Union[str, Dict[str, Any]], expected_format: str = "json", validate_schema: bool = True) -> ParsedResponse:
        """Parse and validate an AI response.

        Args:
            response (Union[str, Dict[str, Any]]): The AI response to parse.
            expected_format (str): The expected format of the response (e.g., "json", "markdown", "docstring").
            validate_schema (bool): Whether to validate the parsed response against a schema.

        Returns:
            ParsedResponse: An object containing the parsed content and metadata about the parsing process.
        """
        start_time = datetime.now()
        errors: List[str] = []
        parsed_content = None

        self._parsing_stats["total_processed"] += 1
        self.logger.info(f"Parsing response, expected format: {expected_format}")

        try:
            if expected_format == "json":
                parsed_content = await self._parse_json_response(response)
            elif expected_format == "markdown":
                parsed_content = await self._parse_markdown_response(response)
            elif expected_format == "docstring":
                parsed_content = await self._parse_docstring_response(response)
            else:
                raise ValueError(f"Unsupported format: {expected_format}")

            validation_success = False
            if parsed_content and validate_schema:
                validation_success = await self._validate_response(parsed_content, expected_format)
                if not validation_success:
                    errors.append("Schema validation failed")
                    self._parsing_stats["validation_failures"] += 1
                    parsed_content = self._create_fallback_response()

            if parsed_content:
                self._parsing_stats["successful_parses"] += 1
            else:
                self._parsing_stats["failed_parses"] += 1
                parsed_content = self._create_fallback_response()

            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Parsing completed in {processing_time} seconds")

            return ParsedResponse(
                content=parsed_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=validation_success,
                errors=errors,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_size": len(str(response)),
                },
            )

        except Exception as e:
            error_message = f"Response parsing failed: {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
            self._parsing_stats["failed_parses"] += 1
            
            return ParsedResponse(
                content=self._create_fallback_response(),
                format_type=expected_format,
                parsing_time=(datetime.now() - start_time).total_seconds(),
                validation_success=False,
                errors=errors,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_size": len(str(response)),
                    "error": str(e)
                },
            )

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response, handling code blocks and cleaning.

        Args:
            response (str): The JSON response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed content, if successful.
        """
        try:
            response = response.strip()
            if "```json" in response and "```" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if start > 7 and end > start:
                    response = response[start:end].strip()

            parsed_content = json.loads(response)

            # Ensuring required fields
            required_fields = {"summary", "description", "args", "returns", "raises"}
            for field in required_fields:
                if field not in parsed_content:
                    if field in {"args", "raises"}:
                        parsed_content[field] = []
                    elif field == "returns":
                        parsed_content[field] = {"type": "Any", "description": ""}
                    else:
                        parsed_content[field] = ""

            if not isinstance(parsed_content["args"], list):
                parsed_content["args"] = []
            if not isinstance(parsed_content["raises"], list):
                parsed_content["raises"] = []
            if not isinstance(parsed_content["returns"], dict):
                parsed_content["returns"] = {"type": "Any", "description": ""}

            self.logger.debug("JSON response parsed successfully")
            return parsed_content

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during JSON response parsing: {e}", exc_info=True)
            return None
        
    async def _validate_response(self, content: Dict[str, Any], format_type: str) -> bool:
        """Validate response against appropriate schema.

        Args:
            content (Dict[str, Any]): The content to validate.
            format_type (str): The format type that dictates which schema to use for validation.

        Returns:
            bool: True if validation is successful, otherwise False.
        """
        try:
            if format_type == "docstring":
                schema = self.docstring_schema["schema"]
                validate(instance=content, schema=schema)
            elif format_type == "function":
                validate(instance=content, schema=self.function_schema["schema"])
            self.logger.debug("Schema validation successful")
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e.message}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during schema validation: {e}", exc_info=True)
            return False

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when parsing fails.

        Returns:
            Dict[str, Any]: A default response indicating documentation generation failure.
        """
        self.logger.info("Creating fallback response due to parsing failure")
        return {
            "summary": "AI-generated documentation not available",
            "description": "Documentation could not be generated by AI service",
            "args": [],
            "returns": {"type": "Any", "description": "Return value not documented"},
            "raises": [],
            "complexity": 1,
        }

    async def _parse_markdown_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a markdown response, handling common formatting issues.

        Args:
            response (str): The markdown response to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed content, if successful.
        """
        try:
            response = response.strip()
            parsed_content = self._extract_markdown_sections(response)
            self.logger.debug("Markdown response parsed successfully")
            return parsed_content if parsed_content else None
        except Exception as e:
            self.logger.error(f"Failed to parse markdown response: {e}", exc_info=True)
            return None

    def _extract_markdown_sections(self, response: str) -> Dict[str, Any]:
        """Extract sections from a markdown response.

        Args:
            response (str): The markdown response to parse.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted sections.
        """
        sections = {}
        current_section = None
        current_content = []

        for line in response.splitlines():
            if line.startswith("#"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.strip("# ").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        self.logger.debug(f"Extracted markdown sections: {list(sections.keys())}")
        return sections
```

### Enhanced `ai_interaction.py`

```python
"""
This module provides classes and functions for handling AI interactions, processing source code,
generating dynamic prompts, and integrating AI-generated documentation back into the source code.

Classes:
    CustomJSONEncoder: Custom JSON encoder that can handle sets and other non-serializable types.
    AIInteractionHandler: Handles AI interactions for generating enriched prompts and managing responses.

Functions:
    serialize_for_logging(obj: Any) -> str: Safely serialize any object for logging purposes.
"""

import ast
import asyncio
import json
import re
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI

from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.metrics import Metrics
from core.response_parsing import ParsedResponse, ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext, ExtractionResult
from exceptions import ConfigurationError, ProcessingError

from logger import LoggerSetup, CorrelationLoggerAdapter

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
        """Override the default JSON encoding.

        Args:
            obj (Any): The object to encode.

        Returns:
            Any: The JSON-encoded version of the object or a string representation.
        """
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, types.ModuleType)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return super().default(obj)


def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging purposes.

    Args:
        obj (Any): The object to serialize.

    Returns:
        str: A JSON string representation of the object.
    """
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"


class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = CorrelationLoggerAdapter(base_logger)
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Optional[Cache] = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: Dict[str, Any] = docstring_schema or load_schema()

        try:
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.logger.info("AI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}")
            raise

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response (Union[str, Dict[str, Any]]): The response to truncate (either string or dictionary).
            length (int): Maximum length of the truncated response.

        Returns:
            str: Truncated string representation of the response.
        """
        try:
            if isinstance(response, dict):
                json_str = json.dumps(response, indent=2)
                return (json_str[:length] + "...") if len(json_str) > length else json_str
            elif isinstance(response, str):
                return (response[:length] + "...") if len(response) > length else response
            else:
                str_response = str(response)
                return (str_response[:length] + "...") if len(str_response) > length else str_response
        except Exception as e:
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Process the source code to extract metadata, interact with the AI, and integrate responses.

        Args:
            source_code (str): The source code to process.

        Returns:
            Optional[Dict[str, Any]]: The updated code and documentation if successful, otherwise None.
        """
        try:
            tree: ast.AST = ast.parse(source_code)
            context = ExtractionContext()
            context.source_code = source_code
            context.tree = tree

            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                match = re.search(r"Module:?\s*([^\n\.]+)", module_docstring)
                if match:
                    context.module_name = match.group(1).strip()

            extractor = CodeExtractor(context)
            extraction_result: Optional[ExtractionResult] = await extractor.extract_code(source_code)
            if not extraction_result:
                self.logger.error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            self.logger.debug("Generated prompt for AI")

            ai_response: Union[str, Dict[str, Any]] = await self._interact_with_ai(prompt)
            self.logger.debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                self.logger.error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                self.logger.error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            self.logger.error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.

        Args:
            ai_response (Dict[str, Any]): The AI-generated response to integrate.
            extraction_result (ExtractionResult): The extraction result containing the source code.

        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            self.logger.debug("Starting AI response integration")
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = self._create_processed_response(ai_response)

            integration_result = self._process_docstrings(processed_response, extraction_result.source_code)
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            if not isinstance(code, str):
                raise ProcessingError("Expected 'code' to be a string in integration result")

            documentation = self._generate_markdown_documentation(ai_response, extraction_result)
            return code, documentation

        except Exception as e:
            self.logger.error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """
        Generate markdown documentation from AI response and extraction result.

        Args:
            ai_response (Dict[str, Any]): The AI response data.
            extraction_result (ExtractionResult): The extraction result containing source code info.

        Returns:
            str: Generated markdown documentation.
        """
        markdown_gen = MarkdownGenerator()
        markdown_context: Dict[str, Any] = {
            "module_name": extraction_result.module_name,
            "file_path": extraction_result.file_path,
            "description": ai_response.get("description", ""),
            "classes": extraction_result.classes,
            "functions": extraction_result.functions,
            "constants": extraction_result.constants,
            "source_code": extraction_result.source_code,
            "ai_documentation": ai_response,
        }
        return markdown_gen.generate(markdown_context)

    def _ensure_required_fields(self, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the AI response has the required fields, and repair if necessary.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            Dict[str, Any]: The AI response with ensured fields.
        """
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            self.logger.error(f"AI response missing required fields: {missing}")

            for field in missing:
                if field == "args":
                    ai_response["args"] = []
                elif field == "returns":
                    ai_response["returns"] = {"type": "None", "description": ""}
                elif field == "raises":
                    ai_response["raises"] = []
                else:
                    ai_response[field] = ""
        return ai_response

    def _create_processed_response(self, ai_response: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Create a list with the processed response.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            List[Dict[str, Union[str, Dict[str, Any]]]]: Processed response for integration.
        """
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """
        Process the docstrings using DocstringProcessor.

        Args:
            processed_response (List[Dict[str, Union[str, Dict[str, Any]]]]): The processed response data.
            source_code (str): The source code to modify.

        Returns:
            Dict[str, Any]: The integration result with updated code and docstrings.
        """
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        self.logger.debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> Union[str, Dict[str, Any]]:
        """
        Interact with the AI model to generate responses.

        Args:
            prompt (str): The prompt to send to the AI model.

        Returns:
            Union[str, Dict[str, Any]]: The AI-generated response.
        """
        try:
            request_params: Dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            self.logger.debug("Sending request to AI")

            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[
                    {"role": "system", "content": "You are a Python documentation expert. Generate complete docstrings in Google format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=request_params.get('temperature', 0.7)
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            self.logger.debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                self.logger.debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse AI response as JSON: {e}")

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        self.logger.debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        self.logger.error(f"Failed to parse extracted JSON: {e2}")

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            self.logger.error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise Exception(f"Unexpected error processing request: {str(e)}")

    async def create_dynamic_prompt(self, extracted_info: Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> str:
        """
        Create a dynamic prompt for the AI model.

        Args:
            extracted_info (Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]): The extracted information for the prompt.

        Returns:
            str: The generated prompt.
        """
        try:
            self.logger.debug("Creating dynamic prompt")
            self.logger.debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

            prompt_parts: List[str] = [
                "Generate a complete Python documentation structure as a single JSON object.\n\n",
                "Required JSON structure:\n",
                "{\n",
                '  "summary": "One-line summary of the code",\n',
                '  "description": "Detailed description of functionality",\n',
                '  "args": [{"name": "param1", "type": "str", "description": "param description"}],\n',
                '  "returns": {"type": "ReturnType", "description": "return description"},\n',
                '  "raises": [{"exception": "ValueError", "description": "error description"}]\n',
                "}\n\n",
                "Code Analysis:\n"
            ]

            if extracted_info.get("module_docstring"):
                prompt_parts.append(f"Current Module Documentation:\n{extracted_info['module_docstring']}\n\n")

            if extracted_info.get("classes"):
                prompt_parts.append("Classes:\n")
                for cls in extracted_info["classes"]:
                    prompt_parts.append(f"- {cls['name']}\n")
                    if cls.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {cls['docstring']}\n")
                    if cls.get("methods"):
                        prompt_parts.append("  Methods:\n")
                        for method in cls["methods"]:
                            prompt_parts.append(f"    - {method['name']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("functions"):
                prompt_parts.append("Functions:\n")
                for func in extracted_info["functions"]:
                    prompt_parts.append(f"- {func['name']}\n")
                    if func.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {func['docstring']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("dependencies"):
                prompt_parts.append("Dependencies:\n")
                for dep_type, deps in extracted_info["dependencies"].items():
                    if dep_type == "maintainability_impact":
                        # Handle maintainability_impact separately
                        prompt_parts.append(f"- {dep_type}: {deps}\n")
                        continue

                    if not isinstance(deps, (list, set, tuple)):
                        self.logger.error(f"Non-iterable dependencies for {dep_type}: {deps} ({type(deps)})")
                        continue

                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            self.logger.debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            self.logger.error(f"Error creating prompt: {e}", exc_info=True)
            raise Exception(f"Error creating dynamic prompt: {str(e)}")

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name (str): Name of the function or class.
            is_class (bool): Whether the target is a class.
            params (Optional[List[Dict[str, Any]]]): List of parameters with their types and descriptions.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring to enhance.
            decorators (Optional[List[str]]): List of decorators applied to the function.
            exceptions (Optional[List[Dict[str, str]]]): List of exceptions raised by the function.

        Returns:
            Dict[str, Any]: Generated docstring content.
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info: Dict[str, Any] = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                stop=["END"],
            )
            response_content: Optional[str] = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            self.logger.info("Generated docstring for %s", func_name)
            return parsed_response.content

        except json.JSONDecodeError as e:
            self.logger.error(
                "JSON decoding error while generating docstring for %s: %s",
                func_name,
                e,
            )
            raise
        except ProcessingError as e:
            self.logger.error(
                "Processing error while generating docstring for %s: %s", func_name, e
            )
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error generating docstring for %s: %s", func_name, e
            )
            raise

    async def _verify_deployment(self) -> bool:
        """
        Verify that the configured deployment exists and is accessible.

        Returns:
            bool: True if the deployment exists and is accessible, otherwise False.
        """
        try:
            test_params: Dict[str, Any] = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }
            self.logger.debug(f"Verifying deployment with parameters: {test_params}")
            response = await self.client.chat.completions.create(**test_params)
            self.logger.debug(f"Deployment verification response: {response}")
            return True
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {e}", exc_info=True)
            return False

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Verifies the deployment configuration and raises a ConfigurationError if the deployment is not accessible.

        Returns:
            AIInteractionHandler: Initialized AIInteractionHandler instance.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.
        """
        self.logger.debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            self.logger.debug("Cache resources have been cleaned up")
        self.logger.info("AIInteractionHandler resources have been cleaned up")
```

### Enhanced `main.py`

```python
import argparse
import asyncio
import re
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

from ai_interaction import AIInteractionHandler
from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docs import DocumentationOrchestrator
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics, MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext
from core.utils import GitUtils
from exceptions import ConfigurationError, DocumentationError
from repository_handler import RepositoryHandler

load_dotenv()

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)
logger = CorrelationLoggerAdapter(base_logger)

class DocumentationGenerator:
    """Class responsible for generating documentation for Python files or repositories."""

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """Initialize the documentation generator.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
        """
        self.config = config or AzureOpenAIConfig.from_env()
        self.logger = logger
        self._initialized = False

        # Store components but don't initialize them yet
        self.api_client: Optional[APIClient] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.doc_orchestrator: Optional[DocumentationOrchestrator] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize components
            self.api_client = APIClient(
                config=self.config,
                response_parser=ResponseParsingService(),
                token_manager=TokenManager()
            )
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=Cache(),
                token_manager=TokenManager(),
                response_parser=ResponseParsingService(),
                metrics=Metrics(),
                docstring_schema=load_schema("docstring_schema")
            )
            self.doc_orchestrator = DocumentationOrchestrator(
                ai_handler=self.ai_handler,
                docstring_processor=DocstringProcessor(metrics=Metrics()),
                code_extractor=CodeExtractor(ExtractionContext()),
                metrics=Metrics(),
                response_parser=ResponseParsingService()
            )
            self.system_monitor = SystemMonitor()
            self.metrics_collector = MetricsCollector()

            # Start monitoring
            await self.system_monitor.start()
            self._initialized = True
            self.logger.info("All components initialized successfully")
        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            await self.cleanup()
            raise ConfigurationError(error_msg) from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file.

        Args:
            file_path (Path): Path to the file to process.
            output_path (Path): Path where the output should be saved.

        Returns:
            bool: True if processing was successful, otherwise False.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info(f"Processing file: {file_path}")

            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            try:
                source_code = file_path.read_text(encoding='utf-8')
                source_code = self._fix_indentation(source_code)
                if not source_code.strip():
                    self.logger.error(f"Empty file: {file_path}")
                    return False
            except UnicodeDecodeError:
                try:
                    source_code = file_path.read_text(encoding='latin-1')
                    source_code = self._fix_indentation(source_code)
                except Exception as e:
                    self.logger.error(f"Failed to read file {file_path}: {e}")
                    return False

            result = await self.ai_handler.process_code(source_code)
            if not result:
                self.logger.error(f"No documentation generated for {file_path}")
                return False

            updated_code = result.get('code')
            documentation = result.get('documentation')

            if not updated_code or not documentation:
                self.logger.error(f"Invalid result for {file_path}")
                return False

            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info(f"Successfully processed {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return False

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation in source code.

        Args:
            source_code (str): The source code to fix.

        Returns:
            str: The corrected source code with consistent indentation.
        """
        lines = source_code.splitlines()
        fixed_lines: List[str] = []

        for line in lines:
            # Convert tabs to spaces
            fixed_line = line.expandtabs(4)
            indent_count = len(fixed_line) - len(fixed_line.lstrip())
            if indent_count > 0:
                proper_indent = (indent_count // 4) * 4
                fixed_line = (" " * proper_indent) + fixed_line.lstrip()

            fixed_lines.append(fixed_line)

        return "\n".join(fixed_lines)

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation.

        Args:
            repo_path (str): The path to the repository to process.
            output_dir (Path): The output directory for documentation.

        Returns:
            bool: True if the repository was processed successfully, otherwise False.
        """
        try:
            if self._is_valid_url(repo_path):
                self.logger.info(f"Cloning repository from URL: {repo_path}")
                async with RepositoryHandler(repo_path=Path.cwd()) as repo_handler:
                    cloned_repo_path = await repo_handler.clone_repository(repo_path)
                    self.logger.info(f"Repository cloned to: {cloned_repo_path}")
                    success = await self._process_local_repository(cloned_repo_path, output_dir)
                    return success
            else:
                self.logger.info(f"Processing local repository at: {repo_path}")
                local_repo_path = Path(repo_path)
                if not local_repo_path.exists():
                    raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                success = await self._process_local_repository(local_repo_path, output_dir)
                return success
        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}")
            return False

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, otherwise False.
        """
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
            r'(?:\S+(?::\S*)?@)?'  # user and password
            r'(?:'
            r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])\.){3}'
            r'(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5])'  # IP address
            r'|'
            r'(?:(?:[a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,})'  # Domain name
            r')'
            r'(?::\d{2,5})?'  # Port
            r'(?:/\S*)?',
            re.IGNORECASE
        )
        return re.match(regex, url) is not None

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Process a local repository for documentation.

        Args:
            repo_path (Path): The path to the local repository.
            output_dir (Path): The output directory for documentation.

        Returns:
            bool: True if the local repository was processed successfully, otherwise False.
        """
        try:
            python_files = GitUtils.get_python_files(repo_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    self.logger.error(f"Failed to process file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error processing local repository: {e}")
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            collected_metrics = self.metrics_collector.get_metrics()
            system_metrics = self.system_monitor.get_metrics()

            print("=== Documentation Generation Metrics ===")
            for metric in collected_metrics['operations']:
                print(f"Operation: {metric['operation_type']}")
                print(f"Success: {metric['success']}")
                print(f"Duration: {metric['duration']} seconds")
                print(f"Usage: {metric['usage']}")
                print(f"Validation Success: {metric['validation_success']}")
                print(f"Timestamp: {metric['timestamp']}")
                print("-" * 40)

            print("=== System Performance Metrics ===")
            print(system_metrics)
            print("-" * 40)

        except Exception as e:
            self.logger.error(f"Error displaying metrics: {e}")
            print(f"Error displaying metrics: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            if self.api_client:
                await self.api_client.close()
            if self.ai_handler:
                await self.ai_handler.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.system_monitor:
                await self.system_monitor.stop()

            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Unexpected error during cleanup: {e}")

async def main(args: argparse.Namespace) -> int:
    """Main function to manage the documentation generation process based on arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    doc_generator = DocumentationGenerator()
    try:
        await doc_generator.initialize()

        if args.repository:
            success = await doc_generator.process_repository(args.repository, Path(args.output))
            if success:
                print("Repository documentation generated successfully.")
            else:
                print("Failed to generate repository documentation.")

        if args.files:
            for file in args.files:
                file_path = Path(file)
                output_path = Path(args.output) / file_path.stem
                success = await doc_generator.process_file(file_path, output_path)
                if success:
                    print(f"Documentation for {file} generated successfully.")
                else:
                    print(f"Failed to generate documentation for {file}.")

        await doc_generator.display_metrics()

    except DocumentationError as de:
        logger.error(f"Documentation generation failed: {de}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        await doc_generator.cleanup()

    return 0

def parse_arguments() -> argparse.Namespace:
    """Parse arguments from the command line.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="URL or local path of the repository to process",
    )
    parser.add_argument("--files", nargs="+", help="Python files to process")
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Documentation generation interrupted by user.")
    except Exception as e:
        logger.error(f"Failed to run documentation generator: {e}")
```

### Enhanced `docstring_processor.py`

```python
"""
Module for processing and validating docstrings.

This module provides utilities for parsing, validating, and formatting docstrings in Python code.
It supports parsing docstrings from various formats, validating them against a schema, integrating
them into source code, and generating updated source code from modified ASTs.

Classes:
    DocstringProcessor: Class for processing and validating docstrings.
"""

import ast
import json
from typing import Optional, Dict, Any, List, Union, Tuple

from docstring_parser import parse as parse_docstring, Docstring

from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError
from core.utils import FormattingUtils, ValidationUtils
from core.schema_loader import load_schema

try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to generate code from AST. "
        "Please install it using 'pip install astor'."
    ) from e

# Set up the logger
base_logger = LoggerSetup.get_logger(__name__)
logger = CorrelationLoggerAdapter(base_logger)

class DocstringProcessor:
    """
    Processes docstrings by parsing and validating them.

    This class provides methods to parse raw docstrings, validate them,
    integrate them into source code, and generate documentation.
    """

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """Initialize the DocstringProcessor.

        Args:
            metrics (Optional[Metrics]): The metrics instance for handling code metrics.
        """
        self.logger = logger
        self.metrics = metrics or Metrics()
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")

    def _validate_docstring_dict(self, docstring_dict: Dict[str, Any]) -> None:
        """Validate that required keys exist in the docstring dictionary.

        Args:
            docstring_dict (Dict[str, Any]): The docstring dictionary to validate.

        Raises:
            DocumentationError: If required keys are missing from the docstring dictionary.
        """
        required_keys = {'summary', 'description', 'args', 'returns', 'raises'}
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            raise DocumentationError(f"Docstring dictionary missing keys: {missing_keys}")

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a docstring from a string or dictionary.

        Args:
            docstring (Union[Dict[str, Any], str]): The docstring to parse.

        Returns:
            DocstringData: A structured representation of the parsed docstring.

        Raises:
            DocumentationError: If the docstring cannot be parsed.
        """
        try:
            if isinstance(docstring, dict):
                self._validate_docstring_dict(docstring)
                if 'complexity' not in docstring:
                    docstring['complexity'] = 1
                return self._create_docstring_data_from_dict(docstring)
            elif isinstance(docstring, str):
                docstring_str = docstring.strip()
                if docstring_str.startswith('{') and docstring_str.endswith('}'):
                    try:
                        doc_dict = json.loads(docstring_str)
                        if 'complexity' not in doc_dict:
                            doc_dict['complexity'] = 1
                        return self.parse(doc_dict)
                    except json.JSONDecodeError as e:
                        self.logger.error("JSON parsing error: %s", e, exc_info=True)
                        raise DocumentationError(f"Invalid JSON format in docstring: {e}") from e

                try:
                    parsed: Docstring = parse_docstring(docstring_str)
                    docstring_dict = {
                        'summary': parsed.short_description or '',
                        'description': parsed.long_description or '',
                        'args': [{'name': p.arg_name, 'type': p.type_name or 'Any', 'description': p.description or ''}
                                 for p in parsed.params],
                        'returns': {
                            'type': parsed.returns.type_name if parsed.returns else 'Any',
                            'description': parsed.returns.description if parsed.returns else ''
                        },
                        'raises': [{'exception': e.type_name, 'description': e.description}
                                   for e in (parsed.raises or [])],
                        'complexity': 1
                    }
                    return self._create_docstring_data_from_dict(docstring_dict)
                except Exception as e:
                    self.logger.error("Docstring parsing error: %s", e, exc_info=True)
                    raise DocumentationError(f"Failed to parse docstring: {e}") from e
            raise DocumentationError("Docstring must be either a dictionary or a string.")
        except DocumentationError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error in parse method: %s", e, exc_info=True)
            raise DocumentationError(f"Unexpected error during parsing: {e}") from e

    def _create_docstring_data_from_dict(self, docstring_dict: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from a dictionary representation.

        Args:
            docstring_dict (Dict[str, Any]): The dictionary containing docstring information.

        Returns:
            DocstringData: The structured docstring data.

        Raises:
            DocumentationError: If creation of DocstringData fails.
        """
        try:
            docstring_dict['args'] = docstring_dict.get('args', [])
            docstring_dict['raises'] = docstring_dict.get('raises', [])
            returns = docstring_dict.get('returns', {})
            if not isinstance(returns, dict):
                returns = {'type': 'Any', 'description': ''}
            if not returns.get('type'):
                returns['type'] = 'Any'
            if not returns.get('description'):
                returns['description'] = ''

            complexity = docstring_dict.get('complexity', 1)

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict['args'],
                returns=returns,
                raises=docstring_dict['raises'],
                complexity=complexity
            )
        except Exception as e:
            self.logger.error("Error creating DocstringData from dict: %s", e, exc_info=True)
            raise DocumentationError(f"Failed to create DocstringData from dict: {e}") from e

    def process_batch(self, doc_entries: List[Dict[str, Any]], source_code: str) -> Dict[str, str]:
        """Process a batch of docstring entries and integrate them into the source code.

        Args:
            doc_entries (List[Dict[str, Any]]): The docstring entries to process.
            source_code (str): The source code to integrate the docstrings into.

        Returns:
            Dict[str, str]: A dictionary containing the updated code and documentation.
        """
        try:
            tree = ast.parse(source_code)
            self.logger.debug(f"Processing {len(doc_entries)} documentation entries")

            processed_entries: List[Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    self.logger.debug(f"Processing entry: {entry}")

                    if not isinstance(entry, dict):
                        self.logger.error(f"Entry is not a dictionary: {type(entry)}", exc_info=True)
                        continue

                    if 'summary' in entry and 'name' not in entry:
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                                docstring = self.format(DocstringData(**entry))
                                processed_entries.append({
                                    'name': node.name,
                                    'docstring': docstring,
                                    'type': type(node).__name__
                                })
                                self.logger.debug(f"Created processed entry for {node.name}")
                                break
                    else:
                        if 'docstring' not in entry and 'summary' in entry:
                            entry['docstring'] = self.format(DocstringData(**entry))
                        processed_entries.append(entry)
                        self.logger.debug(f"Added entry with name: {entry.get('name')}")

                except Exception as e:
                    self.logger.error(f"Error processing entry: {e}", exc_info=True)
                    continue

            if not processed_entries:
                self.logger.error("No valid entries were processed", exc_info=True)
                return {'code': source_code, 'documentation': ""}

            updated_tree: Optional[ast.AST] = self._insert_docstrings(tree, processed_entries)
            if not updated_tree:
                self.logger.error("Failed to update AST with docstrings", exc_info=True)
                return {'code': source_code, 'documentation': ""}

            updated_code: Optional[str] = self._generate_code_from_ast(updated_tree)
            if not updated_code:
                self.logger.error("Failed to generate code from AST", exc_info=True)
                return {'code': source_code, 'documentation': ""}

            documentation: str = self._generate_documentation(processed_entries)
            if not documentation:
                self.logger.error("Failed to generate documentation", exc_info=True)
                return {'code': updated_code, 'documentation': ""}

            self.logger.info("Successfully processed batch", exc_info=True)
            return {'code': updated_code, 'documentation': documentation}

        except Exception as e:
            self.logger.error(f"Unexpected error processing batch: {e}", exc_info=True)
            return {'code': source_code, 'documentation': ""}

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """Generate markdown documentation from processed docstring entries.

        Args:
            doc_entries (List[Dict[str, Any]]): The processed docstring entries.

        Returns:
            str: The generated markdown documentation.
        """
        try:
            self.logger.debug(f"Generating documentation for {len(doc_entries)} entries")
            doc_parts: List[str] = ["# API Documentation\n\n"]

            module_entry = next((entry for entry in doc_entries if entry.get('name') == '__module__'), None)
            if module_entry:
                doc_parts.extend([
                    "## Module Overview\n\n",
                    f"{module_entry.get('docstring', '')}\n\n"
                ])

            class_entries = [entry for entry in doc_entries if entry.get('type') == 'ClassDef']
            if class_entries:
                doc_parts.append("## Classes\n\n")
                for entry in class_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            func_entries = [entry for entry in doc_entries if entry.get('type') in ('FunctionDef', 'AsyncFunctionDef')]
            if func_entries:
                doc_parts.append("## Functions\n\n")
                for entry in func_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            documentation: str = ''.join(doc_parts)
            self.logger.debug(f"Generated documentation length: {len(documentation)}", exc_info=True)
            return documentation

        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}", exc_info=True)
            return ""

    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> Optional[ast.AST]:
        """Insert docstrings into the AST at relevant locations for each entry.

        Args:
            tree (ast.AST): The AST representation of the code.
            doc_entries (List[Dict[str, Any]]): The list of processed docstrings and their locations.

        Returns:
            Optional[ast.AST]: The updated AST with docstrings inserted, or None on failure.
        """
        try:
            docstring_map: Dict[str, str] = {
                entry['name']: entry['docstring']
                for entry in doc_entries
                if 'name' in entry and 'docstring' in entry
            }

            self.logger.debug(f"Created docstring map with {len(docstring_map)} entries", exc_info=True)

            class DocstringInserter(ast.NodeTransformer):
                def visit_Module(self, node: ast.Module) -> ast.Module:
                    self.generic_visit(node)
                    if '__module__' in docstring_map and not ast.get_docstring(node):
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring_map['__module__']))
                        node.body.insert(0, docstring_node)
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                    return self.visit_FunctionDef(node)

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

            transformer = DocstringInserter()
            new_tree = transformer.visit(tree)
            return new_tree

        except Exception as e:
            self.logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return None

    def _generate_code_from_ast(self, tree: ast.AST) -> Optional[str]:
        """Generate source code from an AST.

        Args:
            tree (ast.AST): The AST representation of the code.

        Returns:
            Optional[str]: The generated source code, or None on failure.
        """
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(tree)
            else:
                import astor
                return astor.to_source(tree)
        except Exception as e:
            self.logger.error(f"Error generating code from AST: {e}", exc_info=True)
            return None

    def format(self, data: DocstringData) -> str:
        """Format the docstring data into a human-readable string.

        Args:
            data (DocstringData): The data object containing docstring information.

        Returns:
            str: The formatted docstring.
        """
        return FormattingUtils.format_docstring(data.__dict__)

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate the docstring data against the schema.

        Args:
            data (DocstringData): The data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing success flag and list of validation errors.
        """
        return ValidationUtils.validate_docstring(data.__dict__, self.docstring_schema)
```

These enhancements provide better documentation and type correctness across your modules. Let me know if you require more specific improvements or further explanations.
[Timestamp: 2024/12/07 10:52:23]