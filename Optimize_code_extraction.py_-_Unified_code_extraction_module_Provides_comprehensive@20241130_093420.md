---
epoch: 1732980860642
modelKey: gemini-exp-1121|google
tags:
  - copilot-conversation
---

**user**: Optimize:
"""
code_extraction.py - Unified code extraction module

Provides comprehensive code analysis and extraction functionality for Python
source code, including class and function extraction, metrics calculation, and
dependency analysis.
"""

import importlib.util
import ast
import parso
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
from core.logger import LoggerSetup
from core.metrics import Metrics
from exceptions import ExtractionError

logger = LoggerSetup.get_logger(__name__)  # Initialize logger at module level


@dataclass
class ExtractionContext:
    """Context for extraction operations."""
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    import_context: Optional[Dict[str, Set[str]]] = None
    metrics_enabled: bool = True
    include_source: bool = True
    max_line_length: int = 100
    include_private: bool = False
    include_metrics: bool = True


@dataclass
class ExtractedArgument:
    """Represents a function argument."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function."""
    return_type: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    body_summary: str = ""
    args: List[ExtractedArgument] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Contains the complete extraction results."""
    classes: List['ExtractedClass'] = field(default_factory=list)
    functions: List[ExtractedFunction] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    module_docstring: Optional[str] = None
    imports: Dict[str, Set[str]] = field(default_factory=dict)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedClass(ExtractedElement):
    """Represents an extracted class."""
    bases: List[str] = field(default_factory=list)
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    is_exception: bool = False
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    metaclass: Optional[str] = None


class CodeExtractor:
    """Unified code extraction functionality."""

    def __init__(self, context: Optional[ExtractionContext] = None):
        """Initialize the code extractor with optional context."""
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize self.logger
        self.context = context or ExtractionContext()
        self.metrics_calculator = Metrics()
        self.errors: List[str] = []
        self._current_class: Optional[ast.ClassDef] = None
        self._module_ast: Optional[ast.AST] = None  # Store module AST
        
    def extract_code(self, source_code: str) -> Optional[ExtractionResult]:  
        """  
        Extract all code elements and metadata from the source code.  
  
        Args:  
            source_code: The Python source code to analyze  
  
        Returns:  
            ExtractionResult containing all extracted information, or None  
            if extraction fails due to a syntax error.  
  
        Raises:  
            ExtractionError: If parsing or extraction fails  
        """  
        try:  
            required_imports = set()  
            try:  
                tree = ast.parse(source_code)  
                self._module_ast = tree  # Store module AST  
                self._add_parents(tree)  
  
                # Analyze required imports  
                required_imports = self._analyze_required_imports(tree)  
  
                # Perform extraction  
                classes = self._extract_classes(tree)  
                functions = self._extract_functions(tree)  
                variables = self._extract_variables(tree)  
                module_docstring = ast.get_docstring(tree)  
                imports = self._extract_imports(tree)  
                constants = self._extract_constants(tree)  
                metrics = self._calculate_module_metrics(tree)  
  
                return ExtractionResult(  
                    classes=classes,  
                    functions=functions,  
                    variables=variables,  
                    module_docstring=module_docstring,  
                    imports=imports,  
                    constants=constants,  
                    errors=self.errors,  
                    metrics=metrics  
                )  
  
            except SyntaxError as e:  
                error_msg = f"Syntax error in source code: {str(e)}"  
                self.logger.error(error_msg)  
                self.errors.append(error_msg)  
                # Return None or an empty ExtractionResult with the error  
                return None  # Or return ExtractionResult(errors=self.errors)  
  
        except Exception as e:  
            error_msg = f"Extraction failed: {str(e)}"  
            self.logger.error(error_msg)  
            raise ExtractionError(error_msg) from e 
        
    def _extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self.context.include_private and node.name.startswith('_'):
                    continue
                try:
                    self._current_class = node
                    extracted_class = self._process_class(node)
                    classes.append(extracted_class)
                    self.logger.debug(f"Extracted class: {extracted_class.name}")
                except Exception as e:
                    error_msg = f"Failed to extract class {node.name}: {str(e)}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
                finally:
                    self._current_class = None
        return classes

    def _extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """Extract top-level functions from the AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(getattr(node, 'parent', None), ast.Module):
                if not self.context.include_private and node.name.startswith('_'):
                    continue
                try:
                    extracted_function = self._process_function(node)
                    functions.append(extracted_function)
                    self.logger.debug(f"Extracted function: {extracted_function.name}")
                except Exception as e:
                    error_msg = f"Failed to extract function {node.name}: {str(e)}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
        return functions

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name):
                    var_name = target.id
                    var_type = ast.unparse(node.annotation) if node.annotation else 'Any'
                    var_value = ast.unparse(node.value) if node.value else 'Unknown'

                    variables.append({
                        'name': var_name,
                        'type': var_type,
                        'value': var_value
                    })
                    self.logger.debug(f"Extracted variable: {var_name}")
        return variables

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process a class definition node."""
        metrics = self._calculate_class_metrics(node)
        complexity_warnings = self._get_complexity_warnings(metrics)
        return ExtractedClass(
            name=node.name,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
            source=ast.unparse(node) if self.context.include_source else None,
            metrics=metrics,
            dependencies=self._extract_dependencies(node),
            bases=self._extract_bases(node),
            methods=[self._process_function(n) for n in node.body if isinstance(n, ast.FunctionDef)],
            attributes=self._extract_attributes(node),
            is_exception=self._is_exception_class(node),
            decorators=self._extract_decorators(node),
            instance_attributes=self._extract_instance_attributes(node),
            metaclass=self._extract_metaclass(node),
            complexity_warnings=complexity_warnings
        )

    def _process_function(self, node: ast.FunctionDef) -> ExtractedFunction:
        """Process a function definition node."""
        metrics = self._calculate_function_metrics(node)
        complexity_warnings = self._get_complexity_warnings(metrics)
        return ExtractedFunction(
            name=node.name,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
            source=ast.unparse(node) if self.context.include_source else None,
            metrics=metrics,
            dependencies=self._extract_dependencies(node),
            args=self._extract_arguments(node),
            return_type=self._get_return_type(node),
            is_method=self._is_method(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_generator=self._is_generator(node),
            is_property=self._is_property(node),
            complexity_warnings=complexity_warnings,
            decorators=self._extract_decorators(node),
            body_summary=self._get_body_summary(node),
            raises=self._extract_raises(node)
        )

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node."""
        dependencies = {
            'imports': self._extract_imports(node),
            'calls': self._extract_function_calls(node),
            'attributes': self._extract_attribute_access(node)
        }
        return dependencies

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract function calls from a node."""
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    calls.add(ast.unparse(child.func))
                except Exception as e:
                    self.logger.error(f"Failed to unparse function call: {e}")
        return calls

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """Extract attribute accesses from a node."""
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                try:
                    attributes.add(ast.unparse(child))
                except Exception as e:
                    self.logger.error(f"Failed to unparse attribute access: {e}")
        return attributes

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'method_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            'complexity': self.metrics_calculator.calculate_complexity(node),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
            'inheritance_depth': self._calculate_inheritance_depth(node)
        }

    def _calculate_function_metrics(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
            'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
            'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
            'parameter_count': len(node.args.args),
            'return_complexity': self._calculate_return_complexity(node)
        }

    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate module-level metrics."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'total_lines': len(ast.unparse(tree).splitlines()),
            'complexity': self.metrics_calculator.calculate_complexity(tree),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
            'halstead': self.metrics_calculator.calculate_halstead_metrics(tree)
        }

    # Helper methods

    def _extract_arguments(self, node: ast.FunctionDef) -> List[ExtractedArgument]:
        """Extract and process function arguments."""
        args = []
        defaults = node.args.defaults
        default_offset = len(node.args.args) - len(defaults)

        for i, arg in enumerate(node.args.args):
            default_index = i - default_offset
            default_value = None if default_index < 0 else ast.unparse(defaults[default_index])

            args.append(ExtractedArgument(
                name=arg.arg,
                type_hint=ast.unparse(arg.annotation) if arg.annotation else None,
                default_value=default_value,
                is_required=default_index < 0
            ))
        return args

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes."""
        return [ast.unparse(base) for base in node.bases]

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes."""
        attributes = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign):
                attributes.append({
                    'name': ast.unparse(child.target),
                    'type': ast.unparse(child.annotation) if child.annotation else None,
                    'value': ast.unparse(child.value) if child.value else None
                })
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': None,
                            'value': ast.unparse(child.value)
                        })
        return attributes

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from __init__ method."""
        init_method = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
        if not init_method:
            return []

        attributes = []
        for child in ast.walk(init_method):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes.append({
                            'name': target.attr,
                            'type': None,
                            'value': ast.unparse(child.value)
                        })
        return attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass if specified."""
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return ast.unparse(keyword.value)
        return None

    def _extract_raises(self, node: ast.FunctionDef) -> List[str]:
        """Extract raised exceptions from function body."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_node = child.exc
                # Handle different types of exceptions
                if isinstance(exc_node, ast.Call):
                    # The exception is being instantiated
                    exception_name = self._get_exception_name(exc_node.func)
                    raises.add(exception_name)
                elif isinstance(exc_node, (ast.Name, ast.Attribute)):
                    # The exception is raised directly
                    exception_name = self._get_exception_name(exc_node)
                    raises.add(exception_name)
        return list(raises)

    def _get_exception_name(self, node: ast.AST) -> str:
        """Extract the name of the exception from the node."""
        try:
            return ast.unparse(node)  # Requires Python 3.9+
        except AttributeError:
            # For older Python versions, manually reconstruct the name
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_exception_name(node.value)}.{node.attr}"
            else:
                return "Exception"

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value),
                            'type': type(ast.literal_eval(node.value)).__name__ if isinstance(node.value, ast.Constant) else None
                        })
                        self.logger.debug(f"Extracted constant: {target.id}")
        return constants

    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports."""
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, imports)

        return imports

    def _categorize_import(self, module_name: str, imports: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, local, or third-party."""
        if module_name.startswith('.'):
            imports['local'].add(module_name)
        elif module_name in self._get_stdlib_modules():
            imports['stdlib'].add(module_name)
        else:
            imports['third_party'].add(module_name)
        self.logger.debug(f"Categorized import '{module_name}'")

    @staticmethod
    def _get_stdlib_modules() -> Set[str]:
        """Get a set of standard library module names."""
        import sys
        return set(sys.stdlib_module_names)

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a method."""
        return bool(self._current_class) or any(
            isinstance(parent, ast.ClassDef)
            for parent in ast.walk(node)
            if hasattr(parent, 'body') and node in parent.body
        )

    def _is_generator(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a generator."""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False

    def _is_property(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a property."""
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        return any(
            base.id in {'Exception', 'BaseException'}
            for base in node.bases
            if isinstance(base, ast.Name)
        )

    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Get the return type annotation if present."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _get_body_summary(self, node: ast.FunctionDef) -> str:
        """Generate a summary of the function body."""
        body_lines = ast.unparse(node).split('\n')[1:]  # Skip the definition line
        if len(body_lines) > 5:
            return '\n'.join(body_lines[:5] + ['...'])
        return '\n'.join(body_lines)

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on complexity metrics."""
        warnings = []
        if metrics.get('cyclomatic_complexity', 0) > 10:
            warnings.append("High cyclomatic complexity")
        if metrics.get('cognitive_complexity', 0) > 15:
            warnings.append("High cognitive complexity")
        if metrics.get('maintainability_index', 100) < 20:
            warnings.append("Low maintainability index")
        return warnings

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class."""
        depth = 0
        bases = node.bases

        while bases:
            depth += 1
            new_bases = []
            for base in bases:
                if isinstance(base, ast.Name):
                    base_name = base.id
                    resolved_base = self._resolve_base_class(base_name)
                    if resolved_base:
                        new_bases.extend(resolved_base.bases)
                elif isinstance(base, ast.Attribute):
                    try:
                        base_name = ast.unparse(base)
                        resolved_base = self._resolve_base_class(base_name)
                        if resolved_base:
                            new_bases.extend(resolved_base.bases)
                    except Exception as e:
                        self.logger.error(f"Failed to resolve base class {base}: {e}")
            bases = new_bases

            return depth
    
    def _resolve_base_class(self, base_name: str) -> Optional[ast.ClassDef]:
        """Resolve a base class from the current module or imports."""
        # Check current module
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.ClassDef) and node.name == base_name:
                return node

            # Check imported classes
        import_map = self._get_import_map()
        if base_name in import_map:
            module_name = import_map[base_name]
            return self._resolve_external_class(module_name, base_name)

        return None

    def _get_import_map(self) -> Dict[str, str]:
        """Create a map of imported names to their modules."""
        import_map = {}
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    import_map[alias.asname or alias.name] = node.module
        return import_map

    def _resolve_external_class(self, module_name: str, class_name: str) -> Optional[ast.ClassDef]:
        """Resolve a class from an external module."""
        try:
            module_spec = importlib.util.find_spec(module_name)
            if module_spec is None or module_spec.origin is None:
                self.logger.error(f"Module {module_name} not found.")
                return None

            module_path = module_spec.origin

            try:
                with open(module_path, 'r', encoding='utf-8') as file:
                    module_source = file.read()
            except FileNotFoundError:
                self.logger.error(f"File not found for module {module_name} at path {module_path}.")
                return None
            except IOError as e:
                self.logger.error(f"IO error reading module {module_name} at path {module_path}: {e}")
                return None

            try:
                module_ast = ast.parse(module_source)
            except SyntaxError as e:
                self.logger.error(f"Syntax error parsing module {module_name}: {e}")
                return None

            # Search for the class definition
            for node in ast.walk(module_ast):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return node

        except Exception as e:
            self.logger.error(f"Unexpected error resolving class {class_name} from module {module_name}: {e}")

        return None

    def _calculate_return_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the complexity of return statements."""
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
        return return_count

    def _analyze_required_imports(self, tree: ast.AST) -> Set[str]:
        """
        Analyze the AST to determine required imports that might be missing.

        Args:
            tree: AST to analyze

        Returns:
            Set of required import names
        """
        required = set()

        for node in ast.walk(tree):
            # Check for datetime usage
            if isinstance(node, ast.Name) and node.id in {'datetime', 'timedelta'}:
                required.add('datetime')

            # Check for os.path usage
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'os':
                    required.add('os')
                elif isinstance(node.value, ast.Name) and node.value.id == 'sys':
                    required.add('sys')

            # Check for pathlib.Path usage
            elif isinstance(node, ast.Name) and node.id == 'Path':
                required.add('pathlib')

        return required

    def _add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)

    def _extract_decorators(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> List[str]:
        """Extract decorators from a function or class."""
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        self.logger.debug(f"Extracted decorators for {node.name}: {decorators}")
        return decorators

[Timestamp: 2024/11/30 09:34:20]

**user**: """
code_extraction.py - Unified code extraction module

Provides comprehensive code analysis and extraction functionality for Python
source code, including class and function extraction, metrics calculation, and
dependency analysis.
"""

import importlib.util
import ast
import parso
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
from core.logger import LoggerSetup
from core.metrics import Metrics
from exceptions import ExtractionError

logger = LoggerSetup.get_logger(__name__)  # Initialize logger at module level


@dataclass
class ExtractionContext:
    """Context for extraction operations."""
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    import_context: Optional[Dict[str, Set[str]]] = None
    metrics_enabled: bool = True
    include_source: bool = True
    max_line_length: int = 100
    include_private: bool = False
    include_metrics: bool = True


@dataclass
class ExtractedArgument:
    """Represents a function argument."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True


@dataclass
class ExtractedElement:
    """Base class for extracted code elements."""
    name: str
    lineno: int
    source: Optional[str] = None
    docstring: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractedFunction(ExtractedElement):
    """Represents an extracted function."""
    return_type: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    body_summary: str = ""
    args: List[ExtractedArgument] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Contains the complete extraction results."""
    classes: List['ExtractedClass'] = field(default_factory=list)
    functions: List[ExtractedFunction] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    module_docstring: Optional[str] = None
    imports: Dict[str, Set[str]] = field(default_factory=dict)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedClass(ExtractedElement):
    """Represents an extracted class."""
    bases: List[str] = field(default_factory=list)
    methods: List[ExtractedFunction] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    is_exception: bool = False
    instance_attributes: List[Dict[str, Any]] = field(default_factory=list)
    metaclass: Optional[str] = None


class CodeExtractor:
    """Unified code extraction functionality."""

    def __init__(self, context: Optional[ExtractionContext] = None):
        """Initialize the code extractor with optional context."""
        self.logger = LoggerSetup.get_logger(__name__)  # Initialize self.logger
        self.context = context or ExtractionContext()
        self.metrics_calculator = Metrics()
        self.errors: List[str] = []
        self._current_class: Optional[ast.ClassDef] = None
        self._module_ast: Optional[ast.AST] = None  # Store module AST
        
    def extract_code(self, source_code: str) -> Optional[ExtractionResult]:  
        """  
        Extract all code elements and metadata from the source code.  
  
        Args:  
            source_code: The Python source code to analyze  
  
        Returns:  
            ExtractionResult containing all extracted information, or None  
            if extraction fails due to a syntax error.  
  
        Raises:  
            ExtractionError: If parsing or extraction fails  
        """  
        try:  
            required_imports = set()  
            try:  
                tree = ast.parse(source_code)  
                self._module_ast = tree  # Store module AST  
                self._add_parents(tree)  
  
                # Analyze required imports  
                required_imports = self._analyze_required_imports(tree)  
  
                # Perform extraction  
                classes = self._extract_classes(tree)  
                functions = self._extract_functions(tree)  
                variables = self._extract_variables(tree)  
                module_docstring = ast.get_docstring(tree)  
                imports = self._extract_imports(tree)  
                constants = self._extract_constants(tree)  
                metrics = self._calculate_module_metrics(tree)  
  
                return ExtractionResult(  
                    classes=classes,  
                    functions=functions,  
                    variables=variables,  
                    module_docstring=module_docstring,  
                    imports=imports,  
                    constants=constants,  
                    errors=self.errors,  
                    metrics=metrics  
                )  
  
            except SyntaxError as e:  
                error_msg = f"Syntax error in source code: {str(e)}"  
                self.logger.error(error_msg)  
                self.errors.append(error_msg)  
                # Return None or an empty ExtractionResult with the error  
                return None  # Or return ExtractionResult(errors=self.errors)  
  
        except Exception as e:  
            error_msg = f"Extraction failed: {str(e)}"  
            self.logger.error(error_msg)  
            raise ExtractionError(error_msg) from e 
        
    def _extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self.context.include_private and node.name.startswith('_'):
                    continue
                try:
                    self._current_class = node
                    extracted_class = self._process_class(node)
                    classes.append(extracted_class)
                    self.logger.debug(f"Extracted class: {extracted_class.name}")
                except Exception as e:
                    error_msg = f"Failed to extract class {node.name}: {str(e)}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
                finally:
                    self._current_class = None
        return classes

    def _extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """Extract top-level functions from the AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(getattr(node, 'parent', None), ast.Module):
                if not self.context.include_private and node.name.startswith('_'):
                    continue
                try:
                    extracted_function = self._process_function(node)
                    functions.append(extracted_function)
                    self.logger.debug(f"Extracted function: {extracted_function.name}")
                except Exception as e:
                    error_msg = f"Failed to extract function {node.name}: {str(e)}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
        return functions

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name):
                    var_name = target.id
                    var_type = ast.unparse(node.annotation) if node.annotation else 'Any'
                    var_value = ast.unparse(node.value) if node.value else 'Unknown'

                    variables.append({
                        'name': var_name,
                        'type': var_type,
                        'value': var_value
                    })
                    self.logger.debug(f"Extracted variable: {var_name}")
        return variables

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process a class definition node."""
        metrics = self._calculate_class_metrics(node)
        complexity_warnings = self._get_complexity_warnings(metrics)
        return ExtractedClass(
            name=node.name,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
            source=ast.unparse(node) if self.context.include_source else None,
            metrics=metrics,
            dependencies=self._extract_dependencies(node),
            bases=self._extract_bases(node),
            methods=[self._process_function(n) for n in node.body if isinstance(n, ast.FunctionDef)],
            attributes=self._extract_attributes(node),
            is_exception=self._is_exception_class(node),
            decorators=self._extract_decorators(node),
            instance_attributes=self._extract_instance_attributes(node),
            metaclass=self._extract_metaclass(node),
            complexity_warnings=complexity_warnings
        )

    def _process_function(self, node: ast.FunctionDef) -> ExtractedFunction:
        """Process a function definition node."""
        metrics = self._calculate_function_metrics(node)
        complexity_warnings = self._get_complexity_warnings(metrics)
        return ExtractedFunction(
            name=node.name,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
            source=ast.unparse(node) if self.context.include_source else None,
            metrics=metrics,
            dependencies=self._extract_dependencies(node),
            args=self._extract_arguments(node),
            return_type=self._get_return_type(node),
            is_method=self._is_method(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_generator=self._is_generator(node),
            is_property=self._is_property(node),
            complexity_warnings=complexity_warnings,
            decorators=self._extract_decorators(node),
            body_summary=self._get_body_summary(node),
            raises=self._extract_raises(node)
        )

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node."""
        dependencies = {
            'imports': self._extract_imports(node),
            'calls': self._extract_function_calls(node),
            'attributes': self._extract_attribute_access(node)
        }
        return dependencies

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract function calls from a node."""
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    calls.add(ast.unparse(child.func))
                except Exception as e:
                    self.logger.error(f"Failed to unparse function call: {e}")
        return calls

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """Extract attribute accesses from a node."""
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                try:
                    attributes.add(ast.unparse(child))
                except Exception as e:
                    self.logger.error(f"Failed to unparse attribute access: {e}")
        return attributes

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'method_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            'complexity': self.metrics_calculator.calculate_complexity(node),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
            'inheritance_depth': self._calculate_inheritance_depth(node)
        }

    def _calculate_function_metrics(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
            'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
            'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
            'parameter_count': len(node.args.args),
            'return_complexity': self._calculate_return_complexity(node)
        }

    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate module-level metrics."""
        if not self.context.metrics_enabled:
            return {}

        return {
            'total_lines': len(ast.unparse(tree).splitlines()),
            'complexity': self.metrics_calculator.calculate_complexity(tree),
            'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
            'halstead': self.metrics_calculator.calculate_halstead_metrics(tree)
        }

    # Helper methods

    def _extract_arguments(self, node: ast.FunctionDef) -> List[ExtractedArgument]:
        """Extract and process function arguments."""
        args = []
        defaults = node.args.defaults
        default_offset = len(node.args.args) - len(defaults)

        for i, arg in enumerate(node.args.args):
            default_index = i - default_offset
            default_value = None if default_index < 0 else ast.unparse(defaults[default_index])

            args.append(ExtractedArgument(
                name=arg.arg,
                type_hint=ast.unparse(arg.annotation) if arg.annotation else None,
                default_value=default_value,
                is_required=default_index < 0
            ))
        return args

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes."""
        return [ast.unparse(base) for base in node.bases]

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes."""
        attributes = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign):
                attributes.append({
                    'name': ast.unparse(child.target),
                    'type': ast.unparse(child.annotation) if child.annotation else None,
                    'value': ast.unparse(child.value) if child.value else None
                })
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': None,
                            'value': ast.unparse(child.value)
                        })
        return attributes

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from __init__ method."""
        init_method = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
        if not init_method:
            return []

        attributes = []
        for child in ast.walk(init_method):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes.append({
                            'name': target.attr,
                            'type': None,
                            'value': ast.unparse(child.value)
                        })
        return attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass if specified."""
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return ast.unparse(keyword.value)
        return None

    def _extract_raises(self, node: ast.FunctionDef) -> List[str]:
        """Extract raised exceptions from function body."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_node = child.exc
                # Handle different types of exceptions
                if isinstance(exc_node, ast.Call):
                    # The exception is being instantiated
                    exception_name = self._get_exception_name(exc_node.func)
                    raises.add(exception_name)
                elif isinstance(exc_node, (ast.Name, ast.Attribute)):
                    # The exception is raised directly
                    exception_name = self._get_exception_name(exc_node)
                    raises.add(exception_name)
        return list(raises)

    def _get_exception_name(self, node: ast.AST) -> str:
        """Extract the name of the exception from the node."""
        try:
            return ast.unparse(node)  # Requires Python 3.9+
        except AttributeError:
            # For older Python versions, manually reconstruct the name
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_exception_name(node.value)}.{node.attr}"
            else:
                return "Exception"

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value),
                            'type': type(ast.literal_eval(node.value)).__name__ if isinstance(node.value, ast.Constant) else None
                        })
                        self.logger.debug(f"Extracted constant: {target.id}")
        return constants

    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports."""
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, imports)

        return imports

    def _categorize_import(self, module_name: str, imports: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, local, or third-party."""
        if module_name.startswith('.'):
            imports['local'].add(module_name)
        elif module_name in self._get_stdlib_modules():
            imports['stdlib'].add(module_name)
        else:
            imports['third_party'].add(module_name)
        self.logger.debug(f"Categorized import '{module_name}'")

    @staticmethod
    def _get_stdlib_modules() -> Set[str]:
        """Get a set of standard library module names."""
        import sys
        return set(sys.stdlib_module_names)

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a method."""
        return bool(self._current_class) or any(
            isinstance(parent, ast.ClassDef)
            for parent in ast.walk(node)
            if hasattr(parent, 'body') and node in parent.body
        )

    def _is_generator(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a generator."""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False

    def _is_property(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a property."""
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        return any(
            base.id in {'Exception', 'BaseException'}
            for base in node.bases
            if isinstance(base, ast.Name)
        )

    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Get the return type annotation if present."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _get_body_summary(self, node: ast.FunctionDef) -> str:
        """Generate a summary of the function body."""
        body_lines = ast.unparse(node).split('\n')[1:]  # Skip the definition line
        if len(body_lines) > 5:
            return '\n'.join(body_lines[:5] + ['...'])
        return '\n'.join(body_lines)

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on complexity metrics."""
        warnings = []
        if metrics.get('cyclomatic_complexity', 0) > 10:
            warnings.append("High cyclomatic complexity")
        if metrics.get('cognitive_complexity', 0) > 15:
            warnings.append("High cognitive complexity")
        if metrics.get('maintainability_index', 100) < 20:
            warnings.append("Low maintainability index")
        return warnings

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class."""
        depth = 0
        bases = node.bases

        while bases:
            depth += 1
            new_bases = []
            for base in bases:
                if isinstance(base, ast.Name):
                    base_name = base.id
                    resolved_base = self._resolve_base_class(base_name)
                    if resolved_base:
                        new_bases.extend(resolved_base.bases)
                elif isinstance(base, ast.Attribute):
                    try:
                        base_name = ast.unparse(base)
                        resolved_base = self._resolve_base_class(base_name)
                        if resolved_base:
                            new_bases.extend(resolved_base.bases)
                    except Exception as e:
                        self.logger.error(f"Failed to resolve base class {base}: {e}")
            bases = new_bases

            return depth
    
    def _resolve_base_class(self, base_name: str) -> Optional[ast.ClassDef]:
        """Resolve a base class from the current module or imports."""
        # Check current module
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.ClassDef) and node.name == base_name:
                return node

            # Check imported classes
        import_map = self._get_import_map()
        if base_name in import_map:
            module_name = import_map[base_name]
            return self._resolve_external_class(module_name, base_name)

        return None

    def _get_import_map(self) -> Dict[str, str]:
        """Create a map of imported names to their modules."""
        import_map = {}
        for node in ast.walk(self._module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    import_map[alias.asname or alias.name] = node.module
        return import_map

    def _resolve_external_class(self, module_name: str, class_name: str) -> Optional[ast.ClassDef]:
        """Resolve a class from an external module."""
        try:
            module_spec = importlib.util.find_spec(module_name)
            if module_spec is None or module_spec.origin is None:
                self.logger.error(f"Module {module_name} not found.")
                return None

            module_path = module_spec.origin

            try:
                with open(module_path, 'r', encoding='utf-8') as file:
                    module_source = file.read()
            except FileNotFoundError:
                self.logger.error(f"File not found for module {module_name} at path {module_path}.")
                return None
            except IOError as e:
                self.logger.error(f"IO error reading module {module_name} at path {module_path}: {e}")
                return None

            try:
                module_ast = ast.parse(module_source)
            except SyntaxError as e:
                self.logger.error(f"Syntax error parsing module {module_name}: {e}")
                return None

            # Search for the class definition
            for node in ast.walk(module_ast):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return node

        except Exception as e:
            self.logger.error(f"Unexpected error resolving class {class_name} from module {module_name}: {e}")

        return None

    def _calculate_return_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the complexity of return statements."""
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
        return return_count

    def _analyze_required_imports(self, tree: ast.AST) -> Set[str]:
        """
        Analyze the AST to determine required imports that might be missing.

        Args:
            tree: AST to analyze

        Returns:
            Set of required import names
        """
        required = set()

        for node in ast.walk(tree):
            # Check for datetime usage
            if isinstance(node, ast.Name) and node.id in {'datetime', 'timedelta'}:
                required.add('datetime')

            # Check for os.path usage
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'os':
                    required.add('os')
                elif isinstance(node.value, ast.Name) and node.value.id == 'sys':
                    required.add('sys')

            # Check for pathlib.Path usage
            elif isinstance(node, ast.Name) and node.id == 'Path':
                required.add('pathlib')

        return required

    def _add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)

    def _extract_decorators(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> List[str]:
        """Extract decorators from a function or class."""
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        self.logger.debug(f"Extracted decorators for {node.name}: {decorators}")
        return decorators

[Timestamp: 2024/11/30 09:34:38]