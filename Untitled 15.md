# Untitled 15

```python
import ast
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any, Set, Union, Tuple
import typing
from collections import defaultdict
import math
import re
import builtins
import importlib
import tokenize
import io
import json  # Imported json for potential CLI output

logger = logging.getLogger(__name__)

class CodeElementType(Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"

@dataclass
class SourceLocation:
    """Represents a location in source code."""
    file_path: Path
    start_line: int
    end_line: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None

@dataclass
class Parameter:
    """Represents a function/method parameter."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_optional: bool = False
    is_variadic: bool = False
    is_keyword_only: bool = False
    is_positional_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TypeInfo:
    """Represents type information."""
    raw_type: str
    resolved_type: Optional[Any] = None
    is_optional: bool = False
    is_union: bool = False
    union_types: List[str] = field(default_factory=list)
    is_generic: bool = False
    generic_args: List['TypeInfo'] = field(default_factory=list)
    
    @property
    def formatted_type(self) -> str:
        if self.is_union:
            return f"Union[{', '.join(self.union_types)}]"
        elif self.is_generic:
            args = ', '.join(arg.formatted_type for arg in self.generic_args)
            return f"{self.raw_type}[{args}]"
        return self.raw_type

@dataclass
class ComplexityMetrics:
    """Represents code complexity metrics."""
    cyclomatic: int = 1
    cognitive: int = 0
    halstead: Dict[str, float] = field(default_factory=dict)
    maintainability: float = 0.0
    lines_of_code: int = 0
    comment_ratio: float = 0.0

@dataclass
class FunctionInfo:
    """Represents information about a function."""
    name: str
    type: CodeElementType
    parameters: List[Parameter]
    return_type: Optional[TypeInfo]
    decorators: List[str]
    docstring: Optional[str]
    source_location: SourceLocation
    complexity: ComplexityMetrics
    is_async: bool = False
    is_generator: bool = False
    is_abstract: bool = False
    raises: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClassInfo:
    """Represents information about a class."""
    name: str
    bases: List[str]
    methods: Dict[str, FunctionInfo] = field(default_factory=dict)
    properties: Dict[str, FunctionInfo] = field(default_factory=dict)
    class_variables: Dict[str, TypeInfo] = field(default_factory=dict)
    instance_variables: Dict[str, TypeInfo] = field(default_factory=dict)
    decorators: List[str]
    docstring: Optional[str]
    source_location: SourceLocation
    complexity: ComplexityMetrics
    is_dataclass: bool = False
    is_abstract: bool = False
    interfaces: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleInfo:
    """Represents information about a module."""
    name: str
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    variables: Dict[str, TypeInfo] = field(default_factory=dict)
    imports: Dict[str, str] = field(default_factory=dict)
    docstring: Optional[str]
    source_location: SourceLocation
    complexity: ComplexityMetrics
    is_package: bool = False
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TypeHintProcessor:
    """Processes and validates type hints."""
    
    def __init__(self):
        self._type_cache: Dict[str, Any] = {}
        
    def process_type_hint(self, hint: Any) -> TypeInfo:
        """Process and normalize type hints."""
        if hint is None:
            return TypeInfo(raw_type='Any')
            
        if isinstance(hint, str):
            return self._process_string_hint(hint)
            
        try:
            if hasattr(hint, '__origin__'):  # Generic types
                origin = hint.__origin__
                args = hint.__args__
                
                if origin == Union:
                    return TypeInfo(
                        raw_type='Union',
                        is_union=True,
                        union_types=[self.process_type_hint(arg).raw_type for arg in args]
                    )
                    
                origin_name = origin.__name__
                return TypeInfo(
                    raw_type=origin_name,
                    is_generic=True,
                    generic_args=[self.process_type_hint(arg) for arg in args]
                )
                
            return TypeInfo(raw_type=hint.__name__ if hasattr(hint, '__name__') else str(hint))
            
        except Exception as e:
            logger.warning(f"Error processing type hint: {e}")
            return TypeInfo(raw_type='Any')

    def _process_string_hint(self, hint: str) -> TypeInfo:
        """Process string type hints."""
        try:
            # Handle forward references
            if hint in self._type_cache:
                return self._type_cache[hint]
                
            # Basic types
            if hint in typing.__all__:
                type_obj = getattr(typing, hint)
                return self.process_type_hint(type_obj)
                
            # Built-in types
            if hasattr(builtins, hint):
                return TypeInfo(raw_type=hint)
                
            # Complex type expressions
            compiled = ast.parse(hint, mode='eval').body
            return self._process_type_ast(compiled)
            
        except Exception as e:
            logger.warning(f"Error processing string type hint '{hint}': {e}")
            return TypeInfo(raw_type='Any')

    def _process_type_ast(self, node: ast.AST) -> TypeInfo:
        """Process type hint AST nodes."""
        if isinstance(node, ast.Name):
            return TypeInfo(raw_type=node.id)
            
        elif isinstance(node, ast.Subscript):
            base_type = self._process_type_ast(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = [self._process_type_ast(arg) for arg in node.slice.elts]
            else:
                args = [self._process_type_ast(node.slice)]
                
            base_type.is_generic = True
            base_type.generic_args = args
            return base_type
            
        return TypeInfo(raw_type='Any')

class ComplexityCalculator:
    """Calculates various code complexity metrics."""
    
    def calculate_complexity(self, node: ast.AST) -> ComplexityMetrics:
        """Calculate all complexity metrics for a node."""
        metrics = ComplexityMetrics()
        
        # Cyclomatic complexity
        metrics.cyclomatic = self._calculate_cyclomatic_complexity(node)
        
        # Cognitive complexity
        metrics.cognitive = self._calculate_cognitive_complexity(node)
        
        # Halstead metrics
        metrics.halstead = self._calculate_halstead_metrics(node)
        
        # Lines of code and comments
        metrics.lines_of_code, metrics.comment_ratio = self._calculate_code_metrics(node)
        
        # Maintainability index
        metrics.maintainability = self._calculate_maintainability_index(
            metrics.halstead.get('volume', 0),
            metrics.cyclomatic,
            metrics.lines_of_code
        )
        
        return metrics

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Control flow statements
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            # Exception handling
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            # Boolean operators
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # Conditional expressions
            elif isinstance(child, ast.IfExp):
                complexity += 1
                
        return complexity

    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        
        def process_node(node: ast.AST, level: int):
            nonlocal complexity
            
            # Add complexity based on nesting level
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += level + 1
                
            # Boolean operators
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
            # Exception handling
            elif isinstance(node, ast.ExceptHandler):
                complexity += level + 1
                
            # Recursively process child nodes
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    process_node(child, level + 1)
                else:
                    process_node(child, level)
                    
        process_node(node, 0)
        return complexity

    def _calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.operator):
                operators.add(type(child).__name__)
                total_operators += 1
            elif isinstance(child, ast.Name):
                operands.add(child.id)
                total_operands += 1
                
        # Calculate Halstead metrics
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = total_operators # Total operators
        N2 = total_operands  # Total operands
        
        try:
            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
            effort = volume * difficulty
            time = effort / 18
            bugs = volume / 3000
            
            return {
                'vocabulary': vocabulary,
                'length': length,
                'volume': volume,
                'difficulty': difficulty,
                'effort': effort,
                'time': time,
                'bugs': bugs
            }
        except:
            return {
                'vocabulary': 0,
                'length': 0,
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'time': 0,
                'bugs': 0
            }

    def _calculate_code_metrics(self, node: ast.AST) -> Tuple[int, float]:
        """Calculate lines of code and comment ratio."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return 0, 0.0
            
        try:
            source = ast.get_source_segment(node)
            if not source:
                return 0, 0.0
                
            # Count lines of code
            lines = source.splitlines()
            code_lines = len([line for line in lines if line.strip()])
            
            # Count comments
            comment_lines = 0
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_lines += 1
                    
            comment_ratio = comment_lines / code_lines if code_lines > 0 else 0
            return code_lines, comment_ratio
            
        except:
            return 0, 0.0

    def _calculate_maintainability_index(
        self,
        halstead_volume: float,
        cyclomatic_complexity: int,
        lines_of_code: int
    ) -> float:
        """Calculate maintainability index."""
        try:
            # Original maintainability index formula
            mi = (171 - 5.2 * math.log(halstead_volume) -
                  0.23 * cyclomatic_complexity -
                  16.2 * math.log(lines_of_code)) * 100 / 171
            return max(0, min(100, mi))
        except:
            return 0.0

class CodeAnalyzer(ast.NodeVisitor):
    """Main class for analyzing Python code."""
    
    def __init__(self):
        self.type_processor = TypeHintProcessor()
        self.complexity_calculator = ComplexityCalculator()
        self.current_module: Optional[ModuleInfo] = None
        self.current_class: Optional[ClassInfo] = None
        self.current_function: Optional[FunctionInfo] = None
        
    def analyze_file(self, file_path: Path) -> ModuleInfo:
        """Analyze a Python source file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source)
            self.current_module = ModuleInfo(
                name=file_path.stem,
                classes={},
                functions={},
                variables={},
                imports={},
                docstring=ast.get_docstring(tree),
                source_location=SourceLocation(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(source.splitlines())
                ),
                complexity=self.complexity_calculator.calculate_complexity(tree)
            )
            
            self.visit(tree)
            return self.current_module
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            raise

    def visit_ClassDef(self, node: ast.ClassDef):
        """Process class definitions."""
        previous_class = self.current_class
        
        try:
            # Create class info
            self.current_class = ClassInfo(
                name=node.name,
                bases=[ast.unparse(base) for base in node.bases],
                methods={},
                properties={},
                class_variables={},
                instance_variables={},
                decorators=[ast.unparse(dec) for dec in node.decorator_list],
                docstring=ast.get_docstring(node),
                source_location=SourceLocation(
                    file_path=self.current_module.source_location.file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    start_col=node.col_offset,
                    end_col=node.end_col_offset
                ),
                complexity=self.complexity_calculator.calculate_complexity(node)
            )
            
            # Process class body
            self.generic_visit(node)
            
            # Add class to module
            if self.current_module:
                self.current_module.classes[node.name] = self.current_class
                
        finally:
            self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process function definitions."""
        self._process_function(node, defaults=node.args.defaults)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Process async function definitions."""
        self._process_function(node, is_async=True, defaults=node.args.defaults)

    def _process_function(self, node, is_async: bool = False, defaults: List[ast.expr] = []):
        """Process function and method definitions."""
        previous_function = self.current_function
        
        try:
            # Process parameters
            parameters = self._process_parameters(node.args, defaults)
            
            # Process return type
            return_type = None
            if node.returns:
                return_type = self.type_processor.process_type_hint(
                    self._evaluate_annotation(node.returns)
                )
            
            # Create function info
            self.current_function = FunctionInfo(
                name=node.name,
                type=self._determine_function_type(node, is_async),
                parameters=parameters,
                return_type=return_type,
                decorators=[ast.unparse(dec) for dec in node.decorator_list],
                docstring=ast.get_docstring(node),
                source_location=SourceLocation(
                    file_path=self.current_module.source_location.file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    start_col=node.col_offset,
                    end_col=node.end_col_offset
                ),
                complexity=self.complexity_calculator.calculate_complexity(node),
                is_async=is_async,
                is_generator=self._is_generator(node)
            )
            
            # Process function body
            self.generic_visit(node)
            
            # Add function to appropriate container
            if self.current_class:
                if any(dec.id == 'property' for dec in node.decorator_list 
                      if isinstance(dec, ast.Name)):
                    self.current_class.properties[node.name] = self.current_function
                else:
                    self.current_class.methods[node.name] = self.current_function
            elif self.current_module:
                self.current_module.functions[node.name] = self.current_function
                
        finally:
            self.current_function = previous_function

    def _process_parameters(self, args: ast.arguments, defaults: List[ast.expr]) -> List[Parameter]:
        """Process function parameters, including default values."""
        parameters = []
        default_offset = len(args.args) - len(defaults)

        # Process positional-only parameters
        if hasattr(args, 'posonlyargs'):
            for arg in args.posonlyargs:
                parameters.append(self._create_parameter(arg, is_positional_only=True))

        # Process regular parameters
        for i, arg in enumerate(args.args):
            default_value = None
            if i >= default_offset:
                try:
                    default_value = ast.literal_eval(defaults[i - default_offset])
                except ValueError:
                    default_value = ast.unparse(defaults[i - default_offset])
            param = self._create_parameter(arg)
            param.default_value = default_value
            param.is_optional = default_value is not None
            parameters.append(param)

        # Process variadic positional parameter
        if args.vararg:
            parameters.append(self._create_parameter(args.vararg, is_variadic=True))

        # Process keyword-only parameters
        for arg in args.kwonlyargs:
            parameters.append(self._create_parameter(arg, is_keyword_only=True))
        
        # Process variadic keyword parameter
        if args.kwarg:
            parameters.append(self._create_parameter(args.kwarg, is_variadic=True, is_keyword_only=True))

        return parameters

    def _create_parameter(
        self,
        arg: ast.arg,
        is_positional_only: bool = False,
        is_keyword_only: bool = False,
        is_variadic: bool = False
    ) -> Parameter:
        """Create a Parameter object from an ast.arg node."""
        type_hint = None
        if arg.annotation:
            type_hint = self.type_processor.process_type_hint(
                self._evaluate_annotation(arg.annotation)
            )

        return Parameter(
            name=arg.arg,
            type_hint=type_hint.formatted_type if type_hint else None,
            is_optional=False,  # Will be updated when processing defaults
            is_variadic=is_variadic,
            is_keyword_only=is_keyword_only,
            is_positional_only=is_positional_only
        )

    def _evaluate_annotation(self, node: ast.AST) -> Any:
        """Evaluate a type annotation.
        Handles edge cases like string annotations and forward references.
        """
        try:
            annotation_str = ast.unparse(node)
            # Check for string annotations (forward references or complex types)
            if annotation_str.startswith("'") and annotation_str.endswith("'"):
                return annotation_str[1:-1]  # Return the string without quotes
            return eval(annotation_str, {
                **typing.__dict__,
                **builtins.__dict__
            })
        except (NameError, TypeError) as e:
            logger.debug(f"Could not evaluate annotation {ast.unparse(node)}: {e}")
            return ast.unparse(node)  # Return raw string if evaluation fails

    def _determine_function_type(self, node, is_async) -> CodeElementType:
        """Determine the function type from its context."""
        if self.current_class:
            if is_async:
                return CodeElementType.ASYNC_METHOD
            return CodeElementType.METHOD
        if is_async:
            return CodeElementType.ASYNC_FUNCTION
        return CodeElementType.FUNCTION

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Determine if a function is a generator."""
        for n in ast.walk(node):
            if isinstance(n, (ast.Yield, ast.YieldFrom)):
                return True
        return False

    def visit_Assign(self, node: ast.Assign):
        """Process assignments (without type hints)."""
        if self.current_module:  # Only process at the module level
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)  # Attempt to get literal value
                        type_info = TypeInfo(raw_type=type(value).__name__)  # Use its type
                    except ValueError:
                        type_info = TypeInfo(raw_type="Unknown")  # Unknown type if not a literal
                    self.current_module.variables[target.id] = type_info

    def visit_Try(self, node: ast.Try):
        """Process try-except blocks to capture 'raises' information."""
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name):
                if self.current_function:
                    self.current_function.raises.add(handler.type.id)
        self.generic_visit(node)

# The rest of the methods for the visitor pattern and other functionalities remain unchanged...

```
