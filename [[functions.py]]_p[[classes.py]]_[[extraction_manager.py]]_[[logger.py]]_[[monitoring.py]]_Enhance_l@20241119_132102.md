---
epoch: 1732044062411
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [functions.py](functions.py.md)*p[classes.py](classes.py.md)*[extraction_manager.py](extraction_manager.py.md)*[logger.py](logger.py.md)*[monitoring.py](monitoring.py.md)_Enhance_l@20241119_132102

**user**: [functions.py](simple2/functions.py.md) p[classes.py](simple2/classes.py.md) [extraction_manager.py](simple2/extraction_manager.py.md) [logger.py](simple2/logger.py.md) [monitoring.py](simple2/monitoring.py.md) Enhance logging and debugging in the extraction modules.  
[Timestamp: 2024/11/19 13:21:02]

**ai**: To enhance logging and debugging in the extraction modules, we can add more detailed logging statements that provide insights into the flow of data and the operations being performed. Here’s how you can improve the logging in each of the provided modules:

## [functions.py](simple2/functions.py.md)

1. **Add Entry and Exit Logs**: Log the entry and exit of each method to trace the flow of execution.
2. **Log Key Variable States**: Log important variable states and decisions within methods.

```python
import ast
from typing import List, Dict, Any, Optional, Tuple
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from utils import handle_exceptions

class FunctionExtractor(BaseExtractor):
    """
    Extract function definitions and their metadata from Python source code.
    """

    @handle_exceptions(log_error)
    def extract_functions(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all function definitions and their metadata from the source code.
        """
        log_debug("Entered extract_functions method.")
        if source_code:
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of function definitions.")
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                function_info = self.extract_details(node)
                if function_info:
                    functions.append(function_info)
                    log_info(f"Extracted function '{node.name}' with metadata.")
        log_debug(f"Total functions extracted: {len(functions)}")
        log_debug("Exiting extract_functions method.")
        return functions

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a function definition node.
        """
        log_debug(f"Entered extract_details for node: {node}")
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(f"Expected FunctionDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for function: {node.name}")
        details = self._extract_common_details(node)
        details.update({
            "args": self.extract_parameters(node),
            "return_type": self.extract_return_type(node),
            "decorators": self._extract_decorators(node),
            "exceptions": self._detect_exceptions(node),
            "body_summary": self.get_body_summary(node),
        })
        log_debug(f"Successfully extracted details for function {node.name}")
        log_debug(f"Exiting extract_details for function: {node.name}")
        return details

    # Other methods remain unchanged, but ensure to add similar logging where necessary
```

## [classes.py](simple2/classes.py.md)

1. **Log Method Entry and Exit**: Similar to `functions.py`, log entry and exit points.
2. **Log Important Decisions**: Log decisions like class type checks and metadata extraction.

```python
import ast
from typing import List, Dict, Any, Optional
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from utils import handle_exceptions

class ClassExtractor(BaseExtractor):
    """
    Extract class definitions and their metadata from Python source code.
    """

    @handle_exceptions(log_error)
    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions and their metadata from the source code.
        """
        log_debug("Entered extract_classes method.")
        if source_code:
            log_debug("Parsing source code for class extraction.")
            self.parse_source(source_code)

        if not self.tree:
            log_error("No AST available for extraction")
            return []

        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_info = self.extract_details(node)
                if class_info:
                    classes.append(class_info)
                    log_info(f"Extracted class '{node.name}' with metadata")
        log_debug(f"Total classes extracted: {len(classes)}")
        log_debug("Exiting extract_classes method.")
        return classes

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a class definition node.
        """
        log_debug(f"Entered extract_details for class node: {node}")
        if not isinstance(node, ast.ClassDef):
            log_error(f"Expected ClassDef node, got {type(node).__name__}")
            return {}

        log_debug(f"Extracting details for class: {node.name}")
        details = self._extract_common_details(node)
        details.update({
            "bases": [ast.unparse(base) for base in node.bases],
            "methods": [self.extract_method_details(n) for n in node.body if isinstance(n, ast.FunctionDef)],
            "attributes": self.extract_class_attributes(node),
        })
        log_debug(f"Successfully extracted details for class {node.name}")
        log_debug(f"Exiting extract_details for class: {node.name}")
        return details

    # Other methods remain unchanged, but ensure to add similar logging where necessary
```

## [extraction_manager.py](simple2/extraction_manager.py.md)

1. **Log Start and End of Major Operations**: Log the start and end of metadata extraction.
2. **Log Exception Handling**: Log exceptions with detailed context.

```python
import ast
from typing import Dict, Any, Optional, List, Tuple
from core.logger import log_info, log_error, log_debug
from metrics import Metrics

class ExtractionManager:
    """
    Enhanced extraction manager with support for exception classes and robust error handling.
    """

    def __init__(self):
        """Initialize the extraction manager with a metrics calculator."""
        log_debug("Initializing ExtractionManager.")
        self.metrics = Metrics()

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code with special handling for exception classes.
        """
        log_debug("Starting metadata extraction.")
        try:
            tree = ast.parse(source_code)
            self._add_parent_info(tree)
            classes, functions = [], []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    try:
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            classes.append(metadata)
                        log_debug(f"Extracted {'exception ' if is_exception else ''}class: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting class metadata for {getattr(node, 'name', '<unknown>')}: {e}")
                elif isinstance(node, ast.FunctionDef):
                    try:
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            functions.append(metadata)
                        log_debug(f"Extracted function: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting function metadata for {getattr(node, 'name', '<unknown>')}: {e}")

            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {'classes': classes, 'functions': functions}

        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': []}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': []}

    # Other methods remain unchanged, but ensure to add similar logging where necessary
```

## General Improvements

- **Consistent Log Levels**: Ensure that debug logs are used for tracing execution, info logs for successful operations, and error logs for exceptions.
- **Detailed Error Logs**: Include context in error logs to make debugging easier.
- **Performance Metrics**: Consider logging performance metrics such as execution time for critical methods.

By enhancing logging in this way, you can better trace the execution flow, quickly identify issues, and understand the state of the application at various points in time.  
[Timestamp: 2024/11/19 13:21:57]

**user**: Generate the complete functions.py with updated and enhanced logging throughout the module.  
[Timestamp: 2024/11/19 13:22:03]

**ai**: Here's the updated `functions.py` module with enhanced logging to improve traceability and debugging:

```python
"""
Function Extraction Module

This module provides functionality to extract function definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as parameters, return types, and docstrings.

Version: 1.0.2
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from utils import handle_exceptions

class FunctionExtractor(BaseExtractor):
    """
    Extract function definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about functions, including parameters, return types, and docstrings.
    """

    @handle_exceptions(log_error)
    def extract_functions(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all function definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing function metadata.
        """
        log_debug("Entered extract_functions method.")
        if source_code:
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of function definitions.")
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                log_debug(f"Found function definition: {node.name}")
                function_info = self.extract_details(node)
                if function_info:
                    functions.append(function_info)
                    log_info(f"Extracted function '{node.name}' with metadata.")
        log_debug(f"Total functions extracted: {len(functions)}")
        log_debug("Exiting extract_functions method.")
        return functions

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a function definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            dict: A dictionary containing function details.
        """
        log_debug(f"Entered extract_details for node: {node}")
        if not isinstance(node, ast.FunctionDef):
            log_error(f"Expected FunctionDef node, got {type(node).__name__}")
            raise ValueError(f"Expected FunctionDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for function: {node.name}")
        details = self._extract_common_details(node)
        details.update({
            "args": self.extract_parameters(node),
            "return_type": self.extract_return_type(node),
            "decorators": self._extract_decorators(node),
            "exceptions": self._detect_exceptions(node),
            "body_summary": self.get_body_summary(node),
        })
        log_debug(f"Successfully extracted details for function {node.name}")
        log_debug(f"Exiting extract_details for function: {node.name}")
        return details

    def extract_parameters(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of tuples containing parameter names and types.
        """
        log_debug(f"Extracting parameters for function: {node.name}")
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_type_annotation(arg)
            parameters.append((param_name, param_type))
            log_debug(f"Extracted parameter: {param_name} with type: {param_type}")
        return parameters

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        log_debug(f"Extracting return type for function: {node.name}")
        return_type = ast.unparse(node.returns) if node.returns else "Any"
        log_debug(f"Extracted return type: {return_type}")
        return return_type

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        log_debug(f"Generating body summary for function: {node.name}")
        body_summary = " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."
        log_debug(f"Generated body summary: {body_summary}")
        return body_summary

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """
        Extract decorators from a function definition.

        Args:
            node (ast.FunctionDef): The function node to extract decorators from.

        Returns:
            list: A list of decorator names.
        """
        log_debug(f"Extracting decorators for function: {node.name}")
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        log_debug(f"Extracted decorators: {decorators}")
        return decorators

    def _detect_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """
        Detect exceptions raised within a function.

        Args:
            node (ast.FunctionDef): The function node to analyze.

        Returns:
            list: A list of exception names.
        """
        log_debug(f"Detecting exceptions for function: {node.name}")
        exceptions = []
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Raise):
                if isinstance(sub_node.exc, ast.Name):
                    exceptions.append(sub_node.exc.id)
                elif isinstance(sub_node.exc, ast.Call) and isinstance(sub_node.exc.func, ast.Name):
                    exceptions.append(sub_node.exc.func.id)
        log_debug(f"Detected exceptions: {exceptions}")
        return exceptions

    def _extract_common_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract common details from a function node.

        Args:
            node (ast.FunctionDef): The function node to extract common details from.

        Returns:
            dict: A dictionary containing common function details.
        """
        log_debug(f"Extracting common details for function: {node.name}")
        common_details = {
            "name": node.name,
            "docstring": ast.get_docstring(node) or '',
            "lineno": node.lineno,
        }
        log_debug(f"Extracted common details: {common_details}")
        return common_details

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """
        Get the type annotation of a function argument.

        Args:
            arg (ast.arg): The argument node to extract the type annotation from.

        Returns:
            str: The type annotation as a string.
        """
        log_debug(f"Getting type annotation for argument: {arg.arg}")
        type_annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        log_debug(f"Type annotation for {arg.arg}: {type_annotation}")
        return type_annotation
```

## Key Enhancements

- **Entry and Exit Logs**: Added logging at the entry and exit points of each method to trace the flow of execution.
- **Detailed Variable Logging**: Logged key variable states and decisions, such as parameter extraction and exception detection.
- **Consistent Log Levels**: Used `log_debug` for tracing execution and `log_info` for successful operations. Used `log_error` for exceptions and errors.

This enhanced logging will help in understanding the internal state and flow of the module, making debugging and tracing much easier.  
[Timestamp: 2024/11/19 13:22:58]

**user**: Generate the complete classes.py with updated and enhanced logging throughout the module.  
[Timestamp: 2024/11/19 13:23:00]

**ai**: Here's the updated `classes.py` module with enhanced logging to improve traceability and debugging:

```python
"""
Class Extraction Module

This module provides functionality to extract class definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as methods, attributes, and docstrings.

Version: 1.0.2
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from utils import handle_exceptions

class ClassExtractor(BaseExtractor):
    """
    Extract class definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about classes, including methods, attributes, and docstrings.
    """

    @handle_exceptions(log_error)
    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions and their metadata from the source code.

        Args:
            source_code (Optional[str]): The source code to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing class metadata.
        """
        log_debug("Entered extract_classes method.")
        if source_code:
            log_debug("Parsing source code for class extraction.")
            self.parse_source(source_code)

        if not self.tree:
            log_error("No AST available for extraction")
            return []

        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                log_debug(f"Found class definition: {node.name}")
                class_info = self.extract_details(node)
                if class_info:
                    classes.append(class_info)
                    log_info(f"Extracted class '{node.name}' with metadata")
        log_debug(f"Total classes extracted: {len(classes)}")
        log_debug("Exiting extract_classes method.")
        return classes

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a class definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            Dict[str, Any]: A dictionary containing class details.
        """
        log_debug(f"Entered extract_details for class node: {node}")
        if not isinstance(node, ast.ClassDef):
            log_error(f"Expected ClassDef node, got {type(node).__name__}")
            return {}

        log_debug(f"Extracting details for class: {node.name}")
        details = self._extract_common_details(node)
        details.update({
            "bases": [ast.unparse(base) for base in node.bases],
            "methods": [self.extract_method_details(n) for n in node.body if isinstance(n, ast.FunctionDef)],
            "attributes": self.extract_class_attributes(node),
        })
        log_debug(f"Successfully extracted details for class {node.name}")
        log_debug(f"Exiting extract_details for class: {node.name}")
        return details

    def extract_method_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract detailed information from a method node.

        Args:
            node (ast.FunctionDef): The method node to extract details from.

        Returns:
            dict: A dictionary containing method details.
        """
        log_debug(f"Extracting method details for method: {node.name}")
        method_details = {
            "name": node.name,
            "parameters": self.extract_parameters(node),
            "return_type": self.extract_return_type(node),
            "docstring": self.extract_docstring(node),
            "decorators": self._extract_decorators(node),
            "exceptions": self._detect_exceptions(node),
            "lineno": node.lineno,
            "body_summary": self.get_body_summary(node),
        }
        log_debug(f"Extracted method details: {method_details}")
        return method_details

    def extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of dictionaries containing parameter details.
        """
        log_debug(f"Extracting parameters for method: {node.name}")
        parameters = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg),
                "optional": self._has_default(arg, node),
                "default_value": self._get_default_value(arg, node),
            }
            parameters.append(param_info)
            log_debug(f"Extracted parameter: {param_info}")
        return parameters

    def extract_class_attributes(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract attributes from a class node.

        Args:
            class_node (ast.ClassDef): The class node to extract attributes from.

        Returns:
            list: A list of dictionaries containing attribute details.
        """
        log_debug(f"Extracting attributes for class: {class_node.name}")
        attributes = []
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attr_info = {
                            "name": target.id,
                            "type": self._infer_type(node.value),
                            "lineno": node.lineno,
                        }
                        attributes.append(attr_info)
                        log_debug(f"Extracted attribute: {attr_info}")
        return attributes

    def _infer_type(self, value_node: ast.AST) -> str:
        """
        Infer the type of a value node.

        Args:
            value_node (ast.AST): The value node to infer type from.

        Returns:
            str: The inferred type as a string.
        """
        log_debug(f"Inferring type for value node: {value_node}")
        if isinstance(value_node, ast.Constant):
            inferred_type = type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            inferred_type = "List"
        elif isinstance(value_node, ast.Dict):
            inferred_type = "Dict"
        elif isinstance(value_node, ast.Set):
            inferred_type = "Set"
        elif isinstance(value_node, ast.Tuple):
            inferred_type = "Tuple"
        else:
            inferred_type = "Unknown"
        log_debug(f"Inferred type: {inferred_type}")
        return inferred_type

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        log_debug(f"Extracting return type for method: {node.name}")
        return_type = ast.unparse(node.returns) if node.returns else "Any"
        log_debug(f"Extracted return type: {return_type}")
        return return_type

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        log_debug(f"Generating body summary for method: {node.name}")
        body_summary = " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."
        log_debug(f"Generated body summary: {body_summary}")
        return body_summary

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """
        Extract decorators from a function definition.

        Args:
            node (ast.FunctionDef): The function node to extract decorators from.

        Returns:
            list: A list of decorator names.
        """
        log_debug(f"Extracting decorators for method: {node.name}")
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        log_debug(f"Extracted decorators: {decorators}")
        return decorators

    def _detect_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """
        Detect exceptions raised within a function.

        Args:
            node (ast.FunctionDef): The function node to analyze.

        Returns:
            list: A list of exception names.
        """
        log_debug(f"Detecting exceptions for method: {node.name}")
        exceptions = []
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Raise):
                if isinstance(sub_node.exc, ast.Name):
                    exceptions.append(sub_node.exc.id)
                elif isinstance(sub_node.exc, ast.Call) and isinstance(sub_node.exc.func, ast.Name):
                    exceptions.append(sub_node.exc.func.id)
        log_debug(f"Detected exceptions: {exceptions}")
        return exceptions

    def _extract_common_details(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Extract common details from a class node.

        Args:
            node (ast.ClassDef): The class node to extract common details from.

        Returns:
            dict: A dictionary containing common class details.
        """
        log_debug(f"Extracting common details for class: {node.name}")
        common_details = {
            "name": node.name,
            "docstring": ast.get_docstring(node) or '',
            "lineno": node.lineno,
        }
        log_debug(f"Extracted common details: {common_details}")
        return common_details

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """
        Get the type annotation of a function argument.

        Args:
            arg (ast.arg): The argument node to extract the type annotation from.

        Returns:
            str: The type annotation as a string.
        """
        log_debug(f"Getting type annotation for argument: {arg.arg}")
        type_annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        log_debug(f"Type annotation for {arg.arg}: {type_annotation}")
        return type_annotation

    def _has_default(self, arg: ast.arg, node: ast.FunctionDef) -> bool:
        """
        Check if a function argument has a default value.

        Args:
            arg (ast.arg): The argument node to check.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            bool: True if the argument has a default value, False otherwise.
        """
        log_debug(f"Checking if argument {arg.arg} has a default value.")
        has_default = arg in node.args.defaults
        log_debug(f"Argument {arg.arg} has default: {has_default}")
        return has_default

    def _get_default_value(self, arg: ast.arg, node: ast.FunctionDef) -> Optional[str]:
        """
        Get the default value of a function argument.

        Args:
            arg (ast.arg): The argument node to get the default value for.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            Optional[str]: The default value as a string, or None if not applicable.
        """
        log_debug(f"Getting default value for argument: {arg.arg}")
        try:
            index = node.args.args.index(arg) - len(node.args.args) + len(node.args.defaults)
            if index >= 0:
                default_value = ast.unparse(node.args.defaults[index])
                log_debug(f"Default value for {arg.arg}: {default_value}")
                return default_value
        except ValueError:
            log_error(f"Argument {arg.arg} not found in function arguments.")
        return None
```

## Key Enhancements

- **Entry and Exit Logs**: Added logging at the entry and exit points of each method to trace the flow of execution.
- **Detailed Variable Logging**: Logged key variable states and decisions, such as parameter extraction and exception detection.
- **Consistent Log Levels**: Used `log_debug` for tracing execution and `log_info` for successful operations. Used `log_error` for exceptions and errors.

This enhanced logging will help in understanding the internal state and flow of the module, making debugging and tracing much easier.  
[Timestamp: 2024/11/19 13:23:46]

**user**: Generate the complete extraction_manager.py with updated and enhanced logging throughout the module.  
[Timestamp: 2024/11/19 13:24:01]

**ai**: Here's the updated `extraction_manager.py` module with enhanced logging to improve traceability and debugging:

```python
import ast
from typing import Dict, Any, Optional, List, Tuple
from core.logger import log_info, log_error, log_debug
from metrics import Metrics  # Assuming the metrics module is available

class ExtractionManager:
    """
    Enhanced extraction manager with support for exception classes and robust error handling.
    """

    def __init__(self):
        """Initialize the extraction manager with a metrics calculator."""
        log_debug("Initializing ExtractionManager.")
        self.metrics = Metrics()

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code with special handling for exception classes.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata for classes and functions.
        """
        log_debug("Starting metadata extraction.")
        try:
            tree = ast.parse(source_code)
            self._add_parent_info(tree)
            classes, functions = [], []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    try:
                        log_debug(f"Processing class definition: {node.name}")
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            classes.append(metadata)
                        log_debug(f"Extracted {'exception ' if is_exception else ''}class: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting class metadata for {getattr(node, 'name', '<unknown>')}: {e}")
                elif isinstance(node, ast.FunctionDef):
                    try:
                        log_debug(f"Processing function definition: {node.name}")
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            functions.append(metadata)
                        log_debug(f"Extracted function: {node.name} with metadata: {metadata}")
                    except Exception as e:
                        log_error(f"Error extracting function metadata for {getattr(node, 'name', '<unknown>')}: {e}")

            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {'classes': classes, 'functions': functions}

        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': []}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': []}

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Determine if a class is an exception class.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class is an exception class.
        """
        log_debug(f"Checking if class {node.name} is an exception class.")
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                log_debug(f"Class {node.name} is an exception class.")
                return True
            elif isinstance(base, ast.Attribute) and base.attr in {'Exception', 'BaseException'}:
                log_debug(f"Class {node.name} is an exception class.")
                return True
        log_debug(f"Class {node.name} is not an exception class.")
        return False

    def _extract_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract metadata from a class definition with special handling for exception classes.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Whether this is an exception class.

        Returns:
            Dict[str, Any]: Extracted metadata.
        """
        log_debug(f"Extracting metadata for class: {node.name}")
        try:
            metadata = {
                'name': node.name,
                'docstring': ast.get_docstring(node) or '',
                'lineno': node.lineno,
                'is_exception': is_exception,
                'type': 'exception_class' if is_exception else 'class',
                'complexity': self.metrics.calculate_complexity(node)
            }

            if not is_exception:
                additional_metadata = {
                    'methods': self._extract_methods(node),
                    'bases': [self._format_base(base) for base in node.bases],
                    'decorators': self._extract_decorators(node)
                }
            else:
                additional_metadata = {
                    'bases': [self._format_base(base) for base in node.bases],
                    'error_code': self._extract_error_code(node)
                }

            metadata.update(additional_metadata)
            log_debug(f"Extracted metadata for class {node.name}: {metadata}")
            return metadata

        except (AttributeError, TypeError) as e:
            log_error(f"Error in class metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'exception_class' if is_exception else 'class',
                'error': str(e)
            }

    def _extract_error_code(self, node: ast.ClassDef) -> Optional[str]:
        """
        Extract error code from an exception class if present.

        Args:
            node (ast.ClassDef): The exception class node.

        Returns:
            Optional[str]: The error code if found.
        """
        log_debug(f"Extracting error code for exception class: {node.name}")
        try:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == 'code':
                            if isinstance(item.value, ast.Str):
                                error_code = item.value.s
                            elif isinstance(item.value, ast.Constant):
                                error_code = str(item.value.value)
                            log_debug(f"Extracted error code for {node.name}: {error_code}")
                            return error_code
            log_debug(f"No error code found for {node.name}")
            return None
        except (AttributeError, TypeError) as e:
            log_error(f"Error extracting error code for {node.name}: {e}")
            return None
        
    def _extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract methods from a class definition.

        Args:
            node (ast.ClassDef): The class node to extract methods from.

        Returns:
            List[Dict[str, Any]]: A list of extracted methods.
        """
        log_debug(f"Extracting methods for class: {node.name}")
        methods = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                method_metadata = {
                    'name': stmt.name,
                    'docstring': ast.get_docstring(stmt) or '',
                    'lineno': stmt.lineno,
                    'complexity': self.metrics.calculate_complexity(stmt)
                }
                methods.append(method_metadata)
                log_debug(f"Extracted method: {method_metadata}")
        return methods

    def _format_base(self, base: ast.AST) -> str:
        """
        Format the base class name.

        Args:
            base (ast.AST): The base class node.

        Returns:
            str: The formatted base class name.
        """
        log_debug(f"Formatting base class: {base}")
        if isinstance(base, ast.Name):
            formatted_base = base.id
        elif isinstance(base, ast.Attribute):
            formatted_base = f"{base.value.id}.{base.attr}"
        else:
            formatted_base = "<unknown>"
        log_debug(f"Formatted base class: {formatted_base}")
        return formatted_base

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """
        Extract decorators from a class definition.

        Args:
            node (ast.ClassDef): The class node to extract decorators from.

        Returns:
            List[str]: A list of extracted decorators.
        """
        log_debug(f"Extracting decorators for class: {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        log_debug(f"Extracted decorators: {decorators}")
        return decorators
    
    def _extract_function_metadata(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract metadata from a function definition.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            Dict[str, Any]: Extracted function metadata.
        """
        log_debug(f"Extracting metadata for function: {node.name}")
        try:
            function_metadata = {
                'name': node.name,
                'docstring': ast.get_docstring(node) or '',
                'args': self._extract_arguments(node),
                'return_type': self._extract_return_type(node),
                'decorators': self._extract_decorators(node),
                'lineno': node.lineno,
                'complexity': self.metrics.calculate_complexity(node)
            }
            log_debug(f"Extracted function metadata: {function_metadata}")
            return function_metadata
        except Exception as e:
            log_error(f"Error in function metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'function',
                'error': str(e)
            }

    def _extract_arguments(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract function arguments and their types.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            List[Tuple[str, str]]: List of (argument_name, type_annotation) pairs.
        """
        log_debug(f"Extracting arguments for function: {node.name}")
        args = []
        for arg in node.args.args:
            arg_type = "Any"
            if arg.annotation:
                try:
                    arg_type = ast.unparse(arg.annotation)
                except Exception:
                    log_error(f"Error parsing type annotation for argument {arg.arg}")
            args.append((arg.arg, arg_type))
            log_debug(f"Extracted argument: {arg.arg} with type: {arg_type}")
        return args

    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract function return type.

        Args:
            node (ast.FunctionDef): The function node.

        Returns:
            str: Return type annotation or "Any".
        """
        log_debug(f"Extracting return type for function: {node.name}")
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                log_error(f"Error parsing return type for function {node.name}")
                return_type = "Any"
        else:
            return_type = "Any"
        log_debug(f"Extracted return type: {return_type}")
        return return_type

    def _add_parent_info(self, tree: ast.AST) -> None:
        """
        Add parent node information to each node in an AST.

        Args:
            tree (ast.AST): The Abstract Syntax Tree to process.

        Returns:
            None: Modifies the tree in place.
        """
        log_debug("Adding parent information to AST nodes.")
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)
        log_info("Parent information added to AST nodes.")
```

## Key Enhancements

- **Entry and Exit Logs**: Added logging at the entry and exit points of each method to trace the flow of execution.
- **Detailed Variable Logging**: Logged key variable states and decisions, such as class type checks and metadata extraction.
- **Consistent Log Levels**: Used `log_debug` for tracing execution and `log_info` for successful operations. Used `log_error` for exceptions and errors.

This enhanced logging will help in understanding the internal state and flow of the module, making debugging and tracing much easier.  
[Timestamp: 2024/11/19 13:24:53]
