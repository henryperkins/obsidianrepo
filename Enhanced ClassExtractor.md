```python
class ClassExtractor(BaseExtractor):
    """
    Enhanced class extractor with integrated docstring handling.
    
    This class extracts detailed information about Python classes, including
    methods, attributes, and enhanced docstring processing.
    """

    def __init__(self, source_code: str):
        """Initialize with source code and docstring manager."""
        super().__init__(source_code)
        self.docstring_manager = DocStringManager()

    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions with enhanced docstring processing.
        
        Args:
            source_code: Optional source code to analyze
            
        Returns:
            List of dictionaries containing class metadata
        """
        if source_code:
            self.__init__(source_code)

        if not self.tree:
            log_error("No AST available for extraction")
            return []

        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                try:
                    # Generate unique identifier for this class
                    identifier = f"class:{node.name}:{node.lineno}"
                    
                    # Extract basic class info
                    class_info = self.extract_details(node)
                    
                    if class_info:
                        # Process the class docstring
                        if class_info.get('docstring'):
                            processed_docstring = self.docstring_manager.process_docstring(
                                class_info['docstring'],
                                identifier=identifier,
                                preserve=True
                            )
                            class_info['docstring'] = processed_docstring
                            
                            # Extract docstring sections for metadata
                            sections = self.docstring_manager.extract_sections(processed_docstring)
                            class_info['docstring_metadata'] = self._process_docstring_metadata(sections)
                        
                        # Add to results
                        classes.append(class_info)
                        log_info(f"Extracted class '{node.name}' with metadata")
                    
                except Exception as e:
                    log_error(f"Error extracting class '{getattr(node, 'name', '<unknown>')}': {e}")
                    continue

        log_debug(f"Total classes extracted: {len(classes)}")
        return classes

    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract enhanced details from a class definition node.
        
        Args:
            node: The AST node to extract details from
            
        Returns:
            Dictionary containing class details
        """
        if not isinstance(node, ast.ClassDef):
            log_error(f"Expected ClassDef node, got {type(node).__name__}")
            return {}

        log_debug(f"Extracting details for class: {node.name}")

        try:
            details = self._extract_common_details(node)
            
            # Generate identifier for this class
            identifier = f"class:{node.name}:{node.lineno}"
            
            # Extract and process docstring
            if details.get('docstring'):
                details['docstring'] = self.docstring_manager.process_docstring(
                    details['docstring'],
                    identifier=identifier,
                    preserve=True
                )
            
            # Extract class-specific information
            details.update({
                'bases': [ast.unparse(base) for base in node.bases],
                'methods': self._extract_methods(node, identifier),
                'attributes': self.extract_class_attributes(node),
                'decorators': self._extract_decorators(node),
                'is_dataclass': any(dec.id == 'dataclass' for dec in node.decorator_list 
                                  if isinstance(dec, ast.Name)),
                'inner_classes': self._extract_inner_classes(node)
            })

            # Add complexity metrics
            metrics = Metrics()
            details['complexity_metrics'] = {
                'methods_count': len(details['methods']),
                'attributes_count': len(details['attributes']),
                'inheritance_depth': self._calculate_inheritance_depth(node),
                'maintainability_index': metrics.calculate_maintainability_index(node)
            }

            log_debug(f"Successfully extracted details for class {node.name}")
            return details

        except Exception as e:
            log_error(f"Failed to extract details for class {node.name}: {e}")
            return {}

    def _extract_methods(self, node: ast.ClassDef, class_identifier: str) -> List[Dict[str, Any]]:
        """
        Extract details about class methods with enhanced docstring processing.
        
        Args:
            node: The class node
            class_identifier: Identifier of the parent class
            
        Returns:
            List of dictionaries containing method details
        """
        methods = []
        for method_node in node.body:
            if isinstance(method_node, ast.FunctionDef):
                try:
                    # Generate method identifier
                    method_identifier = f"{class_identifier}.{method_node.name}"
                    
                    method_info = {
                        'name': method_node.name,
                        'parameters': self.extract_parameters(method_node),
                        'return_type': self.extract_return_type(method_node),
                        'decorators': self._extract_decorators(method_node),
                        'is_property': any(dec.id == 'property' for dec in method_node.decorator_list 
                                         if isinstance(dec, ast.Name)),
                        'is_classmethod': any(dec.id == 'classmethod' for dec in method_node.decorator_list 
                                            if isinstance(dec, ast.Name)),
                        'is_staticmethod': any(dec.id == 'staticmethod' for dec in method_node.decorator_list 
                                             if isinstance(dec, ast.Name))
                    }

                    # Process method docstring
                    docstring = ast.get_docstring(method_node)
                    if docstring:
                        processed_docstring = self.docstring_manager.process_docstring(
                            docstring,
                            identifier=method_identifier,
                            preserve=True
                        )
                        method_info['docstring'] = processed_docstring
                        
                        # Extract method metadata
                        sections = self.docstring_manager.extract_sections(processed_docstring)
                        method_info['docstring_metadata'] = self._process_docstring_metadata(sections)

	                    # Add method complexity metrics
	                    method_info['complexity_metrics'] = {
	                        'cyclomatic': Metrics.calculate_cyclomatic_complexity(method_node),
	                        'cognitive': Metrics.calculate_cognitive_complexity(method_node)
	                    }
	                    
	                    methods.append(method_info)
	                    
	                except Exception as e:
	                    log_error(f"Error extracting method {method_node.name}: {e}")
	                    continue
	                    
	        return methods

    def _process_docstring_metadata(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract metadata from docstring sections."""
        metadata = {}
        
        # Extract version info
        if 'Version' in sections:
            metadata['version'] = sections['Version']
            
        # Extract deprecation warnings
        if 'Deprecated' in sections:
            metadata['deprecated'] = True
            metadata['deprecation_reason'] = sections['Deprecated']
            
        # Extract notes and warnings
        if 'Notes' in sections:
            metadata['notes'] = sections['Notes']
        if 'Warning' in sections:
            metadata['warnings'] = sections['Warning']
            
        return metadata

    def _extract_inner_classes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract inner class definitions."""
        inner_classes = []
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                try:
                    inner_class_info = self.extract_details(item)
                    if inner_class_info:
                        inner_classes.append(inner_class_info)
                except Exception as e:
                    log_error(f"Error extracting inner class {item.name}: {e}")
                    continue
        return inner_classes

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class."""
        depth = 0
        for base in node.bases:
            try:
                if isinstance(base, ast.Name):
                    # Simple inheritance
                    depth = max(depth, 1)
                elif isinstance(base, ast.Attribute):
                    # Nested inheritance (e.g., module.Class)
                    depth = max(depth, 2)
            except Exception as e:
                log_error(f"Error calculating inheritance depth: {e}")
                continue
        return depth
```


```python
"""
Class Extraction Module

This module provides functionality to extract class definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as methods, attributes, and docstrings.

Version: 1.0.0
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional
from logger import log_info, log_error, log_debug
from base import BaseExtractor
from docstring.manager import DocStringManager  # Import the DocStringManager

class ClassExtractor(BaseExtractor):
    """
    Enhanced class extractor with integrated docstring handling.
    
    This class extracts detailed information about Python classes, including
    methods, attributes, and enhanced docstring processing.
    """

    def __init__(self, source_code: str):
        """Initialize with source code and docstring manager."""
        super().__init__(source_code)
        self.docstring_manager = DocStringManager()

    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions with enhanced docstring processing.
        
        Args:
            source_code: Optional source code to analyze
            
        Returns:
            List of dictionaries containing class metadata
        """
        if source_code:
            self.__init__(source_code)

        if not self.tree:
            log_error("No AST available for extraction")
            return []

        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                try:
                    # Generate unique identifier for this class
                    identifier = f"class:{node.name}:{node.lineno}"
                    
                    # Extract basic class info
                    class_info = self.extract_details(node)
                    
                    if class_info:
                        # Process the class docstring
                        if class_info.get('docstring'):
                            processed_docstring = self.docstring_manager.process_docstring(
                                class_info['docstring'],
                                identifier=identifier,
                                preserve=True
                            )
                            class_info['docstring'] = processed_docstring
                            
                            # Extract docstring sections for metadata
                            sections = self.docstring_manager.extract_sections(processed_docstring)
                            class_info['docstring_metadata'] = self._process_docstring_metadata(sections)
                        
                        # Add to results
                        classes.append(class_info)
                        log_info(f"Extracted class '{node.name}' with metadata")
                    
                except Exception as e:
                    log_error(f"Error extracting class '{getattr(node, 'name', '<unknown>')}': {e}")
                    continue

        log_debug(f"Total classes extracted: {len(classes)}")
        return classes

    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract enhanced details from a class definition node.
        
        Args:
            node: The AST node to extract details from
            
        Returns:
            Dictionary containing class details
        """
        if not isinstance(node, ast.ClassDef):
            log_error(f"Expected ClassDef node, got {type(node).__name__}")
            return {}

        log_debug(f"Extracting details for class: {node.name}")

        try:
            details = self._extract_common_details(node)
            
            # Generate identifier for this class
            identifier = f"class:{node.name}:{node.lineno}"
            
            # Extract and process docstring
            if details.get('docstring'):
                details['docstring'] = self.docstring_manager.process_docstring(
                    details['docstring'],
                    identifier=identifier,
                    preserve=True
                )
            
            # Extract class-specific information
            details.update({
                'bases': [ast.unparse(base) for base in node.bases],
                'methods': self._extract_methods(node, identifier),
                'attributes': self.extract_class_attributes(node),
                'decorators': self._extract_decorators(node),
                'is_dataclass': any(dec.id == 'dataclass' for dec in node.decorator_list 
                                  if isinstance(dec, ast.Name)),
                'inner_classes': self._extract_inner_classes(node)
            })

            # Add complexity metrics
            metrics = Metrics()
            details['complexity_metrics'] = {
                'methods_count': len(details['methods']),
                'attributes_count': len(details['attributes']),
                'inheritance_depth': self._calculate_inheritance_depth(node),
                'maintainability_index': metrics.calculate_maintainability_index(node)
            }

            log_debug(f"Successfully extracted details for class {node.name}")
            return details

        except Exception as e:
            log_error(f"Failed to extract details for class {node.name}: {e}")
            return {}

    def _extract_methods(self, node: ast.ClassDef, class_identifier: str) -> List[Dict[str, Any]]:
        """
        Extract details about class methods with enhanced docstring processing.
        
        Args:
            node: The class node
            class_identifier: Identifier of the parent class
            
        Returns:
            List of dictionaries containing method details
        """
        methods = []
        for method_node in node.body:
            if isinstance(method_node, ast.FunctionDef):
                try:
                    # Generate method identifier
                    method_identifier = f"{class_identifier}.{method_node.name}"
                    
                    method_info = {
                        'name': method_node.name,
                        'parameters': self.extract_parameters(method_node),
                        'return_type': self.extract_return_type(method_node),
                        'decorators': self._extract_decorators(method_node),
                        'is_property': any(dec.id == 'property' for dec in method_node.decorator_list 
                                         if isinstance(dec, ast.Name)),
                        'is_classmethod': any(dec.id == 'classmethod' for dec in method_node.decorator_list 
                                            if isinstance(dec, ast.Name)),
                        'is_staticmethod': any(dec.id == 'staticmethod' for dec in method_node.decorator_list 
                                             if isinstance(dec, ast.Name))
                    }

                    # Process method docstring
                    docstring = ast.get_docstring(method_node)
                    if docstring:
                        processed_docstring = self.docstring_manager.process_docstring(
                            docstring,
                            identifier=method_identifier,
                            preserve=True
                        )
                        method_info['docstring'] = processed_docstring
                        
                        # Extract method metadata
                        sections = self.docstring_manager.extract_sections(processed_docstring)
                        method_info['docstring_metadata'] = self._process_docstring_metadata(sections)

	                    # Add method complexity metrics
	                    method_info['complexity_metrics'] = {
	                        'cyclomatic': Metrics.calculate_cyclomatic_complexity(method_node),
	                        'cognitive': Metrics.calculate_cognitive_complexity(method_node)
	                    }
	                    
	                    methods.append(method_info)
	                    
	                except Exception as e:
	                    log_error(f"Error extracting method {method_node.name}: {e}")
	                    continue
	                    
	        return methods

    def _process_docstring_metadata(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract metadata from docstring sections."""
        metadata = {}
        
        # Extract version info
        if 'Version' in sections:
            metadata['version'] = sections['Version']
            
        # Extract deprecation warnings
        if 'Deprecated' in sections:
            metadata['deprecated'] = True
            metadata['deprecation_reason'] = sections['Deprecated']
            
        # Extract notes and warnings
        if 'Notes' in sections:
            metadata['notes'] = sections['Notes']
        if 'Warning' in sections:
            metadata['warnings'] = sections['Warning']
            
        return metadata

    def _extract_inner_classes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract inner class definitions."""
        inner_classes = []
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                try:
                    inner_class_info = self.extract_details(item)
                    if inner_class_info:
                        inner_classes.append(inner_class_info)
                except Exception as e:
                    log_error(f"Error extracting inner class {item.name}: {e}")
                    continue
        return inner_classes

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class."""
        depth = 0
        for base in node.bases:
            try:
                if isinstance(base, ast.Name):
                    # Simple inheritance
                    depth = max(depth, 1)
                elif isinstance(base, ast.Attribute):
                    # Nested inheritance (e.g., module.Class)
                    depth = max(depth, 2)
            except Exception as e:
                log_error(f"Error calculating inheritance depth: {e}")
                continue
        return depth
```