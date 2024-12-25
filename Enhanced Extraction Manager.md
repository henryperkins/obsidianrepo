```python
class ExtractionManager:
    """
    Enhanced extraction manager with integrated docstring handling.
    """
    def __init__(self):
        self.class_extractor = None
        self.function_extractor = None
        self.tree = None
        self.source_code = None
        
        # Initialize docstring manager
        self.docstring_manager = DocStringManager()
        
    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata with enhanced docstring processing.
        
        Args:
            source_code (str): The source code to analyze
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing extracted metadata
            for classes and functions
            
        Raises:
            ExtractionError: If extraction fails
        """
        try:
            log_debug("Starting metadata extraction")

            if not self.validate_source_code(source_code):
                raise ExtractionError("Source code validation failed")

            self.source_code = source_code
            self.class_extractor = ClassExtractor(source_code)
            self.function_extractor = FunctionExtractor(source_code)

            classes = []
            functions = []

            # Process all nodes in the AST
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    try:
                        # Generate unique identifier
                        identifier = self._generate_identifier(node)
                        
                        # Extract basic metadata
                        metadata = self.process_node(node)
                        
                        if metadata:
                            # Process docstring if present
                            if 'docstring' in metadata:
                                docstring = metadata['docstring']
                                if docstring:
                                    # Process and preserve the docstring
                                    processed_docstring = self.docstring_manager.process_docstring(
                                        docstring,
                                        identifier=identifier,
                                        preserve=True
                                    )
                                    metadata['docstring'] = processed_docstring
                                    
                                    # Extract additional metadata from docstring
                                    sections = self.docstring_manager.extract_sections(processed_docstring)
                                    metadata['docstring_sections'] = sections
                                    
                                    # Add complexity information from docstring if available
                                    if 'complexity' in sections:
                                        metadata['complexity_info'] = sections['complexity']

                            # Add to appropriate list
                            if isinstance(node, ast.ClassDef):
                                # Process class methods
                                metadata['methods'] = self._process_class_methods(node, identifier)
                                classes.append(metadata)
                                log_debug(f"Extracted class: {node.name}")
                            else:
                                functions.append(metadata)
                                log_debug(f"Extracted function: {node.name}")
                                
                    except Exception as e:
                        log_error(f"Error extracting metadata for {type(node).__name__}: {str(e)}")
                        continue

            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {
                'classes': classes,
                'functions': functions
            }

        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            raise ExtractionError(f"Failed to extract metadata: {str(e)}")

    def _process_class_methods(self, class_node: ast.ClassDef, class_identifier: str) -> Dict[str, Any]:
        """
        Process methods of a class with docstring handling.
        
        Args:
            class_node (ast.ClassDef): The class node to process
            class_identifier (str): The identifier of the parent class
            
        Returns:
            Dict[str, Any]: Dictionary containing method metadata
        """
        methods = {}
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Generate method identifier
                    method_identifier = f"{class_identifier}.{node.name}"
                    
                    # Extract method metadata
                    method_info = self.process_node(node)
                    
                    if method_info and 'docstring' in method_info:
                        # Process and preserve method docstring
                        docstring = method_info['docstring']
                        if docstring:
                            processed_docstring = self.docstring_manager.process_docstring(
                                docstring,
                                identifier=method_identifier,
                                preserve=True
                            )
                            method_info['docstring'] = processed_docstring
                            
                            # Extract method-specific metadata
                            method_info['sections'] = self.docstring_manager.extract_sections(
                                processed_docstring
                            )
                    
                    methods[node.name] = method_info
                    
                except Exception as e:
                    log_error(f"Error processing method {node.name}: {e}")
                    continue
                    
        return methods

    def _generate_identifier(self, node: ast.AST) -> str:
        """
        Generate a unique identifier for a node.
        
        Args:
            node (ast.AST): The AST node
            
        Returns:
            str: Unique identifier
        """
        context = {
            'type': node.__class__.__name__,
            'name': getattr(node, 'name', 'unknown'),
            'lineno': getattr(node, 'lineno', 0),
            'col_offset': getattr(node, 'col_offset', 0)
        }
        return self.docstring_manager.generate_identifier(context)

    def process_node(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Process an individual AST node with enhanced docstring handling.
        
        Args:
            node (ast.AST): The node to process
            
        Returns:
            Optional[Dict[str, Any]]: Node metadata if successful
        """
        try:
            if isinstance(node, ast.ClassDef):
                details = self.class_extractor.extract_details(node)
            elif isinstance(node, ast.FunctionDef):
                details = self.function_extractor.extract_details(node)
                # Calculate additional metrics
                metrics = Metrics()
                details['complexity'] = metrics.calculate_complexity(node)
                details['maintainability_index'] = metrics.calculate_maintainability_index(node)
            else:
                return None

            return details
            
        except Exception as e:
            node_name = getattr(node, 'name', '<unknown>')
            node_type = type(node).__name__
            log_error(f"Error processing {node_type} {node_name}: {str(e)}")
            return None

    def validate_source_code(self, source_code: str) -> bool:
        """
        Validate the provided source code.
        
        Args:
            source_code (str): The source code to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not source_code or not isinstance(source_code, str):
            raise ValueError("Source code must be a non-empty string.")
        
        try:
            self.tree = ast.parse(source_code)
            return True
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {str(e)}")
        except Exception as e:
            raise ValueError(f"Source code validation failed: {str(e)}")

    def get_node_info(self, node: ast.AST) -> Dict[str, str]:
        """
        Get basic information about an AST node.
        
        Args:
            node (ast.AST): The AST node
            
        Returns:
            Dict[str, str]: Node information
        """
        return {
            'type': type(node).__name__,
            'name': getattr(node, 'name', '<unknown>'),
            'line': getattr(node, 'lineno', '<unknown>'),
            'has_docstring': ast.get_docstring(node) is not None
        }

    def is_valid_node(self, node: ast.AST) -> bool:
        """
        Check if a node is valid for processing.
        
        Args:
            node (ast.AST): The node to validate
            
        Returns:
            bool: True if valid
        """
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            return False
            
        required_attrs = ['name', 'body', 'lineno']
        return all(hasattr(node, attr) for attr in required_attrs)
```


```python
"""
Extraction Manager Module

This module provides a manager class for extracting metadata from Python source code.
It coordinates the extraction of class and function metadata using dedicated extractors.

Version: 1.1.0
Author: Development Team
"""

import ast
from typing import Dict, List, Any, Optional
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from metrics import Metrics
from logger import log_info, log_error, log_debug
from docstring.manager import DocStringManager  # Import the DocStringManager

class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass

class ExtractionManager:
    """
    Enhanced extraction manager with integrated docstring handling.
    """

    def __init__(self):
        """Initialize the ExtractionManager without source code."""
        self.class_extractor = None
        self.function_extractor = None
        self.tree = None
        self.source_code = None
        
        # Initialize docstring manager
        self.docstring_manager = DocStringManager()

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata with enhanced docstring processing.
        
        Args:
            source_code (str): The source code to analyze
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing extracted metadata
            for classes and functions
            
        Raises:
            ExtractionError: If extraction fails
        """
        try:
            log_debug("Starting metadata extraction")

            if not self.validate_source_code(source_code):
                raise ExtractionError("Source code validation failed")

            self.source_code = source_code
            self.class_extractor = ClassExtractor(source_code)
            self.function_extractor = FunctionExtractor(source_code)

            classes = []
            functions = []

            # Process all nodes in the AST
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    try:
                        # Generate unique identifier
                        identifier = self._generate_identifier(node)
                        
                        # Extract basic metadata
                        metadata = self.process_node(node)
                        
                        if metadata:
                            # Process docstring if present
                            if 'docstring' in metadata:
                                docstring = metadata['docstring']
                                if docstring:
                                    # Process and preserve the docstring
                                    processed_docstring = self.docstring_manager.process_docstring(
                                        docstring,
                                        identifier=identifier,
                                        preserve=True
                                    )
                                    metadata['docstring'] = processed_docstring
                                    
                                    # Extract additional metadata from docstring
                                    sections = self.docstring_manager.extract_sections(processed_docstring)
                                    metadata['docstring_sections'] = sections
                                    
                                    # Add complexity information from docstring if available
                                    if 'complexity' in sections:
                                        metadata['complexity_info'] = sections['complexity']

                            # Add to appropriate list
                            if isinstance(node, ast.ClassDef):
                                # Process class methods
                                metadata['methods'] = self._process_class_methods(node, identifier)
                                classes.append(metadata)
                                log_debug(f"Extracted class: {node.name}")
                            else:
                                functions.append(metadata)
                                log_debug(f"Extracted function: {node.name}")
                                
                    except Exception as e:
                        log_error(f"Error extracting metadata for {type(node).__name__}: {str(e)}")
                        continue

            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {
                'classes': classes,
                'functions': functions
            }

        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            raise ExtractionError(f"Failed to extract metadata: {str(e)}")

    def _process_class_methods(self, class_node: ast.ClassDef, class_identifier: str) -> Dict[str, Any]:
        """
        Process methods of a class with docstring handling.
        
        Args:
            class_node (ast.ClassDef): The class node to process
            class_identifier (str): The identifier of the parent class
            
        Returns:
            Dict[str, Any]: Dictionary containing method metadata
        """
        methods = {}
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Generate method identifier
                    method_identifier = f"{class_identifier}.{node.name}"
                    
                    # Extract method metadata
                    method_info = self.process_node(node)
                    
                    if method_info and 'docstring' in method_info:
                        # Process and preserve method docstring
                        docstring = method_info['docstring']
                        if docstring:
                            processed_docstring = self.docstring_manager.process_docstring(
                                docstring,
                                identifier=method_identifier,
                                preserve=True
                            )
                            method_info['docstring'] = processed_docstring
                            
                            # Extract method-specific metadata
                            method_info['sections'] = self.docstring_manager.extract_sections(
                                processed_docstring
                            )
                    
                    methods[node.name] = method_info
                    
                except Exception as e:
                    log_error(f"Error processing method {node.name}: {e}")
                    continue
                    
        return methods

    def _generate_identifier(self, node: ast.AST) -> str:
        """
        Generate a unique identifier for a node.
        
        Args:
            node (ast.AST): The AST node
            
        Returns:
            str: Unique identifier
        """
        context = {
            'type': node.__class__.__name__,
            'name': getattr(node, 'name', 'unknown'),
            'lineno': getattr(node, 'lineno', 0),
            'col_offset': getattr(node, 'col_offset', 0)
        }
        return self.docstring_manager.generate_identifier(context)

    def process_node(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Process an individual AST node with enhanced docstring handling.
        
        Args:
            node (ast.AST): The node to process
            
        Returns:
            Optional[Dict[str, Any]]: Node metadata if successful
        """
        try:
            if isinstance(node, ast.ClassDef):
                details = self.class_extractor.extract_details(node)
            elif isinstance(node, ast.FunctionDef):
                details = self.function_extractor.extract_details(node)
                # Calculate additional metrics
                metrics = Metrics()
                details['complexity'] = metrics.calculate_complexity(node)
                details['maintainability_index'] = metrics.calculate_maintainability_index(node)
            else:
                return None

            return details
            
        except Exception as e:
            node_name = getattr(node, 'name', '<unknown>')
            node_type = type(node).__name__
            log_error(f"Error processing {node_type} {node_name}: {str(e)}")
            return None

    def validate_source_code(self, source_code: str) -> bool:
        """
        Validate the provided source code.
        
        Args:
            source_code (str): The source code to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not source_code or not isinstance(source_code, str):
            raise ValueError("Source code must be a non-empty string.")
        
        try:
            self.tree = ast.parse(source_code)
            return True
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {str(e)}")
        except Exception as e:
            raise ValueError(f"Source code validation failed: {str(e)}")

    def get_node_info(self, node: ast.AST) -> Dict[str, str]:
        """
        Get basic information about an AST node.
        
        Args:
            node (ast.AST): The AST node
            
        Returns:
            Dict[str, str]: Node information
        """
        return {
            'type': type(node).__name__,
            'name': getattr(node, 'name', '<unknown>'),
            'line': getattr(node, 'lineno', '<unknown>'),
            'has_docstring': ast.get_docstring(node) is not None
        }

    def is_valid_node(self, node: ast.AST) -> bool:
        """
        Check if a node is valid for processing.
        
        Args:
            node (ast.AST): The node to validate
            
        Returns:
            bool: True if valid
        """
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            return False
            
        required_attrs = ['name', 'body', 'lineno']
        return all(hasattr(node, attr) for attr in required_attrs)
```