### **`logger.py`**
```python
"""
Logging configuration and utilities.
Provides consistent logging across the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

class LoggerSetup:
    """Configures and manages application logging."""

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name
            level: Logging level

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            logger.setLevel(level)
            
            # Create formatters
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_dir / f"{name}.log",
                maxBytes=1024 * 1024,  # 1MB
                backupCount=5
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger

    @staticmethod
    def configure(
        level: str,
        format_str: str,
        log_dir: Optional[str] = None
    ) -> None:
        """
        Configure global logging settings.

        Args:
            level: Logging level
            format_str: Log format string
            log_dir: Optional log directory
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create handlers
        handlers = []
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(format_str))
        handlers.append(console)
        
        # File handler
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path / f"app_{datetime.now():%Y%m%d}.log",
                maxBytes=1024 * 1024,  # 1MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format=format_str,
            handlers=handlers
        )

def log_debug(message: str) -> None:
    """Log debug message."""
    logging.getLogger(__name__).debug(message)

def log_info(message: str) -> None:
    """Log info message."""
    logging.getLogger(__name__).info(message)

def log_warning(message: str) -> None:
    """Log warning message."""
    logging.getLogger(__name__).warning(message)

def log_error(message: str) -> None:
    """Log error message."""
    logging.getLogger(__name__).error(message)
```
---
### **`metrics.py`**
```python
"""
Metrics Module

Provides performance and usage metrics tracking for Azure OpenAI operations.
Focuses on essential metrics while maintaining extensibility.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

@dataclass
class OperationMetrics:
    """Tracks metrics for a specific operation type."""
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    total_tokens: int = 0
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def average_duration(self) -> float:
        """Calculate average operation duration."""
        return self.total_duration / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_tokens(self) -> float:
        """Calculate average tokens per operation."""
        return self.total_tokens / self.total_count if self.total_count > 0 else 0.0

@dataclass
class TokenUsage:
    """Tracks token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    
    def update(self, prompt: int, completion: int, cost: float) -> None:
        """Update token usage and cost."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_cost += cost

class MetricsCollector:
    """Collects and manages various performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = datetime.now()
        self.operations: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self.token_usage = TokenUsage()
        self.recent_operations: List[Dict[str, Any]] = []
        self.max_recent_operations = 100
        logger.info("Metrics collector initialized")

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: Optional[float] = None,
        tokens_used: Optional[Dict[str, int]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Track an operation's metrics.

        Args:
            operation_type: Type of operation
            success: Whether operation succeeded
            duration: Operation duration in seconds
            tokens_used: Token usage details
            error: Error message if failed
        """
        try:
            metrics = self.operations[operation_type]
            metrics.total_count += 1
            
            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1
                if error:
                    metrics.error_counts[error] += 1

            if duration is not None:
                metrics.total_duration += duration

            if tokens_used:
                prompt_tokens = tokens_used.get('prompt_tokens', 0)
                completion_tokens = tokens_used.get('completion_tokens', 0)
                cost = self._calculate_cost(prompt_tokens, completion_tokens)
                
                metrics.total_tokens += prompt_tokens + completion_tokens
                self.token_usage.update(prompt_tokens, completion_tokens, cost)

            # Track recent operations
            self._add_recent_operation(
                operation_type=operation_type,
                success=success,
                duration=duration,
                tokens_used=tokens_used,
                error=error
            )

            logger.debug(f"Tracked {operation_type} operation: success={success}")

        except Exception as e:
            logger.error(f"Error tracking metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict[str, Any]: Metrics summary
        """
        try:
            runtime = (datetime.now() - self.start_time).total_seconds()
            
            metrics = {
                'runtime_seconds': runtime,
                'operations': {},
                'token_usage': {
                    'prompt_tokens': self.token_usage.prompt_tokens,
                    'completion_tokens': self.token_usage.completion_tokens,
                    'total_tokens': (
                        self.token_usage.prompt_tokens + 
                        self.token_usage.completion_tokens
                    ),
                    'total_cost': self.token_usage.total_cost
                },
                'operations_per_minute': self._calculate_operations_per_minute(runtime)
            }

            # Add operation-specific metrics
            for op_type, op_metrics in self.operations.items():
                metrics['operations'][op_type] = {
                    'total_count': op_metrics.total_count,
                    'success_count': op_metrics.success_count,
                    'failure_count': op_metrics.failure_count,
                    'success_rate': op_metrics.success_rate,
                    'average_duration': op_metrics.average_duration,
                    'average_tokens': op_metrics.average_tokens,
                    'common_errors': dict(
                        sorted(
                            op_metrics.error_counts.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                    )
                }

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {'error': str(e)}

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost based on token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            float: Calculated cost
        """
        # Example rates - adjust based on your Azure OpenAI pricing
        PROMPT_RATE = 0.0004  # per 1K tokens
        COMPLETION_RATE = 0.0008  # per 1K tokens
        
        prompt_cost = (prompt_tokens / 1000) * PROMPT_RATE
        completion_cost = (completion_tokens / 1000) * COMPLETION_RATE
        
        return prompt_cost + completion_cost

    def _calculate_operations_per_minute(self, runtime_seconds: float) -> float:
        """
        Calculate operations per minute.

        Args:
            runtime_seconds: Total runtime in seconds

        Returns:
            float: Operations per minute
        """
        total_operations = sum(
            metrics.total_count for metrics in self.operations.values()
        )
        if runtime_seconds > 0:
            return (total_operations / runtime_seconds) * 60
        return 0.0

    def _add_recent_operation(self, **kwargs) -> None:
        """
        Add operation to recent operations list.

        Args:
            **kwargs: Operation details
        """
        kwargs['timestamp'] = datetime.now().isoformat()
        self.recent_operations.append(kwargs)
        
        # Maintain maximum size
        if len(self.recent_operations) > self.max_recent_operations:
            self.recent_operations.pop(0)

    def get_recent_operations(self) -> List[Dict[str, Any]]:
        """
        Get list of recent operations.

        Returns:
            List[Dict[str, Any]]: Recent operations
        """
        return self.recent_operations.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        self.start_time = datetime.now()
        self.operations.clear()
        self.token_usage = TokenUsage()
        self.recent_operations.clear()
        logger.info("Metrics reset")

    def get_operation_metrics(self, operation_type: str) -> Dict[str, Any]:
        """
        Get metrics for specific operation type.

        Args:
            operation_type: Type of operation

        Returns:
            Dict[str, Any]: Operation metrics
        """
        metrics = self.operations.get(operation_type, OperationMetrics())
        return {
            'total_count': metrics.total_count,
            'success_count': metrics.success_count,
            'failure_count': metrics.failure_count,
            'success_rate': metrics.success_rate,
            'average_duration': metrics.average_duration,
            'average_tokens': metrics.average_tokens,
            'error_counts': dict(metrics.error_counts)
        }
```
---
### **`ai_interaction.py`**
```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from openai import AsyncAzureOpenAI, OpenAIError

from core.logger import LoggerSetup
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.monitoring import MetricsCollector
from exceptions import AIServiceError, TokenLimitError, ProcessingError

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """
    Manages AI model interactions with integrated monitoring and caching.
    Handles Azure OpenAI API communication for documentation generation.
    """

    def __init__(
        self,
        config: AzureOpenAIConfig,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the AI interaction handler.

        Args:
            config: Azure OpenAI configuration
            cache: Optional cache instance
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.cache = cache
        self.metrics = metrics_collector
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        
        logger.info("AI Interaction Handler initialized")

    async def process_code(self, source_code: str) -> Tuple[str, str]:
        """
        Process source code to generate documentation.

        Args:
            source_code: Source code to process

        Returns:
            Tuple[str, str]: (updated_code, documentation)

        Raises:
            ProcessingError: If processing fails
            AIServiceError: If API interaction fails
        """
        operation_start = datetime.now()
        
        try:
            # Check cache first if enabled
            if self.cache:
                cache_key = f"doc:{hash(source_code)}"
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    logger.debug("Cache hit for documentation")
                    return cached_result.get('code', ''), cached_result.get('docs', '')

            # Generate documentation
            docs = await self._generate_documentation(source_code)
            
            # Update source code with documentation
            updated_code = await self._update_code(source_code, docs)
            
            # Cache result if enabled
            if self.cache:
                await self.cache.save_docstring(
                    cache_key,
                    {'code': updated_code, 'docs': docs}
                )

            # Track metrics
            if self.metrics:
                duration = (datetime.now() - operation_start).total_seconds()
                await self.metrics.track_operation(
                    operation_type='process_code',
                    success=True,
                    duration=duration
                )

            return updated_code, docs

        except Exception as e:
            if self.metrics:
                duration = (datetime.now() - operation_start).total_seconds()
                await self.metrics.track_operation(
                    operation_type='process_code',
                    success=False,
                    duration=duration,
                    error=str(e)
                )
            raise AIServiceError(f"Failed to process code: {str(e)}")

    async def _generate_documentation(self, source_code: str) -> str:
        """
        Generate documentation using Azure OpenAI.

        Args:
            source_code: Source code to document

        Returns:
            str: Generated documentation
        """
        try:
            # Prepare the prompt
            prompt = self._create_documentation_prompt(source_code)
            
            # Make API request with retries
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.deployment_name,
                        messages=[
                            {"role": "system", "content": "You are a documentation expert that generates clear, concise, and accurate documentation for Python code."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    # Extract and validate documentation
                    documentation = response.choices[0].message.content
                    if not documentation:
                        raise ProcessingError("Empty response from AI model")
                    
                    return documentation

                except OpenAIError as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            raise AIServiceError(f"Failed to generate documentation: {str(e)}")

    async def _update_code(self, source_code: str, documentation: str) -> str:
        """
        Update source code with generated documentation.

        Args:
            source_code: Original source code
            documentation: Generated documentation

        Returns:
            str: Updated source code
        """
        try:
            # Parse the documentation and update the code
            # This is a simplified version - you might want to use a proper parser
            return f'"""\n{documentation}\n"""\n\n{source_code}'
        except Exception as e:
            logger.error(f"Code update failed: {e}")
            raise ProcessingError(f"Failed to update code: {str(e)}")

    def _create_documentation_prompt(self, source_code: str) -> str:
        """
        Create prompt for documentation generation.

        Args:
            source_code: Source code to document

        Returns:
            str: Generated prompt
        """
        return f"""
Please generate comprehensive documentation for the following Python code.
Include:
- Module overview
- Class and function documentation
- Parameter descriptions
- Return value descriptions
- Usage examples where appropriate

Code:
{source_code}

Generate the documentation in markdown format.
"""

    async def close(self) -> None:
        """Close the AI interaction handler and cleanup resources."""
        try:
            await self.client.close()
            logger.info("AI Interaction Handler closed successfully")
        except Exception as e:
            logger.error(f"Error closing AI handler: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
```
---
### **`docstring_utils.py`**
```Python
"""Utilities for docstring generation and validation using Azure OpenAI."""

from typing import Dict, List, Any, Tuple
import re
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class DocstringValidator:
    """Validates and processes docstrings."""

    def __init__(self):
        """Initialize validator with basic requirements."""
        self.required_sections = ['summary', 'args', 'returns']
        self.min_length = {
            'summary': 10,
            'description': 10
        }

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate docstring content and structure.

        Args:
            docstring_data: Dictionary containing docstring sections

        Returns:
            Tuple[bool, List[str]]: Validation result and error messages
        """
        errors = []

        # Check required sections
        for section in self.required_sections:
            if section not in docstring_data:
                errors.append(f"Missing required section: {section}")

        # Validate content
        if 'summary' in docstring_data:
            if len(docstring_data['summary'].strip()) < self.min_length['summary']:
                errors.append("Summary too short")

        # Validate parameters
        if 'args' in docstring_data:
            param_errors = self._validate_parameters(docstring_data['args'])
            errors.extend(param_errors)

        # Validate return value
        if 'returns' in docstring_data:
            return_errors = self._validate_return(docstring_data['returns'])
            errors.extend(return_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Docstring validation failed: {errors}")

        return is_valid, errors

    def _validate_parameters(self, parameters: List[Dict[str, Any]]) -> List[str]:
        """Validate parameter documentation."""
        errors = []
        
        if not isinstance(parameters, list):
            return ["Parameters must be a list"]

        for param in parameters:
            if not isinstance(param, dict):
                errors.append("Invalid parameter format")
                continue

            if 'name' not in param:
                errors.append("Parameter missing name")
            if 'type' not in param:
                errors.append(f"Parameter {param.get('name', '?')} missing type")
            if 'description' not in param:
                errors.append(f"Parameter {param.get('name', '?')} missing description")

        return errors

    def _validate_return(self, returns: Dict[str, Any]) -> List[str]:
        """Validate return value documentation."""
        errors = []

        if not isinstance(returns, dict):
            return ["Return value must be a dictionary"]

        if 'type' not in returns:
            errors.append("Return missing type")
        if 'description' not in returns:
            errors.append("Return missing description")

        return errors

def parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse docstring into structured format.

    Args:
        docstring: Raw docstring text

    Returns:
        Dict[str, Any]: Parsed docstring sections
    """
    if not docstring:
        return {
            "summary": "",
            "args": [],
            "returns": {"type": "None", "description": "No return value."}
        }

    sections = {
        "summary": "",
        "args": [],
        "returns": {"type": "None", "description": "No return value."}
    }

    lines = docstring.split('\n')
    current_section = 'summary'
    current_content = []

    for line in lines:
        line = line.strip()
        
        # Check for section headers
        if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'returns:')):
            # Save previous section
            if current_section == 'summary' and current_content:
                sections['summary'] = ' '.join(current_content)
            current_content = []
            
            # Update current section
            if any(line.lower().startswith(x) for x in ['args:', 'arguments:', 'parameters:']):
                current_section = 'args'
            else:
                current_section = 'returns'
            continue

        # Add content to current section
        if line:
            current_content.append(line)

    # Process final section
    if current_content:
        if current_section == 'summary':
            sections['summary'] = ' '.join(current_content)
        elif current_section == 'args':
            sections['args'] = _parse_parameters('\n'.join(current_content))
        elif current_section == 'returns':
            sections['returns'] = _parse_return('\n'.join(current_content))

    return sections

def _parse_parameters(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter section into structured format."""
    params = []
    param_pattern = r'(\w+)(?:\s*$([^)]+)$)?\s*:\s*(.+)'
    
    for line in params_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(param_pattern, line)
        if match:
            name, type_str, description = match.groups()
            params.append({
                'name': name,
                'type': type_str or 'Any',
                'description': description.strip()
            })

    return params

def _parse_return(return_str: str) -> Dict[str, str]:
    """Parse return section into structured format."""
    return_info = {
        'type': 'None',
        'description': return_str.strip() or 'No return value.'
    }

    # Try to extract type information
    type_match = re.match(r'(\w+):\s*(.+)', return_str)
    if type_match:
        return_info['type'] = type_match.group(1)
        return_info['description'] = type_match.group(2).strip()

    return return_info
```
---
### **`ast_analyzer.py`**
```python
"""
AST Analysis Module

Provides comprehensive analysis of Python source code using Abstract Syntax Trees (AST).
Handles code structure analysis, complexity calculation, and metadata extraction.
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

@dataclass
class CodeElement:
    """Represents a code element (function or class)."""
    name: str
    type: str
    docstring: Optional[str]
    lineno: int
    complexity: int
    parent: Optional[str] = None

@dataclass
class Parameter:
    """Represents a function/method parameter."""
    name: str
    type: str
    default: Optional[str] = None
    is_optional: bool = False

class ASTAnalyzer:
    """Analyzes Python code using AST with comprehensive analysis capabilities."""

    def __init__(self, source_code: Optional[str] = None):
        """
        Initialize AST analyzer.

        Args:
            source_code: Optional source code to analyze immediately
        """
        self.tree = None
        if source_code:
            self.tree = self.parse_source_code(source_code)
        logger.info("AST analyzer initialized")

    def parse_source_code(self, source_code: str) -> ast.AST:
        """
        Parse source code into an AST.

        Args:
            source_code: Source code to parse

        Returns:
            ast.AST: Parsed abstract syntax tree

        Raises:
            SyntaxError: If source code has syntax errors
        """
        try:
            tree = ast.parse(source_code)
            self.add_parent_info(tree)
            self.tree = tree
            return tree
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise

    def analyze_code(self, tree: Optional[ast.AST] = None) -> Dict[str, Any]:
        """
        Perform comprehensive code analysis.

        Args:
            tree: Optional AST to analyze (uses self.tree if not provided)

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            tree = tree or self.tree
            if not tree:
                raise ValueError("No AST available for analysis")

            return {
                'imports': self._analyze_imports(tree),
                'classes': self._analyze_classes(tree),
                'functions': self._analyze_functions(tree),
                'complexity': self._analyze_complexity(tree),
                'metadata': self._extract_metadata(tree)
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {}

    def _analyze_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Analyze import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'type': 'import',
                        'name': name.name,
                        'asname': name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports.append({
                        'type': 'from',
                        'module': node.module,
                        'name': name.name,
                        'asname': name.asname
                    })
        return imports

    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze class definitions."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'bases': [self.get_name(base) for base in node.bases],
                    'methods': self._analyze_methods(node),
                    'attributes': self._analyze_attributes(node),
                    'decorators': [self.get_name(d) for d in node.decorator_list],
                    'lineno': node.lineno,
                    'complexity': self._calculate_complexity(node)
                })
        return classes

    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node):
                functions.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'parameters': self._analyze_parameters(node),
                    'returns': self.get_return_annotation(node),
                    'decorators': [self.get_name(d) for d in node.decorator_list],
                    'lineno': node.lineno,
                    'complexity': self._calculate_complexity(node),
                    'async': isinstance(node, ast.AsyncFunctionDef)
                })
        return functions

    def _analyze_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Analyze class methods."""
        methods = []
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'parameters': self._analyze_parameters(node),
                    'returns': self.get_return_annotation(node),
                    'decorators': [self.get_name(d) for d in node.decorator_list],
                    'is_property': any(
                        isinstance(d, ast.Name) and d.id == 'property'
                        for d in node.decorator_list
                    ),
                    'is_classmethod': any(
                        isinstance(d, ast.Name) and d.id == 'classmethod'
                        for d in node.decorator_list
                    ),
                    'is_staticmethod': any(
                        isinstance(d, ast.Name) and d.id == 'staticmethod'
                        for d in node.decorator_list
                    ),
                    'async': isinstance(node, ast.AsyncFunctionDef),
                    'complexity': self._calculate_complexity(node)
                })
        return methods

    def _analyze_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Analyze function parameters."""
        params = []
        for i, arg in enumerate(node.args.args):
            if arg.arg == 'self':
                continue
                
            default_offset = len(node.args.args) - len(node.args.defaults)
            default = None
            if i >= default_offset:
                default = ast.unparse(node.args.defaults[i - default_offset])

            params.append({
                'name': arg.arg,
                'type': self.get_annotation(arg.annotation),
                'default': default,
                'is_optional': default is not None
            })

        # Handle *args and **kwargs
        if node.args.vararg:
            params.append({
                'name': f"*{node.args.vararg.arg}",
                'type': self.get_annotation(node.args.vararg.annotation),
                'is_vararg': True
            })
        if node.args.kwarg:
            params.append({
                'name': f"**{node.args.kwarg.arg}",
                'type': self.get_annotation(node.args.kwarg.annotation),
                'is_kwarg': True
            })

        return params

    def _analyze_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Analyze class attributes."""
        attributes = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                attributes.append({
                    'name': item.target.id,
                    'type': self.get_annotation(item.annotation),
                    'has_default': item.value is not None,
                    'is_class_var': self._is_class_var(item)
                })
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': 'Any',
                            'has_default': True,
                            'is_class_var': True
                        })
        return attributes

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        return {
            'cyclomatic': self._calculate_complexity(tree),
            'cognitive': self._calculate_cognitive_complexity(tree),
            'nesting': self._calculate_max_nesting(tree)
        }

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Break, ast.Continue)):
                complexity += 1
        return complexity

    def _calculate_cognitive_complexity(self, node: ast.AST, nesting: int = 0) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += nesting + 1
                complexity += self._calculate_cognitive_complexity(child, nesting + 1)
            elif isinstance(child, ast.BoolOp):
                complexity += nesting + len(child.values) - 1
            elif isinstance(child, (ast.Break, ast.Continue)):
                complexity += nesting + 1
        return complexity

    def _calculate_max_nesting(self, node: ast.AST) -> int:
        """Calculate maximum nesting level."""
        def get_nesting(node: ast.AST, current: int = 0) -> int:
            max_nest = current
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    max_nest = max(max_nest, get_nesting(child, current + 1))
            return max_nest
        return get_nesting(node)

    def _extract_metadata(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract general code metadata."""
        return {
            'module_docstring': ast.get_docstring(tree),
            'total_lines': self._count_lines(tree),
            'code_elements': self._get_code_elements(tree),
            'dependencies': self._analyze_imports(tree)
        }

    def _count_lines(self, tree: ast.AST) -> Dict[str, int]:
        """Count different types of lines in the code."""
        return {
            'total': len(self.tree.body),
            'code': sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt)),
            'docstring': sum(1 for node in ast.walk(tree) if ast.get_docstring(node)),
            'comments': sum(1 for node in ast.walk(tree) if isinstance(node, ast.Expr) 
                          and isinstance(node.value, ast.Str))
        }

    def _get_code_elements(self, tree: ast.AST) -> List[CodeElement]:
        """Get all code elements with their details."""
        elements = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                elements.append(CodeElement(
                    name=node.name,
                    type=type(node).__name__,
                    docstring=ast.get_docstring(node),
                    lineno=node.lineno,
                    complexity=self._calculate_complexity(node),
                    parent=self._get_parent_name(node)
                ))
        return elements

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)

    @staticmethod
    def get_annotation(node: Optional[ast.AST]) -> str:
        """Get type annotation as string."""
        if node is None:
            return "Any"
        try:
            return ast.unparse(node)
        except Exception:
            return "Any"

    @staticmethod
    def get_return_annotation(node: ast.FunctionDef) -> str:
        """Get return type annotation."""
        return ast.unparse(node.returns) if node.returns else "Any"

    @staticmethod
    def get_name(node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{ASTAnalyzer.get_name(node.value)}.{node.attr}"
        return ast.unparse(node)

    @staticmethod
    def _is_method(node: ast.FunctionDef) -> bool:
        """Check if function is a method."""
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    @staticmethod
    def _is_class_var(node: ast.AnnAssign) -> bool:
        """Check if attribute is a class variable."""
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _get_parent_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of the node's parent."""
        parent = getattr(node, 'parent', None)
        if isinstance(parent, ast.ClassDef):
            return parent.name
        return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a comprehensive analysis summary."""
        if not self.tree:
            return {}

        analysis = self.analyze_code()
        return {
            'summary': {
                'total_classes': len(analysis['classes']),
                'total_functions': len(analysis['functions']),
                'total_imports': len(analysis['imports']),
                'complexity_score': analysis['complexity'],
                'lines': self._count_lines(self.tree)
            },
            'details': analysis
        }
```
---
### **`extraction_manager.py`**
```python
"""Manages code extraction and metadata collection."""

import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup
from .base import BaseExtractor

logger = LoggerSetup.get_logger(__name__)

class ExtractionManager(BaseExtractor):
    """Manages extraction of code elements and their metadata."""

    def extract_metadata(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract metadata from source code."""
        try:
            return {
                'functions': self._extract_functions(),
                'classes': self._extract_classes()
            }
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {'functions': [], 'classes': []}

    def _extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function definitions and metadata."""
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    functions.append({
                        **self.extract_common_details(node),
                        'args': [
                            (arg.arg, self.get_annotation(arg.annotation))
                            for arg in node.args.args
                        ],
                        'return_type': self.get_annotation(node.returns),
                        'decorators': [
                            d.id for d in node.decorator_list 
                            if isinstance(d, ast.Name)
                        ]
                    })
                except Exception as e:
                    logger.error(f"Error extracting function {node.name}: {e}")
        return functions

    def _extract_classes(self) -> List[Dict[str, Any]]:
        """Extract class definitions and metadata."""
        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                try:
                    classes.append({
                        **self.extract_common_details(node),
                        'bases': [
                            self.get_annotation(base) for base in node.bases
                        ],
                        'methods': self._extract_methods(node),
                        'is_exception': self._is_exception_class(node)
                    })
                except Exception as e:
                    logger.error(f"Error extracting class {node.name}: {e}")
        return classes

    def _extract_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract methods from a class."""
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                try:
                    methods.append({
                        'name': node.name,
                        'docstring': self.extract_docstring(node),
                        'args': [
                            (arg.arg, self.get_annotation(arg.annotation))
                            for arg in node.args.args
                            if arg.arg != 'self'
                        ],
                        'return_type': self.get_annotation(node.returns)
                    })
                except Exception as e:
                    logger.error(f"Error extracting method {node.name}: {e}")
        return methods

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                return True
        return False
```
---
### **`base.py`**
```python
"""Base extraction module providing core AST analysis functionality."""

import ast
from typing import Optional, Dict, Any, List
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class BaseExtractor:
    """Base class for code extraction with essential functionality."""
    
    def __init__(self, source_code: str):
        """Initialize with source code."""
        try:
            self.tree = ast.parse(source_code)
            self.source_code = source_code
        except SyntaxError as e:
            logger.error(f"Failed to parse source code: {e}")
            raise

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node."""
        try:
            return ast.get_docstring(node)
        except Exception as e:
            logger.error(f"Failed to extract docstring: {e}")
            return None

    def get_annotation(self, node: Optional[ast.AST]) -> str:
        """Get type annotation from an AST node."""
        if node is None:
            return "Any"
        try:
            return ast.unparse(node)
        except Exception:
            return "Any"

    def extract_common_details(self, node: ast.AST) -> Dict[str, Any]:
        """Extract common details from an AST node."""
        return {
            'name': getattr(node, 'name', '<unknown>'),
            'docstring': self.extract_docstring(node),
            'lineno': getattr(node, 'lineno', 0)
        }
```
---
### **`utils.py`**
```python
"""
Utility functions for code analysis and documentation generation.
Provides common functionality used across the codebase.
"""

import os
import ast
import json
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

# Type aliases for clarity
PathLike = Union[str, Path]
JsonDict = Dict[str, Any]

def handle_exceptions(
    error_handler: Optional[Callable] = None,
    default_return: Any = None
) -> Callable:
    """
    Decorator for handling exceptions with optional custom error handler.

    Args:
        error_handler: Optional function to handle errors
        default_return: Value to return on error

    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler(f"Error in {func.__name__}: {str(e)}")
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler(f"Error in {func.__name__}: {str(e)}")
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class FileHandler:
    """Handles file operations with error handling and logging."""

    @staticmethod
    @handle_exceptions(default_return="")
    async def read_file(filepath: PathLike, encoding: str = 'utf-8') -> str:
        """
        Read file content asynchronously.

        Args:
            filepath: Path to the file
            encoding: File encoding (default: utf-8)

        Returns:
            str: File content
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        return await asyncio.to_thread(path.read_text, encoding=encoding)

    @staticmethod
    @handle_exceptions(default_return=False)
    async def write_file(filepath: PathLike, content: str, encoding: str = 'utf-8') -> bool:
        """
        Write content to file asynchronously.

        Args:
            filepath: Path to the file
            content: Content to write
            encoding: File encoding (default: utf-8)

        Returns:
            bool: Success status
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content, encoding=encoding)
        return True

    @staticmethod
    @handle_exceptions(default_return={})
    async def read_json(filepath: PathLike) -> JsonDict:
        """
        Read and parse JSON file asynchronously.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dict: Parsed JSON data
        """
        content = await FileHandler.read_file(filepath)
        return json.loads(content) if content else {}

    @staticmethod
    @handle_exceptions(default_return=False)
    async def write_json(filepath: PathLike, data: JsonDict, indent: int = 2) -> bool:
        """
        Write data to JSON file asynchronously.

        Args:
            filepath: Path to the JSON file
            data: Data to write
            indent: JSON indentation (default: 2)

        Returns:
            bool: Success status
        """
        content = json.dumps(data, indent=indent)
        return await FileHandler.write_file(filepath, content)

class CodeAnalysisUtils:
    """Utilities for code analysis and AST manipulation."""

    @staticmethod
    def get_qualified_name(node: ast.AST) -> str:
        """
        Get fully qualified name from AST node.

        Args:
            node: AST node

        Returns:
            str: Qualified name
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{CodeAnalysisUtils.get_qualified_name(node.value)}.{node.attr}"
        return "unknown"

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            tree: AST to process
        """
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)

    @staticmethod
    def get_docstring(node: ast.AST) -> Optional[str]:
        """
        Get docstring from AST node safely.

        Args:
            node: AST node

        Returns:
            Optional[str]: Docstring if found
        """
        try:
            return ast.get_docstring(node)
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return None

    @staticmethod
    def get_function_signature(node: ast.FunctionDef) -> str:
        """
        Get function signature as string.

        Args:
            node: Function node

        Returns:
            str: Function signature
        """
        try:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            return f"def {node.name}({', '.join(args)}){returns}"
        except Exception as e:
            logger.error(f"Error getting function signature: {e}")
            return node.name

class HashUtils:
    """Utilities for hashing and caching."""

    @staticmethod
    def generate_hash(content: str) -> str:
        """
        Generate hash for content.

        Args:
            content: Content to hash

        Returns:
            str: Hash value
        """
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def generate_cache_key(prefix: str, *args: Any) -> str:
        """
        Generate cache key from arguments.

        Args:
            prefix: Key prefix
            *args: Arguments to include in key

        Returns:
            str: Cache key
        """
        content = ':'.join(str(arg) for arg in args)
        return f"{prefix}:{HashUtils.generate_hash(content)}"

class PathUtils:
    """Utilities for path handling."""

    @staticmethod
    def ensure_directory(path: PathLike) -> Path:
        """
        Ensure directory exists.

        Args:
            path: Directory path

        Returns:
            Path: Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_python_files(
        directory: PathLike,
        exclude_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Get all Python files in directory.

        Args:
            directory: Directory to search
            exclude_dirs: Directories to exclude

        Returns:
            List[Path]: List of Python file paths
        """
        exclude_dirs = set(exclude_dirs or [])
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
                    
        return python_files

    @staticmethod
    def get_relative_path(path: PathLike, base: PathLike) -> str:
        """
        Get relative path.

        Args:
            path: Path to make relative
            base: Base path

        Returns:
            str: Relative path
        """
        return str(Path(path).relative_to(base))

def create_result(
    success: bool,
    data: Any = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized result dictionary.

    Args:
        success: Operation success status
        data: Result data
        error: Error message if any

    Returns:
        Dict[str, Any]: Result dictionary
    """
    result = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
    
    if success:
        result["data"] = data
    else:
        result["error"] = error or "Unknown error"
        
    return result

def format_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format error with context for logging.

    Args:
        error: Exception instance
        context: Additional context

    Returns:
        Dict[str, Any]: Formatted error
    """
    return {
        "error_type": type(error).__name__,
        "message": str(error),
        "context": context or {},
        "timestamp": datetime.now().isoformat()
    }

@handle_exceptions(default_return=[])
def split_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Split text into code and non-code blocks.

    Args:
        text: Text to split

    Returns:
        List[Dict[str, str]]: List of blocks with type and content
    """
    blocks = []
    current_block = {"type": "text", "content": []}
    
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            # Save current block if not empty
            if current_block["content"]:
                blocks.append({
                    "type": current_block["type"],
                    "content": '\n'.join(current_block["content"])
                })
            # Switch block type
            current_block = {
                "type": "code" if current_block["type"] == "text" else "text",
                "content": []
            }
        else:
            current_block["content"].append(line)
            
    # Add final block
    if current_block["content"]:
        blocks.append({
            "type": current_block["type"],
            "content": '\n'.join(current_block["content"])
        })
        
    return blocks
```
---
### **`main.py`**

```python
"""
Main module for documentation generation using Azure OpenAI.
Handles workflow orchestration and component management.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig
from core.cache import Cache
from core.monitoring import MetricsCollector
from ai_interaction import AIInteractionHandler
from exceptions import WorkflowError

# Initialize logging
logger = LoggerSetup.get_logger(__name__)

class DocumentationProcessor:
    """Handles the core documentation processing logic."""

    def __init__(self, components: 'AsyncComponentManager', config: AzureOpenAIConfig):
        """Initialize with components and configuration."""
        self.components = components
        self.config = config
        self.metrics = components.components.get("metrics")

    async def process_file(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process a single file for documentation generation.

        Args:
            file_path: Path to source file
            output_dir: Output directory for documentation

        Returns:
            Dict[str, Any]: Processing results
        """
        start_time = datetime.now()
        metrics = {
            "file_path": str(file_path),
            "start_time": start_time.isoformat(),
            "status": "pending"
        }

        try:
            # Read source code
            source_code = await self._read_source_code(file_path)
            if not source_code.strip():
                return self._create_result(
                    "skipped",
                    message="Empty source file",
                    metrics=metrics
                )

            # Process with AI handler
            ai_handler = self.components.components["ai_handler"]
            code, docs = await ai_handler.process_code(source_code)

            if not docs:
                return self._create_result(
                    "failed",
                    message="Documentation generation failed",
                    metrics=metrics
                )

            # Save results
            result = await self._save_documentation(
                code, docs, file_path, output_dir
            )

            # Update metrics
            metrics.update({
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "status": "success"
            })

            return self._create_result(
                "success",
                result=result,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            metrics.update({
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "status": "failed",
                "error": str(e)
            })
            return self._create_result(
                "failed",
                error=str(e),
                metrics=metrics
            )

    async def _read_source_code(self, file_path: Path) -> str:
        """Read source code with encoding detection."""
        try:
            return await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return await asyncio.to_thread(file_path.read_text, encoding="latin-1")

    async def _save_documentation(
        self,
        code: str,
        docs: str,
        file_path: Path,
        output_dir: Path
    ) -> Dict[str, str]:
        """Save generated documentation and updated code."""
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save documentation
            doc_path = output_dir / f"{file_path.stem}_docs.md"
            await asyncio.to_thread(doc_path.write_text, docs, encoding="utf-8")

            # Save updated code (optional)
            code_path = output_dir / f"{file_path.stem}_updated.py"
            await asyncio.to_thread(code_path.write_text, code, encoding="utf-8")

            return {
                "documentation_path": str(doc_path),
                "code_path": str(code_path)
            }

        except Exception as e:
            raise Exception(f"Failed to save documentation: {e}")

    def _create_result(
        self,
        status: str,
        result: Optional[Dict] = None,
        message: str = "",
        error: str = "",
        metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            "status": status,
            "result": result or {},
            "message": message,
            "error": error,
            "metrics": metrics or {}
        }

class AsyncComponentManager:
    """Manages async components lifecycle."""

    def __init__(self, config: AzureOpenAIConfig):
        """Initialize with configuration."""
        self.config = config
        self.components: Dict[str, Any] = {}

    async def __aenter__(self) -> 'AsyncComponentManager':
        """Initialize components in correct order."""
        try:
            # Initialize monitoring
            self.components["metrics"] = MetricsCollector()

            # Initialize cache if enabled
            if self.config.cache_enabled:
                self.components["cache"] = await Cache.create(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password
                )

            # Initialize AI handler
            self.components["ai_handler"] = AIInteractionHandler(
                config=self.config,
                cache=self.components.get("cache"),
                metrics_collector=self.components["metrics"]
            )

            return self

        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup components in reverse order."""
        errors = []
        for component_name in reversed(list(self.components)):
            try:
                component = self.components[component_name]
                if hasattr(component, "close"):
                    await component.close()
            except Exception as e:
                errors.append(f"Error closing {component_name}: {e}")
                logger.error(f"Error closing {component_name}: {e}")

        self.components.clear()

        if errors:
            raise WorkflowError(
                "Error during cleanup",
                details={'cleanup_errors': errors}
            )

class WorkflowOrchestrator:
    """Orchestrates the documentation generation workflow."""

    def __init__(self, config: AzureOpenAIConfig):
        """Initialize with configuration."""
        self.config = config

    async def run(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run the documentation generation workflow.

        Args:
            source_path: Source code path (file or directory)
            output_dir: Output directory for documentation

        Returns:
            Dict[str, Any]: Workflow results
        """
        workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = datetime.now()
        results = {
            "workflow_id": workflow_id,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "skipped_files": 0
        }

        try:
            async with AsyncComponentManager(self.config) as manager:
                processor = DocumentationProcessor(manager, self.config)

                if source_path.is_file():
                    result = await processor.process_file(source_path, output_dir)
                    self._update_results(results, result)
                else:
                    for file in source_path.rglob("*.py"):
                        result = await processor.process_file(file, output_dir)
                        self._update_results(results, result)

                # Generate summary
                await self._generate_summary(
                    results,
                    output_dir,
                    workflow_id,
                    start_time
                )

                return results

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise WorkflowError(f"Workflow {workflow_id} failed: {e}")

    def _update_results(self, results: Dict[str, Any], file_result: Dict[str, Any]):
        """Update workflow results with file processing outcome."""
        results["processed_files"] += 1
        
        status = file_result.get("status", "failed")
        if status == "success":
            results["successful_files"] += 1
        elif status == "skipped":
            results["skipped_files"] += 1
        else:
            results["failed_files"] += 1

    async def _generate_summary(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        workflow_id: str,
        start_time: datetime
    ) -> None:
        """Generate workflow summary."""
        duration = (datetime.now() - start_time).total_seconds()
        
        summary = [
            "# Documentation Generation Summary\n",
            f"Workflow ID: {workflow_id}",
            f"Duration: {duration:.2f} seconds",
            f"Processed Files: {results['processed_files']}",
            f"Successful: {results['successful_files']}",
            f"Failed: {results['failed_files']}",
            f"Skipped: {results['skipped_files']}"
        ]

        summary_path = output_dir / f"summary_{workflow_id}.md"
        await asyncio.to_thread(
            summary_path.write_text,
            "\n".join(summary),
            encoding="utf-8"
        )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation using Azure OpenAI"
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Source code file or directory"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for documentation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    return parser.parse_args()

async def main() -> None:
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Load configuration
        config = AzureOpenAIConfig.from_env()
        
        # Setup logging
        LoggerSetup.configure(
            config.log_level,
            config.log_format,
            Path(config.log_directory)
        )

        # Run workflow
        orchestrator = WorkflowOrchestrator(config)
        results = await orchestrator.run(args.source_path, args.output_dir)
        
        logger.info("Documentation generation completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```
---
### **`config.py`**
```python
"""
Configuration management for Azure OpenAI integration.
Handles loading and validation of configuration settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
from pathlib import Path
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

@dataclass
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI integration."""

    # Azure OpenAI settings
    endpoint: str = field(default="")
    api_key: str = field(default="")
    api_version: str = field(default="2024-02-15-preview")
    deployment_name: str = field(default="")
    model_name: str = field(default="gpt-4")

    # Request settings
    max_tokens: int = field(default=1000)
    temperature: float = field(default=0.7)
    request_timeout: int = field(default=30)
    max_retries: int = field(default=3)
    retry_delay: int = field(default=2)

    # Rate limiting
    max_requests_per_minute: int = field(default=60)
    max_tokens_per_minute: int = field(default=150000)
    batch_size: int = field(default=5)

    # Caching settings
    cache_enabled: bool = field(default=True)
    cache_ttl: int = field(default=3600)
    redis_host: str = field(default="localhost")
    redis_port: int = field(default=6379)
    redis_db: int = field(default=0)
    redis_password: Optional[str] = field(default=None)

    # Logging settings
    log_level: str = field(default="INFO")
    log_format: str = field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_directory: str = field(default="logs")

    @classmethod
    def from_env(cls) -> 'AzureOpenAIConfig':
        """
        Create configuration from environment variables.

        Returns:
            AzureOpenAIConfig: Configuration instance
        """
        try:
            config = cls(
                # Azure OpenAI settings
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_KEY", ""),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                model_name=os.getenv("MODEL_NAME", "gpt-4"),

                # Request settings
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                retry_delay=int(os.getenv("RETRY_DELAY", "2")),

                # Rate limiting
                max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60")),
                max_tokens_per_minute=int(os.getenv("MAX_TOKENS_PER_MINUTE", "150000")),
                batch_size=int(os.getenv("BATCH_SIZE", "5")),

                # Cache settings
                cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_db=int(os.getenv("REDIS_DB", "0")),
                redis_password=os.getenv("REDIS_PASSWORD"),

                # Logging settings
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_format=os.getenv(
                    "LOG_FORMAT",
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
                log_directory=os.getenv("LOG_DIRECTORY", "logs")
            )

            if not config.validate():
                raise ValueError("Invalid configuration")

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validate required fields
            if not all([self.endpoint, self.api_key, self.deployment_name]):
                raise ValueError("Missing required Azure OpenAI credentials")

            # Validate numeric values
            if not 0 <= self.temperature <= 1:
                raise ValueError("Temperature must be between 0 and 1")

            if self.max_tokens <= 0:
                raise ValueError("Max tokens must be positive")

            # Validate rate limits
            if self.max_requests_per_minute <= 0:
                raise ValueError("Max requests per minute must be positive")

            # Validate cache settings
            if self.cache_enabled and self.cache_ttl <= 0:
                raise ValueError("Cache TTL must be positive")

            # Ensure log directory exists
            log_dir = Path(self.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            # Azure OpenAI settings
            "endpoint": self.endpoint,
            "deployment_name": self.deployment_name,
            "model_name": self.model_name,
            "api_version": self.api_version,

            # Request settings
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,

            # Rate limiting
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "batch_size": self.batch_size,

            # Cache settings
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,

            # Logging settings
            "log_level": self.log_level,
            "log_directory": self.log_directory
        }

    def update(self, **kwargs) -> None:
        """
        Update configuration settings.

        Args:
            **kwargs: Settings to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration setting: {key}")

        if not self.validate():
            raise ValueError("Invalid configuration after update")
```
---
### **`response_parser.py`**
```python
"""
Response Parser Module

Handles parsing and validation of Azure OpenAI API responses.
Ensures consistent and reliable output formatting.
"""

import json
from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ResponseParser:
    """Parses and validates Azure OpenAI API responses."""

    # Define response schema
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "documentation": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "description"]
                    }
                },
                "required": ["summary", "description"]
            }
        },
        "required": ["documentation"]
    }

    def __init__(self):
        """Initialize response parser."""
        self._validation_cache = {}
        logger.info("Response parser initialized")

    async def parse_response(
        self,
        response: str,
        expected_format: str = 'json'
    ) -> Optional[Dict[str, Any]]:
        """
        Parse and validate API response.

        Args:
            response: Raw API response
            expected_format: Expected response format ('json' or 'markdown')

        Returns:
            Optional[Dict[str, Any]]: Parsed response
        """
        try:
            if expected_format == 'json':
                return await self._parse_json_response(response)
            return await self._parse_markdown_response(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response."""
        try:
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            # Parse JSON
            data = json.loads(response.strip())
            
            # Validate against schema
            if not self._validate_response(data):
                logger.error("Response validation failed")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON response: {e}")
            return None

    async def _parse_markdown_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse markdown response into structured format."""
        try:
            sections = self._split_markdown_sections(response)
            
            return {
                'documentation': {
                    'summary': sections.get('summary', ''),
                    'description': sections.get('description', ''),
                    'parameters': self._parse_parameters(sections.get('parameters', '')),
                    'returns': self._parse_returns(sections.get('returns', ''))
                }
            }

        except Exception as e:
            logger.error(f"Error parsing markdown response: {e}")
            return None

    def _validate_response(self, data: Dict[str, Any]) -> bool:
        """Validate response against schema."""
        try:
            validate(instance=data, schema=self.RESPONSE_SCHEMA)
            return True
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return False

    def _split_markdown_sections(self, markdown: str) -> Dict[str, str]:
        """Split markdown into sections."""
        sections = {}
        current_section = 'description'
        current_content = []

        for line in markdown.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []

                # Update current section
                section_name = line.lstrip('#').strip().lower()
                current_section = section_name
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _parse_parameters(self, params_text: str) -> List[Dict[str, str]]:
        """Parse parameter section from markdown."""
        params = []
        current_param = None

        for line in params_text.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                # New parameter
                if ':' in line:
                    name, rest = line[2:].split(':', 1)
                    current_param = {
                        'name': name.strip(),
                        'type': 'Any',
                        'description': rest.strip()
                    }
                    params.append(current_param)
            elif current_param and line:
                # Continue previous parameter description
                current_param['description'] += ' ' + line

        return params

    def _parse_returns(self, returns_text: str) -> Dict[str, str]:
        """Parse returns section from markdown."""
        if ':' in returns_text:
            type_str, description = returns_text.split(':', 1)
            return {
                'type': type_str.strip(),
                'description': description.strip()
            }
        return {
            'type': 'None',
            'description': returns_text.strip() or 'No return value.'
        }
    
```
---
### **`monitoring.py`**
```python
"""
Monitoring Module

Provides system monitoring and performance tracking for Azure OpenAI operations.
Focuses on essential metrics while maintaining efficiency.
"""

import psutil
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class SystemMonitor:
    """Monitors system resources and performance metrics."""

    def __init__(self, check_interval: int = 60):
        """
        Initialize system monitor.

        Args:
            check_interval: Interval between system checks in seconds
        """
        self.check_interval = check_interval
        self.start_time = datetime.now()
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        logger.info("System monitor initialized")

    async def start(self) -> None:
        """Start monitoring system resources."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop monitoring system resources."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                self._store_metrics(metrics)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store collected metrics."""
        for key, value in metrics.items():
            if key != 'timestamp':
                self._metrics[key].append({
                    'timestamp': metrics['timestamp'],
                    'value': value
                })

        # Keep only last hour of metrics
        max_entries = 3600 // self.check_interval
        for key in self._metrics:
            if len(self._metrics[key]) > max_entries:
                self._metrics[key] = self._metrics[key][-max_entries:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        try:
            current_metrics = self._collect_system_metrics()
            runtime = (datetime.now() - self.start_time).total_seconds()

            return {
                'current': current_metrics,
                'runtime_seconds': runtime,
                'averages': self._calculate_averages(),
                'status': self._get_system_status()
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}

    def _calculate_averages(self) -> Dict[str, float]:
        """Calculate average values for metrics."""
        averages = {}
        for key, values in self._metrics.items():
            if values:
                if key == 'cpu':
                    averages[key] = sum(v['value']['percent'] for v in values) / len(values)
                elif key in ['memory', 'disk']:
                    averages[key] = sum(v['value']['percent'] for v in values) / len(values)
        return averages

    def _get_system_status(self) -> str:
        """Determine overall system status."""
        try:
            current = self._collect_system_metrics()
            
            # Define thresholds
            CPU_THRESHOLD = 90
            MEMORY_THRESHOLD = 90
            DISK_THRESHOLD = 90

            if (current.get('cpu', {}).get('percent', 0) > CPU_THRESHOLD or
                current.get('memory', {}).get('percent', 0) > MEMORY_THRESHOLD or
                current.get('disk', {}).get('percent', 0) > DISK_THRESHOLD):
                return 'critical'
            elif (current.get('cpu', {}).get('percent', 0) > CPU_THRESHOLD * 0.8 or
                  current.get('memory', {}).get('percent', 0) > MEMORY_THRESHOLD * 0.8 or
                  current.get('disk', {}).get('percent', 0) > DISK_THRESHOLD * 0.8):
                return 'warning'
            return 'healthy'
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return 'unknown'

    async def __aenter__(self) -> 'SystemMonitor':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
```
---
### **`markdown_generator.py`**
```python
"""
Markdown Documentation Generator Module

Generates structured markdown documentation from code analysis results.
Provides consistent and well-formatted documentation output.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import ast
import re

from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class MarkdownDocumentationGenerator:
    """Generates standardized markdown documentation for Python code."""

    def __init__(self, source_code: str, module_path: Optional[str] = None):
        """
        Initialize markdown generator.

        Args:
            source_code: Source code to document
            module_path: Optional module path for documentation
        """
        self.source_code = source_code
        self.module_path = Path(module_path) if module_path else Path("module.py")
        self.tree = ast.parse(source_code)
        self.docstring = ast.get_docstring(self.tree) or ""
        self.changes: List[str] = []
        logger.info(f"Markdown generator initialized for {self.module_path}")

    async def generate_markdown(self) -> str:
        """
        Generate complete markdown documentation.

        Returns:
            str: Generated markdown documentation
        """
        try:
            sections = [
                await self._generate_header(),
                await self._generate_overview(),
                await self._generate_installation(),
                await self._generate_classes_section(),
                await self._generate_functions_section(),
                await self._generate_examples(),
                await self._generate_changes_section()
            ]
            
            return "\n\n".join(filter(None, sections))
            
        except Exception as e:
            logger.error(f"Failed to generate markdown: {e}")
            return f"# Documentation Generation Failed\n\nError: {str(e)}"

    async def _generate_header(self) -> str:
        """Generate module header section."""
        return f"# {self.module_path.stem}\n\n" + self._format_badges()

    def _format_badges(self) -> str:
        """Generate status badges."""
        return (
            "![Python](https://img.shields.io/badge/python-3.7%2B-blue)\n"
            "![Status](https://img.shields.io/badge/status-stable-green)\n"
            f"![Updated](https://img.shields.io/badge/updated-{datetime.now().strftime('%Y---%m')}-blue)\n"
        )

    async def _generate_overview(self) -> str:
        """Generate overview section."""
        overview = "## Overview\n\n"
        if self.docstring:
            overview += self.docstring
        else:
            overview += "*No module description available.*"
        return overview

    async def _generate_installation(self) -> str:
        """Generate installation instructions."""
        return (
            "## Installation\n\n"
            "```bash\n"
            f"pip install {self.module_path.stem}\n"
            "```"
        )

    async def _generate_classes_section(self) -> str:
        """Generate classes section with methods."""
        classes = [node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)]
        if not classes:
            return ""

        sections = ["## Classes"]
        
        for cls in classes:
            sections.append(self._format_class(cls))

        return "\n\n".join(sections)

    def _format_class(self, cls: ast.ClassDef) -> str:
        """Format class documentation."""
        doc = f"### {cls.name}\n\n"
        
        # Add inheritance info
        if cls.bases:
            bases = ", ".join(ast.unparse(base) for base in cls.bases)
            doc += f"*Inherits from: `{bases}`*\n\n"

        # Add docstring
        if cls_doc := ast.get_docstring(cls):
            doc += f"{cls_doc}\n\n"

        # Add methods
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
        if methods:
            doc += "#### Methods\n\n"
            for method in methods:
                doc += self._format_method(method)

        return doc

    def _format_method(self, method: ast.FunctionDef) -> str:
        """Format method documentation."""
        # Format signature
        signature = self._format_signature(method)
        doc = f"##### `{signature}`\n\n"

        # Add docstring
        if method_doc := ast.get_docstring(method):
            doc += f"{method_doc}\n\n"

        return doc

    async def _generate_functions_section(self) -> str:
        """Generate functions section."""
        functions = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and not isinstance(node.parent, ast.ClassDef)
        ]
        
        if not functions:
            return ""

        sections = ["## Functions"]
        
        for func in functions:
            sections.append(self._format_function(func))

        return "\n\n".join(sections)

    def _format_function(self, func: ast.FunctionDef) -> str:
        """Format function documentation."""
        # Format signature
        signature = self._format_signature(func)
        doc = f"### `{signature}`\n\n"

        # Add docstring
        if func_doc := ast.get_docstring(func):
            doc += f"{func_doc}\n\n"

        return doc

    def _format_signature(self, node: ast.FunctionDef) -> str:
        """Format function/method signature."""
        args = []
        
        # Process arguments
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # Add return type
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        
        return f"{node.name}({', '.join(args)}){returns}"

    async def _generate_examples(self) -> str:
        """Generate examples section."""
        return (
            "## Examples\n\n"
            "```python\n"
            f"from {self.module_path.stem} import *\n\n"
            "# Basic usage example\n"
            "...\n"
            "```"
        )

    async def _generate_changes_section(self) -> str:
        """Generate recent changes section."""
        if not self.changes:
            today = datetime.now().strftime('%Y-%m-%d')
            self.changes.append(f"[{today}] Initial documentation generated")

        return (
            "## Recent Changes\n\n" +
            "\n".join(f"- {change}" for change in self.changes)
        )

    def add_change(self, description: str) -> None:
        """
        Add a change entry to the documentation.

        Args:
            description: Change description
        """
        date = datetime.now().strftime('%Y-%m-%d')
        self.changes.append(f"[{date}] {description}")
```
---
### **`cache.py`**
```python
"""
Cache module for storing and retrieving AI-generated docstrings.
Provides both in-memory and Redis-based caching options.
"""

import json
import time
from typing import Optional, Any, Dict, Union
from datetime import datetime
import asyncio
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class Cache:
    """Flexible caching system with optional Redis support."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enabled: bool = True,
        ttl: int = 3600,
        prefix: str = "docstring:"
    ):
        """
        Initialize cache with configuration.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            enabled: Whether caching is enabled
            ttl: Default TTL in seconds
            prefix: Cache key prefix
        """
        self.enabled = enabled
        self.ttl = ttl
        self.prefix = prefix
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        self._lock = asyncio.Lock()

        # Use in-memory cache by default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        
        # Initialize Redis if needed
        self._redis = None
        if enabled:
            self._init_redis(host, port, db, password)

    def _init_redis(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str]
    ) -> None:
        """Initialize Redis connection if available."""
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            logger.info("Redis cache initialized")
        except ImportError:
            logger.warning("Redis not available, using in-memory cache")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")

    @classmethod
    async def create(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        **kwargs
    ) -> 'Cache':
        """
        Create and initialize cache instance.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            **kwargs: Additional configuration

        Returns:
            Cache: Initialized cache instance
        """
        cache = cls(host, port, db, password, **kwargs)
        if cache._redis:
            try:
                await cache._redis.ping()
                logger.info("Redis connection verified")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                cache._redis = None
        return cache

    async def get_cached_docstring(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached docstring by key.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Optional[Dict[str, Any]]: Cached data or default
        """
        if not self.enabled:
            return default

        cache_key = f"{self.prefix}{key}"
        try:
            # Try Redis first
            if self._redis:
                async with self._lock:
                    data = await self._redis.get(cache_key)
                if data:
                    self._stats['hits'] += 1
                    return json.loads(data)
            
            # Fallback to in-memory cache
            if cache_key in self._cache:
                if self._is_valid(cache_key):
                    self._stats['hits'] += 1
                    return self._cache[cache_key]
                else:
                    await self.invalidate(key)

            self._stats['misses'] += 1
            return default

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Cache get error: {e}")
            return default

    async def save_docstring(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Save docstring data to cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Optional TTL override

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        cache_key = f"{self.prefix}{key}"
        expiration = ttl or self.ttl

        try:
            serialized = json.dumps(data)
            
            # Try Redis first
            if self._redis:
                async with self._lock:
                    await self._redis.set(
                        cache_key,
                        serialized,
                        ex=expiration
                    )
            
            # Always update in-memory cache as backup
            self._cache[cache_key] = data
            self._timestamps[cache_key] = time.time() + expiration
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Cache save error: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate cached entry.

        Args:
            key: Cache key

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        cache_key = f"{self.prefix}{key}"
        try:
            # Remove from Redis
            if self._redis:
                async with self._lock:
                    await self._redis.delete(cache_key)
            
            # Remove from in-memory cache
            self._cache.pop(cache_key, None)
            self._timestamps.pop(cache_key, None)
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Cache invalidation error: {e}")
            return False

    def _is_valid(self, key: str) -> bool:
        """Check if cached entry is still valid."""
        return time.time() < self._timestamps.get(key, 0)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        stats = {
            'enabled': self.enabled,
            'stats': self._stats.copy()
        }

        if self._redis:
            try:
                info = await self._redis.info()
                stats.update({
                    'redis_connected': True,
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_clients': info.get('connected_clients', 0)
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                stats['redis_connected'] = False

        return stats

    async def clear(self) -> bool:
        """
        Clear all cached entries.

        Returns:
            bool: Success status
        """
        try:
            # Clear Redis
            if self._redis:
                async with self._lock:
                    keys = await self._redis.keys(f"{self.prefix}*")
                    if keys:
                        await self._redis.delete(*keys)
            
            # Clear in-memory cache
            self._cache.clear()
            self._timestamps.clear()
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def close(self) -> None:
        """Close cache connections."""
        if self._redis:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

    async def __aenter__(self) -> 'Cache':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
```
---
### **`exceptions.py`**
```python
"""
Custom exceptions for the documentation generation system.
Provides clear error handling and categorization.
"""

class DocumentationError(Exception):
    """Base exception for documentation-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ConfigurationError(DocumentationError):
    """Raised when there are configuration-related issues."""
    pass

class AIServiceError(DocumentationError):
    """Raised when there are issues with the Azure OpenAI service."""
    pass

class TokenLimitError(DocumentationError):
    """Raised when token limits are exceeded."""
    pass

class ProcessingError(DocumentationError):
    """Raised when there are issues processing code or documentation."""
    pass

class ValidationError(DocumentationError):
    """Raised when validation fails."""
    pass

class CacheError(DocumentationError):
    """Raised when there are caching-related issues."""
    pass

class ExtractorError(DocumentationError):
    """Raised when there are issues extracting code information."""
    pass

class ParserError(DocumentationError):
    """Raised when there are issues parsing responses or code."""
    pass
```