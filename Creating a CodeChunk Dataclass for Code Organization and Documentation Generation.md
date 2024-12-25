## code_chunk.py

```python
"""
code_chunk.py

Defines the CodeChunk dataclass for representing segments of code with associated
metadata and analysis capabilities. Provides core functionality for code
organization and documentation generation, including AST-based merging and splitting.
"""

from __future__ import annotations
import uuid
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import ast
from token_utils import TokenManager, TokenizationError, TokenizationResult
import itertools
import logging
from radon.complexity import cc_visit

# Configure logger
logger = logging.getLogger(__name__)



class ChunkType(Enum):
    """Enumeration of possible chunk types."""
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    NESTED_FUNCTION = "nested_function"
    CLASS_METHOD = "class_method"
    STATIC_METHOD = "static_method"
    PROPERTY = "property"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    DECORATOR = "decorator"


@dataclass(frozen=True)  # ChunkMetadata is immutable
class ChunkMetadata:
    """Stores metadata about a code chunk, including complexity."""
    start_line: int
    end_line: int
    chunk_type: ChunkType
    token_count: int = 0
    dependencies: Set[int] = field(default_factory=set)
    used_by: Set[int] = field(default_factory=set)
    complexity: Optional[float] = None  # Cached complexity

    def __post_init__(self):
        # Complexity is handled in CodeChunk.__post_init__
        pass  # No need for __post_init__ here


dataclass(frozen=True)
class CodeChunk:
    """Immutable representation of a code chunk with metadata."""

    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_content: str
    language: str  # Keep language attribute
    is_async: bool = False
    decorator_list: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_chunk_id: Optional[int] = None
    chunk_id: int = field(init=False)
    _chunk_counter = itertools.count()
    _tokens: Optional[List[int]] = field(default=None, init=False, repr=False)
    metadata: ChunkMetadata = field(init=False) # Only define metadata and _tokens ONCE


    @property
    def tokens(self) -> Optional[List[int]]:
        """Returns cached tokens."""
        return self._tokens

    @property
    def token_count(self) -> int:
        """Returns the number of tokens."""
        return len(self._tokens) if self._tokens else 0

    def __post_init__(self):  # Correct implementation
        """Initializes CodeChunk with tokens and complexity."""
        object.__setattr__(self, "chunk_id", next(self._chunk_counter))

        try:
            token_result: TokenizationResult = TokenManager.count_tokens(self.chunk_content, include_special_tokens=True)
            if token_result.error:
                raise ValueError(f"Token counting failed: {token_result.error}")

            object.__setattr__(self, "_tokens", token_result.tokens)

            complexity = self._calculate_complexity(self.chunk_content)

            metadata = ChunkMetadata(
                start_line=self.start_line,
                end_line=self.end_line,
                chunk_type=self._determine_chunk_type(),
                token_count=self.token_count,  # Use the property
                complexity=complexity
            )
            object.__setattr__(self, "metadata", metadata)

        except TokenizationError as e:
            logger.error(f"Tokenization error in chunk: {str(e)}")
            object.__setattr__(self, "token_count", 0)
            object.__setattr__(self, "_tokens", [])
            # Create metadata with default values in case of error
            metadata = ChunkMetadata(
                start_line=self.start_line,
                end_line=self.end_line,
                chunk_type=self._determine_chunk_type(),
                token_count=0,
                complexity=None
            )
            object.__setattr__(self, "metadata", metadata)

	 def _calculate_complexity(self, code: str) -> Optional[float]:
        """Calculates complexity using radon."""
        logger.debug(f"Calculating complexity for chunk (lines {self.start_line}-{self.end_line})")
        try:
            complexity_blocks = cc_visit(code)
            calculated_complexity = sum(block.complexity for block in complexity_blocks)
            logger.debug(f"Calculated complexity: {calculated_complexity}")
            return calculated_complexity
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return None

    def add_dependency(self, other: CodeChunk) -> CodeChunk:
        """
        Adds a dependency relationship between chunks immutably.

        Args:
            other: The CodeChunk that this chunk depends on.

        Returns:
            A new CodeChunk object with the updated dependencies.
        """
        if self.chunk_id == other.chunk_id:
            logger.warning("Attempting to add self-dependency, skipping.")
            return self

        if other.chunk_id in self.metadata.dependencies:
            logger.debug(f"Dependency {other.chunk_id} already exists for chunk {self.chunk_id}, skipping.")
            return self

        new_dependencies = self.metadata.dependencies.union({other.chunk_id})
        new_self_metadata = replace(self.metadata, dependencies=new_dependencies)
        new_self = replace(self, metadata=new_self_metadata)

        new_other_used_by = other.metadata.used_by.union({self.chunk_id})
        new_other_metadata = replace(other.metadata, used_by=new_other_used_by)
        new_other = replace(other, metadata=new_other_metadata)

        return new_self

    def _determine_chunk_type(self) -> ChunkType:
        """Determines the type of this chunk based on its properties."""
        if self.class_name and self.function_name:
            if 'staticmethod' in self.decorator_list:
                return ChunkType.STATIC_METHOD
            elif 'classmethod' in self.decorator_list:
                return ChunkType.CLASS_METHOD
            elif 'property' in self.decorator_list:
                return ChunkType.PROPERTY
            elif self.is_async:
                return ChunkType.ASYNC_METHOD
            return ChunkType.METHOD
        elif self.class_name:
            return ChunkType.CLASS
        elif self.function_name:
            if self.is_async:
                return ChunkType.ASYNC_FUNCTION
            return ChunkType.NESTED_FUNCTION if self.parent_chunk_id else ChunkType.FUNCTION
        elif self.decorator_list:  # Check for decorators before defaulting to MODULE
            return ChunkType.DECORATOR
        return ChunkType.MODULE  # Default to MODULE if no other type is identified

    def _calculate_hash(self) -> str:
        """Calculates a SHA256 hash of the chunk content."""
        return hashlib.sha256(self.chunk_content.encode('utf-8')).hexdigest()

    def get_context_string(self) -> str:
        """Returns a concise string representation of the chunk's context."""
        parts = [
            f"File: {self.file_path}",
            f"Lines: {self.start_line}-{self.end_line}"
        ]

        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.function_name:
            prefix = "Async " if self.is_async else ""
            parts.append(f"{prefix}Function: {self.function_name}")
        if self.decorator_list:
            parts.append(f"Decorators: {', '.join(self.decorator_list)}")

        return ", ".join(parts)

    def get_hierarchy_path(self) -> str:
        """Returns the full hierarchy path of the chunk."""
        parts = [Path(self.file_path).stem]  # Use stem for module name
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def can_merge_with(self, other: 'CodeChunk') -> bool:
        """
        Determines if this chunk can be merged with another using AST analysis.

        Args:
            other: Another chunk to potentially merge with.

        Returns:
            bool: True if chunks can be merged.
        """
        if not (
            self.file_path == other.file_path and
            self.language == other.language and
            self.end_line + 1 == other.start_line
        ):
            return False

        # Use AST to check if merging maintains valid syntax
        combined_content = self.chunk_content + '\n' + other.chunk_content
        try:
            ast.parse(combined_content)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def merge(chunk1: 'CodeChunk', chunk2: 'CodeChunk') -> 'CodeChunk':
        """
        Creates a new chunk by merging two chunks using AST analysis.

        Args:
            chunk1: First chunk to merge.
            chunk2: Second chunk to merge.

        Returns:
            CodeChunk: New merged chunk.

        Raises:
            ValueError: If chunks cannot be merged.
        """
        if not chunk1.can_merge_with(chunk2):
            raise ValueError("Chunks cannot be merged, AST validation failed.")

        combined_content = chunk1.chunk_content + '\n' + chunk2.chunk_content
        tokens = TokenManager.count_tokens(combined_content)

        # Aggregate dependencies and used_by
        combined_dependencies = chunk1.metadata.dependencies.union(chunk2.metadata.dependencies)
        combined_used_by = chunk1.metadata.used_by.union(chunk2.metadata.used_by)

        # Determine new chunk type
        chunk_type = chunk1.metadata.chunk_type  # Simplistic approach; can be enhanced

        # Calculate complexity
        complexity = chunk1.metadata.complexity
        if chunk2.metadata.complexity is not None:
            complexity += chunk2.metadata.complexity

        new_metadata = ChunkMetadata(
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            chunk_type=chunk_type,
            token_count=chunk1.token_count + chunk2.token_count,
            dependencies=combined_dependencies,
            used_by=combined_used_by,
            complexity=complexity
        )

        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            chunk_content=combined_content,
            token_count=tokens.token_count,
            language=chunk1.language,
            is_async=chunk1.is_async or chunk2.is_async,
            decorator_list=list(set(chunk1.decorator_list + chunk2.decorator_list)),
            docstring=chunk1.docstring or chunk2.docstring,
            parent_chunk_id=chunk1.parent_chunk_id,
            metadata=new_metadata
        )

    def get_possible_split_points(self) -> List[int]:
        """
        Returns a list of line numbers where the chunk can be split without breaking syntax.

        Returns:
            List[int]: List of valid split line numbers.
        """
        try:
            tree = ast.parse(self.chunk_content)
        except SyntaxError:
            # If the chunk has invalid syntax, it can't be split safely
            return []

        possible_split_points = []
        for node in ast.walk(tree):  # Use ast.walk to traverse all nodes
            if isinstance(node, (ast.stmt, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for statements, class definitions, and function definitions
                if hasattr(node, 'lineno'):
                    split_line = self.start_line + node.lineno - 1
                    if split_line < self.end_line:  # Avoid splitting on the last line
                        possible_split_points.append(split_line)

        return sorted(set(possible_split_points))  # Remove duplicates and sort

    def split(
        self,
        split_point: int
    ) -> List['CodeChunk']:
        """
        Splits chunk at specified line number using AST analysis.

        Args:
            split_point: Line number to split at.

        Returns:
            List[CodeChunk]: List of split chunks.

        Raises:
            ValueError: If split point is invalid or violates boundary conditions.
        """
        valid_split_points = self.get_possible_split_points()

        # Boundary Checks
        if split_point == self.start_line:
            raise ValueError(f"Cannot split at the very beginning of the chunk (line {split_point}).")
        if split_point == self.end_line:
            raise ValueError(f"Cannot split at the very end of the chunk (line {split_point}).")

        if split_point not in valid_split_points:
            raise ValueError(
                f"Invalid split point at line {split_point}. "
                f"Valid split points are: {valid_split_points}"
            )

        lines = self.chunk_content.splitlines(keepends=True)
        split_idx = split_point - self.start_line

        # Ensure split_idx is within the bounds of the lines list
        if split_idx <= 0 or split_idx >= len(lines):
            raise ValueError(
                f"Split index {split_idx} derived from split point {split_point} is out of bounds."
            )

        # Create first chunk
        chunk1_content = ''.join(lines[:split_idx])
        tokens1 = TokenManager.count_tokens(chunk1_content)

        chunk1 = CodeChunk(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=split_point - 1,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk1_content,
            token_count=tokens1.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, end_line=split_point - 1)
        )

        # Create second chunk
        chunk2_content = ''.join(lines[split_idx:])
        tokens2 = TokenManager.count_tokens(chunk2_content)

        chunk2 = CodeChunk(
            file_path=self.file_path,
            start_line=split_point,
            end_line=self.end_line,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk2_content,
            token_count=tokens2.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, start_line=split_point)
        )

        return [chunk1, chunk2]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'complexity': self.metadata.complexity,
            'token_count': self.token_count,
            'start_line': self.metadata.start_line,
            'end_line': self.metadata.end_line,
            'type': self.metadata.chunk_type.value,
            'has_docstring': self.docstring is not None,
            'is_async': self.is_async,
            'decorator_count': len(self.decorator_list),
            'has_parent': self.parent_chunk_id is not None
        }

    def __repr__(self) -> str:
        """Returns a detailed string representation of the chunk."""
        content_preview = (
            f"{self.chunk_content[:50]}..."
            if len(self.chunk_content) > 50
            else self.chunk_content
        ).replace('\n', '\\n')

        return (
            f"CodeChunk(file='{self.file_path}', "
            f"lines={self.start_line}-{self.end_line}, "
            f"type={self.metadata.chunk_type.value}, "
            f"content='{content_preview}', "
            f"tokens={self.token_count})"
        )

```

## metrics_utils.py

```python
"""
metrics_utils.py - Utility module to support metrics-related operations.

Provides helper functions and classes that assist in metrics calculations,
including metadata extraction, embedding calculations, code churn and duplication analysis,
and formatting and severity determination of metrics.
"""

import os
import json
import logging
import difflib
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps, partial
from time import perf_counter
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Iterable

from concurrent.futures import ThreadPoolExecutor

# External dependencies
from git import Repo, GitError, InvalidGitRepositoryError, NoSuchPathError
from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.decomposition import PCA
import ast
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --------------------------
# Custom Exceptions
# --------------------------

class BareRepositoryError(Exception):
    """Custom exception for bare repositories."""
    pass

# --------------------------
# Data Classes
# --------------------------

@dataclass
class CodeMetadata:
    """Metadata extracted from code."""
    function_name: str
    variable_names: List[str]
    complexity: float
    halstead_volume: float
    maintainability_index: float

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics."""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1000000.0
    code_churn_high: int = 1000
    code_churn_warning: int = 500
    code_duplication_high: float = 30.0  # Percentage
    code_duplication_warning: float = 10.0

    @classmethod
    def from_dict(cls, thresholds_dict: Dict[str, Any]) -> 'MetricsThresholds':
        """Creates a MetricsThresholds instance from a dictionary with validation."""
        validated_dict = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in thresholds_dict:
                value = thresholds_dict[field_name]
                if not isinstance(value, field_type):  # Simplified type checking
                    try:
                        validated_dict[field_name] = field_type(value)  # Attempt type conversion
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid value for threshold '{field_name}': {value} - {e}") from e
                else:
                    validated_dict[field_name] = value

        # Additional validation for percentage fields
        percentage_fields = ['code_duplication_high', 'code_duplication_warning']
        for pf in percentage_fields:
            if pf in validated_dict:
                pct = validated_dict[pf]
                if not (0.0 <= pct <= 100.0):
                    raise ValueError(f"Threshold '{pf}' must be between 0 and 100.")

        return cls(**validated_dict)

    @classmethod
    def load_from_file(cls, config_path: str) -> 'MetricsThresholds':
        """Loads metric thresholds from a JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                thresholds_dict = json.load(f)
                logger.debug(f"Loaded thresholds from {config_path}: {thresholds_dict}")
            return cls.from_dict(thresholds_dict)
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except ValueError as e:  # Catch validation errors
            logger.error(f"Invalid threshold values in config: {e}")
            raise

# --------------------------
# Decorators
# --------------------------

T = TypeVar('T')  # Generic type variable for the return value

def safe_metric_calculation(default_value: T = None, metric_name: str = "metric") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for safe metric calculation with specific error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                error_message = f"ValueError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except TypeError as e:
                error_message = f"TypeError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except Exception as e:
                error_message = f"Unexpected error during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
        return wrapper
    return decorator

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = perf_counter() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

# --------------------------
# Helper Functions
# --------------------------

def get_metric_severity(metric_name: str, value: Union[int, float], thresholds: MetricsThresholds) -> str:
    """
    Determines the severity level of a metric value based on thresholds.
    """
    logger.debug(f"Evaluating severity for metric '{metric_name}' with value {value}")
    severity = "normal"

    if metric_name == "complexity":
        if value >= thresholds.complexity_high:
            severity = "high"
        elif value >= thresholds.complexity_warning:
            severity = "warning"
    elif metric_name == "maintainability_index":
        if value < thresholds.maintainability_low:
            severity = "low"
    elif metric_name == "halstead_effort":
        if value > thresholds.halstead_effort_high:
            severity = "high"
    elif metric_name == "code_churn":
        if value >= thresholds.code_churn_high:
            severity = "high"
        elif value >= thresholds.code_churn_warning:
            severity = "warning"
    elif metric_name == "code_duplication":
        if value >= thresholds.code_duplication_high:
            severity = "high"
        elif value >= thresholds.code_duplication_warning:
            severity = "warning"

    logger.debug(f"Severity determined: {severity}")
    return severity

def format_metric_value(metric_name: str, value: Union[int, float]) -> str:
    """
    Formats metric values for display.
    """
    logger.debug(f"Formatting metric '{metric_name}' with value {value}")

    if metric_name in ["maintainability_index", "complexity"]:
        formatted = f"{value:.2f}"
    elif metric_name == "halstead_effort":
        formatted = f"{value:,.0f}"
    elif metric_name == "code_churn":
        formatted = f"{int(value)} lines"
    elif metric_name == "code_duplication":
        formatted = f"{value:.2f}%"
    else:
        formatted = str(value)

    logger.debug(f"Formatted value: {formatted}")
    return formatted

# --------------------------
# Embedding Calculators
# --------------------------

class MultiLayerEmbeddingCalculator:
    """Generates multi-layer embeddings from code using transformer models."""

    def __init__(self, model_name: str = 'microsoft/codebert-base', pca_components: int = 384):
        logger.debug(f"Initializing MultiLayerEmbeddingCalculator with model '{model_name}' and PCA components {pca_components}")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pca = PCA(n_components=pca_components)
        self.layer_weights = [0.1, 0.2, 0.3, 0.4]
        self._fit_pca()

    def _fit_pca(self, sample_embeddings: Optional[List[np.ndarray]] = None):
        """
        Fit PCA on a sample of embeddings to initialize PCA transformation.
        This method should be called with representative data to capture variance.
        """
        # Placeholder for fitting PCA. In practice, you should fit on a large representative dataset.
        logger.debug("Fitting PCA on sample embeddings.")
        if sample_embeddings:
            combined = np.vstack(sample_embeddings)
            self.pca.fit(combined)
            logger.debug("PCA fitting completed.")
        else:
            # Fit PCA with random data if no samples provided (not ideal)
            random_data = np.random.rand(100, self.pca.n_components * 10)
            self.pca.fit(random_data)
            logger.debug("PCA fitted with random data as placeholder.")

    def calculate_multi_layer_embedding(self, code: str) -> np.ndarray:
        """
        Calculate a multi-layer embedding for the given code snippet.
        """
        logger.debug("Calculating multi-layer embedding.")
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_four_layers = outputs.hidden_states[-4:]
        combined_embedding = torch.zeros_like(last_four_layers[0][0])
        for layer, weight in zip(last_four_layers, self.layer_weights):
            combined_embedding += weight * layer[0]

        combined_embedding = torch.mean(combined_embedding, dim=0)
        embedding_np = combined_embedding.numpy()
        reduced_embedding = self.pca.transform(embedding_np.reshape(1, -1))

        logger.debug("Multi-layer embedding calculated and reduced.")
        return reduced_embedding.flatten()

class EnhancedEmbeddingCalculator:
    """Combines various embeddings and metadata into a unified embedding vector."""

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        code_model: str = 'microsoft/codebert-base',
        pca_components: int = 384
    ):
        logger.debug(f"Initializing EnhancedEmbeddingCalculator with embedding model '{embedding_model}' and code model '{code_model}'")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.multi_layer_calculator = MultiLayerEmbeddingCalculator(code_model, pca_components)
        self.metadata_weights = {
            'content': 0.3,
            'multi_layer': 0.3,
            'function_name': 0.1,
            'variable_names': 0.1,
            'complexity': 0.05,
            'halstead_volume': 0.05,
            'maintainability_index': 0.1
        }

    def calculate_enhanced_embedding(self, code: str, metadata: CodeMetadata) -> np.ndarray:
        """
        Combine various embeddings and metadata into a single normalized embedding vector.
        """
        logger.debug("Calculating enhanced embedding.")
        content_embedding = self.embedding_model.encode(code)
        multi_layer_embedding = self.multi_layer_calculator.calculate_multi_layer_embedding(code)
        function_name_embedding = self.embedding_model.encode(metadata.function_name)
        variable_names_embedding = self.embedding_model.encode(" ".join(metadata.variable_names))

        complexity_norm = self._normalize_value(metadata.complexity, 0, 50)
        halstead_volume_norm = self._normalize_value(metadata.halstead_volume, 0, 1000)
        maintainability_index_norm = self._normalize_value(metadata.maintainability_index, 0, 100)

        combined_embedding = (
            self.metadata_weights['content'] * content_embedding +
            self.metadata_weights['multi_layer'] * multi_layer_embedding +
            self.metadata_weights['function_name'] * function_name_embedding +
            self.metadata_weights['variable_names'] * variable_names_embedding +
            self.metadata_weights['complexity'] * complexity_norm * np.ones_like(content_embedding) +
            self.metadata_weights['halstead_volume'] * halstead_volume_norm * np.ones_like(content_embedding) +
            self.metadata_weights['maintainability_index'] * maintainability_index_norm * np.ones_like(content_embedding)
        )

        norm = np.linalg.norm(combined_embedding)
        if norm == 0:
            logger.warning("Combined embedding norm is zero. Returning zero vector.")
            return combined_embedding
        normalized_embedding = combined_embedding / norm

        logger.debug("Enhanced embedding calculated and normalized.")
        return normalized_embedding

    def _normalize_value(self, value: float, min_value: float, max_value: float) -> float:
        """Normalize a value to a 0-1 range based on provided min and max."""
        normalized = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0
        logger.debug(f"Normalized value: {normalized} (value: {value}, min: {min_value}, max: {max_value})")
        return normalized

    def set_metadata_weights(self, new_weights: Dict[str, float]) -> None:
        """Update the weights for metadata features."""
        logger.debug(f"Setting new metadata weights: {new_weights}")
        if not np.isclose(sum(new_weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
        self.metadata_weights.update(new_weights)
        logger.debug("Metadata weights updated.")

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        similarity = np.dot(embedding1, embedding2)
        logger.debug(f"Calculated similarity: {similarity}")
        return similarity

class EmbeddingManager:
    """Manages embedding calculations and similarity assessments."""

    def __init__(self, embedding_calculator: Optional[EnhancedEmbeddingCalculator] = None):
        self.embedding_calculator = embedding_calculator or EnhancedEmbeddingCalculator()

    def get_embedding(self, code: str, metadata: CodeMetadata) -> np.ndarray:
        """Generates an enhanced embedding for the given code and metadata."""
        embedding = self.embedding_calculator.calculate_enhanced_embedding(code, metadata)
        logger.debug(f"Generated embedding: {embedding}")
        return embedding

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculates the similarity between two embeddings."""
        similarity = self.embedding_calculator.calculate_similarity(embedding1, embedding2)
        logger.debug(f"Similarity between embeddings: {similarity}")
        return similarity

# --------------------------
# Code Churn and Duplication Analysis
# --------------------------

class CodeChurnAnalyzer:
    """Analyzes code churn in a Git repository."""

    def __init__(self, repo_path: str, since_days: int = 30):
        self.repo_path = repo_path
        self.since = datetime.now() - timedelta(days=since_days)
        self.repo = self._load_repo()

    def _load_repo(self) -> Repo:
        """Loads the Git repository."""
        try:
            repo = Repo(self.repo_path)
            if repo.bare:
                logger.error("Cannot analyze churn for a bare repository.")
                raise BareRepositoryError("Repository is bare.")
            logger.debug(f"Loaded Git repository from '{self.repo_path}'")
            return repo
        except (NoSuchPathError, InvalidGitRepositoryError) as e:
            logger.error(f"Error accessing Git repository: {e}")
            raise ValueError(f"Invalid Git repository: {self.repo_path}") from e

    def calculate_code_churn(self) -> int:
        """
        Calculates code churn by summing insertions and deletions in Git commits since a given date.
        """
        logger.debug(f"Calculating code churn since {self.since.isoformat()}")
        churn = 0
        try:
            commits = list(self.repo.iter_commits(since=self.since))
            logger.debug(f"Found {len(commits)} commits since {self.since.isoformat()}")
            for commit in commits:
                insertions = commit.stats.total.get('insertions', 0)
                deletions = commit.stats.total.get('deletions', 0)
                churn += insertions + deletions
                logger.debug(f"Commit {commit.hexsha}: +{insertions} -{deletions}")
            logger.info(f"Total code churn: {churn} lines")
            return churn
        except GitError as e:
            logger.error(f"Git error while calculating churn: {e}")
            raise

def calculate_code_duplication(
    repo_path: str,
    extensions: Optional[Iterable[str]] = None,
    min_duplication_block_size: int = 5
) -> float:
    """
    Calculates code duplication percentage by comparing file contents using difflib.

    Note: This implementation has O(n^2) time complexity and may be inefficient for large repositories.
    Consider using specialized tools like jscpd or SonarQube for better performance.
    """
    logger.debug(f"Calculating code duplication for repository at '{repo_path}' with extensions {extensions}.")

    duplicated_lines = 0
    total_lines = 0
    file_content_batches = []

    if extensions is None:
        extensions = {".py"}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                        # Create batches of lines
                        for i in range(0, len(content), 1000):
                            batch = content[i:i + 1000]
                            file_content_batches.append((file_path, batch))
                            total_lines += len(batch)
                            logger.debug(f"Read {len(batch)} lines (batch) from {file_path}")
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Skipped file '{file_path}' due to error: {e}")
                    continue  # Skip unreadable files

    num_batches = len(file_content_batches)
    logger.debug(f"Total batches to compare: {num_batches}")

    for i in range(num_batches):
        for j in range(i + 1, num_batches):  # Compare each batch with every other batch
            _, batch1 = file_content_batches[i]
            _, batch2 = file_content_batches[j]

            seq = difflib.SequenceMatcher(None, batch1, batch2)
            for block in seq.get_matching_blocks():
                if block.size >= min_duplication_block_size:
                    duplicated_lines += block.size
                    logger.debug(f"Found duplicated block of size {block.size} between batches {i} and {j}")

    duplication_percentage = (duplicated_lines / total_lines * 100) if total_lines > 0 else 0.0
    logger.info(f"Total duplicated lines: {duplicated_lines} out of {total_lines} lines")
    logger.info(f"Duplication percentage: {duplication_percentage:.2f}%")

    return duplication_percentage

# --------------------------
# Metadata Extraction
# --------------------------

def extract_metadata(code: str, metrics: Dict[str, Any]) -> CodeMetadata:
    """
    Extract metadata such as function name and variable names from code using AST.
    """
    logger.debug("Extracting metadata from code using AST.")
    function_name = ""
    variable_names = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                for arg in node.args.args:
                    variable_names.append(arg.arg)
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                variable_names.append(target.id)
    except SyntaxError as e:
        logger.warning(f"Syntax error while parsing code: {e}")

    complexity = metrics.get('cyclomatic_complexity', 0)
    halstead_volume = metrics.get('halstead', {}).get('volume', 0.0)
    maintainability_index = metrics.get('maintainability_index', 0.0)

    metadata = CodeMetadata(
        function_name=function_name,
        variable_names=variable_names,
        complexity=complexity,
        halstead_volume=halstead_volume,
        maintainability_index=maintainability_index
    )

    logger.debug(f"Metadata extracted: {metadata}")
    return metadata

# --------------------------
# Embedding Integration
# --------------------------

def calculate_embedding_metrics(code: str, embedding_calculator: EnhancedEmbeddingCalculator, metadata: CodeMetadata) -> np.ndarray:
    """
    Calculates the enhanced embedding for a given code snippet using provided metadata.
    """
    logger.debug("Calculating embedding metrics.")
    embedding = embedding_calculator.calculate_enhanced_embedding(code, metadata)
    logger.debug(f"Enhanced embedding calculated: {embedding}")
    return embedding

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Utilities Example Usage")
    parser.add_argument('--repo_path', type=str, required=True, help='Path to the Git repository.')
    parser.add_argument('--thresholds', type=str, default='thresholds.json', help='Path to thresholds JSON file.')
    args = parser.parse_args()

    # Load thresholds
    try:
        thresholds = MetricsThresholds.load_from_file(args.thresholds)
    except Exception as e:
        logger.error(f"Failed to load thresholds: {e}")
        thresholds = MetricsThresholds()  # Use default thresholds

    # Initialize analyzers and calculators
    churn_analyzer = CodeChurnAnalyzer(repo_path=args.repo_path, since_days=30)
    embedding_calculator = EnhancedEmbeddingCalculator()
    embedding_manager = EmbeddingManager(embedding_calculator=embedding_calculator)

    # Calculate code churn
    try:
        churn = churn_analyzer.calculate_code_churn()
        severity = get_metric_severity("code_churn", churn, thresholds)
        formatted_churn = format_metric_value("code_churn", churn)
        logger.info(f"Code Churn: {formatted_churn} (Severity: {severity})")
    except Exception as e:
        logger.error(f"Error calculating code churn: {e}")

    # Calculate code duplication
    try:
        duplication = calculate_code_duplication(args.repo_path)
        severity = get_metric_severity("code_duplication", duplication, thresholds)
        formatted_duplication = format_metric_value("code_duplication", duplication)
        logger.info(f"Code Duplication: {formatted_duplication} (Severity: {severity})")
    except Exception as e:
        logger.error(f"Error calculating code duplication: {e}")

    # Example: Calculate embedding for a specific file
    example_file_path = os.path.join(args.repo_path, 'example.py')  # Replace with an actual file path
    if os.path.exists(example_file_path):
        try:
            with open(example_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                # Assume that metrics.py has been used to calculate metrics for this code
                # Here, we simulate metrics for demonstration purposes
                metrics = {
                    'cyclomatic_complexity': 12.0,
                    'maintainability_index': 45.0,
                    'halstead': {'effort': 50000.0}
                }
                metadata = extract_metadata(code, metrics)
                embedding = calculate_embedding_metrics(code, embedding_calculator, metadata)
                logger.info(f"Enhanced Embedding for {example_file_path}: {embedding}")
        except Exception as e:
            logger.error(f"Error calculating embedding for {example_file_path}: {e}")
    else:
        logger.warning(f"Example file '{example_file_path}' does not exist.")


```

## token_utils.py

```python
# token_utils.py

"""
token_utils.py - Enhanced tokenization utilities with embedding capabilities.

This module provides utilities for tokenizing text using various tokenizer models,
counting tokens, decoding tokens back to text, handling special tokens, validating
token limits, splitting text based on token limits, and batch token processing.
Additionally, it integrates advanced embedding calculations and similarity measurements
for enhanced text analysis.
"""

import tiktoken
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import logging
from enum import Enum
import threading
import numpy as np
from metrics_utils import EnhancedEmbeddingCalculator, CodeMetadata

logger = logging.getLogger(__name__)

class TokenizerModel(Enum):
    """Supported tokenizer models."""
    GPT4 = "cl100k_base"
    GPT3 = "p50k_base"
    CODEX = "p50k_edit"

@dataclass
class TokenizationResult:
    """Results from a tokenization operation."""
    tokens: List[int]
    token_count: int
    encoding_name: str
    special_tokens: Dict[str, int] = None
    error: Optional[str] = None

class TokenizationError(Exception):
    """Custom exception for tokenization errors."""
    pass

class TokenManager:
    """
    Manages tokenization operations with caching, thread-safety, and embedding functionalities.

    Features:
        - Token counting and decoding.
        - Handling of special tokens.
        - Validation against token limits.
        - Splitting text based on token limits.
        - Batch token processing.
        - Enhanced embedding calculations and similarity measurements.
    """

    _encoders = {}  # Cache for different encoders
    _default_model = TokenizerModel.GPT4
    _lock = threading.Lock()  # Lock for thread safety
    _embedding_calculator: Optional[EnhancedEmbeddingCalculator] = None  # Lazy initialization

    @classmethod
    def get_encoder(cls, model: TokenizerModel = None) -> tiktoken.Encoding:
        """
        Retrieves the appropriate encoder instance.

        Args:
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            tiktoken.Encoding: Encoder instance.

        Raises:
            TokenizationError: If encoder creation fails.
        """
        with cls._lock:  # Ensure thread-safe access
            try:
                model = model or cls._default_model
                if model not in cls._encoders:
                    logger.debug(f"Creating new encoder for model: {model.value}")
                    cls._encoders[model] = tiktoken.get_encoding(model.value)
                return cls._encoders[model]
            except Exception as e:
                logger.error(f"Failed to create encoder for model {model}: {e}")
                raise TokenizationError(f"Failed to create encoder: {str(e)}")

    @classmethod
    def count_tokens(
        cls,
        text: Union[str, List[str]],
        model: TokenizerModel = None,
        include_special_tokens: bool = False
    ) -> TokenizationResult:
        """
        Counts tokens in the provided text with enhanced error handling.

        Args:
            text (Union[str, List[str]]): Text to tokenize.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.
            include_special_tokens (bool, optional): Whether to count special tokens. Defaults to False.

        Returns:
            TokenizationResult: Tokenization results.

        Raises:
            TokenizationError: If tokenization fails.
        """
        logger.debug(f"Counting tokens for text: {text[:50]}...")  # Truncate text for logging
        try:
            if not text:
                logger.warning("Empty input provided for token counting.")
                return TokenizationResult([], 0, "", error="Empty input")

            encoder = cls.get_encoder(model)  # Get encoder (thread-safe)
            model = model or cls._default_model

            if isinstance(text, list):
                text = " ".join(text)

            tokens = encoder.encode(text)
            logger.debug(f"Encoded tokens: {tokens[:10]}...")  # Truncate tokens for logging

            special_tokens = None
            if include_special_tokens:
                special_tokens = cls._count_special_tokens(text, encoder)

            return TokenizationResult(
                tokens=tokens,
                token_count=len(tokens),
                encoding_name=model.value,
                special_tokens=special_tokens
            )

        except TokenizationError:
            raise  # Re-raise custom exceptions without modification
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise TokenizationError(f"Failed to count tokens: {str(e)}")

    @classmethod
    def decode_tokens(
        cls,
        tokens: List[int],
        model: TokenizerModel = None
    ) -> str:
        """
        Decodes a list of tokens back to text.

        Args:
            tokens (List[int]): List of token IDs.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            str: Decoded text.

        Raises:
            TokenizationError: If decoding fails.
        """
        logger.debug(f"Decoding tokens: {tokens[:10]}...")  # Truncate tokens for logging
        try:
            if not tokens:
                logger.warning("Empty token list provided for decoding.")
                return ""

            encoder = cls.get_encoder(model)
            decoded_text = encoder.decode(tokens)
            logger.debug(f"Decoded text: {decoded_text[:50]}...")  # Truncate text for logging
            return decoded_text

        except TokenizationError:
            raise
        except Exception as e:
            logger.error(f"Token decoding error: {e}")
            raise TokenizationError(f"Failed to decode tokens: {str(e)}")

    @classmethod
    def _count_special_tokens(
        cls,
        text: str,
        encoder: tiktoken.Encoding
    ) -> Dict[str, int]:
        """
        Counts special tokens in the text.

        Args:
            text (str): Text to analyze.
            encoder (tiktoken.Encoding): Encoder instance.

        Returns:
            Dict[str, int]: Counts of special tokens.
        """
        special_tokens = {}
        try:
            # Example: count newlines and code blocks
            special_tokens["newlines"] = text.count("\n")
            special_tokens["code_blocks"] = text.count("```")
            logger.debug(f"Special tokens counted: {special_tokens}")
            return special_tokens
        except Exception as e:
            logger.warning(f"Error counting special tokens: {e}")
            return {}

    @classmethod
    def validate_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> bool:
        """
        Checks if the text exceeds the specified token limit.

        Args:
            text (str): Text to check.
            max_tokens (int): Maximum allowed tokens.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            bool: True if within limit, False otherwise.
        """
        try:
            result = cls.count_tokens(text, model)
            logger.debug(f"Token count {result.token_count} compared to max {max_tokens}")
            return result.token_count <= max_tokens
        except TokenizationError as e:
            logger.error(f"Validation failed: {e}")
            return False

    @classmethod
    def split_by_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> List[str]:
        """
        Splits the text into chunks, each not exceeding the specified token limit.

        Args:
            text (str): Text to split.
            max_tokens (int): Maximum tokens per chunk.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            List[str]: List of text chunks.

        Raises:
            TokenizationError: If splitting fails.
        """
        logger.debug(f"Splitting text with max tokens per chunk: {max_tokens}")
        try:
            encoder = cls.get_encoder(model)
            tokens = encoder.encode(text)
            chunks = []
            current_chunk = []
            current_count = 0

            for token in tokens:
                if current_count + 1 > max_tokens:
                    decoded_chunk = encoder.decode(current_chunk)
                    chunks.append(decoded_chunk)
                    logger.debug(f"Created chunk with {current_count} tokens.")
                    current_chunk = []
                    current_count = 0

                current_chunk.append(token)
                current_count += 1

            if current_chunk:
                decoded_chunk = encoder.decode(current_chunk)
                chunks.append(decoded_chunk)
                logger.debug(f"Created final chunk with {current_count} tokens.")

            logger.info(f"Total chunks created: {len(chunks)}")
            return chunks

        except TokenizationError:
            raise
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise TokenizationError(f"Failed to split text: {str(e)}")

    @classmethod
    def estimate_tokens_from_chars(cls, text: str) -> int:
        """
        Provides a rough estimation of token count based on character length.

        Useful for quick checks before full tokenization.

        Args:
            text (str): Text to estimate.

        Returns:
            int: Estimated token count.
        """
        estimate = len(text) // 4  # GPT models average ~4 characters per token
        logger.debug(f"Estimated tokens from characters: {estimate}")
        return estimate

    @classmethod
    def batch_count_tokens(
        cls,
        texts: List[str],
        model: TokenizerModel = None
    ) -> List[TokenizationResult]:
        """
        Counts tokens for multiple texts efficiently.

        Args:
            texts (List[str]): List of texts to tokenize.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            List[TokenizationResult]: Results for each text.
        """
        logger.debug(f"Batch counting tokens for {len(texts)} texts.")
        results = []
        try:
            encoder = cls.get_encoder(model)

            for idx, text in enumerate(texts):
                try:
                    tokens = encoder.encode(text)
                    result = TokenizationResult(
                        tokens=tokens,
                        token_count=len(tokens),
                        encoding_name=model.value if model else cls._default_model.value
                    )
                    results.append(result)
                    logger.debug(f"Text {idx+1}: {result.token_count} tokens.")
                except Exception as e:
                    logger.error(f"Error in batch tokenization for text {idx+1}: {e}")
                    results.append(TokenizationResult(
                        tokens=[],
                        token_count=0,
                        encoding_name="",
                        error=str(e)
                    ))

            return results

        except TokenizationError as e:
            logger.error(f"Batch tokenization failed: {e}")
            # Return empty results with errors
            return [TokenizationResult(
                        tokens=[],
                        token_count=0,
                        encoding_name="",
                        error=str(e)
                    ) for _ in texts]

    @classmethod
    def clear_cache(cls):
        """
        Clears the encoder cache.
        """
        with cls._lock:
            cls._encoders.clear()
            logger.info("Encoder cache cleared.")

    @classmethod
    def _initialize_embedding_calculator(cls):
        """
        Initializes the EnhancedEmbeddingCalculator if not already initialized.
        """
        if cls._embedding_calculator is None:
            logger.debug("Initializing EnhancedEmbeddingCalculator.")
            cls._embedding_calculator = EnhancedEmbeddingCalculator()

    @classmethod
    def get_enhanced_embedding(cls, code: str, metadata: CodeMetadata) -> np.ndarray:
        """
        Generates an enhanced embedding for the given code and metadata.

        Args:
            code (str): Code snippet to embed.
            metadata (CodeMetadata): Metadata associated with the code.

        Returns:
            np.ndarray: Enhanced embedding vector.

        Raises:
            TokenizationError: If embedding calculation fails.
        """
        logger.debug(f"Generating enhanced embedding for code: {code[:50]}...")
        try:
            cls._initialize_embedding_calculator()
            embedding = cls._embedding_calculator.calculate_enhanced_embedding(code, metadata)
            logger.debug(f"Generated embedding of shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate enhanced embedding: {e}")
            raise TokenizationError(f"Failed to generate enhanced embedding: {str(e)}")

    @classmethod
    def calculate_similarity(cls, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates the similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.

        Returns:
            float: Similarity score.

        Raises:
            TokenizationError: If similarity calculation fails.
        """
        logger.debug("Calculating similarity between two embeddings.")
        try:
            cls._initialize_embedding_calculator()
            similarity = cls._embedding_calculator.calculate_similarity(embedding1, embedding2)
            logger.debug(f"Calculated similarity: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            raise TokenizationError(f"Failed to calculate similarity: {str(e)}")

    @classmethod
    def set_metadata_weights(cls, new_weights: Dict[str, float]) -> None:
        """
        Sets new weights for metadata features in embedding calculations.

        Args:
            new_weights (Dict[str, float]): New weights for metadata features.

        Raises:
            TokenizationError: If setting weights fails.
        """
        logger.debug(f"Setting new metadata weights: {new_weights}")
        try:
            cls._initialize_embedding_calculator()
            cls._embedding_calculator.set_metadata_weights(new_weights)
            logger.info("Metadata weights updated successfully.")
        except Exception as e:
            logger.error(f"Failed to set metadata weights: {e}")
            raise TokenizationError(f"Failed to set metadata weights: {str(e)}")

```

## utils.py

```python
"""
utils.py

Core utility functions and classes for code processing, file handling,
metrics calculations, and token management. Provides the foundational
functionality for the documentation generation system.
"""

import os
import sys
import json
import logging
import asyncio
import tiktoken
import pathspec
import subprocess
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Set, Any, Optional, Union, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
import tokenize
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import coverage
from coverage.files import PathAliases
from coverage.misc import CoverageException
import aiofiles
from aiofiles import os as aio_os
import re
from math import log2  # Add this for maintainability index calculation


# Try importing lizard first, fall back to radon if not available
try:
    import lizard
    USE_LIZARD = True
except ImportError:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
    USE_LIZARD = False

logger = logging.getLogger(__name__)

# Default configurations for metrics thresholds
DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 10, "medium": 20, "high": 30}
DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 100, "medium": 500, "high": 1000},
    "difficulty": {"low": 10, "medium": 20, "high": 30},
    "effort": {"low": 500, "medium": 1000, "high": 2000}
}
DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 50, "medium": 70, "high": 85}

# Enhanced default patterns with more comprehensive coverage
DEFAULT_EXCLUDED_PATTERNS = {
    'dirs': {
        # Version Control
        '.git', '.svn', '.hg', '.bzr',
        # Python
        '__pycache__', '.pytest_cache', '.mypy_cache', '.coverage',
        'htmlcov', '.tox', '.nox',
        # Virtual Environments
        'venv', '.venv', 'env', '.env', 'virtualenv',
        # Node.js
        'node_modules',
        # Build/Distribution
        'build', 'dist', '.eggs', '*.egg-info',
        # IDE
        '.idea', '.vscode', '.vs', '.settings',
        # Other
        'tmp', 'temp', '.tmp', '.temp'
    },
    'files': {
        # System
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        # Python
        '*.pyc', '*.pyo', '*.pyd', '.python-version',
        '.coverage', 'coverage.xml', '.coverage.*',
        # Package Management
        'pip-log.txt', 'pip-delete-this-directory.txt',
        'poetry.lock', 'Pipfile.lock',
        # Environment
        '.env', '.env.*',
        # IDE
        '*.swp', '*.swo', '*~',
        # Build
        '*.spec', '*.manifest',
        # Documentation
        '*.pdf', '*.doc', '*.docx',
        # Other
        '*.log', '*.bak', '*.tmp'
    },
    'extensions': {
        # Python
        '.pyc', '.pyo', '.pyd', '.so',
        # Compilation
        '.o', '.obj', '.dll', '.dylib',
        # Package
        '.egg', '.whl',
        # Cache
        '.cache',
        # Documentation
        '.pdf', '.doc', '.docx',
        # Media
        '.jpg', '.jpeg', '.png', '.gif', '.ico',
        '.mov', '.mp4', '.avi',
        '.mp3', '.wav',
        # Archives
        '.zip', '.tar.gz', '.tgz', '.rar'
    }
}

# Language mappings with metadata
LANGUAGE_MAPPING = {
    ".py": {
        "name": "python",
        "comment_symbol": "#",
        "doc_strings": ['"""', "'''"],
        "supports_type_hints": True
    },
    ".js": {
        "name": "javascript",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": False
    },
    ".ts": {
        "name": "typescript",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    },
    ".java": {
        "name": "java",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    },
    ".go": {
        "name": "go",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    }
}

# Custom Exceptions
class MetricsError(Exception):
    """Base exception for metrics-related errors."""
    pass

class FileOperationError(Exception):
    """Base exception for file operation errors."""
    pass

class CoverageFormatError(Exception):
    """Raised when coverage format is invalid or unrecognized."""
    pass

# Data Classes
@dataclass
class TokenResult:
    """Contains token analysis results."""
    tokens: List[str]
    token_count: int
    encoding_name: str
    special_tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None

@dataclass
class ComplexityMetrics:
    """Container for complexity metrics with default values."""
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 100.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0
    halstead_time: float = 0.0
    type_hint_coverage: float = 0.0

@dataclass
class CoverageData:
    """Container for coverage metrics."""
    line_rate: float = 0.0
    branch_rate: float = 0.0
    complexity: float = 0.0
    timestamp: str = ""
    source_file: str = ""

# Core Utility Classes
class TokenManager:
    """Manages token counting and analysis with caching."""
    
    _encoders = {}
    _lock = threading.Lock()
    _cache = {}
    _max_cache_size = 1000
    
    @classmethod
    def get_encoder(cls, model_name: str = "gpt-4") -> Any:
        """Gets or creates a tiktoken encoder with thread safety."""
        with cls._lock:
            if model_name not in cls._encoders:
                try:
                    if model_name.startswith("gpt-4"):
                        encoding_name = "cl100k_base"
                    elif model_name.startswith("gpt-3"):
                        encoding_name = "p50k_base"
                    else:
                        encoding_name = "cl100k_base"  # default
                    
                    cls._encoders[model_name] = tiktoken.get_encoding(encoding_name)
                except Exception as e:
                    logger.error(f"Error creating encoder for {model_name}: {e}")
                    raise
            
            return cls._encoders[model_name]

    @classmethod
    def count_tokens(
        cls,
        text: str,
        model_name: str = "gpt-4",
        use_cache: bool = True
    ) -> TokenResult:
        """Counts tokens in text using tiktoken with caching."""
        if not text:
            return TokenResult([], 0, "", error="Empty text")
            
        if use_cache:
            cache_key = hash(text + model_name)
            if cache_key in cls._cache:
                return cls._cache[cache_key]
        
        try:
            encoder = cls.get_encoder(model_name)
            tokens = encoder.encode(text)
            special_tokens = cls._count_special_tokens(text)
            
            result = TokenResult(
                tokens=tokens,
                token_count=len(tokens),
                encoding_name=encoder.name,
                special_tokens=special_tokens
            )
            
            if use_cache:
                with cls._lock:
                    if len(cls._cache) >= cls._max_cache_size:
                        cls._cache.pop(next(iter(cls._cache)))
                    cls._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return TokenResult([], 0, "", error=str(e))

    @staticmethod
    def _count_special_tokens(text: str) -> Dict[str, int]:
        """Counts special tokens like newlines and code blocks."""
        return {
            "newlines": text.count("\n"),
            "code_blocks": text.count("```"),
            "inline_code": text.count("`") - (text.count("```") * 3)
        }

class FileHandler:
    """Handles file operations with caching and error handling."""
    
    _content_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 100
    _executor = ThreadPoolExecutor(max_workers=4)
    
    @classmethod
    async def read_file(
        cls,
        file_path: Union[str, Path],
        use_cache: bool = True,
        encoding: str = 'utf-8'
    ) -> Optional[str]:
        """Reads file content asynchronously with caching."""
        file_path = str(file_path)
        
        try:
            if use_cache:
                with cls._cache_lock:
                    cache_key = f"{file_path}:{encoding}"
                    if cache_key in cls._content_cache:
                        return cls._content_cache[cache_key]
        
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
        
            if use_cache:
                with cls._cache_lock:
                    if len(cls._content_cache) >= cls._max_cache_size:
                        cls._content_cache.pop(next(iter(cls._content_cache)))
                    cls._content_cache[cache_key] = content
        
            return content
            
        except UnicodeDecodeError:
            logger.warning(
                f"UnicodeDecodeError for {file_path} with {encoding}, "
                "trying with error handling"
            )
            try:
                async with aiofiles.open(
                    file_path,
                    'r',
                    encoding=encoding,
                    errors='replace'
                ) as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with robust error handling."""

    def __init__(self):
        self.using_lizard = USE_LIZARD
        logger.info(f"Using {'lizard' if self.using_lizard else 'radon'} for metrics calculation")

    def calculate_metrics(
        self,
        code: str,
        file_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive code metrics.
        
        Args:
            code: Source code to analyze
            file_path: Optional file path for coverage data
            language: Programming language
            
        Returns:
            Dict containing calculated metrics
        """
        try:
            metrics = ComplexityMetrics()
            
            if self.using_lizard:
                metrics = self._calculate_lizard_metrics(code)
            else:
                metrics = self._calculate_radon_metrics(code)

            # Add language-specific metrics if available
            if language:
                self._add_language_metrics(metrics, code, language)

            return self._prepare_metrics_output(metrics)

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._prepare_metrics_output(ComplexityMetrics())
            
    def _calculate_maintainability_index(self, code: str) -> float:
        """
        Calculates maintainability index using a modified version of the SEI formula.
        
        The formula considers:
        - Lines of code
        - Cyclomatic complexity
        - Halstead volume
        
        Args:
            code (str): Source code to analyze
            
        Returns:
            float: Maintainability index (0-100)
        """
        try:
            # Count lines of code (excluding empty lines and comments)
            lines = [
                line.strip()
                for line in code.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]
            loc = len(lines)
            
            if loc == 0:
                return 100.0
            
            # Calculate average line length as a complexity factor
            avg_line_length = sum(len(line) for line in lines) / loc
            
            # Count control structures as a basic complexity measure
            control_structures = len(re.findall(
                r'\b(if|else|elif|for|while|try|except|with)\b',
                code
            ))
            
            # Basic Halstead volume approximation
            operators = len(re.findall(
                r'[\+\-\*/=<>!&|%]+|and|or|not|in|is',
                code
            ))
            operands = len(re.findall(r'\b[a-zA-Z_]\w*\b', code))
            
            # Modified SEI formula
            vol = (operators + operands) * log2(operators + operands) if operators + operands > 0 else 0
            cc = control_structures / loc
            
            mi = 171 - 5.2 * log2(vol + 1) - 0.23 * cc * 100 - 16.2 * log2(loc)
            
            # Normalize to 0-100 scale
            return max(0.0, min(100.0, mi))
            
        except Exception as e:
            logger.error(f"Error calculating maintainability index: {str(e)}")
            return 100.0

    def _calculate_lizard_metrics(self, code: str) -> ComplexityMetrics:
        """Calculates metrics using lizard."""
        try:
            analysis = lizard.analyze_file.analyze_source_code("temp.py", code)
            
            # Calculate average complexity
            total_complexity = sum(func.cyclomatic_complexity for func in analysis.function_list)
            avg_complexity = (
                total_complexity / len(analysis.function_list)
                if analysis.function_list
                else 0.0
            )

            return ComplexityMetrics(
                cyclomatic_complexity=avg_complexity,
                maintainability_index=self._calculate_maintainability_index(code),
                halstead_volume=analysis.nloc,  # Using NLOC as a proxy
                halstead_difficulty=avg_complexity,  # Using complexity as proxy
                halstead_effort=analysis.nloc * avg_complexity
            )

        except Exception as e:
            logger.error(f"Error in lizard metrics calculation: {str(e)}")
            return ComplexityMetrics()

    def _calculate_radon_metrics(self, code: str) -> ComplexityMetrics:
        """Calculates metrics using radon with robust error handling."""
        metrics = ComplexityMetrics()

        try:
            # Calculate cyclomatic complexity
            try:
                cc_blocks = radon_cc.cc_visit(code)
                total_cc = sum(block.complexity for block in cc_blocks)
                metrics.cyclomatic_complexity = (
                    total_cc / len(cc_blocks) if cc_blocks else 0.0
                )
            except Exception as e:
                logger.warning(f"Error calculating cyclomatic complexity: {str(e)}")

            # Calculate maintainability index
            try:
                mi_result = radon_metrics.mi_visit(code, multi=False)
                if isinstance(mi_result, (int, float)):
                    metrics.maintainability_index = float(mi_result)
            except Exception as e:
                logger.warning(f"Error calculating maintainability index: {str(e)}")

            # Calculate Halstead metrics
            try:
                h_visit_result = radon_metrics.h_visit(code)
                
                # Handle different return types from h_visit
                if isinstance(h_visit_result, (list, tuple)) and h_visit_result:
                    h_metrics = h_visit_result[0]
                elif hasattr(h_visit_result, 'h1'):  # Single object
                    h_metrics = h_visit_result
                else:
                    raise ValueError("Invalid Halstead metrics format")

                # Safely extract Halstead metrics
                metrics.halstead_volume = getattr(h_metrics, 'volume', 0.0)
                metrics.halstead_difficulty = getattr(h_metrics, 'difficulty', 0.0)
                metrics.halstead_effort = getattr(h_metrics, 'effort', 0.0)
                metrics.halstead_bugs = getattr(h_metrics, 'bugs', 0.0)
                metrics.halstead_time = getattr(h_metrics, 'time', 0.0)

            except Exception as e:
                logger.warning(f"Error calculating Halstead metrics: {str(e)}")

        except Exception as e:
            logger.error(f"Error in radon metrics calculation: {str(e)}")

        return metrics

    def _add_language_metrics(
        self,
        metrics: ComplexityMetrics,
        code: str,
        language: str
    ) -> None:
        """Adds language-specific metrics if available."""
        try:
            if language in LANGUAGE_MAPPING:
                lang_info = LANGUAGE_MAPPING[language]
                
                # Add language-specific calculations here
                if lang_info["supports_type_hints"]:
                    # Example: Count type hints in Python
                    if language == "python":
                        import ast
                        try:
                            tree = ast.parse(code)
                            type_hints = sum(
                                1 for node in ast.walk(tree)
                                if isinstance(node, ast.AnnAssign)
                                or (isinstance(node, ast.FunctionDef) and node.returns)
                            )
                            metrics.type_hint_coverage = type_hints
                        except Exception as e:
                            logger.warning(f"Error analyzing type hints: {str(e)}")

        except Exception as e:
            logger.warning(f"Error adding language metrics: {str(e)}")

    def _prepare_metrics_output(self, metrics: ComplexityMetrics) -> Dict[str, Any]:
        """Prepares the final metrics output dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "complexity": metrics.cyclomatic_complexity,
            "maintainability_index": metrics.maintainability_index,
            "halstead": {
                "volume": metrics.halstead_volume,
                "difficulty": metrics.halstead_difficulty,
                "effort": metrics.halstead_effort,
                "bugs": metrics.halstead_bugs,
                "time": metrics.halstead_time
            }
        }

class CoverageHandler:
    """Handles multiple coverage report formats."""

    SUPPORTED_FORMATS = {'.coverage', '.xml', '.json', '.sqlite'}

    def __init__(self):
        self._coverage = None
        self._aliases = PathAliases()

    def get_test_coverage(
        self,
        file_path: Union[str, Path],
        coverage_path: Union[str, Path]
    ) -> Optional[CoverageData]:
        """Gets test coverage data from various report formats."""
        try:
            coverage_path = Path(coverage_path)
            file_path = Path(file_path)

            if not coverage_path.exists():
                logger.warning(f"Coverage file not found: {coverage_path}")
                return None

            if not file_path.exists():
                logger.warning(f"Source file not found: {file_path}")
                return None

            coverage_path = coverage_path.resolve()
            file_path = file_path.resolve()

            handler_map = {
                '.coverage': self._get_coverage_from_sqlite,
                '.xml': self._get_coverage_from_xml,
                '.json': self._get_coverage_from_json,
                '.sqlite': self._get_coverage_from_sqlite
            }

            if coverage_path.suffix not in handler_map:
                logger.warning(f"Unsupported coverage format: {coverage_path.suffix}")
                return None

            coverage_data = handler_map[coverage_path.suffix](coverage_path, file_path)
            if coverage_data:
                self._validate_coverage_data(coverage_data)
            return coverage_data

        except Exception as e:
            logger.error(f"Error getting test coverage: {str(e)}")
            return None

    def _get_coverage_from_sqlite(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from SQLite format."""
        try:
            conn = sqlite3.connect(str(coverage_path))
            cursor = conn.cursor()

            # Example query, adjust based on actual coverage DB schema
            cursor.execute("""
                SELECT line_rate, branch_rate, complexity, timestamp
                FROM coverage
                WHERE filename = ?
            """, (str(file_path),))
            row = cursor.fetchone()

            conn.close()

            if row:
                line_rate, branch_rate, complexity, timestamp = row
                return CoverageData(
                    line_rate=line_rate,
                    branch_rate=branch_rate,
                    complexity=complexity,
                    timestamp=datetime.fromtimestamp(timestamp).isoformat(),
                    source_file=str(file_path)
                )
            else:
                logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading SQLite coverage data: {e}")
            return None

    def _get_coverage_from_xml(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from XML format."""
        try:
            tree = ET.parse(coverage_path)
            root = tree.getroot()

            # Example parsing, adjust based on actual XML schema
            for file_elem in root.findall('.//file'):
                if file_elem.get('name') == str(file_path):
                    line_rate = float(file_elem.get('line-rate', 0.0))
                    branch_rate = float(file_elem.get('branch-rate', 0.0))
                    complexity = float(file_elem.get('complexity', 0.0))
                    timestamp = datetime.now().isoformat()  # XML might not have timestamp

                    return CoverageData(
                        line_rate=line_rate,
                        branch_rate=branch_rate,
                        complexity=complexity,
                        timestamp=timestamp,
                        source_file=str(file_path)
                    )

            logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
            return None

        except Exception as e:
            logger.error(f"Error reading XML coverage data: {e}")
            return None

    def _get_coverage_from_json(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from JSON format."""
        try:
            with open(coverage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Example parsing, adjust based on actual JSON schema
            file_coverage = data.get('files', {}).get(str(file_path), {})
            if file_coverage:
                return CoverageData(
                    line_rate=file_coverage.get('line_rate', 0.0),
                    branch_rate=file_coverage.get('branch_rate', 0.0),
                    complexity=file_coverage.get('complexity', 0.0),
                    timestamp=datetime.now().isoformat(),  # JSON might not have timestamp
                    source_file=str(file_path)
                )
            else:
                logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading JSON coverage data: {e}")
            return None

    def _calculate_complexity(self, analysis: Any) -> float:
        """Calculates complexity from analysis data."""
        try:
            # Implement complexity calculation based on analysis
            return float(analysis.complexity)
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return 0.0

    def _get_relative_path(self, file_path: Path) -> str:
        """Gets the relative path of a file."""
        try:
            return str(file_path.relative_to(self.repo_path))
        except ValueError:
            return str(file_path)

    def _validate_coverage_data(self, data: CoverageData) -> None:
        """Validates the coverage data."""
        try:
            if not (0.0 <= data.line_rate <= 1.0):
                raise CoverageFormatError("Line rate out of bounds")
            if not (0.0 <= data.branch_rate <= 1.0):
                raise CoverageFormatError("Branch rate out of bounds")
            # Add more validation as needed
        except CoverageFormatError as e:
            logger.error(f"Invalid coverage data: {e}")
            raise

class PathFilter:
    """Handles file path filtering based on various exclusion patterns."""
    
    def __init__(
        self,
        repo_path: Union[str, Path],
        excluded_dirs: Optional[Set[str]] = None,
        excluded_files: Optional[Set[str]] = None,
        skip_types: Optional[Set[str]] = None
    ):
        self.repo_path = Path(repo_path)
        self.excluded_dirs = (excluded_dirs or set()) | DEFAULT_EXCLUDED_PATTERNS['dirs']
        self.excluded_files = (excluded_files or set()) | DEFAULT_EXCLUDED_PATTERNS['files']
        self.skip_types = (skip_types or set()) | DEFAULT_EXCLUDED_PATTERNS['extensions']
        self.gitignore = load_gitignore(repo_path)
        
        self.file_patterns = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            self.excluded_files
        )

    def should_include_path(self, path: Path, relative_to: Optional<Path] = None) -> bool:
        """Determines if a path should be included based on exclusion rules."""
        try:
            check_path = path.relative_to(relative_to) if relative_to else path
            
            if any(part.startswith('.') for part in check_path.parts):
                return False
                
            if any(part in self.excluded_dirs for part in check_path.parts):
                return False
                
            if self.file_patterns.match_file(str(check_path)):
                return False
                
            if check_path.suffix.lower() in self.skip_types:
                return False
                
            if self.gitignore.match_file(str(check_path)):
                return False
                
            return True
                
        except Exception as e:
            logger.warning(f"Error checking path {path}: {str(e)}")
            return False

@lru_cache(maxsize=128)
def load_gitignore(repo_path: Union[str, Path]) -> pathspec.PathSpec:
    """Loads and caches .gitignore patterns."""
    patterns = []
    gitignore_path = Path(repo_path) / '.gitignore'
    
    try:
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
                logger.debug(f"Loaded {len(patterns)} patterns from .gitignore")
        else:
            logger.debug("No .gitignore file found")
            
    except Exception as e:
        logger.warning(f"Error reading .gitignore: {str(e)}")
        
    return pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern,
        patterns
    )

def get_all_file_paths(
    repo_path: Union[str, Path],
    excluded_dirs: Optional[Set[str]] = None,
    excluded_files: Optional[Set[str]] = None,
    skip_types: Optional[Set[str]] = None,
    follow_symlinks: bool = False
) -> List[str]:
    """Gets all file paths in a repository with improved filtering."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return []
        
    path_filter = PathFilter(
        repo_path,
        excluded_dirs,
        excluded_files,
        skip_types
    )
    
    included_paths = []
    
    try:
        for root, dirs, files in os.walk(repo_path, followlinks=follow_symlinks):
            root_path = Path(root)
            
            # Filter directories in-place
            dirs[:] = [
                d for d in dirs
                if path_filter.should_include_path(root_path / d, repo_path)
            ]
            
            # Filter and add files
            for file in files:
                file_path = root_path / file
                if path_filter.should_include_path(file_path, repo_path):
                    included_paths.append(str(file_path))
                    
        logger.info(
            f"Found {len(included_paths)} files in {repo_path} "
            f"(excluded: dirs={len(path_filter.excluded_dirs)}, "
            f"files={len(path_filter.excluded_files)}, "
            f"types={len(path_filter.skip_types)})"
        )
        
        return included_paths
        
    except Exception as e:
        logger.error(f"Error walking repository: {str(e)}")
        return []

# Initialize global instances
metrics_calculator = EnhancedMetricsCalculator()
coverage_handler = CoverageHandler()

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None
) -> bool:
    """Sets up logging configuration."""
    try:
        if not log_format:
            log_format = (
                "%(asctime)s [%(levelname)s] "
                "%(name)s:%(lineno)d - %(message)s"
            )
        
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=handlers
        )
        
        # Set lower level for external libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        return True
        
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return False

```

## metrics.py

```python
"""
metrics.py - Core logic for calculating and analyzing code metrics.

Provides functionalities to calculate various code metrics, including
cyclomatic complexity, maintainability index, and Halstead metrics.
Implements robust error handling, logging, and aggregation of metrics data.
"""

import ast
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import wraps, partial
from time import perf_counter
from typing import Dict, Any, Optional, Union, Callable, TypeVar

from concurrent.futures import ThreadPoolExecutor

from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --------------------------
# Custom Exceptions
# --------------------------

class MetricsCalculationError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class HalsteadCalculationError(MetricsCalculationError):
    """Exception for Halstead metrics calculation errors."""
    pass

class ComplexityCalculationError(MetricsCalculationError):
    """Exception for cyclomatic complexity calculation errors."""
    pass

# --------------------------
# Data Classes
# --------------------------

@dataclass
class MetricsResult:
    """Data class for storing metrics calculation results."""
    file_path: str
    timestamp: datetime
    execution_time: float
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics."""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1000000.0
    code_churn_high: int = 1000
    code_churn_warning: int = 500
    code_duplication_high: float = 30.0  # Percentage
    code_duplication_warning: float = 10.0

    @classmethod
    def from_dict(cls, thresholds_dict: Dict[str, Any]) -> 'MetricsThresholds':
        """Creates a MetricsThresholds instance from a dictionary with validation."""
        validated_dict = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in thresholds_dict:
                value = thresholds_dict[field_name]
                if not isinstance(value, field_type):  # Simplified type checking
                    try:
                        validated_dict[field_name] = field_type(value)  # Attempt type conversion
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid value for threshold '{field_name}': {value} - {e}") from e
                else:
                    validated_dict[field_name] = value

        # Additional validation for percentage fields
        percentage_fields = ['code_duplication_high', 'code_duplication_warning']
        for pf in percentage_fields:
            if pf in validated_dict:
                pct = validated_dict[pf]
                if not (0.0 <= pct <= 100.0):
                    raise ValueError(f"Threshold '{pf}' must be between 0 and 100.")

        return cls(**validated_dict)

    @classmethod
    def load_from_file(cls, config_path: str) -> 'MetricsThresholds':
        """Loads metric thresholds from a JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                thresholds_dict = json.load(f)
                logger.debug(f"Loaded thresholds from {config_path}: {thresholds_dict}")
            return cls.from_dict(thresholds_dict)
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except ValueError as e:  # Catch validation errors
            logger.error(f"Invalid threshold values in config: {e}")
            raise

# --------------------------
# Decorators
# --------------------------

T = TypeVar('T')  # Generic type variable for the return value

def safe_metric_calculation(default_value: T = None, metric_name: str = "metric") -> Callable[Callable..., T](Callable...,%20T.md), Callable[..., T]]:
    """Decorator for safe metric calculation with specific error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                error_message = f"ValueError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except TypeError as e:
                error_message = f"TypeError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except Exception as e:
                error_message = f"Unexpected error during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
        return wrapper
    return decorator

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = perf_counter() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

# --------------------------
# Metric Calculation Functions
# --------------------------

@safe_metric_calculation(default_value={}, metric_name="Halstead Metrics")
def calculate_halstead_metrics(code: str) -> Dict[str, Union[int, float]]:
    """Calculates Halstead complexity metrics."""
    halstead_reports = h_visit(code)

    if not halstead_reports:
        raise HalsteadCalculationError("No Halstead metrics found")

    metrics = halstead_reports[0] if isinstance(halstead_reports, list) else halstead_reports

    return {
        'h1': metrics.h1,
        'h2': metrics.h2,
        'N1': metrics.N1,
        'N2': metrics.N2,
        'vocabulary': metrics.vocabulary,
        'length': metrics.length,
        'calculated_length': metrics.calculated_length,
        'volume': metrics.volume,
        'difficulty': metrics.difficulty,
        'effort': metrics.effort,
        'time': metrics.time,
        'bugs': metrics.bugs
    }

@safe_metric_calculation(default_value=0.0, metric_name="Maintainability Index")
def calculate_maintainability_index(code: str) -> float:
    """Calculates the Maintainability Index."""
    mi_value = mi_visit(code, multi=False)
    return float(mi_value)

@safe_metric_calculation(default_value={}, metric_name="Cyclomatic Complexity")
def calculate_cyclomatic_complexity(code: str) -> Dict[str, float]:
    """Calculates Cyclomatic Complexity."""
    complexity_visitor = ComplexityVisitor.from_code(code)
    function_complexity = {}
    for block in complexity_visitor.functions + complexity_visitor.classes:
        function_complexity[block.name] = float(block.complexity)
    return function_complexity

# --------------------------
# Metrics Aggregation and Analysis
# --------------------------

@log_execution_time
def calculate_code_metrics(code: str, file_path: Optional[str] = None, language: str = "python") -> MetricsResult:
    """Calculates code metrics, handling different languages, with optimized thread pooling and validation."""
    start_time = perf_counter()
    logger.info(f"Starting metrics calculation for {file_path or 'unknown file'}")

    try:
        if not isinstance(code, str):
            raise ValueError(f"Invalid code type: {type(code)}")

        if not code.strip():
            raise ValueError("Empty code provided")

        metrics_data = {}

        if language.lower() == "python":
            metrics_data['halstead'] = calculate_halstead_metrics(code)
            metrics_data['maintainability_index'] = calculate_maintainability_index(code)
            metrics_data['function_complexity'] = calculate_cyclomatic_complexity(code)

            complexities = list(metrics_data['function_complexity'].values())
            metrics_data['cyclomatic_complexity'] = sum(complexities) / len(complexities) if complexities else 0.0
        else:
            # Placeholder for other languages
            metrics_data['halstead'] = {}
            metrics_data['maintainability_index'] = 0.0
            metrics_data['function_complexity'] = {}
            metrics_data['cyclomatic_complexity'] = 0.0
            logger.warning(f"Metrics calculation for language '{language}' is not implemented.")

        validated_metrics = validate_metrics(metrics_data)
        quality_score = calculate_quality_score(validated_metrics)
        validated_metrics['quality'] = quality_score

        validated_metrics['raw'] = {
            'code_size': len(code),
            'num_functions': len(metrics_data.get('function_complexity', {})),
            'calculation_time': perf_counter() - start_time
        }

        logger.info(f"Metrics calculation completed for {file_path or 'unknown file'}")
        logger.debug(f"Calculated metrics: {validated_metrics}")

        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=True,
            metrics=validated_metrics
        )

    except MetricsCalculationError as e:
        error_msg = f"Metrics calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=False,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during metrics calculation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=False,
            error=error_msg
        )

# --------------------------
# Metrics Validation and Scoring
# --------------------------

def validate_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Validates calculated metrics against expected ranges and logs warnings."""
    validated = metrics.copy()

    halstead = validated.get('halstead', {})
    for metric, value in halstead.items():
        if isinstance(value, (int, float)) and value < 0:
            logger.warning(f"Invalid Halstead {metric}: {value} (expected non-negative)")
            validated['halstead'][metric] = 0

    mi = validated.get('maintainability_index', 0.0)
    if not (0 <= mi <= 100):
        logger.warning(f"Invalid Maintainability Index: {mi} (expected 0-100)")
        validated['maintainability_index'] = max(0, min(mi, 100))

    cyclomatic = validated.get('cyclomatic_complexity', 0.0)
    if cyclomatic < 1:
        logger.warning(f"Unusually low Cyclomatic Complexity: {cyclomatic} (expected >= 1)")
    if cyclomatic > 50:
        logger.warning(f"Very high Cyclomatic Complexity: {cyclomatic} (consider refactoring)")

    function_complexity = validated.get('function_complexity', {})
    for func, complexity in function_complexity.items():
        if complexity < 0:
            logger.warning(f"Invalid complexity for function '{func}': {complexity} (expected non-negative)")
            validated['function_complexity'][func] = 0
        if complexity > 30:
            logger.warning(f"High complexity for function '{func}': {complexity} (consider refactoring)")

    return validated

def calculate_quality_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates a normalized quality score based on metrics."""
    quality_scores = {
        'maintainability': normalize_score(metrics['maintainability_index'], 0, 100, 0.4),
        'complexity': normalize_score(metrics['cyclomatic_complexity'], 1, 30, 0.3, inverse=True),
        'halstead_effort': normalize_score(metrics['halstead'].get('effort', 0), 0, 1000000, 0.3, inverse=True)
    }
    quality_scores['overall'] = sum(quality_scores.values()) / len(quality_scores)
    return quality_scores

def normalize_score(value: float, min_val: float, max_val: float, weight: float = 1.0, inverse: bool = False) -> float:
    """Normalizes a metric value to a 0-1 scale."""
    try:
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        if inverse:
            normalized = 1.0 - normalized
        return normalized * weight
    except Exception as e:
        logger.error(f"Error normalizing score: {e}")
        return 0.0

# --------------------------
# Metrics Analyzer
# --------------------------

class MetricsAnalyzer:
    """Analyzes and aggregates metrics across multiple files."""

    def __init__(self, thresholds: Optional[MetricsThresholds] = None):
        self.metrics_history: List[MetricsResult] = []
        self.error_count = 0
        self.warning_count = 0
        self.thresholds = thresholds or MetricsThresholds()

    def add_result(self, result: MetricsResult):
        """Adds a metrics result and updates error/warning counts."""
        self.metrics_history.append(result)
        if not result.success:
            self.error_count += 1
            logger.error(f"Metrics calculation failed for {result.file_path}: {result.error}")
        elif result.metrics:  # Only check for warnings if metrics are available
            self._check_metrics_warnings(result)

    def _check_metrics_warnings(self, result: MetricsResult):
        """Checks for and logs warnings about concerning metric values."""
        metrics = result.metrics
        if metrics['maintainability_index'] < self.thresholds.maintainability_low:
            self.warning_count += 1
            logger.warning(f"Very low maintainability index ({metrics['maintainability_index']:.2f}) in {result.file_path}")
        for func, complexity in metrics['function_complexity'].items():
            if complexity > self.thresholds.complexity_high:
                self.warning_count += 1
                logger.warning(f"High cyclomatic complexity ({complexity}) in function '{func}' in {result.file_path}")
        if metrics['halstead'].get('effort', 0) > self.thresholds.halstead_effort_high:
            self.warning_count += 1
            logger.warning(f"High Halstead effort ({metrics['halstead']['effort']:.2f}) in {result.file_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Generates a summary of processed metrics."""
        successful_results = [r for r in self.metrics_history if r.success and r.metrics]
        avg_metrics = self._calculate_average_metrics(successful_results) if successful_results else None
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'processed_files': len(self.metrics_history),
            'success_rate': (len(successful_results) / len(self.metrics_history) * 100) if self.metrics_history else 0,
            'average_metrics': avg_metrics,
            'execution_times': self._calculate_execution_time_summary()
        }

    def _calculate_average_metrics(self, results: List[MetricsResult]) -> Dict[str, Any]:
        """Calculates average metrics from successful results."""
        try:
            avg_maintainability = sum(r.metrics['maintainability_index'] for r in results) / len(results)
            avg_complexity = sum(
                sum(r.metrics['function_complexity'].values()) / max(1, len(r.metrics['function_complexity']))
                for r in results
            ) / len(results)
            avg_halstead = {
                metric: sum(r.metrics['halstead'][metric] for r in results) / len(results)
                for metric in ['volume', 'difficulty', 'effort']
            }
            return {
                'maintainability_index': avg_maintainability,
                'cyclomatic_complexity': avg_complexity,
                'halstead': avg_halstead
            }
        except Exception as e:
            logger.error(f"Error calculating average metrics: {e}")
            return {}  # Or return a dictionary with default values

    def _calculate_execution_time_summary(self) -> Dict[str, float]:
        """Calculates execution time summary statistics."""
        if not self.metrics_history:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0}  # Return default values if no history
        times = [r.execution_time for r in self.metrics_history]
        return {
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times)
        }

    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """
        Identifies files with concerning metrics based on provided thresholds.

        Returns:
            List of problematic files and their issues.
        """
        problematic_files = []

        for result in self.metrics_history:
            if not (result.success and result.metrics):
                continue

            issues = []
            metrics = result.metrics

            if metrics['maintainability_index'] < self.thresholds.maintainability_low:
                issues.append({
                    'type': 'maintainability',
                    'value': metrics['maintainability_index'],
                    'threshold': self.thresholds.maintainability_low
                })

            high_complexity_functions = [
                (func, complexity)
                for func, complexity in metrics['function_complexity'].items()
                if complexity > self.thresholds.complexity_high
            ]
            if high_complexity_functions:
                issues.append({
                    'type': 'complexity',
                    'functions': high_complexity_functions,
                    'threshold': self.thresholds.complexity_high
                })

            if metrics['halstead'].get('effort', 0) > self.thresholds.halstead_effort_high:
                issues.append({
                    'type': 'halstead_effort',
                    'value': metrics['halstead']['effort'],
                    'threshold': self.thresholds.halstead_effort_high
                })

            if issues:
                problematic_files.append({
                    'file_path': result.file_path,
                    'issues': issues
                })

        return problematic_files

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Calculate code metrics for a repository.")
    parser.add_argument('repo_path', type=str, help='Path to the Git repository.')
    parser.add_argument('--thresholds', type=str, default='thresholds.json', help='Path to thresholds JSON file.')
    args = parser.parse_args()

    try:
        thresholds = MetricsThresholds.load_from_file(args.thresholds)
    except Exception as e:
        logger.error(f"Failed to load thresholds: {e}")
        thresholds = MetricsThresholds()  # Use default thresholds

    analyzer = MetricsAnalyzer(thresholds=thresholds)

    # Create a thread pool for concurrent processing
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for root, _, files in os.walk(args.repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(
                        calculate_code_metrics,
                        code=open(file_path, 'r', encoding='utf-8').read(),
                        file_path=file_path,
                        language="python"
                    ))

        for future in futures:
            try:
                result = future.result()
                analyzer.add_result(result)

                if result.success and result.metrics:
                    # Determine severity for each metric
                    cc_severity = "N/A"
                    mi_severity = "N/A"
                    halstead_severity = "N/A"

                    # Assuming severity functions are in metrics_utils.py and imported appropriately
                    # from metrics_utils import get_metric_severity, format_metric_value

                    # For demonstration, using placeholder severity
                    cc_severity = "high" if result.metrics['cyclomatic_complexity'] > thresholds.complexity_high else (
                        "warning" if result.metrics['cyclomatic_complexity'] > thresholds.complexity_warning else "normal"
                    )
                    mi_severity = "low" if result.metrics['maintainability_index'] < thresholds.maintainability_low else "normal"
                    halstead_effort = result.metrics['halstead'].get('effort', 0)
                    halstead_severity = "high" if halstead_effort > thresholds.halstead_effort_high else "normal"

                    # Format metrics
                    formatted_cc = f"{result.metrics['cyclomatic_complexity']:.2f}"
                    formatted_mi = f"{result.metrics['maintainability_index']:.2f}"
                    formatted_halstead = f"{halstead_effort:,.0f}"

                    logger.info(f"File: {result.file_path}")
                    logger.info(f"  Cyclomatic Complexity: {formatted_cc} (Severity: {cc_severity})")
                    logger.info(f"  Maintainability Index: {formatted_mi} (Severity: {mi_severity})")
                    logger.info(f"  Halstead Effort: {formatted_halstead} (Severity: {halstead_severity})")

            except Exception as e:
                logger.error(f"Error processing file: {e}")

    # Aggregate and report metrics
    summary = analyzer.get_summary()
    logger.info("Aggregated Metrics:")
    for metric, value in summary['average_metrics'].items():
        if metric != 'halstead':
            severity = "N/A"  # Placeholder for severity
            formatted_value = f"{value:.2f}"
            logger.info(f"  {metric.replace('_', ' ').title()}: {formatted_value} (Severity: {severity})")
        else:
            for hal_metric, hal_value in value.items():
                severity = "high" if hal_value > thresholds.halstead_effort_high else "normal"
                formatted_value = f"{hal_value:,.0f}"
                logger.info(f"  Halstead {hal_metric.title()}: {formatted_value} (Severity: {severity})")

    # Display summary
    logger.info(f"Processed Files: {summary['processed_files']}")
    logger.info(f"Success Rate: {summary['success_rate']:.2f}%")
    logger.info(f"Errors: {summary['error_count']}, Warnings: {summary['warning_count']}")
    logger.info(f"Execution Time - Min: {summary['execution_times']['min']:.2f}s, "
                f"Max: {summary['execution_times']['max']:.2f}s, "
                f"Avg: {summary['execution_times']['avg']:.2f}s")

    # Identify and report problematic files
    problematic_files = analyzer.get_problematic_files()
    if problematic_files:
        logger.info("Problematic Files:")
        for pf in problematic_files:
            logger.info(f"  File: {pf['file_path']}")
            for issue in pf['issues']:
                if issue['type'] == 'maintainability':
                    logger.info(f"    - Maintainability Index: {issue['value']} (Threshold: {issue['threshold']})")
                elif issue['type'] == 'complexity':
                    for func, complexity in issue['functions']:
                        logger.info(f"    - Function '{func}' Complexity: {complexity} (Threshold: {issue['threshold']})")
                elif issue['type'] == 'halstead_effort':
                    logger.info(f"    - Halstead Effort: {issue['value']} (Threshold: {issue['threshold']})")
    else:
        logger.info("No problematic files detected based on the provided thresholds.")


```

## openai_model.py

```python
# openai_model.py
import openai
import logging
import json
from typing import List, Dict, Any

# Set up logging for OpenAI model integration
logger = logging.getLogger(__name__)

class OpenAIModel:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_prompt(self, base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Creates a structured prompt for OpenAI gpt-4o model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for OpenAI model.")
        return prompt

    def calculate_tokens(self, prompt: List[Dict[str, str]]) -> int:
        """Calculates an approximate token count for OpenAI model prompts."""
        # Using a rough token calculation (OpenAI has ~4 chars per token as a baseline)
        return sum(len(item['content']) for item in prompt) // 4

    def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Fetches documentation generation from OpenAI gpt-4o model."""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        logger.info("OpenAI documentation generated successfully.")
        return response.choices[0].message['content']

```

## write_documentation_report.py

```python
"""
write_documentation_report.py

Enhanced documentation report generation with template support, robust error handling,
and improved Markdown generation. Provides comprehensive documentation formats with
metrics, badges, and formatting utilities.
"""

import re
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Type
from dataclasses import dataclass
from datetime import datetime
from jinja2 import Environment, PackageLoader, select_autoescape, Template
import aiofiles
import aiofiles.os
from functools import lru_cache

from utils import (
    FileHandler,
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
    TokenManager,
    sanitize_filename
)

logger = logging.getLogger(__name__)

# Global write lock for thread safety
write_lock = asyncio.Lock()

class DocumentationError(Exception):
    """Base exception for documentation-related errors."""
    pass

class TemplateError(DocumentationError):
    """Raised when template processing fails."""
    pass

class FileWriteError(DocumentationError):
    """Raised when file writing fails."""
    pass

@dataclass
class BadgeConfig:
    """Configuration for badge generation."""
    metric_name: str
    value: Union[int, float]
    thresholds: Dict[str, int]
    logo: Optional[str] = None
    style: str = "flat-square"
    label_color: Optional[str] = None
    
    def get_color(self) -> str:
        """Determines badge color based on thresholds."""
        low, medium, high = (
            self.thresholds["low"],
            self.thresholds["medium"],
            self.thresholds["high"]
        )
        
        if self.value <= low:
            return "success"
        elif self.value <= medium:
            return "yellow"
        else:
            return "critical"

class BadgeGenerator:
    """Enhanced badge generation with caching and templates."""
    
    _badge_template = (
        "![{label}](https://img.shields.io/badge/"
        "{encoded_label}-{value}-{color}"
        "?style={style}{logo_part}{label_color_part})"
    )
    
    @classmethod
    @lru_cache(maxsize=128)
    def generate_badge(cls, config: BadgeConfig) -> str:
        """
        Generates a Markdown badge with caching.
        
        Args:
            config: Badge configuration
            
        Returns:
            str: Markdown badge string
        """
        try:
            label = config.metric_name.replace("_", " ").title()
            encoded_label = label.replace(" ", "%20")
            color = config.get_color()
            
            # Format value based on type
            if isinstance(config.value, float):
                value = f"{config.value:.2f}"
            else:
                value = str(config.value)
            
            # Add optional components
            logo_part = f"&logo={config.logo}" if config.logo else ""
            label_color_part = (
                f"&labelColor={config.label_color}"
                if config.label_color
                else ""
            )
            
            return cls._badge_template.format(
                label=label,
                encoded_label=encoded_label,
                value=value,
                color=color,
                style=config.style,
                logo_part=logo_part,
                label_color_part=label_color_part
            )
            
        except Exception as e:
            logger.error(f"Error generating badge: {e}")
            return ""
    
    @classmethod
    def generate_all_badges(cls, metrics: Dict[str, Any]) -> str:
        """
        Generates all relevant badges for metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            str: Combined badge string
        """
        badges = []
        
        try:
            # Complexity badge
            if complexity := metrics.get("complexity"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Complexity",
                    value=complexity,
                    thresholds=DEFAULT_COMPLEXITY_THRESHOLDS,
                    logo="codeClimate"
                )))
            
            # Halstead metrics badges
            if halstead := metrics.get("halstead"):
                halstead_configs = [
                    BadgeConfig(
                        metric_name="Volume",
                        value=halstead["volume"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["volume"],
                        logo="stackOverflow"
                    ),
                    BadgeConfig(
                        metric_name="Difficulty",
                        value=halstead["difficulty"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["difficulty"],
                        logo="codewars"
                    ),
                    BadgeConfig(
                        metric_name="Effort",
                        value=halstead["effort"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["effort"],
                        logo="atlassian"
                    )
                ]
                badges.extend(
                    cls.generate_badge(config)
                    for config in halstead_configs
                )
            
            # Maintainability badge
            if mi := metrics.get("maintainability_index"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Maintainability",
                    value=mi,
                    thresholds=DEFAULT_MAINTAINABILITY_THRESHOLDS,
                    logo="codeclimate"
                )))
            
            # Test coverage badge if available
            if coverage := metrics.get("test_coverage", {}).get("line_rate"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Coverage",
                    value=coverage,
                    thresholds={"low": 80, "medium": 60, "high": 0},
                    logo="testCoverage"
                )))
            
            return " ".join(badges)
            
        except Exception as e:
            logger.error(f"Error generating badges: {e}")
            return ""

class MarkdownFormatter:
    """Enhanced Markdown formatting with template support."""
    
    def __init__(self):
        self.env = Environment(
            loader=PackageLoader('documentation', 'templates'),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.env.filters['truncate_description'] = self.truncate_description
        self.env.filters['sanitize_text'] = self.sanitize_text
    
    @staticmethod
    def truncate_description(
        description: str,
        max_length: int = 100,
        ellipsis: str = "..."
    ) -> str:
        """
        Truncates description with word boundary awareness.
        
        Args:
            description: Text to truncate
            max_length: Maximum length
            ellipsis: Ellipsis string
            
        Returns:
            str: Truncated description
        """
        if not description or len(description) <= max_length:
            return description
        
        truncated = description[:max_length]
        # Find last word boundary
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        
        return truncated + ellipsis
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Sanitizes text for Markdown with improved character handling.
        
        Args:
            text: Text to sanitize
            
        Returns:
            str: Sanitized text
        """
        # Escape Markdown special characters
        special_chars = r'[`*_{}[\]()#+\-.!|]'
        text = re.sub(
            special_chars,
            lambda m: '\\' + m.group(0),
            str(text)
        )
        
        # Replace newlines and returns
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Normalize whitespace
        return ' '.join(text.split())
    
    def format_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        alignment: Optional[List[str]] = None
    ) -> str:
        """
        Formats data into a Markdown table with alignment support.
        
        Args:
            headers: Column headers
            rows: Table rows
            alignment: Column alignments ('left', 'center', 'right')
            
        Returns:
            str: Formatted Markdown table
        """
        if not headers or not rows:
            return ""
            
        try:
            # Sanitize headers
            headers = [self.sanitize_text(str(header)) for header in headers]
            
            # Set default alignment if not provided
            if not alignment:
                alignment = ['left'] * len(headers)
            
            # Create alignment string
            align_map = {
                'left': ':---',
                'center': ':---:',
                'right': '---:'
            }
            separators = [
                align_map.get(align, ':---')
                for align in alignment
            ]
            
            # Format headers and separator
            table_lines = [
                f"| {' | '.join(headers)} |",
                f"| {' | '.join(separators)} |"
            ]
            
            # Format rows
            for row in rows:
                # Ensure row has correct number of columns
                row = (row + [''] * len(headers))[:len(headers)]
                sanitized_row = [
                    self.sanitize_text(str(cell))
                    for cell in row
                ]
                table_lines.append(
                    f"| {' | '.join(sanitized_row)} |"
                )
            
            return '\n'.join(table_lines)
            
        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return ""
class DocumentationGenerator:
    """Enhanced documentation generation with template support."""
    
    def __init__(self):
        self.formatter = MarkdownFormatter()
        self.badge_generator = BadgeGenerator()
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Loads and validates templates."""
        try:
            self.templates = {
                'main': self.formatter.env.get_template('main.md.j2'),
                'function': self.formatter.env.get_template('function.md.j2'),
                'class': self.formatter.env.get_template('class.md.j2'),
                'metric': self.formatter.env.get_template('metric.md.j2'),
                'summary': self.formatter.env.get_template('summary.md.j2')
            }
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise TemplateError(f"Failed to load templates: {e}")
    
    async def generate_documentation(
        self,
        documentation: Dict[str, Any],
        language: str,
        file_path: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates comprehensive documentation using templates.
        
        Args:
            documentation: Documentation data
            language: Programming language
            file_path: Source file path
            metrics: Optional additional metrics
            
        Returns:
            str: Generated documentation
            
        Raises:
            DocumentationError: If documentation generation fails
        """
        try:
            # Generate badges
            badges = self.badge_generator.generate_all_badges(
                metrics or documentation.get("metrics", {})
            )
            
            # Generate different sections
            language_info = self._get_language_info(language)
            functions_doc = await self._generate_functions_section(
                documentation.get("functions", [])
            )
            classes_doc = await self._generate_classes_section(
                documentation.get("classes", [])
            )
            metrics_doc = await self._generate_metrics_section(
                documentation.get("metrics", {}),
                metrics or {}
            )
            summary = await self._generate_summary_section(
                documentation,
                language_info
            )
            
            # Combine everything using the main template
            content = await self._render_template(
                self.templates['main'],
                {
                    'file_name': Path(file_path).name,
                    'badges': badges,
                    'language': language_info,
                    'summary': summary,
                    'functions': functions_doc,
                    'classes': classes_doc,
                    'metrics': metrics_doc,
                    'documentation': documentation
                }
            )
            
            # Generate table of contents
            toc = self._generate_toc(content)
            
            return f"# Table of Contents\n\n{toc}\n\n{content}"
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise DocumentationError(f"Documentation generation failed: {e}")
    
    async def _generate_functions_section(
        self,
        functions: List[Dict[str, Any]]
    ) -> str:
        """Generates functions documentation using templates."""
        if not functions:
            return ""
        
        try:
            function_docs = []
            for func in functions:
                doc = await self._render_template(
                    self.templates['function'],
                    {'function': func}
                )
                function_docs.append(doc)
            
            return "\n\n".join(function_docs)
        except Exception as e:
            logger.error(f"Error generating functions section: {e}")
            return ""
    
    async def _generate_classes_section(
        self,
        classes: List[Dict[str, Any]]
    ) -> str:
        """Generates classes documentation using templates."""
        if not classes:
            return ""
        
        try:
            class_docs = []
            for cls in classes:
                doc = await self._render_template(
                    self.templates['class'],
                    {'class': cls}
                )
                class_docs.append(doc)
            
            return "\n\n".join(class_docs)
        except Exception as e:
            logger.error(f"Error generating classes section: {e}")
            return ""
    
    async def _generate_metrics_section(
        self,
        doc_metrics: Dict[str, Any],
        additional_metrics: Dict[str, Any]
    ) -> str:
        """Generates metrics documentation using templates."""
        try:
            combined_metrics = {**doc_metrics, **additional_metrics}
            return await self._render_template(
                self.templates['metric'],
                {'metrics': combined_metrics}
            )
        except Exception as e:
            logger.error(f"Error generating metrics section: {e}")
            return ""
    
    async def _generate_summary_section(
        self,
        documentation: Dict[str, Any],
        language_info: Dict[str, Any]
    ) -> str:
        """Generates summary section using templates."""
        try:
            return await self._render_template(
                self.templates['summary'],
                {
                    'documentation': documentation,
                    'language': language_info
                }
            )
        except Exception as e:
            logger.error(f"Error generating summary section: {e}")
            return ""
    
    @staticmethod
    def _get_language_info(language: str) -> Dict[str, Any]:
        """Gets language-specific information."""
        from utils import LANGUAGE_MAPPING
        
        for ext, info in LANGUAGE_MAPPING.items():
            if info["name"] == language:
                return info
        return {"name": language}
    
    async def _render_template(
        self,
        template: Template,
        context: Dict[str, Any]
    ) -> str:
        """
        Renders a template asynchronously.
        
        Args:
            template: Jinja2 template
            context: Template context
            
        Returns:
            str: Rendered content
        """
        try:
            # Run template rendering in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                template.render,
                context
            )
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise TemplateError(f"Failed to render template: {e}")
    
    def _generate_toc(self, content: str) -> str:
        """
        Generates table of contents from content.
        
        Args:
            content: Markdown content
            
        Returns:
            str: Table of contents
        """
        toc_entries = []
        current_level = 0
        
        for line in content.splitlines():
            if line.startswith('#'):
                # Count heading level
                level = 0
                while line.startswith('#'):
                    level += 1
                    line = line[1:]
                
                # Extract heading text
                heading = line.strip()
                if not heading:
                    continue
                
                # Create anchor
                anchor = heading.lower()
                anchor = re.sub(r'[^\w\- ]', '', anchor)
                anchor = anchor.replace(' ', '-')
                
                # Add TOC entry
                indent = '  ' * (level - 1)
                toc_entries.append(
                    f"{indent}- [{heading}](#{anchor})"
                )
        
        return '\n'.join(toc_entries)

async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,
    metrics: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Writes documentation to JSON and Markdown files.
    
    Args:
        documentation: Documentation data
        language: Programming language
        file_path: Source file path
        repo_root: Repository root path
        output_dir: Output directory
        project_id: Project identifier
        metrics: Optional additional metrics
        
    Returns:
        Optional[Dict[str, Any]]: Written documentation or None if failed
    """
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None
    
    try:
        async with write_lock:
            # Create output directory
            project_output_dir = Path(output_dir) / project_id
            await aiofiles.os.makedirs(
                project_output_dir,
                exist_ok=True
            )
            
            # Prepare paths
            relative_path = Path(file_path).relative_to(repo_root)
            safe_filename = sanitize_filename(relative_path.name)
            base_path = project_output_dir / safe_filename
            
            # Write JSON documentation
            json_path = base_path.with_suffix(".json")
            try:
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(json.dumps(
                        documentation,
                        indent=2,
                        sort_keys=True
                    ))
            except Exception as e:
                logger.error(f"Error writing JSON to {json_path}: {e}")
                raise FileWriteError(f"Failed to write JSON: {e}")
            
            # Generate and write Markdown if requested
            if documentation.get("generate_markdown", True):
                try:
                    generator = DocumentationGenerator()
                    markdown_content = await generator.generate_documentation(
                        documentation,
                        language,
                        file_path,
                        metrics
                    )
                    
                    md_path = base_path.with_suffix(".md")
                    async with aiofiles.open(md_path, "w") as f:
                        await f.write(markdown_content)
                except Exception as e:
                    logger.error(f"Error writing Markdown to {md_path}: {e}")
                    raise FileWriteError(f"Failed to write Markdown: {e}")
            
            logger.info(f"Documentation written to {json_path}")
            return documentation
            
    except Exception as e:
        logger.error(f"Error writing documentation report: {e}")
        raise DocumentationError(f"Documentation write failed: {e}")
```

## gemini_model.py

```python
# gemini_model.py

"""
gemini_model.py

Handles interaction with the Gemini model, including token counting and 
documentation generation via the Gemini API.
"""

import aiohttp
import logging
import json
from typing import List, Dict, Any
from utils import TokenManager  # Import TokenManager
from token_utils import TokenManager

logger = logging.getLogger(__name__)

class GeminiModel:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    async def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> Dict[str, Any]:
        """Fetches documentation from the Gemini model API."""
        url = f"{self.endpoint}/generate-docs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Gemini documentation generated successfully.")
                    return data.get("documentation", {})
                else:
                    logger.error(f"Error generating documentation from Gemini: {response.status}")
                    return {}

    def calculate_tokens(self, base_info: str, context: str, chunk_content: str, schema: str) -> int:
        """
        Calculates token count for Gemini model prompts using TokenManager.

        Args:
            base_info: Project and style information
            context: Related code/documentation
            chunk_content: Content of the chunk being documented
            schema: JSON schema

        Returns:
            Total token count
        """
        total = 0
        for text in [base_info, context, chunk_content, schema]:
            token_result = TokenManager.count_tokens(text)
            total += token_result.token_count
        return total

    def generate_prompt(self, base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates prompt structure specifically for the Gemini model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for Gemini model.")
        return prompt    

```

## process_manager.py

```python
"""
process_manager.py

Enhanced documentation generation process manager with improved provider handling,
metrics tracking, and error management. Coordinates file processing, model
interactions, and documentation generation across multiple AI providers.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import yaml
import threading
import uuid

from utils import (
    FileHandler,
    MetricsCalculator,
    CodeFormatter,
    setup_logging,
    get_all_file_paths,
    load_json_schema
)
from file_handlers import (
    FileProcessor,
    APIHandler,
    ProcessingResult,
    ChunkManager,
    HierarchicalContextManager
)
from write_documentation_report import DocumentationGenerator

logger = logging.getLogger(__name__)

@dataclass
class ProviderMetrics:
    """Tracks metrics for an AI provider."""
    api_calls: int = 0
    api_errors: int = 0
    total_tokens: int = 0
    average_latency: float = 0.0
    rate_limit_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    successful_chunks: int = 0

    def update_latency(self, latency: float) -> None:
        """Updates average latency with new value."""
        if self.api_calls == 0:
            self.average_latency = latency
        else:
            self.average_latency = (
                (self.average_latency * (self.api_calls - 1) + latency) /
                self.api_calls
            )

    def record_error(self, error_type: str) -> None:
        """Records an API error."""
        self.api_errors += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of provider metrics."""
        error_rate = (self.api_errors / self.api_calls * 100) if self.api_calls > 0 else 0
        tokens_per_call = (self.total_tokens / self.api_calls) if self.api_calls > 0 else 0
        hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        retries_per_call = (self.retry_count / self.api_calls) if self.api_calls > 0 else 0

        return {
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "error_rate": error_rate,
            "average_latency": self.average_latency,
            "total_tokens": self.total_tokens,
            "tokens_per_call": tokens_per_call,
            "rate_limit_hits": self.rate_limit_hits,
            "cache_efficiency": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate
            },
            "retries": {
                "total": self.retry_count,
                "per_call": retries_per_call
            },
            "error_breakdown": self.error_types,
            "successful_chunks": self.successful_chunks
        }

@dataclass
class ProcessingMetrics:
    """Enhanced processing metrics with detailed tracking."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    provider_metrics: Dict[str, ProviderMetrics] = field(default_factory=lambda: {})
    error_types: Dict[str, int] = field(default_factory=dict)
    processing_times: List[float] = field(default_factory=list)

    def record_file_result(
        self,
        success: bool,
        processing_time: float,
        error_type: Optional[str] = None
    ) -> None:
        """Records file processing result."""
        self.processed_files += 1
        if success:
            self.successful_files += 1
        else:
            self.failed_files += 1
            if error_type:
                self.error_types[error_type] = (
                    self.error_types.get(error_type, 0) + 1
                )
        self.processing_times.append(processing_time)

    def get_provider_metrics(self, provider: str) -> ProviderMetrics:
        """Gets or creates metrics for a provider."""
        if provider not in self.provider_metrics:
            self.provider_metrics[provider] = ProviderMetrics()
        return self.provider_metrics[provider]

    def get_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive metrics summary."""
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else (datetime.now() - self.start_time).total_seconds()
        )
        return {
            "duration": {
                "seconds": duration,
                "formatted": str(datetime.now() - self.start_time)
            },
            "files": {
                "total": self.total_files,
                "processed": self.processed_files,
                "successful": self.successful_files,
                "failed": self.failed_files,
                "success_rate": (
                    self.successful_files / self.total_files * 100
                    if self.total_files > 0
                    else 0
                )
            },
            "chunks": {
                "total": self.total_chunks,
                "successful": self.successful_chunks,
                "success_rate": (
                    self.successful_chunks / self.total_chunks * 100
                    if self.total_chunks > 0
                    else 0
                )
            },
            "processing_times": {
                "average": (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times
                    else 0
                ),
                "min": min(self.processing_times) if self.processing_times else 0,
                "max": max(self.processing_times) if self.processing_times else 0
            },
            "providers": {
                provider: metrics.get_summary()
                for provider, metrics in self.provider_metrics.items()
            },
            "errors": {
                "total": self.failed_files,
                "types": self.error_types,
                "rate": (
                    self.failed_files / self.processed_files * 100
                    if self.processed_files > 0
                    else 0
                )
            }
        }

class ProviderConfig(BaseModel):
    """Enhanced provider configuration with validation."""
    name: str
    endpoint: str
    api_key: str
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    timeout: float = 30.0
    chunk_overlap: int = 200  # Token overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size in tokens
    max_parallel_chunks: int = 3

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v < 1 or v > 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        return v

class DocumentationRequest(BaseModel):
    """Enhanced API request model."""
    file_paths: List[str]
    skip_types: Optional[List[str]] = []
    project_info: Optional[str] = ""
    style_guidelines: Optional[str] = ""
    safe_mode: Optional[bool] = False
    project_id: str
    provider: str = "azure"
    max_concurrency: Optional[int] = 5
    priority: Optional[str] = "normal"
    callback_url: Optional[str] = None

    @validator("provider")
    def validate_provider(cls, v):
        valid_providers = {"azure", "gemini", "openai"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        if v not in {"low", "normal", "high"}:
            raise ValueError("Priority must be low, normal, or high")
        return v

class DocumentationResponse(BaseModel):
    """Enhanced API response model."""
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None

_manager_instance = None
_manager_lock = threading.Lock()

def get_manager_instance() -> 'DocumentationProcessManager':
    """Gets or creates DocumentationProcessManager instance."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            # Load provider configurations
            provider_configs = {
                name: ProviderConfig(**config)
                for name, config in load_provider_configs().items()
            }

            _manager_instance = DocumentationProcessManager(
                repo_root=os.getenv("REPO_ROOT", "."),
                output_dir=os.getenv("OUTPUT_DIR", "./docs"),
                provider_configs=provider_configs,
                max_concurrency=int(os.getenv("MAX_CONCURRENCY", 5)),
                cache_dir=os.getenv("CACHE_DIR")
            )

    return _manager_instance

def load_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Loads provider configurations from environment/files."""
    import os
    import yaml
    import json
    from pathlib import Path

    config_file = os.getenv('PROVIDER_CONFIG_FILE')

    if config_file:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Provider config file not found: {config_file}")

        # Try to read the file
        try:
            with open(config_path, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError("Unsupported config file format. Must be .yaml or .json")
        except Exception as e:
            raise ValueError(f"Failed to load provider config file: {e}")

    else:
        # Use default configuration
        config = {
            "azure": {
                "name": "azure",
                "endpoint": os.getenv("AZURE_ENDPOINT", "https://your-azure-endpoint.com"),
                "api_key": os.getenv("AZURE_API_KEY", "your-azure-api-key"),
                "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME", "your-deployment-name"),
                "api_version": os.getenv("AZURE_API_VERSION", "2023-01-01"),
                "model_name": os.getenv("AZURE_MODEL_NAME", "gpt-3"),
                "max_tokens": int(os.getenv("AZURE_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("AZURE_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("AZURE_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("AZURE_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("AZURE_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("AZURE_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("AZURE_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("AZURE_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("AZURE_MAX_PARALLEL_CHUNKS", "3"))
            },
            "gemini": {
                "name": "gemini",
                "endpoint": os.getenv("GEMINI_ENDPOINT", "https://your-gemini-endpoint.com"),
                "api_key": os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
                "model_name": os.getenv("GEMINI_MODEL_NAME", "gemini-model"),
                "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("GEMINI_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("GEMINI_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("GEMINI_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("GEMINI_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("GEMINI_MAX_PARALLEL_CHUNKS", "3"))
            },
            "openai": {
                "name": "openai",
                "endpoint": os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/engines"),
                "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
                "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-3"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("OPENAI_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("OPENAI_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("OPENAI_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("OPENAI_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("OPENAI_MAX_PARALLEL_CHUNKS", "3"))
            }
        }

    # Return the configuration
    return config

def calculate_estimated_completion(
    request: DocumentationRequest
) -> datetime:
    """Calculates estimated completion time."""
    import os
    from datetime import datetime, timedelta

    # For simplicity, we can assume that processing each file takes some base time plus some time proportional to its size.

    base_time_per_file = 5  # seconds
    time_per_kb = 0.1  # seconds per kilobyte

    total_time = 0.0

    for file_path in request.file_paths:
        try:
            file_size = os.path.getsize(file_path)  # in bytes
            file_time = base_time_per_file + (file_size / 1024) * time_per_kb
            total_time += file_time
        except Exception as e:
            # If file size can't be determined, use default time
            total_time += base_time_per_file

    estimated_completion_time = datetime.now() + timedelta(seconds=total_time)

    return estimated_completion_time

class DocumentationProcessManager:
    """Enhanced documentation process manager with improved error handling."""

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider_configs: Dict[str, ProviderConfig],
        function_schema: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 5,
        cache_dir: Optional[str] = None
    ):
        """
        Initializes the documentation process manager.

        Args:
            repo_root: Repository root path
            output_dir: Output directory
            provider_configs: Configuration for AI providers
            function_schema: Documentation schema
            max_concurrency: Maximum concurrent operations
            cache_dir: Directory for caching
        """
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider_configs = provider_configs
        self.function_schema = function_schema
        self.max_concurrency = max_concurrency

        # Initialize metrics
        self.metrics = ProcessingMetrics()

        # Initialize semaphores
        self.api_semaphore = asyncio.Semaphore(max_concurrency)
        self.file_semaphore = asyncio.Semaphore(max_concurrency * 2)

        # Initialize managers
        self.context_manager = HierarchicalContextManager(
            cache_dir=cache_dir
        )
        self.metrics_calculator = MetricsCalculator()
        self.doc_generator = DocumentationGenerator()

        # Initialize thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max(os.cpu_count() - 1, 1)
        )

        # Task tracking
        self._active_tasks: Dict[str, Dict[asyncio.Task, str]] = {}
        self._task_progress: Dict[str, float] = {}
        self._task_status: Dict[str, str] = {}

        logger.info(
            f"Initialized DocumentationProcessManager with providers: "
            f"{', '.join(provider_configs.keys())}"
        )

    async def process_files(
        self,
        request: DocumentationRequest,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Processes multiple files with improved task management and error handling.

        Args:
            request: Documentation request
            task_id: Unique task identifier

        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        try:
            self.metrics = ProcessingMetrics()
            self.metrics.total_files = len(request.file_paths)

            # Store task status
            self._task_status[task_id] = "in_progress"
            self._task_progress[task_id] = 0.0

            # Get provider configuration
            provider_config = self.provider_configs.get(request.provider)
            if not provider_config:
                raise ValueError(f"Unsupported provider: {request.provider}")

            # Initialize ProviderMetrics for the provider
            provider_metrics = self.metrics.get_provider_metrics(request.provider)

            # Initialize processing components
            async with aiohttp.ClientSession() as session:
                api_handler = APIHandler(
                    session=session,
                    config=provider_config,
                    semaphore=self.api_semaphore,
                    provider_metrics=provider_metrics  # Pass ProviderMetrics
                )

                file_processor = FileProcessor(
                    context_manager=self.context_manager,
                    api_handler=api_handler,
                    provider_config=provider_config,
                    provider_metrics=provider_metrics  # Pass ProviderMetrics
                )

                # Process files with priority handling
                prioritized_files = self._prioritize_files(
                    request.file_paths,
                    request.priority
                )

                results = []
                total_files = len(prioritized_files)
                completed = 0  # Initialize completed files counter

                # Create tasks for all files
                tasks = {}
                for file_path in prioritized_files:
                    task = asyncio.create_task(
                        self._process_file(
                            file_path=file_path,
                            processor=file_processor,
                            request=request,
                            timeout=provider_config.timeout
                        )
                    )
                    tasks[task] = file_path  # Map task to file path

                # Keep track of active tasks
                self._active_tasks[task_id] = tasks

                for future in asyncio.as_completed(tasks.keys()):
                    file_path = tasks[future]
                    try:
                        result = await future
                        results.append(result)
                        self.metrics.record_file_result(
                            success=result["success"],
                            processing_time=result.get("processing_time", 0.0),
                            error_type=result.get("error_type")
                        )
                    except asyncio.CancelledError:
                        logger.warning(f"Processing of {file_path} was cancelled.")
                        self.metrics.record_file_result(
                            success=False,
                            processing_time=0.0,
                            error_type="CancelledError"
                        )
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": "Processing was cancelled",
                            "error_type": "CancelledError"
                        })
                    except Exception as e:
                        error_type = type(e).__name__
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        self.metrics.record_file_result(
                            success=False,
                            processing_time=0.0,
                            error_type=error_type
                        )
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": str(e),
                            "error_type": error_type
                        })
                    finally:
                        completed += 1  # Increment completed files counter
                        # Update progress based on files processed
                        progress = (completed / total_files) * 100
                        self._update_task_progress(task_id, progress)

                        # Send progress callback if configured
                        if request.callback_url:
                            await self._send_progress_callback(
                                request.callback_url,
                                progress,
                                completed,
                                total_files
                            )

                # Update final metrics
                self.metrics.end_time = datetime.now()

                # Mark task as completed
                self._task_status[task_id] = "completed"
                self._task_progress[task_id] = 100.0
                # Remove from active tasks
                del self._active_tasks[task_id]

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "results": results,
                    "metrics": self.metrics.get_summary()
                }

        except Exception as e:
            logger.error(f"Critical error in process_files: {str(e)}")
            # Mark task as failed
            self._task_status[task_id] = "failed"
            self._task_progress[task_id] = 100.0
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            raise

    async def _process_file(
        self,
        file_path: str,
        processor: FileProcessor,
        request: DocumentationRequest,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Processes a single file with improved error handling and timeouts."""
        start_time = datetime.now()

        try:
            if timeout:
                result = await asyncio.wait_for(
                    processor.process_file(
                        file_path=file_path,
                        skip_types=set(request.skip_types),
                        project_info=request.project_info,
                        style_guidelines=request.style_guidelines,
                        repo_root=str(self.repo_root),
                        output_dir=str(self.output_dir),
                        provider=request.provider,
                        project_id=request.project_id,
                        safe_mode=request.safe_mode
                    ),
                    timeout=timeout
                )
            else:
                result = await processor.process_file(
                    file_path=file_path,
                    skip_types=set(request.skip_types),
                    project_info=request.project_info,
                    style_guidelines=request.style_guidelines,
                    repo_root=str(self.repo_root),
                    output_dir=str(self.output_dir),
                    provider=request.provider,
                    project_id=request.project_id,
                    safe_mode=request.safe_mode
                )

            processing_time = (
                datetime.now() - start_time
            ).total_seconds()

            return {
                "file_path": file_path,
                "success": result.success,
                "content": result.content,
                "error": result.error,
                "processing_time": processing_time,
                "retries": result.retries
            }

        except asyncio.TimeoutError:
            logger.error(f"Processing of {file_path} timed out.")
            return {
                "file_path": file_path,
                "success": False,
                "error": "Processing timed out",
                "error_type": "TimeoutError",
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }
        except asyncio.CancelledError:
            logger.warning(f"Processing of {file_path} was cancelled.")
            return {
                "file_path": file_path,
                "success": False,
                "error": "Processing was cancelled",
                "error_type": "CancelledError",
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }

    def _prioritize_files(
        self,
        file_paths: List[str],
        priority: str
    ) -> List[str]:
        """Prioritizes files based on various factors."""
        if priority == "low":
            return file_paths

        try:
            # Calculate file scores
            file_scores = []
            for file_path in file_paths:
                score = 0

                # Check file size
                size = Path(file_path).stat().st_size
                score += min(size / 1024, 100)  # Size score (max 100)

                # Check modification time
                mtime = Path(file_path).stat().st_mtime
                age_hours = (
                    datetime.now().timestamp() - mtime
                ) / 3600
                score += max(0, 100 - age_hours)  # Age score

                # Add file type priority
                ext = Path(file_path).suffix.lower()
                type_scores = {
                    '.py': 100,
                    '.js': 90,
                    '.ts': 90,
                    '.java': 85,
                    '.cpp': 85,
                    '.h': 80
                }
                score += type_scores.get(ext, 50)

                file_scores.append((score, file_path))

            # Sort by score (descending for high priority)
            file_scores.sort(reverse=priority == "high")
            return [f[1] for f in file_scores]

        except Exception as e:
            logger.warning(f"Error in file prioritization: {e}")
            return file_paths

    def _update_task_progress(
        self,
        task_id: str,
        progress: float
    ) -> None:
        """Updates task progress."""
        self._task_progress[task_id] = progress
        if progress >= 100:
            self._task_status[task_id] = "completed"
        else:
            self._task_status[task_id] = "in_progress"

    async def _send_progress_callback(
        self,
        callback_url: str,
        progress: float,
        processed: int,
        total: int
    ) -> None:
        """Sends progress callback to specified URL."""
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    callback_url,
                    json={
                        "progress": progress,
                        "processed_files": processed,
                        "total_files": total,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.warning(f"Error sending progress callback: {e}")

    async def get_task_status(
        self,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Gets status of a specific task."""
        if task_id not in self._task_status:
            return None

        return {
            "task_id": task_id,
            "status": self._task_status[task_id],
            "progress": self._task_progress.get(task_id, 0.0)
        }

    async def cancel_task(self, task_id: str) -> None:
        """Cancels a running task."""
        if task_id in self._active_tasks:
            tasks = self._active_tasks[task_id]
            for task in tasks.keys():
                task.cancel()
            self._task_status[task_id] = "cancelled"
            self._task_progress[task_id] = 100.0
            del self._active_tasks[task_id]
            logger.info(f"Task {task_id} has been cancelled.")
        else:
            logger.warning(f"Task {task_id} not found or already completed.")

    async def cleanup(self) -> None:
        """Cleans up resources."""
        try:
            # Cancel active tasks
            for task_dict in self._active_tasks.values():
                for task in task_dict.keys():
                    task.cancel()

            # Cleanup thread pool
            self.thread_pool.shutdown(wait=True)

            # Clear context manager
            await self.context_manager.clear_context()

            # Clear task tracking
            self._active_tasks.clear()
            self._task_progress.clear()
            self._task_status.clear()

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# FastAPI app setup
app = FastAPI(title="Documentation Generator API")

@app.post("/api/documentation/generate", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest,
    background_tasks: BackgroundTasks
) -> DocumentationResponse:
    """API endpoint to generate documentation."""
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Load provider configurations
        provider_configs = {
            name: ProviderConfig(**config)
            for name, config in load_provider_configs().items()
        }

        manager = DocumentationProcessManager(
            repo_root=os.getenv("REPO_ROOT"),
            output_dir=os.getenv("OUTPUT_DIR"),
            provider_configs=provider_configs,
            max_concurrency=request.max_concurrency or 5,
            cache_dir=os.getenv("CACHE_DIR")
        )

        # Start processing task
        background_tasks.add_task(
            manager.process_files,
            request=request,
            task_id=task_id  # Pass task_id to process_files
        )

        return DocumentationResponse(
            task_id=task_id,
            status="started",
            progress=0.0,
            estimated_completion=calculate_estimated_completion(request)
        )

    except Exception as e:
        logger.error(f"Error in generate_documentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Documentation generation failed: {str(e)}"
        )

@app.get(
    "/api/documentation/status/{task_id}",
    response_model=DocumentationResponse
)
async def get_status(task_id: str) -> Dict[str, Any]:
    """API endpoint to get documentation generation status."""
    try:
        # Get manager instance
        manager = get_manager_instance()

        status = await manager.get_task_status(task_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        return status

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    # Setup logging
    setup_logging(
        log_file=os.getenv("LOG_FILE"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

    # Start API server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEBUG", False))
    )

```

## context_manager.py

```python
"""
context_manager.py

Manages code context and relationships between code chunks with enhanced caching,
metrics tracking, hierarchical organization, dependency graphs, and semantic similarity.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles
from collections import deque
import shutil
import hashlib
from functools import lru_cache

import networkx as nx
from sentence_transformers import SentenceTransformer

from code_chunk import CodeChunk
from token_utils import TokenManager, TokenizerModel, TokenizationError
from metrics import MetricsResult
from metrics_utils import calculate_code_metrics_with_metadata, CodeMetadata

logger = logging.getLogger(__name__)


# Custom Exceptions
class ChunkNotFoundError(Exception):
    """Raised when a requested chunk cannot be found."""
    pass


class InvalidChunkError(Exception):
    """Raised when a chunk is invalid or corrupted."""
    pass


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass


# Data Classes
@dataclass
class ChunkLocation:
    """Represents the location of a chunk in the project hierarchy."""
    project_path: str
    module_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    def get_hierarchy_path(self) -> str:
        """Returns the full hierarchical path of the chunk."""
        parts = [self.module_path]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def overlaps_with(self, other: 'ChunkLocation') -> bool:
        """Checks if this location overlaps with another."""
        if self.module_path != other.module_path:
            return False
        return (self.start_line <= other.end_line and
                self.end_line >= other.start_line)


@dataclass
class ChunkMetadata:
    """Metadata for a code chunk."""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    hash: str = ""
    dependencies: Set[str] = field(default_factory=set)
    metrics: Optional[MetricsResult] = None

    def update_hash(self, content: str) -> None:
        """Updates the content hash."""
        self.hash = hashlib.sha256(content.encode()).hexdigest()
        self.last_modified = datetime.now()


# Node Class for Hierarchical Structure
class Node:
    """Node in the context tree."""

    def __init__(
        self,
        name: str,
        chunk: Optional[CodeChunk] = None,
        location: Optional[ChunkLocation] = None,
        metadata: Optional[ChunkMetadata] = None
    ):
        self.name = name
        self.chunk = chunk
        self.location = location
        self.metadata = metadata or ChunkMetadata()
        self.children: Dict[str, 'Node'] = {}
        self.parent: Optional['Node'] = None

    def add_child(self, child: 'Node') -> None:
        """Adds a child node."""
        self.children[child.name] = child
        child.parent = self

    def remove_child(self, name: str) -> None:
        """Removes a child node."""
        if name in self.children:
            self.children[name].parent = None
            del self.children[name]

    def get_ancestors(self) -> List['Node']:
        """Gets all ancestor nodes."""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> List['Node']:
        """Gets all descendant nodes."""
        descendants = []
        queue = deque(self.children.values())
        while queue:
            node = queue.popleft()
            descendants.append(node)
            queue.extend(node.children.values())
        return descendants


# HierarchicalContextManager Class
class HierarchicalContextManager:
    """Manages code chunks and documentation with a tree structure, caching, dependency graphs, and semantic similarity."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 1000,
        token_model: TokenizerModel = TokenizerModel.GPT4,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initializes the context manager.

        Args:
            cache_dir: Directory for persistent cache
            max_cache_size: Maximum number of items in memory cache
            token_model: Tokenizer model to use
            embedding_model: Sentence transformer model for embeddings
        """
        self._root = Node("root")
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._max_cache_size = max_cache_size
        self._token_model = token_model
        self._metrics: Dict[str, Any] = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_chunks': 0,
            'total_tokens': 0
        }

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Dependency Graph for Advanced Relationship Management
        self._dependency_graph = nx.DiGraph()
        self._embedding_model = SentenceTransformer(embedding_model)

    # Core Methods

    async def add_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Adds a code chunk to the tree with validation, dependency tracking, and metrics.

        Args:
            chunk: CodeChunk to add

        Raises:
            InvalidChunkError: If chunk is invalid
            TokenizationError: If token counting fails
        """
        try:
            # Validate chunk
            if not chunk.chunk_content.strip():
                raise InvalidChunkError("Empty chunk content")

            # Count tokens
            token_result = TokenManager.count_tokens(
                chunk.chunk_content,
                model=self._token_model
            )

            location = ChunkLocation(
                project_path=str(Path(chunk.file_path).parent),
                module_path=Path(chunk.file_path).stem,
                class_name=chunk.class_name,
                function_name=chunk.function_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line
            )

            metadata = ChunkMetadata(
                token_count=token_result.token_count,
                dependencies=set()
            )
            metadata.update_hash(chunk.chunk_content)

            async with self._lock:
                # Check for overlapping chunks
                if self._has_overlap(location):
                    logger.warning(f"Overlapping chunk detected at {location.get_hierarchy_path()}")

                # Add to tree
                path = location.get_hierarchy_path().split(".")
                current = self._root

                for part in path:
                    if part not in current.children:
                        current.children[part] = Node(part)
                    current = current.children[part]

                current.chunk = chunk
                current.location = location
                current.metadata = metadata

                # Update metrics
                self._metrics['total_chunks'] += 1
                self._metrics['total_tokens'] += token_result.token_count

                logger.debug(f"Added chunk {chunk.chunk_id} to context")

                # Add to dependency graph
                self._add_to_dependency_graph(chunk)

        except (InvalidChunkError, TokenizationError) as e:
            logger.error(f"Validation error adding chunk: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding chunk: {str(e)}")
            raise

    def _has_overlap(self, location: ChunkLocation) -> bool:
        """Checks if a location overlaps with existing chunks."""
        for node in self._get_module_nodes(location.module_path):
            if node.location and node.location.overlaps_with(location):
                return True
        return False

    def _get_module_nodes(self, module_path: str) -> List[Node]:
        """Gets all nodes in a module."""
        nodes = []
        if module_path in self._root.children:
            module_node = self._root.children[module_path]
            nodes.extend([module_node] + module_node.get_descendants())
        return nodes

    def _add_to_dependency_graph(self, chunk: CodeChunk) -> None:
        """
        Adds a chunk to the dependency graph based on semantic similarity.

        Args:
            chunk: CodeChunk to add
        """
        self._dependency_graph.add_node(chunk.chunk_id, chunk=chunk)

        related_chunks = self._find_related_chunks(chunk)
        for related_chunk_id in related_chunks:
            self._dependency_graph.add_edge(chunk.chunk_id, related_chunk_id)

    def _find_related_chunks(self, chunk: CodeChunk, threshold: float = 0.7) -> List[str]:
        """
        Finds related chunks based on semantic similarity.

        Args:
            chunk: CodeChunk to compare
            threshold: Similarity threshold

        Returns:
            List of related chunk IDs
        """
        chunk_embedding = self._embedding_model.encode(chunk.chunk_content)
        related_chunks = []

        for node_id in self._dependency_graph.nodes:
            other_chunk = self._dependency_graph.nodes[node_id]['chunk']
            other_embedding = self._embedding_model.encode(other_chunk.chunk_content)
            similarity = self._cosine_similarity(chunk_embedding, other_embedding)

            if similarity > threshold:
                related_chunks.append(node_id)

        return related_chunks

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        """Calculates cosine similarity between two vectors."""
        return float((vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    async def get_context_for_function(
        self,
        module_path: str,
        function_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """
        Gets context chunks for a function with token limit awareness.

        Args:
            module_path: Path to the module
            function_name: Name of the function
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Relevant context chunks
        """
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, function_name]
            node = self._find_node(path)

            if node and node.chunk:
                context_chunks = self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id,
                    language=language,
                    max_tokens=max_tokens
                )

                return context_chunks
            return []

    async def get_context_for_class(
        self,
        module_path: str,
        class_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a class."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, class_name]
            node = self._find_node(path)

            if node:
                return self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id if node.chunk else "",
                    language=language,
                    max_tokens=max_tokens
                )
            return []

    async def get_context_for_module(
        self,
        module_path: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a module."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name]
            node = self._find_node(path)

            if node:
                return self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id if node.chunk else "",
                    language=language,
                    max_tokens=max_tokens
                )
            return []

    def _get_context_from_graph(
        self,
        chunk_id: str,
        language: str,
        max_tokens: int
    ) -> List[CodeChunk]:
        """
        Performs intelligent context gathering using dependency graph traversal with token limiting.

        Args:
            chunk_id: Starting chunk ID
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Context chunks within token limit
        """
        context_chunks = []
        total_tokens = 0
        visited = set()
        queue = deque([(chunk_id, 0)])  # (chunk_id, depth)

        while queue and total_tokens < max_tokens:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            current_chunk = self._dependency_graph.nodes[current_id]['chunk']

            if current_chunk.language == language:
                if total_tokens + current_chunk.token_count <= max_tokens:
                    context_chunks.append(current_chunk)
                    total_tokens += current_chunk.token_count

                    # Enqueue related chunks based on dependencies
                    for neighbor in self._dependency_graph.neighbors(current_id):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

            # Optionally, you can limit the depth to avoid too deep traversal
            if depth >= 5:  # Example depth limit
                continue

        return context_chunks

    def _get_context_from_tree(
        self,
        node: Node,
        language: str,
        max_tokens: int
    ) -> List[CodeChunk]:
        """
        Performs intelligent context gathering using BFS with token limiting.

        Args:
            node: Starting node
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Context chunks within token limit
        """
        context_chunks = []
        total_tokens = 0
        visited = set()
        queue = deque([(node, 0)])  # (node, depth)

        while queue and total_tokens < max_tokens:
            current, depth = queue.popleft()

            if current.chunk_id in visited:
                continue

            visited.add(current.chunk_id)

            if (current.chunk and
                current.chunk.language == language and
                current.metadata):

                # Check if adding this chunk would exceed token limit
                if (total_tokens + current.metadata.token_count <= max_tokens):
                    context_chunks.append(current.chunk)
                    total_tokens += current.metadata.token_count

                    # Add related chunks based on dependencies
                    for dep_id in current.metadata.dependencies:
                        dep_node = self._find_node_by_id(dep_id)
                        if dep_node and dep_node.chunk_id not in visited:
                            queue.append((dep_node, depth + 1))

            # Add siblings and children with priority based on depth
            siblings = [
                (n, depth) for n in current.parent.children.values()
                if n.chunk_id not in visited
            ] if current.parent else []

            children = [
                (n, depth + 1) for n in current.children.values()
                if n.chunk_id not in visited
            ]

            # Prioritize closer relationships
            queue.extend(sorted(siblings + children, key=lambda x: x[1]))

        return context_chunks

    # Caching Methods

    async def _cache_documentation(
        self,
        chunk_id: str,
        documentation: Dict[str, Any]
    ) -> None:
        """Caches documentation to disk with error handling."""
        if not self._cache_dir:
            return

        cache_path = self._cache_dir / f"{chunk_id}.json"
        try:
            async with aiofiles.open(cache_path, 'w', encoding="utf-8") as f:
                await f.write(json.dumps({
                    'documentation': documentation,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'token_count': self._get_doc_token_count(documentation),
                        'hash': self._calculate_doc_hash(documentation)
                    }
                }, indent=2))
            logger.debug(f"Cached documentation for chunk {chunk_id}")

        except Exception as e:
            logger.error(f"Failed to cache documentation: {e}")
            raise CacheError(f"Cache write failed: {str(e)}")

    def _get_doc_token_count(self, documentation: Dict[str, Any]) -> int:
        """Calculates token count for documentation."""
        try:
            return TokenManager.count_tokens(
                json.dumps(documentation),
                model=self._token_model
            ).token_count
        except TokenizationError:
            return 0

    def _calculate_doc_hash(self, documentation: Dict[str, Any]) -> str:
        """Calculates hash for documentation content."""
        return hashlib.sha256(
            json.dumps(documentation, sort_keys=True).encode()
        ).hexdigest()

    async def get_documentation_for_chunk(
        self,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves documentation with caching and validation.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Optional[Dict[str, Any]]: Documentation if found

        Raises:
            ChunkNotFoundError: If chunk not found
            CacheError: If cache operations fail
        """
        async with self._lock:
            # Check memory cache
            if chunk_id in self._docs:
                self._metrics['cache_hits'] += 1
                return self._docs[chunk_id]

            # Check disk cache
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                if cache_path.exists():
                    try:
                        async with aiofiles.open(
                            cache_path, 'r', encoding="utf-8"
                        ) as f:
                            cached = json.loads(await f.read())

                        # Validate cache
                        node = self._find_node_by_id(chunk_id)
                        if node and node.metadata:
                            cached_hash = cached.get('metadata', {}).get('hash')
                            if cached_hash == node.metadata.hash:
                                doc = cached['documentation']
                                self._docs[chunk_id] = doc
                                self._metrics['cache_hits'] += 1
                                return doc

                    except Exception as e:
                        logger.error(f"Cache read failed: {e}")

            self._metrics['cache_misses'] += 1
            return None

    async def update_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Updates an existing chunk with change detection.

        Args:
            chunk: Updated chunk

        Raises:
            ChunkNotFoundError: If chunk not found
            InvalidChunkError: If chunk is invalid
        """
        async with self._lock:
            path = chunk.get_hierarchy_path().split(".")
            node = self._find_node(path)

            if not node:
                raise ChunkNotFoundError(
                    f"No chunk found for path: {'.'.join(path)}"
                )

            # Calculate new metadata
            token_result = TokenManager.count_tokens(
                chunk.chunk_content,
                model=self._token_model
            )

            new_metadata = ChunkMetadata(
                token_count=token_result.token_count,
                dependencies=node.metadata.dependencies if node.metadata else set()
            )
            new_metadata.update_hash(chunk.chunk_content)

            # Check if content actually changed
            if (node.metadata and
                node.metadata.hash == new_metadata.hash):
                logger.debug(f"Chunk {chunk.chunk_id} unchanged, skipping update")
                return

            # Update node
            node.chunk = chunk
            node.metadata = new_metadata

            # Update dependency graph
            self._dependency_graph.remove_node(chunk.chunk_id)
            self._add_to_dependency_graph(chunk)

            # Invalidate cached documentation
            self._docs.pop(chunk.chunk_id, None)
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk.chunk_id}.json"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached documentation: {e}")

            logger.debug(f"Updated chunk {chunk.chunk_id}")

    async def remove_code_chunk(self, chunk_id: str) -> None:
        """
        Removes a chunk and its documentation.

        Args:
            chunk_id: Chunk to remove

        Raises:
            ChunkNotFoundError: If chunk not found
        """
        async with self._lock:
            node = self._find_node_by_id(chunk_id)
            if not node:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")

            # Remove from tree
            if node.parent:
                node.parent.remove_child(node.name)

            # Remove from dependency graph
            if self._dependency_graph.has_node(chunk_id):
                self._dependency_graph.remove_node(chunk_id)

            # Remove documentation
            self._docs.pop(chunk_id, None)

            # Remove from cache
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached documentation: {e}")

            # Update metrics
            if node.metadata:
                self._metrics['total_chunks'] -= 1
                self._metrics['total_tokens'] -= node.metadata.token_count

            logger.debug(f"Removed chunk {chunk_id}")

    async def clear_context(self) -> None:
        """Clears all chunks, documentation, and cache."""
        async with self._lock:
            self._root = Node("root")
            self._docs.clear()
            self._metrics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_chunks': 0,
                'total_tokens': 0
            }

            if self._cache_dir:
                try:
                    shutil.rmtree(self._cache_dir)
                    self._cache_dir.mkdir()
                    logger.debug("Cache directory cleared")
                except OSError as e:
                    logger.error(f"Error clearing cache directory: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current metrics."""
        return {
            **self._metrics,
            'cache_hit_ratio': (
                self._metrics['cache_hits'] /
                (self._metrics['cache_hits'] + self._metrics['cache_misses'])
                if (self._metrics['cache_hits'] + self._metrics['cache_misses']) > 0
                else 0
            ),
            'avg_tokens_per_chunk': (
                self._metrics['total_tokens'] / self._metrics['total_chunks']
                if self._metrics['total_chunks'] > 0
                else 0
            )
        }

    def _find_node_by_id(self, chunk_id: str) -> Optional[Node]:
        """Finds a node by chunk ID."""
        queue = deque([self._root])
        while queue:
            current = queue.popleft()
            if current.chunk and current.chunk.chunk_id == chunk_id:
                return current
            queue.extend(current.children.values())
        return None

    def _find_node(self, path: List[str]) -> Optional[Node]:
        """Finds a node by path."""
        current = self._root
        for part in path:
            if part not in current.children:
                return None
            current = current.children[part]
        return current

    async def optimize_cache(self) -> None:
        """Optimizes cache by removing least recently used items."""
        if len(self._docs) > self._max_cache_size:
            sorted_docs = sorted(
                self._docs.items(),
                key=lambda x: x[1].get('metadata', {}).get('last_modified', datetime.min)
            )
            to_remove = sorted_docs[:-self._max_cache_size]
            for chunk_id, _ in to_remove:
                self._docs.pop(chunk_id)
                logger.debug(f"Optimized cache by removing chunk {chunk_id}")

    async def get_related_chunks(
        self,
        chunk_id: str,
        max_distance: int = 2
    ) -> List[CodeChunk]:
        """
        Gets related chunks based on dependencies and proximity.

        Args:
            chunk_id: Starting chunk
            max_distance: Maximum relationship distance

        Returns:
            List[CodeChunk]: Related chunks
        """
        node = self._find_node_by_id(chunk_id)
        if not node:
            return []

        related = []
        visited = set()
        queue = deque([(node, 0)])  # (node, distance)

        while queue:
            current, distance = queue.popleft()

            if distance > max_distance:
                continue

            if current.chunk_id in visited:
                continue

            visited.add(current.chunk_id)

            if current.chunk and current != node:
                related.append(current.chunk)

            # Add dependencies from the dependency graph
            if self._dependency_graph.has_node(current.chunk_id):
                for dep_id in self._dependency_graph.predecessors(current.chunk_id):
                    dep_node = self._find_node_by_id(dep_id)
                    if dep_node and dep_node.chunk_id not in visited:
                        queue.append((dep_node, distance + 1))

            # Add siblings
            if current.parent:
                for sibling in current.parent.children.values():
                    if sibling != current and sibling.chunk_id not in visited:
                        queue.append((sibling, distance + 1))

            # Add children
            for child in current.children.values():
                if child.chunk_id not in visited:
                    queue.append((child, distance + 1))

        return related

    # Helper Methods

    async def _process_dependencies(self, chunk: CodeChunk) -> None:
        """Processes and updates dependencies for a chunk."""
        # Placeholder for any additional dependency processing logic
        pass

```

## setup_logging.py

```python
import logging
import logging.handlers

def setup_logging(log_file: str, log_level: str = "INFO", formatter_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """Sets up logging configuration."""
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.hasHandlers():
        # Create a rotating file handler
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)  # Log all levels to file

            # Create a formatter
            formatter = logging.Formatter(formatter_str)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except (IOError, OSError) as e:
            print(f"Failed to set up file handler for logging: {e}")
            return False  # Indicate failure

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Log INFO and above to console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return True  # Indicate success
```

## results.py

```python
# results.py

"""
results.py

Contains data classes for storing processing results, such as FileProcessingResult.
These classes are used across multiple modules to standardize result storage and access.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class FileProcessingResult:
    """
    Stores the result of processing a file.

    Attributes:
        file_path: Path to the processed file
        success: Whether processing succeeded
        error: Error message if processing failed
        documentation: Generated documentation if successful
        metrics: Metrics about the processing
        chunk_count: Number of chunks processed
        successful_chunks: Number of successfully processed chunks
        timestamp: When the processing completed
    """
    file_path: str
    success: bool
    error: Optional[str] = None
    documentation: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    chunk_count: int = 0
    successful_chunks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
```

## chunk_manager.py

```python
"""
chunk_manager.py

Manages code chunking operations with context awareness, utilizing AST analysis
to create meaningful and syntactically valid code segments.
"""

import ast
import logging
import os
from typing import List, Optional, Dict, Set
from code_chunk import CodeChunk
from token_utils import TokenManager

logger = logging.getLogger(__name__)

class ChunkManager:
    """Manages code chunking operations."""

    def __init__(self, max_tokens: int = 4096, overlap: int = 200, repo_path: str = "."):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.token_manager = TokenManager()
        self.project_structure = self.analyze_project_structure(repo_path)

    def analyze_project_structure(self, repo_path: str) -> Dict:
        """Analyzes the project's directory structure."""
        project_structure = {"modules": {}}
        for root, dirs, files in os.walk(repo_path):
            if "__init__.py" in files:
                module_name = os.path.basename(root)
                project_structure["modules"][module_name] = {
                    "files": [os.path.join(root, f) for f in files if f.endswith(".py")],
                    "dependencies": []
                }

        for module_name, module_data in project_structure["modules"].items():
            for file_path in module_data["files"]:
                with open(file_path, "r") as f:
                    code = f.read()
                    try:
                        tree = ast.parse(code)
                        analyzer = DependencyAnalyzer(project_structure)
                        analyzer.visit(tree)
                        module_data["dependencies"] = list(analyzer.dependencies)
                    except SyntaxError:
                        logger.warning(f"Syntax error in {file_path}, skipping dependency analysis.")

        return project_structure

    def create_chunks(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Creates code chunks with context awareness."""
        if language.lower() == "python":
            return self._create_python_chunks(code, file_path)
        else:
            return self._create_simple_chunks(code, file_path, language)

    def _create_python_chunks(self, code: str, file_path: str) -> List[CodeChunk]:
        """Creates chunks for Python code using AST analysis."""
        try:
            tree = ast.parse(code)
            chunks = []
            current_chunk_lines = []
            current_chunk_start = 1
            current_token_count = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if current_chunk_lines:
                        chunks.append(self._create_chunk_from_lines(
                            current_chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))
                        current_chunk_lines = []
                        current_token_count = 0

                    current_chunk_start = node.lineno
                    current_chunk_lines.extend(code.splitlines()[node.lineno - 1:node.end_lineno])
                    current_token_count += self.token_manager.count_tokens("\n".join(current_chunk_lines)).token_count

                    while current_token_count >= self.max_tokens - self.overlap:
                        split_point = self._find_split_point(node, current_chunk_lines)
                        if split_point is None:
                            logger.warning(f"Chunk too large to split: {node.name} in {file_path}")
                            break

                        chunk_lines = current_chunk_lines[:split_point]
                        chunks.append(self._create_chunk_from_lines(
                            chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))

                        current_chunk_start += len(chunk_lines)
                        current_chunk_lines = current_chunk_lines[split_point:]
                        current_token_count = self.token_manager.count_tokens("\n".join(current_chunk_lines)).token_count

                elif current_chunk_lines:
                    current_chunk_lines.append(code.splitlines()[node.lineno - 1])
                    current_token_count += self.token_manager.count_tokens(code.splitlines()[node.lineno - 1]).token_count

                    if current_token_count >= self.max_tokens - self.overlap:
                        chunks.append(self._create_chunk_from_lines(
                            current_chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))
                        current_chunk_lines = []
                        current_token_count = 0
                        current_chunk_start = node.lineno

            if current_chunk_lines:
                chunks.append(self._create_chunk_from_lines(
                    current_chunk_lines,
                    current_chunk_start,
                    file_path,
                    "python"
                ))

            return chunks

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error creating Python chunks: {e}")
            return []

    def _create_simple_chunks(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Creates chunks based on lines of code with a simple token count limit."""
        chunks = []
        current_chunk_lines = []
        current_chunk_start = 1
        current_token_count = 0

        for i, line in enumerate(code.splitlines(), 1):
            current_chunk_lines.append(line)
            current_token_count += self.token_manager.count_tokens(line).token_count

            if current_token_count >= self.max_tokens - self.overlap:
                chunks.append(self._create_chunk_from_lines(
                    current_chunk_lines,
                    current_chunk_start,
                    file_path,
                    language
                ))
                current_chunk_lines = []
                current_token_count = 0
                current_chunk_start = i + 1

        if current_chunk_lines:
            chunks.append(self._create_chunk_from_lines(
                current_chunk_lines,
                current_chunk_start,
                file_path,
                language
            ))

        return chunks

    def _create_chunk_from_lines(
        self,
        lines: List[str],
        start_line: int,
        file_path: str,
        language: str
    ) -> CodeChunk:
        """Creates a CodeChunk from a list of lines."""
        chunk_content = "\n".join(lines)
        end_line = start_line + len(lines) - 1
        chunk_id = f"{file_path}:{start_line}-{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language=language,
        )

    async def get_contextual_chunks(
        self,
        chunk: CodeChunk,
        all_chunks: List[CodeChunk],
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Retrieves contextually relevant chunks for a given chunk."""
        context_chunks = [chunk]
        current_tokens = chunk.token_count
        visited_chunks = {chunk.chunk_id}

        module_name = self._get_module_name(chunk.file_path)
        if module_name:
            module_dependencies = self.project_structure["modules"][module_name].get("dependencies", [])
            for dep_module in module_dependencies:
                for c in all_chunks:
                    if (c.file_path.startswith(dep_module.replace(".", "/")) and
                        c.chunk_id not in visited_chunks and
                        current_tokens + c.token_count <= max_tokens):
                        context_chunks.append(c)
                        current_tokens += c.token_count
                        visited_chunks.add(c.chunk_id)

        return context_chunks

    def _get_module_name(self, file_path: str) -> Optional[str]:
        """Gets the module name from a file path."""
        for module_name, module_data in self.project_structure["modules"].items():
            if file_path in module_data["files"]:
                return module_name
        return None

    def _find_split_point(self, node: ast.AST, lines: List[str]) -> Optional[int]:
        """Finds a suitable split point within a function or class."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i, line in enumerate(lines):
                if line.strip().startswith("return") or line.strip().startswith("yield"):
                    return i + 1
        elif isinstance(node, ast.ClassDef):
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    return i
        return None

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyzes dependencies within a Python file."""

    def __init__(self, project_structure: Dict):
        self.project_structure = project_structure
        self.dependencies: Set[str] = set()
        self.scope_stack: List[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call):
        called_func_name = self._get_called_function_name(node)
        if called_func_name and not self._is_in_current_scope(called_func_name):
            self._add_dependency_if_exists(called_func_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            object_name = node.value.id
            attribute_name = node.attr
            full_name = f"{object_name}.{attribute_name}"
            if not self._is_in_current_scope(full_name):
                self._add_dependency_if_exists(full_name)
        self.generic_visit(node)

    def _get_called_function_name(self, node: ast.Call) -> Optional[str]:
        """Gets the name of the called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _is_in_current_scope(self, name: str) -> bool:
        """Checks if a name is defined in the current scope."""
        return any(name in [n.name for n in self.scope_stack if hasattr(n, 'name')])

    def _add_dependency_if_exists(self, name: str):
        """Adds a dependency if the name exists in another module."""
        for other_module_name, other_module_data in self.project_structure["modules"].items():
            if any(name in f for f in other_module_data["files"]):
                self.dependencies.add(other_module_name)
```

## shared_functions.py

```python
# shared_functions.py

"""
shared_functions.py

Provides shared functions for both Gemini and Azure model integrations,
such as token encoding, prompt token calculation, logging setup, and 
data transformation utilities.
"""

import json
import logging
from typing import List, Dict, Any
from token_utils import TokenManager

logger = logging.getLogger(__name__)

# Define default thresholds and configurations
DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 5, "medium": 10, "high": 15}
DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 500, "medium": 1000, "high": 2000},
    "difficulty": {"low": 5, "medium": 10, "high": 20},
    "effort": {"low": 500, "medium": 1000, "high": 2000},
}
DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 65, "medium": 85, "high": 100}

def calculate_prompt_tokens(base_info: str, context: str, chunk_content: str, schema: str) -> int:
    """
    Calculates total tokens for the prompt content using TokenManager.

    Args:
        base_info: Project and style information
        context: Related code/documentation
        chunk_content: Content of the chunk being documented
        schema: JSON schema

    Returns:
        Total token count
    """
    total = 0
    for text in [base_info, context, chunk_content, schema]:
        token_result = TokenManager.count_tokens(text)
        total += token_result.token_count
    return total

def format_prompt(base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates prompt messages to be used in the documentation generation request.

    Args:
        base_info: Information about the project
        context: Contextual information from other parts of the code
        chunk_content: Code chunk content
        schema: JSON schema for structure

    Returns:
        List of messages as prompt input
    """
    schema_str = json.dumps(schema, indent=2)
    return [
        {"role": "system", "content": base_info},
        {"role": "user", "content": context},
        {"role": "assistant", "content": chunk_content},
        {"role": "schema", "content": schema_str}
    ]

def log_error(message: str, exc: Exception):
    """Logs errors with traceback information."""
    logger.error(f"{message}: {str(exc)}", exc_info=True)

```

## file_handlers.py

```python
"""
file_handlers.py

Contains classes and functions for handling file processing, API interactions,
chunk management, and context management for the documentation generation process.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import aiohttp

from utils import (
    should_process_file,
    get_language,
    FileHandler,
    CodeFormatter,
    TokenManager,
    ChunkAnalyzer,
    ProcessingResult,
    CodeChunk,
    ChunkValidationError,
    ChunkTooLargeError,
    ChunkingError,
    HierarchicalContextManager,
    MetricsCalculator,
    write_documentation_report
)
from chunk_manager import ChunkManager  # Import the new ChunkManager

logger = logging.getLogger(__name__)

class APIHandler:
    """Enhanced API interaction handler with better error handling and rate limiting."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: 'ProviderConfig',
        semaphore: asyncio.Semaphore,
        provider_metrics: 'ProviderMetrics'
    ):
        self.session = session
        self.config = config
        self.semaphore = semaphore
        self.provider_metrics = provider_metrics
        self._rate_limit_tokens = {}  # Track rate limits per endpoint
        self._rate_limit_lock = asyncio.Lock()


class FileProcessor:
    """Enhanced file processing with improved error handling and metrics."""

    def __init__(
        self,
        context_manager: HierarchicalContextManager,
        api_handler: APIHandler,
        provider_config: 'ProviderConfig',
        provider_metrics: 'ProviderMetrics',
        repo_path: str
    ):
        self.context_manager = context_manager
        self.api_handler = api_handler
        self.provider_config = provider_config
        self.provider_metrics = provider_metrics
        self.chunk_manager = ChunkManager(
            max_tokens=provider_config.max_tokens,
            overlap=provider_config.chunk_overlap,
            repo_path=repo_path
        )
        self.metrics_calculator = MetricsCalculator()

    def _build_prompt(
        self,
        context_chunk: List[CodeChunk],
        project_info: str,
        style_guidelines: str
    ) -> List[Dict[str, str]]:
        """Builds the prompt for the AI model with multi-level context."""
        main_chunk = context_chunk[0]  # The main chunk being documented

        # 1. Direct Context (Code Chunk)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate documentation for the following {main_chunk.language} code:\n\n```{main_chunk.language}\n{main_chunk.content}\n```"}
        ]

        # 2. Dependency Context
        dependencies = [
            f"* `{c.function_name}` ({self._get_module_name(c.file_path)}): {self._get_brief_description(c)}"
            for c in context_chunk[1:] if c.function_name
        ]
        if dependencies:
            prompt.append({"role": "user", "content": f"Dependencies:\n{chr(10).join(dependencies)}"})

        # 3. Extended Context (Module Summary)
        module_name = self._get_module_name(main_chunk.file_path)
        if module_name:
            module_summary = self._get_module_summary(module_name)  # You'll need to implement this
            prompt.append({"role": "user", "content": f"Module Summary:\n{module_summary}"})

        # Add project info and style guidelines
        if project_info:
            prompt.append({"role": "user", "content": f"Project Information:\n{project_info}"})
        if style_guidelines:
            prompt.append({"role": "user", "content": f"Style Guidelines:\n{style_guidelines}"})

        return prompt

    def _get_module_name(self, file_path: str) -> Optional[str]:
        """Extracts the module name from a file path."""
        # Implement logic to determine module name based on file path
        pass

    def _get_brief_description(self, chunk: CodeChunk) -> str:
        """Returns a brief description of a code chunk (e.g., first line of docstring)."""
        # Implement logic to extract a short description
        pass

    def _get_module_summary(self, module_name: str) -> str:
        """Retrieves the summary for a module."""
        # Implement logic to get the module summary, perhaps from a cache or analysis
        pass

    async def fetch_completion(
        self,
        prompt: List[Dict[str, str]],
        provider: str
    ) -> ProcessingResult:
        """
        Fetches completion with enhanced error handling and rate limiting.

        Args:
            prompt: Formatted prompt messages
            provider: AI provider name

        Returns:
            ProcessingResult: Processing result
        """
        start_time = datetime.now()
        attempt = 0
        last_error = None

        while attempt < self.config.max_retries:
            try:
                # Check rate limits
                await self._wait_for_rate_limit(provider)

                async with self.semaphore:
                    # Record API call attempt
                    self.provider_metrics.api_calls += 1

                    # Make API request
                    result = await self._make_api_request(prompt, provider)

                    # Calculate latency
                    latency = (datetime.now() - start_time).total_seconds()
                    self.provider_metrics.update_latency(latency)

                    # Update total tokens used
                    tokens_used = self._extract_tokens_used(result)
                    self.provider_metrics.total_tokens += tokens_used

                    # Update rate limit tracking
                    await self._update_rate_limits(provider, result)

                    processing_time = (
                        datetime.now() - start_time
                    ).total_seconds()

                    return ProcessingResult(
                        success=True,
                        content=result,
                        processing_time=processing_time
                    )

            except aiohttp.ClientError as e:
                error_type = "NetworkError"
                should_retry = True
                last_error = e
            except asyncio.TimeoutError:
                error_type = "TimeoutError"
                should_retry = True
                last_error = e
            except Exception as e:
                error_type = type(e).__name__
                should_retry = self._should_retry_error(str(e))
                last_error = e

            # Record error
            self.provider_metrics.record_error(error_type)

            if should_retry and attempt < self.config.max_retries - 1:
                attempt += 1
                self.provider_metrics.retry_count += 1
                delay = self._calculate_retry_delay(attempt, error_type)
                logger.warning(
                    f"Retry {attempt}/{self.config.max_retries} "
                    f"after {delay}s. Error: {error_type}"
                )
                await asyncio.sleep(delay)
                continue
            else:
                error_msg = f"API request failed: {error_type}"
                logger.error(error_msg)
                break

        processing_time = (datetime.now() - start_time).total_seconds()
        return ProcessingResult(
            success=False,
            error=str(last_error),
            retries=attempt,
            processing_time=processing_time
        )

    async def _make_api_request(
        self,
        prompt: List[Dict[str, str]],
        provider: str
    ) -> Dict[str, Any]:
        """Makes actual API request based on provider."""
        try:
            if provider == "azure":
                return await self._fetch_azure_completion(prompt)
            elif provider == "gemini":
                return await self._fetch_gemini_completion(prompt)
            elif provider == "openai":
                return await self._fetch_openai_completion(prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"API request error for {provider}: {str(e)}")
            raise

    async def _fetch_azure_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from Azure OpenAI service."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key
        }
        params = {
            "api-version": self.config.api_version
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        url = f"{self.config.endpoint}/openai/deployments/{self.config.deployment_name}/chat/completions"

        async with self.session.post(
            url, headers=headers, params=params, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_gemini_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from Gemini AI service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "model": self.config.model_name
        }
        url = f"{self.config.endpoint}/v1/chat/completions"

        async with self.session.post(
            url, headers=headers, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_openai_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from OpenAI service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "model": self.config.model_name
        }
        url = f"{self.config.endpoint}/v1/chat/completions"

        async with self.session.post(
            url, headers=headers, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _extract_tokens_used(self, response: Dict[str, Any]) -> int:
        """Extracts the number of tokens used from the API response."""
        usage = response.get('usage', {})
        return usage.get('total_tokens', 0)

    def _should_retry_error(self, error_message: str) -> bool:
        """Determines if an error should trigger a retry."""
        retry_patterns = [
            r"rate limit",
            r"timeout",
            r"too many requests",
            r"server error",
            r"503",
            r"429",
            r"connection",
            r"network",
            r"reset by peer"
        ]
        return any(
            re.search(pattern, error_message.lower())
            for pattern in retry_patterns
        )

    def _calculate_retry_delay(
        self,
        attempt: int,
        error_type: str
    ) -> float:
        """Calculates retry delay with exponential backoff and jitter."""
        base_delay = self.config.retry_delay
        max_delay = min(base_delay * (2 ** attempt), 60)  # Cap at 60 seconds

        # Add jitter (±25% of base delay)
        import random
        jitter = random.uniform(-0.25, 0.25) * base_delay

        # Increase delay for rate limit errors
        if "rate limit" in error_type.lower():
            max_delay *= 1.5

        return max(0.1, min(max_delay + jitter, 60))

    async def _wait_for_rate_limit(self, provider: str) -> None:
        """Waits if rate limit is reached."""
        async with self._rate_limit_lock:
            if provider in self._rate_limit_tokens:
                tokens, reset_time = self._rate_limit_tokens[provider]
                if tokens <= 0 and datetime.now() < reset_time:
                    wait_time = (reset_time - datetime.now()).total_seconds()
                    logger.warning(
                        f"Rate limit reached for {provider}. "
                        f"Waiting {wait_time:.1f}s"
                    )
                    self.provider_metrics.rate_limit_hits += 1
                    await asyncio.sleep(wait_time)

    async def _update_rate_limits(
        self,
        provider: str,
        response: Dict[str, Any]
    ) -> None:
        """Updates rate limit tracking based on response headers."""
        async with self._rate_limit_lock:
            # Extract rate limit info from response headers
            headers = response.get("headers", {})
            remaining = int(headers.get("x-ratelimit-remaining", 1))
            reset = int(headers.get("x-ratelimit-reset", 0))

            if reset > 0:
                reset_time = datetime.fromtimestamp(reset)
                self._rate_limit_tokens[provider] = (remaining, reset_time)

class FileProcessor:
    """Enhanced file processing with improved error handling and metrics."""

    def __init__(
        self,
        context_manager: HierarchicalContextManager,
        api_handler: APIHandler,
        provider_config: 'ProviderConfig',
        provider_metrics: 'ProviderMetrics'
    ):
        self.context_manager = context_manager
        self.api_handler = api_handler
        self.provider_config = provider_config
        self.provider_metrics = provider_metrics
        self.chunk_manager = ChunkManager(self.provider_config)
        self.metrics_calculator = MetricsCalculator()

    async def process_file(
        self,
        file_path: str,
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        repo_root: str,
        output_dir: str,
        provider: str,
        project_id: str,
        safe_mode: bool = False
    ) -> ProcessingResult:
        """Processes a single file with enhanced error handling."""
        start_time = datetime.now()

        try:
            # Basic validation
            if not should_process_file(file_path, skip_types):
                return ProcessingResult(
                    success=False,
                    error="File type excluded",
                    processing_time=0.0
                )

            # Read file content
            content = await FileHandler.read_file(file_path)
            if content is None:
                return ProcessingResult(
                    success=False,
                    error="Failed to read file",
                    processing_time=0.0
                )

            # Get language and validate
            language = get_language(file_path)
            if not language:
                return ProcessingResult(
                    success=False,
                    error="Unsupported language",
                    processing_time=0.0
                )

            # Create and process chunks
            try:
                chunks = self.chunk_manager.create_chunks(
                    content,
                    file_path,
                    language
                )
                self.provider_metrics.total_chunks += len(chunks)
            except ChunkingError as e:
                return ProcessingResult(
                    success=False,
                    error=f"Chunking error: {str(e)}",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds()
                )

            # Process chunks
            chunk_results = await self._process_chunks(
                chunks=chunks,
                project_info=project_info,
                style_guidelines=style_guidelines,
                provider=provider
            )

            # Combine documentation
            combined_doc = await self._combine_documentation(
                chunk_results=chunk_results,
                file_path=file_path,
                language=language
            )

            if not combined_doc:
                return ProcessingResult(
                    success=False,
                    error="Failed to combine documentation",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds()
                )

            # Write documentation if not in safe mode
            if not safe_mode:
                doc_result = await write_documentation_report(
                    documentation=combined_doc,
                    language=language,
                    file_path=file_path,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    project_id=project_id
                )

                if not doc_result:
                    return ProcessingResult(
                        success=False,
                        error="Failed to write documentation",
                        processing_time=(
                            datetime.now() - start_time
                        ).total_seconds()
                    )

            return ProcessingResult(
                success=True,
                content=combined_doc,
                processing_time=(
                    datetime.now() - start_time
                ).total_seconds()
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=(
                    datetime.now() - start_time
                ).total_seconds()
            )

    async def _process_chunks(
        self,
        chunks: List[CodeChunk],
        project_info: str,
        style_guidelines: str,
        provider: str
    ) -> List[ProcessingResult]:
        """Processes chunks with improved parallel handling."""
        results = []
        max_parallel_chunks = self.provider_config.max_parallel_chunks

        for i in range(0, len(chunks), max_parallel_chunks):
            chunk_group = chunks[i:i + max_parallel_chunks]
            tasks = [
                self._process_chunk(
                    chunk=chunk,
                    project_info=project_info,
                    style_guidelines=style_guidelines,
                    provider=provider
                )
                for chunk in chunk_group
            ]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(chunk_group, group_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk: {str(result)}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(result)
                    ))
                else:
                    results.append(result)
                    if result.success and result.content:
                        # Store successful results in context manager
                        try:
                            await self.context_manager.add_doc_chunk(
                                chunk.chunk_id,
                                result.content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to store chunk result: {str(e)}"
                            )
                        # Update successful chunks count
                        self.provider_metrics.successful_chunks += 1

        return results

    async def _process_chunk(
        self,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str,
        provider: str
    ) -> ProcessingResult:
        """Processes a single code chunk."""
        try:
            prompt = self._build_prompt(
                chunk=chunk,
                project_info=project_info,
                style_guidelines=style_guidelines
            )

            result = await self.api_handler.fetch_completion(
                prompt=prompt,
                provider=provider
            )

            if result.success:
                # Extract content from API response
                content = self._extract_content(result.content)
                return ProcessingResult(
                    success=True,
                    content=content,
                    processing_time=result.processing_time
                )
            else:
                return ProcessingResult(
                    success=False,
                    error=result.error,
                    processing_time=result.processing_time
                )

        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=0.0
            )

    def _build_prompt(
        self,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str
    ) -> List[Dict[str, str]]:
        """Builds the prompt for the AI model."""
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate documentation for the following code:\n\n{chunk.content}"}
        ]
        if project_info:
            prompt.append({"role": "user", "content": f"Project information:\n{project_info}"})
        if style_guidelines:
            prompt.append({"role": "user", "content": f"Style guidelines:\n{style_guidelines}"})
        return prompt

    def _extract_content(self, api_response: Dict[str, Any]) -> str:
        """Extracts content from the API response."""
        choices = api_response.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '')
        return ''

    async def _combine_documentation(
        self,
        chunk_results: List[ProcessingResult],
        file_path: str,
        language: str
    ) -> str:
        """Combines documentation from chunk results."""
        documentation = ""
        for result in chunk_results:
            if result.success and result.content:
                documentation += result.content + "\n\n"
        return documentation.strip()

class ChunkManager:
    """Manages code chunking operations."""

    def __init__(
        self,
        config: 'ProviderConfig',
        analyzer: Optional[ChunkAnalyzer] = None
    ):
        self.config = config
        self.analyzer = analyzer or ChunkAnalyzer()git  
        self.token_manager = TokenManager()

    def create_chunks(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[CodeChunk]:
        """
        Creates code chunks with smart splitting and validation.

        Args:
            content: Source code content
            file_path: Path to source file
            language: Programming language

        Returns:
            List[CodeChunk]: List of code chunks

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            lines = content.splitlines()
            chunks = []
            current_chunk = []
            for i, line in enumerate(lines):
                current_chunk.append(line)
                current_chunk_str = "\n".join(current_chunk)
                token_count = self.token_manager.count_tokens(current_chunk_str)
                if token_count >= self.config.max_tokens - self.config.chunk_overlap:
                    # Find a split point
                    split_line = i
                    while split_line > 0 and not self.analyzer.is_valid_split(lines[split_line]):
                        split_line -= 1
                    if split_line == 0:
                        raise ChunkTooLargeError("No valid split point found")
                    chunk_content = "\n".join(current_chunk[:split_line - len(current_chunk)])
                    if self.analyzer.is_valid_chunk(chunk_content, language):
                        chunk = self._create_chunk(
                            chunk_content,
                            split_line - len(current_chunk),
                            split_line,
                            file_path,
                            language
                        )
                        chunks.append(chunk)
                        # Start new chunk with overlap
                        overlap_start = max(0, split_line - self.config.chunk_overlap)
                        current_chunk = lines[overlap_start:i + 1]
                    else:
                        raise ChunkValidationError(f"Invalid chunk at line {split_line}")
            # Add final chunk
            if current_chunk:
                final_content = "\n".join(current_chunk)
                if self.analyzer.is_valid_chunk(final_content, language):
                    chunk = self._create_chunk(
                        final_content,
                        len(lines) - len(current_chunk) + 1,
                        len(lines),
                        file_path,
                        language
                    )
                    chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise ChunkingError(f"Chunking failed: {str(e)}") from e

    def _create_chunk(
        self,
        content: str,
        start_line: int,
        end_line: int,
        file_path: str,
        language: str
    ) -> CodeChunk:
        """Creates a CodeChunk object."""
        chunk_id = f"{file_path}:{start_line}-{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            content=content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language=language
        )
```

## main.py

```python
# main.py
import os
import sys
import logging
import argparse
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from azure_model import AzureModel
from gemini_model import GeminiModel
from openai_model import OpenAIModel
from process_manager import DocumentationProcessManager
from utils import DEFAULT_EXCLUDED_FILES, DEFAULT_EXCLUDED_DIRS, DEFAULT_SKIP_TYPES, load_config, load_function_schema, get_all_file_paths
from logging_config import setup_logging

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and insert docstrings using Azure OpenAI, Gemini, or OpenAI models.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--provider", help="Choose AI provider: 'azure', 'gemini', or 'openai'", default="azure")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default="schemas/function_schema.json")
    parser.add_argument("--doc-output-dir", help="Directory to save documentation files", default="documentation")
    parser.add_argument("--project-id", help="Unique identifier for the project", required=True)
    return parser.parse_args()

def validate_api_config(provider: str) -> Optional[str]:
    """Validates required API configuration based on the selected provider."""
    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            return "Missing AZURE_OPENAI_API_KEY"
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            return "Missing AZURE_OPENAI_ENDPOINT"
        if not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
            return "Missing AZURE_OPENAI_DEPLOYMENT"
    elif provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return "Missing GEMINI_API_KEY"
        if not os.getenv("GEMINI_ENDPOINT"):
            return "Missing GEMINI_ENDPOINT"
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return "Missing OPENAI_API_KEY"
    else:
        return f"Unsupported provider: {provider}"
    return None

async def main():
    """Main function."""
    args = parse_arguments()
    load_dotenv()

    # Configure logging 
    log_file = "documentation_generation.log"
    if not setup_logging(log_file, log_level=args.log_level):
        print("Failed to set up logging. Exiting...")
        sys.exit(1)

    logger.info("Starting documentation generation process...")  # Example log message

    repo_path = args.repo_path
    config_path = args.config
    output_dir = args.doc_output_dir

    # Validate API configuration based on provider
    if error := validate_api_config(args.provider):
        logger.error(f"Configuration error: {error}")
        sys.exit(1)

    try:
        # Initialize the appropriate model based on provider
        client = None
        if args.provider == "azure":
            client = AzureModel(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("API_VERSION", "2023-05-15")
            )
        elif args.provider == "gemini":
            client = GeminiModel(
                api_key=os.getenv("GEMINI_API_KEY"),
                endpoint=os.getenv("GEMINI_ENDPOINT")
            )
        elif args.provider == "openai":
            client = OpenAIModel(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            logger.error(f"Unsupported provider: {args.provider}")
            sys.exit(1)

        # Load configuration, schema, and file paths
        excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        excluded_files = set(DEFAULT_EXCLUDED_FILES)
        skip_types_set = set(DEFAULT_SKIP_TYPES)
        if args.skip_types:
            skip_types_set.update(ext.strip() for ext in args.skip_types.split(","))

        project_info, style_guidelines = load_config(config_path, excluded_dirs, excluded_files, skip_types_set)
        project_info = args.project_info or project_info
        style_guidelines = args.style_guidelines or style_guidelines

        function_schema = load_function_schema(args.schema)
        file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files, skip_types_set)

        # Initialize DocumentationProcessManager
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider=args.provider,
            azure_config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                "api_version": os.getenv("API_VERSION", "2024-05-01-preview")  # Moved inside azure_config
            },
            gemini_config={
                "api_key": os.getenv("GEMINI_API_KEY"),
                "endpoint": os.getenv("GEMINI_ENDPOINT")
            },
            openai_config={
                "api_key": os.getenv("OPENAI_API_KEY")
            },
            function_schema=function_schema,
            max_concurrency=args.concurrency
        )

        results = await manager.process_files(
            file_paths=file_paths,
            skip_types=skip_types_set,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=args.safe_mode  # safe_mode is still present
        )

        logger.info(f"Documentation generation completed. Results: {results}")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

```
## language_functions/base_handler.py

```python
"""
base_handler.py

This module defines the abstract base class `BaseHandler` for language-specific handlers.
Each handler is responsible for extracting code structure, inserting docstrings/comments,
and validating code for a specific programming language.

Classes:
    - BaseHandler: Abstract base class defining the interface for all language handlers.
"""

from __future__ import annotations  # For forward references in type hints
import abc
import logging
from typing import Dict, Any, Optional, List

from metrics import MetricsAnalyzer  # Import for type hinting

logger = logging.getLogger(__name__)

class BaseLanguageHandler(ABC):
    """
    Abstract base class for language-specific code handlers.

    Provides a common interface and shared functionality for handling different programming languages
    in the documentation generation process.  Subclasses must implement the `extract_structure` and
    `validate_code` methods.
    """

	 def __init__(self, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer):
        """
        Initializes the BaseLanguageHandler.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.
            metrics_analyzer: The metrics analyzer object for collecting code metrics.
        """
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer

    @abc.abstractmethod
    async def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the code (classes, functions, etc.).

        Subclasses must implement this method to parse the source code and identify key components
        such as classes, functions, methods, variables, and other relevant elements.

        Args:
            code (str): The source code to analyze.
            file_path (str): Path to the source file.

        Returns:
            Dict[str, Any]: A dictionary representing the code structure, including details
                            like classes, functions, variables, and their attributes.  Should conform
                            to the provided `function_schema`.
        """
        raise NotImplementedError
        
    def insert_docstrings(self, code: str, documentation: Dict[str, Any], docstring_format: str = "default") -> str:
        """
        Inserts docstrings into the code based on the documentation.

        Provides a default implementation that logs the docstring insertion attempt. Subclasses can override
        this method to implement language-specific docstring insertion logic.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.
            docstring_format (str, optional): The format of the docstrings to be inserted. Defaults to "default".

        Returns:
            str: The source code, potentially with inserted documentation. The default implementation returns
                 the original code unchanged.
        """
        logger.info(f"Inserting docstrings (format: {docstring_format})...")
        return code  # Return the original code if no specific logic is implemented

    @abc.abstractmethod
	 def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified code for syntax correctness.

        Subclasses must implement this method to ensure that the code remains syntactically correct after
        inserting docstrings/comments. It may involve compiling the code or running
        language-specific linters/validators.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        raise NotImplementedError
	 @abc.abstractmethod # Make this abstract if all handlers need to implement it
    def _calculate_complexity(self, code: str) -> Optional[float]:
        """
        Calculates code complexity.

        This method provides a default implementation that returns None. Subclasses can override
        this method to provide language-specific complexity calculations.

        Args:
            code (str): The source code to analyze.

        Returns:
            Optional[float]: The calculated complexity, or None if not implemented.
        """
        raise NotImplementedError
```

## language_functions/python_handler.py

```python
# python_handler.py

import ast
import logging
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from jsonschema import validate, ValidationError
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, h_visit
from .base_handler import BaseHandler
from metrics import (
    calculate_code_metrics,
    DEFAULT_EMPTY_METRICS,
    validate_metrics,
    calculate_quality_score,
    normalize_score,
    get_default_halstead_metrics,
    MetricsAnalyzer,
    MetricsThresholds
)
logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer):
        """Initialize the Python handler."""
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer

    async def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts the structure of the Python code, calculates complexity, and validates against the schema.

        Checklist:
        - [x] Parsing: Uses ast module.
        - [x] Data Structure: Conforms to function_schema.json.
        - [x] Schema Validation: Implemented using jsonschema.validate.
        - [x] Metrics Calculation: Uses radon and metrics.py.
        - [x] Language-Specific Features: Extracts decorators, argument types, return types.
        """
        logger.info(f"Extracting structure for file: {file_path}")
        try:
            if metrics is None:
                metrics_result = calculate_code_metrics(code, file_path, language="python")
                
                # Await the coroutine before accessing attributes
                metrics_result = await metrics_result 
                
                if metrics_result.success:
                    metrics = metrics_result.metrics
                else:
                    logger.warning(f"Metrics calculation failed for {file_path}: {metrics_result.error}")
                    metrics = DEFAULT_EMPTY_METRICS

                self.metrics_analyzer.add_result(metrics_result)

            tree = ast.parse(code)
            code_structure = {
                "docstring_format": "Google",
                "summary": "",
                "changes_made": [],
                "functions": [],
                "classes": [],
                "variables": [],
                "constants": [],
                "imports": [],
                "metrics": metrics,
            }

            class CodeVisitor(ast.NodeVisitor):
                def __init__(self, file_path: str):
                    self.scope_stack = []
                    self.file_path = file_path
                    self.current_class = None

                def _calculate_complexity(self, node):
                    try:
                        complexity_blocks = cc_visit(ast.unparse(node))
                        total_complexity = sum(block.complexity for block in complexity_blocks)
                        complexity_rank = cc_rank(total_complexity)
                        return total_complexity, complexity_rank
                    except Exception as e:
                        logger.error(f"Error calculating complexity: {e}")
                        return 0, "A"

                def visit_Module(self, node):
                    code_structure["summary"] = ast.get_docstring(node) or ""
                    for n in node.body:
                        if isinstance(n, (ast.Import, ast.ImportFrom)):
                            module_name = n.module if isinstance(n, ast.ImportFrom) else None
                            for alias in n.names:
                                imported_name = alias.name
                                full_import_path = f"{module_name}.{imported_name}" if module_name else imported_name
                                code_structure["imports"].append(full_import_path)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node, is_async=False):
                    self.scope_stack.append(node)
                    function_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "args": self._get_args(node),  # Extract arguments with type annotations
                        "async": is_async,
                        "returns": self._get_return_type(node),  # Extract return type
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                    }

                    complexity, rank = self._calculate_complexity(node)
                    function_info["complexity"] = complexity
                    function_info["complexity_rank"] = rank

                    code_structure["functions"].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def _get_args(self, node):
                    """Extracts function arguments with type annotations."""
                    args = []
                    for arg in node.args.args:
                        if arg.arg != "self":
                            arg_info = {"name": arg.arg}
                            if arg.annotation:
                                arg_info["type"] = ast.unparse(arg.annotation)
                            args.append(arg_info)
                    return args

                def _get_return_type(self, node):
                    """Extracts the return type annotation."""
                    if node.returns:
                        return ast.unparse(node.returns)
                    return None

                def visit_ClassDef(self, node):
                    self.current_class = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                    }
                    complexity, rank = self._calculate_complexity(node)
                    self.current_class["complexity"] = complexity
                    self.current_class["complexity_rank"] = rank

                    code_structure["classes"].append(self.current_class)
                    self.scope_stack.append(node)
                    self.generic_visit(node)
                    self.scope_stack.pop()
                    self.current_class = None

                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            is_constant = var_name.isupper()
                            try:
                                var_value = ast.literal_eval(node.value)
                                var_type = type(var_value).__name__
                            except ValueError:
                                var_value = None
                                var_type = "Unknown"

                            var_info = {
                                "name": var_name,
                                "type": var_type,
                                "value": var_value,
                                "lineno": node.lineno,
                                "end_lineno": node.end_lineno,
                            }
                            if is_constant:
                                code_structure["constants"].append(var_info)
                            else:
                                code_structure["variables"].append(var_info)

            visitor = CodeVisitor(file_path)
            visitor.visit(tree)

            try:
                validate(instance=code_structure, schema=self.function_schema)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e}")

            return code_structure

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}
        except Exception as e:
            logger.error(f"Error extracting Python structure from {file_path}: {e}", exc_info=True)
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts docstrings into the Python code.

        Checklist:
        - [x] Docstring Generation: Generates Google-style and NumPy-style docstrings.
        - [ ] Docstring Formats: Handles Google and NumPy formats (placeholders for others).
        - [x] Insertion Method: Uses AST manipulation.
        - [x] Error Handling: Includes error handling and logging.
        - [x] Preservation of Existing Docstrings: Allows preserving existing docstrings.
        """
        logger.info("Inserting docstrings...")
        try:
            tree = ast.parse(code)
            docstring_format = documentation.get("docstring_format", "Google")
            transformer = DocstringTransformer(documentation, docstring_format, preserve_existing=False)
            modified_tree = transformer.visit(tree)
            return ast.unparse(modified_tree)
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the Python code using pylint.

        Checklist:
        - [x] Validation Tool: Uses pylint.
        - [x] Error Handling: Handles validation errors.
        - [x] Temporary Files: Uses and cleans up temporary files.
        """
        logger.info("Validating code...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
                temp_file.write(code.encode("utf-8"))
                temp_file_path = temp_file.name

            result = subprocess.run(
                ["pylint", temp_file_path],
                capture_output=True,
                text=True,
                check=False
            )
            os.unlink(temp_file_path)

            if result.returncode == 0:
                logger.info("Code validation passed.")
                return True
            else:
                logger.error(f"Code validation failed: {result.stdout}\n{result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error validating code: {e}", exc_info=True)
            return False

class DocstringTransformer(ast.NodeTransformer):
    """Transformer for inserting docstrings into AST nodes."""

    def __init__(self, documentation: Dict[str, Any], docstring_format: str, preserve_existing=False):
        self.documentation = documentation
        self.docstring_format = docstring_format
        self.preserve_existing = preserve_existing

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Adds or updates docstring to function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(func["docstring"], self.docstring_format, func.get("args", []), func.get("returns"))
                    node.docstring = docstring
                break
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Adds or updates docstring to async function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(func["docstring"], self.docstring_format, func.get("args", []), func.get("returns"))
                    node.docstring = docstring
                break
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Adds or updates docstring to class definitions."""
        for cls in self.documentation.get("classes", []):
            if cls["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(cls["docstring"], self.docstring_format)
                    node.docstring = docstring
                break
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Adds or updates docstring to the module."""
        if not self.preserve_existing or not ast.get_docstring(node):
            docstring = self._format_docstring(self.documentation.get("summary", ""), self.docstring_format)
            node.docstring = docstring
        return node

    def _format_docstring(self, docstring: str, format: str = "Google", args: List[Dict] = None, returns: Optional[str] = None) -> str:
        """Formats the docstring according to the specified format."""
        if format == "Google":
            formatted_docstring = docstring.strip() + "\n\n"
            if args:
                formatted_docstring += "Args:\n"
                for arg in args:
                    formatted_docstring += f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}\n"
            if returns:
                formatted_docstring += f"\nReturns:\n    {returns}\n"
            return formatted_docstring
        elif format == "NumPy":
            formatted_docstring = docstring.strip() + "\n\n"
            if args:
                formatted_docstring += "Parameters\n----------\n"
                for arg in args:
                    formatted_docstring += f"{arg['name']} : {arg.get('type', 'Any')}\n    {arg.get('description', '')}\n"
            if returns:
                formatted_docstring += "\nReturns\n-------\n"
                formatted_docstring += f"{returns}\n"
            return formatted_docstring
        # ... (Handle other formats like reStructuredText)
        return docstring

```

## azure_model.py

```python
# azure_model.py

"""
azure_model.py

Handles interaction with the Azure OpenAI API, including token counting,
documentation generation, and API request logic.
"""

import aiohttp
import logging
import json
from typing import List, Dict, Any
from utils import TokenManager  # Import TokenManager
from token_utils import TokenManager

logger = logging.getLogger(__name__)

class AzureModel:
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version

    async def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> Dict[str, Any]:
        """Fetches documentation from the Azure OpenAI API."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Azure OpenAI documentation generated successfully.")
                    choice = data.get("choices", [{}])[0]
                    return choice.get("message", {}).get("content", {})
                else:
                    logger.error(f"Error generating documentation from Azure: {response.status}")
                    return {}

    def calculate_tokens(self, base_info: str, context: str, chunk_content: str, schema: str) -> int:
        """
        Calculates token count for Azure model prompts using TokenManager.

        Args:
            base_info: Project and style information
            context: Related code/documentation
            chunk_content: Content of the chunk being documented
            schema: JSON schema

        Returns:
            Total token count
        """
        total = 0
        for text in [base_info, context, chunk_content, schema]:
            token_result = TokenManager.count_tokens(text)
            total += token_result.token_count
        return total

    def generate_prompt(self, base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates prompt structure specifically for the Azure model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for Azure model.")
        return prompt

```

## language_functions/js_ts_handler.py

```python
# js_ts_handler.py

import os
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from jsonschema import validate, ValidationError

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JSDocStyle(Enum):
    JSDOC = "jsdoc"
    TSDOC = "tsdoc"

@dataclass
class MetricsResult:
    complexity: int
    maintainability: float
    halstead: Dict[str, float]
    function_metrics: Dict[str, Dict[str, Any]]

class JSTsHandler(BaseHandler):

    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    async def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts the structure of the JavaScript/TypeScript code.

        Checklist:
        - [x] Parsing: Uses external js_ts_parser.js script.
        - [x] Data Structure: Conforms to function_schema.json.
        - [x] Schema Validation: Implemented using jsonschema.validate.
        - [x] Metrics Calculation: Uses external js_ts_metrics.js script.
        - [x] Language-Specific Features: Extracts React components.
        """
        logger.info(f"Extracting structure for file: {file_path}")
        try:
            is_typescript = self._is_typescript_file(file_path)
            parser_options = self._get_parser_options(is_typescript)
            input_data = {
                "code": code,
                "language": "typescript" if is_typescript else "javascript",
                "filePath": file_path or "unknown",
                "options": parser_options
            }

            # Get metrics
            metrics_result = self._calculate_metrics(code, is_typescript)
            if metrics_result is None:
                return self._get_empty_structure("Metrics calculation failed")

            # Run parser script
            parsed_data = self._run_parser_script(input_data)
            if parsed_data is None:
                return self._get_empty_structure("Parsing failed")

            # Map parsed data to function_schema.json structure
            structured_data = {
                "docstring_format": "JSDoc" if not is_typescript else "TSDoc",
                "summary": parsed_data.get("summary", ""),
                "changes_made": [],  # Placeholder for changelog
                "functions": self._map_functions(parsed_data.get("functions", []), metrics_result.function_metrics),
                "classes": self._map_classes(parsed_data.get("classes", []), metrics_result.function_metrics),
                "variables": parsed_data.get("variables", []),
                "constants": parsed_data.get("constants", []),
                "imports": parsed_data.get("imports", []),
                "metrics": {
                    "complexity": metrics_result.complexity,
                    "halstead": metrics_result.halstead,
                    "maintainability_index": metrics_result.maintainability,
                }
            }

            # React analysis
            if self._is_react_file(file_path):
                react_info = self._analyze_react_components(code, is_typescript)
                if react_info is not None:
                    structured_data["react_components"] = react_info

            # Schema validation
            try:
                validate(instance=structured_data, schema=self.function_schema)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e}")

            return structured_data

        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def _map_functions(self, functions: List[Dict], function_metrics: Dict) -> List[Dict]:
        """Maps function data to the schema."""
        mapped_functions = []
        for func in functions:
            func_name = func.get("name", "")
            metrics = function_metrics.get(func_name, {})
            mapped_functions.append({
                "name": func_name,
                "docstring": func.get("docstring", ""),
                "args": func.get("params", []),
                "async": func.get("async", False),
                "returns": {"type": func.get("returnType", ""), "description": ""},  # Map return type
                "complexity": metrics.get("complexity", 0),
                "halstead": metrics.get("halstead", {})
            })
        return mapped_functions

    def _map_classes(self, classes: List[Dict], function_metrics: Dict) -> List[Dict]:
        """Maps class data to the schema."""
        mapped_classes = []
        for cls in classes:
            mapped_methods = self._map_functions(cls.get("methods", []), function_metrics)
            mapped_classes.append({
                "name": cls.get("name", ""),
                "docstring": cls.get("docstring", ""),
                "methods": mapped_methods
            })
        return mapped_classes

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts JSDoc/TSDoc comments into JavaScript/TypeScript code.

        Checklist:
        - [x] Docstring Generation: Generates JSDoc/TSDoc style comments.
        - [x] Docstring Formats: Handles JSDoc and TSDoc based on file type.
        - [x] Insertion Method: Uses external js_ts_inserter.js script.
        - [x] Error Handling: Includes error handling and logging.
        - [x] Preservation of Existing Docstrings: Controlled by script options.
        """
        logger.info("Inserting docstrings...")
        try:
            is_typescript = self._is_typescript_file(documentation.get("file_path"))
            doc_style = JSDocStyle.TSDOC if is_typescript else JSDocStyle.JSDOC

            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "typescript" if is_typescript else "javascript",
                "options": {
                    "style": doc_style.value,
                    "includeTypes": is_typescript,
                    "preserveExisting": True  # Or False, depending on your requirement
                }
            }

            updated_code = self._run_inserter_script(input_data)
            return updated_code if updated_code is not None else code

        except Exception as e:
            logger.error(f"Error inserting documentation: {str(e)}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates JavaScript/TypeScript code using ESLint.

        Checklist:
        - [x] Validation Tool: Uses ESLint.
        - [x] Error Handling: Handles validation errors.
        - [x] Temporary Files: Uses and cleans up temporary files.
        """
        logger.info("Validating code...")
        try:
            if not file_path:
                logger.warning("File path not provided for validation")
                return True

            is_typescript = self._is_typescript_file(file_path)
            config_path = self._get_eslint_config(is_typescript)

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ts' if is_typescript else '.js',
                encoding='utf-8',
                delete=False
            ) as tmp:
                tmp.write(code)
                temp_path = tmp.name

            try:
                result = subprocess.run(
                    ["eslint", "--config", config_path, temp_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.debug("ESLint validation passed.")
                else:
                    logger.error(f"ESLint validation failed: {result.stdout}\n{result.stderr}")
                return result.returncode == 0
            finally:
                try:
                    os.unlink(temp_path)
                except OSError as e:
                    logger.error(f"Error deleting temporary file {temp_path}: {e}")

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False

    def _calculate_metrics(self, code: str, is_typescript: bool) -> Optional[MetricsResult]:
        """Calculates code metrics using the js_ts_metrics.js script."""
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "sourceType": "module",
                    "loc": True,
                    "cyclomatic": True,
                    "halstead": True,
                    "maintainability": True
                }
            }
            result = self._run_script(
                script_name="js_ts_metrics.js",
                input_data=input_data,
                error_message="Metrics calculation failed"
            )
            logger.debug(f"Metrics calculation result: {result}")

            if result is None:
                logger.error("Metrics calculation returned None.")
                return None

            if not isinstance(result, dict):
                logger.error(f"Metrics result is not a dictionary: {type(result)}")
                return None

            required_keys = ["complexity", "maintainability", "halstead", "functions"]
            if not all(key in result for key in required_keys):
                missing_keys = [key for key in required_keys if key not in result]
                logger.error(f"Metrics result is missing keys: {missing_keys}")
                return None

            if not isinstance(result["halstead"], dict):
                logger.error("Halstead metrics should be a dictionary.")
                return None

            if not isinstance(result["functions"], dict):
                logger.error("Function metrics should be a dictionary.")
                return None

            return MetricsResult(
                complexity=result.get("complexity", 0),
                maintainability=result.get("maintainability", 0.0),
                halstead=result.get("halstead", {}),
                function_metrics=result.get("functions", {})
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return None

    def _analyze_react_components(self, code: str, is_typescript: bool) -> Optional[Dict[str, Any]]:
        """Analyzes React components using the react_analyzer.js script."""
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "plugins": ["jsx", "react"]
                }
            }
            result = self._run_script(
                script_name="react_analyzer.js",
                input_data=input_data,
                error_message="React analysis failed"
            )
            logger.debug(f"React analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing React components: {str(e)}", exc_info=True)
            return None

    def _get_parser_options(self, is_typescript: bool) -> Dict[str, Any]:
        """Returns parser options for the js_ts_parser.js script."""
        options = {
            "sourceType": "module",
            "plugins": [
                "jsx",
                "decorators-legacy",
                ["decorators", {"decoratorsBeforeExport": True}],
                "classProperties",
                "classPrivateProperties",
                "classPrivateMethods",
                "exportDefaultFrom",
                "exportNamespaceFrom",
                "dynamicImport",
                "nullishCoalescingOperator",
                "optionalChaining",
            ]
        }

        if is_typescript:
            options["plugins"].extend([
                "typescript"
            ])

        return options

    def _is_typescript_file(self, file_path: Optional[str]) -> bool:
        """Checks if a file is a TypeScript file."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.ts', '.tsx'))

    def _is_react_file(self, file_path: Optional[str]) -> bool:
        """Checks if a file is a React file (JSX or TSX)."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.jsx', '.tsx'))

    def _get_eslint_config(self, is_typescript: bool) -> str:
        """Returns the path to the appropriate ESLint config file."""
        config_name = '.eslintrc.typescript.json' if is_typescript else '.eslintrc.json'
        return os.path.join(self.script_dir, config_name)

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        """Returns an empty structure dictionary with a reason."""
        return {
            "classes": [],
            "functions": [],
            "variables": [],
            "constants": [],
            "imports": [],
            "exports": [],
            "react_components": [],
            "summary": f"Empty structure: {reason}" if reason else "Empty structure",
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "complexity": 0,
            "maintainability_index": 0,
            "function_metrics": {}
        }

    def _run_parser_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs the js_ts_parser.js script and returns the parsed data."""
        return self._run_script(
            script_name="js_ts_parser.js",
            input_data=input_data,
            error_message="Parsing failed"
        )

    def _run_inserter_script(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Runs the js_ts_inserter.js script and returns the updated code."""
        result = self._run_script(
            script_name="js_ts_inserter.js",
            input_data=input_data,
            error_message="Error running inserter"
        )
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("code")
        else:
            logger.error("Inserter script did not return code string.")
            return None

    def _run_script(self, script_name: str, input_data: Dict[str, Any], error_message: str) -> Any:
        """
        Runs a Node.js script with improved error handling and encoding management.
        """
        try:
            script_path = os.path.join(self.script_dir, script_name)
            if not os.path.isfile(script_path):
                logger.error(f"Script not found: {script_path}")
                return None

            logger.debug(f"Running script: {script_path} with input data: {input_data}")

            # Convert input data to JSON string with proper encoding handling
            try:
                input_json = json.dumps(input_data, ensure_ascii=False)
                input_bytes = input_json.encode('utf-8', errors='surrogateescape')
            except UnicodeEncodeError as e:
                logger.error(f"Unicode encoding error in input data: {e}", exc_info=True)
                return None

            process = subprocess.run(
                ['node', script_path],
                input=input_json,  # Pass the JSON string
                capture_output=True,
                text=True, 
                check=True,
                timeout=60
            )

            if process.returncode != 0:
                logger.error(f"{error_message}: {process.stderr}")
                return None

            output = process.stdout.strip()
            logger.debug(f"Script Output ({script_name}): {output}")

            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                if script_name == "js_ts_inserter.js":
                    # If inserter script returns plain code, not JSON
                    return output
                logger.error(f"{error_message}: Invalid JSON output. Error: {e}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: Process error: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"{error_message}: Script timed out after 60 seconds")
            return None
        except Exception as e:
            logger.error(f"{error_message}: Unexpected error: {e}", exc_info=True)
            return None

```

## language_functions/language_functions.py

```python
"""
language_functions.py

This module provides utility functions for handling different programming languages within the documentation generation pipeline.
It includes functions to retrieve the appropriate language handler and to insert docstrings/comments into source code based on AI-generated documentation.

Functions:
    - get_handler(language, function_schema): Retrieves the appropriate handler for a given programming language.
    - insert_docstrings(original_code, documentation, language, schema_path): Inserts docstrings/comments into the source code using the specified language handler.
"""

import json
import logging
import subprocess
from typing import Dict, Any, Optional

from .base_handler import BaseHandler
from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from utils import load_function_schema  # Import for dynamic schema loading

logger = logging.getLogger(__name__)


def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    This function matches the provided programming language with its corresponding handler class.
    If the language is supported, it returns an instance of the handler initialized with the given function schema.
    If the language is unsupported, it logs a warning and returns None.

    Args:
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    language = language.lower()
    if language == "python":
        return PythonHandler(function_schema)
    elif language == "java":
        return JavaHandler(function_schema)
    elif language in ["javascript", "js", "typescript", "ts"]:
        return JSTsHandler(function_schema)
    elif language == "go":
        return GoHandler(function_schema)
    elif language in ["cpp", "c++", "cxx"]:
        return CppHandler(function_schema)
    elif language in ["html", "htm"]:
        return HTMLHandler(function_schema)
    elif language == "css":
        return CSSHandler(function_schema)
    else:
        logger.warning(f"No handler available for language: {language}")
        return None


def insert_docstrings(
    original_code: str, 
    documentation: Dict[str, Any], 
    language: str, 
    schema_path: str  # schema_path is now required
) -> str:
    """
    Inserts docstrings/comments into code based on the specified programming language.

    This function dynamically loads the function schema from a JSON file, retrieves the appropriate
    language handler, and uses it to insert documentation comments into the original source code.
    If any errors occur during schema loading or docstring insertion, the original code is returned.

    Args:
        original_code (str): The original source code to be documented.
        documentation (Dict[str, Any]): Documentation details obtained from AI, typically including descriptions of functions, classes, and methods.
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        schema_path (str): Path to the function schema JSON file, which defines the structure and expected documentation format.

    Returns:
        str: The source code with inserted documentation comments, or the original code if errors occur.
    """
    logger.debug(f"Processing docstrings for language: {language}")

    try:
        # Load the function schema from the provided schema path
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading function schema: {e}")
        return original_code  # Return original code on schema loading error
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during schema loading: {e}", exc_info=True)
        return original_code

    # Retrieve the appropriate handler for the specified language
    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    try:
        # Use the handler to insert docstrings/comments into the original code
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code  # Return original code on docstring insertion error

```

## language_functions/__init__.py

```python
"""
language_functions Package

This package provides language-specific handlers for extracting code structures,
inserting documentation comments (docstrings), and validating code across various
programming languages. It includes handlers for languages such as Python, Java,
JavaScript/TypeScript, Go, C++, HTML, and CSS.

Modules:
    - python_handler.py: Handler for Python code.
    - java_handler.py: Handler for Java code.
    - js_ts_handler.py: Handler for JavaScript and TypeScript code.
    - go_handler.py: Handler for Go code.
    - cpp_handler.py: Handler for C++ code.
    - html_handler.py: Handler for HTML code.
    - css_handler.py: Handler for CSS code.
    - base_handler.py: Abstract base class defining the interface for all handlers.

Functions:
    - get_handler(language, function_schema): Factory function to retrieve the appropriate language handler.

Example:
    ```python
    from language_functions import get_handler
    from utils import load_function_schema

    function_schema = load_function_schema('path/to/schema.json')
    handler = get_handler('python', function_schema)
    if handler:
        updated_code = handler.insert_docstrings(original_code, documentation)
    ```
"""

import logging
from typing import Dict, Any, Optional

from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from .base_handler import BaseHandler
from .language_functions import insert_docstrings  # Import the function
from metrics import MetricsAnalyzer  # Import MetricsAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["get_handler", "insert_docstrings"]

def get_handler(language: str, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    Args:
        language (str): The programming language of the source code.
        function_schema (Dict[str, Any]): The schema defining functions.
        metrics_analyzer (MetricsAnalyzer): The metrics analyzer object.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    # Normalize the language string to lowercase to ensure case-insensitive matching
    language = language.lower()
    
    # Map of supported languages to their handlers
    handlers = {
        "python": PythonHandler,
        "java": JavaHandler,
        "javascript": JSTsHandler,
        "js": JSTsHandler,
        "typescript": JSTsHandler,
        "ts": JSTsHandler,
        "go": GoHandler,
        "cpp": CppHandler,
        "c++": CppHandler,
        "cxx": CppHandler,
        "html": HTMLHandler,
        "htm": HTMLHandler,
        "css": CSSHandler
    }

    handler_class = handlers.get(language)
    if handler_class:
        return handler_class(function_schema, metrics_analyzer)
    else:
        logger.debug(f"No handler available for language: {language}")
        return None  # Return None instead of raising an exception
```

# Directory: schemas

## schemas/function_schema.json

```json
{
  "functions": [
    {
      "name": "generate_documentation",
      "description": "Generates documentation for code structures.",
      "parameters": {
        "type": "object",
        "properties": {
          "docstring_format": {
            "type": "string",
            "description": "Format of the docstring (e.g., Google, JSDoc, TSDoc, NumPy, reST).",
            "enum": ["Google", "JSDoc", "TSDoc", "NumPy", "reST"]
          },
          "summary": {
            "type": "string",
            "description": "A detailed summary of the file."
          },
          "changes_made": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of changes made to the file."
          },
          "functions": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "docstring": { "type": "string" },
                "args": {
                  "type": "array",
                  "items": { "type": "string" }
                },
                "async": { "type": "boolean" },
                "complexity": { "type": "integer", "description": "Cyclomatic complexity of the function." },
                "halstead": {
                  "type": "object",
                  "properties": {
                    "volume": { "type": "number" },
                    "difficulty": { "type": "number" },
                    "effort": { "type": "number" }
                  },
                  "description": "Halstead complexity metrics for the function.",
                  "required": []
                }
              },
              "required": ["name", "docstring", "args", "async"]
            },
            "description": "List of functions."
          },
          "classes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "docstring": { "type": "string" },
                "methods": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "name": { "type": "string" },
                      "docstring": { "type": "string" },
                      "args": {
                        "type": "array",
                        "items": { "type": "string" }
                      },
                      "async": { "type": "boolean" },
                      "type": { "type": "string", "description": "Return type or purpose of the method." },
                      "complexity": { "type": "integer", "description": "Cyclomatic complexity of the method." },
                      "halstead": {
                        "type": "object",
                        "properties": {
                          "volume": { "type": "number" },
                          "difficulty": { "type": "number" },
                          "effort": { "type": "number" }
                        },
                        "description": "Halstead complexity metrics for the method.",
                        "required": []
                      }
                    },
                    "required": ["name", "docstring", "args", "async", "type"]
                  }
                }
              },
              "required": ["name", "docstring", "methods"]
            }
          },
          "halstead": {
            "type": "object",
            "properties": {
              "volume": { "type": "number" },
              "difficulty": { "type": "number" },
              "effort": { "type": "number" }
            },
            "description": "Overall Halstead complexity metrics for the file.",
            "required": []
          },
          "maintainability_index": {
            "type": "number",
            "description": "Maintainability index of the code."
          },
          "variables": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "type": { "type": "string" },
                "description": { "type": "string" }
              },
              "required": ["name", "type", "description"]
            }
          },
          "constants": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "type": { "type": "string" },
                "description": { "type": "string" }
              },
              "required": ["name", "type", "description"]
            }
          }
        },
        "required": ["docstring_format", "summary", "functions", "classes"]
      }
    }
  ]
}

```



