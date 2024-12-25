---
epoch: 1731234710787
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [core_-_types.py](core_-_types.py.md)*[analysis_-_code.py](analysis_-_code.py.md)*[analysis_-_security.py](analysis_-_security.py.md)_[[utils@20241110_043150

**user**: [core - types.py](core%20-%20types.py.md) [analysis - code.py](analysis%20-%20code.py.md) [analysis - security.py](analysis%20-%20security.py.md) [utils - logging.py](utils%20-%20logging.py.md) [Context Retrieval Layer](Context%20Retrieval%20Layer.md) [Pattern Learning Implementation](Pattern%20Learning%20Implementation.md)[UnifiedRelationshipGraph Implementation](UnifiedRelationshipGraph%20Implementation.md) [Adaptive Memory Manager Implementation](Adaptive%20Memory%20Manager%20Implementation.md) [PredictiveContextLoader Implementation](PredictiveContextLoader%20Implementation.md) [Advanced Context Management](Advanced%20Context%20Management.md) [SharedContextPool Implementation](SharedContextPool%20Implementation.md) [Semantic Code Relationships](Semantic%20Code%20Relationships.md) [DynamicTokenBudget Implementation](DynamicTokenBudget%20Implementation.md) [Final Integration Layer](Final%20Integration%20Layer.md) [DocuScribe Storage and Language Model Integration](DocuScribe%20Storage%20and%20Language%20Model%20Integration.md) [DocuScribe Learning and Analysis Systems](DocuScribe%20Learning%20and%20Analysis%20Systems.md)  
[Timestamp: 2024/11/10 04:31:50]

**ai**: Based on the provided context, I'll help you generate high-quality code solutions. I understand that you want me to act as an adaptive code generation assistant in a RAG system, focusing on maintainable, efficient, and modular code.

What specific code would you like me to help you generate? I can assist with:

1. Extending existing components (e.g., adding features to the ContextRetrieval, PatternLearning, or SecurityAnalyzer classes)
2. Creating new components that integrate with the existing system
3. Implementing specific functionality within the DocuScribe framework
4. Writing test cases for any of the existing components

Please let me know what type of code you'd like to work on, and I'll help you create a solution that follows best practices and integrates well with the existing codebase.

Some key capabilities I can help with based on the context:

- Code analysis and pattern detection
- Security vulnerability scanning
- Context management and retrieval
- Token budget optimization
- Vector storage and embedding
- Language model integration
- Performance monitoring and optimization

Let me know what specific area you'd like to focus on!  
[Timestamp: 2024/11/10 04:32:00]

**user**: docuscribe/  
├── core/  
│ ├── **init**.py  
│ ├── integration.py <- "Final Integration Layer.md"  
│ ├── types.py <- "core/types.py"  
│ └── utils.py <- "core/utils.py"  
│  
├── context/  
│ ├── pool.py <- "SharedContextPool Implementation.txt"  
│ ├── relationship.py <- "UnifiedRelationshipGraph Implementation.txt"  
│ └── tokens.py <- "DynamicTokenBudget Implementation.txt"  
│  
├── memory/  
│ ├── manager.py <- "AdaptiveMemoryManager Implementation.txt"  
│ ├── prediction.py <- "PredictiveContextLoader Implementation.txt"  
│ └── compression.py <- "Context Compressor Implementation.txt"  
│  
├── learning/  
│ ├── core.py <- "learning - core.py"  
│ ├── patterns.py <- "Pattern Learning Implementation.txt"  
│ └── strategies.py <- "EmbeddingFeatureExtractor, ReinforcementLearningStrategy..."  
│  
├── analysis/  
│ ├── code.py <- "analysis - code.py"  
│ ├── semantic.py <- "Semantic Code Relationships.md"  
│ └── security.py <- analysis - security.py  
│  
├── integrations/  
│ ├── git/ <- (No direct documentation)  
│ ├── ide/ <- (No direct documentation)  
│ └── api/ <- "DocuScribe Storage and LLM Integration.md"  
│  
└── utils/  
    ├── monitoring.py <- "Analysis Metrics and Monitoring Implementation"  
    └── logging.py <- "utils - logging.py"  
[Timestamp: 2024/11/10 04:33:34]

**ai**: I'll help you create the missing core utility module (`core/utils.py`) that will provide essential functionality used across the DocuScribe system. Based on the existing codebase and architecture, here's a comprehensive implementation:

```python
# docuscribe/core/utils.py

import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DocuScribeError(Exception):
    """Base exception class for DocuScribe errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()

class ValidationError(DocuScribeError):
    """Raised when validation fails"""
    pass

class OperationError(DocuScribeError):
    """Raised when an operation fails"""
    pass

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential: bool = True
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Whether to use exponential backoff
        
    Returns:
        Result of the function call
        
    Raises:
        OperationError: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break
                
            delay = min(
                max_delay,
                base_delay * (2 ** attempt if exponential else 1)
            )
            
            logger.warning(
                f"Retry attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    raise OperationError(
        f"Operation failed after {max_retries} attempts",
        details={"last_error": str(last_error)}
    )

def async_timed():
    """Decorator to measure async function execution time"""
    def wrapper(func: Callable):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                logger.debug(
                    f"Function '{func.__name__}' took {elapsed:.2f} seconds"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"Function '{func.__name__}' failed after {elapsed:.2f} seconds: {e}"
                )
                raise
        return wrapped
    return wrapper

class AsyncCache:
    """Thread-safe async cache with TTL support"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        async with self._lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            if (datetime.now() - entry['timestamp']).total_seconds() > self.ttl_seconds:
                del self.cache[key]
                return None
                
            return entry['value']
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        async with self._lock:
            self.cache[key] = {
                'value': value,
                'timestamp': datetime.now()
            }
    
    async def clear(self) -> None:
        """Clear all cached entries"""
        async with self._lock:
            self.cache.clear()

class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    async def read_file(path: Union[str, Path]) -> str:
        """Read file content asynchronously"""
        path = Path(path)
        try:
            with ThreadPoolExecutor() as pool:
                content = await asyncio.get_event_loop().run_in_executor(
                    pool,
                    path.read_text
                )
            return content
        except Exception as e:
            raise OperationError(f"Failed to read file: {e}")
    
    @staticmethod
    async def write_file(
        path: Union[str, Path],
        content: str,
        create_dirs: bool = True
    ) -> None:
        """Write content to file asynchronously"""
        path = Path(path)
        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            with ThreadPoolExecutor() as pool:
                await asyncio.get_event_loop().run_in_executor(
                    pool,
                    path.write_text,
                    content
                )
        except Exception as e:
            raise OperationError(f"Failed to write file: {e}")

class HashUtils:
    """Utilities for hashing operations"""
    
    @staticmethod
    def hash_content(content: Union[str, bytes]) -> str:
        """Generate SHA-256 hash of content"""
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Generate deterministic hash of dictionary"""
        serialized = json.dumps(data, sort_keys=True)
        return HashUtils.hash_content(serialized)

class ValidationUtils:
    """Utilities for validation operations"""
    
    @staticmethod
    def validate_token_count(text: str, max_tokens: int) -> bool:
        """Validate token count is within limit"""
        # Simple approximation: words + punctuation
        token_count = len(text.split()) + len([c for c in text if c in '.,!?;:'])
        return token_count <= max_tokens
    
    @staticmethod
    def validate_file_path(path: Union[str, Path]) -> bool:
        """Validate file path is safe and within project"""
        path = Path(path)
        try:
            # Resolve to absolute path
            resolved = path.resolve()
            # Check if path is within project directory
            return resolved.is_relative_to(Path.cwd())
        except Exception:
            return False

class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def record_metric(self, name: str, value: float) -> None:
        """Record a metric value"""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    async def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        async with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
                
            values = self.metrics[name]
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                '
[Timestamp: 2024/11/10 04:34:11]

**user**: Merge with this one: """
# DocuScribe Utilities
Core utility functions and helper classes for DocuScribe.

## Overview
This module provides common utilities used across DocuScribe components, including:
- File operations
- Text processing
- Error handling
- Logging setup
- Performance monitoring
- Cache management

## Key Features
- Async file operations
- Text normalization
- Code parsing helpers
- Error tracking
- Performance metrics
"""

import ast
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
import time
from functools import wraps
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)

# File Operations
async def read_file_async(path: Path) -> str:
    """Read file content asynchronously."""
    try:
        return await asyncio.to_thread(path.read_text)
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise

async def write_file_async(path: Path, content: str) -> None:
    """Write file content asynchronously."""
    try:
        await asyncio.to_thread(path.write_text, content)
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        raise

def find_python_files(directory: Path) -> Generator[Path, None, None]:
    """Find all Python files in directory recursively."""
    for path in directory.rglob("*.py"):
        if path.is_file():
            yield path

# Code Processing
def normalize_code(code: str) -> str:
    """Normalize code formatting."""
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except Exception:
        return code

def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from AST node."""
    try:
        return ast.get_docstring(node)
    except Exception:
        return None

def calculate_code_hash(code: str) -> str:
    """Calculate hash of code content."""
    return hashlib.sha256(code.encode()).hexdigest()

# Performance Monitoring
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    async def __aenter__(self) -> 'Timer':
        self.start_time = time.monotonic()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.monotonic() - self.start_time
        logger.debug(f"{self.operation_name} took {duration:.3f} seconds")
        
        # Record metric
        await record_metric(
            metric_name=f"operation_duration_{self.operation_name}",
            value=duration
        )

class Cache:
    """Simple cache with size limit and TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key not in self.cache:
                return None
                
            timestamp = self.timestamps[key]
            if time.monotonic() - timestamp > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
                
            return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with cleanup if needed."""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                await self._cleanup()
                
            self.cache[key] = value
            self.timestamps[key] = time.monotonic()
    
    async def _cleanup(self) -> None:
        """Remove oldest entries when cache is full."""
        current_time = time.monotonic()
        expired = [
            k for k, t in self.timestamps.items()
            if current_time - t > self.ttl_seconds
        ]
        
        for key in expired:
            del self.cache[key]
            del self.timestamps[key]
            
        if len(self.cache) >= self.max_size:
            # Remove 20% oldest entries
            sorted_keys = sorted(
                self.timestamps.items(),
                key=lambda x: x[1]
            )[:len(self.cache) // 5]
            
            for key, _ in sorted_keys:
                del self.cache[key]
                del self.timestamps[key]

# Error Handling
async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """Retry an async function with exponential backoff."""
    retries = 0
    current_delay = delay
    
    while True:
        try:
            return await func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise
                
            logger.warning(
                f"Attempt {retries} failed: {str(e)}. "
                f"Retrying in {current_delay} seconds."
            )
            
            await asyncio.sleep(current_delay)
            current_delay *= backoff

# Metrics Collection
async def record_metric(metric_name: str, value: float) -> None:
    """Record a metric value."""
    try:
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        # Format metric data
        metric_data = {
            'name': metric_name,
            'value': value,
            'timestamp': timestamp
        }
        
        # Append to metrics file
        async with aiofiles.open('metrics.jsonl', 'a') as f:
            await f.write(json.dumps(metric_data) + '\n')
            
    except Exception as e:
        logger.error(f"Error recording metric: {e}")

# Validation Utilities
def validate_python_code(code: str) -> bool:
    """Validate Python code syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def validate_file_path(path: Path) -> bool:
    """Validate file path exists and is readable."""
    return path.exists() and path.is_file() and os.access(path, os.R_OK)

# Setup Utilities
def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration."""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default'],
                'level': level,
                'propagate': True
            }
        }
    }
    
    if log_file:
        config['handlers']['file'] = {
            'level': level,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': log_file,
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(config)
[Timestamp: 2024/11/10 04:34:18]

**ai**: I'll merge the two utility implementations and create a comprehensive `core/utils.py` that combines the best features of both while maintaining consistency with the existing type system. Here's the merged implementation:

```python
"""
DocuScribe Utilities
Core utility functions and helper classes for DocuScribe.

Features:
- Async file operations with retry logic
- Code processing and validation
- Performance monitoring and metrics
- Cache management with TTL
- Error handling with backoff
- Logging configuration
"""

import ast
import asyncio
import functools
import hashlib
import json
import logging
import logging.config
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from .types import CodeEntity, EntityType, Location, PathLike

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Exception Classes
class DocuScribeError(Exception):
    """Base exception class for DocuScribe errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()

class ValidationError(DocuScribeError):
    """Raised when validation fails"""
    pass

class OperationError(DocuScribeError):
    """Raised when an operation fails"""
    pass

# Async Utilities
class Timer:
    """Context manager for timing async operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    async def __aenter__(self) -> 'Timer':
        self.start_time = time.monotonic()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.monotonic() - self.start_time
        logger.debug(f"{self.operation_name} took {duration:.3f} seconds")
        await record_metric(f"operation_duration_{self.operation_name}", duration)

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential: bool = True
) -> T:
    """Retry an async function with exponential backoff"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break
                
            delay = min(
                max_delay,
                base_delay * (2 ** attempt if exponential else 1)
            )
            
            logger.warning(
                f"Retry attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    raise OperationError(
        f"Operation failed after {max_retries} attempts",
        details={"last_error": str(last_error)}
    )

# File Operations
class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    async def read_file(path: PathLike) -> str:
        """Read file content asynchronously"""
        path = Path(path)
        try:
            with ThreadPoolExecutor() as pool:
                content = await asyncio.get_event_loop().run_in_executor(
                    pool, path.read_text
                )
            return content
        except Exception as e:
            raise OperationError(f"Failed to read file: {e}")
    
    @staticmethod
    async def write_file(
        path: PathLike,
        content: str,
        create_dirs: bool = True
    ) -> None:
        """Write content to file asynchronously"""
        path = Path(path)
        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            with ThreadPoolExecutor() as pool:
                await asyncio.get_event_loop().run_in_executor(
                    pool, path.write_text, content
                )
        except Exception as e:
            raise OperationError(f"Failed to write file: {e}")
    
    @staticmethod
    def find_python_files(directory: Path) -> Generator[Path, None, None]:
        """Find all Python files in directory recursively"""
        for path in directory.rglob("*.py"):
            if path.is_file():
                yield path

# Code Processing
class CodeUtils:
    """Utilities for code processing"""
    
    @staticmethod
    def normalize_code(code: str) -> str:
        """Normalize code formatting"""
        try:
            tree = ast.parse(code)
            return ast.unparse(tree)
        except Exception:
            return code
    
    @staticmethod
    def extract_docstring(node: ast.AST) -> Optional[str]:
        """Extract docstring from AST node"""
        try:
            return ast.get_docstring(node)
        except Exception:
            return None
    
    @staticmethod
    def validate_python_code(code: str) -> bool:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

# Cache Management
class AsyncCache:
    """Thread-safe async cache with TTL and size limit"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        async with self._lock:
            if key not in self.cache:
                return None
                
            timestamp = self.timestamps[key]
            if time.monotonic() - timestamp > self.ttl_seconds:
                await self._remove_entry(key)
                return None
                
            return self.cache[key]['value']
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with cleanup if needed"""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                await self._cleanup()
                
            self.cache[key] = {'value': value}
            self.timestamps[key] = time.monotonic()
    
    async def _cleanup(self) -> None:
        """Remove expired and oldest entries"""
        current_time = time.monotonic()
        
        # Remove expired entries
        expired = [
            k for k, t in self.timestamps.items()
            if current_time - t > self.ttl_seconds
        ]
        for key in expired:
            await self._remove_entry(key)
        
        # Remove oldest entries if still too large
        if len(self.cache) >= self.max_size:
            sorted_keys = sorted(
                self.timestamps.items(),
                key=lambda x: x[1]
            )[:len(self.cache) // 5]  # Remove oldest 20%
            
            for key, _ in sorted_keys:
                await self._remove_entry(key)
    
    async def _remove_entry(self, key: str) -> None:
        """Remove a cache entry"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

# Metrics Collection
class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def record_metric(self, name: str, value: float) -> None:
        """Record a metric
[Timestamp: 2024/11/10 04:34:55]

**user**: Continue
[Timestamp: 2024/11/10 04:35:00]

**ai**: Here's the continuation of the `core/utils.py` file:

```python
# Metrics Collection (continued)
    async def record_metric(self, name: str, value: float) -> None:
        """Record a metric value"""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    async def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        async with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
                
            values = self.metrics[name]
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }
    
    async def save_metrics(self, file_path: PathLike) -> None:
        """Save metrics to file"""
        async with self._lock:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    name: await self.get_statistics(name)
                    for name in self.metrics
                }
            }
            
            await FileUtils.write_file(
                file_path,
                json.dumps(metrics_data, indent=2)
            )

# Global metrics collector instance
metrics_collector = MetricsCollector()

async def record_metric(metric_name: str, value: float) -> None:
    """Global function to record metrics"""
    await metrics_collector.record_metric(metric_name, value)

# Validation Utilities
class ValidationUtils:
    """Utilities for validation operations"""
    
    @staticmethod
    def validate_token_count(text: str, max_tokens: int) -> bool:
        """Validate token count is within limit"""
        # Simple approximation: words + punctuation
        token_count = len(text.split()) + len([c for c in text if c in '.,!?;:'])
        return token_count <= max_tokens
    
    @staticmethod
    def validate_file_path(path: PathLike) -> bool:
        """Validate file path is safe and within project"""
        path = Path(path)
        try:
            resolved = path.resolve()
            return (
                resolved.exists() and
                resolved.is_file() and
                os.access(resolved, os.R_OK) and
                resolved.is_relative_to(Path.cwd())
            )
        except Exception:
            return False
    
    @staticmethod
    def validate_code_entity(entity: CodeEntity) -> bool:
        """Extended validation for code entities"""
        if not bool(entity.name) or not isinstance(entity.entity_type, EntityType):
            return False
            
        location = entity.location
        if not isinstance(location, Location):
            return False
            
        return (
            location.line_start >= 0 and
            location.line_end >= location.line_start and
            (location.column_start is None or location.column_start >= 0) and
            (location.column_end is None or location.column_end >= location.column_start)
        )

# Logging Setup
def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Setup logging configuration"""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': log_format or '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default'],
                'level': level,
                'propagate': True
            }
        }
    }
    
    if log_file:
        config['handlers']['file'] = {
            'level': level,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': log_file,
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(config)

# Decorators
def async_timed():
    """Decorator to measure async function execution time"""
    def wrapper(func: Callable):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                await record_metric(
                    f"function_duration_{func.__name__}",
                    elapsed
                )
                
                logger.debug(
                    f"Function '{func.__name__}' took {elapsed:.2f} seconds"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"Function '{func.__name__}' failed after {elapsed:.2f} seconds: {e}"
                )
                raise
        return wrapped
    return wrapper

def validate_input(**validators: Callable):
    """Decorator to validate function inputs"""
    def wrapper(func: Callable):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            # Combine args and kwargs with function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument with its validator
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}'",
                            details={"value": value}
                        )
            
            return await func(*args, **kwargs)
        return wrapped
    return wrapper

# Hash Utilities
class HashUtils:
    """Utilities for hashing operations"""
    
    @staticmethod
    def hash_content(content: Union[str, bytes]) -> str:
        """Generate SHA-256 hash of content"""
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Generate deterministic hash of dictionary"""
        serialized = json.dumps(data, sort_keys=True)
        return HashUtils.hash_content(serialized)
    
    @staticmethod
    def hash_code_entity(entity: CodeEntity) -> str:
        """Generate hash for code entity"""
        content = f"{entity.name}:{entity.entity_type.value}:{entity.location.file}"
        return HashUtils.hash_content(content)
```

This merged implementation:
1. Combines all utility functions from both sources
2. Maintains consistency with the type system
3. Adds improved error handling and validation
4. Includes comprehensive metrics collection
5. Provides enhanced caching capabilities
6. Adds useful decorators for timing and validation
7. Includes robust logging setup
8. Adds hash utilities for various data types

The code is organized into logical sections with clear class hierarchies and follows best practices for async Python code. It provides a solid foundation for utility functions used throughout the DocuScribe system.  
[Timestamp: 2024/11/10 04:35:35]

**user**: Generate context - pool.py  
[Timestamp: 2024/11/10 04:36:17]

**ai**: I'll help you create the `context/pool.py` file that implements the SharedContextPool for DocuScribe. This implementation will incorporate the context management and vector storage capabilities discussed in the documentation.

```python
"""
SharedContextPool Implementation
Manages a pool of code contexts with efficient storage and retrieval capabilities.

Features:
- Vector-based similarity search
- Caching with TTL
- Async operations
- Relationship tracking
- Performance monitoring
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import faiss
from sentence_transformers import SentenceTransformer

from ..core.types import (
    CodeEntity,
    EntityType,
    Location,
    EmbeddingVector
)
from ..core.utils import (
    AsyncCache,
    Timer,
    ValidationUtils,
    HashUtils,
    record_metric
)

logger = logging.getLogger(__name__)

@dataclass
class ContextMetadata:
    """Metadata for stored contexts"""
    source: str
    language: str = "python"
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    relationships: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    additional: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StoredContext:
    """Represents a stored context with its metadata"""
    content: str
    metadata: ContextMetadata
    embedding: Optional[np.ndarray] = None
    entity: Optional[CodeEntity] = None
    vector_id: Optional[int] = None

@dataclass
class SearchResult:
    """Represents a context search result"""
    context_id: str
    content: str
    score: float
    metadata: ContextMetadata

class SharedContextPool:
    """
    Central context storage with vector-based similarity search.
    Manages code contexts, their relationships, and efficient retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        similarity_threshold: float = 0.7
    ):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Storage
        self.contexts: Dict[str, StoredContext] = {}
        self.index_to_id: Dict[int, str] = {}
        self.id_to_index: Dict[str, int] = {}
        
        # Cache
        self.cache = AsyncCache(max_size=cache_size, ttl_seconds=cache_ttl)
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        
        # Locks
        self._write_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        
        logger.info(
            f"Initialized SharedContextPool with embedding dimension {self.embedding_dim}"
        )

    async def add_context(
        self,
        content: str,
        metadata: Dict[str, Any],
        entity: Optional[CodeEntity] = None
    ) -> Optional[str]:
        """
        Add a new context to the pool.
        
        Args:
            content: The context content
            metadata: Context metadata
            entity: Optional code entity
            
        Returns:
            Context ID if successful, None otherwise
        """
        async with self._write_lock:
            try:
                # Generate context ID
                context_id = HashUtils.hash_content(content)
                
                # Check if already exists
                if context_id in self.contexts:
                    await self._update_existing_context(context_id, metadata)
                    return context_id
                
                # Generate embedding
                async with Timer("generate_embedding"):
                    embedding = self.embedding_model.encode(content)
                
                # Create metadata
                context_metadata = ContextMetadata(
                    source=metadata.get('source', 'unknown'),
                    language=metadata.get('language', 'python'),
                    tags=set(metadata.get('tags', [])),
                    additional=metadata
                )
                
                # Create stored context
                stored_context = StoredContext(
                    content=content,
                    metadata=context_metadata,
                    embedding=embedding,
                    entity=entity
                )
                
                # Add to storage
                self.contexts[context_id] = stored_context
                
                # Update index
                await self._update_index(context_id, embedding)
                
                # Update relationships
                if entity:
                    await self._update_relationships(context_id, entity)
                
                await record_metric("contexts_added", 1)
                return context_id
                
            except Exception as e:
                logger.error(f"Error adding context: {e}")
                return None

    async def get_context(
        self,
        query: str,
        max_results: int = 5,
        threshold: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Retrieve contexts similar to the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            threshold: Optional similarity threshold override
            include_metadata: Whether to include full metadata
            
        Returns:
            List of search results
        """
        try:
            # Check cache
            cache_key = f"query:{HashUtils.hash_content(query)}:{max_results}"
            if cached := await self.cache.get(cache_key):
                await record_metric("cache_hits", 1)
                return cached
            
            # Generate query embedding
            async with Timer("query_embedding"):
                query_embedding = self.embedding_model.encode(query)
            
            # Search index
            async with Timer("vector_search"):
                results = await self._search_similar(
                    query_embedding,
                    max_results,
                    threshold or self.similarity_threshold
                )
            
            # Format results
            search_results = []
            for context_id, score in results:
                if context := self.contexts.get(context_id):
                    await self._update_access(context_id)
                    
                    search_results.append(SearchResult(
                        context_id=context_id,
                        content=context.content,
                        score=score,
                        metadata=context.metadata if include_metadata else None
                    ))
            
            # Cache results
            await self.cache.set(cache_key, search_results)
            await record_metric("cache_misses", 1)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    async def _update_index(
        self,
        context_id: str,
        embedding: np.ndarray
    ) -> None:
        """Update FAISS index with new embedding"""
        async with self._index_lock:
            try:
                # Add to index
                vector_id = len(self.index_to_id)
                self.index.add(embedding.reshape(1, -1))
                
                # Update mappings
                self.index_to_id[vector_id] = context_id
                self.id_to_index[context_id] = vector_id
                
                # Update stored context
                self.contexts[context_id].vector_id = vector_id
                
            except Exception as e:
                logger.error(f"Error updating index: {e}")
                raise

    async def _search_similar(
        self,
        query_embedding: np.ndarray,
        k: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        """Search for similar
[Timestamp: 2024/11/10 04:36:54]

**user**: Continue
[Timestamp: 2024/11/10 04:36:59]

**ai**: I'll continue with the `context/pool.py` implementation:

```python
    async def _search_similar(
        self,
        query_embedding: np.ndarray,
        k: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        """Search for similar contexts using FAISS"""
        async with self._index_lock:
            try:
                # Ensure k doesn't exceed index size
                k = min(k, len(self.index_to_id))
                if k == 0:
                    return []
                
                # Search index
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1),
                    k
                )
                
                # Convert distances to similarity scores
                max_distance = np.max(distances)
                if max_distance == 0:
                    similarities = np.ones_like(distances)
                else:
                    similarities = 1 - (distances / max_distance)
                
                # Filter and format results
                results = []
                for idx, similarity in zip(indices[0], similarities[0]):
                    if similarity < threshold:
                        continue
                        
                    context_id = self.index_to_id.get(int(idx))
                    if not context_id:
                        continue
                        
                    context = self.contexts.get(context_id)
                    if not context:
                        continue
                        
                    # Combine similarity with importance
                    final_score = similarity * 0.7 + context.metadata.importance * 0.3
                    results.append((context_id, float(final_score)))
                
                return sorted(results, key=lambda x: x[1], reverse=True)
                
            except Exception as e:
                logger.error(f"Error searching similar contexts: {e}")
                return []

    async def _update_existing_context(
        self,
        context_id: str,
        new_metadata: Dict[str, Any]
    ) -> None:
        """Update metadata for existing context"""
        try:
            context = self.contexts[context_id]
            
            # Update metadata
            context.metadata.last_accessed = datetime.now()
            context.metadata.access_count += 1
            context.metadata.additional.update(new_metadata)
            
            # Update importance based on access patterns
            time_factor = 1.0 / (1.0 + (datetime.now() - context.metadata.created_at).days)
            access_factor = min(1.0, context.metadata.access_count / 100)
            context.metadata.importance = 0.4 + (0.3 * time_factor + 0.3 * access_factor)
            
        except Exception as e:
            logger.error(f"Error updating existing context: {e}")
            raise

    async def _update_relationships(
        self,
        context_id: str,
        entity: CodeEntity
    ) -> None:
        """Update relationships for a context based on its code entity"""
        try:
            context = self.contexts[context_id]
            
            # Find similar contexts
            if context.embedding is not None:
                similar = await self._search_similar(
                    context.embedding,
                    k=5,
                    threshold=self.similarity_threshold
                )
                
                # Update relationships
                for other_id, score in similar:
                    if other_id != context_id:
                        context.metadata.relationships.add(other_id)
                        if other := self.contexts.get(other_id):
                            other.metadata.relationships.add(context_id)
            
            # Add entity-based relationships
            if entity.metadata.get('related_entities'):
                for related_id in entity.metadata['related_entities']:
                    context.metadata.relationships.add(related_id)
                    
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

    async def _update_access(self, context_id: str) -> None:
        """Update access patterns for a context"""
        try:
            if context := self.contexts.get(context_id):
                context.metadata.last_accessed = datetime.now()
                context.metadata.access_count += 1
                await record_metric("context_accesses", 1)
                
        except Exception as e:
            logger.error(f"Error updating access: {e}")

    async def remove_context(self, context_id: str) -> bool:
        """Remove a context from the pool"""
        async with self._write_lock:
            try:
                if context := self.contexts.pop(context_id, None):
                    # Remove from index
                    if context.vector_id is not None:
                        self.index_to_id.pop(context.vector_id, None)
                        self.id_to_index.pop(context_id, None)
                    
                    # Remove relationships
                    for related_id in context.metadata.relationships:
                        if related := self.contexts.get(related_id):
                            related.metadata.relationships.remove(context_id)
                    
                    await record_metric("contexts_removed", 1)
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Error removing context: {e}")
                return False

    async def optimize(self) -> None:
        """Optimize the context pool"""
        async with self._write_lock:
            try:
                # Remove old contexts
                await self._cleanup_old_contexts()
                
                # Rebuild index if needed
                if len(self.contexts) > len(self.index_to_id):
                    await self._rebuild_index()
                
                # Update relationships
                await self._update_all_relationships()
                
                await record_metric("optimizations_performed", 1)
                
            except Exception as e:
                logger.error(f"Error optimizing context pool: {e}")

    async def _cleanup_old_contexts(self) -> None:
        """Remove old or unused contexts"""
        try:
            current_time = datetime.now()
            to_remove = []
            
            for context_id, context in self.contexts.items():
                # Remove if not accessed in 30 days and low importance
                if (
                    (current_time - context.metadata.last_accessed).days > 30 and
                    context.metadata.importance < 0.3 and
                    len(context.metadata.relationships) < 3
                ):
                    to_remove.append(context_id)
            
            # Remove contexts
            for context_id in to_remove:
                await self.remove_context(context_id)
                
        except Exception as e:
            logger.error(f"Error cleaning up contexts: {e}")

    async def _rebuild_index(self) -> None:
        """Rebuild FAISS index"""
        async with self._index_lock:
            try:
                # Create new index
                new_index = faiss.IndexFlatL2(self.embedding_dim)
                new_index_to_id = {}
                new_id_to_index = {}
                
                # Add all embeddings
                embeddings = []
                context_ids = []
                
                for context_id, context in self.contexts.items():
                    if context.embedding is not None:
                        embeddings.append(context.embedding)
                        context_ids.append(context_id)
                
                if embeddings:
                    embeddings_array = np.vstack(embeddings)
                    new_index.add(embeddings_array)
                    
                    # Update mappings
                    for i, context_id in enumerate(context_ids):
                        new_index_to_id[i] = context_id
                        new_id_to_index[context_id] = i
                        self.contexts[context_id].vector_id = i
                
                # Replace old index and mappings
                self.index = new_index
                self.index_to_id = new_index_to_id
                self.id_to_index = new_id_to_index
                
                await record_metric("index_rebuilds", 1)
                
            except Exception as e:
                logger.error(f"Error rebuilding index: {e}")

    async def get_stats
[Timestamp: 2024/11/10 04:37:41]

**user**: Continue
[Timestamp: 2024/11/10 04:38:06]

**ai**: I'll continue with the remaining methods for the `SharedContextPool` class in `context/pool.py`:

```python
    async def get_stats(self) -> Dict[str, Any]:
        """Get current context pool statistics"""
        try:
            active_contexts = sum(
                1 for c in self.contexts.values()
                if (datetime.now() - c.metadata.last_accessed).days < 7
            )
            
            avg_importance = np.mean([
                c.metadata.importance
                for c in self.contexts.values()
            ]) if self.contexts else 0.0
            
            avg_relationships = np.mean([
                len(c.metadata.relationships)
                for c in self.contexts.values()
            ]) if self.contexts else 0.0
            
            return {
                'total_contexts': len(self.contexts),
                'active_contexts': active_contexts,
                'index_size': len(self.index_to_id),
                'avg_importance': float(avg_importance),
                'avg_relationships': float(avg_relationships),
                'embedding_dimension': self.embedding_dim,
                'cache_size': await self.cache.get_size(),
                'last_optimization': self.last_optimization.isoformat() if hasattr(self, 'last_optimization') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    async def update_context(
        self,
        context_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing context's content and/or metadata.
        
        Args:
            context_id: ID of context to update
            content: Optional new content
            metadata_updates: Optional metadata updates
            
        Returns:
            Boolean indicating success
        """
        async with self._write_lock:
            try:
                if context := self.contexts.get(context_id):
                    # Update content if provided
                    if content is not None and content != context.content:
                        # Generate new embedding
                        new_embedding = self.embedding_model.encode(content)
                        
                        # Update index
                        if context.vector_id is not None:
                            async with self._index_lock:
                                self.index.remove_ids(np.array([context.vector_id]))
                                self.index.add(new_embedding.reshape(1, -1))
                        
                        context.content = content
                        context.embedding = new_embedding
                    
                    # Update metadata if provided
                    if metadata_updates:
                        context.metadata.additional.update(metadata_updates)
                        if 'tags' in metadata_updates:
                            context.metadata.tags.update(metadata_updates['tags'])
                        if 'importance' in metadata_updates:
                            context.metadata.importance = float(metadata_updates['importance'])
                    
                    context.metadata.last_accessed = datetime.now()
                    await record_metric("context_updates", 1)
                    return True
                    
                return False
                
            except Exception as e:
                logger.error(f"Error updating context: {e}")
                return False

    async def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search contexts using metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to match
            max_results: Maximum number of results to return
            
        Returns:
            List of matching search results
        """
        try:
            results = []
            
            for context_id, context in self.contexts.items():
                matches = True
                for key, value in metadata_filters.items():
                    if key == 'tags':
                        if not context.metadata.tags.intersection(value):
                            matches = False
                            break
                    elif key not in context.metadata.additional or context.metadata.additional[key] != value:
                        matches = False
                        break
                
                if matches:
                    results.append(SearchResult(
                        context_id=context_id,
                        content=context.content,
                        score=context.metadata.importance,
                        metadata=context.metadata
                    ))
            
            # Sort by importance and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    async def get_related_contexts(
        self,
        context_id: str,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Get contexts related to a given context.
        
        Args:
            context_id: ID of the context to find relations for
            max_results: Maximum number of results to return
            
        Returns:
            List of related contexts
        """
        try:
            if context := self.contexts.get(context_id):
                results = []
                
                # Get directly related contexts
                for related_id in context.metadata.relationships:
                    if related := self.contexts.get(related_id):
                        results.append(SearchResult(
                            context_id=related_id,
                            content=related.content,
                            score=related.metadata.importance,
                            metadata=related.metadata
                        ))
                
                # If we need more results, find semantically similar contexts
                if len(results) < max_results and context.embedding is not None:
                    similar = await self._search_similar(
                        context.embedding,
                        k=max_results - len(results),
                        threshold=self.similarity_threshold
                    )
                    
                    for similar_id, score in similar:
                        if similar_id not in context.metadata.relationships:
                            if similar := self.contexts.get(similar_id):
                                results.append(SearchResult(
                                    context_id=similar_id,
                                    content=similar.content,
                                    score=score,
                                    metadata=similar.metadata
                                ))
                
                return sorted(results, key=lambda x: x.score, reverse=True)[:max_results]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting related contexts: {e}")
            return []

    async def bulk_add_contexts(
        self,
        contexts: List[Tuple[str, Dict[str, Any], Optional[CodeEntity]]]
    ) -> List[Optional[str]]:
        """
        Add multiple contexts efficiently.
        
        Args:
            contexts: List of (content, metadata, entity) tuples
            
        Returns:
            List of context IDs (None for failed additions)
        """
        async with self._write_lock:
            try:
                context_ids = []
                embeddings = []
                valid_contexts = []
                
                # Generate embeddings in batch
                contents = [c[0] for c in contexts]
                batch_embeddings = self.embedding_model.encode(contents)
                
                # Process each context
                for i, (content, metadata, entity) in enumerate(contexts):
                    try:
                        context_id = HashUtils.hash_content(content)
                        
                        # Skip if already exists
                        if context_id in self.contexts:
                            await self._update_existing_context(context_id, metadata)
                            context_ids.append(context_id)
                            continue
                        
                        # Create metadata
                        context_metadata = ContextMetadata(
                            source=metadata.get('source', 'unknown'),
                            language=metadata.get('language', 'python'),
                            tags=set(metadata.get('tags', [])),
                            additional=metadata
                        )
                        
                        # Create stored context
                        stored_context = StoredContext(
                            content=content,
                            metadata=context_metadata,
                            embedding=batch_embeddings[i],
                            entity=entity
                        )
                        
                        self.contexts[context_id] = stored_context
                        
[Timestamp: 2024/11/10 04:38:44]

**user**: Finish it up
[Timestamp: 2024/11/10 04:38:51]

**ai**: Certainly! Let's complete the `bulk_add_contexts` method and wrap up the `SharedContextPool` implementation:

```python
                        # Store valid context and its embedding
                        valid_contexts.append((context_id, stored_context))
                        embeddings.append(batch_embeddings[i])
                        context_ids.append(context_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing context: {e}")
                        context_ids.append(None)
                
                # Update index with new embeddings
                if valid_contexts:
                    await self._update_index_bulk(valid_contexts, embeddings)
                
                # Update relationships for new contexts
                for context_id, stored_context in valid_contexts:
                    if stored_context.entity:
                        await self._update_relationships(context_id, stored_context.entity)
                
                await record_metric("bulk_contexts_added", len(valid_contexts))
                return context_ids
                
            except Exception as e:
                logger.error(f"Error adding contexts in bulk: {e}")
                return [None] * len(contexts)

    async def _update_index_bulk(
        self,
        valid_contexts: List[Tuple[str, StoredContext]],
        embeddings: List[np.ndarray]
    ) -> None:
        """Update FAISS index with multiple embeddings"""
        async with self._index_lock:
            try:
                # Add embeddings to index
                embeddings_array = np.vstack(embeddings)
                self.index.add(embeddings_array)
                
                # Update mappings
                for i, (context_id, stored_context) in enumerate(valid_contexts):
                    vector_id = len(self.index_to_id)
                    self.index_to_id[vector_id] = context_id
                    self.id_to_index[context_id] = vector_id
                    stored_context.vector_id = vector_id
                
            except Exception as e:
                logger.error(f"Error updating index in bulk: {e}")

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        pool = SharedContextPool()
        
        # Add a single context
        context_id = await pool.add_context(
            content="def example_function(): pass",
            metadata={"source": "example.py"}
        )
        print(f"Added context ID: {context_id}")
        
        # Retrieve contexts
        results = await pool.get_context("example_function")
        for result in results:
            print(f"Found context: {result.content} with score {result.score}")
        
        # Bulk add contexts
        bulk_contexts = [
            ("def func1(): pass", {"source": "file1.py"}, None),
            ("def func2(): pass", {"source": "file2.py"}, None)
        ]
        bulk_ids = await pool.bulk_add_contexts(bulk_contexts)
        print(f"Bulk added context IDs: {bulk_ids}")
        
        # Get stats
        stats = await pool.get_stats()
        print(f"Context pool stats: {stats}")

    asyncio.run(main())
```

## Key Features of the `SharedContextPool`

1. **Context Management**: 
   - Add, update, and remove contexts with metadata.
   - Bulk operations for efficiency.

2. **Vector-Based Search**:
   - Uses FAISS for fast similarity search.
   - Supports semantic search based on embeddings.

3. **Caching**:
   - Async cache with TTL to optimize repeated queries.

4. **Relationship Tracking**:
   - Tracks relationships between contexts based on similarity and metadata.

5. **Performance Monitoring**:
   - Records metrics for operations like context addition, search, and optimization.

6. **Concurrency**:
   - Uses async locks to manage concurrent access to shared resources.

This implementation provides a robust framework for managing and retrieving code contexts efficiently, leveraging vector embeddings for semantic understanding.  
[Timestamp: 2024/11/10 04:39:10]

**user**: learning - patterns.py  
[Timestamp: 2024/11/10 04:39:29]

**ai**: Let's create the `learning/patterns.py` file, which will focus on pattern learning and recognition within the DocuScribe system. This module will be responsible for identifying and managing patterns in code, usage, and behavior.

```python
"""
Pattern Learning Implementation
Provides functionality for recognizing and managing patterns in code and usage.

Features:
- Pattern recognition with clustering
- Pattern storage and retrieval
- Pattern evaluation and scoring
- Async operations for scalability
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import DBSCAN

from ..core.utils import Timer, record_metric

logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    """Represents a learned pattern"""
    pattern_id: str
    pattern_type: str  # e.g., 'code', 'usage', 'behavior'
    features: Dict[str, float]
    confidence: float
    frequency: int
    last_seen: datetime
    impact_score: float
    related_patterns: Set[str] = field(default_factory=set)

@dataclass
class PatternEvaluation:
    """Tracks pattern effectiveness"""
    accuracy: float
    relevance: float
    stability: float
    impact: float
    timestamp: datetime = field(default_factory=datetime.now)

class PatternRecognizer:
    """Advanced pattern recognition with learning capabilities"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_history: Dict[str, List[PatternEvaluation]] = {}
        self.min_confidence = min_confidence
        self.cluster_model = DBSCAN(eps=0.3, min_samples=2)
        self._lock = asyncio.Lock()

    async def recognize_patterns(
        self,
        data: Dict[str, Any],
        pattern_type: str = 'general'
    ) -> List[Pattern]:
        """Recognizes patterns in input data"""
        async with self._lock:
            try:
                # Extract features
                features = await self._extract_features(data)
                
                # Identify patterns
                patterns = await self._identify_patterns(features, pattern_type)
                
                # Update pattern database
                for pattern in patterns:
                    await self._update_pattern(pattern)
                
                # Evaluate and filter patterns
                valid_patterns = [
                    p for p in patterns
                    if p.confidence >= self.min_confidence
                ]
                
                await record_metric("patterns_recognized", len(valid_patterns))
                return valid_patterns
                
            except Exception as e:
                logger.error(f"Pattern recognition error: {e}")
                return []

    async def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extracts numerical features from input data"""
        try:
            features = []
            
            # Process different types of features
            if 'numerical_features' in data:
                features.extend(self._process_numerical_features(
                    data['numerical_features']
                ))
                
            if 'categorical_features' in data:
                features.extend(self._process_categorical_features(
                    data['categorical_features']
                ))
                
            if 'temporal_features' in data:
                features.extend(await self._process_temporal_features(
                    data['temporal_features']
                ))
                
            if 'structural_features' in data:
                features.extend(await self._process_structural_features(
                    data['structural_features']
                ))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.array([])

    def _process_numerical_features(
        self,
        features: Dict[str, float]
    ) -> List[float]:
        """Processes numerical features"""
        try:
            processed = []
            
            for key, value in features.items():
                # Normalize value
                if isinstance(value, (int, float)):
                    normalized = value / max(abs(value), 1.0)
                    processed.append(normalized)
                
            return processed
            
        except Exception as e:
            logger.error(f"Numerical feature processing error: {e}")
            return []

    def _process_categorical_features(
        self,
        features: Dict[str, str]
    ) -> List[float]:
        """Processes categorical features using one-hot encoding"""
        try:
            processed = []
            
            for key, value in features.items():
                # Create category hash for one-hot encoding
                category_hash = hash(f"{key}:{value}") % 2
                processed.append(float(category_hash))
                
            return processed
            
        except Exception as e:
            logger.error(f"Categorical feature processing error: {e}")
            return []

    async def _process_temporal_features(
        self,
        features: Dict[str, datetime]
    ) -> List[float]:
        """Processes temporal features"""
        try:
            processed = []
            current_time = datetime.now()
            
            for key, timestamp in features.items():
                if isinstance(timestamp, datetime):
                    # Convert to relative time features
                    time_diff = (current_time - timestamp).total_seconds()
                    
                    # Add multiple temporal aspects
                    processed.extend([
                        1.0 / (1.0 + time_diff),  # Recency
                        1.0 / (1.0 + (time_diff % 86400)),  # Daily pattern
                        1.0 / (1.0 + (time_diff % 3600))  # Hourly pattern
                    ])
                    
            return processed
            
        except Exception as e:
            logger.error(f"Temporal feature processing error: {e}")
            return []

    async def _process_structural_features(
        self,
        features: Dict[str, Any]
    ) -> List[float]:
        """Processes structural features"""
        try:
            processed = []
            
            for key, structure in features.items():
                if isinstance(structure, dict):
                    # Process depth
                    depth = self._calculate_structure_depth(structure)
                    processed.append(depth / 10.0)  # Normalize depth
                    
                    # Process breadth
                    breadth = len(structure)
                    processed.append(breadth / 100.0)  # Normalize breadth
                    
                    # Process complexity
                    complexity = self._calculate_structure_complexity(structure)
                    processed.append(complexity)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Structural feature processing error: {e}")
            return []

    def _calculate_structure_depth(self, structure: Dict) -> int:
        """Calculates the depth of a nested structure"""
        if not isinstance(structure, dict) or not structure:
            return 0
        return 1 + max(
            self._calculate_structure_depth(v) if isinstance(v, dict) else 0
            for v in structure.values()
        )

    def _calculate_structure_complexity(self, structure: Dict) -> float:
        """Calculates structural complexity score"""
        try:
            # Count nodes
            node_count = sum(1 for _ in self._iter_structure(structure))
            
            # Count edges
            edge_count = sum(
                1 for _ in self._iter_structure(structure)
                if isinstance(_, dict)
            )
            
            # Calculate complexity score
            if node_count > 0:
                return min(1.0, (edge_count / node_count) * 0.5)
            return 0.0
            
        except Exception as e:
            logger.error(f"Complexity calculation error: {e}")
            return 0.0

    def _iter_structure(self, obj: Any) -> Any:
        """Recursively iterates through a structure"""
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                yield from self._iter_structure(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._iter_structure(item)
        else:
            yield obj

    async def _identify_patterns(
        self,
        features: np.ndarray,
        pattern_type: str
    ) -> List[Pattern]:
        """Identifies patterns in feature space"""
        try:
            patterns = []
            
            # Cluster features
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            
            clusters = self.cluster_model.fit_predict(features)
            
            # Process each cluster
            unique_clusters = set(clusters)
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue
                    
                # Get cluster features
                cluster_mask = clusters == cluster_id
                cluster_features = features[cluster_mask]
                
                # Create pattern
                pattern = await self._create_pattern(
                    cluster_features,
                    pattern_type
                )
                
                if pattern:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern identification error: {e}")
            return []

    async def _create_pattern(
        self,
        features: np.ndarray,
        pattern_type: str
    ) -> Optional[Pattern]:
        """Creates a pattern from feature cluster"""
        try:
            # Calculate pattern characteristics
            mean_features = np.mean(features, axis=0)
            std_features = np.std(features, axis=0)
            
            # Convert to feature dictionary
            feature_dict = {
                f'feature_{i}': float(mean_features[i])
                for i in range(len(mean_features))
            }
            
            # Calculate confidence based on cluster tightness
            confidence = 1.0 / (1.0 + np.mean(std_features))
            
            # Create pattern
            pattern = Pattern(
                pattern_id=f"{pattern_type}_{datetime.now().timestamp()}",
                pattern_type=pattern_type,
                features=feature_dict,
                confidence=confidence,
                frequency=1,
                last_seen=datetime.now(),
                impact_score=0.0
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Pattern creation error: {e}")
            return None

    async def _update_pattern(self, pattern: Pattern) -> None:
        """Updates pattern database with new pattern"""
        try:
            if pattern.pattern_id in self.patterns:
                # Update existing pattern
                existing = self.patterns[pattern.pattern_id]
                existing.frequency += 1
                existing.last_seen = pattern.last_seen
                existing.confidence = (
                    existing.confidence * 0.9 +
                    pattern.confidence * 0.1
                )
                
                # Update relationships
                await self._update_pattern_relationships(
                    existing,
                    pattern
                )
            else:
                # Add new pattern
                self.patterns[pattern.pattern_id] = pattern
                
        except Exception as e:
            logger.error(f"Pattern update error: {e}")

    async def _update_pattern_relationships(
        self,
        existing: Pattern,
        new: Pattern
    ) -> None:
        """Updates relationships between patterns"""
        try:
            # Find related patterns
            for other_id, other_pattern in self.patterns.items():
                if other_id != existing.pattern_id:
                    similarity = await self._calculate_pattern_similarity(
                        existing,
                        other_pattern
                    )
                    
                    if similarity > self.min_confidence:
                        existing.related_patterns.add(other_id)
                        other_pattern.related_patterns.add(existing.pattern_id)
                        
        except Exception as e:
            logger.error(f"Pattern relationship update error: {e}")

    async def _calculate_pattern_similarity(
        self,
        pattern1: Pattern,
        pattern2: Pattern
    ) -> float:
        """Calculates similarity between patterns"""
        try:
            # Get common features
            common_features = set(pattern1.features.keys()) & set(pattern2.features.keys())
            
            if not common_features:
                return 0.0
                
            # Calculate similarity for common features
            similarities = []
            for feature in common_features:
                value1 = pattern1.features[feature]
                value2 = pattern2.features[feature]
                similarity = 1.0 - abs(value1 - value2)
                similarities.append(similarity)
                
            return np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Pattern similarity calculation error: {e}")
            return 0.0

    async def evaluate_pattern(
        self,
        pattern_id: str
    ) -> Optional[PatternEvaluation]:
        """Evaluates pattern effectiveness"""
        try:
            if pattern_id not in self.patterns:
                return None
                
            pattern = self.patterns[pattern_id]
            
            # Calculate accuracy based on confidence history
            history = self.pattern_history[pattern_id]
            if history:
                accuracy = np.mean([
                    eval.accuracy for eval in history[-10:]
                ])
            else:
                accuracy = pattern.confidence
                
            # Calculate relevance based on frequency and recency
            age = (datetime.now() - pattern.last_seen).total_seconds()
            relevance = pattern.frequency / (1.0 + np.log1p(age))
            
            # Calculate stability
            if len(history) >= 2:
                confidence_changes = [
                    abs(history[i].accuracy - history[i-1].accuracy)
                    for i in range(1, len(history))
                ]
                stability = 1.0 - min(1.0, np.mean(confidence_changes))
            else:
                stability = 1.0
                
            # Create evaluation
            evaluation = PatternEvaluation(
                accuracy=accuracy,
                relevance=relevance,
                stability=stability,
                impact=pattern.impact_score
            )
            
            # Update history
            self.pattern_history[pattern_id].append(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Pattern evaluation error: {e}")
            return None

    async def get_pattern_analytics(self) -> Dict[str, Any]:
        """Gets analytics about learned patterns"""
        try:
            analytics = {
                'total_patterns': len(self.patterns),
                'pattern_types': defaultdict(int),
                'average_confidence': 0.0,
                'average_frequency': 0.0,
                'pattern_relationships': 0,
                'evaluations': {}
            }
            
            if not self.patterns:
                return analytics
            
            # Calculate statistics
            confidences = []
            frequencies = []
            relationships = 0
            
            for pattern in self.patterns.values():
                analytics['pattern_types'][pattern.pattern_type] += 1
                confidences.append(pattern.confidence)
                frequencies.append(pattern.frequency)
                relationships += len(pattern.related_patterns)
                
                # Get latest evaluation
                evaluation = await self.evaluate_pattern(pattern.pattern_id)
                if evaluation:
                    analytics['evaluations'][pattern.pattern_id] = {
                        'accuracy': evaluation.accuracy,
                        'relevance': evaluation.relevance,
                        'stability': evaluation.stability,
                        'impact': evaluation.impact
                    }
            
            # Calculate averages
            analytics['average_confidence'] = np.mean(confidences)
            analytics['average_frequency'] = np.mean(frequencies)
            analytics['pattern_relationships'] = relationships
            
            return analytics
            
        except Exception as e:
            logger.error(f"Pattern analytics error: {e}")
            return {}
```

## Key Features of the `PatternRecognizer`

1. **Pattern Recognition**:
   - Uses clustering (DBSCAN) to identify patterns in feature space.
   - Supports various feature types: numerical, categorical, temporal, and structural.

2. **Pattern Management**:
   - Stores and updates patterns with metadata including confidence, frequency, and relationships.
   - Evaluates patterns for effectiveness and relevance.

3. **Async Operations**:
   - Uses async locks to ensure thread safety during concurrent operations.
   - Records metrics for recognized patterns.

4. **Pattern Analytics**:
   - Provides analytics on stored patterns, including types, confidence, and relationships.

This implementation provides a robust framework for recognizing and managing patterns in code and usage data, enhancing the adaptability and intelligence of the DocuScribe system.  
[Timestamp: 2024/11/10 04:40:46]
