**user**: [api_client.py](simple2/api_client.py.md) [docs.py](simple2/docs.py.md) [response_parser.py](simple2/response_parser.py.md) [documentation_analyzer.py](./documentation_analyzer.py.md) [interaction.py](interaction.py.md) [main.py](simple2/main.py.md)
[Timestamp: 2024/11/15 02:41:04]

**ai**: I'll help you analyze the codebase and suggest improvements for the docstring generation system. Here are some key observations and recommendations:

1. **API Client Improvements**:
```python
class AzureOpenAIClient:
    def __init__(self, ..., rate_limit_config: Optional[Dict] = None):
        # Add rate limiting configuration
        self.rate_limiter = RateLimiter(**(rate_limit_config or {
            'requests_per_minute': 60,
            'burst_size': 10
        }))
        
    async def get_docstring(self, ...) -> Optional[Dict[str, Any]]:
        # Add rate limiting
        async with self.rate_limiter:
            return await self._get_docstring_with_retry(...)
            
    async def _get_docstring_with_retry(self, ...):
        # Implement exponential backoff retry logic
        for attempt in range(self.max_retries):
            try:
                return await self._make_api_request(...)
            except RateLimitError:
                await self.handle_rate_limits(attempt)
```

2. **Response Parser Enhancement**:
```python
class ResponseParser:
    def validate_response_schema(self, response: Dict[str, Any]) -> bool:
        """
        Validate response against expected schema with detailed error reporting.
        
        Args:
            response (Dict[str, Any]): Response to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            jsonschema.validate(response, RESPONSE_SCHEMA)
            return True
        except ValidationError as e:
            log_error(f"Schema validation failed: {e.message}")
            log_error(f"Failed path: {' -> '.join(str(p) for p in e.path)}")
            return False
```

3. **Documentation Analyzer Improvements**:
```python
class DocumentationAnalyzer:
    def analyze_docstring_quality(self, docstring: str) -> Dict[str, Any]:
        """
        Analyze docstring quality with metrics.
        
        Args:
            docstring (str): Docstring to analyze
            
        Returns:
            Dict[str, Any]: Quality metrics including completeness, clarity score
        """
        metrics = {
            'completeness': self._check_completeness(docstring),
            'clarity_score': self._calculate_clarity_score(docstring),
            'has_examples': bool(re.search(r'Examples?:', docstring)),
            'has_type_hints': self._check_type_hints(docstring)
        }
        return metrics
```

4. **Interaction Handler Enhancement**:
```python
class InteractionHandler:
    async def process_all_functions(self, source_code: str) -> Tuple[str, str]:
        """
        Process functions with improved batching and progress tracking.
        
        Args:
            source_code (str): Source code to process
            
        Returns:
            Tuple[str, str]: Updated code and documentation
        """
        functions = self.extraction_manager.extract_metadata(source_code)
        
        # Add progress tracking
        progress = Progress()
        async for batch in self._batch_generator(functions):
            results = await asyncio.gather(
                *[self.process_function(func) for func in batch]
            )
            progress.update(len(batch))
            
        return self._update_source_code(results), self._generate_documentation(results)
```

5. **Main Workflow Improvements**:
```python
async def run_workflow(args: argparse.Namespace) -> None:
    """
    Enhanced workflow with better error handling and reporting.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        # Add configuration validation
        config = Configuration.from_args(args)
        if not config.validate():
            raise ValueError("Invalid configuration")
            
        # Add progress reporting
        async with ProcessingContext(config) as ctx:
            for file in ctx.get_python_files():
                await ctx.process_file(file)
                
    except Exception as e:
        log_exception("Workflow failed", exc_info=e)
        sys.exit(1)
```

6. **New Features to Add**:

```python
# Add parallel processing support
class ParallelProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        
    async def process_files(self, files: List[str]) -> List[Result]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_batch,
            files
        )

# Add caching improvements
class CacheManager:
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = aioredis.from_url(**redis_config)
        
    async def get_cached_result(self, key: str) -> Optional[Dict]:
        try:
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            log_error(f"Cache error: {e}")
            return None
```

These improvements focus on:
- Better error handling and reporting
- Rate limiting and retry logic
- Enhanced response validation
- Progress tracking
- Parallel processing
- Improved caching
- Better configuration management

Would you like me to elaborate on any of these improvements or suggest additional enhancements?
[Timestamp: 2024/11/15 02:41:28]

**user**: [Azure OpenAI Strategy Guide](./Azure%20OpenAI%20Strategy%20Guide.md)
[Timestamp: 2024/11/15 02:41:30]

**ai**: Based on the Azure OpenAI Strategy Guide and the existing codebase, I'll suggest some strategic improvements to enhance the docstring generation system:

1. **Enhanced Rate Limiting and Retry Logic**:
```python
class RateLimiter:
    """Rate limiter for Azure OpenAI API requests."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token for API request."""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * (self.requests_per_minute / 60)
            )
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.requests_per_minute / 60)
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_update = now
```

2. **Improved Structured Output Generation**:
```python
class DocstringGenerator:
    """Generate structured docstrings using Azure OpenAI."""

    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                },
                "returns": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    }
                },
                "raises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exception": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                }
            }
        }

    async def generate_docstring(self, function_info: Dict) -> Dict:
        """Generate structured docstring using function information."""
        response = await self.client.get_docstring(
            func_name=function_info['name'],
            params=function_info['args'],
            return_type=function_info['returns'],
            complexity_score=function_info.get('complexity_score', 0),
            existing_docstring=function_info.get('docstring', ''),
            schema=self.schema
        )
        return response
```

3. **Advanced RAG Implementation**:
```python
class DocstringRAG:
    """Retrieval-Augmented Generation for docstring generation."""

    def __init__(self, client: AzureOpenAIClient, search_client: SearchClient):
        self.client = client
        self.search_client = search_client

    async def get_similar_functions(self, function_info: Dict) -> List[Dict]:
        """Retrieve similar functions from the codebase."""
        query_vector = await self.client.get_embeddings(
            f"{function_info['name']} {function_info['docstring']}"
        )
        
        results = self.search_client.search(
            search_text=function_info['name'],
            vector_queries=[{
                "vector": query_vector,
                "k": 3,
                "fields": "content_vector"
            }],
            select=["content", "docstring"]
        )
        
        return list(results)

    async def generate_enhanced_docstring(self, function_info: Dict) -> str:
        """Generate docstring using similar function examples."""
        similar_functions = await self.get_similar_functions(function_info)
        
        context = "\n".join([
            f"Similar function example:\n{func['docstring']}"
            for func in similar_functions
        ])
        
        response = await self.client.get_docstring(
            func_name=function_info['name'],
            params=function_info['args'],
            return_type=function_info['returns'],
            complexity_score=function_info.get('complexity_score', 0),
            existing_docstring=function_info.get('docstring', ''),
            additional_context=context
        )
        
        return response
```

4. **Enhanced Monitoring and Metrics**:
```python
class DocstringMetrics:
    """Track and analyze docstring generation metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def log_generation(self, function_name: str, metrics: Dict):
        """Log metrics for a docstring generation."""
        self.metrics['generations'].append({
            'function': function_name,
            'timestamp': time.time(),
            'tokens_used': metrics.get('tokens', 0),
            'response_time': metrics.get('response_time', 0),
            'success': metrics.get('success', False)
        })

    def get_summary(self) -> Dict:
        """Get summary of docstring generation metrics."""
        total_generations = len(self.metrics['generations'])
        if not total_generations:
            return {'error': 'No generations recorded'}

        successful = sum(1 for g in self.metrics['generations'] if g['success'])
        
        return {
            'total_generations': total_generations,
            'success_rate': successful / total_generations,
            'average_response_time': statistics.mean(
                g['response_time'] for g in self.metrics['generations']
            ),
            'total_tokens': sum(g['tokens_used'] for g in self.metrics['generations']),
            'runtime': time.time() - self.start_time
        }
```

These improvements focus on:
- Robust rate limiting and retry logic
- Structured output generation with schema validation
- RAG implementation for better docstring quality
- Comprehensive monitoring and metrics

Would you like me to elaborate on any of these improvements or suggest additional enhancements?
[Timestamp: 2024/11/15 02:41:57]

**user**: Provide full implementations 
[Timestamp: 2024/11/15 02:47:28]

**ai**: I'll provide complete implementations for the enhanced docstring generation system. I'll break this down into several key modules:

1. First, let's create a new `rate_limiter.py`:

```python
"""
rate_limiter.py - Rate Limiting Module

This module provides rate limiting functionality for Azure OpenAI API requests,
implementing token bucket algorithm with burst handling.

Classes:
    RateLimiter: Manages API request rates using token bucket algorithm
    AsyncRateLimiter: Async version of rate limiter for concurrent operations
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
from logger import log_info, log_error, log_debug

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    burst_size: int = 10
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass

class AsyncRateLimiter:
    """
    Asynchronous rate limiter using token bucket algorithm.
    
    Attributes:
        requests_per_minute (int): Maximum requests allowed per minute
        burst_size (int): Maximum burst size allowed
        tokens (float): Current number of tokens available
        last_update (float): Timestamp of last token update
        lock (asyncio.Lock): Lock for thread-safe operations
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter with configuration.

        Args:
            config (RateLimitConfig): Rate limiting configuration
        """
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        log_debug(f"Rate limiter initialized with {config.requests_per_minute} rpm")

    async def acquire(self) -> bool:
        """
        Acquire a token for API request.

        Returns:
            bool: True if token acquired, False if rate limited
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded after retries
        """
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Replenish tokens based on time passed
            self.tokens = min(
                self.config.burst_size,
                self.tokens + time_passed * (self.config.requests_per_minute / 60.0)
            )
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.config.requests_per_minute / 60.0)
                log_debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                
                if wait_time > self.config.max_delay:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded. Required wait time: {wait_time:.2f}s"
                    )
                
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_update = now
            log_debug(f"Token acquired. {self.tokens:.2f} tokens remaining")
            return True

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

class RateLimitHandler:
    """
    Handles rate limiting with retries and backoff.
    
    Attributes:
        limiter (AsyncRateLimiter): Rate limiter instance
        config (RateLimitConfig): Rate limiting configuration
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limit handler.

        Args:
            config (Optional[RateLimitConfig]): Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        self.limiter = AsyncRateLimiter(self.config)

    async def execute_with_rate_limit(self, func, *args, **kwargs):
        """
        Execute function with rate limiting and retries.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function execution

        Raises:
            RateLimitExceededError: If rate limit is exceeded after all retries
        """
        for attempt in range(self.config.max_retries):
            try:
                async with self.limiter:
                    return await func(*args, **kwargs)
            except RateLimitExceededError as e:
                if attempt == self.config.max_retries - 1:
                    log_error(f"Rate limit exceeded after {attempt + 1} attempts")
                    raise
                
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay
                )
                log_info(f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            except Exception as e:
                log_error(f"Error during rate-limited execution: {str(e)}")
                raise

    async def check_rate_limit(self) -> bool:
        """
        Check if rate limit allows for a request.

        Returns:
            bool: True if request is allowed, False otherwise
        """
        try:
            return await self.limiter.acquire()
        except RateLimitExceededError:
            return False

    def get_status(self) -> dict:
        """
        Get current rate limiter status.

        Returns:
            dict: Current rate limiter status
        """
        return {
            "tokens_available": self.limiter.tokens,
            "requests_per_minute": self.config.requests_per_minute,
            "burst_size": self.config.burst_size,
            "last_update": self.limiter.last_update
        }
```

2. Next, let's create a new `docstring_generator.py`:

```python
"""
docstring_generator.py - Enhanced Docstring Generation Module

This module provides advanced docstring generation capabilities using Azure OpenAI,
with support for RAG, structured output, and comprehensive error handling.

Classes:
    DocstringGenerator: Main class for generating enhanced docstrings
    DocstringRAG: Implements Retrieval-Augmented Generation for docstrings
    DocstringMetrics: Tracks and analyzes docstring generation metrics
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict
import statistics
import time

from azure.search.documents import SearchClient
from rate_limiter import RateLimitHandler, RateLimitConfig
from logger import log_info, log_error, log_debug

@dataclass
class DocstringSchema:
    """Schema for structured docstring generation."""
    summary: str
    args: List[Dict[str, str]]
    returns: Dict[str, str]
    raises: Optional[List[Dict[str, str]]] = None
    examples: Optional[List[str]] = None
    notes: Optional[str] = None

class DocstringGenerator:
    """
    Enhanced docstring generator using Azure OpenAI with structured output.
    
    Attributes:
        client (AzureOpenAIClient): Azure OpenAI client instance
        rate_limiter (RateLimitHandler): Rate limiting handler
        metrics (DocstringMetrics): Metrics tracking instance
    """

    def __init__(self, client, rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize docstring generator.

        Args:
            client: Azure OpenAI client instance
            rate_limit_config (Optional[RateLimitConfig]): Rate limiting configuration
        """
        self.client = client
        self.rate_limiter = RateLimitHandler(rate_limit_config)
        self.metrics = DocstringMetrics()
        log_debug

    async def generate_structured_docstring(self, function_info: Dict) -> Optional[DocstringSchema]:
        """
        Generate structured docstring using function information.

        Args:
            function_info (Dict): Function metadata including name, args, returns, etc.

        Returns:
            Optional[DocstringSchema]: Generated docstring in structured format
        """
        try:
            start_time = time.time()
            
            async def _generate():
                response = await self.client.get_docstring(
                    func_name=function_info['name'],
                    params=function_info['args'],
                    return_type=function_info['returns'],
                    complexity_score=function_info.get('complexity_score', 0),
                    existing_docstring=function_info.get('docstring', ''),
                    decorators=function_info.get('decorators', []),
                    exceptions=function_info.get('exceptions', [])
                )
                
                if not response or 'content' not in response:
                    raise ValueError("Invalid response from API")
                
                return DocstringSchema(
                    summary=response['content']['summary'],
                    args=[{
                        'name': param[0],
                        'type': param[1],
                        'description': response['content'].get('args', {}).get(param[0], '')
                    } for param in function_info['args']],
                    returns={
                        'type': function_info['returns'],
                        'description': response['content'].get('returns', '')
                    },
                    raises=[{
                        'exception': exc,
                        'description': response['content'].get('raises', {}).get(exc, '')
                    } for exc in function_info.get('exceptions', [])]
                )

            result = await self.rate_limiter.execute_with_rate_limit(_generate)
            
            self.metrics.log_generation(
                function_info['name'],
                {
                    'success': True,
                    'response_time': time.time() - start_time,
                    'tokens': result.get('usage', {}).get('total_tokens', 0)
                }
            )
            
            return result

        except Exception as e:
            log_error(f"Error generating docstring for {function_info['name']}: {str(e)}")
            self.metrics.log_generation(
                function_info['name'],
                {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e)
                }
            )
            return None

class DocstringRAG:
    """
    Implements Retrieval-Augmented Generation for improved docstring generation.
    
    Attributes:
        client (AzureOpenAIClient): Azure OpenAI client instance
        search_client (SearchClient): Azure Cognitive Search client
        embedding_cache (Dict): Cache for function embeddings
    """

    def __init__(self, client, search_client: SearchClient):
        """
        Initialize DocstringRAG.

        Args:
            client: Azure OpenAI client instance
            search_client (SearchClient): Azure Cognitive Search client
        """
        self.client = client
        self.search_client = search_client
        self.embedding_cache = {}
        log_debug("DocstringRAG initialized")

    async def get_function_embedding(self, function_info: Dict) -> List[float]:
        """
        Get or generate embedding for a function.

        Args:
            function_info (Dict): Function metadata

        Returns:
            List[float]: Function embedding vector
        """
        cache_key = f"{function_info['name']}:{hash(str(function_info['args']))}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embedding_text = (
            f"{function_info['name']} "
            f"{' '.join(arg[0] for arg in function_info['args'])} "
            f"{function_info.get('docstring', '')}"
        )
        
        response = await self.client.get_embeddings(embedding_text)
        self.embedding_cache[cache_key] = response
        return response

    async def get_similar_functions(self, function_info: Dict, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar functions using vector search.

        Args:
            function_info (Dict): Function metadata
            top_k (int): Number of similar functions to retrieve

        Returns:
            List[Dict]: Similar functions with their docstrings
        """
        try:
            vector = await self.get_function_embedding(function_info)
            
            results = self.search_client.search(
                search_text=function_info['name'],
                vector_queries=[{
                    "vector": vector,
                    "k": top_k,
                    "fields": "content_vector"
                }],
                select=["content", "docstring", "name", "similarity_score"]
            )
            
            similar_functions = []
            for result in results:
                similar_functions.append({
                    'name': result['name'],
                    'docstring': result['docstring'],
                    'similarity_score': result.get('similarity_score', 0)
                })
            
            log_debug(f"Retrieved {len(similar_functions)} similar functions")
            return similar_functions

        except Exception as e:
            log_error(f"Error retrieving similar functions: {str(e)}")
            return []

    async def generate_enhanced_docstring(self, function_info: Dict) -> Optional[DocstringSchema]:
        """
        Generate enhanced docstring using similar function examples.

        Args:
            function_info (Dict): Function metadata

        Returns:
            Optional[DocstringSchema]: Generated docstring with RAG enhancement
        """
        try:
            similar_functions = await self.get_similar_functions(function_info)
            
            context = "\n\n".join([
                f"Similar function '{func['name']}':\n{func['docstring']}"
                for func in similar_functions
            ])
            
            enhanced_info = function_info.copy()
            enhanced_info['context'] = context
            
            generator = DocstringGenerator(self.client)
            return await generator.generate_structured_docstring(enhanced_info)

        except Exception as e:
            log_error(f"Error generating enhanced docstring: {str(e)}")
            return None

class DocstringMetrics:
    """
    Tracks and analyzes docstring generation metrics.
    
    Attributes:
        metrics (defaultdict): Collection of generation metrics
        start_time (float): Start time of metrics collection
    """

    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def log_generation(self, function_name: str, metrics: Dict):
        """
        Log metrics for a docstring generation.

        Args:
            function_name (str): Name of the function
            metrics (Dict): Metrics to log
        """
        self.metrics['generations'].append({
            'function': function_name,
            'timestamp': time.time(),
            'tokens_used': metrics.get('tokens', 0),
            'response_time': metrics.get('response_time', 0),
            'success': metrics.get('success', False),
            'error': metrics.get('error')
        })
        log_debug(f"Logged metrics for {function_name}")

    def get_summary(self) -> Dict:
        """
        Get summary of docstring generation metrics.

        Returns:
            Dict: Summary statistics of generations
        """
        total_generations = len(self.metrics['generations'])
        if not total_generations:
            return {'error': 'No generations recorded'}

        successful = sum(1 for g in self.metrics['generations'] if g['success'])
        response_times = [g['response_time'] for g in self.metrics['generations']]
        
        summary = {
            'total_generations': total_generations,
            'success_rate': successful / total_generations,
            'average_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'total_tokens': sum(g['tokens_used'] for g in self.metrics['generations']),
            'runtime': time.time() - self.start_time,
            'errors': [g['error'] for g in self.metrics['generations'] if g.get('error')],
            'performance_percentiles': {
                '95th': statistics.quantiles(response_times, n=20)[-1],
                '99th': statistics.quantiles(response_times, n=100)[-1]
            }
        }
        
        log_info(f"Generated metrics summary: {json.dumps(summary, indent=2)}")
        return summary

    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format.

        Args:
            format (str): Export format ('json' or 'csv')

        Returns:
            str: Formatted metrics data
        """
        if format == 'json':
            return json.dumps({
                'summary': self.get_summary(),
                'detailed_metrics': self.metrics
            }, indent=2)
        elif format == 'csv':
            # Implement CSV export if needed
            pass
        else:
            raise ValueError(f"Unsupported export format: {format}")
```

Now, let's create a new module `docstring_workflow.py` to orchestrate the entire docstring generation process:

```python
"""
docstring_workflow.py - Docstring Generation Workflow

This module orchestrates the entire docstring generation process, including
batch processing, caching, and error handling.

Classes:
    DocstringWorkflow: Main workflow orchestrator for docstring generation
    WorkflowConfig: Configuration for the docstring generation workflow
    WorkflowMetrics: Metrics tracking for the entire workflow
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from azure.search.documents import SearchClient
from docstring_generator import DocstringGenerator, DocstringRAG, DocstringMetrics
from rate_limiter import RateLimitConfig
from logger import log_info, log_error, log_debug

@dataclass
class WorkflowConfig:
    """Configuration for docstring generation workflow."""
    batch_size: int = 5
    max_retries: int = 3
    cache_ttl: int = 3600
    rate_limit_config: Optional[RateLimitConfig] = None
    output_dir: Path = Path("generated_docs")
    enable_rag: bool = True
    enable_caching: bool = True

class DocstringWorkflow:
    """
    Orchestrates the docstring generation workflow.
    
    Attributes:
        config (WorkflowConfig): Workflow configuration
        generator (DocstringGenerator): Docstring generator instance
        rag (Optional[DocstringRAG]): RAG implementation if enabled
        metrics (WorkflowMetrics): Workflow metrics tracking
    """

    def __init__(
        self,
        client,
        search_client: Optional[SearchClient] = None,
        config: Optional[WorkflowConfig] = None
    ):
        """
        Initialize workflow orchestrator.

        Args:
            client: Azure OpenAI client instance
            search_client (Optional[SearchClient]): Search client for RAG
            config (Optional[WorkflowConfig]): Workflow configuration
        """
        self.config = config or WorkflowConfig()
        self.generator = DocstringGenerator(client, self.config.rate_limit_config)
        self.rag = DocstringRAG(client, search_client) if search_client and self.config.enable_rag else None
        self.metrics = WorkflowMetrics()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        log_debug("DocstringWorkflow initialized")

    async def process_batch(self, functions: List[Dict]) -> List[Optional[Dict]]:
        """
        Process a batch of functions for docstring generation.

        Args:
            functions (List[Dict]): List of function metadata

        Returns:
            List[Optional[Dict]]: Generated docstrings and metadata
        """
        tasks = []
        for func_info in functions:
            if self.config.enable_rag and self.rag:
                tasks.append(self.rag.generate_enhanced_docstring(func_info))
            else:
                tasks.append(self.generator.generate_structured_docstring(func_info))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results = []
            
            for func_info, result in zip(functions, results):
                if isinstance(result, Exception):
                    log_error(f"Error processing {func_info['name']}: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append({
                        'function': func_info['name'],
                        'docstring': result,
                        'timestamp': time.time()
                    })
            
            return processed_results

        except Exception as e:
            log_error(f"Batch processing error: {str(e)}")
            return [None] * len(functions)

    async def process_all_functions(self, functions: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Process all functions with batch processing and metrics tracking.

        Args:
            functions (List[Dict]): List of all functions to process

        Returns:
            Tuple[List[Dict], Dict]: Processed results and workflow metrics
        """
        start_time = time.time()
        all_results = []
        
        try:
            for i in range(0, len(functions), self.config.batch_size):
                batch = functions[i:i + self.config.batch_size]
                log_info(f"Processing batch {i // self.config.batch_size + 1}")
                
                results = await self.process_batch(batch)
                all_results.extend(results)
                
                # Update metrics
                self.metrics.update_batch_metrics(results)
                
                # Save intermediate results
                await self.save_intermediate_results(all_results)

            workflow_metrics = self.metrics.get_workflow_summary(
                total_time=time.time() - start_time
            )
            
            return all_results, workflow_metrics

        except Exception as e:
            log_error(f"Error in workflow execution: {str(e)}")
            raise

    async def save_intermediate_results(self, results: List[Dict]):
        """
        Save intermediate results to prevent data loss.

        Args:
            results (List[Dict]): Results to save
        """
        try:
            intermediate_file = self.config.output_dir / "intermediate_results.json"
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2)
            log_debug("Saved intermediate results")
        except Exception as e:
            log_error(f"Error saving intermediate results: {str(e)}")

    def generate_report(self, results: List[Dict], metrics: Dict) -> str:
        """
        Generate a comprehensive report of the docstring generation process.

        Args:
            results (List[Dict]): Processing results
            metrics (Dict): Workflow metrics

        Returns:
            str: Formatted report
        """
        report = [
            "# Docstring Generation Report\n",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## Summary\n",
            f"- Total functions processed: {len(results)}",
            f"- Successful generations: {metrics['successful_generations']}",
            f"- Failed generations: {metrics['failed_generations']}",
            f"- Total processing time: {metrics['total_time']:.2f}s",
            f"- Average processing time per function: {metrics['avg_time_per_function']:.2f}s",
            f"- Success rate: {metrics['success_rate']*100:.1f}%\n",
            "\n## Detailed Results\n"
        ]

        # Add detailed results for each function
        for result in results:
            if result:
                report.extend([
                    f"### Function: {result['function']}\n",
                    "```python",
                    str(result['docstring']),
                    "```\n"
                ])

        # Add error summary if any
        if metrics.get('errors'):
            report.extend([
                "\n## Errors\n",
                *[f"- {error}" for error in metrics['errors']]
            ])

        return "\n".join(report)

class WorkflowMetrics:
    """
    Tracks and analyzes metrics for the entire docstring generation workflow.
    
    Attributes:
        batch_metrics (List[Dict]): Metrics for each processed batch
        error_log (List[Dict]): Log of errors encountered
        start_time (float): Workflow start time
    """

    def __init__(self):
        """Initialize workflow metrics tracking."""
        self.batch_metrics = []
        self.error_log = []
        self.start_time = time.time()
        self.function_metrics = defaultdict(dict)

    def update_batch_metrics(self, batch_results: List[Dict]):
        """
        Update metrics with results from a processed batch.

        Args:
            batch_results (List[Dict]): Results from batch processing
        """
        batch_metrics = {
            'timestamp': time.time(),
            'size': len(batch_results),
            'successful': sum(1 for r in batch_results if r is not None),
            'failed': sum(1 for r in batch_results if r is None)
        }
        
        self.batch_metrics.append(batch_metrics)
        
        # Update individual function metrics
        for result in batch_results:
            if result:
                self.function_metrics[result['function']].update({
                    'status': 'success',
                    'timestamp': result['timestamp']
                })
            else:
                self.function_metrics[result['function']].update({
                    'status': 'failed',
                    'timestamp': time.time()
                })

    def log_error(self, error: str, context: Dict = None):
        """
        Log an error with context.

        Args:
            error (str): Error message
            context (Dict): Error context
        """
        self.error_log.append({
            'timestamp': time.time(),
            'error': error,
            'context': context or {}
        })

    def get_workflow_summary(self, total_time: float) -> Dict:
        """
        Get comprehensive workflow metrics summary.

        Args:
            total_time (float): Total workflow execution time

        Returns:
            Dict: Workflow metrics summary
        """
        total_functions = sum(batch['size'] for batch in self.batch_metrics)
        successful_generations = sum(batch['successful'] for batch in self.batch_metrics)
        
        return {
            'total_functions': total_functions,
            'successful_generations': successful_generations,
            'failed_generations': total_functions - successful_generations,
            'success_rate': successful_generations / total_functions if total_functions > 0 else 0,
            'total_time': total_time,
            'avg_time_per_function': total_time / total_functions if total_functions > 0 else 0,
            'batch_statistics': {
                'total_batches': len(self.batch_metrics),
                'avg_batch_size': statistics.mean([b['size'] for b in self.batch_metrics]),
                'avg_batch_success_rate': statistics.mean([b['successful']/b['size'] for b in self.batch_metrics])
            },
            'errors': [{'message': e['error'], 'context': e['context']} for e in self.error_log],
            'performance_metrics': self._calculate_performance_metrics()
        }

    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate detailed performance metrics.

        Returns:
            Dict: Performance metrics
        """
        function_times = [
            metrics['timestamp'] - self.start_time
            for metrics in self.function_metrics.values()
        ]
        
        return {
            'min_processing_time': min(function_times, default=0),
            'max_processing_time': max(function_times, default=0),
            'avg_processing_time': statistics.mean(function_times) if function_times else 0,
            'median_processing_time': statistics.median(function_times) if function_times else 0,
            'percentiles': {
                '95th': statistics.quantiles(function_times, n=20)[-1] if function_times else 0,
                '99th': statistics.quantiles(function_times, n=100)[-1] if function_times else 0
            }
        }
```

Now, let's create a new module `docstring_cache.py` for handling caching:

```python
"""
docstring_cache.py - Docstring Caching System

This module provides caching functionality for generated docstrings to improve
performance and reduce API calls.

Classes:
    DocstringCache: Manages caching of generated docstrings
    CacheConfig: Configuration for the caching system
"""

import asyncio
from typing import Optional, Dict, Any
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
import aioredis

from logger import log_info, log_error, log_debug

@dataclass
class CacheConfig:
    """Configuration for docstring caching."""
    redis_url: str = "redis://localhost:6379"
    ttl: int = 3600  # 1 hour
    namespace: str = "docstring"
    enable_local_cache: bool = True
    local_cache_size: int = 1000
    local_cache_ttl: int = 300  # 5 minutes

class DocstringCache:
    """
    Manages caching of generated docstrings.
    
    Attributes:
        config (CacheConfig): Cache configuration
        redis (aioredis.Redis): Redis client instance
        local_cache (Dict): In-memory cache
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the caching system.

        Args:
            config (Optional[CacheConfig]): Cache configuration
        """
        self.config = config or CacheConfig()
        self.redis = None
        self.local_cache = {}
        self.local_cache_timestamps = {}
        log_debug("DocstringCache initialized")

    async def connect(self):
        """Establish connection to Redis."""
        try:
            self.redis = await aioredis.from_url(self.config.redis_url)
            log_info("Connected to Redis cache")
        except Exception as e:
            log_error(f"Failed to connect to Redis: {str(e)}")
            raise

    def generate_cache_key(self, function_info: Dict) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_info (Dict): Function metadata

        Returns:
            str: Cache key
        """
        key_content = {
            'name': function_info['name'],
            'args': function_info['args'],
            'returns': function_info['returns'],
            'decorators': function_info.get('decorators', []),
            'exceptions': function_info.get('exceptions', [])
        }
        
        key_string = json.dumps(key_content, sort_keys=True)
        return f"{self.config.namespace}:{hashlib.sha256(key_string.encode()).hexdigest()}"

    async def get(self, function_info: Dict) -> Optional[Dict]:
        """
        Retrieve cached docstring for a function.

        Args:
            function_info (Dict): Function metadata

        Returns:
            Optional[Dict]: Cached docstring data if available
        """
        cache_key = self.generate_cache_key(function_info)

        # Check local cache first
        if self.config.enable_local_cache:
            local_result = self._get_from_local_cache(cache_key)
            if local_result:
                log_debug(f"Local cache hit for {function_info['name']}")
                return local
                ```
	            