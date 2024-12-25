---
epoch: 1731672312004
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [api_client.py](api_client.py.md)*[main.py](main.py.md)[extraction_manager.py](extraction_manager.py.md)*[response_parser.py](response_parser.py.md)*[cache.py](cache.py.md)*[[interac@20241115_060512

**user**: [api_client.py](simple2/api_client.py.md) [main.py](simple2/main.py.md)[extraction_manager.py](simple2/extraction_manager.py.md) [response_parser.py](simple2/response_parser.py.md) [cache.py](simple2/cache.py.md) [interaction.py](interaction.py.md) [documentation_analyzer.py](./documentation_analyzer.py.md) [docs.py](simple2/docs.py.md) [monitoring.py](simple2/monitoring.py.md)  
[Timestamp: 2024/11/15 06:05:12]

**ai**: Based on the provided code files, I'll help you understand the system and suggest improvements. This appears to be a comprehensive docstring generation system using Azure OpenAI. Let me break down the key components and suggest some enhancements.

Key Components:
1. API Client (api_client.py) - Handles Azure OpenAI API interactions
2. Main Workflow (main.py) - Orchestrates the docstring generation process
3. Documentation Management (docs.py) - Handles documentation generation and formatting
4. Cache System (cache.py) - Provides caching functionality
5. Monitoring (monitoring.py) - Tracks system metrics and performance

Here are some suggested improvements and additions:

1. Add a Configuration Manager:

```python
# config_manager.py
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class APIConfig:
    endpoint: str
    api_key: str
    model: str
    api_version: str

@dataclass
class CacheConfig:
    host: str
    port: int
    db: int
    ttl: int

@dataclass
class SystemConfig:
    api: APIConfig
    cache: CacheConfig
    batch_size: int
    output_dir: str
    log_level: str

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> SystemConfig:
        with open(self.config_path) as f:
            config_data = yaml.safe_load(f)
            return SystemConfig(
                api=APIConfig(**config_data['api']),
                cache=CacheConfig(**config_data['cache']),
                batch_size=config_data['batch_size'],
                output_dir=config_data['output_dir'],
                log_level=config_data['log_level']
            )

    def get_api_config(self) -> APIConfig:
        return self.config.api

    def get_cache_config(self) -> CacheConfig:
        return self.config.cache
```

2. Add a Rate Limiter:

```python
# rate_limiter.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.requests: Dict[datetime, int] = {}
        
    async def acquire(self) -> None:
        now = datetime.now()
        self._cleanup_old_requests(now)
        
        current_requests = sum(self.requests.values())
        if current_requests >= self.requests_per_minute:
            wait_time = self._calculate_wait_time(now)
            await asyncio.sleep(wait_time)
            
        self.requests[now] = self.requests.get(now, 0) + 1
        
    def _cleanup_old_requests(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_size)
        self.requests = {
            ts: count for ts, count in self.requests.items()
            if ts > cutoff
        }
        
    def _calculate_wait_time(self, now: datetime) -> float:
        oldest_request = min(self.requests.keys())
        return (oldest_request + timedelta(seconds=self.window_size) - now).total_seconds()
```

3. Add Error Recovery and Retry Logic:

```python
# error_recovery.py
import asyncio
from typing import Callable, Any, Optional
from functools import wraps

class RetryStrategy:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
        raise last_error

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    retry_strategy = RetryStrategy(max_retries, base_delay)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_strategy.execute(func, *args, **kwargs)
        return wrapper
    return decorator
```

4. Add a Progress Tracker:

```python
# progress_tracker.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class ProgressEntry:
    function_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None

class ProgressTracker:
    def __init__(self):
        self.entries: Dict[str, ProgressEntry] = {}
        
    def start_function(self, function_name: str) -> None:
        self.entries[function_name] = ProgressEntry(
            function_name=function_name,
            status="in_progress",
            start_time=datetime.now()
        )
        
    def complete_function(self, function_name: str, error: Optional[str] = None) -> None:
        if function_name in self.entries:
            entry = self.entries[function_name]
            entry.end_time = datetime.now()
            entry.status = "error" if error else "completed"
            entry.error = error
            
    def get_progress_report(self) -> Dict[str, Any]:
        total = len(self.entries)
        completed = len([e for e in self.entries.values() if e.status == "completed"])
        failed = len([e for e in self.entries.values() if e.status == "error"])
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": total - completed - failed,
            "completion_percentage": (completed / total * 100) if total > 0 else 0
        }
```

These improvements would enhance the system's reliability, maintainability, and monitoring capabilities. The Configuration Manager provides centralized configuration management, the Rate Limiter prevents API throttling, the Error Recovery system adds robust retry logic, and the Progress Tracker helps monitor the docstring generation process.

Would you like me to provide more specific improvements for any particular component or explain any of these suggestions in more detail?  
[Timestamp: 2024/11/15 06:05:41]

**user**: [logger.py](simple2/logger.py.md)[token_management.py](simple2/token_management.py.md) [schema.py](simple2/schema.py.md) [base.py](simple2/base.py.md) [functions.py](simple2/functions.py.md) [classes.py](simple2/classes.py.md) [metrics.py](simple2/metrics.py.md) [utils.py](simple2/utils.py.md) [docstring_schema.json](./docstring_schema.json.md) Is the rest of the source code.  
[Timestamp: 2024/11/15 06:06:04]

**ai**: Based on the additional code files provided, I'll suggest some more focused improvements and integrations:

1. Add a Docstring Quality Analyzer:

```python
# docstring_analyzer.py
from typing import Dict, List, Optional
from schema import DocstringSchema
import ast
from metrics import Metrics

class DocstringQualityAnalyzer:
    """Analyzes and scores docstring quality."""
    
    def __init__(self):
        self.metrics = Metrics()
        
    def analyze_docstring(self, node: ast.AST, docstring: str) -> Dict[str, float]:
        """
        Analyze docstring quality and provide scores.
        
        Args:
            node: AST node containing the docstring
            docstring: The docstring to analyze
            
        Returns:
            Dict containing quality metrics
        """
        scores = {
            'completeness': self._score_completeness(docstring),
            'clarity': self._score_clarity(docstring),
            'consistency': self._score_consistency(docstring),
            'complexity_match': self._score_complexity_match(node, docstring)
        }
        
        scores['overall'] = sum(scores.values()) / len(scores)
        return scores
        
    def _score_completeness(self, docstring: str) -> float:
        """Score docstring completeness (0-1)."""
        required_sections = ['Args', 'Returns', 'Raises']
        present_sections = sum(1 for section in required_sections 
                             if section.lower() in docstring.lower())
        return present_sections / len(required_sections)
        
    def _score_clarity(self, docstring: str) -> float:
        """Score docstring clarity (0-1)."""
        words = docstring.split()
        if not words:
            return 0.0
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        return min(1.0, 2.0 / avg_word_length)  # Prefer shorter words
        
    def _score_consistency(self, docstring: str) -> float:
        """Score docstring style consistency (0-1)."""
        style_markers = {
            'google': ['Args:', 'Returns:', 'Raises:'],
            'sphinx': [':param', ':return:', ':raises'],
            'numpy': ['Parameters', '----------', 'Returns']
        }
        
        style_scores = {
            style: sum(1 for marker in markers if marker in docstring)
            for style, markers in style_markers.items()
        }
        
        return max(style_scores.values()) / max(len(markers) 
                                              for markers in style_markers.values())
                                              
    def _score_complexity_match(self, node: ast.AST, docstring: str) -> float:
        """Score how well docstring detail matches code complexity."""
        complexity = self.metrics.calculate_complexity(node)
        docstring_detail = len(docstring.split())
        
        # More complex code should have more detailed docstrings
        expected_detail = complexity * 10
        return min(1.0, docstring_detail / expected_detail)
```

2. Add a Token Budget Manager:

```python
# token_budget.py
from dataclasses import dataclass
from typing import Optional, Dict
import time

@dataclass
class TokenBudget:
    """Manages token usage and budgets."""
    
    max_tokens_per_minute: int
    max_tokens_per_hour: int
    max_tokens_per_day: int
    
    def __init__(
        self,
        max_tokens_per_minute: int = 90_000,
        max_tokens_per_hour: int = 3_600_000,
        max_tokens_per_day: int = 80_000_000
    ):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_tokens_per_hour = max_tokens_per_hour
        self.max_tokens_per_day = max_tokens_per_day
        
        self.usage = {
            'minute': {'tokens': 0, 'reset_time': time.time()},
            'hour': {'tokens': 0, 'reset_time': time.time()},
            'day': {'tokens': 0, 'reset_time': time.time()}
        }
        
    async def check_and_update_budget(self, tokens: int) -> bool:
        """
        Check if token usage is within budget and update usage.
        
        Args:
            tokens: Number of tokens to use
            
        Returns:
            bool: True if within budget, False otherwise
        """
        current_time = time.time()
        self._reset_if_needed(current_time)
        
        # Check all time windows
        if (self.usage['minute']['tokens'] + tokens > self.max_tokens_per_minute or
            self.usage['hour']['tokens'] + tokens > self.max_tokens_per_hour or
            self.usage['day']['tokens'] + tokens > self.max_tokens_per_day):
            return False
            
        # Update usage
        self.usage['minute']['tokens'] += tokens
        self.usage['hour']['tokens'] += tokens
        self.usage['day']['tokens'] += tokens
        
        return True
        
    def _reset_if_needed(self, current_time: float) -> None:
        """Reset usage counters if time window has passed."""
        if current_time - self.usage['minute']['reset_time'] >= 60:
            self.usage['minute'] = {'tokens': 0, 'reset_time': current_time}
            
        if current_time - self.usage['hour']['reset_time'] >= 3600:
            self.usage['hour'] = {'tokens': 0, 'reset_time': current_time}
            
        if current_time - self.usage['day']['reset_time'] >= 86400:
            self.usage['day'] = {'tokens': 0, 'reset_time': current_time}
```

3. Add a Docstring Template Manager:

```python
# template_manager.py
from typing import Dict, Optional
import jinja2
from schema import DocstringSchema

class DocstringTemplateManager:
    """Manages docstring templates and generation."""
    
    def __init__(self):
        self.env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        self._load_templates()
        
    def _load_templates(self):
        """Load default templates."""
        self.templates = {
            'function': self.env.from_string('''
                {{ description }}

                Args:
                    {% for param in parameters %}
                    {{ param.name }} ({{ param.type }}): {{ param.description }}
                    {% endfor %}

                Returns:
                    {{ returns.type }}: {{ returns.description }}

                {% if raises %}
                Raises:
                    {% for exception in raises %}
                    {{ exception.type }}: {{ exception.description }}
                    {% endfor %}
                {% endif %}

                {% if examples %}
                Examples:
                    {% for example in examples %}
                    {{ example }}
                    {% endfor %}
                {% endif %}
            '''.strip()),
            
            'class': self.env.from_string('''
                {{ description }}

                Attributes:
                    {% for attr in attributes %}
                    {{ attr.name }} ({{ attr.type }}): {{ attr.description }}
                    {% endfor %}

                {% if methods %}
                Methods:
                    {% for method in methods %}
                    {{ method.name }}: {{ method.description }}
                    {% endfor %}
                {% endif %}
            '''.strip())
        }
        
    def generate_docstring(
        self,
        template_type: str,
        context: Dict,
        style: str = 'google'
    ) -> str:
        """
        Generate a docstring using a template.
        
        Args:
            template_type: Type of template to use
            context: Template context data
            style: Docstring style to use
            
        Returns:
            str: Generated docstring
        """
        template = self.templates.get(template_type)
        if
[Timestamp: 2024/11/15 06:07:25]

**user**: Merge this rate_limiter.py module into the existing [[token_management.py]]

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

[Timestamp: 2024/11/15 06:08:33]

**ai**: I'll help you merge the rate limiting functionality into the token management module. Here's the integrated version:

```python
"""
Token and Rate Management Module

Handles token counting, optimization, and rate limiting for Azure OpenAI API requests.
Provides efficient token estimation, prompt optimization, and request rate management.

Version: 1.2.0
Author: Development Team
"""

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import tiktoken
from logger import log_debug, log_error, log_info

@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

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
    """Asynchronous rate limiter using token bucket algorithm."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        log_debug(f"Rate limiter initialized with {config.requests_per_minute} rpm")

    async def acquire(self) -> bool:
        """Acquire a token for API request."""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
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

class TokenManager:
    """Token and rate management for Azure OpenAI API requests."""

    MODEL_LIMITS = {
        "gpt-4": {"max_tokens": 8192, "cost_per_1k_prompt": 0.03, "cost_per_1k_completion": 0.06},
        "gpt-4-32k": {"max_tokens": 32768, "cost_per_1k_prompt": 0.06, "cost_per_1k_completion": 0.12},
        "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_1k_prompt": 0.0015, "cost_per_1k_completion": 0.002},
        "gpt-3.5-turbo-16k": {"max_tokens": 16384, "cost_per_1k_prompt": 0.003, "cost_per_1k_completion": 0.004}
    }

    def __init__(
        self,
        model: str = "gpt-4",
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        """Initialize TokenManager with model configuration and rate limiting."""
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_config = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["gpt-4"])
        self.rate_limiter = AsyncRateLimiter(rate_limit_config or RateLimitConfig())
        log_debug(f"TokenManager initialized for model: {model}")

    async def execute_with_rate_limit(self, func, *args, **kwargs):
        """Execute function with rate limiting and retries."""
        for attempt in range(self.rate_limiter.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                return await func(*args, **kwargs)
            except RateLimitExceededError as e:
                if attempt == self.rate_limiter.config.max_retries - 1:
                    log_error(f"Rate limit exceeded after {attempt + 1} attempts")
                    raise
                
                delay = min(
                    self.rate_limiter.config.base_delay * (2 ** attempt),
                    self.rate_limiter.config.max_delay
                )
                log_info(f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

    @lru_cache(maxsize=128)
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text with caching."""
        try:
            tokens = len(self.encoding.encode(text))
            log_debug(f"Estimated {tokens} tokens for text")
            return tokens
        except Exception as e:
            log_error(f"Error estimating tokens: {e}")
            return 0

    async def optimize_prompt_with_rate_limit(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """Optimize prompt with rate limiting."""
        return await self.execute_with_rate_limit(
            self.optimize_prompt,
            text,
            max_tokens,
            preserve_sections
        )

    def optimize_prompt(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """Optimize prompt to fit within token limits."""
        # ... (rest of the existing optimize_prompt implementation)

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage and cost."""
        # ... (existing _calculate_usage implementation)

    async def validate_request_with_rate_limit(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> Tuple[bool, Dict[str, int], str]:
        """Validate request with rate limiting."""
        return await self.execute_with_rate_limit(
            self.validate_request,
            prompt,
            max_completion_tokens
        )

    def get_rate_limit_status(self) -> dict:
        """Get current rate limiter status."""
        return {
            "tokens_available": self.rate_limiter.tokens,
            "requests_per_minute": self.rate_limiter.config.requests_per_minute,
            "burst_size": self.rate_limiter.config.burst_size,
            "last_update": self.rate_limiter.last_update
        }

# Maintain backward compatibility
@lru_cache(maxsize=128)
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Legacy function for token estimation."""
    manager = TokenManager(model)
    return manager.estimate_tokens(text)

async def optimize_prompt_async(text: str, max_tokens: int = 4000) -> str:
    """Async function for prompt optimization with rate limiting."""
    manager = TokenManager()
    optimized_text, _ = await manager.optimize_prompt_with_rate_limit(text, max_tokens)
    return optimized_text

def optimize_prompt(text: str, max_
[Timestamp: 2024/11/15 06:09:12]

**user**: Continue
[Timestamp: 2024/11/15 06:09:18]

**ai**: Here's the completion of the merged token_management.py file:

```python
def optimize_prompt(text: str, max_tokens: int = 4000) -> str:
    """Legacy function for prompt optimization."""
    manager = TokenManager()
    optimized_text, _ = manager.optimize_prompt(text, max_tokens)
    return optimized_text

class TokenBudget:
    """Manages token usage budgets across different time windows."""
    
    def __init__(
        self,
        max_tokens_per_minute: int = 90_000,
        max_tokens_per_hour: int = 3_600_000,
        max_tokens_per_day: int = 80_000_000
    ):
        """Initialize token budget tracking."""
        self.limits = {
            'minute': max_tokens_per_minute,
            'hour': max_tokens_per_hour,
            'day': max_tokens_per_day
        }
        self.usage = {
            'minute': {'tokens': 0, 'reset_time': time.time()},
            'hour': {'tokens': 0, 'reset_time': time.time()},
            'day': {'tokens': 0, 'reset_time': time.time()}
        }
        self.lock = asyncio.Lock()

    async def check_and_update_budget(self, tokens: int) -> bool:
        """
        Check if token usage is within budget and update usage.
        
        Args:
            tokens: Number of tokens to use
            
        Returns:
            bool: True if within budget, False otherwise
        """
        async with self.lock:
            current_time = time.time()
            self._reset_if_needed(current_time)
            
            # Check all time windows
            for window, limit in self.limits.items():
                if self.usage[window]['tokens'] + tokens > limit:
                    log_debug(f"Token budget exceeded for {window} window")
                    return False
                
            # Update usage
            for window in self.usage:
                self.usage[window]['tokens'] += tokens
            
            log_debug(f"Token budget updated: {tokens} tokens used")
            return True

    def _reset_if_needed(self, current_time: float) -> None:
        """Reset usage counters if time window has passed."""
        windows = {
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        
        for window, duration in windows.items():
            if current_time - self.usage[window]['reset_time'] >= duration:
                self.usage[window] = {
                    'tokens': 0,
                    'reset_time': current_time
                }
                log_debug(f"Reset token budget for {window} window")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current budget status for all time windows.
        
        Returns:
            Dict containing usage statistics for each time window
        """
        return {
            window: {
                'used': self.usage[window]['tokens'],
                'remaining': self.limits[window] - self.usage[window]['tokens'],
                'limit': self.limits[window],
                'reset_time': self.usage[window]['reset_time']
            }
            for window in self.limits
        }

class TokenRateManager:
    """
    Combines token management, rate limiting, and budget tracking.
    
    This class provides a unified interface for managing token usage,
    rate limiting, and budget constraints for API requests.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        rate_limit_config: Optional[RateLimitConfig] = None,
        token_budget_config: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the token rate manager.
        
        Args:
            model: The model identifier
            rate_limit_config: Configuration for rate limiting
            token_budget_config: Configuration for token budgets
        """
        self.token_manager = TokenManager(model, rate_limit_config)
        self.token_budget = TokenBudget(**(token_budget_config or {}))
        log_debug(f"TokenRateManager initialized for model: {model}")

    async def process_request(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """
        Process a request with token optimization, rate limiting, and budget checking.
        
        Args:
            text: The text to process
            max_tokens: Maximum allowed tokens
            preserve_sections: Sections to preserve during optimization
            
        Returns:
            Tuple containing optimized text and token usage statistics
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Estimate tokens
        estimated_tokens = self.token_manager.estimate_tokens(text)
        
        # Check token budget
        if not await self.token_budget.check_and_update_budget(estimated_tokens):
            raise RateLimitExceededError("Token budget exceeded")
        
        # Process with rate limiting
        return await self.token_manager.optimize_prompt_with_rate_limit(
            text,
            max_tokens,
            preserve_sections
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of token usage and rate limiting.
        
        Returns:
            Dict containing token usage, rate limiting, and budget statistics
        """
        return {
            'rate_limit_status': self.token_manager.get_rate_limit_status(),
            'token_budget_status': self.token_budget.get_status(),
            'model_config': self.token_manager.model_config
        }

async def test_token_rate_manager():
    """Test the TokenRateManager functionality."""
    try:
        manager = TokenRateManager()
        test_text = "This is a test prompt" * 100
        
        log_info("Starting token rate manager test")
        optimized_text, usage = await manager.process_request(
            test_text,
            max_tokens=1000
        )
        
        log_info(f"Test successful: {usage}")
        status = manager.get_status()
        log_info(f"Current status: {status}")
        
    except Exception as e:
        log_error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_token_rate_manager())
```

This merged version combines token management, rate limiting, and budget tracking into a single cohesive module. Key improvements include:

1. Added `TokenBudget` class for tracking token usage across different time windows
2. Created `TokenRateManager` as a unified interface for all token-related operations
3. Integrated rate limiting with token optimization
4. Added comprehensive status reporting
5. Improved error handling and logging
6. Added async support throughout
7. Maintained backward compatibility with existing functions

The module now provides a complete solution for managing tokens, rate limits, and budgets in Azure OpenAI API requests. Users can either use the individual components (TokenManager, AsyncRateLimiter, TokenBudget) or the unified TokenRateManager for simpler integration.  
[Timestamp: 2024/11/15 06:09:49]

**user**: Take this docstring_generator.py module and merge it into [docs.py](simple2/docs.py.md)

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

[Timestamp: 2024/11/15 06:10:50]

**ai**: Let's merge the `docstring_generator.py` module into the existing `docs.py` module. This will integrate the enhanced docstring generation capabilities directly into the documentation generation system. Here's how you can do it:

```python
#!/usr/bin/env python3
"""
docs.py - Documentation Generation System

This module provides a comprehensive system for generating documentation from Python source code,
including docstring management, markdown generation, and documentation workflow automation.

Classes:
    DocStringManager: Manages docstring operations for source code files.
    DocStringParser: Handles parsing and extraction of docstrings from Python source code.
    DocStringGenerator: Generates docstrings for Python code elements.
    MarkdownGenerator: Generates markdown documentation from Python code elements.
    DocumentationManager: Manages the overall documentation generation process.
    DocstringGenerator: Enhanced docstring generator using Azure OpenAI with structured output.
    DocstringRAG: Implements Retrieval-Augmented Generation for docstrings.
    DocstringMetrics: Tracks and analyzes docstring generation metrics.

Functions:
    main(): Demonstrates usage of the documentation system.
"""

import ast
import logging
import inspect
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import json
import statistics
import time

# Existing classes and functions...

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
        logging.debug("DocstringGenerator initialized.")

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
            logging.error(f"Error generating docstring for {function_info['name']}: {str(e)}")
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
        logging.debug("DocstringRAG initialized")

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
            
            logging.debug(f"Retrieved {len(similar_functions)} similar functions")
            return similar_functions

        except Exception as e:
            logging.error(f"Error retrieving similar functions: {str(e)}")
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
            logging.error(f"Error generating enhanced docstring: {str(e)}")
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
        logging.debug(f"Logged metrics for {function_name}")

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
        
        logging.info(f"Generated metrics summary: {json.dumps(summary, indent=2)}")
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

# The rest of the existing classes and functions...

def main():
    """
    Main function to demonstrate usage of the documentation system.
    Sets up logging, initializes the DocumentationManager, processes directories,
    generates documentation, and saves it to files.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize DocumentationManager
        doc_manager = DocumentationManager(output_dir="generated_docs")

        # Example usage
        source_dir = "."  # Current directory
        docs_map = doc_manager.process_directory(source_dir)

        # Generate and save documentation for each file
        for file_path, content in docs_map.items():
            output_file = Path("generated_docs") / Path(file_path).with_suffix('.md').name
            doc_manager.save_documentation(content, output_file)

        # Generate and save index
        index_content = doc_manager.generate_index(docs_map)
        doc_manager.save_documentation(index_content, "generated_docs/index.md")

        logger.info("Documentation generation completed successfully")

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

## Key Integrations

- **DocstringGenerator**: Integrated into the module to handle enhanced docstring generation using Azure OpenAI.
- **DocstringRAG**: Provides Retrieval-Augmented Generation capabilities for improved docstring quality.
- **DocstringMetrics**: Tracks and analyzes metrics for docstring generation, integrated for comprehensive monitoring.

This merged module now combines the functionalities of docstring generation, management, and documentation into a single, cohesive system.  
[Timestamp: 2024/11/15 06:11:45]

**user**: Merge this docstring_workflow.py into [interaction.py](interaction.py.md)

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

[Timestamp: 2024/11/15 06:12:57]

**ai**: Let's merge the `docstring_workflow.py` module into the existing `interaction.py` module. This will integrate the workflow orchestration directly into the interaction handler, allowing for a more streamlined process management. Here's how you can do it:

```python
"""
Interaction Handler Module

This module manages the orchestration of docstring generation with integrated caching,
monitoring, and error handling. It processes functions in batches and interacts with
the Azure OpenAI API to generate documentation.

Version: 1.2.0
Author: Development Team
"""

import asyncio
import hashlib
import os
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import statistics

from dotenv import load_dotenv
from api_client import AzureOpenAIClient
from docs import DocStringManager
from cache import Cache
from logger import log_info, log_error, log_debug
from monitoring import SystemMonitor
from extraction_manager import ExtractionManager
from docstring_generator import DocstringGenerator, DocstringRAG, DocstringMetrics
from rate_limiter import RateLimitConfig
from azure.search.documents import SearchClient

# Load environment variables from .env file
load_dotenv()

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

class InteractionHandler:
    """
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5,
        search_client: Optional[SearchClient] = None,
        config: Optional[WorkflowConfig] = None
    ):
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            endpoint (Optional[str]): The Azure OpenAI endpoint.
            api_key (Optional[str]): The API key for Azure OpenAI.
            cache_config (Optional[Dict]): Configuration for the cache.
            batch_size (int): Number of functions to process concurrently.
            search_client (Optional[SearchClient]): Search client for RAG.
            config (Optional[WorkflowConfig]): Workflow configuration.
        """
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_client = AzureOpenAIClient(endpoint=endpoint, api_key=api_key)
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        self.config = config or WorkflowConfig()
        self.generator = DocstringGenerator(self.api_client, self.config.rate_limit_config)
        self.rag = DocstringRAG(self.api_client, search_client) if search_client and self.config.enable_rag else None
        self.metrics = WorkflowMetrics()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        log_info("Interaction Handler initialized with batch processing capability")

    async def process_all_functions(self, source_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions.")
        try:
            # Extract metadata using the centralized manager
            metadata = self.extraction_manager.extract_metadata(source_code)
            functions = metadata['functions']
            log_info(f"Extracted {len(functions)} functions from source code.")

            # Process functions in batches
            results = []
            for i in range(0, len(functions), self.batch_size):
                batch = functions[i: i + self.batch_size]
                log_debug(f"Processing batch of functions: {[func['name'] for func in batch]}")
                batch_results = await self.process_batch(batch)
                results.extend(batch_results)

            # Update source code and generate documentation
            manager = DocStringManager(source_code)
            documentation_entries = []

            for function_info, result in zip(functions, results):
                if result and result['docstring']:
                    manager.insert_docstring(function_info["node"], result['docstring'])
                    documentation_entries.append({
                        "function_name": function_info["name"],
                        "complexity_score": result.get("complexity_score", 0),
                        "docstring": result['docstring'],
                        "summary": result.get("summary", ""),
                        "changelog": result.get("changelog", ""),
                    })

            updated_code = manager.update_source_code(documentation_entries)
            documentation = manager.generate_markdown_documentation(documentation_entries)

            # Save the generated markdown documentation
            output_file = self.config.output_dir / "documentation.md"
            manager.save_documentation(documentation, output_file)

            # Log final metrics
            self.monitor.log_batch_completion(len(functions))
            log_info("Batch processing completed successfully.")

            return updated_code, documentation

        except Exception as e:
            log_error(f"Error in batch processing: {str(e)}")
            return None, None

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

    async def process_function(self, source_code: str, function_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function.
            function_info (Dict): Metadata about the function to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata.
        """
        async with self.semaphore:
            func_name = function_info.get('name', 'unknown')
            try:
                start_time = time.time()
                function_node = function_info['node']

                # Use extracted metadata for docstring generation
                response = await self.api_client.get_docstring(
                    func_name=function_info['name'],
                    params=function_info['args'],
                    return_type=function_info['returns'],
                    complexity_score=function_info.get('complexity_score', 0),
                    existing_docstring=function_info['docstring'],
                    decorators=function_info['decorators'],
                    exceptions=function_info.get('exceptions', []),
                    max_tokens=4000,
                    temperature=0.2
                )

                if not response:
                    return None, None

                # Process response
                docstring_data = response['content']

                # Cache the result
                await self.cache.save_docstring(
                    self._generate_cache_key(function_node),
                    {
                        'docstring': docstring_data['docstring'],
                        'metadata': {
                            'summary': docstring_data['summary'],
                            'complexity_score': docstring_data['complexity_score'],
                            'changelog': docstring_data.get('changelog', 'Initial documentation')
                        }
                    }
                )

                # Log metrics
                self.monitor.log_operation_complete(
                    func_name=func_name,
                    duration=time.time() - start_time,
                    tokens=response['usage']['total_tokens']
                )

                log_info(f"Processed function '{func_name}' successfully.")
                return docstring_data['docstring'], docstring_data

            except TimeoutError:
                log_error(f"Timeout occurred while processing function {func_name}.")
                return None, None
            except Exception as e:
                self.monitor.log_error_event(f"Error processing function {func_name}: {str(e)}")
                return None, None

    def _generate_cache_key(self, function_node: ast.FunctionDef) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_node (ast.FunctionDef): The function node to generate a key for.

        Returns:
            str: The generated cache key.
        """
        func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
        return hashlib.md5(func_signature.encode()).hexdigest()

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
```

## Key Integrations

- **DocstringWorkflow**: Integrated into the InteractionHandler to manage the entire docstring generation process.
- **WorkflowConfig and WorkflowMetrics**: Provide configuration and metrics tracking for the workflow.
- **Batch Processing**: Handles functions in batches for efficient processing and resource management.
- **Enhanced Error Handling**: Improved error logging and handling throughout the workflow.

This merged module now combines the functionalities of interaction handling, docstring generation, and workflow management into a single, cohesive system.  
[Timestamp: 2024/11/15 06:14:04]
