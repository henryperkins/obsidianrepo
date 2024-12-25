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