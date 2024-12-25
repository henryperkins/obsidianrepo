```python
# api_rate_limiter.py

from typing import Dict, Optional, DefaultDict
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
from core.logger import LoggerSetup

@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class WindowStats:
    """Statistics for a time window."""
    request_count: int = 0
    token_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, wait_time: float):
        self.wait_time = wait_time
        super().__init__(message)

class EnhancedRateLimiter:
    """Thread-safe rate limiter with window tracking."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = LoggerSetup.get_logger("rate_limiter")
        self._lock = asyncio.Lock()  # Synchronization lock
        self._windows: DefaultDict[str, WindowStats] = defaultdict(WindowStats)
        self._global_window = WindowStats()
        
    async def acquire(self, tokens: int = 0, namespace: str = "default") -> None:
        """
        Acquire rate limit permission with proper synchronization.
        
        Args:
            tokens: Estimated token usage
            namespace: Rate limit namespace
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        async with self._lock:  # Thread-safe access to shared state
            try:
                # Check and update both global and namespace windows
                await self._check_and_update_window(self._global_window, tokens)
                if namespace != "default":
                    await self._check_and_update_window(
                        self._windows[namespace],
                        tokens
                    )
                
                # Record successful acquisition
                await self._record_acquisition(namespace, tokens)
                
            except RateLimitExceededError as e:
                self.logger.warning(
                    f"Rate limit exceeded for namespace {namespace}. "
                    f"Waiting {e.wait_time:.2f}s"
                )
                raise
            except Exception as e:
                self.logger.error(f"Error in rate limiter: {str(e)}")
                raise
    
    async def _check_and_update_window(
        self,
        window: WindowStats,
        tokens: int
    ) -> None:
        """
        Check and update rate limit window with atomic operations.
        
        Args:
            window: Window to check and update
            tokens: Number of tokens to acquire
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        now = datetime.now()
        window_age = (now - window.last_reset).total_seconds()
        
        # Reset window if expired
        if window_age >= 60:
            window.request_count = 0
            window.token_count = 0
            window.last_reset = now
            return
        
        # Check limits
        if (window.request_count >= self.config.requests_per_minute or
            window.token_count + tokens > self.config.tokens_per_minute):
            wait_time = 60 - window_age
            raise RateLimitExceededError(
                "Rate limit exceeded",
                wait_time=wait_time
            )
        
        # Update counters atomically
        window.request_count += 1
        window.token_count += tokens
    
    async def _record_acquisition(self, namespace: str, tokens: int) -> None:
        """Record successful rate limit acquisition with metrics."""
        try:
            await self.metrics.record_rate_limit_acquisition(
                namespace=namespace,
                tokens=tokens,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.error(f"Failed to record metrics: {str(e)}")
    
    async def wait_if_needed(
        self,
        tokens: int = 0,
        namespace: str = "default"
    ) -> None:
        """
        Wait if rate limit is exceeded, with retries.
        
        Args:
            tokens: Estimated token usage
            namespace: Rate limit namespace
        """
        for attempt in range(self.config.max_retries):
            try:
                await self.acquire(tokens, namespace)
                return
            except RateLimitExceededError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                wait_time = e.wait_time + (self.config.retry_delay * attempt)
                self.logger.info(
                    f"Rate limit exceeded, waiting {wait_time:.2f}s "
                    f"(attempt {attempt + 1}/{self.config.max_retries})"
                )
                await asyncio.sleep(wait_time)
    
    async def get_window_stats(
        self,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for rate limit window.
        
        Args:
            namespace: Optional namespace to get stats for
            
        Returns:
            Dict[str, Any]: Window statistics
        """
        async with self._lock:
            if namespace and namespace in self._windows:
                window = self._windows[namespace]
            else:
                window = self._global_window
            
            now = datetime.now()
            window_age = (now - window.last_reset).total_seconds()
            
            return {
                "request_count": window.request_count,
                "token_count": window.token_count,
                "window_age": window_age,
                "requests_remaining": max(
                    0,
                    self.config.requests_per_minute - window.request_count
                ),
                "tokens_remaining": max(
                    0,
                    self.config.tokens_per_minute - window.token_count
                )
            }
    
    async def reset_window(self, namespace: Optional[str] = None) -> None:
        """
        Reset rate limit window with proper synchronization.
        
        Args:
            namespace: Optional namespace to reset
        """
        async with self._lock:
            if namespace:
                if namespace in self._windows:
                    self._windows[namespace] = WindowStats()
            else:
                self._global_window = WindowStats()
                self._windows.clear()

# Service-specific rate limiters configuration
DEFAULT_RATE_LIMITS = {
    "azure": RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=90000
    ),
    "openai": RateLimitConfig(
        requests_per_minute=50,
        tokens_per_minute=80000
    ),
    "anthropic": RateLimitConfig(
        requests_per_minute=40,
        tokens_per_minute=100000
    )
}
```