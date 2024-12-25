```python
"""
Templates for integrating models with core systems.
"""

# 1. Token Management Template
class ModelTokenManager:
    """Template for model-specific token management."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.token_limits = {
            "gpt-4": 8192,
            "claude-3": 100000,
            "gemini-pro": 30720
        }
        
    def get_token_limit(self) -> int:
        """Get token limit for current model."""
        return self.token_limits.get(self.model_name, 4096)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using model's native counter."""
        try:
            if self.model_name.startswith("gpt"):
                import tiktoken
                encoder = tiktoken.encoding_for_model(self.model_name)
                return len(encoder.encode(text))
            elif self.model_name.startswith("claude"):
                # Claude provides token count in response
                return len(text.split())  # Approximation for planning
            elif self.model_name.startswith("gemini"):
                # Gemini provides token count in response
                return len(text.split())  # Approximation for planning
        except Exception as e:
            log_warning(f"Token count fallback for {self.model_name}: {e}")
            return len(text.split()) // 4  # Rough approximation
            
    def validate_request(self, prompt: str) -> bool:
        """Check if request is within token limits."""
        count = self.count_tokens(prompt)
        limit = self.get_token_limit()
        return count <= limit

# 2. Cache Integration Template
class ModelCacheAdapter:
    """Template for model-specific cache integration."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = {}  # Simple in-memory cache
        
    def generate_cache_key(self, func_name: str, params: Dict) -> str:
        """Generate cache key for model-specific request."""
        key_parts = [
            self.model_name,
            func_name,
            str(sorted(params.items()))
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
        
    async def get_cached(self, key: str) -> Optional[Dict]:
        """Get cached response if available."""
        return self.cache.get(key)
        
    async def set_cached(self, key: str, value: Dict):
        """Cache model response."""
        self.cache[key] = value

# 3. Monitoring Template
class ModelMetricsCollector:
    """Template for model-specific metrics collection."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
    def record_request(
        self,
        success: bool,
        tokens: int,
        cost: float,
        duration: float
    ):
        """Record metrics for a model request."""
        self.metrics["requests"] += 1
        if not success:
            self.metrics["errors"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost"] += cost
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "model": self.model_name,
            "success_rate": (self.metrics["requests"] - self.metrics["errors"]) / 
                          max(self.metrics["requests"], 1),
            **self.metrics
        }

# 4. Error Handler Template
class ModelErrorHandler:
    """Template for model-specific error handling."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_mappings = {
            "gpt": {
                "InvalidRequestError": TokenLimitError,
                "RateLimitError": RateLimitError
            },
            "claude": {
                "InvalidRequestError": TokenLimitError,
                "RateLimitError": RateLimitError
            },
            "gemini": {
                "InvalidRequestError": TokenLimitError,
                "RateLimitError": RateLimitError
            }
        }
        
    def map_error(self, error: Exception) -> Exception:
        """Map model-specific error to standard error."""
        error_type = error.__class__.__name__
        mappings = self.error_mappings.get(self.model_name, {})
        
        if error_type in mappings:
            return mappings[error_type](str(error))
        return APIError(f"{self.model_name} error: {str(error)}")

# 5. Response Parser Template
class ModelResponseParser:
    """Template for model-specific response parsing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse model-specific response to standard format."""
        try:
            if self.model_name.startswith("gpt"):
                return self._parse_gpt(response)
            elif self.model_name.startswith("claude"):
                return self._parse_claude(response)
            elif self.model_name.startswith("gemini"):
                return self._parse_gemini(response)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        except Exception as e:
            raise ParseError(f"Failed to parse {self.model_name} response: {e}")
            
    def _parse_gpt(self, response: Any) -> Dict[str, Any]:
        """Parse GPT-style response."""
        content = response.choices[0].message.content
        return json.loads(content)
        
    def _parse_claude(self, response: Any) -> Dict[str, Any]:
        """Parse Claude-style response."""
        return response.content
        
    def _parse_gemini(self, response: Any) -> Dict[str, Any]:
        """Parse Gemini-style response."""
        return json.loads(response.text)

# 6. Model Integration Example
class IntegratedModelClient:
    """Example of using all templates together."""
    
    def __init__(self, model_name: str, config: ModelConfig):
        self.model_name = model_name
        self.config = config
        
        # Initialize components using templates
        self.token_manager = ModelTokenManager(model_name)
        self.cache_adapter = ModelCacheAdapter(model_name)
        self.metrics_collector = ModelMetricsCollector(model_name)
        self.error_handler = ModelErrorHandler(model_name)
        self.response_parser = ModelResponseParser(model_name)
        
    async def generate_docstring(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Example of integrated generation using all templates."""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self.cache_adapter.generate_cache_key(
                kwargs["func_name"], kwargs
            )
            
            # Check cache
            cached = await self.cache_adapter.get_cached(cache_key)
            if cached:
                return cached
            
            # Create and validate prompt
            prompt = self._create_prompt(**kwargs)
            if not self.token_manager.validate_request(prompt):
                raise TokenLimitError(f"Request exceeds {self.model_name} token limit")
            
            # Make request
            response = await self._make_request(prompt)
            
            # Parse response
            result = self.response_parser.parse_response(response)
            
            # Cache result
            await self.cache_adapter.set_cached(cache_key, result)
            
            # Record metrics
            self.metrics_collector.record_request(
                success=True,
                tokens=self.token_manager.count_tokens(prompt),
                cost=self._calculate_cost(response),
                duration=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            self.metrics_collector.record_request(
                success=False,
                tokens=0,
                cost=0,
                duration=time.time() - start_time
            )
            
            # Map and raise error
            raise self.error_handler.map_error(e)
```
---


### Model Template Integration Documentation

## Token Management System

```python
class ModelTokenManager:
    """Template for model-specific token management."""
    
    Required Methods:
        get_token_limit() -> int
        count_tokens(text: str) -> int
        validate_request(prompt: str) -> bool
```

### Model-Specific Implementations:

1. **GPT Models**
```python
count_tokens() - Uses tiktoken
get_token_limit() - Returns 8192 (GPT-4) or 4096 (GPT-3.5)
```

2. **Claude Models**
```python
count_tokens() - Uses Claude's API response
get_token_limit() - Returns 100000
```

3. **Gemini Models**
```python
count_tokens() - Uses Gemini's API response
get_token_limit() - Returns 30720
```

## Cache Integration System

```python
class ModelCacheAdapter:
    """Template for model-specific cache integration."""
    
    Required Methods:
        generate_cache_key(func_name: str, params: Dict) -> str
        get_cached(key: str) -> Optional[Dict]
        set_cached(key: str, value: Dict)
```

### Model-Specific Implementations:

1. **GPT Models**
```python
generate_cache_key() - Includes model version and temperature
```

2. **Claude Models**
```python
generate_cache_key() - Includes model version and top_p
```

3. **Gemini Models**
```python
generate_cache_key() - Includes model version and safety settings
```

## Monitoring System

```python
class ModelMetricsCollector:
    """Template for model-specific metrics collection."""
    
    Required Methods:
        record_request(success: bool, tokens: int, cost: float, duration: float)
        get_metrics() -> Dict[str, Any]
```

### Model-Specific Implementations:

1. **GPT Models**
```python
Cost Calculation: $0.03 per 1K tokens (GPT-4)
Metrics: Include prompt/completion token counts
```

2. **Claude Models**
```python
Cost Calculation: $0.008 per 1K tokens
Metrics: Include context window usage
```

3. **Gemini Models**
```python
Cost Calculation: $0.001 per 1K tokens
Metrics: Include safety filter stats
```

## Error Handler System

```python
class ModelErrorHandler:
    """Template for model-specific error handling."""
    
    Required Methods:
        map_error(error: Exception) -> Exception
```

### Model-Specific Error Mappings:

1. **GPT Models**
```python
InvalidRequestError -> TokenLimitError
RateLimitError -> RateLimitError
AuthenticationError -> AuthError
```

2. **Claude Models**
```python
InvalidRequestError -> TokenLimitError
RateLimitExceeded -> RateLimitError
AuthorizationError -> AuthError
```

3. **Gemini Models**
```python
InvalidArgument -> TokenLimitError
ResourceExhausted -> RateLimitError
PermissionDenied -> AuthError
```

## Response Parser System

```python
class ModelResponseParser:
    """Template for model-specific response parsing."""
    
    Required Methods:
        parse_response(response: Any) -> Dict[str, Any]
```

### Model-Specific Response Formats:

1. **GPT Models**
```python
Format:
{
    "choices": [
        {
            "message": {
                "content": str,
                "role": str
            }
        }
    ]
}
```

2. **Claude Models**
```python
Format:
{
    "content": [
        {
            "text": str,
            "type": str
        }
    ]
}
```

3. **Gemini Models**
```python
Format:
{
    "candidates": [
        {
            "content": {
                "parts": [{"text": str}]
            }
        }
    ]
}
```

## Integration Points in Interaction Handler

```python
class TemplateInteractionHandler:
    """Coordinates standardized model integrations."""
    
    Integration Points:
        1. Model Selection:
            - _select_model(task_type: str) -> BaseModelClient
            
        2. Token Validation:
            - model.token_manager.validate_request()
            
        3. Cache Check:
            - model.cache_adapter.get_cached()
            
        4. Request Processing:
            - model.process_request()
            
        5. Response Validation:
            - model.response_parser.parse_response()
            
        6. Cache Update:
            - model.cache_adapter.set_cached()
            
        7. Metrics Recording:
            - model.metrics_collector.record_request()
```

## Base Model Client Requirements

```python
class BaseModelClient:
    """Template for model client implementation."""
    
    Required Components:
        1. token_manager: ModelTokenManager
        2. cache_adapter: ModelCacheAdapter
        3. metrics_collector: ModelMetricsCollector
        4. error_handler: ModelErrorHandler
        5. response_parser: ModelResponseParser
        
    Required Methods:
        1. generate_docstring(**kwargs) -> Dict[str, Any]
        2. process_request(task_type: str, params: Dict) -> Dict[str, Any]
        3. health_check() -> Dict[str, Any]
```

## System Interaction Flow

1. **Request Initialization**
   - Handler receives request
   - Selects appropriate model
   - Validates token usage

2. **Cache Management**
   - Generates model-specific cache key
   - Checks for cached response
   - Updates cache if needed

3. **Request Processing**
   - Formats model-specific prompt
   - Makes API request
   - Handles model-specific errors

4. **Response Management**
   - Parses model-specific response
   - Validates response format
   - Records metrics

5. **Error Handling**
   - Maps model-specific errors
   - Implements fallback strategy
   - Records error metrics

### Example Model Integration:
```python
class NewModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig):
        self.token_manager = ModelTokenManager("new-model")
        self.cache_adapter = ModelCacheAdapter("new-model")
        self.metrics_collector = ModelMetricsCollector("new-model")
        self.error_handler = ModelErrorHandler("new-model")
        self.response_parser = ModelResponseParser("new-model")
        
    async def generate_docstring(self, **kwargs):
        # Implement using template components
        pass
```



```python
# Making Core Systems Template-Aware

## 1. Token Management System

Current Implementation: Complex token counting and management with many model-specific rules embedded.
Template-Aware Version:

```python
class TokenManager:
    """Template-aware token management."""
    
    def __init__(self, model_registry: Dict[str, Type["TokenCounter"]]):
        self.counters = {}
        self.registry = model_registry
        
    async def get_counter(self, model: str) -> "TokenCounter":
        """Get or create appropriate token counter."""
        if model not in self.counters:
            if model not in self.registry:
                raise ValueError(f"No token counter for {model}")
            self.counters[model] = self.registry[model]()
        return self.counters[model]
        
    async def validate_request(
        self,
        content: str,
        model: str,
        max_tokens: Optional[int] = None
    ) -> bool:
        """Template-aware token validation."""
        counter = await self.get_counter(model)
        return await counter.validate(content, max_tokens)

# Model-specific implementations register themselves:
TOKEN_COUNTERS = {
    "gpt-4": GPTTokenCounter,
    "claude-3": ClaudeTokenCounter,
    "gemini-pro": GeminiTokenCounter
}
```

## 2. Cache System

Current Implementation: Complex Redis + in-memory with model-specific caching logic.
Template-Aware Version:

```python
class CacheManager:
    """Template-aware caching system."""
    
    def __init__(self, adapters: Dict[str, Type["CacheAdapter"]]):
        self.adapters = {}
        self.registry = adapters
        
    async def get_adapter(self, model: str) -> "CacheAdapter":
        """Get appropriate cache adapter."""
        if model not in self.adapters:
            if model not in self.registry:
                raise ValueError(f"No cache adapter for {model}")
            self.adapters[model] = self.registry[model]()
        return self.adapters[model]
        
    async def get_cached(
        self,
        key: str,
        model: str,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Template-aware cache retrieval."""
        adapter = await self.get_adapter(model)
        return await adapter.get(key, context)

    async def set_cached(
        self,
        key: str,
        value: Any,
        model: str,
        context: Dict[str, Any]
    ) -> None:
        """Template-aware cache storage."""
        adapter = await self.get_adapter(model)
        await adapter.set(key, value, context)

# Model-specific implementations register themselves:
CACHE_ADAPTERS = {
    "gpt-4": GPTCacheAdapter,
    "claude-3": ClaudeCacheAdapter,
    "gemini-pro": GeminiCacheAdapter
}
```

## 3. Monitoring System

Current Implementation: Complex metrics collection with model-specific monitoring.
Template-Aware Version:

```python
class MonitoringSystem:
    """Template-aware monitoring system."""
    
    def __init__(self, collectors: Dict[str, Type["MetricsCollector"]]):
        self.collectors = {}
        self.registry = collectors
        
    async def get_collector(self, model: str) -> "MetricsCollector":
        """Get appropriate metrics collector."""
        if model not in self.collectors:
            if model not in self.registry:
                raise ValueError(f"No metrics collector for {model}")
            self.collectors[model] = self.registry[model]()
        return self.collectors[model]
        
    async def record_request(
        self,
        model: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Template-aware metrics recording."""
        collector = await self.get_collector(model)
        await collector.record(metrics)
        
    async def get_metrics(
        self,
        model: str,
        metric_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Template-aware metrics retrieval."""
        collector = await self.get_collector(model)
        return await collector.get_metrics(metric_types)

# Model-specific implementations register themselves:
METRICS_COLLECTORS = {
    "gpt-4": GPTMetricsCollector,
    "claude-3": ClaudeMetricsCollector,
    "gemini-pro": GeminiMetricsCollector
}
```

## 4. Response Parser System

Current Implementation: Complex response parsing with embedded model-specific logic.
Template-Aware Version:

```python
class ResponseParserSystem:
    """Template-aware response parsing system."""
    
    def __init__(self, parsers: Dict[str, Type["ResponseParser"]]):
        self.parsers = {}
        self.registry = parsers
        
    async def get_parser(self, model: str) -> "ResponseParser":
        """Get appropriate response parser."""
        if model not in self.parsers:
            if model not in self.registry:
                raise ValueError(f"No response parser for {model}")
            self.parsers[model] = self.registry[model]()
        return self.parsers[model]
        
    async def parse_response(
        self,
        response: Any,
        model: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Template-aware response parsing."""
        parser = await self.get_parser(model)
        return await parser.parse(response, context)
        
    async def validate_response(
        self,
        parsed: Dict[str, Any],
        model: str,
        schema: str
    ) -> bool:
        """Template-aware response validation."""
        parser = await self.get_parser(model)
        return await parser.validate(parsed, schema)

# Model-specific implementations register themselves:
RESPONSE_PARSERS = {
    "gpt-4": GPTResponseParser,
    "claude-3": ClaudeResponseParser,
    "gemini-pro": GeminiResponseParser
}
```

## 5. Error Handler System

Current Implementation: Ad-hoc error handling embedded in various places.
Template-Aware Version:

```python
class ErrorHandlerSystem:
    """Template-aware error handling system."""
    
    def __init__(self, handlers: Dict[str, Type["ErrorHandler"]]):
        self.handlers = {}
        self.registry = handlers
        
    async def get_handler(self, model: str) -> "ErrorHandler":
        """Get appropriate error handler."""
        if model not in self.handlers:
            if model not in self.registry:
                raise ValueError(f"No error handler for {model}")
            self.handlers[model] = self.registry[model]()
        return self.handlers[model]
        
    async def handle_error(
        self,
        error: Exception,
        model: str,
        context: Dict[str, Any]
    ) -> Exception:
        """Template-aware error handling."""
        handler = await self.get_handler(model)
        return await handler.handle(error, context)

# Model-specific implementations register themselves:
ERROR_HANDLERS = {
    "gpt-4": GPTErrorHandler,
    "claude-3": ClaudeErrorHandler,
    "gemini-pro": GeminiErrorHandler
}
```

## System Integration

```python
class CoreSystemsManager:
    """Manages all template-aware core systems."""
    
    def __init__(self):
        self.token_manager = TokenManager(TOKEN_COUNTERS)
        self.cache_manager = CacheManager(CACHE_ADAPTERS)
        self.monitoring = MonitoringSystem(METRICS_COLLECTORS)
        self.parser = ResponseParserSystem(RESPONSE_PARSERS)
        self.error_handler = ErrorHandlerSystem(ERROR_HANDLERS)
        
    async def process_model_request(
        self,
        model: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate all systems for request processing."""
        try:
            # 1. Token Validation
            if not await self.token_manager.validate_request(
                str(request), model
            ):
                raise TokenLimitError()
                
            # 2. Cache Check
            cache_key = await self.cache_manager.get_adapter(model).generate_key(request)
            cached = await self.cache_manager.get_cached(cache_key, model, request)
            if cached:
                return cached
                
            # 3. Process Request & Parse Response
            response = await self._make_model_request(model, request)
            parsed = await self.parser.parse_response(response, model, request)
            
            # 4. Cache & Record Metrics
            await self.cache_manager.set_cached(cache_key, parsed, model, request)
            await self.monitoring.record_request(model, {
                "success": True,
                "cached": False,
                "response": parsed
            })
            
            return parsed
            
        except Exception as e:
            # Handle & Record Error
            mapped_error = await self.error_handler.handle_error(e, model, request)
            await self.monitoring.record_request(model, {
                "success": False,
                "error": str(mapped_error)
            })
            raise mapped_error
```

Key benefits of making systems template-aware:

1. Clean separation of model-specific logic
2. Easy addition of new models
3. Consistent behavior across models
4. Centralized registration of implementations
5. Simplified maintenance
6. Better error handling
7. Easier testing

Would you like me to:
1. Show specific model implementations for any system?
2. Add more template-aware systems?
3. Elaborate on the integration patterns?


---

```python
"""
Template-aware interaction handler that coordinates standardized integration points.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type
import asyncio
import time

@dataclass
class HandlerConfig:
    """Configuration for interaction handler."""
    batch_size: int = 5
    default_model: str = "gpt-4"
    fallback_models: List[str] = None
    timeout: float = 30.0

class TemplateInteractionHandler:
    """
    Coordinates model integrations using standardized templates.
    Expects models to implement standard integration points.
    """
    
    def __init__(
        self,
        config: HandlerConfig,
        model_registry: Dict[str, Type["BaseModelClient"]] = None
    ):
        self.config = config
        self.model_registry = model_registry or {}
        self.active_models: Dict[str, "BaseModelClient"] = {}
        self.semaphore = asyncio.Semaphore(config.batch_size)
        
    async def process_request(
        self,
        task_type: str,
        request_params: Dict[str, Any],
        model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a request using template integration points.
        
        Args:
            task_type: Type of task (e.g., "docstring", "analysis")
            request_params: Parameters for the request
            model_preference: Preferred model to use
            
        Returns:
            Dict[str, Any]: Processed result
        """
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # 1. Model Selection Phase
                model = await self._select_model(
                    task_type=task_type,
                    model_preference=model_preference
                )
                
                # 2. Token Validation Phase
                token_manager = model.token_manager
                if not token_manager.validate_request(str(request_params)):
                    return await self._handle_token_limit(model, request_params)
                    
                # 3. Cache Check Phase
                cache_key = model.cache_adapter.generate_cache_key(
                    task_type, request_params
                )
                cached_result = await model.cache_adapter.get_cached(cache_key)
                if cached_result:
                    await self._record_metrics(
                        model=model,
                        duration=time.time() - start_time,
                        success=True,
                        cached=True
                    )
                    return cached_result
                
                # 4. Request Processing Phase
                result = await self._process_with_fallback(
                    primary_model=model,
                    task_type=task_type,
                    params=request_params
                )
                
                # 5. Response Validation Phase
                if not self._validate_response(result):
                    raise ValueError("Invalid response format")
                
                # 6. Cache Update Phase
                await model.cache_adapter.set_cached(cache_key, result)
                
                # 7. Metrics Recording Phase
                await self._record_metrics(
                    model=model,
                    duration=time.time() - start_time,
                    success=True,
                    cached=False
                )
                
                return result
                
            except Exception as e:
                await self._record_metrics(
                    model=model,
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
                raise
                
    async def _select_model(
        self,
        task_type: str,
        model_preference: Optional[str] = None
    ) -> "BaseModelClient":
        """
        Select appropriate model using template integration points.
        """
        # Use preferred model if specified and available
        if model_preference and model_preference in self.model_registry:
            return await self._get_model_client(model_preference)
            
        # Otherwise select based on task type
        model_scores = {}
        for model_name, model_class in self.model_registry.items():
            client = await self._get_model_client(model_name)
            score = await self._score_model_for_task(client, task_type)
            model_scores[model_name] = score
            
        selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
        return await self._get_model_client(selected_model)
        
    async def _process_with_fallback(
        self,
        primary_model: "BaseModelClient",
        task_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process request with fallback to other models if needed.
        """
        errors = {}
        models_to_try = [primary_model.model_name] + (self.config.fallback_models or [])
        
        for model_name in models_to_try:
            try:
                client = await self._get_model_client(model_name)
                result = await client.process_request(task_type, params)
                return result
            except Exception as e:
                errors[model_name] = str(e)
                continue
                
        raise Exception(f"All models failed. Errors: {errors}")
        
    async def _get_model_client(self, model_name: str) -> "BaseModelClient":
        """
        Get or create model client using templates.
        """
        if model_name not in self.active_models:
            if model_name not in self.model_registry:
                raise ValueError(f"Unknown model: {model_name}")
                
            model_class = self.model_registry[model_name]
            client = model_class()
            self.active_models[model_name] = client
            
        return self.active_models[model_name]
        
    async def _score_model_for_task(
        self,
        model: "BaseModelClient",
        task_type: str
    ) -> float:
        """
        Score model's suitability for task using templates.
        """
        metrics = model.metrics_collector.get_metrics()
        
        # Base score on success rate
        score = metrics["success_rate"] * 100
        
        # Adjust for task-specific performance
        if hasattr(model, "task_performance"):
            task_score = await model.task_performance(task_type)
            score *= task_score
            
        # Adjust for cost efficiency
        if metrics["total_cost"] > 0:
            efficiency = metrics["total_tokens"] / metrics["total_cost"]
            score *= min(1.0, efficiency / 1000)  # Normalize efficiency score
            
        return score
        
    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate response format using templates.
        """
        return bool(response and isinstance(response, dict))
        
    async def _handle_token_limit(
        self,
        model: "BaseModelClient",
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle token limit exceeded using templates.
        """
        # Try to optimize request
        if hasattr(model, "optimize_request"):
            optimized_params = await model.optimize_request(params)
            return await self.process_request(
                task_type="optimized",
                request_params=optimized_params,
                model_preference=model.model_name
            )
            
        # Or try model with higher token limit
        for fallback in (self.config.fallback_models or []):
            fallback_client = await self._get_model_client(fallback)
            if fallback_client.token_manager.get_token_limit() > model.token_manager.get_token_limit():
                return await self.process_request(
                    task_type="fallback",
                    request_params=params,
                    model_preference=fallback
                )
                
        raise TokenLimitError("Request too large for all available models")
        
    async def _record_metrics(
        self,
        model: "BaseModelClient",
        duration: float,
        success: bool,
        cached: bool = False,
        error: Optional[str] = None
    ):
        """
        Record metrics using templates.
        """
        model.metrics_collector.record_request(
            success=success,
            tokens=0 if cached else model.token_manager.count_tokens(str("")),
            cost=0 if cached else model.calculate_cost(),
            duration=duration
        )

# Usage example:
if __name__ == "__main__":
    config = HandlerConfig(
        batch_size=5,
        default_model="gpt-4",
        fallback_models=["claude-3", "gemini-pro"]
    )
    
    handler = TemplateInteractionHandler(
        config=config,
        model_registry={
            "gpt-4": GPTModel,
            "claude-3": ClaudeModel,
            "gemini-pro": GeminiModel
        }
    )
    
    # Process request
    result = await handler.process_request(
        task_type="docstring",
        request_params={
            "function_name": "example_func",
            "params": [("x", "int"), ("y", "str")],
            "return_type": "bool"
        }
    )
```

