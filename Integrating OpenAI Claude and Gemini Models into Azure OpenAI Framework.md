Based on the provided code, I'll help you understand the key changes and additions made to support Gemini, Claude, and OpenAI models alongside the existing Azure OpenAI implementation.

Here's a summary of the main components and their interactions:

1. **Model Factory Pattern**:
```python
class AIClientFactory:
    @staticmethod
    def create_client(config: AIModelConfig) -> Optional[BaseAIClient]:
        """Create appropriate AI client based on configuration."""
        if config.model_type == "azure":
            return AzureOpenAIClient(config)
        elif config.model_type == "claude":
            return ClaudeClient(config)
        elif config.model_type == "gemini":
            return GeminiClient(config)
        elif config.model_type == "openai":
            return OpenAIClient(config)
```

2. **Base Client Interface**:
```python
class BaseAIClient(ABC):
    @abstractmethod
    async def generate_docstring(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def close(self):
        pass
```

3. **Model-Specific Configurations**:
```python
@dataclass
class OpenAIConfig(AIModelConfig):
    api_key: str
    organization_id: Optional[str] = None
    model_name: str = "gpt-4"
    # ...

@dataclass
class GeminiConfig(AIModelConfig):
    api_key: str
    project_id: Optional[str] = None
    model_name: str = "gemini-pro"
    # ...
```

Key Features Added:

1. **Unified Interface**: All model clients implement the same base interface, making them interchangeable.

2. **Model-Specific Features**:
   - OpenAI: Supports streaming and structured JSON output
   - Gemini: Includes document processing capabilities
   - Claude: Offers JSON mode and contextual embeddings

3. **Error Handling & Retries**:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def generate_docstring(self, **kwargs):
    # Implementation
```

4. **Monitoring Integration**:
```python
class SystemMonitor:
    def log_api_request(self, endpoint: str, tokens: int, ...):
        # Track metrics for all models
```

To use different models:

```python
# Create configuration
config = GeminiConfig.from_env()  # or OpenAIConfig, ClaudeConfig

# Create client using factory
client = AIClientFactory.create_client(config)

# Generate docstring
docstring = await client.generate_docstring(
    func_name="example_function",
    params=[("param1", "str"), ("param2", "int")],
    return_type="bool",
    complexity_score=2
)
```

The monitoring system tracks metrics across all models:
```python
monitor = SystemMonitor()
monitor.log_api_request(
    endpoint="gemini/generate",
    tokens=100,
    response_time=0.5,
    status="success",
    prompt_tokens=50,
    completion_tokens=50,
    estimated_cost=0.001
)
```

Would you like me to explain any specific aspect in more detail?

I'll identify the key integration points for the new models (OpenAI, Gemini, and Claude) in the existing system:

1. **Configuration Integration Points**:
```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIModelConfig:
    """Base configuration class for all AI models"""
    model_type: str
    max_tokens: int
    temperature: float
    request_timeout: int
    max_retries: int
    retry_delay: int

# Add model configs to existing configuration system
MODEL_CONFIGS = {
    "azure": AzureOpenAIConfig,
    "openai": OpenAIConfig,
    "gemini": GeminiConfig,
    "claude": ClaudeConfig
}

def get_model_config(model_type: str) -> AIModelConfig:
    """Factory method for model configurations"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_CONFIGS[model_type].from_env()
```

2. **Token Management Integration**:
```python
# token_management.py
class TokenManager:
    # Add new model token limits
    MODEL_LIMITS.update({
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06
        },
        "gemini-pro": {
            "max_tokens": 30720,
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002
        },
        "claude-2": {
            "max_tokens": 100000,
            "cost_per_1k_prompt": 0.008,
            "cost_per_1k_completion": 0.024
        }
    })

    def get_model_tokenizer(self, model_type: str):
        """Get appropriate tokenizer for model"""
        tokenizers = {
            "openai": "tiktoken",
            "azure": "tiktoken",
            "gemini": "sentencepiece",
            "claude": "claude_tokenizer"
        }
        return self._initialize_tokenizer(tokenizers[model_type])
```

3. **Monitoring Integration**:
```python
# monitoring.py
class SystemMonitor:
    def __init__(self):
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }

    def log_model_specific_metrics(
        self,
        model_type: str,
        operation: str,
        metrics: Dict[str, Any]
    ):
        """Log metrics specific to each model type"""
        if model_type not in self.metrics_by_model:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.metrics_by_model[model_type][operation] = metrics
```

4. **Cache Integration**:
```python
# cache.py
class Cache:
    def generate_cache_key(
        self,
        model_type: str,
        function_name: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key with model type"""
        key_components = [
            model_type,
            function_name,
            hashlib.md5(str(params).encode()).hexdigest()
        ]
        return ":".join(key_components)
```

5. **Main Workflow Integration**:
```python
# main.py
async def process_file(file_path: str, args: argparse.Namespace) -> None:
    """Update main processing function to support multiple models"""
    
    # Get model configuration
    model_config = get_model_config(args.model_type)
    
    # Create client using factory
    client = AIClientFactory.create_client(model_config)
    
    # Initialize components with model awareness
    token_manager = TokenManager(model_type=args.model_type)
    monitor = SystemMonitor()
    cache = Cache()

    try:
        async with client:
            # Process file with selected model
            result = await client.generate_docstring(...)
            
            # Log model-specific metrics
            monitor.log_model_specific_metrics(
                model_type=args.model_type,
                operation="generate_docstring",
                metrics=result.get("metrics", {})
            )
    except Exception as e:
        log_error(f"Error processing with {args.model_type}: {e}")
```

6. **Command Line Integration**:
```python
# main.py
def setup_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["azure", "openai", "gemini", "claude"],
        default="azure",
        help="AI model to use for docstring generation"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration file"
    )
    return parser
```

7. **Error Handling Integration**:
```python
# exceptions.py
class ModelError(Exception):
    """Base class for model-specific errors"""
    def __init__(self, model_type: str, message: str):
        self.model_type = model_type
        super().__init__(f"{model_type} error: {message}")

class TokenLimitError(ModelError):
    """Error for token limit exceeded"""
    pass

class ModelAPIError(ModelError):
    """Error for API-related issues"""
    pass
```

8. **Response Parser Integration**:
```python
# response_parser.py
class ResponseParser:
    """Unified response parsing for all models"""
    
    def parse_response(
        self,
        model_type: str,
        response: Any
    ) -> Dict[str, Any]:
        """Parse response based on model type"""
        parsers = {
            "azure": self._parse_azure_response,
            "openai": self._parse_openai_response,
            "gemini": self._parse_gemini_response,
            "claude": self._parse_claude_response
        }
        
        if model_type not in parsers:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return parsers[model_type](response)
```

9. **Documentation Generator Integration**:
```python
# docs.py
class DocStringManager:
    def format_docstring(
        self,
        model_type: str,
        content: Dict[str, Any]
    ) -> str:
        """Format docstring based on model type"""
        formatters = {
            "azure": self._format_azure_docstring,
            "openai": self._format_openai_docstring,
            "gemini": self._format_gemini_docstring,
            "claude": self._format_claude_docstring
        }
        return formatters[model_type](content)
```

These integration points ensure that:
- All models can be configured consistently
- Token management works across different models
- Monitoring captures model-specific metrics
- Caching works with all models
- Error handling is consistent
- Response parsing is standardized
- Documentation formatting is unified

Would you like me to elaborate on any of these integration points?

'll provide more detailed context for each integration point:

1. **Configuration Integration**:
```python
# config.py
"""
Configuration system for multiple AI models.
Provides base configuration class and model-specific configurations
with environment variable loading and validation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os

@dataclass
class AIModelConfig:
    """Base configuration for all AI models with common parameters."""
    model_type: str
    max_tokens: int
    temperature: float
    request_timeout: int
    max_retries: int
    retry_delay: int
    
    @classmethod
    def from_env(cls) -> "AIModelConfig":
        """Load configuration from environment variables."""
        return cls(
            model_type=os.getenv("AI_MODEL_TYPE", "azure"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2"))
        )

    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Add validation logic
        pass

# Model-specific configurations inherit from base
@dataclass
class OpenAIConfig(AIModelConfig):
    """OpenAI-specific configuration parameters."""
    api_key: str
    organization_id: Optional[str] = None
    model_name: str = "gpt-4"
    # Add OpenAI specific parameters

@dataclass
class GeminiConfig(AIModelConfig):
    """Google Gemini-specific configuration."""
    api_key: str
    project_id: Optional[str] = None
    model_name: str = "gemini-pro"
    # Add Gemini specific parameters

# Configuration factory and registry
MODEL_CONFIGS: Dict[str, type] = {
    "azure": AzureOpenAIConfig,
    "openai": OpenAIConfig,
    "gemini": GeminiConfig,
    "claude": ClaudeConfig
}
```

2. **Token Management Integration**:
```python
# token_management.py
"""
Token management system supporting multiple AI models.
Handles token counting, cost calculation, and rate limiting
for different model types.
"""

from typing import Dict, Any, Optional
import tiktoken
import sentencepiece
from anthropic import Anthropic

class TokenManager:
    """Manages tokens across different AI models."""
    
    # Token limits and pricing for all supported models
    MODEL_LIMITS: Dict[str, Dict[str, Any]] = {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06
        },
        "gemini-pro": {
            "max_tokens": 30720,
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002
        },
        # Add other model limits
    }

    def __init__(self, model_type: str):
        """Initialize token manager for specific model type."""
        self.model_type = model_type
        self.tokenizer = self.get_model_tokenizer(model_type)
        # Initialize model-specific components

    def get_model_tokenizer(self, model_type: str):
        """Get appropriate tokenizer for model type."""
        # Implementation for different tokenizers
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens using model-specific tokenizer."""
        # Implementation for token counting
        pass

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model-specific pricing."""
        # Implementation for cost calculation
        pass
```

3. **Monitoring Integration**:
```python
# monitoring.py
"""
Monitoring system for tracking metrics across different AI models.
Handles logging, metrics collection, and reporting for all model types.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import json

@dataclass
class ModelMetrics:
    """Metrics structure for model operations."""
    model_type: str
    operation: str
    tokens_used: int
    response_time: float
    status: str
    cost: float
    error: Optional[str] = None

class SystemMonitor:
    """System-wide monitoring for all AI models."""
    
    def __init__(self):
        """Initialize monitoring system with model-specific metrics."""
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }
        self.start_time = time.time()

    def log_model_specific_metrics(
        self,
        model_type: str,
        operation: str,
        metrics: Dict[str, Any]
    ):
        """Log metrics specific to each model type."""
        # Implementation for metric logging
        pass

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        # Implementation for report generation
        pass
```

4. **Cache Integration**:
```python
# cache.py
"""
Caching system supporting multiple AI models.
Handles cache key generation, storage, and retrieval
with model-specific considerations.
"""

import hashlib
import json
from typing import Optional, Dict, Any
import redis

class Cache:
    """Multi-model caching system."""
    
    def __init__(self, redis_config: Optional[Dict[str, Any]] = None):
        """Initialize cache with Redis connection."""
        self.redis = redis.Redis(**(redis_config or {}))

    def generate_cache_key(
        self,
        model_type: str,
        function_name: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate unique cache key including model type."""
        # Implementation for cache key generation
        pass

    async def get_cached_response(
        self,
        model_type: str,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached response with model context."""
        # Implementation for cache retrieval
        pass

    async def cache_response(
        self,
        model_type: str,
        key: str,
        response: Dict[str, Any],
        ttl: int = 3600
    ):
        """Cache response with model-specific TTL."""
        # Implementation for caching
        pass
```

5. **Main Workflow Integration**:
```python
# main.py
"""
Main workflow system supporting multiple AI models.
Handles file processing, model selection, and orchestration
of the docstring generation process.
"""

import asyncio
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

async def process_file(
    file_path: str,
    args: argparse.Namespace
) -> None:
    """Process file using selected AI model."""
    
    # Get model configuration
    model_config = get_model_config(args.model_type)
    
    # Initialize components
    client = AIClientFactory.create_client(model_config)
    token_manager = TokenManager(model_type=args.model_type)
    monitor = SystemMonitor()
    cache = Cache()

    try:
        async with client:
            # Process file implementation
            pass
    except Exception as e:
        # Error handling implementation
        pass

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser with model selection."""
    # Implementation for argument parser
    pass

async def main():
    """Main entry point with model selection and processing."""
    # Implementation for main workflow
    pass
```

Would you like me to provide the complete implementation for any of these components?


Based on the provided resources, I'll help outline the key integration points for implementing OpenAI, Claude, and Gemini models alongside Azure OpenAI. Here's a structured approach:

1. **Model Factory Integration**:
```python
# model_factory.py
from typing import Optional, Dict
from base_client import BaseAIClient
from config import AIModelConfig

class AIClientFactory:
    """Factory for creating AI model clients with unified interface."""
    
    _clients: Dict[str, type] = {
        "azure": AzureOpenAIClient,
        "openai": OpenAIClient,
        "claude": ClaudeClient,
        "gemini": GeminiClient
    }
    
    @classmethod
    def register_client(cls, model_type: str, client_class: type):
        """Register new model client."""
        cls._clients[model_type] = client_class
    
    @classmethod
    def create_client(cls, config: AIModelConfig) -> Optional[BaseAIClient]:
        """Create appropriate AI client based on configuration."""
        if config.model_type not in cls._clients:
            raise ValueError(f"Unsupported model type: {config.model_type}")
            
        client_class = cls._clients[config.model_type]
        return client_class(config)
```

2. **Configuration System**:
```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIModelConfig:
    """Base configuration for all AI models."""
    model_type: str
    max_tokens: int
    temperature: float
    request_timeout: int
    max_retries: int
    retry_delay: int

@dataclass
class OpenAIConfig(AIModelConfig):
    api_key: str
    organization_id: Optional[str] = None
    model_name: str = "gpt-4"

@dataclass
class ClaudeConfig(AIModelConfig):
    api_key: str
    model_name: str = "claude-3-opus-20240229"

@dataclass
class GeminiConfig(AIModelConfig):
    api_key: str
    project_id: Optional[str] = None
    model_name: str = "gemini-pro"
```

3. **Client Implementations**:
```python
# base_client.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAIClient(ABC):
    """Base interface for all AI model clients."""
    
    @abstractmethod
    async def generate_docstring(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate docstring using the AI model."""
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of AI service."""
        pass

    @abstractmethod
    async def batch_process(self, prompts: List[str], **kwargs) -> List[Optional[Dict[str, Any]]]:
        """Process multiple prompts in batch."""
        pass
```

4. **Monitoring Integration**:
```python
# monitoring.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelMetrics:
    """Metrics structure for model operations."""
    model_type: str
    operation: str
    tokens_used: int
    response_time: float
    status: str
    cost: float
    error: Optional[str] = None

class SystemMonitor:
    """Unified monitoring for all AI models."""
    
    def __init__(self):
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }
        
    def log_model_metrics(
        self,
        model_type: str,
        metrics: ModelMetrics
    ):
        """Log metrics for specific model."""
        if model_type not in self.metrics_by_model:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.metrics_by_model[model_type][metrics.operation] = metrics
```

5. **Token Management**:
```python
# token_management.py
class TokenManager:
    """Token management for multiple AI models."""
    
    MODEL_LIMITS = {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06
        },
        "claude-3": {
            "max_tokens": 100000,
            "cost_per_1k_prompt": 0.008,
            "cost_per_1k_completion": 0.024
        },
        "gemini-pro": {
            "max_tokens": 30720,
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002
        }
    }
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.tokenizer = self._get_tokenizer(model_type)
```

6. **Cache Integration**:
```python
# cache.py
class Cache:
    """Multi-model caching system."""
    
    def generate_cache_key(
        self,
        model_type: str,
        function_name: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key including model type."""
        key_components = [
            model_type,
            function_name,
            hashlib.md5(str(params).encode()).hexdigest()
        ]
        return ":".join(key_components)
```

7. **Error Handling**:
```python
# exceptions.py
class ModelError(Exception):
    """Base class for model-specific errors."""
    def __init__(self, model_type: str, message: str):
        self.model_type = model_type
        super().__init__(f"{model_type} error: {message}")

class TokenLimitError(ModelError):
    """Error for token limit exceeded."""
    pass

class ModelAPIError(ModelError):
    """Error for API-related issues."""
    pass
```

These integration points provide:
- Unified interface through BaseAIClient
- Consistent configuration system
- Centralized monitoring
- Shared token management
- Common caching mechanism
- Standardized error handling

Would you like me to elaborate on any of these components or provide additional implementation details?