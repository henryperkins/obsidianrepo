---
epoch: 1731886789071
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [api_client.py](api_client.py.md)*[gemini_model.py](gemini_model.py.md)*[claude_model.py](claude_model.py.md)*[openai_model.py](openai_model.py.md)*[base_client.py](base_client.py.md)_[[m@20241117_173949

**user**: [api_client.py](simple2/api_client.py.md) [gemini_model.py](gemini_model.py.md) [claude_model.py](claude_model.py.md) [openai_model.py](openai_model.py.md) [base_client.py](base_client.py.md) [model_factory.py](model_factory.py.md) [interaction_handler.py](interaction_handler.py.md) [api_interaction.py](api_interaction.py.md) [config.py](simple2/config.py.md) [Untitled 4](./Untitled%204.md)  
[Timestamp: 2024/11/17 17:39:49]

**ai**: Based on the code provided, I'll help create a comprehensive AI model integration system. Let's create a new file called `ai_integration.py` that will tie everything together:

```python
"""
AI Integration Module

This module provides a unified interface for interacting with multiple AI models
(Azure OpenAI, OpenAI, Claude, Gemini) with advanced features like:
- Automatic model selection and fallback
- Load balancing
- Caching
- Error handling
- Metrics tracking
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from model_factory import AIClientFactory
from config import AIModelConfig, AzureOpenAIConfig, OpenAIConfig, ClaudeConfig, GeminiConfig
from logger import log_info, log_error, log_debug
from cache import Cache
from monitoring import SystemMonitor
from metrics import Metrics

@dataclass
class ModelPriority:
    """Configuration for model priority and fallback."""
    primary: str
    fallbacks: List[str]
    timeout: int = 30
    max_retries: int = 3

class AIIntegration:
    """
    Unified interface for AI model interactions with advanced features.
    """

    def __init__(
        self,
        configs: Dict[str, AIModelConfig],
        model_priority: Optional[ModelPriority] = None,
        cache_config: Optional[Dict] = None
    ):
        """
        Initialize the AI integration system.

        Args:
            configs: Dictionary of model configurations
            model_priority: Optional priority configuration for models
            cache_config: Optional cache configuration
        """
        self.configs = configs
        self.model_priority = model_priority or ModelPriority(
            primary="azure",
            fallbacks=["openai", "claude", "gemini"]
        )
        
        # Initialize components
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.metrics = Metrics()
        self.clients = {}
        
        # Initialize AI clients
        self._initialize_clients()
        
        log_info("AI Integration system initialized successfully")

    def _initialize_clients(self):
        """Initialize AI clients for each configured model."""
        for model_type, config in self.configs.items():
            try:
                self.clients[model_type] = AIClientFactory.create_client(config)
                log_info(f"Initialized {model_type} client")
            except Exception as e:
                log_error(f"Failed to initialize {model_type} client: {e}")

    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate docstring using configured AI models with fallback support.

        Args:
            func_name: Name of the function
            params: List of parameter names and types
            return_type: Return type of the function
            complexity_score: Complexity score of the function
            existing_docstring: Existing docstring if any
            decorators: List of decorators
            exceptions: List of exceptions

        Returns:
            Optional[Dict[str, Any]]: Generated docstring and metadata
        """
        # Check cache first
        cache_key = f"docstring:{func_name}:{hash(str(params))}"
        cached_response = await self.cache.get(cache_key)
        if cached_response:
            log_info(f"Cache hit for function: {func_name}")
            return cached_response

        # Try primary model first
        primary_client = self.clients.get(self.model_priority.primary)
        if primary_client:
            try:
                response = await primary_client.generate_docstring(
                    func_name=func_name,
                    params=params,
                    return_type=return_type,
                    complexity_score=complexity_score,
                    existing_docstring=existing_docstring,
                    decorators=decorators,
                    exceptions=exceptions
                )
                if response:
                    await self.cache.set(cache_key, response)
                    return response
            except Exception as e:
                log_error(f"Primary model failed: {e}")

        # Try fallback models
        for model_type in self.model_priority.fallbacks:
            client = self.clients.get(model_type)
            if not client:
                continue

            try:
                response = await client.generate_docstring(
                    func_name=func_name,
                    params=params,
                    return_type=return_type,
                    complexity_score=complexity_score,
                    existing_docstring=existing_docstring,
                    decorators=decorators,
                    exceptions=exceptions
                )
                if response:
                    await self.cache.set(cache_key, response)
                    return response
            except Exception as e:
                log_error(f"Fallback model {model_type} failed: {e}")

        log_error(f"All models failed for function: {func_name}")
        return None

    async def batch_generate_docstrings(
        self,
        functions: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate docstrings for multiple functions with load balancing.

        Args:
            functions: List of function metadata
            batch_size: Size of batches for processing

        Returns:
            List[Optional[Dict[str, Any]]]: List of generated docstrings
        """
        results = []
        
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            batch_tasks = [
                self.generate_docstring(**func) for func in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for func, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    log_error(f"Error processing {func['func_name']}: {result}")
                    results.append(None)
                else:
                    results.append(result)
                    
            # Add delay between batches for rate limiting
            if i + batch_size < len(functions):
                await asyncio.sleep(0.1)
                
        return results

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all configured models.

        Returns:
            Dict[str, Any]: Health status for all models
        """
        health_status = {}
        
        for model_type, client in self.clients.items():
            try:
                status = await client.health_check()
                health_status[model_type] = status
            except Exception as e:
                health_status[model_type] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                
        return health_status

    async def close(self):
        """Close all client connections."""
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                log_error(f"Error closing client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

```

Usage example:

```python
# Initialize configurations
configs = {
    "azure": AzureOpenAIConfig.from_env(),
    "openai": OpenAIConfig.from_env(),
    "claude": ClaudeConfig.from_env(),
    "gemini": GeminiConfig.from_env()
}

# Configure model priority
model_priority = ModelPriority(
    primary="azure",
    fallbacks=["openai", "claude", "gemini"],
    timeout=30,
    max_retries=3
)

# Initialize AI integration
ai = AIIntegration(
    configs=configs,
[Timestamp: 2024/11/17 17:41:06]

**user**: Just trying to see about the template system.In [[Untitled 4]]
[Timestamp: 2024/11/17 17:42:24]

**ai**: It looks like you're experimenting with the template system in Obsidian. In your note titled "Untitled 4," you might want to include a template or structure for organizing your thoughts or code snippets related to AI model integration. Here's a suggestion for what you could include in that note:

---

# AI Integration Template

## Overview
Provide a brief description of the purpose of this integration module and what it aims to achieve.

## Configuration
- **Azure OpenAI Configuration**: Details about the Azure OpenAI setup.
- **OpenAI Configuration**: Details about the OpenAI setup.
- **Claude Configuration**: Details about the Claude setup.
- **Gemini Configuration**: Details about the Gemini setup.

## Model Priority
Outline the priority and fallback strategy for model selection.

## Features
- **Automatic Model Selection**: Describe how the system selects models.
- **Load Balancing**: Explain the load balancing mechanism.
- **Caching**: How caching is implemented and its benefits.
- **Error Handling**: Strategies for handling errors.
- **Metrics Tracking**: What metrics are tracked and how.

## Usage Example
Provide a code snippet or example of how to use the AI integration module.

```python
# Example code for using the AI Integration module
configs = {
    "azure": AzureOpenAIConfig.from_env(),
    "openai": OpenAIConfig.from_env(),
    "claude": ClaudeConfig.from_env(),
    "gemini": GeminiConfig.from_env()
}

model_priority = ModelPriority(
    primary="azure",
    fallbacks=["openai", "claude", "gemini"],
    timeout=30,
    max_retries=3
)

ai_integration = AIIntegration(
    configs=configs,
    model_priority=model_priority
)

# Example usage
async def main():
    async with ai_integration as ai:
        result = await ai.generate_docstring(
            func_name="example_function",
            params=[("param1", "str"), ("param2", "int")],
            return_type="bool",
            complexity_score=5,
            existing_docstring=""
        )
        print(result)

# Run the example
# asyncio.run(main())
```

## Notes

- Any additional notes or considerations for the integration.

---

Feel free to customize this template to better fit your needs or the specific details of your AI integration project.  
[Timestamp: 2024/11/17 17:43:06]
