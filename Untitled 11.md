# Untitled 11

```python
"""
Claude-3 Sonnet provider implementation with template awareness.
Uses external schema definitions for validation and response formatting.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from anthropic import AsyncAnthropic

from ..base import BaseLLMProvider, TokenUsage, GenerationMetrics
from ..templates import DocstringStyle, TemplateConfig
from ..schemas import get_schema, validate_schema, DOCSTRING_SCHEMAS
from ..logger import get_logger

logger = get_logger(__name__)

class ClaudeProvider(BaseLLMProvider):
    """
    Claude-3 Sonnet provider with template-aware generation capabilities.
    Optimized for structured docstring generation with context awareness.
    """
    
    MODEL_NAME = "claude-3-sonnet-20241022"
    MAX_TOKENS = 200000  # Claude-3's context window
    
    def __init__(self, config: 'ProviderConfig'):
        """Initialize Claude provider with configuration."""
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key)
        
        # Template-specific configurations
        self.style_configs = {
            DocstringStyle.GOOGLE: {
                'system_prompt': (
                    "You are a documentation expert generating Google-style "
                    "docstrings. Follow PEP 257 and Google's style guide strictly. "
                    "Structure output as valid JSON matching the provided schema."
                ),
                'response_format': {
                    'type': 'json_object',
                    'schema': get_schema('google')
                }
            },
            DocstringStyle.NUMPY: {
                'system_prompt': (
                    "You are a documentation expert generating NumPy-style "
                    "docstrings. Follow NumPy documentation guidelines strictly. "
                    "Structure output as valid JSON matching the provided schema."
                ),
                'response_format': {
                    'type': 'json_object',
                    'schema': get_schema('numpy')
                }
            }
            # Additional styles automatically loaded from DOCSTRING_SCHEMAS
        }
        
        # Load any additional styles from schema registry
        for style_name in DOCSTRING_SCHEMAS:
            if style_name not in self.style_configs:
                self.style_configs[DocstringStyle(style_name)] = {
                    'system_prompt': (
                        f"You are a documentation expert generating {style_name}-style "
                        f"docstrings. Follow the {style_name} documentation guidelines "
                        "strictly. Structure output as valid JSON matching the provided schema."
                    ),
                    'response_format': {
                        'type': 'json_object',
                        'schema': get_schema(style_name)
                    }
                }
    
    async def generate(
        self,
        prompt: str,
        style: DocstringStyle = DocstringStyle.GOOGLE,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate docstring using Claude-3 with template awareness.
        
        Args:
            prompt: The formatted prompt for generation
            style: Docstring style to use
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generated docstring and metadata
            
        Raises:
            ValueError: If docstring style is not supported
            Exception: If generation or validation fails
        """
        try:
            style_config = self.style_configs.get(style)
            if not style_config:
                raise ValueError(f"Unsupported docstring style: {style}")
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": style_config['system_prompt']
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Configure generation parameters
            params = {
                "model": self.MODEL_NAME,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "response_format": style_config['response_format']
            }
            
            # Generate response
            response = await self.client.messages.create(
                messages=messages,
                **params
            )
            
            # Process and validate response
            content = response.content[0].text
            if response_format := style_config['response_format']:
                try:
                    content = json.loads(content)
                    # Use external schema validation
                    if not validate_schema(style.value, content):
                        raise ValueError("Response failed schema validation")
                except Exception as e:
                    logger.error(f"Response validation failed: {e}")
                    raise
            
            # Track metrics
            self.track_generation(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                style=style
            )
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Claude's tokenizer."""
        try:
            count = await self.client.count_tokens(text)
            return count
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split()) * 2  # Rough estimation fallback
    
    def track_generation(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        style: DocstringStyle
    ) -> None:
        """Track generation metrics with style awareness."""
        total_tokens = prompt_tokens + completion_tokens
        
        self.metrics.append(
            GenerationMetrics(
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                tokens=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                ),
                style=style
            )
        )
    
    def get_style_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get generation metrics broken down by docstring style."""
        style_metrics = {}
        
        for style in DocstringStyle:
            style_generations = [
                m for m in self.metrics 
                if m.style == style and m.success
            ]
            
            if style_generations:
                total_tokens = sum(
                    m.tokens.total_tokens for m in style_generations
                )
                avg_tokens = total_tokens / len(style_generations)
                
                style_metrics[style.value] = {
                    "total_generations": len(style_generations),
                    "total_tokens": total_tokens,
                    "average_tokens": avg_tokens,
                    "success_rate": self._calculate_success_rate(style)
                }
        
        return style_metrics
    
    def _calculate_success_rate(self, style: DocstringStyle) -> float:
        """Calculate success rate for a specific docstring style."""
        style_attempts = [m for m in self.metrics if m.style == style]
        if not style_attempts:
            return 0.0
            
        successful = len([m for m in style_attempts if m.success])
        return (successful / len(style_attempts)) * 100
```
