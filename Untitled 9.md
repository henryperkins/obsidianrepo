# Untitled 9

```python
"""
Template-aware orchestration system for managing LLM providers and docstring generation.
"""

from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
import asyncio
from datetime import datetime

from .providers import BaseLLMProvider, ClaudeProvider
from .templates import DocstringStyle, DocstringTemplate
from .storage import Storage
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for docstring generation."""
    max_retries: int = 3
    timeout: float = 30.0
    cache_ttl: int = 3600
    parallel_attempts: int = 2

@dataclass
class GenerationResult:
    """Result of docstring generation."""
    content: Dict[str, Any]
    provider: str
    style: DocstringStyle
    tokens_used: int
    generation_time: float
    from_cache: bool = False

class TemplateOrchestrator:
    """
    Orchestrates template-aware docstring generation across multiple providers.
    Handles provider selection, template application, and generation coordination.
    """
    
    def __init__(
        self,
        providers: Dict[str, BaseLLMProvider],
        templates: Dict[DocstringStyle, DocstringTemplate],
        storage: Optional[Storage] = None,
        config: Optional[GenerationConfig] = None
    ):
        """Initialize orchestrator with providers and templates."""
        self.providers = providers
        self.templates = templates
        self.storage = storage
        self.config = config or GenerationConfig()
        
        # Provider capabilities mapping
        self.provider_capabilities = self._map_provider_capabilities()
        
        # Template compatibility matrix
        self.template_compatibility = self._build_compatibility_matrix()
    
    def _map_provider_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Map provider capabilities and preferences."""
        capabilities = {}
        
        for name, provider in self.providers.items():
            if isinstance(provider, ClaudeProvider):
                capabilities[name] = {
                    'max_tokens': 200000,
                    'preferred_styles': [
                        DocstringStyle.GOOGLE,
                        DocstringStyle.NUMPY
                    ],
                    'supports_json': True,
                    'supports_streaming': True
                }
            # Add mappings for other providers...
        
        return capabilities
    
    def _build_compatibility_matrix(self) -> Dict[str, List[DocstringStyle]]:
        """Build provider-template compatibility matrix."""
        matrix = {}
        
        for provider_name, capabilities in self.provider_capabilities.items():
            compatible_styles = []
            
            for style in DocstringStyle:
                if style in capabilities.get('preferred_styles', []):
                    compatible_styles.append(style)
                elif self._check_style_compatibility(provider_name, style):
                    compatible_styles.append(style)
            
            matrix[provider_name] = compatible_styles
        
        return matrix
    
    def _check_style_compatibility(
        self,
        provider: str,
        style: DocstringStyle
    ) -> bool:
        """Check if provider can handle given style."""
        capabilities = self.provider_capabilities.get(provider, {})
        
        # Check basic requirements
        if style.requires_json and not capabilities.get('supports_json'):
            return False
            
        # Check token requirements
        if style.min_tokens > capabilities.get('max_tokens', 0):
            return False
        
        return True
    
    def _select_provider(
        self,
        style: DocstringStyle,
        preferred_provider: Optional[str] = None
    ) -> str:
        """Select best provider for given style."""
        if preferred_provider:
            if preferred_provider in self.providers and \
               style in self.template_compatibility[preferred_provider]:
                return preferred_provider
        
        # Find providers supporting this style
        compatible_providers = [
            name for name, styles in self.template_compatibility.items()
            if style in styles
        ]
        
        if not compatible_providers:
            raise ValueError(f"No compatible provider for style: {style}")
        
        # Select based on capabilities
        return max(
            compatible_providers,
            key=lambda p: (
                style in self.provider_capabilities[p].get('preferred_styles', []),
                self.provider_capabilities[p].get('max_tokens', 0)
            )
        )
    
    async def generate_docstring(
        self,
        code_metadata: Dict[str, Any],
        style: DocstringStyle = DocstringStyle.GOOGLE,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate docstring with style awareness and provider selection.
        
        Args:
            code_metadata: Extracted code metadata
            style: Desired docstring style
            preferred_provider: Preferred provider to use
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult containing generated docstring
        """
        # Check cache first
        if self.storage:
            cache_key = self._generate_cache_key(code_metadata, style)
            cached = await self.storage.get(cache_key)
            if cached:
                return GenerationResult(
                    content=cached['content'],
                    provider=cached['provider'],
                    style=style,
                    tokens_used=cached['tokens_used'],
                    generation_time=0,
                    from_cache=True
                )
        
        # Select provider and template
        provider_name = self._select_provider(style, preferred_provider)
        provider = self.providers[provider_name]
        template = self.templates[style]
        
        # Initialize generators
        primary_task = self._generate_with_provider(
            provider=provider,
            template=template,
            code_metadata=code_metadata,
            **kwargs
        )
        
        tasks = [primary_task]
        
        # Add fallback if configured
        if self.config.parallel_attempts > 1:
            fallback_provider = self._select_fallback_provider(
                style,
                exclude=[provider_name]
            )
            if fallback_provider:
                fallback_task = self._generate_with_provider(
                    provider=self.providers[fallback_provider],
                    template=template,
                    code_metadata=code_metadata,
                    **kwargs
                )
                tasks.append(fallback_task)
        
        # Wait for first successful result
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from fir
```
