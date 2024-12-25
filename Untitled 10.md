# Untitled 10

```python
"""
Template-aware orchestration system for managing LLM providers and docstring generation.
"""

from typing import Dict, Any, Optional, List, Type, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
import hashlib

from .providers import BaseLLMProvider, ClaudeProvider
from .templates import DocstringStyle, DocstringTemplate
from .schemas import validate_schema
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
        
        # Metrics tracking
        self.generation_metrics: List[Dict[str, Any]] = []
    
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
            else:
                # Get capabilities from provider if available
                if hasattr(provider, 'get_capabilities'):
                    capabilities[name] = provider.get_capabilities()
                else:
                    # Default capabilities
                    capabilities[name] = {
                        'max_tokens': 4000,
                        'preferred_styles': [DocstringStyle.GOOGLE],
                        'supports_json': True,
                        'supports_streaming': False
                    }
        
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
        
        # Select based on capabilities and success metrics
        return max(
            compatible_providers,
            key=lambda p: (
                style in self.provider_capabilities[p].get('preferred_styles', []),
                self._get_provider_success_rate(p, style),
                self.provider_capabilities[p].get('max_tokens', 0)
            )
        )
    
    def _select_fallback_provider(
        self,
        style: DocstringStyle,
        exclude: List[str]
    ) -> Optional[str]:
        """Select fallback provider for given style."""
        compatible_providers = [
            name for name, styles in self.template_compatibility.items()
            if style in styles and name not in exclude
        ]
        
        if not compatible_providers:
            return None
            
        # Select based on success metrics
        return max(
            compatible_providers,
            key=lambda p: self._get_provider_success_rate(p, style)
        )
    
    def _get_provider_success_rate(
        self,
        provider: str,
        style: DocstringStyle
    ) -> float:
        """Calculate provider success rate for given style."""
        provider_attempts = [
            m for m in self.generation_metrics
            if m['provider'] == provider and m['style'] == style
        ]
        
        if not provider_attempts:
            return 0.0
        
        successful = len([
            m for m in provider_attempts
            if m['success'] and not m['from_cache']
        ])
        return successful / len(provider_attempts)
    
    async def _generate_with_provider(
        self,
        provider: BaseLLMProvider,
        template: DocstringTemplate,
        code_metadata: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate docstring using specific provider and template."""
        start_time = datetime.now()
        success = False
        
        try:
            # Format prompt using template
            prompt = template.format_prompt(
                metadata=code_metadata,
                **template.get_provider_params(provider.__class__.__name__)
            )
            
            # Generate with retries
            for attempt in range(self.config.max_retries):
                try:
                    result = await asyncio.wait_for(
                        provider.generate(
                            prompt=prompt,
                            style=template.style,
                            **kwargs
                        ),
                        timeout=self.config.timeout
                    )
                    
                    # Validate result
                    if validate_schema(template.style.value, result['content']):
                        success = True
                        break
                        
                except asyncio.TimeoutError:
                    if attempt == self.config.max_retries - 1:
                        raise
                    continue
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            if not success:
                raise ValueError("Failed to generate valid docstring")
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Track metrics
            self.generation_metrics.append({
                'provider': provider.__class__.__name__,
                'style': template.style,
                'success': True,
                'tokens_used': result['usage']['total_tokens'],
                'generation_time': generation_time,
                'from_cache': False
            })
            
            return GenerationResult(
                content=result['content'],
                provider=provider.__class__.__name__,
                style=template.style,
                tokens_used=result['usage']['total_tokens'],
                generation_time=generation_time
            )
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Track failure metrics
            self.generation_metrics.append({
                'provider': provider.__class__.__name__,
                'style': template.style,
                'success': False,
                'error': str(e),
                'generation_time': generation_time,
                'from_cache': False
            })
            
            raise
    
    def _generate_cache_key(
        self,
        metadata: Dict[str, Any],
        style: DocstringStyle
    ) -> str:
        """Generate cache key from metadata and style."""
        # Create deterministic string from metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        
        # Generate hash
        key = f"{style.value}:{metadata_str}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def generate_docstring(
        self,
        code_metadata: Dict[str, Any],
        style: DocstringStyle = DocstringStyle.GOOGLE,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate docstring with style awareness and provider selection."""
        # Check cache first
        if self.storage:
            cache_key = self._generate_cache_key(code_metadata, style)
            cached = await self.storage.get(cache_key)
            if cached:
                # Track cache hit
                self.generation_metrics.append({
                    'provider': cached['provider'],
                    'style': style,
                    'success': True,
                    'tokens_used': cached['tokens_used'],
                    'generation_time': 0,
                    'from_cache': True
                })
                
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
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.config.timeout
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from first completed task
        try:
            result = done.pop().result()
            
            # Cache successful result
            if self.storage and not result.from_cache:
                cache_key = self._generate_cache_key(code_metadata, style)
                await self.storage.set(
                    cache_key,
                    {
                        'content': result.content,
                        'provider': result.provider,
                        'tokens_used': result.tokens_used,
                        'timestamp': datetime.now().isoformat()
                    },
                    ttl=self.config.cache_ttl
                )
            
            return result
            
        except Exception as e:
            logger.error(f"All generation attempts failed: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive generation metrics."""
        metrics = {
            'providers': {},
            'styles': {},
            'cache': {
                'hits': len([m for m in self.generation_metrics if m['from_cache']]),
                'misses': len([m for m in self.generation_metrics if not m['from_cache']])
            }
        }
        
        # Provider metrics
        for provider in self.providers:
            provider_attempts = [
                m for m in self.generation_metrics
                if m['provider'] == provider
            ]
            
            if provider_attempts:
                success_rate = len([
                    m for m in provider_attempts
                    if m['success']
                ]) / len(provider_attempts)
                
                metrics['providers'][provider] = {
                    'total_attempts': len(provider_attempts),
                    'success_rate': success_rate,
                    'average_tokens': sum(
                        m.get('tokens_used', 0) for m in provider_attempts
                    ) / len(provider_attempts),
                    'average_time': sum(
                        m['generation_time'] for m in provider_attempts
                    ) / len(provider_attempts)
                }
        
        # Style metrics
        for style in DocstringStyle:
            style_attempts = [
                m for m in self.generation_metrics
                if m['style'] == style
            ]
            
            if style_attempts:
                metrics['styles'][style.value] = {
                    'total_attempts': len(style_attempts),
                    'success_rate': len([
                        m for m in style_attempts if m['success']
                    ]) / len(style_attempts),
                    'provider_breakdown': self._get_style_provider_breakdown(style)
                }
        
        return metrics
    
    def _get_style_provider_breakdown(
        self,
        style: DocstringStyle
    ) -> Dict[str, Dict[str, Any]]:
        """Get provider breakdown for specific style."""
        breakdown = {}
        
        style_attempts = [
            m for m in self.generation_metrics
            if m['style'] == style
        ]
        
        for provider in self.providers:
            provider_attempts = [
                m for m in style_attempts
                if m['provider'] == provider
            ]
            
            if provider_attempts:
                breakdown[provider] = {
                    'attempts': len(provider_attempts),
                    'success_rate': len([
                        m for m in provider_attempts if m['success']
                    ]) / len(provider_attempts)
                }
        
        return breakdown
    
    async def clear_cache(self) -> None:
        """Clear all cached results."""
        if self.storage:
            await self.storage.clear()
```
