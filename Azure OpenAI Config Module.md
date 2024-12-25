```python
# config.py
"""
Configuration module for Azure OpenAI service integration.
Handles environment variables, validation, and configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import urllib.parse
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

@dataclass(frozen=True)
class CacheConfig:
    """Cache-specific configuration settings."""
    enabled: bool = field(default_factory=lambda: get_env_bool("CACHE_ENABLED", False))
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))
    password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    ttl: int = field(default_factory=lambda: get_env_int("CACHE_TTL", 3600))
    prefix: str = field(default_factory=lambda: os.getenv("CACHE_PREFIX", "docstring:"))

@dataclass(frozen=True)
class TokenConfig:
    """Token management configuration settings."""
    max_tokens_per_minute: int = field(
        default_factory=lambda: get_env_int("MAX_TOKENS_PER_MINUTE", 150000)
    )
    token_buffer: int = field(
        default_factory=lambda: get_env_int("TOKEN_BUFFER", 100)
    )
    batch_size: int = field(
        default_factory=lambda: get_env_int("BATCH_SIZE", 5)
    )
    max_retries: int = field(
        default_factory=lambda: get_env_int("MAX_RETRIES", 3)
    )

@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring configuration settings."""
    enabled: bool = field(
        default_factory=lambda: get_env_bool("MONITORING_ENABLED", True)
    )
    check_interval: int = field(
        default_factory=lambda: get_env_int("MONITORING_INTERVAL", 60)
    )
    metrics_enabled: bool = field(
        default_factory=lambda: get_env_bool("METRICS_ENABLED", True)
    )
    log_metrics: bool = field(
        default_factory=lambda: get_env_bool("LOG_METRICS", True)
    )
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_threshold': float(os.getenv("CPU_THRESHOLD", "90")),
        'memory_threshold': float(os.getenv("MEMORY_THRESHOLD", "90")),
        'token_usage_threshold': float(os.getenv("TOKEN_USAGE_THRESHOLD", "80"))
    })

@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI service."""
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "azure"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4"))
    deployment_id: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", ""))
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY", ""))
    api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-09-01-preview")
    )

    # Add sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    token: TokenConfig = field(default_factory=TokenConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Model-specific settings
    max_tokens: int = field(default_factory=lambda: get_env_int("MAX_TOKENS", 6000))
    temperature: float = field(default_factory=lambda: get_env_float("TEMPERATURE", 0.4))
    request_timeout: int = field(default_factory=lambda: get_env_int("REQUEST_TIMEOUT", 30))

    def validate(self) -> bool:
        """Validate the configuration settings."""
        if not self.model_type:
            logger.error("Model type is required")
            return False

        if not self.endpoint:
            logger.error("Azure OpenAI endpoint is required")
            return False
        else:
            parsed_url = urllib.parse.urlparse(self.endpoint)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error("Invalid Azure OpenAI endpoint URL")
                return False

        if not self.api_key:
            logger.error("Azure OpenAI API key is required")
            return False

        if not self.deployment_id:
            logger.error("Azure OpenAI deployment ID is required")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive information."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "deployment_id": self.deployment_id,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "cache_enabled": self.cache.enabled,
            "monitoring_enabled": self.monitoring.enabled,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

# monitoring.py
class SystemMonitor:
    """Monitors system resources and AI service usage."""

    def __init__(
        self,
        config: MonitoringConfig,
        token_manager: Optional[TokenManager] = None,
        cache: Optional[Cache] = None
    ) -> None:
        """Initialize system monitor."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config
        self.token_manager = token_manager
        self.cache = cache
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.alert_history: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """Start monitoring system resources."""
        if self._running or not self.config.enabled:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info("System monitoring started")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Collect token usage metrics
                if self.token_manager:
                    token_metrics = self.token_manager.get_usage_stats()
                    metrics['token_usage'] = token_metrics

                # Collect cache metrics
                if self.cache:
                    cache_metrics = await self.cache.get_stats()
                    metrics['cache'] = cache_metrics

                # Store metrics
                self._store_metrics(metrics)

                # Check thresholds and send alerts
                await self._check_thresholds(metrics)

                # Sleep until next check
                await asyncio.sleep(self.config.check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.check_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }

            return metrics
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}

    async def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against configured thresholds."""
        alerts = []

        # Check CPU usage
        if metrics.get('cpu', {}).get('percent', 0) > self.config.alert_thresholds['cpu_threshold']:
            alerts.append({
                'type': 'CPU',
                'message': f"High CPU usage: {metrics['cpu']['percent']}%",
                'level': 'warning'
            })

        # Check memory usage
        if metrics.get('memory', {}).get('percent', 0) > self.config.alert_thresholds['memory_threshold']:
            alerts.append({
                'type': 'Memory',
                'message': f"High memory usage: {metrics['memory']['percent']}%",
                'level': 'warning'
            })

        # Check token usage
        if 'token_usage' in metrics:
            usage_percent = (metrics['token_usage']['total_tokens'] / 
                           self.token_manager.get_model_limits()['max_tokens']) * 100
            if usage_percent > self.config.alert_thresholds['token_usage_threshold']:
                alerts.append({
                    'type': 'Token',
                    'message': f"High token usage: {usage_percent:.1f}%",
                    'level': 'warning'
                })

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Alert: {alert['type']} - {alert['message']}")
            self.alert_history.append({
                **alert,
                'timestamp': datetime.now().isoformat()
            })

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        try:
            current_metrics = self._collect_system_metrics()
            return {
                'current': current_metrics,
                'averages': self._calculate_averages(),
                'alerts': self.alert_history[-10:],  # Last 10 alerts
                'status': self._get_system_status()
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {'error': str(e)}

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("System monitoring stopped")
```

```python
# ai_interaction.py
class AIInteractionHandler:
    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache  # Cache instance injected
            self.token_manager = token_manager
            self.response_parser = response_parser
            self.metrics = metrics

            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    def _generate_cache_key(self, extracted_info: Dict[str, Any]) -> str:
        """Generate a unique cache key based on input data."""
        try:
            # Create a deterministic string representation of the input
            key_components = [
                extracted_info.get('source', ''),
                str(extracted_info.get('metrics', {})),
                str(sorted(extracted_info.get('args', []))),
                str(extracted_info.get('returns', {})),
                str(sorted(extracted_info.get('raises', [])))
            ]
            key_string = '|'.join(key_components)
            
            # Generate hash for the key
            return f"ai_doc:{hashlib.md5(key_string.encode()).hexdigest()}"
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return None

    async def process_code(
        self,
        extracted_info: Dict[str, Any],
        cache_key: Optional[str] = None,
    ) -> Optional[DocumentationData]:
        """Process code information and generate documentation."""
        try:
            # Generate cache key if not provided
            cache_key = cache_key or self._generate_cache_key(extracted_info)
            
            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        return DocumentationData(**cached_result)
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Generate prompt
            prompt = self._generate_prompt(extracted_info)

            # Get request parameters from token manager
            request_params = await self.token_manager.validate_and_prepare_request(prompt)

            # Make API request
            completion = await self.client.chat.completions.create(**request_params)

            # Process completion through token manager
            content, usage = await self.token_manager.process_completion(completion)

            if not content:
                raise ValidationError("Empty response from AI service")

            # Parse and validate
            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format="docstring",
                validate_schema=True,
            )

            if not parsed_response.validation_success:
                self.logger.warning(f"Response validation had errors: {parsed_response.errors}")
                return None

            # Create documentation data
            doc_data = DocumentationData(
                module_info={
                    "name": extracted_info.get("module_name", "Unknown"),
                    "file_path": extracted_info.get("file_path", "Unknown")
                },
                ai_content=parsed_response.content,
                docstring_data=self.docstring_processor.parse(parsed_response.content),
                code_metadata=extracted_info,
                source_code=extracted_info.get("source_code"),
                metrics=extracted_info.get("metrics")
            )

            # Cache the result if enabled
            if self.cache and cache_key:
                try:
                    await self.cache.save_docstring(
                        cache_key,
                        doc_data.to_dict(),  # Convert to dictionary for caching
                        expire=self.config.cache_ttl
                    )
                    self.logger.debug(f"Cached result for key: {cache_key}")
                except Exception as e:
                    self.logger.error(f"Cache storage error: {e}")

            return doc_data

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    async def invalidate_cache(self, extracted_info: Dict[str, Any]) -> bool:
        """Invalidate cache for specific code information."""
        try:
            if not self.cache:
                return True

            cache_key = self._generate_cache_key(extracted_info)
            if cache_key:
                await self.cache.invalidate(cache_key)
                self.logger.info(f"Invalidated cache for key: {cache_key}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if not self.cache:
                return {"enabled": False}
            return await self.cache.get_stats()
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close and clean up resources."""
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

# Example usage in main.py:
async def setup_cache(config: AzureOpenAIConfig) -> Optional[Cache]:
    """Setup cache instance based on configuration."""
    if config.cache_enabled:
        try:
            cache = Cache(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                password=config.redis_password,
                enabled=True,
                ttl=config.cache_ttl,
                prefix="docstring:"
            )
            # Test connection
            await cache._initialize_connection()
            return cache
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            return None
    return None

# In DocumentationGenerator initialization:
class DocumentationGenerator:
    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = await setup_cache(self.config)
            
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=self.cache,  # Pass cache instance
                token_manager=self.token_manager,
                response_parser=self.response_parser,
                metrics=self.metrics
            )
            # ... rest of initialization
```

```python
"""
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from ai_interaction import AIInteractionHandler
from api_client import APIClient
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from docstring_processor import DocstringProcessor
from docs import DocumentationOrchestrator
from repository_handler import RepositoryHandler
from token_management import TokenManager
from exceptions import ConfigurationError, DocumentationError

load_dotenv()
logger = LoggerSetup.get_logger(__name__)

class DocumentationGenerator:
    """Documentation Generator orchestrating all components."""

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """Initialize the documentation generator with all necessary components."""
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Load configuration
            self.config = config or AzureOpenAIConfig.from_env()

            # Initialize core components
            self.metrics = MetricsCollector()
            self.cache = self._setup_cache()
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_id=self.config.deployment_id,
                config=self.config,
                metrics_collector=self.metrics
            )
            
            # Initialize API and response handling
            self.response_parser = ResponseParsingService()
            self.api_client = APIClient(
                config=self.config,
                response_parser=self.response_parser
            )
            
            # Initialize processors
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)
            
            # Initialize handlers
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=self.cache,
                token_manager=self.token_manager,
                response_parser=self.response_parser,
                metrics=self.metrics
            )

            # Initialize orchestrators
            self.doc_orchestrator = DocumentationOrchestrator(
                ai_handler=self.ai_handler,
                docstring_processor=self.docstring_processor
            )

            # Initialize monitoring
            self.system_monitor = SystemMonitor(
                check_interval=60,
                token_manager=self.token_manager
            )

            self._initialized = False
            self.repo_handler = None

        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentationGenerator: {e}")
            raise ConfigurationError(f"Initialization failed: {e}") from e

    def _setup_cache(self) -> Optional[Cache]:
        """Setup cache if enabled in configuration."""
        if self.config.cache_enabled:
            return Cache(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                enabled=True
            )
        return None

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """Initialize components that depend on runtime arguments."""
        try:
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()
                self.logger.info(f"Repository handler initialized with path: {base_path}")

            await self.system_monitor.start()
            self._initialized = True
            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize: {e}") from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """
        Process a single Python file and generate documentation.

        Args:
            file_path: Path to the Python file
            output_path: Path where documentation should be written

        Returns:
            bool: True if processing was successful
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.suffix == '.py':
                raise ValueError(f"Not a Python file: {file_path}")

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate documentation
            source_code = file_path.read_text(encoding='utf-8')
            updated_code, documentation = await self.doc_orchestrator.generate_documentation(
                source_code=source_code,
                file_path=file_path,
                module_name=file_path.stem
            )

            # Write output files
            if updated_code:
                file_path.write_text(updated_code, encoding='utf-8')
            if documentation:
                output_path.write_text(documentation, encoding='utf-8')

            self.logger.info(f"Documentation generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    async def process_repository(
        self,
        repo_path_or_url: str,
        output_dir: Path = Path("docs")
    ) -> bool:
        """
        Process an entire repository and generate documentation.

        Args:
            repo_path_or_url: Local path or Git URL to repository
            output_dir: Output directory for documentation

        Returns:
            bool: True if processing was successful
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        is_url = bool(urlparse(repo_path_or_url).scheme)
        temp_dir = None

        try:
            # Handle repository setup
            if is_url:
                temp_dir = Path(tempfile.mkdtemp())
                repo_path = await self.repo_handler.clone_repository(
                    repo_path_or_url,
                    temp_dir
                )
            else:
                repo_path = Path(repo_path_or_url)

            if not repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process all Python files
            python_files = list(repo_path.rglob("*.py"))
            if not python_files:
                self.logger.warning("No Python files found in repository")
                return False

            success = True
            for file_path in python_files:
                relative_path = file_path.relative_to(repo_path)
                output_path = output_dir / relative_path.with_suffix('.md')
                
                if not await self.process_file(file_path, output_path):
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Repository processing failed: {e}")
            return False

        finally:
            # Cleanup temporary directory if used
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)

    async def cleanup(self) -> None:
        """Clean up all resources."""
        components = [
            (self.system_monitor, "System Monitor"),
            (self.repo_handler, "Repository Handler"),
            (self.ai_handler, "AI Handler"),
            (self.cache, "Cache"),
        ]

        for component, name in components:
            if component:
                try:
                    if hasattr(component, 'close'):
                        await component.close()
                    elif hasattr(component, '__aexit__'):
                        await component.__aexit__(None, None, None)
                    self.logger.debug(f"{name} cleaned up successfully")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name}: {e}")

        self._initialized = False

async def main(args: argparse.Namespace) -> int:
    """Main application entry point."""
    doc_generator = None
    try:
        # Initialize documentation generator
        config = AzureOpenAIConfig.from_env()
        doc_generator = DocumentationGenerator(config)
        await doc_generator.initialize()

        # Process based on arguments
        if args.repository:
            success = await doc_generator.process_repository(
                args.repository,
                Path(args.output)
            )
        elif args.files:
            success = True
            for file_path in args.files:
                output_path = Path(args.output) / Path(file_path).with_suffix('.md').name
                if not await doc_generator.process_file(Path(file_path), output_path):
                    success = False
        else:
            logger.error("No input specified")
            return 1

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

    finally:
        if doc_generator:
            await doc_generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="Local path or Git URL of the repository to process"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Python files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        exit_code = asyncio.run(main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
```

```python
# types.py
@dataclass
class DocumentationData:
    """Standardized documentation data structure."""
    module_info: Dict[str, str]  # Basic module information
    ai_content: Dict[str, Any]   # AI-generated content
    docstring_data: DocstringData  # Processed docstring data
    code_metadata: Dict[str, Any]  # Code analysis metadata
    source_code: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# ai_interaction.py
class AIInteractionHandler:
    async def process_code(self, extracted_info: Dict[str, Any], cache_key: Optional[str] = None) -> Optional[DocumentationData]:
        try:
            # Get AI response as before
            prompt = self._generate_prompt(extracted_info)
            completion = await self.client.chat.completions.create(...)
            parsed_response = await self.response_parser.parse_response(...)

            if parsed_response.validation_success:
                return DocumentationData(
                    module_info={
                        "name": extracted_info.get("module_name", "Unknown"),
                        "file_path": extracted_info.get("file_path", "Unknown")
                    },
                    ai_content=parsed_response.content,
                    docstring_data=self.docstring_processor.parse(parsed_response.content),
                    code_metadata=extracted_info,
                    source_code=extracted_info.get("source_code"),
                    metrics=extracted_info.get("metrics")
                )
            return None

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

# docs.py
class DocumentationOrchestrator:
    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        try:
            # Process code through AI handler
            doc_data = await self.ai_handler.process_code(
                context.metadata,
                cache_key=context.get_cache_key()
            )
            
            if not doc_data:
                raise DocumentationError("Failed to generate documentation data")

            # Validate the docstring data
            valid, errors = self.docstring_processor.validate(doc_data.docstring_data)
            if not valid:
                self.logger.warning(f"Docstring validation errors: {errors}")

            # Generate markdown using complete context
            markdown_context = {
                "module_name": doc_data.module_info["name"],
                "file_path": doc_data.module_info["file_path"],
                "description": doc_data.docstring_data.description,
                "classes": context.classes,
                "functions": context.functions,
                "constants": context.constants,
                "source_code": doc_data.source_code if context.include_source else None,
                "ai_documentation": doc_data.ai_content,
                "metrics": doc_data.metrics
            }

            documentation = self.markdown_generator.generate(markdown_context)
            return doc_data.source_code, documentation

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate documentation: {e}")
```


```python
# markdown_generator.py

class MarkdownGenerator:
    """Generates formatted markdown documentation."""

    def __init__(self):
        """Initialize markdown generator."""
        self.logger = LoggerSetup.get_logger(__name__)

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate formatted markdown documentation.
        
        Args:
            context: Documentation context including all metadata and content
            
        Returns:
            Formatted markdown string
        """
        sections = [
            self._generate_header(context),
            self._generate_overview(context),
            self._generate_installation(context),
            self._generate_usage(context),
            self._generate_api_reference(context),
            self._generate_examples(context),
            self._generate_metrics(context)
        ]

        return "\n\n".join(filter(None, sections))

    def _generate_header(self, context: Dict[str, Any]) -> str:
        """Generate module header section."""
        module_name = context.get("module_name", "Unknown Module")
        return (
            f"# {module_name}\n\n"
            f"**File Path:** `{context.get('file_path', 'Unknown')}`\n\n"
            f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"## Description\n\n{context.get('description', 'No description provided.')}"
        )

    def _generate_overview(self, context: Dict[str, Any]) -> str:
        """Generate module overview section."""
        ai_docs = context.get('ai_documentation', {})
        return (
            "## Overview\n\n"
            f"{ai_docs.get('summary', 'No summary available.')}\n\n"
            "### Key Features\n\n"
            f"{self._extract_key_features(ai_docs.get('description', ''))}\n\n"
            "### Dependencies\n\n"
            f"{self._format_dependencies(context.get('dependencies', {}))}"
        )

    def _generate_installation(self, context: Dict[str, Any]) -> str:
        """Generate installation instructions section."""
        dependencies = context.get('dependencies', {})
        if not dependencies:
            return ""

        requirements = []
        for dep_type, deps in dependencies.items():
            if deps:
                requirements.extend(deps)

        if not requirements:
            return ""

        return (
            "## Installation\n\n"
            "```bash\n"
            "pip install " + " ".join(requirements) + "\n"
            "```"
        )

    def _generate_api_reference(self, context: Dict[str, Any]) -> str:
        """Generate API reference section."""
        sections = ["## API Reference\n"]

        # Add classes
        if context.get('classes'):
            sections.append("### Classes\n")
            for cls in context['classes']:
                sections.append(self._format_class(cls))

        # Add functions
        if context.get('functions'):
            sections.append("### Functions\n")
            for func in context['functions']:
                sections.append(self._format_function(func))

        # Add constants
        if context.get('constants'):
            sections.append("### Constants\n")
            sections.append(self._format_constants(context['constants']))

        return "\n".join(sections)

    def _format_class(self, cls: Dict[str, Any]) -> str:
        """Format class documentation."""
        doc = [f"#### {cls['name']}\n"]
        
        # Add inheritance info
        if cls.get('bases'):
            doc.append(f"*Inherits from: {', '.join(cls['bases'])}*\n")

        # Add description
        if cls.get('docstring'):
            doc.append(f"{cls['docstring']}\n")

        # Add metrics/complexity warning if needed
        if cls.get('metrics', {}).get('complexity', 0) > 10:
            doc.append("> ⚠️ **High Complexity Warning**\n")

        # Add methods
        if cls.get('methods'):
            doc.append("**Methods:**\n")
            for method in cls['methods']:
                doc.append(self._format_method(method))

        # Add attributes
        if cls.get('attributes'):
            doc.append("**Attributes:**\n")
            for attr in cls['attributes']:
                doc.append(f"- `{attr['name']}` ({attr.get('type', 'Any')}): {attr.get('description', '')}")

        return "\n".join(doc)

    def _format_method(self, method: Dict[str, Any]) -> str:
        """Format method documentation."""
        signature = f"##### `{method['name']}("
        
        # Format arguments
        args = []
        for arg in method.get('args', []):
            arg_str = f"{arg['name']}: {arg.get('type', 'Any')}"
            if arg.get('default_value'):
                arg_str += f" = {arg['default_value']}"
            args.append(arg_str)
        
        signature += ", ".join(args)
        signature += f") -> {method.get('return_type', 'None')}`\n"

        doc = [signature]

        # Add description
        if method.get('description'):
            doc.append(f"{method['description']}\n")

        # Add argument details
        if method.get('args'):
            doc.append("**Arguments:**")
            for arg in method['args']:
                doc.append(f"- `{arg['name']}` ({arg.get('type', 'Any')}): {arg.get('description', '')}")
            doc.append("")

        # Add return info
        if method.get('returns'):
            doc.append(f"**Returns:** {method['returns'].get('description', '')}")

        # Add raises info
        if method.get('raises'):
            doc.append("\n**Raises:**")
            for exc in method['raises']:
                doc.append(f"- `{exc['exception']}`: {exc.get('description', '')}")

        return "\n".join(doc)

    def _format_function(self, func: Dict[str, Any]) -> str:
        """Format function documentation."""
        # Similar to _format_method but for standalone functions
        signature = f"#### `{func['name']}("
        args = []
        for arg in func.get('args', []):
            arg_str = f"{arg['name']}: {arg.get('type', 'Any')}"
            if arg.get('default_value'):
                arg_str += f" = {arg['default_value']}"
            args.append(arg_str)
        
        signature += ", ".join(args)
        signature += f") -> {func.get('return_type', 'None')}`\n"

        doc = [signature]

        # Add description
        if func.get('description'):
            doc.append(f"{func['description']}\n")

        # Add argument details
        if func.get('args'):
            doc.append("**Arguments:**")
            for arg in func['args']:
                doc.append(f"- `{arg['name']}` ({arg.get('type', 'Any')}): {arg.get('description', '')}")
            doc.append("")

        # Add return info
        if func.get('returns'):
            doc.append(f"**Returns:** {func['returns'].get('description', '')}")

        # Add raises info
        if func.get('raises'):
            doc.append("\n**Raises:**")
            for exc in func['raises']:
                doc.append(f"- `{exc['exception']}`: {exc.get('description', '')}")

        return "\n".join(doc)

    def _format_constants(self, constants: List[Dict[str, Any]]) -> str:
        """Format constants documentation."""
        doc = []
        for const in constants:
            doc.append(f"#### `{const['name']}`")
            if const.get('type'):
                doc.append(f"**Type:** `{const['type']}`")
            if const.get('value'):
                doc.append(f"**Value:** `{const['value']}`")
            if const.get('description'):
                doc.append(f"\n{const['description']}")
            doc.append("")
        return "\n".join(doc)

    def _generate_examples(self, context: Dict[str, Any]) -> str:
        """Generate examples section."""
        examples = context.get('examples', [])
        if not examples:
            return ""

        doc = ["## Examples\n"]
        for i, example in enumerate(examples, 1):
            doc.append(f"### Example {i}")
            if example.get('description'):
                doc.append(example['description'])
            if example.get('code'):
                doc.append("```python")
                doc.append(example['code'])
                doc.append("```")
            if example.get('output'):
                doc.append("\nOutput:")
                doc.append("```")
                doc.append(example['output'])
                doc.append("```")
            doc.append("")

        return "\n".join(doc)

    def _generate_metrics(self, context: Dict[str, Any]) -> str:
        """Generate metrics section."""
        metrics = context.get('metrics', {})
        if not metrics:
            return ""

        doc = ["## Code Metrics\n"]
        
        # Maintainability index
        if 'maintainability' in metrics:
            doc.append(f"- **Maintainability Index:** {metrics['maintainability']:.2f}")
            doc.append(self._get_maintainability_badge(metrics['maintainability']))

        # Complexity metrics
        if 'complexity' in metrics:
            doc.append(f"- **Cyclomatic Complexity:** {metrics['complexity']}")
            if metrics['complexity'] > 10:
                doc.append("  > ⚠️ **High Complexity Warning**")

        # Additional metrics
        if 'lines' in metrics:
            doc.append(f"- **Lines of Code:** {metrics['lines']}")
        if 'classes' in metrics:
            doc.append(f"- **Number of Classes:** {metrics['classes']}")
        if 'functions' in metrics:
            doc.append(f"- **Number of Functions:** {metrics['functions']}")

        return "\n".join(doc)

    def _get_maintainability_badge(self, index: float) -> str:
        """Get maintainability badge based on index."""
        if index >= 80:
            return "![Maintainability A](https://img.shields.io/badge/maintainability-A-brightgreen)"
        elif index >= 60:
            return "![Maintainability B](https://img.shields.io/badge/maintainability-B-yellow)"
        else:
            return "![Maintainability C](https://img.shields.io/badge/maintainability-C-red)"

    def _extract_key_features(self, description: str) -> str:
        """Extract key features from description."""
        # Simple extraction of bullet points or key phrases
        features = []
        for line in description.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                features.append(line)
            elif line.endswith('.') and len(line.split()) < 15:
                features.append(f"- {line}")

        if not features:
            return "- No specific features listed"

        return "\n".join(features)

    def _format_dependencies(self, dependencies: Dict[str, Set[str]]) -> str:
        """Format dependencies section."""
        if not dependencies:
            return "No dependencies listed."

        sections = []
        if 'stdlib' in dependencies and dependencies['stdlib']:
            sections.append("**Standard Library:**\n" + 
                          "\n".join(f"- `{dep}`" for dep in sorted(dependencies['stdlib'])))
        
        if 'third_party' in dependencies and dependencies['third_party']:
            sections.append("**Third Party:**\n" + 
                          "\n".join(f"- `{dep}`" for dep in sorted(dependencies['third_party'])))
        
        if 'local' in dependencies and dependencies['local']:
            sections.append("**Local:**\n" + 
                          "\n".join(f"- `{dep}`" for dep in sorted(dependencies['local'])))

        return "\n\n".join(sections)
```

```python
# token_management.py
class TokenManager:
    """Manages all token-related operations for Azure OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4",
        deployment_id: Optional[str] = None,
        config: Optional[AzureOpenAIConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config or AzureOpenAIConfig.from_env()
        self.model = self._get_model_name(deployment_id, model)
        self.deployment_id = deployment_id
        self.metrics_collector = metrics_collector

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model, self.config.model_limits["gpt-4"]
        )

        # Initialize counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def validate_and_prepare_request(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate and prepare a request with token management.

        Args:
            prompt: The prompt to send to the API
            max_tokens: Optional maximum tokens for completion
            temperature: Optional temperature setting

        Returns:
            Dict containing the validated request parameters

        Raises:
            TokenLimitError: If request exceeds token limits
        """
        try:
            prompt_tokens = self.estimate_tokens(prompt)
            max_completion = max_tokens or (self.model_config["chunk_size"] - prompt_tokens)
            total_tokens = prompt_tokens + max_completion

            if total_tokens > self.model_config["max_tokens"]:
                raise TokenLimitError("Request exceeds model token limit")

            request_params = {
                "model": self.deployment_id or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or 0.3
            }

            # Track the request
            self.track_request(prompt_tokens, max_completion)

            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}")
            raise

    async def process_completion(
        self, 
        completion: Any
    ) -> Tuple[str, Dict[str, int]]:
        """Process completion response and track token usage.

        Args:
            completion: The completion response from the API

        Returns:
            Tuple of (completion content, usage statistics)
        """
        try:
            content = completion.choices[0].message.content
            usage = completion.usage.model_dump() if hasattr(completion, 'usage') else {}

            # Update token counts
            if usage:
                self.total_completion_tokens += usage.get('completion_tokens', 0)
                self.total_prompt_tokens += usage.get('prompt_tokens', 0)

                if self.metrics_collector:
                    await self.metrics_collector.track_operation(
                        "token_usage",
                        True,
                        0,
                        usage,
                        metadata={
                            "model": self.model,
                            "deployment_id": self.deployment_id
                        }
                    )

            return content, usage

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}")
            raise

    @lru_cache(maxsize=1024)
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            tokens = len(self.encoding.encode(text))
            return tokens
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}")
            raise

# ai_interaction.py
class AIInteractionHandler:
    """Handles AI interactions, delegating token management to TokenManager."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.token_manager = token_manager
            self.response_parser = response_parser
            self.metrics = metrics
            
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    async def process_code(
        self,
        extracted_info: Dict[str, Any],
        cache_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process code information and generate documentation."""
        try:
            # Check cache
            if cache_key and self.cache:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return cached_result

            # Generate prompt
            prompt = self._generate_prompt(extracted_info)

            # Let TokenManager handle token validation and request preparation
            request_params = await self.token_manager.validate_and_prepare_request(prompt)

            # Make API request
            completion = await self.client.chat.completions.create(**request_params)

            # Process completion through TokenManager
            content, usage = await self.token_manager.process_completion(completion)

            if not content:
                raise ValidationError("Empty response from AI service")

            # Parse and validate
            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format="docstring",
                validate_schema=True,
            )

            if not parsed_response.validation_success:
                self.logger.warning(f"Response validation had errors: {parsed_response.errors}")
                return None

            # Cache if enabled
            if cache_key and self.cache:
                await self.cache.save_docstring(cache_key, parsed_response.content)

            return parsed_response.content

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None
```


```python
"""AI Interaction Handler Module.

Manages interactions with Azure OpenAI API, focusing on prompt generation
and API communication with validated data.
"""

from typing import Any, Dict, Optional, Tuple
from openai import AsyncAzureOpenAI

from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.response_parsing import ResponseParsingService
from exceptions import ExtractionError, ValidationError

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """Handler for AI interactions with Azure OpenAI API.
    
    Focuses on dynamic prompt generation and processing AI responses.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.response_parser = response_parser
            self.metrics = metrics
            
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    async def process_code(
        self,
        extracted_info: Dict[str, Any],
        cache_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process code information and generate documentation.

        Args:
            extracted_info: Pre-processed code information from extractors
            cache_key: Optional cache key for storing results

        Returns:
            Optional[Dict[str, Any]]: AI-generated documentation data
        """
        try:
            # Check cache first
            if cache_key and self.cache:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return cached_result

            # Generate dynamic prompt based on extracted info
            prompt = self._generate_prompt(extracted_info)

            # Get AI response
            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )

            if not completion.choices[0].message.content:
                raise ValidationError("Empty response from AI service")

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response=completion.choices[0].message.content,
                expected_format="docstring",
                validate_schema=True,
            )

            if not parsed_response.validation_success:
                self.logger.warning(f"Response validation had errors: {parsed_response.errors}")
                return None

            # Cache the result if enabled
            if cache_key and self.cache:
                await self.cache.save_docstring(cache_key, parsed_response.content)

            return parsed_response.content

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _generate_prompt(self, extracted_info: Dict[str, Any]) -> str:
        """Generate a dynamic prompt based on extracted code information.

        Args:
            extracted_info: Extracted code information and metadata

        Returns:
            str: Generated prompt for AI service
        """
        context_blocks = []

        # Add code context
        if "source" in extracted_info:
            context_blocks.append(
                "CODE TO DOCUMENT:\n"
                "```python\n"
                f"{extracted_info['source']}\n"
                "```\n"
            )

        # Add existing docstring context if available
        if "existing_docstring" in extracted_info:
            context_blocks.append(
                "EXISTING DOCUMENTATION:\n"
                f"{extracted_info['existing_docstring']}\n"
            )

        # Add code complexity information
        if "metrics" in extracted_info:
            metrics = extracted_info["metrics"]
            context_blocks.append(
                "CODE METRICS:\n"
                f"- Complexity: {metrics.get('complexity', 'N/A')}\n"
                f"- Maintainability: {metrics.get('maintainability_index', 'N/A')}\n"
            )

        # Add function/method context
        if "args" in extracted_info:
            args_info = "\n".join(
                f"- {arg['name']}: {arg['type']}"
                for arg in extracted_info["args"]
            )
            context_blocks.append(
                "FUNCTION ARGUMENTS:\n"
                f"{args_info}\n"
            )

        # Add return type information
        if "returns" in extracted_info:
            context_blocks.append(
                "RETURN TYPE:\n"
                f"{extracted_info['returns']['type']}\n"
            )

        # Add exception information
        if "raises" in extracted_info:
            raises_info = "\n".join(
                f"- {exc['exception']}"
                for exc in extracted_info["raises"]
            )
            context_blocks.append(
                "EXCEPTIONS RAISED:\n"
                f"{raises_info}\n"
            )

        # Combine all context blocks
        context = "\n\n".join(context_blocks)

        # Add the base prompt template
        prompt_template = (
            "Generate documentation for the provided code as a JSON object.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "```json\n"
            "{\n"
            '  "summary": "A brief one-line summary of the function/method",\n'
            '  "description": "Detailed description of the functionality",\n'
            '  "args": [\n'
            "    {\n"
            '      "name": "string - parameter name",\n'
            '      "type": "string - parameter data type",\n'
            '      "description": "string - brief description of the parameter"\n'
            "    }\n"
            "  ],\n"
            '  "returns": {\n'
            '    "type": "string - return data type",\n'
            '    "description": "string - brief description of return value"\n'
            "  },\n"
            '  "raises": [\n'
            "    {\n"
            '      "exception": "string - exception class name",\n'
            '      "description": "string - circumstances under which raised"\n'
            "    }\n"
            "  ],\n"
            '  "complexity": "integer - McCabe complexity score"\n'
            "}\n"
            "```\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "IMPORTANT NOTES:\n"
            "1. Always include a 'complexity' field with an integer value\n"
            "2. If complexity > 10, note this in the description with [WARNING]\n"
            "3. Never set complexity to null or omit it\n"
            "4. Provide detailed, specific descriptions\n"
            "5. Ensure all type hints are accurate Python types\n\n"
            "Respond with only the JSON object. Do not include any other text."
        )

        return prompt_template

    async def close(self) -> None:
        """Close and clean up resources."""
        try:
            if self.client:
                await self.client.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        await self.close()
```


```python
"""
Documentation generation orchestration module.
Coordinates documentation processes between various components.
"""

from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from core.logger import LoggerSetup
from core.types import DocumentationContext
from core.markdown_generator import MarkdownGenerator
from exceptions import DocumentationError

logger = LoggerSetup.get_logger(__name__)

class DocumentationOrchestrator:
    """Orchestrates the documentation generation process."""

    def __init__(self, ai_handler, docstring_processor, markdown_generator: Optional[MarkdownGenerator] = None):
        """
        Initialize the documentation orchestrator.

        Args:
            ai_handler: Handler for AI interactions
            docstring_processor: Processor for docstrings
            markdown_generator: Generator for markdown output
        """
        self.logger = LoggerSetup.get_logger(__name__)
        self.ai_handler = ai_handler
        self.docstring_processor = docstring_processor
        self.markdown_generator = markdown_generator or MarkdownGenerator()

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """
        Generate documentation for the given context.

        Args:
            context: Documentation context containing source code and metadata

        Returns:
            Tuple[str, str]: Updated source code and generated documentation

        Raises:
            DocumentationError: If documentation generation fails
        """
        try:
            self.logger.info("Starting documentation generation")
            start_time = datetime.now()

            # Process code through AI handler
            result = await self.ai_handler.process_code(
                context.source_code,
                extracted_info=context.metadata,
                context=context
            )
            
            if not result:
                raise DocumentationError("AI processing failed")
            
            updated_code, ai_docs = result

            # Validate and process docstrings
            docstring_data = self.docstring_processor.parse(ai_docs)
            valid, errors = self.docstring_processor.validate(docstring_data)
            
            if not valid:
                self.logger.warning(f"Docstring validation errors: {errors}")

            # Generate markdown documentation
            markdown_context = {
                "module_name": context.metadata.get("module_name", "Unknown Module"),
                "file_path": context.metadata.get("file_path", "Unknown Path"),
                "description": docstring_data.description,
                "classes": context.classes,
                "functions": context.functions,
                "constants": context.constants,
                "changes": context.changes,
                "source_code": updated_code if context.include_source else None,
                "ai_documentation": ai_docs
            }

            documentation = self.markdown_generator.generate(markdown_context)

            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Documentation generated in {processing_time:.2f} seconds")

            return updated_code, documentation

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate documentation: {e}")

    async def generate_module_documentation(self, file_path: Path, output_dir: Path) -> None:
        """
        Generate documentation for a single module.

        Args:
            file_path: Path to the Python file
            output_dir: Output directory for documentation

        Raises:
            DocumentationError: If module documentation generation fails
        """
        try:
            self.logger.info(f"Generating documentation for {file_path}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.with_suffix('.md').name

            # Read source code
            source_code = file_path.read_text(encoding='utf-8')

            # Create documentation context
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat()
                }
            )

            # Generate documentation
            _, documentation = await self.generate_documentation(context)

            # Write output
            output_path.write_text(documentation, encoding='utf-8')
            self.logger.info(f"Documentation written to {output_path}")

        except Exception as e:
            self.logger.error(f"Module documentation generation failed: {e}")
            raise DocumentationError(f"Failed to generate module documentation: {e}")

    async def __aenter__(self) -> 'DocumentationOrchestrator':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass
```

```python
# docstring_processor.py
class DocstringProcessor:
    """Processes docstrings by parsing and validating them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """Initialize docstring processor."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a raw docstring into structured format."""
        try:
            if isinstance(docstring, dict):
                # Ensure 'returns' is a dictionary, providing a default if needed
                returns = docstring.get('returns')
                if not isinstance(returns, dict):
                    returns = {'type': 'Any', 'description': ''}
                    docstring['returns'] = returns

                return DocstringData(
                    summary=docstring.get('summary', ''),
                    description=docstring.get('description', ''),
                    args=docstring.get('args', []),
                    returns=returns,
                    raises=docstring.get('raises', []),
                    complexity=docstring.get('complexity', 1)
                )

            # If it's a string, try to parse as JSON
            if isinstance(docstring, str) and docstring.strip().startswith('{'):
                try:
                    doc_dict = json.loads(docstring)
                    return self.parse(doc_dict)
                except json.JSONDecodeError:
                    pass

            # Parse as a regular docstring string
            parsed = parse_docstring(docstring)
            return DocstringData(
                summary=parsed.short_description or '',
                description=parsed.long_description or '',
                args=[{
                    'name': param.arg_name,
                    'type': param.type_name or 'Any',
                    'description': param.description or ''
                } for param in parsed.params],
                returns={
                    'type': parsed.returns.type_name if parsed.returns else 'Any',
                    'description': parsed.returns.description if parsed.returns else ''
                },
                raises=[{
                    'exception': e.type_name or 'Exception',
                    'description': e.description or ''
                } for e in parsed.raises] if parsed.raises else []
            )

        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise DocumentationError(f"Failed to parse docstring: {e}")

    def format(self, data: DocstringData) -> str:
        """Format structured docstring data into a string."""
        lines = []

        if data.summary:
            lines.extend([data.summary, ""])

        if data.description and data.description != data.summary:
            lines.extend([data.description, ""])

        if data.args:
            lines.append("Args:")
            for arg in data.args:
                arg_desc = f"    {arg['name']} ({arg['type']}): {arg['description']}"
                if arg.get('optional', False):
                    arg_desc += " (Optional)"
                if 'default_value' in arg:
                    arg_desc += f", default: {arg['default_value']}"
                lines.append(arg_desc)
            lines.append("")

        if data.returns:
            lines.append("Returns:")
            lines.append(f"    {data.returns['type']}: {data.returns['description']}")
            lines.append("")

        if data.raises:
            lines.append("Raises:")
            for exc in data.raises:
                lines.append(f"    {exc['exception']}: {exc['description']}")
            lines.append("")

        return "\n".join(lines).strip()

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate docstring data against requirements."""
        errors = []
        
        # Required fields validation
        if not data.summary:
            errors.append("Missing summary")
        if not data.description:
            errors.append("Missing description")
            
        # Args validation
        for arg in data.args:
            if not arg.get('name'):
                errors.append("Missing argument name")
            if not arg.get('type'):
                errors.append(f"Missing type for argument {arg.get('name', 'unknown')}")
            if not arg.get('description'):
                errors.append(f"Missing description for argument {arg.get('name', 'unknown')}")
                
        # Returns validation
        if not data.returns:
            errors.append("Missing returns information")
        elif not data.returns.get('type'):
            errors.append("Missing return type")
        elif not data.returns.get('description'):
            errors.append("Missing return description")
            
        return len(errors) == 0, errors

# code_extractor.py
class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def extract_docstring_info(self, node: ast.AST) -> Dict[str, Any]:
        """Extract docstring and related metadata from an AST node."""
        try:
            docstring = ast.get_docstring(node) or ""
            
            # Extract argument information
            args = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = self._extract_function_args(node)
            
            # Extract return information
            returns = self._extract_return_info(node)
            
            # Extract raised exceptions
            raises = self._extract_raises(node)
            
            # Calculate complexity
            complexity = self.metrics_calculator.calculate_complexity(node)
            
            return {
                'docstring': docstring,
                'args': args,
                'returns': returns,
                'raises': raises,
                'complexity': complexity,
                'metadata': {
                    'lineno': getattr(node, 'lineno', None),
                    'name': getattr(node, 'name', None),
                    'type': type(node).__name__
                }
            }
        except Exception as e:
            self.logger.error(f"Error extracting docstring info: {e}")
            raise

    def _extract_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract function argument information."""
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'type': self.ast_utils.get_name(arg.annotation) if arg.annotation else 'Any',
                'description': '',  # Will be filled by docstring processor
            }
            # Handle default values
            if node.args.defaults:
                default_index = len(node.args.args) - len(node.args.defaults)
                if node.args.args.index(arg) >= default_index:
                    default_value = self.ast_utils.get_name(
                        node.args.defaults[node.args.args.index(arg) - default_index]
                    )
                    arg_info['default_value'] = default_value
                    arg_info['optional'] = True
            args.append(arg_info)
        return args

    def _extract_return_info(self, node: ast.AST) -> Dict[str, Any]:
        """Extract return type information."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            return {
                'type': self.ast_utils.get_name(node.returns),
                'description': ''  # Will be filled by docstring processor
            }
        return {'type': 'Any', 'description': ''}

    def _extract_raises(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extract exception information from the AST."""
        raises = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                exc_name = self._get_exception_name(child.exc)
                if exc_name:
                    raises.add(exc_name)
        return [{'exception': exc, 'description': ''} for exc in raises]
```


```python
# ai_interaction.py changes
class AIInteractionHandler:
    def __init__(self, config: Optional[AzureOpenAIConfig] = None,
                 cache: Optional[Cache] = None,
                 token_manager: Optional[TokenManager] = None,
                 response_parser: Optional[ResponseParsingService] = None,
                 metrics: Optional[Metrics] = None) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.metrics = metrics or Metrics()
            self.context = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False,
            )
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.token_manager = token_manager
            self.response_parser = response_parser
            self.code_extractor = CodeExtractor(context=self.context)
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    async def process_code(self, source_code: str, cache_key: Optional[str] = None,
                          extracted_info: Optional[Dict[str, Any]] = None,
                          context: Optional[DocumentationContext] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            # Check cache first if enabled
            if cache_key and self.cache:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return cached_result.get("updated_code"), cached_result.get("documentation")

            # Process and validate source code
            processed_code = source_code.strip()
            tree = ast.parse(processed_code)

            # Extract metadata if not provided
            if not extracted_info:
                extraction_result = self.code_extractor.extract_code(processed_code, self.context)
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    "module_docstring": extraction_result.module_docstring,
                    "metrics": extraction_result.metrics,
                }

            # Generate prompt and get AI response
            prompt = self._create_function_calling_prompt(processed_code, extracted_info)
            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )

            # Parse and validate the response
            parsed_response = await self.response_parser.parse_response(
                response=completion.choices[0].message.content,
                expected_format="docstring",
                validate_schema=True
            )

            if not parsed_response.validation_success:
                raise ValidationError("Response validation failed")

            # Process the docstring data
            docstring_data = self.docstring_processor.parse(parsed_response.content)

            # Update the AST with new docstrings
            modified_tree = self._update_ast_with_docstrings(tree, docstring_data)
            updated_code = ast.unparse(modified_tree)

            # Prepare AI documentation
            ai_documentation = self._prepare_ai_documentation(docstring_data)

            # Update cache if enabled
            if cache_key and self.cache:
                await self.cache.save_docstring(cache_key, {
                    "updated_code": updated_code,
                    "documentation": ai_documentation,
                })

            return updated_code, ai_documentation

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _update_ast_with_docstrings(self, tree: ast.AST, docstring_data: DocstringData) -> ast.AST:
        """Update AST with new docstrings using DocstringProcessor."""
        tree = self.docstring_processor.insert_docstring(tree, 
            self.docstring_processor.format(docstring_data))
        return tree

    def _prepare_ai_documentation(self, docstring_data: DocstringData) -> Dict[str, Any]:
        """Prepare AI documentation from docstring data."""
        return {
            "summary": docstring_data.summary or "No summary provided",
            "description": docstring_data.description or "No description provided",
            "args": docstring_data.args or [],
            "returns": docstring_data.returns or {"type": "Any", "description": ""},
            "raises": docstring_data.raises or [],
            "complexity": docstring_data.complexity or 1,
        }

# metrics.py additions
class Metrics:
    def calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate all metrics for a class."""
        return {
            'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
            'complexity': self.calculate_complexity(node),
            'maintainability': self.calculate_maintainability_index(node),
            'inheritance_depth': self._calculate_inheritance_depth(node),
            'attribute_count': len(self._get_attributes(node))
        }

    def calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate all metrics for a function."""
        return {
            'cyclomatic_complexity': self.calculate_cyclomatic_complexity(node),
            'cognitive_complexity': self.calculate_cognitive_complexity(node),
            'maintainability_index': self.calculate_maintainability_index(node),
            'parameter_count': len(node.args.args),
            'return_complexity': self._calculate_return_complexity(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }

    def calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate all metrics for a module."""
        return {
            'complexity': self.calculate_complexity(tree),
            'maintainability': self.calculate_maintainability_index(tree),
            'lines': len(ast.unparse(tree).splitlines()),
            'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'functions': len([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
        }

# Update extractors to use centralized metrics
class ClassExtractor:
    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        return self.metrics_calculator.calculate_class_metrics(node)

class FunctionExtractor:
    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        return self.metrics_calculator.calculate_function_metrics(node)

class CodeExtractor:
    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        return self.metrics_calculator.calculate_module_metrics(tree)
```




Hello! It looks like you've shared code related to generating documentation using the Azure OpenAI service, involving classes like `AIInteractionHandler`, `DocumentationGenerator`, and others.

Could you please provide more details on how I can assist you with this code? For example:

- **Are you encountering any errors or issues when running this code?** If so, please share the error messages or describe the unexpected behavior.
- **Do you need help optimizing or refactoring certain parts of the code?**
- **Are you looking for a code review or suggestions on best practices?**
- **Do you have specific questions about how certain components work or how to integrate them?**

Providing additional context will help me give you a more precise and helpful response.