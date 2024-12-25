```python
# config.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import json
from dotenv import load_dotenv
from core.logger import LoggerSetup, log_operation

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ValidationError(ConfigError):
    """Exception for configuration validation failures."""
    pass

class EnvironmentError(ConfigError):
    """Exception for environment configuration errors."""
    pass

@dataclass
class APICredentials:
    """Configuration for API credentials."""
    api_key: str
    endpoint: Optional[str] = None
    org_id: Optional[str] = None
    deployment_name: Optional[str] = None

    def validate(self) -> None:
        """Validate credential configuration."""
        if not self.api_key:
            raise ValidationError("API key is required")
        if self.endpoint and not self.endpoint.startswith(('http://', 'https://')):
            raise ValidationError("Invalid endpoint URL format")

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int = 3
    retry_delay: float = 1.0

    def validate(self) -> None:
        """Validate rate limit configuration."""
        if self.requests_per_minute <= 0:
            raise ValidationError("requests_per_minute must be positive")
        if self.tokens_per_minute <= 0:
            raise ValidationError("tokens_per_minute must be positive")
        if self.max_retries < 0:
            raise ValidationError("max_retries cannot be negative")
        if self.retry_delay <= 0:
            raise ValidationError("retry_delay must be positive")

@dataclass
class ServiceConfig:
    """Configuration for individual services."""
    credentials: APICredentials
    rate_limits: RateLimitConfig
    endpoints: Dict[str, str] = field(default_factory=dict)
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate service configuration."""
        self.credentials.validate()
        self.rate_limits.validate()
        for endpoint in self.endpoints.values():
            if not endpoint.startswith(('http://', 'https://')):
                raise ValidationError(f"Invalid endpoint URL format: {endpoint}")

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    services: Dict[str, ServiceConfig]
    cache_enabled: bool = True
    cache_ttl: int = 3600
    embedding_dimension: int = 1536
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 3

    def validate(self) -> None:
        """Validate RAG configuration."""
        if not self.services:
            raise ValidationError("At least one service must be configured")
        for service in self.services.values():
            service.validate()
        if self.cache_ttl <= 0:
            raise ValidationError("cache_ttl must be positive")
        if self.embedding_dimension <= 0:
            raise ValidationError("embedding_dimension must be positive")
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValidationError("chunk_overlap cannot be negative")
        if self.retriever_k <= 0:
            raise ValidationError("retriever_k must be positive")

class ConfigurationManager:
    """Enhanced configuration management with validation."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("configuration")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self._config: Dict[str, Any] = {}
        self._load_environment()
    
    @log_operation("load_environment")
    def _load_environment(self) -> None:
        """Load environment configuration."""
        try:
            # Load environment variables
            env_file = {
                "development": ".env.development",
                "staging": ".env.staging",
                "production": ".env.production"
            }.get(self.environment, ".env")
            
            if not os.path.exists(env_file):
                raise EnvironmentError(f"Environment file not found: {env_file}")
            
            load_dotenv(env_file)
            self.logger.info(f"Loaded environment from {env_file}")
            
        except Exception as e:
            raise EnvironmentError(f"Failed to load environment: {str(e)}")
    
    @log_operation("load_configuration")
    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigError: If configuration loading fails
        """
        try:
            # Load base configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load environment overrides
            env_config_path = f"{config_path}.{self.environment}"
            if os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    env_config = json.load(f)
                config = self._merge_configurations(config, env_config)
            
            # Build RAG configuration
            rag_config = self._build_rag_config(config)
            
            # Validate configuration
            rag_config.validate()
            
            self._config = config
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}")
    
    def _merge_configurations(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge base and override configurations."""
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._merge_configurations(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _build_rag_config(self, config: Dict[str, Any]) -> RAGConfig:
        """Build RAG configuration from raw config."""
        services = {}
        
        for service_name, service_config in config.get("services", {}).items():
            services[service_name] = ServiceConfig(
                credentials=APICredentials(
                    api_key=os.getenv(f"{service_name.upper()}_API_KEY"),
                    endpoint=service_config.get("endpoint"),
                    org_id=service_config.get("org_id"),
                    deployment_name=service_config.get("deployment_name")
                ),
                rate_limits=RateLimitConfig(
                    requests_per_minute=service_config["rate_limits"]["requests_per_minute"],
                    tokens_per_minute=service_config["rate_limits"]["tokens_per_minute"]
                ),
                endpoints=service_config.get("endpoints", {}),
                model_configs=service_config.get("model_configs", {})
            )
        
        return RAGConfig(
            services=services,
            cache_enabled=config.get("cache_enabled", True),
            cache_ttl=config.get("cache_ttl", 3600),
            embedding_dimension=config.get("embedding_dimension", 1536),
            chunk_size=config.get("chunk_size", 500),
            chunk_overlap=config.get("chunk_overlap", 50),
            retriever_k=config.get("retriever_k", 3)
        )
    
    def get_service_config(self, service: str) -> ServiceConfig:
        """
        Get configuration for a specific service.
        
        Args:
            service: Service name
            
        Returns:
            ServiceConfig: Service configuration
            
        Raises:
            ConfigError: If service is not configured
        """
        if not self._config:
            raise ConfigError("Configuration not loaded")
        
        try:
            return self._build_rag_config(self._config).services[service]
        except KeyError:
            raise ConfigError(f"Service not configured: {service}")
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        if not self._config:
            raise ConfigError("Configuration not loaded")
        
        try:
            current = self._config
            for part in key.split('.'):
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def validate_required_settings(self, required_settings: List[str]) -> None:
        """
        Validate required configuration settings.
        
        Args:
            required_settings: List of required setting keys
            
        Raises:
            ValidationError: If required settings are missing
        """
        missing_settings = []
        
        for setting in required_settings:
            if self.get_value(setting) is None:
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValidationError(
                f"Missing required configuration settings: {', '.join(missing_settings)}"
            )
    
    @log_operation("reload_configuration")
    def reload_configuration(self) -> None:
        """Reload configuration from current sources."""
        if not self._config:
            raise ConfigError("No configuration loaded to reload")
        
        current_config = self._config.copy()
        try:
            self._load_environment()
            self.load_configuration(current_config["_source_path"])
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self._config = current_config
            raise ConfigError(f"Failed to reload configuration: {str(e)}")
```