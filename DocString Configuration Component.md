```python
"""
DocString Configuration

This module provides configuration management for the docstring system,
including style preferences and preservation settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
from enum import Enum

class DocStringStyle(Enum):
    """Enumeration of supported docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    CUSTOM = "custom"

@dataclass
class FormatConfig:
    """Configuration for docstring formatting."""
    style: DocStringStyle = DocStringStyle.GOOGLE
    indentation: int = 4
    line_length: int = 80
    blank_lines_between_sections: int = 1
    section_order: List[str] = field(default_factory=lambda: [
        "Description",
        "Args",
        "Returns",
        "Raises",
        "Yields",
        "Examples",
        "Notes"
    ])

@dataclass
class PreservationConfig:
    """Configuration for docstring content preservation."""
    enabled: bool = True
    storage_dir: Optional[Path] = None
    preserve_custom_sections: bool = True
    preserve_decorators: bool = True
    preserve_examples: bool = True
    ttl_days: int = 30

@dataclass
class ValidationConfig:
    """Configuration for docstring validation."""
    enforce_style: bool = True
    require_description: bool = True
    require_param_description: bool = True
    require_return_description: bool = True
    require_examples: bool = False
    max_description_length: int = 1000
    min_description_length: int = 10

class DocStringConfig:
    """
    Configuration manager for the docstring system.
    
    This class manages all configuration aspects of the docstring system,
    including loading from files and providing defaults.
    """
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        format_config: Optional[FormatConfig] = None,
        preservation_config: Optional[PreservationConfig] = None,
        validation_config: Optional[ValidationConfig] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to a configuration file
            format_config: Optional format configuration
            preservation_config: Optional preservation configuration
            validation_config: Optional validation configuration
        """
        self.format_config = format_config or FormatConfig()
        self.preservation_config = preservation_config or PreservationConfig()
        self.validation_config = validation_config or ValidationConfig()
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: Path):
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to the configuration file (JSON or YAML)
        
        Raises:
            ValueError: If the configuration file is invalid
        """
        try:
            config_data = self._load_file(config_file)
            
            # Update format config
            if 'format' in config_data:
                format_data = config_data['format']
                self.format_config = FormatConfig(
                    style=DocStringStyle(format_data.get('style', self.format_config.style.value)),
                    indentation=format_data.get('indentation', self.format_config.indentation),
                    line_length=format_data.get('line_length', self.format_config.line_length),
                    blank_lines_between_sections=format_data.get(
                        'blank_lines_between_sections',
                        self.format_config.blank_lines_between_sections
                    ),
                    section_order=format_data.get('section_order', self.format_config.section_order)
                )
            
            # Update preservation config
            if 'preservation' in config_data:
                pres_data = config_data['preservation']
                self.preservation_config = PreservationConfig(
                    enabled=pres_data.get('enabled', self.preservation_config.enabled),
                    storage_dir=Path(pres_data['storage_dir']) if 'storage_dir' in pres_data else self.preservation_config.storage_dir,
                    preserve_custom_sections=pres_data.get(
                        'preserve_custom_sections',
                        self.preservation_config.preserve_custom_sections
                    ),
                    preserve_decorators=pres_data.get(
                        'preserve_decorators',
                        self.preservation_config.preserve_decorators
                    ),
                    preserve_examples=pres_data.get(
                        'preserve_examples',
                        self.preservation_config.preserve_examples
                    ),
                    ttl_days=pres_data.get('ttl_days', self.preservation_config.ttl_days)
                )
            
            # Update validation config
            if 'validation' in config_data:
                val_data = config_data['validation']
                self.validation_config = ValidationConfig(
                    enforce_style=val_data.get('enforce_style', self.validation_config.enforce_style),
                    require_description=val_data.get(
                        'require_description',
                        self.validation_config.require_description
                    ),
                    require_param_description=val_data.get(
                        'require_param_description',
                        self.validation_config.require_param_description
                    ),
                    require_return_description=val_data.get(
                        'require_return_description',
                        self.validation_config.require_return_description
                    ),
                    require_examples=val_data.get(
                        'require_examples',
                        self.validation_config.require_examples
                    ),
                    max_description_length=val_data.get(
                        'max_description_length',
                        self.validation_config.max_description_length
                    ),
                    min_description_length=val_data.get(
                        'min_description_length',
                        self.validation_config.min_description_length
                    )
                )
        
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _load_file(self, config_file: Path) -> Dict[str, Any]:
        """Load and parse a configuration file."""
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file) as f:
            if config_file.suffix == '.json':
                return json.load(f)
            elif config_file.suffix in ('.yml', '.yaml'):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format")

    def save_config(self, config_file: Path):
        """
        Save current configuration to a file.
        
        Args:
            config_file: Path to save the configuration
        """
        config_data = {
            'format': {
                'style': self.format_config.style.value,
                'indentation': self.format_config.indentation,
                'line_length': self.format_config.line_length,
                'blank_lines_between_sections': self.format_config.blank_lines_between_sections,
                'section_order': self.format_config.section_order
            },
            'preservation': {
                'enabled': self.preservation_config.enabled,
                'storage_dir': str(self.preservation_config.storage_dir) if self.preservation_config.storage_dir else None,
                'preserve_custom_sections': self.preservation_config.preserve_custom_sections,
                'preserve_decorators': self.preservation_config.preserve_decorators,
                'preserve_examples': self.preservation_config.preserve_examples,
                'ttl_days': self.preservation_config.ttl_days
            },
            'validation': {
                'enforce_style': self.validation_config.enforce_style,
                'require_description': self.validation_config.require_description,
                'require_param_description': self.validation_config.require_param_description,
                'require_return_description': self.validation_config.require_return_description,
                'require_examples': self.validation_config.require_examples,
                'max_description_length': self.validation_config.max_description_length,
                'min_description_length': self.validation_config.min_description_length
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix == '.json':
                    json.dump(config_data, f, indent=2)
                elif config_file.suffix in ('.yml', '.yaml'):
                    yaml.safe_dump(config_data, f, default_flow_style=False)
                else:
                    raise ValueError("Unsupported configuration file format")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def get_style_guide(self) -> StyleGuide:
        """
        Create a StyleGuide instance from the current format configuration.
        
        Returns:
            StyleGuide: A configured style guide instance
        """
        return StyleGuide(
            indentation=self.format_config.indentation,
            section_order=self.format_config.section_order,
            format_rules={
                StyleRule.INDENTATION: self.format_config.indentation,
                StyleRule.SPACING: self.format_config.blank_lines_between_sections,
                StyleRule.FORMAT: self.format_config.style
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary
        """
        return {
            'format': {
                'style': self.format_config.style.value,
                'indentation': self.format_config.indentation,
                'line_length': self.format_config.line_length,
                'blank_lines_between_sections': self.format_config.blank_lines_between_sections,
                'section_order': self.format_config.section_order
            },
            'preservation': {
                'enabled': self.preservation_config.enabled,
                'storage_dir': str(self.preservation_config.storage_dir) if self.preservation_config.storage_dir else None,
                'preserve_custom_sections': self.preservation_config.preserve_custom_sections,
                'preserve_decorators': self.preservation_config.preserve_decorators,
                'preserve_examples': self.preservation_config.preserve_examples,
                'ttl_days': self.preservation_config.ttl_days
            },
            'validation': {
                'enforce_style': self.validation_config.enforce_style,
                'require_description': self.validation_config.require_description,
                'require_param_description': self.validation_config.require_param_description,
                'require_return_description': self.validation_config.require_return_description,
                'require_examples': self.validation_config.require_examples,
                'max_description_length': self.validation_config.max_description_length,
                'min_description_length': self.validation_config.min_description_length
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocStringConfig':
        """
        Create a configuration instance from a dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            DocStringConfig: A new configuration instance
        """
        format_config = FormatConfig(
            style=DocStringStyle(data['format']['style']),
            indentation=data['format']['indentation'],
            line_length=data['format']['line_length'],
            blank_lines_between_sections=data['format']['blank_lines_between_sections'],
            section_order=data['format']['section_order']
        )
        
        preservation_config = PreservationConfig(
            enabled=data['preservation']['enabled'],
            storage_dir=Path(data['preservation']['storage_dir']) if data['preservation']['storage_dir'] else None,
            preserve_custom_sections=data['preservation']['preserve_custom_sections'],
            preserve_decorators=data['preservation']['preserve_decorators'],
            preserve_examples=data['preservation']['preserve_examples'],
            ttl_days=data['preservation']['ttl_days']
        )
        
        validation_config = ValidationConfig(
            enforce_style=data['validation']['enforce_style'],
            require_description=data['validation']['require_description'],
            require_param_description=data['validation']['require_param_description'],
            require_return_description=data['validation']['require_return_description'],
            require_examples=data['validation']['require_examples'],
            max_description_length=data['validation']['max_description_length'],
            min_description_length=data['validation']['min_description_length']
        )
        
        return cls(
            format_config=format_config,
            preservation_config=preservation_config,
            validation_config=validation_config
        )
```


```python
"""
Configuration Module for Azure OpenAI Integration and DocString Management

This module manages configuration settings for Azure OpenAI services and the docstring system,
including environment-specific settings, model parameters, style preferences, and preservation settings.

Version: 1.0.0
Author: Development Team
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import json
import yaml
import logging
from enum import Enum

# Load environment variables
load_dotenv()

class DocStringStyle(Enum):
    """Enumeration of supported docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    CUSTOM = "custom"

@dataclass
class FormatConfig:
    """Configuration for docstring formatting."""
    style: DocStringStyle = DocStringStyle.GOOGLE
    indentation: int = 4
    line_length: int = 80
    blank_lines_between_sections: int = 1
    section_order: List[str] = field(default_factory=lambda: [
        "Description",
        "Args",
        "Returns",
        "Raises",
        "Yields",
        "Examples",
        "Notes"
    ])

@dataclass
class PreservationConfig:
    """Configuration for docstring content preservation."""
    enabled: bool = True
    storage_dir: Optional[Path] = None
    preserve_custom_sections: bool = True
    preserve_decorators: bool = True
    preserve_examples: bool = True
    ttl_days: int = 30

@dataclass
class ValidationConfig:
    """Configuration for docstring validation."""
    enforce_style: bool = True
    require_description: bool = True
    require_param_description: bool = True
    require_return_description: bool = True
    require_examples: bool = False
    max_description_length: int = 1000
    min_description_length: int = 10

class DocStringConfig:
    """
    Configuration manager for the docstring system.
    
    This class manages all configuration aspects of the docstring system,
    including loading from files and providing defaults.
    """
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        format_config: Optional[FormatConfig] = None,
        preservation_config: Optional[PreservationConfig] = None,
        validation_config: Optional[ValidationConfig] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to a configuration file
            format_config: Optional format configuration
            preservation_config: Optional preservation configuration
            validation_config: Optional validation configuration
        """
        self.format_config = format_config or FormatConfig()
        self.preservation_config = preservation_config or PreservationConfig()
        self.validation_config = validation_config or ValidationConfig()
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: Path):
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to the configuration file (JSON or YAML)
        
        Raises:
            ValueError: If the configuration file is invalid
        """
        try:
            config_data = self._load_file(config_file)
            
            # Update format config
            if 'format' in config_data:
                format_data = config_data['format']
                self.format_config = FormatConfig(
                    style=DocStringStyle(format_data.get('style', self.format_config.style.value)),
                    indentation=format_data.get('indentation', self.format_config.indentation),
                    line_length=format_data.get('line_length', self.format_config.line_length),
                    blank_lines_between_sections=format_data.get(
                        'blank_lines_between_sections',
                        self.format_config.blank_lines_between_sections
                    ),
                    section_order=format_data.get('section_order', self.format_config.section_order)
                )
            
            # Update preservation config
            if 'preservation' in config_data:
                pres_data = config_data['preservation']
                self.preservation_config = PreservationConfig(
                    enabled=pres_data.get('enabled', self.preservation_config.enabled),
                    storage_dir=Path(pres_data['storage_dir']) if 'storage_dir' in pres_data else self.preservation_config.storage_dir,
                    preserve_custom_sections=pres_data.get(
                        'preserve_custom_sections',
                        self.preservation_config.preserve_custom_sections
                    ),
                    preserve_decorators=pres_data.get(
                        'preserve_decorators',
                        self.preservation_config.preserve_decorators
                    ),
                    preserve_examples=pres_data.get(
                        'preserve_examples',
                        self.preservation_config.preserve_examples
                    ),
                    ttl_days=pres_data.get('ttl_days', self.preservation_config.ttl_days)
                )
            
            # Update validation config
            if 'validation' in config_data:
                val_data = config_data['validation']
                self.validation_config = ValidationConfig(
                    enforce_style=val_data.get('enforce_style', self.validation_config.enforce_style),
                    require_description=val_data.get(
                        'require_description',
                        self.validation_config.require_description
                    ),
                    require_param_description=val_data.get(
                        'require_param_description',
                        self.validation_config.require_param_description
                    ),
                    require_return_description=val_data.get(
                        'require_return_description',
                        self.validation_config.require_return_description
                    ),
                    require_examples=val_data.get(
                        'require_examples',
                        self.validation_config.require_examples
                    ),
                    max_description_length=val_data.get(
                        'max_description_length',
                        self.validation_config.max_description_length
                    ),
                    min_description_length=val_data.get(
                        'min_description_length',
                        self.validation_config.min_description_length
                    )
                )
        
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _load_file(self, config_file: Path) -> Dict[str, Any]:
        """Load and parse a configuration file."""
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file) as f:
            if config_file.suffix == '.json':
                return json.load(f)
            elif config_file.suffix in ('.yml', '.yaml'):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format")

    def save_config(self, config_file: Path):
        """
        Save current configuration to a file.
        
        Args:
            config_file: Path to save the configuration
        """
        config_data = {
            'format': {
                'style': self.format_config.style.value,
                'indentation': self.format_config.indentation,
                'line_length': self.format_config.line_length,
                'blank_lines_between_sections': self.format_config.blank_lines_between_sections,
                'section_order': self.format_config.section_order
            },
            'preservation': {
                'enabled': self.preservation_config.enabled,
                'storage_dir': str(self.preservation_config.storage_dir) if self.preservation_config.storage_dir else None,
                'preserve_custom_sections': self.preservation_config.preserve_custom_sections,
                'preserve_decorators': self.preservation_config.preserve_decorators,
                'preserve_examples': self.preservation_config.preserve_examples,
                'ttl_days': self.preservation_config.ttl_days
            },
            'validation': {
                'enforce_style': self.validation_config.enforce_style,
                'require_description': self.validation_config.require_description,
                'require_param_description': self.validation_config.require_param_description,
                'require_return_description': self.validation_config.require_return_description,
                'require_examples': self.validation_config.require_examples,
                'max_description_length': self.validation_config.max_description_length,
                'min_description_length': self.validation_config.min_description_length
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix == '.json':
                    json.dump(config_data, f, indent=2)
                elif config_file.suffix in ('.yml', '.yaml'):
                    yaml.safe_dump(config_data, f, default_flow_style=False)
                else:
                    raise ValueError("Unsupported configuration file format")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def get_style_guide(self) -> 'StyleGuide':
        """
        Create a StyleGuide instance from the current format configuration.
        
        Returns:
            StyleGuide: A configured style guide instance
        """
        return StyleGuide(
            indentation=self.format_config.indentation,
            section_order=self.format_config.section_order,
            format_rules={
                StyleRule.INDENTATION: self.format_config.indentation,
                StyleRule.SPACING: self.format_config.blank_lines_between_sections,
                StyleRule.FORMAT: self.format_config.style
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary
        """
        return {
            'format': {
                'style': self.format_config.style.value,
                'indentation': self.format_config.indentation,
                'line_length': self.format_config.line_length,
                'blank_lines_between_sections': self.format_config.blank_lines_between_sections,
                'section_order': self.format_config.section_order
            },
            'preservation': {
                'enabled': self.preservation_config.enabled,
                'storage_dir': str(self.preservation_config.storage_dir) if self.preservation_config.storage_dir else None,
                'preserve_custom_sections': self.preservation_config.preserve_custom_sections,
                'preserve_decorators': self.preservation_config.preserve_decorators,
                'preserve_examples': self.preservation_config.preserve_examples,
                'ttl_days': self.preservation_config.ttl_days
            },
            'validation': {
                'enforce_style': self.validation_config.enforce_style,
                'require_description': self.validation_config.require_description,
                'require_param_description': self.validation_config.require_param_description,
                'require_return_description': self.validation_config.require_return_description,
                'require_examples': self.validation_config.require_examples,
                'max_description_length': self.validation_config.max_description_length,
                'min_description_length': self.validation_config.min_description_length
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocStringConfig':
        """
        Create a configuration instance from a dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            DocStringConfig: A new configuration instance
        """
        format_config = FormatConfig(
            style=DocStringStyle(data['format']['style']),
            indentation=data['format']['indentation'],
            line_length=data['format']['line_length'],
            blank_lines_between_sections=data['format']['blank_lines_between_sections'],
            section_order=data['format']['section_order']
        )
        
        preservation_config = PreservationConfig(
            enabled=data['preservation']['enabled'],
            storage_dir=Path(data['preservation']['storage_dir']) if data['preservation']['storage_dir'] else None,
            preserve_custom_sections=data['preservation']['preserve_custom_sections'],
            preserve_decorators=data['preservation']['preserve_decorators'],
            preserve_examples=data['preservation']['preserve_examples'],
            ttl_days=data['preservation']['ttl_days']
        )
        
        validation_config = ValidationConfig(
            enforce_style=data['validation']['enforce_style'],
            require_description=data['validation']['require_description'],
            require_param_description=data['validation']['require_param_description'],
            require_return_description=data['validation']['require_return_description'],
            require_examples=data['validation']['require_examples'],
            max_description_length=data['validation']['max_description_length'],
            min_description_length=data['validation']['min_description_length']
        )
        
        return cls(
            format_config=format_config,
            preservation_config=preservation_config,
            validation_config=validation_config
        )

@dataclass
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI."""
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    model_name: str
    max_tokens: int
    temperature: float
    max_retries: int
    retry_delay: int
    request_timeout: int

    @classmethod
    def from_env(cls, environment: Optional[str] = None) -> 'AzureOpenAIConfig':
        """
        Create configuration from environment variables.
        
        Args:
            environment: Optional environment name (dev/prod)
            
        Returns:
            AzureOpenAIConfig: Configuration instance
        """
        endpoint_key = f"AZURE_OPENAI_ENDPOINT_{environment.upper()}" if environment else "AZURE_OPENAI_ENDPOINT"
        
        return cls(
            endpoint=os.getenv(endpoint_key, os.getenv("AZURE_OPENAI_ENDPOINT", "")),
            api_key=os.getenv("AZURE_OPENAI_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            model_name=os.getenv("MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("MAX_TOKENS", 4000)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_retries=int(os.getenv("MAX_RETRIES", 3)),
            retry_delay=int(os.getenv("RETRY_DELAY", 2)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 30))
        )

    def validate(self) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid
        """
        required_fields = [
            self.endpoint,
            self.api_key,
            self.api_version,
            self.deployment_name
        ]
        missing_fields = [field for field in required_fields if not field]
        if missing_fields:
            logging.error(f"Missing configuration fields: {missing_fields}")
        return not missing_fields

# Create default configuration instance
default_azure_config = AzureOpenAIConfig.from_env()
default_docstring_config = DocStringConfig()
```