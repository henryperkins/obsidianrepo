```python
# core/templates/schemas/google.py
"""
JSON schema definitions for Google-style docstrings.
"""

from typing import Dict, Any

GOOGLE_DOCSTRING_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "minLength": 1,
            "description": "Brief summary of the function or class"
        },
        "description": {
            "type": ["string", "null"],
            "description": "Detailed description of the function or class"
        },
        "args": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1
                    },
                    "type": {
                        "type": "string",
                        "minLength": 1
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1
                    },
                    "default": {
                        "type": ["string", "null"],
                        "description": "Default value if any"
                    },
                    "optional": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["name", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "minLength": 1
                },
                "description": {
                    "type": "string",
                    "minLength": 1
                }
            },
            "required": ["type"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "minLength": 1
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1
                    }
                },
                "required": ["type", "description"]
            }
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": ["string", "null"]
                    },
                    "code": {
                        "type": "string",
                        "minLength": 1
                    }
                },
                "required": ["code"]
            }
        },
        "notes": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1
            }
        }
    },
    "required": ["summary"]
}

# Provider-specific schema extensions
AZURE_EXTENSIONS = {
    "properties": {
        "complexity": {
            "type": "integer",
            "minimum": 0,
            "description": "Code complexity score"
        },
        "token_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Token count estimate"
        }
    }
}

ANTHROPIC_EXTENSIONS = {
    "properties": {
        "xml_tags": {
            "type": "boolean",
            "description": "Use XML tags in output"
        },
        "structured_output": {
            "type": "boolean",
            "description": "Return structured data"
        }
    }
}

def get_provider_schema(provider_type: str) -> Dict[str, Any]:
    """Get provider-specific schema."""
    base_schema = GOOGLE_DOCSTRING_SCHEMA.copy()
    
    extensions = {
        "azure": AZURE_EXTENSIONS,
        "anthropic": ANTHROPIC_EXTENSIONS
    }
    
    if provider_type in extensions:
        # Deep merge extensions
        for key, value in extensions[provider_type]["properties"].items():
            base_schema["properties"][key] = value
    
    return base_schema

def validate_provider_schema(
    provider_type: str,
    content: Dict[str, Any]
) -> bool:
    """Validate content against provider-specific schema."""
    from jsonschema import validate, ValidationError
    
    try:
        schema = get_provider_schema(provider_type)
        validate(instance=content, schema=schema)
        return True
    except ValidationError:
        return False
```

```python
# core/templates/docstring/google.py
"""
Google-style docstring template implementation.
"""

from typing import List, Dict, Any, Optional
import textwrap
import re
from ..base import DocstringTemplate, DocstringStyle, TemplateConfig
from ..schemas import validate_schema

class GoogleDocstringTemplate(DocstringTemplate):
    """Google-style docstring template."""
    
    def __init__(
        self,
        style: DocstringStyle = DocstringStyle.GOOGLE,
        config: Optional[TemplateConfig] = None
    ):
        """Initialize Google-style template."""
        super().__init__(style, config)
        
        # Section order
        self.section_order = [
            "args",
            "returns",
            "raises",
            "examples",
            "notes"
        ]
    
    def format(self, **kwargs) -> str:
        """Format complete docstring."""
        # Validate input
        if not self.validate(kwargs):
            raise ValueError("Invalid docstring content")
        
        sections = []
        
        # Add summary and description
        sections.append(kwargs.get("summary", ""))
        if description := kwargs.get("description"):
            sections.append("")
            sections.append(description)
        
        # Add remaining sections in order
        for section in self.section_order:
            if content := kwargs.get(section):
                sections.append("")
                if section == "args":
                    sections.append(self.format_args(content))
                elif section == "returns":
                    sections.append(self.format_returns(content))
                elif section == "raises":
                    sections.append(self.format_raises(content))
                elif section == "examples":
                    sections.append(self.format_examples(content))
                elif section == "notes":
                    sections.append(self.format_notes(content))
        
        # Format final docstring
        docstring = "\n".join(sections)
        return self._format_final(docstring)
    
    def format_args(self, args: List[Dict[str, Any]]) -> str:
        """Format arguments section."""
        if not args:
            return ""
            
        lines = ["Args:"]
        for arg in args:
            # Format argument line
            arg_line = f"{self.config.indent}{arg['name']}"
            if "type" in arg:
                arg_line += f" ({arg['type']})"
            arg_line += f": {arg['description']}"
            
            # Wrap description
            wrapped = textwrap.fill(
                arg_line,
                width=self.config.wrap_length,
                subsequent_indent=self.config.indent * 2,
                break_long_words=False,
                break_on_hyphens=False
            )
            lines.append(wrapped)
            
        return "\n".join(lines)
    
    def format_returns(self, returns: Dict[str, Any]) -> str:
        """Format returns section."""
        if not returns:
            return ""
            
        lines = ["Returns:"]
        
        # Format return line
        return_line = f"{self.config.indent}{returns['type']}"
        if "description" in returns:
            return_line += f": {returns['description']}"
            
        # Wrap description
        wrapped = textwrap.fill(
            return_line,
            width=self.config.wrap_length,
            subsequent_indent=self.config.indent * 2,
            break_long_words=False,
            break_on_hyphens=False
        )
        lines.append(wrapped)
        
        return "\n".join(lines)
    
    def format_raises(self, raises: List[Dict[str, Any]]) -> str:
        """Format raises section."""
        if not raises:
            return ""
            
        lines = ["Raises:"]
        for error in raises:
            # Format error line
            error_line = f"{self.config.indent}{error['type']}"
            if "description" in error:
                error_line += f": {error['description']}"
                
            # Wrap description
            wrapped = textwrap.fill(
                error_line,
                width=self.config.wrap_length,
                subsequent_indent=self.config.indent * 2,
                break_long_words=False,
                break_on_hyphens=False
            )
            lines.append(wrapped)
            
        return "\n".join(lines)
    
    def format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples section."""
        if not examples:
            return ""
            
        lines = ["Examples:"]
        for example in examples:
            if "description" in example:
                desc_line = f"{self.config.indent}{example['description']}"
                wrapped = textwrap.fill(
                    desc_line,
                    width=self.config.wrap_length,
                    subsequent_indent=self.config.indent * 2
                )
                lines.append(wrapped)
            
            if "code" in example:
                lines.append(f"{self.config.indent}```python")
                # Indent code lines
                code_lines = example["code"].strip().split("\n")
                lines.extend(f"{self.config.indent}{line}" for line in code_lines)
                lines.append(f"{self.config.indent}```")
                
            lines.append("")
            
        return "\n".join(lines[:-1])  # Remove last empty line
    
    def format_notes(self, notes: List[str]) -> str:
        """Format notes section."""
        if not notes:
            return ""
            
        lines = ["Notes:"]
        for note in notes:
            note_line = f"{self.config.indent}{note}"
            wrapped = textwrap.fill(
                note_line,
                width=self.config.wrap_length,
                subsequent_indent=self.config.indent * 2,
                break_long_words=False,
                break_on_hyphens=False
            )
            lines.append(wrapped)
            
        return "\n".join(lines)
    
    def validate(self, content: Dict[str, Any]) -> bool:
        """Validate docstring content against schema."""
        return validate_schema("google", content)
    
    def _format_final(self, docstring: str) -> str:
        """Format final docstring with proper quotes and spacing."""
        # Add quote style
        quoted = f'"""\n{docstring}\n"""'
        
        # Normalize line endings
        normalized = re.sub(r'\r\n?', '\n', quoted)
        
        # Ensure consistent spacing
        spaced = re.sub(
            r'\n{3,}',
            '\n' * (self.config.section_spacing + 1),
            normalized
        )
        
        return spaced

# Register template
register_template("google", GoogleDocstringTemplate)
```

```python
# core/templates/base.py
"""
Base template system defining core interfaces and functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, ClassVar
from dataclasses import dataclass
from enum import Enum

@dataclass
class TemplateConfig:
    """Template configuration settings."""
    indent: str = "    "
    wrap_length: int = 80
    section_spacing: int = 1
    allow_markdown: bool = True
    extra_config: Optional[Dict[str, Any]] = None

class DocstringStyle(Enum):
    """Supported docstring styles."""
    GOOGLE = 'google'
    NUMPY = 'numpy'
    SPHINX = 'sphinx'
    EPYDOC = 'epydoc'
    
    def configure(self, **kwargs) -> 'DocstringStyle':
        """Configure style with custom settings."""
        style = self.__class__(self.value)
        style._config = TemplateConfig(**kwargs)
        return style

class ProviderType(Enum):
    """Supported LLM providers."""
    AZURE = 'azure'
    ANTHROPIC = 'anthropic'
    OPENAI = 'openai'
    GEMINI = 'gemini'
    COHERE = 'cohere'

class TemplateBase(ABC):
    """Base class for all templates."""
    
    _registry: ClassVar[Dict[str, Type['TemplateBase']]] = {}
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize template with configuration."""
        self.config = config or TemplateConfig()
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        pass
    
    @abstractmethod
    def validate(self, content: Any) -> bool:
        """Validate content against template schema."""
        pass
    
    @classmethod
    def register(cls, name: str, template_class: Type['TemplateBase']) -> None:
        """Register a new template class."""
        if not issubclass(template_class, TemplateBase):
            raise ValueError("Template must inherit from TemplateBase")
        cls._registry[name] = template_class
    
    @classmethod
    def get_template(cls, name: str) -> Type['TemplateBase']:
        """Get registered template class."""
        if name not in cls._registry:
            raise ValueError(f"Unknown template: {name}")
        return cls._registry[name]

class DocstringTemplate(TemplateBase):
    """Base class for docstring templates."""
    
    def __init__(
        self,
        style: DocstringStyle,
        config: Optional[TemplateConfig] = None
    ):
        """Initialize docstring template."""
        super().__init__(config)
        self.style = style
    
    @abstractmethod
    def format_args(self, args: List[Dict[str, Any]]) -> str:
        """Format arguments section."""
        pass
    
    @abstractmethod
    def format_returns(self, returns: Dict[str, Any]) -> str:
        """Format returns section."""
        pass
    
    @abstractmethod
    def format_raises(self, raises: List[Dict[str, Any]]) -> str:
        """Format raises section."""
        pass
    
    @abstractmethod
    def format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples section."""
        pass

class PromptTemplate:
    """Template for generating prompts."""
    
    def __init__(
        self,
        template: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        provider_configs: Dict[ProviderType, Dict[str, Any]]
    ):
        """Initialize prompt template."""
        self.template = template
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.provider_configs = provider_configs
    
    def format_prompt(self, **kwargs) -> str:
        """Format prompt template with provided values."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt parameter: {e}")

def register_template(name: str, template_class: Type[TemplateBase]) -> None:
    """Register a new template."""
    TemplateBase.register(name, template_class)

def get_template(name: str) -> Type[TemplateBase]:
    """Get registered template."""
    return TemplateBase.get_template(name)

def create_template(
    name: str,
    style: Optional[DocstringStyle] = None,
    config: Optional[TemplateConfig] = None
) -> TemplateBase:
    """Create template instance."""
    template_class = get_template(name)
    if issubclass(template_class, DocstringTemplate):
        if not style:
            raise ValueError("Docstring template requires style")
        return template_class(style=style, config=config)
    return template_class(config=config)
```

---

# Template System Documentation

## Directory Structure
```
core/
├── templates/
│   ├── __init__.py
│   ├── base.py              # Base template definitions
│   ├── docstring/
│   │   ├── __init__.py
│   │   ├── google.py       # Google style templates
│   │   ├── numpy.py        # NumPy style templates
│   │   ├── sphinx.py       # Sphinx style templates
│   │   └── schemas.py      # JSON schemas for validation
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── azure.py        # Azure-specific templates
│   │   ├── anthropic.py    # Anthropic-specific templates
│   │   ├── openai.py       # OpenAI-specific templates
│   │   ├── gemini.py       # Gemini-specific templates
│   │   └── cohere.py       # Cohere-specific templates
│   └── schemas/
│       ├── __init__.py
│       ├── input.py        # Input validation schemas
│       └── output.py       # Output validation schemas
```

## Core Components

### 1. Base Templates (base.py)
Defines the fundamental template interfaces and shared functionality:
- Template registration
- Schema validation
- Variable substitution
- Template inheritance

### 2. Docstring Templates (docstring/)
Style-specific docstring templates with format specifications:
- Google style (google.py)
- NumPy style (numpy.py)
- Sphinx style (sphinx.py)
- Style-specific schemas (schemas.py)

### 3. Provider Templates (providers/)
Provider-specific optimizations and configurations:
- Azure OpenAI templates
- Anthropic templates
- Google Gemini templates
- OpenAI templates
- Cohere templates

### 4. Schema Definitions (schemas/)
JSON schemas for input/output validation:
- Input schemas (input.py)
- Output schemas (output.py)
- Schema utilities and helpers

## Usage Examples

### 1. Basic Template Usage
```python
from core.templates import DocstringTemplate, DocstringStyle

# Create template
template = DocstringTemplate(style=DocstringStyle.GOOGLE)

# Format docstring
docstring = template.format(
    name="example_function",
    args=[{"name": "x", "type": "int", "description": "Input value"}],
    returns={"type": "str", "description": "Processed result"}
)
```

### 2. Provider-Specific Template
```python
from core.templates.providers import AzureTemplate

# Create Azure-optimized template
template = AzureTemplate(
    style=DocstringStyle.GOOGLE,
    max_tokens=4000,
    temperature=0.7
)

# Generate with Azure-specific features
result = await template.generate(
    code_metadata=metadata,
    use_functions=True
)
```

### 3. Custom Template Registration
```python
from core.templates import register_template, TemplateBase

class CustomTemplate(TemplateBase):
    """Custom documentation template."""
    
    def format(self, **kwargs):
        # Custom formatting logic
        pass
    
    def validate(self, content):
        # Custom validation logic
        pass

# Register template
register_template("custom", CustomTemplate)
```

## Key Features

### 1. Template Inheritance
Templates support inheritance for specialization:
```python
class ExtendedTemplate(GoogleDocstringTemplate):
    """Extended Google-style template."""
    
    def format(self, **kwargs):
        # Add custom formatting
        base_result = super().format(**kwargs)
        return self._add_extensions(base_result)
```

### 2. Schema Validation
Built-in schema validation for inputs/outputs:
```python
from core.templates.schemas import validate_input

# Validate template input
errors = validate_input(
    template_name="google",
    data={
        "name": "example_function",
        "args": [...],
        "returns": {...}
    }
)
```

### 3. Provider Optimization
Provider-specific template optimizations:
```python
# Azure-optimized template
azure_template = AzureTemplate(
    style=DocstringStyle.GOOGLE,
    optimize_tokens=True,
    use_functions=True
)

# Anthropic-optimized template
anthropic_template = AnthropicTemplate(
    style=DocstringStyle.NUMPY,
    use_xml=True,
    structured_output=True
)
```

### 4. Style Configuration
Configure docstring styles:
```python
# Configure Google style
google_style = DocstringStyle.GOOGLE.configure(
    indent="    ",
    section_order=[
        "Args",
        "Returns",
        "Raises",
        "Examples"
    ],
    wrap_length=80
)

# Use configured style
template = DocstringTemplate(style=google_style)
```

### 5. Schema Customization
Customize validation schemas:
```python
from core.templates.schemas import extend_schema

# Extend base schema
custom_schema = extend_schema(
    "google",
    additional_properties={
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    }
)
```

## Template Guidelines

### 1. Style Consistency
- Follow language-specific docstring conventions
- Maintain consistent formatting within each style
- Use proper indentation and spacing

### 2. Content Structure
- Clear section separation
- Logical parameter ordering
- Consistent type annotations
- Proper return value documentation

### 3. Provider Optimizations
- Token-efficient formatting
- Provider-specific features usage
- Proper error handling
- Rate limit consideration

### 4. Schema Validation
- Input validation before processing
- Output validation after generation
- Clear error messages
- Proper type checking

### 5. Extension Points
- Template inheritance
- Hook methods
- Custom formatters
- Schema extensions


```python
# main.py
"""
Enhanced main entry point for the documentation generator with multi-LLM support.
Supports multiple LLM providers with intelligent fallback and load balancing.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import json

from core.config import Config
from core.storage import Storage
from core.logger import setup_script_logger
from core.providers.registry import ProviderRegistry, ProviderPool
from core.templates import ProviderType, DocstringStyle
from core.integrated_generator import IntegratedDocGenerator

logger = setup_script_logger('docgen')

def setup_providers(config: Config) -> ProviderPool:
    """Set up LLM providers from configuration."""
    providers = {}
    
    # Try setting up each provider
    for provider_type in ProviderType:
        try:
            provider = ProviderRegistry.from_environment(provider_type)
            if provider:
                providers[provider_type] = provider
                logger.info(f"Successfully initialized {provider_type} provider")
        except Exception as e:
            logger.warning(f"Failed to initialize {provider_type} provider: {e}")
    
    if not providers:
        raise ValueError("No LLM providers could be initialized")
    
    return ProviderPool(
        providers=providers,
        preferred_provider=config.preferred_provider
    )

async def process_file(
    file_path: Path,
    generator: IntegratedDocGenerator,
    output_dir: Path,
    style: DocstringStyle = DocstringStyle.GOOGLE
) -> bool:
    """
    Process a single Python file.
    
    Args:
        file_path: Path to Python file
        generator: Document generator instance
        output_dir: Output directory
        style: Docstring style to use
        
    Returns:
        bool: True if processing was successful
    """
    try:
        logger.info(f"Processing {file_path}")
        
        # Read source code
        source_code = file_path.read_text(encoding='utf-8')
        
        # Generate documentation
        updated_code, markdown_docs = await generator.process_code(
            source_code=source_code,
            file_path=file_path,
            style=style
        )
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save updated code
        code_path = output_dir / file_path.name
        code_path.write_text(updated_code, encoding='utf-8')
        
        # Save documentation
        docs_path = output_dir / f"{file_path.stem}_docs.md"
        docs_path.write_text(markdown_docs, encoding='utf-8')
        
        # Save metrics
        metrics = await generator.get_metrics()
        metrics_path = output_dir / f"{file_path.stem}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        
        logger.info(f"Successfully processed {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False

async def process_directory(
    directory: Path,
    generator: IntegratedDocGenerator,
    output_dir: Path,
    style: DocstringStyle = DocstringStyle.GOOGLE,
    concurrent_files: int = 5
) -> int:
    """
    Process all Python files in a directory.
    
    Args:
        directory: Directory containing Python files
        generator: Document generator instance
        output_dir: Output directory
        style: Docstring style to use
        concurrent_files: Number of files to process concurrently
        
    Returns:
        int: Number of successfully processed files
    """
    try:
        # Find all Python files
        py_files = list(directory.rglob("*.py"))
        if not py_files:
            logger.error(f"No Python files found in {directory}")
            return 0
        
        # Process files in batches
        total_success = 0
        for i in range(0, len(py_files), concurrent_files):
            batch = py_files[i:i + concurrent_files]
            tasks = [
                process_file(
                    file_path=file_path,
                    generator=generator,
                    output_dir=output_dir,
                    style=style
                )
                for file_path in batch
            ]
            
            # Process batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if isinstance(r, bool) and r)
            total_success += success_count
            
            logger.info(
                f"Batch {i//concurrent_files + 1}: "
                f"Processed {success_count} out of {len(batch)} files"
            )
        
        # Save overall metrics
        metrics = await generator.get_metrics()
        metrics_path = output_dir / "processing_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding='utf-8'
        )
        
        logger.info(
            f"Total: Successfully processed {total_success} "
            f"out of {len(py_files)} files"
        )
        return total_success
        
    except Exception as e:
        logger.error(f"Directory processing failed: {e}")
        return 0

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced AI-powered Python documentation generator"
    )
    
    # Source and output paths
    parser.add_argument(
        "source",
        type=Path,
        help="Source file or directory to process"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for generated documentation"
    )
    
    # LLM configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=[p.value for p in ProviderType],
        help="Preferred LLM provider"
    )
    
    # Documentation style
    parser.add_argument(
        "--style",
        type=str,
        choices=[s.value for s in DocstringStyle],
        default=DocstringStyle.GOOGLE.value,
        help="Docstring style to use"
    )
    
    # Processing options
    parser.add_argument(
        "--concurrent-files",
        type=int,
        default=5,
        help="Number of files to process concurrently"
    )
    parser.add_argument(
        "--cache-url",
        type=str,
        help="Redis URL for caching (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = Config.from_env()
        if args.provider:
            config.preferred_provider = ProviderType(args.provider)
        
        # Initialize storage
        storage = None
        if args.cache_url:
            storage = Storage(args.cache_url)
            await storage.connect()
        
        # Set up LLM providers
        provider_pool = setup_providers(config)
        
        # Initialize generator
        generator = IntegratedDocGenerator(
            config=config,
            provider_pool=provider_pool,
            storage=storage,
            docstring_style=DocstringStyle(args.style)
        )
        
        # Process source
        if args.source.is_file():
            success = await process_file(
                file_path=args.source,
                generator=generator,
                output_dir=args.output,
                style=DocstringStyle(args.style)
            )
            exit_code = 0 if success else 1
        elif args.source.is_dir():
            success_count = await process_directory(
                directory=args.source,
                generator=generator,
                output_dir=args.output,
                style=DocstringStyle(args.style),
                concurrent_files=args.concurrent_files
            )
            exit_code = 0 if success_count > 0 else 1
        else:
            logger.error(f"Invalid source path: {args.source}")
            exit_code = 1
        
        # Cleanup
        if storage:
            await storage.close()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
```

```python
# core/providers/base.py
"""
Base class for LLM providers defining the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
from dataclasses import dataclass

from ..templates import PromptTemplate
from ..schema import SchemaValidator
from ..logger import get_logger

logger = get_logger(__name__)

@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: datetime = datetime.now()

@dataclass
class GenerationMetrics:
    """Generation metrics tracking."""
    start_time: datetime
    end_time: datetime
    success: bool
    tokens: Optional[TokenUsage] = None
    error: Optional[str] = None
    provider_metrics: Optional[Dict[str, Any]] = None

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: 'ProviderConfig'):
        """Initialize provider with configuration."""
        self.config = config
        self.validator = SchemaValidator()
        self.metrics: List[GenerationMetrics] = []
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate completion from prompt."""
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    async def validate_response(
        self,
        response: Any,
        expected_schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Validate provider response against schema."""
        pass
    
    async def safe_generate(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate with retry logic and error handling."""
        start_time = datetime.now()
        
        for attempt in range(max_retries):
            try:
                # Check token count
                token_count = await self.count_tokens(prompt)
                if token_count > self.config.max_tokens:
                    raise ValueError(f"Token count {token_count} exceeds limit {self.config.max_tokens}")
                
                # Generate response
                response = await self.generate(prompt, **kwargs)
                
                # Track metrics
                end_time = datetime.now()
                self.metrics.append(
                    GenerationMetrics(
                        start_time=start_time,
                        end_time=end_time,
                        success=True,
                        tokens=TokenUsage(
                            prompt_tokens=token_count,
                            completion_tokens=await self.count_tokens(str(response)),
                            total_tokens=token_count + await self.count_tokens(str(response))
                        )
                    )
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Track failure metrics
                    end_time = datetime.now()
                    self.metrics.append(
                        GenerationMetrics(
                            start_time=start_time,
                            end_time=end_time,
                            success=False,
                            error=str(e)
                        )
                    )
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of generation metrics."""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]
        
        total_tokens = sum(
            m.tokens.total_tokens for m in successful if m.tokens
        )
        
        avg_time = sum(
            (m.end_time - m.start_time).total_seconds()
            for m in self.metrics
        ) / len(self.metrics)
        
        return {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(self.metrics) * 100,
            "average_time": avg_time,
            "total_tokens": total_tokens,
            "common_errors": self._get_common_errors()
        }
    
    def _get_common_errors(self) -> Dict[str, int]:
        """Get counts of common errors."""
        error_counts: Dict[str, int] = {}
        
        for metric in self.metrics:
            if not metric.success and metric.error:
                error_counts[metric.error] = error_counts.get(metric.error, 0) + 1
        
        # Sort by frequency and return top 5
        return dict(
            sorted(
                error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )
```

```python
# core/providers/registry.py
"""
Registry system for LLM providers with unified interfaces and provider-specific optimizations.
Supports Azure OpenAI, Anthropic, Gemini, OpenAI, and Cohere with consistent error handling.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
import asyncio
from functools import partial

from .base import BaseLLMProvider
from .azure import AzureProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .cohere import CohereProvider

from ..templates import ProviderType, PromptTemplate
from ..logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProviderConfig:
    """Provider-specific configuration."""
    api_key: str
    model_name: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    extra_config: Dict[str, Any] = None

class ProviderRegistry:
    """Registry for managing multiple LLM providers."""
    
    _providers: Dict[ProviderType, Type[BaseLLMProvider]] = {
        ProviderType.AZURE: AzureProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.COHERE: CohereProvider
    }
    
    @classmethod
    def register_provider(
        cls,
        provider_type: ProviderType,
        provider_class: Type[BaseLLMProvider]
    ) -> None:
        """Register a new provider."""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider must inherit from BaseLLMProvider")
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def get_provider(
        cls,
        provider_type: ProviderType,
        config: ProviderConfig
    ) -> BaseLLMProvider:
        """Get provider instance with configuration."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def from_environment(
        cls,
        provider_type: ProviderType
    ) -> Optional[BaseLLMProvider]:
        """Create provider from environment variables."""
        try:
            config = {
                ProviderType.AZURE: partial(
                    ProviderConfig,
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_VERSION"),
                    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
                ),
                ProviderType.ANTHROPIC: partial(
                    ProviderConfig,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model_name=os.getenv("ANTHROPIC_MODEL", "claude-2")
                ),
                ProviderType.GEMINI: partial(
                    ProviderConfig,
                    api_key=os.getenv("GEMINI_API_KEY"),
                    model_name=os.getenv("GEMINI_MODEL", "gemini-pro")
                ),
                ProviderType.OPENAI: partial(
                    ProviderConfig,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4")
                ),
                ProviderType.COHERE: partial(
                    ProviderConfig,
                    api_key=os.getenv("COHERE_API_KEY"),
                    model_name=os.getenv("COHERE_MODEL", "command")
                )
            }
            
            if provider_type not in config:
                raise ValueError(f"Unsupported provider type: {provider_type}")
                
            provider_config = config[provider_type]()
            if not provider_config.api_key:
                raise ValueError(f"Missing API key for {provider_type}")
                
            return cls.get_provider(provider_type, provider_config)
            
        except Exception as e:
            logger.error(f"Error creating provider from environment: {e}")
            return None
    
class ProviderPool:
    """Pool of LLM providers with fallback and load balancing."""
    
    def __init__(
        self,
        providers: Dict[ProviderType, BaseLLMProvider],
        preferred_provider: Optional[ProviderType] = None
    ):
        """Initialize provider pool."""
        self.providers = providers
        self.preferred_provider = preferred_provider
        self.fallback_order = self._determine_fallback_order()
    
    def _determine_fallback_order(self) -> List[ProviderType]:
        """Determine provider fallback order based on capabilities."""
        # Prioritize preferred provider
        order = []
        if self.preferred_provider:
            order.append(self.preferred_provider)
        
        # Add remaining providers in priority order
        priority = [
            ProviderType.AZURE,  # Most capable
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.GEMINI,
            ProviderType.COHERE
        ]
        
        for provider_type in priority:
            if provider_type in self.providers and provider_type not in order:
                order.append(provider_type)
        
        return order
    
    async def execute(
        self,
        template: PromptTemplate,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Execute prompt with fallback."""
        for provider_type in self.fallback_order:
            provider = self.providers.get(provider_type)
            if not provider:
                continue
            
            try:
                return await provider.generate(
                    template.format_prompt(**kwargs),
                    **template.provider_configs.get(provider_type, {})
                )
            except Exception as e:
                logger.error(f"Provider {provider_type} failed: {e}")
                continue
        
        raise RuntimeError("All providers failed")
```

```python
# core/integrated_generator.py
"""
Integrated documentation generator that combines the simplified core system
with multi-LLM capabilities and template-driven generation.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio

from .config import Config
from .analyzer import CodeAnalyzer, CodeMetadata
from .storage import Storage
from .templates import (
    ProviderType, DocstringStyle, DOCSTRING_PROMPTS
)
from .orchestrator import LLMOrchestrator
from .schema import SchemaValidator
from .docgen import DocstringGenerator
from .logger import get_logger

logger = get_logger(__name__)

class IntegratedDocGenerator:
    """
    Unified documentation generator that combines simplified core
    with multi-LLM capabilities.
    """
    
    def __init__(
        self,
        config: Config,
        storage: Optional[Storage] = None,
        docstring_style: DocstringStyle = DocstringStyle.GOOGLE,
        provider_type: Optional[ProviderType] = None
    ):
        """Initialize the integrated generator."""
        self.config = config
        self.storage = storage
        
        # Initialize components
        self.analyzer = CodeAnalyzer(None)  # Will be set per file
        self.validator = SchemaValidator()
        self.docstring_generator = DocstringGenerator(style=docstring_style.value)
        
        # Set up LLM orchestrator
        provider_configs = {
            provider_type or ProviderType.AZURE: config.to_provider_config()
        }
        self.orchestrator = LLMOrchestrator(
            providers=provider_configs,
            storage=storage,
            default_style=docstring_style
        )
    
    async def process_code(
        self,
        source_code: str,
        file_path: Optional[Path] = None
    ) -> Tuple[str, str]:
        """
        Process source code to generate documentation.
        
        Args:
            source_code: Source code to process
            file_path: Optional path for the source file
            
        Returns:
            Tuple of (updated source code, markdown documentation)
        """
        try:
            # Analyze code
            self.analyzer.source_code = source_code
            metadata = self.analyzer.analyze()
            
            # Validate metadata
            if errors := self.validator.validate_metadata(metadata):
                raise ValueError(f"Invalid code metadata: {errors}")
            
            # Process each item with LLM
            updated_items = []
            for category in ['classes', 'functions']:
                for item in metadata[category]:
                    # Generate docstring
                    result = await self.orchestrator.generate_docstring(
                        code_metadata=item.to_dict()
                    )
                    
                    # Format docstring
                    formatted_docstring = self.docstring_generator.generate(
                        content=result.content,
                        metadata=result.metadata
                    )
                    
                    updated_items.append({
                        'name': item.name,
                        'type': category[:-1],  # Remove 's' from category
                        'docstring': formatted_docstring,
                        'lineno': item.lineno
                    })
            
            # Update source code with new docstrings
            updated_code = self._update_source(source_code, updated_items)
            
            # Generate markdown documentation
            markdown_docs = self._generate_markdown(
                updated_items,
                file_path.stem if file_path else None
            )
            
            return updated_code, markdown_docs
            
        except Exception as e:
            logger.error(f"Error processing code: {e}")
            raise
    
    def _update_source(
        self,
        source_code: str,
        updated_items: List[Dict[str, Any]]
    ) -> str:
        """Update source code with new docstrings."""
        self.analyzer.source_code = source_code
        return self.analyzer.update_docstrings(updated_items)
    
    def _generate_markdown(
        self,
        items: List[Dict[str, Any]],
        module_name: Optional[str] = None
    ) -> str:
        """Generate markdown documentation."""
        sections = []
        
        # Add header
        if module_name:
            sections.append(f"# {module_name}")
            sections.append("")
        
        # Add classes
        classes = [item for item in items if item['type'] == 'class']
        if classes:
            sections.append("## Classes")
            sections.append("")
            for cls in classes:
                sections.append(f"### {cls['name']}")
                sections.append(cls['docstring'])
                sections.append("")
        
        # Add functions
        functions = [item for item in items if item['type'] == 'function']
        if functions:
            sections.append("## Functions")
            sections.append("")
            for func in functions:
                sections.append(f"### {func['name']}")
                sections.append(func['docstring'])
                sections.append("")
        
        return "\n".join(sections)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return await self.orchestrator.get_metrics()
    
    async def clear_cache(self) -> None:
        """Clear all cached results."""
        if self.storage:
            await self.storage.clear()

# Update main script
async def process_files(
    source_paths: List[Path],
    output_dir: Path,
    config: Config,
    storage: Optional[Storage] = None,
    docstring_style: DocstringStyle = DocstringStyle.GOOGLE,
    provider_type: Optional[ProviderType] = None
) -> int:
    """
    Process multiple Python files.
    
    Args:
        source_paths: List of files to process
        output_dir: Output directory
        config: Configuration
        storage: Optional storage system
        docstring_style: Docstring style to use
        provider_type: Specific provider to use
        
    Returns:
        Number of successfully processed files
    """
    try:
        # Initialize generator
        generator = IntegratedDocGenerator(
            config=config,
            storage=storage,
            docstring_style=docstring_style,
            provider_type=provider_type
        )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        success_count = 0
        for path in source_paths:
            try:
                logger.info(f"Processing {path}")
                
                # Read source
                source_code = path.read_text(encoding='utf-8')
                
                # Generate documentation
                updated_code, markdown_docs = await generator.process_code(
                    source_code,
                    path
                )
                
                # Write outputs
                code_path = output_dir / path.name
                docs_path = output_dir / f"{path.stem}_docs.md"
                
                code_path.write_text(updated_code, encoding='utf-8')
                docs_path.write_text(markdown_docs, encoding='utf-8')
                
                success_count += 1
                logger.info(f"Successfully processed {path}")
                
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
        
        # Log metrics
        metrics = await generator.get_metrics()
        logger.info(f"Processing completed. Metrics: {metrics}")
        
        return success_count
        
    except Exception as e:
        logger.error(f"Failed to process files: {e}")
        return 0
```