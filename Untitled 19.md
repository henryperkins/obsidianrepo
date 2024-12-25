```python
"""
Documentation Generator main module.
Handles initialization, processing, and cleanup of documentation generation.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from urllib.parse import urlparse
import tempfile
from core.logger import LoggerSetup
from core.cache import Cache
from core.metrics import Metrics, MetricsCollector
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.types import DocumentationContext
from core.docs import DocStringManager
from repository_handler import RepositoryHandler
from exceptions import ConfigurationError, DocumentationError

load_dotenv()
logger = LoggerSetup.get_logger(__name__)

class DocumentationGenerator:
    """
    Documentation Generator Class.
    Handles initialization, processing, and cleanup of components for documentation generation.
    """

    def __init__(self) -> None:
        """Initialize the documentation generator with empty component references."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config: Optional[AzureOpenAIConfig] = None
        self.cache: Optional[Cache] = None
        self.metrics: Optional[Metrics] = None
        self.token_manager: Optional[TokenManager] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.repo_handler: Optional[RepositoryHandler] = None
        self.docstring_processor: Optional[DocstringProcessor] = None
        self._initialized = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """
        Initialize all components in correct order with proper dependency injection.

        Args:
            base_path: Optional base path for repository operations

        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            # 1. First load configuration
            self.config = AzureOpenAIConfig.from_env()
            if not self.config.validate():
                raise ConfigurationError("Invalid configuration")

            # 2. Initialize metrics
            self.metrics = Metrics()
            self.metrics_collector = MetricsCollector()

            # 3. Initialize cache if enabled
            if self.config.cache_enabled:
                self.cache = Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=True
                )
                await self.cache._initialize_connection()

            # 4. Initialize token manager
            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_id,
                config=self.config,
                metrics_collector=self.metrics_collector
            )

            # 5. Initialize docstring processor
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)

            # 6. Initialize AI handler
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=self.cache,
                token_manager=self.token_manager,
                code_extractor=None,
                response_parser=None,
                metrics=self.metrics
            )

            # 7. Initialize repository handler if path provided
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()

            self._initialized = True
            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up all resources safely."""
        cleanup_errors = []

        # List of (component, name, cleanup_method)
        components = [
            (self.repo_handler, "Repository Handler", 
             lambda x: x.__aexit__(None, None, None) if hasattr(x, '__aexit__') else None),
            (self.ai_handler, "AI Handler", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            (self.token_manager, "Token Manager", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            (self.cache, "Cache", 
             lambda x: x.close() if hasattr(x, 'close') else None),
            (self.metrics_collector, "Metrics Collector", 
             lambda x: x.close() if hasattr(x, 'close') else None)
        ]

        for component, name, cleanup_func in components:
            if component is not None:
                try:
                    if cleanup_func:
                        result = cleanup_func(component)
                        if asyncio.iscoroutine(result):
                            await result
                    logger.info(f"{name} cleaned up successfully")
                except Exception as e:
                    error_msg = f"Error cleaning up {name}: {str(e)}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
                    continue  # Continue with other cleanups even if one fails

        self._initialized = False

        if cleanup_errors:
            logger.error("Some components failed to cleanup properly")
            for error in cleanup_errors:
                logger.error(f"- {error}")

    async def process_file(self, file_path: Path, output_base: Path) -> Optional[Tuple[str, str]]:
        """
        Process a single file and generate documentation.

        Args:
            file_path: Path to the file to process
            output_base: Base path for output files

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails
        """
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()

        try:
            # Create normalized paths
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not file_path.suffix == '.py':
                raise ValueError(f"Not a Python file: {file_path}")

            # Create output directory structure
            relative_path = file_path.relative_to(output_base) if output_base else file_path
            output_dir = Path("docs") / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Read and validate source code
            try:
                source_code = file_path.read_text(encoding='utf-8')
                if not source_code.strip():
                    raise ValueError("Empty source code")
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                source_code = file_path.read_text(encoding='latin-1')
                self.logger.warning(f"Used latin-1 fallback encoding for {file_path}")

            # Process with AI handler
            cache_key = f"doc:{file_path.stem}:{hash(source_code.encode())}"
            result = await self.ai_handler.process_code(
                source_code=source_code,
                cache_key=cache_key
            )

            if not result:
                raise DocumentationError("AI processing failed")

            updated_code, ai_docs = result

            # Create documentation context
            context = DocumentationContext(
                source_code=updated_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "ai_generated": ai_docs,
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat()
                }
            )

            # Generate documentation
            doc_manager = DocStringManager(
                context=context,
                docstring_processor=self.docstring_processor,
                markdown_generator=None  # Will use default
            )

            documentation = await doc_manager.generate_documentation()

            # Save documentation
            output_path = output_dir / f"{file_path.stem}.md"
            output_path.write_text(documentation, encoding='utf-8')

            # Track metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            if self.metrics_collector:
                await self.metrics_collector.track_operation(
                    operation_type="documentation_generation",
                    success=True,
                    duration=processing_time,
                    metadata={
                        "file_path": str(file_path),
                        "output_path": str(output_path),
                        "processing_time": processing_time
                    }
                )

            return updated_code, documentation

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            if self.metrics_collector:
                await self.metrics_collector.track_operation(
                    operation_type="documentation_generation",
                    success=False,
                    duration=(datetime.now() - start_time).total_seconds(),
                    error=str(e),
                    metadata={
                        "file_path": str(file_path)
                    }
                )
            return None

    async def process_repository(self, repo_path_or_url: str) -> int:
        """Process an entire repository."""
        try:
            if not self._initialized:
                raise RuntimeError("DocumentationGenerator not initialized")

            start_time = datetime.now()
            processed_files = 0
            failed_files = []
            is_url = urlparse(repo_path_or_url).scheme != ''

            try:
                # Handle repository setup
                if is_url:
                    logger.info(f"Cloning repository from URL: {repo_path_or_url}")
                    if not self.repo_handler:
                        self.repo_handler = RepositoryHandler(repo_path=Path(tempfile.mkdtemp()))
                    repo_path = await self.repo_handler.clone_repository(repo_path_or_url)
                else:
                    repo_path = Path(repo_path_or_url).resolve()
                    if not repo_path.exists():
                        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                    if not self.repo_handler:
                        self.repo_handler = RepositoryHandler(repo_path=repo_path)

                logger.info(f"Starting repository processing: {repo_path}")
                python_files = self.repo_handler.get_python_files()
                
                if not python_files:
                    logger.warning("No Python files found")
                    return 0

                # Process files
                with tqdm(python_files, desc="Processing files") as progress:
                    for file_path in progress:
                        try:
                            result = await self.process_file(file_path, repo_path)
                            if result:
                                processed_files += 1
                            else:
                                failed_files.append((file_path, "Processing failed"))
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            failed_files.append((file_path, str(e)))
                        
                        progress.set_postfix(
                            processed=processed_files,
                            failed=len(failed_files)
                        )

                # Log summary
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info("\nProcessing Summary:")
                logger.info(f"Total files: {len(python_files)}")
                logger.info(f"Successfully processed: {processed_files}")
                logger.info(f"Failed: {len(failed_files)}")
                logger.info(f"Total processing time: {processing_time:.2f} seconds")

                if failed_files:
                    logger.error("\nFailed files:")
                    for file_path, error in failed_files:
                        logger.error(f"- {file_path}: {error}")

                return 0 if processed_files > 0 else 1

            finally:
                if is_url and self.repo_handler:
                    await self.repo_handler.cleanup()
                    logger.info("Repository cleanup completed")

        except Exception as e:
            logger.error(f"Repository processing failed: {e}")
            return 1

    async def main(self, args: argparse.Namespace) -> int:
        """Main application entry point."""
        try:
            await self.initialize()
            
            if args.repository:
                return await self.process_repository(args.repository)
            elif args.files:
                success = True
                for file_path in args.files:
                    try:
                        result = await self.process_file(Path(file_path), None)
                        if not result:
                            success = False
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        success = False
                return 0 if success else 1
            else:
                logger.error("No input specified")
                return 1

        except Exception as e:
            logger.error(f"Application error: {e}")
            return 1

        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files"
    )
    parser.add_argument(
        '--repository',
        type=str,
        help='Local path or Git URL of the repository to process'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Python files to process'
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        generator = DocumentationGenerator()
        exit_code = asyncio.run(generator.main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
```


```python
"""
Response parsing service with consistent error handling and validation.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup
from core.schema_loader import load_schema
from core.docstring_processor import DocstringProcessor, DocstringData
from core.metrics import Metrics
from exceptions import ValidationError as CustomValidationError, MetricsError

logger = LoggerSetup.get_logger(__name__)

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self, metrics: Optional[Metrics] = None):
        """Initialize the response parsing service."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.docstring_processor = DocstringProcessor(metrics=metrics or Metrics())
        self.docstring_schema = load_schema('docstring_schema')
        self.function_schema = load_schema('function_tools_schema')
        self._parsing_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'validation_failures': 0
        }

    async def parse_response(
        self, 
        response: str,
        expected_format: str = 'json',
        max_retries: int = 3,
        validate_schema: bool = True
    ) -> ParsedResponse:
        """
        Parse and validate an AI response.

        Args:
            response: Raw response string to parse
            expected_format: Expected format ('json', 'markdown', 'docstring')
            max_retries: Maximum number of parsing attempts
            validate_schema: Whether to validate against schema

        Returns:
            ParsedResponse: Structured response data with metadata

        Raises:
            CustomValidationError: If validation fails
        """
        start_time = datetime.now()
        errors = []
        parsed_content = None

        self._parsing_stats['total_processed'] += 1

        try:
            # Try parsing with retries
            for attempt in range(max_retries):
                try:
                    if expected_format == 'json':
                        parsed_content = await self._parse_json_response(response)
                    elif expected_format == 'markdown':
                        parsed_content = await self._parse_markdown_response(response)
                    elif expected_format == 'docstring':
                        parsed_content = await self._parse_docstring_response(response)
                    else:
                        raise ValueError(f"Unsupported format: {expected_format}")

                    if parsed_content:
                        break

                except Exception as e:
                    errors.append(f"Parsing attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue

            # Validate if requested and content was parsed
            validation_success = False
            if parsed_content and validate_schema:
                validation_success = await self._validate_response(
                    parsed_content, 
                    expected_format
                )
                if not validation_success:
                    errors.append("Schema validation failed")
                    self._parsing_stats['validation_failures'] += 1

            # Update success/failure stats
            if parsed_content:
                self._parsing_stats['successful_parses'] += 1
            else:
                self._parsing_stats['failed_parses'] += 1
                parsed_content = self._create_fallback_response()

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return ParsedResponse(
                content=parsed_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=validation_success,
                errors=errors,
                metadata={
                    'attempts': len(errors) + 1,
                    'timestamp': datetime.now().isoformat(),
                    'response_size': len(response)
                }
            )

        except CustomValidationError as e:
            self.logger.error(f"Response parsing failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during response parsing: {e}")
            raise CustomValidationError(f"Failed to parse response: {str(e)}")

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response, handling code blocks and cleaning."""
        try:
            # Clean up response
            response = response.strip()
            
            # Extract JSON from code blocks if present
            if '```json' in response and '```' in response:
                start = response.find('```json') + 7
                end = response.rfind('```')
                if start > 7 and end > start:
                    response = response[start:end]
            
            # Remove any non-JSON content
            response = response.strip()
            if not response.startswith('{'):
                start = response.find('{')
                if start >= 0:
                    response = response[start:]
            if not response.endswith('}'):
                end = response.rfind('}')
                if end >= 0:
                    response = response[:end+1]

            return json.loads(response.strip())

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None

    async def _parse_markdown_response(self, response: str) -> Dict[str, Any]:
        """Parse markdown response into structured format."""
        sections = {
            'summary': '',
            'description': '',
            'args': [],
            'returns': {'type': 'Any', 'description': ''},
            'raises': []
        }

        current_section = None
        section_content = []

        for line in response.split('\n'):
            line = line.strip()
            
            if line.lower().startswith('# '):
                sections['summary'] = line[2:].strip()
            elif line.lower().startswith('## arguments') or line.lower().startswith('## args'):
                current_section = 'args'
                section_content = []
            elif line.lower().startswith('## returns'):
                if section_content and current_section == 'args':
                    sections['args'].extend(self._parse_arg_section(section_content))
                current_section = 'returns'
                section_content = []
            elif line.lower().startswith('## raises'):
                if current_section == 'returns':
                    sections['returns'] = self._parse_return_section(section_content)
                current_section = 'raises'
                section_content = []
            elif line:
                section_content.append(line)

        # Process final section
        if section_content:
            if current_section == 'args':
                sections['args'].extend(self._parse_arg_section(section_content))
            elif current_section == 'returns':
                sections['returns'] = self._parse_return_section(section_content)
            elif current_section == 'raises':
                sections['raises'].extend(self._parse_raises_section(section_content))

        return sections

    async def _parse_docstring_response(self, response: str) -> Dict[str, Any]:
        """Parse docstring response using DocstringProcessor."""
        docstring_data = self.docstring_processor.parse(response)
        return {
            'summary': docstring_data.summary,
            'description': docstring_data.description,
            'args': docstring_data.args,
            'returns': docstring_data.returns,
            'raises': docstring_data.raises,
            'complexity': docstring_data.complexity
        }

    def _parse_arg_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse argument section content."""
        args = []
        current_arg = None

        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                if current_arg:
                    args.append(current_arg)
                parts = line[2:].split(':')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    description = ':'.join(parts[1:]).strip()
                    current_arg = {
                        'name': name,
                        'type': self._extract_type(name),
                        'description': description
                    }
            elif current_arg and line:
                current_arg['description'] += ' ' + line

        if current_arg:
            args.append(current_arg)

        return args

    def _parse_return_section(self, lines: List[str]) -> Dict[str, str]:
        """Parse return section content."""
        if not lines:
            return {'type': 'None', 'description': ''}

        return_text = ' '.join(lines)
        if ':' in return_text:
            type_str, description = return_text.split(':', 1)
            return {
                'type': type_str.strip(),
                'description': description.strip()
            }
        return {
            'type': 'Any',
            'description': return_text.strip()
        }

    def _parse_raises_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse raises section content."""
        raises = []
        current_exception = None

        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                if current_exception:
                    raises.append(current_exception)
                parts = line[2:].split(':')
                if len(parts) >= 2:
                    exception = parts[0].strip()
                    description = ':'.join(parts[1:]).strip()
                    current_exception = {
                        'exception': exception,
                        'description': description
                    }
            elif current_exception and line:
                current_exception['description'] += ' ' + line

        if current_exception:
            raises.append(current_exception)

        return raises

    def _extract_type(self, text: str) -> str:
        """Extract type hints from text."""
        if '(' in text and ')' in text:
            type_hint = text[text.find('(') + 1:text.find(')')]
            return type_hint
        return 'Any'

    async def _validate_response(
        self,
        content: Dict[str, Any],
        format_type: str
    ) -> bool:
        """Validate response against appropriate schema."""
        try:
            if format_type == 'docstring':
                validate(instance=content, schema=self.docstring_schema)
            elif format_type == 'function':
                validate(instance=content, schema=self.function_schema)
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e}")
            return False

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when parsing fails."""
        return {
            'summary': 'Documentation unavailable',
            'description': 'Documentation could not be generated.',
            'args': [],
            'returns': {
                'type': 'Any',
                'description': 'Return value not documented'
            },
            'raises': [],
            'complexity': 1
        }

    def get_stats(self) -> Dict[str, int]:
        """Get current parsing statistics."""
        return self._parsing_stats.copy()

    async def __aenter__(self) -> 'ResponseParsingService':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # Cleanup if needed

class ParsedResponse:
    """Data class for structured response data."""

    def __init__(
        self,
        content: Any,
        format_type: str,
        parsing_time: float,
        validation_success: bool,
        errors: List[str],
        metadata: Dict[str, Any]
    ):
        self.content = content
        self.format_type = format_type
        self.parsing_time = parsing_time
        self.validation_success = validation_success
        self.errors = errors
        self.metadata = metadata
```

```python
"""
Docstring processing module for parsing, formatting, and inserting docstrings.
"""

import ast
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from jsonschema import validate, ValidationError as JsonValidationError
from core.logger import LoggerSetup
from core.metrics import Metrics
from exceptions import DocumentationError, MetricsError

logger = LoggerSetup.get_logger(__name__)

@dataclass
class DocstringData:
    """Structured representation of a docstring."""
    summary: str
    description: str
    args: List[Dict[str, Any]]
    returns: Dict[str, Any]
    raises: List[Dict[str, Any]]
    complexity: Optional[int] = None

class DocstringProcessor:
    """Processes docstrings by parsing, validating, and formatting them."""

    def __init__(self, metrics: Optional[Metrics] = None):
        """Initialize the docstring processor."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.metrics = metrics or Metrics()
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for validating docstring structure."""
        try:
            return {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "description"]
                    },
                    "raises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["exception", "description"]
                        }
                    },
                    "complexity": {"type": "integer"}
                },
                "required": ["summary", "description", "args", "returns", "raises", "complexity"]
            }
        except Exception as e:
            self.logger.error(f"Error loading JSON schema: {e}")
            raise DocumentationError(f"Failed to load JSON schema: {e}")

    def parse(self, docstring: str) -> DocstringData:
        """
        Parse a raw docstring into structured format.

        Args:
            docstring: Raw docstring text to parse.

        Returns:
            DocstringData: Structured docstring data.

        Raises:
            DocumentationError: If parsing fails.
        """
        try:
            if not docstring.strip():
                return DocstringData("", "", [], {}, [])

            # Parsing logic using external libraries or custom implementation
            # This is a placeholder for actual parsing
            return DocstringData(
                summary="Function summary",
                description="Function description",
                args=[{'name': 'param1', 'type': 'str', 'description': 'Description of param1'}],
                returns={'type': 'int', 'description': 'Return value'},
                raises=[{'exception': 'ValueError', 'description': 'Description of raised exception'}],
                complexity=self.metrics.calculate_complexity(ast.parse(docstring))
            )

        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}")
            raise DocumentationError(f"Failed to parse docstring: {e}")

    def format(self, data: DocstringData, style: str = 'google') -> str:
        """
        Format structured docstring data into a string.

        Args:
            data: Structured docstring data.
            style: The docstring style to use ('google', 'numpy', 'sphinx').

        Returns:
            str: Formatted docstring.
        """
        try:
            lines = []

            # Add summary if present
            if data.summary:
                lines.extend([data.summary, ""])

            # Add description
            if data.description:
                lines.extend([data.description, ""])

            # Add arguments section
            if data.args:
                lines.append("Args:")
                for arg in data.args:
                    lines.append(
                        f"    {arg['name']} ({arg.get('type', 'Any')}): "
                        f"{arg.get('description', '')}"
                    )
                lines.append("")

            # Add returns section
            if data.returns:
                lines.append("Returns:")
                lines.append(
                    f"    {data.returns.get('type', 'Any')}: "
                    f"{data.returns.get('description', '')}"
                )
                lines.append("")

            # Add raises section
            if data.raises:
                lines.append("Raises:")
                for exc in data.raises:
                    lines.append(
                        f"    {exc.get('exception', 'Exception')}: "
                        f"{exc.get('description', '')}"
                    )
                lines.append("")

            # Add complexity score if present
            if data.complexity is not None:
                warning = " ⚠️" if data.complexity > 10 else ""
                lines.append(f"Complexity Score: {data.complexity}{warning}")

            return "\n".join(lines).strip()

        except Exception as e:
            self.logger.error(f"Error formatting docstring: {e}")
            raise DocumentationError(f"Failed to format docstring: {e}")

    def insert_docstring(self, node: ast.AST, docstring: str) -> ast.AST:
        """
        Insert docstring into an AST node.

        Args:
            node: The AST node to update.
            docstring: The docstring to insert.

        Returns:
            ast.AST: The updated node.

        Raises:
            DocumentationError: If insertion fails.
        """
        try:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError(f"Invalid node type for docstring: {type(node)}")

            docstring_node = ast.Expr(value=ast.Constant(value=docstring))

            # Remove existing docstring if present
            if node.body and isinstance(node.body[0], ast.Expr) and \
               isinstance(node.body[0].value, ast.Constant):
                node.body.pop(0)

            # Insert new docstring
            node.body.insert(0, docstring_node)
            return node

        except Exception as e:
            self.logger.error(f"Error inserting docstring: {e}")
            raise DocumentationError(f"Failed to insert docstring: {e}")

    def update_docstring(self, existing: str, new_content: str) -> str:
        """
        Update an existing docstring with new content.

        Args:
            existing: Existing docstring.
            new_content: New content to merge.

        Returns:
            str: Updated docstring.
        """
        try:
            # Parse both docstrings
            existing_data = self.parse(existing)
            new_data = self.parse(new_content)

            # Merge data, preferring new content but keeping existing if new is empty
            merged = DocstringData(
                summary=new_data.summary or existing_data.summary,
                description=new_data.description or existing_data.description,
                args=new_data.args or existing_data.args,
                returns=new_data.returns or existing_data.returns,
                raises=new_data.raises or existing_data.raises,
                complexity=new_data.complexity or existing_data.complexity
            )

            return self.format(merged)

        except Exception as e:
            self.logger.error(f"Error updating docstring: {e}")
            raise DocumentationError(f"Failed to update docstring: {e}")
```

```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API using centralized response parsing.
"""

import asyncio
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import ast
import openai
from openai import AsyncAzureOpenAI
from core.logger import LoggerSetup
from core.cache import Cache
from core.metrics import MetricsCollector, Metrics
from core.config import AzureOpenAIConfig
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.extraction.types import ExtractionContext
from core.types import ProcessingResult, AIHandler, DocumentationContext
from api.token_management import TokenManager
from api.api_client import APIClient
from core.response_parsing import ResponseParsingService
from exceptions import ValidationError, ExtractionError

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler(AIHandler):
    def __init__(
        self, 
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        code_extractor: Optional[CodeExtractor] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Initialize metrics first
            self.metrics = metrics or Metrics()
            
            # Create extraction context
            self.context = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False
            )
            
            # Initialize components with context
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_id,
                config=self.config
            )
            self.code_extractor = code_extractor or CodeExtractor(context=self.context)
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)
            self.response_parser = response_parser or ResponseParsingService()
            
            # Initialize API client
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code before parsing."""
        try:
            processed_code = source_code.strip()
            self.logger.debug("Preprocessed source code")
            return processed_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        context: Optional[DocumentationContext] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Process source code to generate documentation.

        Args:
            source_code: The source code to process
            cache_key: Optional cache key for storing results
            extracted_info: Optional pre-extracted code information
            context: Optional extraction context

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails

        Raises:
            ExtractionError: If code extraction fails
            ValidationError: If response validation fails
        """
        try:
            # Check cache first if enabled
            if cache_key and self.cache:
                result = await self._get_cached_docstring(cache_key)
                if result:
                    return result

            # Process and validate source code
            processed_code = self._preprocess_code(source_code)
            tree = self._parse_ast(processed_code)

            # Extract metadata if not provided
            if not extracted_info:
                extracted_info = await self._extract_code_info(processed_code, context)

            # Generate prompt and call API
            prompt = self._generate_docstring_prompt(processed_code, extracted_info)
            completion = await self._call_openai_api(prompt)

            # Parse and validate response
            parsed_response = await self._parse_and_validate_response(completion)
            if parsed_response.validation_success:
                docstring_data = self.docstring_processor.parse(parsed_response.content)
                updated_code, documentation = await self._apply_docstrings(tree, docstring_data)

                # Cache the result if caching is enabled
                if cache_key and self.cache:
                    await self._cache_result(cache_key, updated_code, documentation)

                return updated_code, documentation
            else:
                self.logger.warning(f"Response parsing had errors: {parsed_response.errors}")
                return None

        except ExtractionError as e:
            self.logger.error(f"Extraction error: {e}")
            raise
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _parse_ast(self, source_code: str) -> ast.AST:
        """Parse the source code into an AST."""
        try:
            return ast.parse(source_code)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {e}")
            raise ExtractionError(f"Failed to parse code: {e}")

    async def _extract_code_info(
        self, 
        source_code: str,
        context: Optional[DocumentationContext] = None
    ) -> Dict[str, Any]:
        """Extract metadata from source code if not provided."""
        ctx = self.context
        if isinstance(context, DocumentationContext):
            ctx = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False
            )
        extraction_result = self.code_extractor.extract_code(source_code, ctx)
        if not extraction_result:
            raise ExtractionError("Failed to extract code information")
        return {
            'module_docstring': extraction_result.module_docstring,
            'metrics': extraction_result.metrics
        }

    def _generate_docstring_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Generate the prompt for docstring generation."""
        return (
            "Generate documentation for the provided code as a JSON object.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "```json\n"
            "{\n"
            '  "summary": "A brief one-line summary of the function/method",\n'
            '  "description": "Detailed description of the functionality",\n'
            '  "args": [\n'
            '    {\n'
            '      "name": "string - parameter name",\n'
            '      "type": "string - parameter data type",\n'
            '      "description": "string - brief description of the parameter"\n'
            '    }\n'
            '  ],\n'
            '  "returns": {\n'
            '    "type": "string - return data type",\n'
            '    "description": "string - brief description of return value"\n'
            '  },\n'
            '  "raises": [\n'
            '    {\n'
            '      "exception": "string - exception class name",\n'
            '      "description": "string - circumstances under which raised"\n'
            '    }\n'
            '  ],\n'
            '  "complexity": "integer - McCabe complexity score"\n'
            '}\n'
            '```\n\n'
            "VALIDATION REQUIREMENTS:\n"
            "1. All fields shown above are required\n"
            "2. All strings must be descriptive and clear\n"
            "3. Types must be accurate Python types\n"
            "4. Complexity must be a positive integer\n"
            "5. If complexity > 10, note this in the description with [WARNING]\n\n"
            "CODE TO DOCUMENT:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n\n"
            "IMPORTANT:\n"
            "1. Always include a 'complexity' field with an integer value\n"
            "2. If complexity cannot be determined, use 1 as default\n"
            "3. Never set complexity to null or omit it\n\n"
            "Respond with only the JSON object. Do not include any other text."
        )

    async def _call_openai_api(self, prompt: str) -> Any:
        """Call Azure OpenAI API with the generated prompt."""
        try:
            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error during Azure OpenAI API call: {e}")
            raise

    async def _parse_and_validate_response(self, response_content: str) -> ParsedResponse:
        """Parse and validate the API response."""
        if response_content is None:
            raise ValidationError("Empty response from AI service")
        
        return await self.response_parser.parse_response(
            response=response_content,
            expected_format='docstring',
            validate_schema=True
        )

    async def _apply_docstrings(
        self,
        tree: ast.AST,
        docstring_data: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Apply docstrings to the AST and return updated code and documentation."""
        transformer = DocstringTransformer(self.docstring_processor, docstring_data)
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        # Convert back to source code
        updated_code = ast.unparse(modified_tree)
        documentation = self.docstring_processor.format(docstring_data)
        return updated_code, documentation

    async def _get_cached_docstring(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """Retrieve cached docstring if available."""
        try:
            cached_result = await self.cache.get_cached_docstring(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for key: {cache_key}")
                code = cached_result.get("updated_code")
                docs = cached_result.get("documentation")
                return (code, docs) if isinstance(code, str) and isinstance(docs, str) else None
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
        return None

    async def _cache_result(self, cache_key: str, updated_code: str, documentation: str) -> None:
        """Cache the docstring result."""
        try:
            await self.cache.store_docstring(
                cache_key,
                {
                    "updated_code": updated_code,
                    "documentation": documentation
                }
            )
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

class DocstringTransformer(ast.NodeTransformer):
    def __init__(self, docstring_processor, docstring_data):
        self.docstring_processor = docstring_processor
        self.docstring_data = docstring_data

    def visit_Module(self, node):
        # Handle module-level docstring
        if self.docstring_data.summary:
            module_docstring = self.docstring_processor.format(self.docstring_data)
            node = self.docstring_processor.insert_docstring(node, module_docstring)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Handle class docstrings
        if hasattr(self.docstring_data, 'classes') and \
        node.name in self.docstring_data.classes:
            class_data = self.docstring_data.classes[node.name]
            class_docstring = self.docstring_processor.format(class_data)
            node = self.docstring_processor.insert_docstring(node, class_docstring)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Handle function docstrings
        if hasattr(self.docstring_data, 'functions') and \
        node.name in self.docstring_data.functions:
            func_data = self.docstring_data.functions[node.name]
            func_docstring = self.docstring_processor.format(func_data)
            node = self.docstring_processor.insert_docstring(node, func_docstring)
        return self.generic_visit(node)
```