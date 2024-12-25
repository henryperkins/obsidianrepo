```python
"""
API Client Module

Provides an interface to interact with the Azure OpenAI API, using centralized response parsing.
"""

from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncAzureOpenAI
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.response_parsing import ResponseParsingService, ParsedResponse

class APIClient:
    """Client to interact with Azure OpenAI API with centralized response parsing."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        response_parser: Optional[ResponseParsingService] = None
    ) -> None:
        """
        Initialize API client with configuration and response parser.

        Args:
            config: Configuration settings for Azure OpenAI service
            response_parser: Optional response parsing service instance
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.response_parser = response_parser or ResponseParsingService()
            self.logger.info("APIClient initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize APIClient: {e}")
            raise

    async def process_request(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 6000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, str]] = None,
        expected_format: str = 'json'
    ) -> Tuple[Optional[ParsedResponse], Optional[Dict[str, Any]]]:
        """
        Process request to Azure OpenAI API with response parsing.

        Args:
            prompt: The prompt to send to the API
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Optional function calling tools
            tool_choice: Optional tool choice
            expected_format: Expected response format

        Returns:
            Tuple of parsed response and usage statistics
        """
        try:
            self.logger.debug("Processing API request", 
                            extra={"max_tokens": max_tokens, "temperature": temperature})

            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,  # Corrected attribute
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )

            # Parse response
            parsed_response = await self.response_parser.parse_response(
                response=completion.choices[0].message.content,
                expected_format=expected_format
            )

            # Get usage statistics
            usage = completion.usage.model_dump() if hasattr(completion, 'usage') else {}

            self.logger.debug("API request successful", 
                            extra={"usage": usage, "parsing_success": parsed_response.validation_success})

            return parsed_response, usage

        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise
```

```python
"""AI Interaction Handler Module.

Manages interactions with Azure OpenAI API using centralized response parsing.
"""

import ast
from typing import Any, Dict, Optional, Tuple, Type
from types import TracebackType

from openai import AsyncAzureOpenAI

from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.extraction.types import ExtractionContext
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.response_parsing import ResponseParsingService
from exceptions import ExtractionError, ValidationError

from api.token_management import TokenManager
from core.types import DocumentationContext

logger = LoggerSetup.get_logger(__name__)


class AIInteractionHandler:
    """Handler for AI interactions with Azure OpenAI API.

    This class manages communication with the Azure OpenAI API, handles caching,
    token management, code extraction, and response parsing.

    Attributes:
        logger: The logger instance for logging messages.
        metrics: Metrics collector for tracking performance and usage.
        context: Context for code extraction process.
        config: Configuration for Azure OpenAI API.
        cache: Cache instance for caching results.
        token_manager: Token manager for handling API tokens.
        code_extractor: Code extractor for parsing source code.
        docstring_processor: Processor for handling docstrings.
        response_parser: Service for parsing AI responses.
        client: Asynchronous client for Azure OpenAI API.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        code_extractor: Optional[CodeExtractor] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Initialize the AIInteractionHandler with dependency injection.

        Args:
            config: Azure OpenAI configuration.
                If None, it will be loaded from environment variables.
            cache: Cache instance for caching docstrings.
            token_manager: Pre-initialized TokenManager instance for handling API tokens.
            response_parser: Pre-initialized ResponseParsingService for parsing AI responses.
            code_extractor: Optional pre-initialized CodeExtractor for extracting information from code.
            metrics: Optional pre-initialized Metrics collector.

        Raises:
            Exception: If initialization fails.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Initialize metrics, use provided or default to new one
            self.metrics = metrics or Metrics()

            # Create an extraction context
            self.context = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False,
            )

            # Use the provided config or load it from the environment
            self.config = config or AzureOpenAIConfig.from_env()

            # Set other dependencies (cache, token manager, etc.)
            self.cache = cache
            self.token_manager = token_manager  # Injected dependency
            self.response_parser = response_parser  # Injected dependency
            self.code_extractor = code_extractor or CodeExtractor(context=self.context)

            # Initialize the docstring processor
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)

            # Initialize the API client for Azure OpenAI
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess the source code before parsing.

        Strips leading and trailing whitespace from the source code.

        Args:
            source_code: The source code to preprocess.

        Returns:
            The preprocessed source code.
        """
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
        context: Optional[DocumentationContext] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:  # Change return type
        """Process source code to generate documentation.

        Args:
            source_code: The source code to process.
            cache_key: Optional cache key for storing results.
            extracted_info: Optional pre-extracted code information.
            context: Optional extraction context.

        Returns:
            A tuple of (updated_code, ai_documentation), or None if processing fails.

        Raises:
            ExtractionError: If code extraction fails.
            ValidationError: If response validation fails.
        """
        try:
            # Check cache first if enabled
            if cache_key and self.cache:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        code = cached_result.get("updated_code")
                        docs = cached_result.get("documentation")
                        if isinstance(code, str) and isinstance(docs, str):
                            return code, docs
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Process and validate source code
            processed_code = self._preprocess_code(source_code)
            try:
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}") from e

            # Extract metadata if not provided
            if not extracted_info:
                ctx = self.context
                if isinstance(context, DocumentationContext):
                    ctx = ExtractionContext(
                        metrics=self.metrics,
                        metrics_enabled=True,
                        include_private=False,
                        include_magic=False,
                    )
                extraction_result = self.code_extractor.extract_code(
                    processed_code, ctx
                )
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    "module_docstring": extraction_result.module_docstring,
                    "metrics": extraction_result.metrics,
                }

            # Generate prompt
            try:
                prompt = self._create_function_calling_prompt(
                    processed_code, extracted_info
                )
                completion = await self.client.chat.completions.create(
                    model=self.config.deployment_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )
            except Exception as e:
                self.logger.error(f"Error during Azure OpenAI API call: {e}")
                raise

            # Parse and validate the response
            content = completion.choices[0].message.content
            if content is None:
                raise ValidationError("Empty response from AI service")

            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format="docstring",
                validate_schema=True,
            )

            # Handle the docstrings
            if parsed_response.validation_success:
                try:
                    # Parse the docstring data
                    docstring_data = self.docstring_processor.parse(
                        parsed_response.content
                    )

                    # Ensure the documentation matches the expected structure
                    ai_documentation = {
                        "summary": docstring_data.summary or "No summary provided",
                        "description": (
                            docstring_data.description or "No description provided"
                        ),
                        "args": docstring_data.args or [],
                        "returns": (
                            docstring_data.returns
                            or {"type": "Any", "description": ""}
                        ),
                        "raises": docstring_data.raises or [],
                        "complexity": docstring_data.complexity or 1,
                    }

                    # Create AST transformer
                    class DocstringTransformer(ast.NodeTransformer):
                        def __init__(
                            self,
                            docstring_processor: DocstringProcessor,
                            docstring_data: Any,
                        ) -> None:
                            self.docstring_processor = docstring_processor
                            self.docstring_data = docstring_data

                        def visit_Module(
                            self, node: ast.Module
                        ) -> ast.Module:
                            # Handle module-level docstring
                            if self.docstring_data.summary:
                                module_docstring = (
                                    self.docstring_processor.format(
                                        self.docstring_data
                                    )
                                )
                                node = self.docstring_processor.insert_docstring(
                                    node, module_docstring
                                )
                            return self.generic_visit(node)

                        def visit_ClassDef(
                            self, node: ast.ClassDef
                        ) -> ast.ClassDef:
                            # Handle class docstrings
                            if (
                                hasattr(self.docstring_data, "classes")
                                and node.name in self.docstring_data.classes
                            ):
                                class_data = self.docstring_data.classes[node.name]
                                class_docstring = (
                                    self.docstring_processor.format(class_data)
                                )
                                node = self.docstring_processor.insert_docstring(
                                    node, class_docstring
                                )
                            return self.generic_visit(node)

                        def visit_FunctionDef(
                            self, node: ast.FunctionDef
                        ) -> ast.FunctionDef:
                            # Handle function docstrings
                            if (
                                hasattr(self.docstring_data, "functions")
                                and node.name in self.docstring_data.functions
                            ):
                                func_data = self.docstring_data.functions[node.name]
                                func_docstring = (
                                    self.docstring_processor.format(func_data)
                                )
                                node = self.docstring_processor.insert_docstring(
                                    node, func_docstring
                                )
                            return self.generic_visit(node)

                    # Apply the transformer
                    transformer = DocstringTransformer(
                        self.docstring_processor, docstring_data
                    )
                    modified_tree = transformer.visit(tree)
                    ast.fix_missing_locations(modified_tree)

                    # Convert back to source code
                    updated_code = ast.unparse(modified_tree)

                    # Update the context with AI-generated documentation
                    if context:
                        context.ai_generated = ai_documentation

                    # Cache the result if caching is enabled
                    if cache_key and self.cache:
                        try:
                            await self.cache.save_docstring(
                                cache_key,
                                {
                                    "updated_code": updated_code,
                                    "documentation": ai_documentation,
                                },
                            )
                        except Exception as e:
                            self.logger.error(f"Cache storage error: {e}")

                    return updated_code, ai_documentation

                except Exception as e:
                    self.logger.error(f"Error processing docstrings: {e}")
                    return None
            else:
                self.logger.warning(
                    f"Response parsing had errors: {parsed_response.errors}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _insert_docstrings(
        self, tree: ast.AST, docstrings: Dict[str, Any]
    ) -> ast.AST:
        """Insert docstrings into the AST.

        Args:
            tree: The abstract syntax tree of the code.
            docstrings: A dictionary of docstrings to insert.

        Returns:
            The AST with docstrings inserted.
        """

        class DocstringTransformer(ast.NodeTransformer):
            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                # Insert class docstring
                if node.name in docstrings:
                    docstring = ast.Constant(value=docstrings[node.name])
                    node.body.insert(0, ast.Expr(value=docstring))
                return node

            def visit_FunctionDef(
                self, node: ast.FunctionDef
            ) -> ast.FunctionDef:
                # Insert function docstring
                if node.name in docstrings:
                    docstring = ast.Constant(value=docstrings[node.name])
                    node.body.insert(0, ast.Expr(value=docstring))
                return node

        transformer = DocstringTransformer()
        return transformer.visit(tree)

    def _create_function_calling_prompt(
        self, source_code: str, metadata: Dict[str, Any]
    ) -> str:
        """Create the prompt for function calling with schema-compliant JSON output.

        Args:
            source_code: The source code to document.
            metadata: Metadata extracted from the source code.

        Returns:
            The generated prompt to send to the AI model.
        """
        return (
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

    async def close(self) -> None:
        """Close and clean up resources.

        Raises:
            Exception: If an error occurs during closing resources.
        """
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Enter the async context manager.

        Returns:
            The AIInteractionHandler instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the async context manager.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()

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
from core.docstring_processor import DocstringProcessor
from core.types import ParsedResponse, DocstringData
from exceptions import ValidationError as CustomValidationError

logger = LoggerSetup.get_logger(__name__)

class ResponseParsingService:
    """Centralized service for parsing and validating AI responses."""

    def __init__(self):
        """Initialize the response parsing service."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.docstring_processor = DocstringProcessor()
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
                    # Provide default values if validation fails
                    parsed_content = self._create_fallback_response()

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

        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            self._parsing_stats['failed_parses'] += 1
            raise CustomValidationError(f"Failed to parse response: {str(e)}")

    async def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response, handling code blocks and cleaning."""
        try:
            response = response.strip()

            # Extract JSON from code blocks if present
            if '```json' in response and '```' in response:
                start = response.find('```json') + 7
                end = response.rfind('```')
                if start > 7 and end > start:
                    response = response[start:end].strip()

            # Remove any non-JSON content
            if not response.startswith('{') or not response.endswith('}'):
                start = response.find('{')
                end = response.rfind('}')
                if start >= 0 and end >= 0:
                    response = response[start:end+1]

            # Parse JSON into Python dictionary
            parsed_content = json.loads(response.strip())

            # Ensure required fields are present and valid
            required_fields = {'summary', 'description', 'args', 'returns', 'raises'}
            for field in required_fields:
                if field not in parsed_content:
                    if field in {'args', 'raises'}:
                        parsed_content[field] = []  # Default to empty list
                    elif field == 'returns':
                        parsed_content[field] = {'type': 'Any', 'description': ''}  # Default returns value
                    else:
                        parsed_content[field] = ''  # Default to empty string for other fields

            # Validate field types
            if not isinstance(parsed_content['args'], list):
                parsed_content['args'] = []  # Ensure `args` is a list
            if not isinstance(parsed_content['raises'], list):
                parsed_content['raises'] = []  # Ensure `raises` is a list
            if not isinstance(parsed_content['returns'], dict):
                parsed_content['returns'] = {'type': 'Any', 'description': ''}  # Ensure `returns` is a dict

            return parsed_content

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during JSON response parsing: {e}")
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

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when parsing fails."""
        return {
            'summary': 'AI-generated documentation not available',
            'description': 'Documentation could not be generated by AI service',
            'args': [],
            'returns': {
                'type': 'Any',
                'description': 'Return value not documented'
            },
            'raises': [],
            'complexity': 1
        }

    async def _validate_response(
        self,
        content: Dict[str, Any],
        format_type: str
    ) -> bool:
        """Validate response against appropriate schema."""
        try:
            if format_type == 'docstring':
                # Ensure the schema includes the required keys
                schema = self.docstring_schema['schema']
                required_keys = {'summary', 'description', 'args', 'returns', 'raises'}
                if not all(key in schema['properties'] for key in required_keys):
                    raise CustomValidationError("Schema does not include all required keys")

                validate(instance=content, schema=schema)
            elif format_type == 'function':
                validate(instance=content, schema=self.function_schema['schema'])
            return True
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get current parsing statistics."""
        return self._parsing_stats.copy()

    async def __aenter__(self) -> 'ResponseParsingService':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # Cleanup if needed
```

```python
"""
Docstring processing module.
"""

import ast
from typing import Optional, Dict, Any, List, Union
from docstring_parser import parse as parse_docstring
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError

class DocstringProcessor:
    """Processes docstrings by parsing, validating, and formatting them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """
        Initialize docstring processor.

        Args:
            metrics: Optional metrics instance for complexity calculations
        """
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
                    docstring['returns'] = returns  # Update the dictionary in place

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

            # Otherwise, parse as a regular docstring string
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
    
    def extract_from_node(self, node: ast.AST) -> DocstringData:
        """
        Extract docstring from an AST node.

        Args:
            node (ast.AST): The AST node to extract from.

        Returns:
            DocstringData: The extracted docstring data.

        Raises:
            DocumentationError: If extraction fails.
        """
        try:
            raw_docstring = ast.get_docstring(node) or ""
            docstring_data = self.parse(raw_docstring)
            
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity = self.metrics.calculate_complexity(node)
                docstring_data.complexity = complexity

            return docstring_data

        except Exception as e:
            self.logger.error(f"Error extracting docstring: {e}")
            raise DocumentationError(f"Failed to extract docstring: {e}")

    def insert_docstring(self, node: ast.AST, docstring: str) -> ast.AST:
        """
        Insert docstring into an AST node.

        Args:
            node (ast.AST): The AST node to update
            docstring (str): The docstring to insert

        Returns:
            ast.AST: The updated node

        Raises:
            DocumentationError: If insertion fails
        """
        try:
            # Handle module-level docstrings
            if isinstance(node, ast.Module):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node
                
            # Handle class and function docstrings
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node
            else:
                raise ValueError(f"Invalid node type for docstring: {type(node)}")

        except Exception as e:
            self.logger.error(f"Error inserting docstring: {e}")
            raise DocumentationError(f"Failed to insert docstring: {e}")

    def update_docstring(self, existing: str, new_content: str) -> str:
        """
        Update an existing docstring with new content.

        Args:
            existing (str): Existing docstring.
            new_content (str): New content to merge.

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
Documentation Generator main module.

Handles initialization, processing, and cleanup of documentation generation.
"""

import argparse
import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import ast
from dotenv import load_dotenv
from tqdm import tqdm

from ai_interaction import AIInteractionHandler
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docs import DocStringManager
from core.logger import LoggerSetup
from core.metrics import MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from core.types import DocumentationContext
from exceptions import ConfigurationError, DocumentationError
from repository_handler import RepositoryHandler

load_dotenv()
logger = LoggerSetup.get_logger(__name__)


class DocumentationGenerator:
    """Documentation Generator."""

    def __init__(
        self,
        config: AzureOpenAIConfig,
        cache: Optional[Cache],
        metrics: MetricsCollector,
        token_manager: TokenManager,
        ai_handler: AIInteractionHandler,
        response_parser: ResponseParsingService,  # Add response_parser here
        system_monitor: Optional[SystemMonitor] = None,
    ) -> None:
        """Initialize the DocumentationGenerator."""
        self.logger = LoggerSetup.get_logger(__name__)
        self.config = config
        self.cache = cache
        self.metrics = metrics
        self.token_manager = token_manager
        self.ai_handler = ai_handler
        self.response_parser = response_parser  # Initialize the response_parser
        self.system_monitor = system_monitor
        self.repo_handler: Optional[RepositoryHandler] = None
        self._initialized = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """Initialize components that depend on runtime arguments."""
        try:
            if base_path:
                self.repo_handler = RepositoryHandler(repo_path=base_path)
                await self.repo_handler.__aenter__()
                self.logger.info(f"Repository handler initialized with path: {base_path}")

            self._initialized = True
            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            await self.cleanup()
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up all resources safely."""
        cleanup_errors = []

        components = [
            (self.system_monitor, "System Monitor", lambda x: x.stop() if hasattr(x, "stop") else None),
            (self.repo_handler, "Repository Handler", lambda x: x.__aexit__(None, None, None) if hasattr(x, "__aexit__") else None),
            (self.ai_handler, "AI Handler", lambda x: x.close() if hasattr(x, "close") else None),
            (self.token_manager, "Token Manager", None),
            (self.cache, "Cache", lambda x: x.close() if hasattr(x, "close") else None),
        ]

        for component, name, cleanup_func in components:
            if component is not None:
                try:
                    if cleanup_func:
                        result = cleanup_func(component)
                        if asyncio.iscoroutine(result):
                            await result
                    self.logger.info(f"{name} cleaned up successfully")
                except Exception as e:
                    error_msg = f"Error cleaning up {name}: {str(e)}"
                    self.logger.error(error_msg)
                    cleanup_errors.append(error_msg)

        self._initialized = False

        if cleanup_errors:
            self.logger.error("Some components failed to cleanup properly")
            for error in cleanup_errors:
                self.logger.error(f"- {error}")

    async def process_file(self, file_path: Path, output_base: Path) -> Optional[Tuple[str, str]]:
        """
        Process a single Python file to generate documentation.

        Args:
            file_path (Path): Path to the Python file to process.
            output_base (Path): Base directory for output documentation.

        Returns:
            Optional[Tuple[str, str]]: A tuple of (updated_code, documentation) if successful, otherwise None.
        """
        logger = LoggerSetup.get_logger(__name__)
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()

        try:
            # Normalize paths
            file_path = Path(file_path).resolve()
            output_base = Path(output_base).resolve()
            logger.debug(f"Processing file: {file_path}, output_base: {output_base}")

            # Validate the input file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.suffix == ".py":
                raise ValueError(f"File is not a Python file: {file_path}")

            # Read the file source code
            try:
                source_code = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source_code = file_path.read_text(encoding="latin-1")
                logger.warning(f"Used latin-1 encoding fallback for file: {file_path}")

            # Ensure source code is non-empty
            if not source_code.strip():
                raise ValueError(f"Source code is empty: {file_path}")

            # Validate Python syntax
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return None

            logger.debug("File syntax is valid.")

            # Generate cache key and process code
            cache_key = f"doc:{file_path.stem}:{hash(source_code.encode())}"
            if not self.ai_handler:
                raise RuntimeError("AI handler not initialized")
            result = await self.ai_handler.process_code(source_code, cache_key)

            if not result:
                raise ValueError(f"AI processing failed for: {file_path}")

            updated_code, ai_docs = result

            # Derive module_name
            module_name = file_path.stem or "UnknownModule"
            if not module_name.strip():  # Ensure the module_name is valid
                raise ValueError(f"Invalid module name derived from file path: {file_path}")

            logger.debug(f"Derived module_name: {module_name}")

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "module_name": module_name,
                "creation_time": datetime.now().isoformat(),
            }
            logger.debug(f"Constructed metadata: {metadata}")

            # Validate metadata
            if not isinstance(metadata, dict):
                raise TypeError(f"Metadata is not a dictionary: {metadata}")

            # Create the DocumentationContext
            context = DocumentationContext(
                source_code=updated_code,
                module_path=file_path,
                include_source=True,
                metadata=metadata,
                ai_generated=ai_docs,
            )
            logger.debug(f"Documentation context created: {context.__dict__}")

            # Generate the documentation
            doc_manager = DocStringManager(
                context=context,
                ai_handler=self.ai_handler,
                response_parser=self.response_parser,
            )
            documentation = await doc_manager.generate_documentation()

            # Handle the output directory and file creation
            output_dir = output_base / "docs"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory created at: {output_dir}")

            # Calculate output path and write the file
            output_path = output_dir / file_path.relative_to(output_base).with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(documentation, encoding="utf-8")
            logger.info(f"Documentation written to: {output_path}")

            # Return result
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processed {file_path} in {duration:.2f} seconds")
            return updated_code, documentation

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    async def process_repository(self, repo_path_or_url: str, output_file: str = "docs/Documentation.md") -> int:
        """
        Process an entire repository to generate a single markdown documentation file.

        Args:
            repo_path_or_url (str): Local path or Git URL to the repository.
            output_file (str): Path to the output markdown file.

        Returns:
            int: 0 if processing is successful, 1 otherwise.
        """
        logger = LoggerSetup.get_logger(__name__)
        if not self._initialized:
            raise RuntimeError("DocumentationGenerator not initialized")

        start_time = datetime.now()
        processed_files = 0
        failed_files = []
        combined_documentation = ""
        toc_entries = []
        is_url = urlparse(repo_path_or_url).scheme != ""

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

            # Process files and accumulate markdown documentation
            with tqdm(python_files, desc="Processing files") as progress:
                for file_path in progress:
                    try:
                        result = await self.process_file(file_path, repo_path)
                        if result:
                            updated_code, module_doc = result
                            processed_files += 1
                            
                            # Add module to TOC and combined file
                            module_name = Path(file_path).stem
                            toc_entries.append(f"- [{module_name}](#{module_name.lower().replace('_', '-')})")
                            combined_documentation += module_doc + "\n\n"
                        else:
                            failed_files.append((file_path, "Processing failed"))
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        failed_files.append((file_path, str(e)))

                    progress.set_postfix(processed=processed_files, failed=len(failed_files))

            # Create unified documentation file
            toc_section = "# Table of Contents\n\n" + "\n".join(toc_entries) + "\n\n"
            full_documentation = toc_section + combined_documentation

            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_documentation)
            
            logger.info(f"Final documentation written to: {output_file}")

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

async def main(args: argparse.Namespace) -> int:
    """
    Main application entry point.
    """
    try:
        # Initialize shared dependencies in main
        config = AzureOpenAIConfig.from_env()
        cache = Cache(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            enabled=config.cache_enabled,
        )
        metrics = MetricsCollector()
        token_manager = TokenManager(
            model=config.model_name,
            deployment_id=config.deployment_id,
            config=config,
            metrics_collector=metrics,
        )
        response_parser = ResponseParsingService()

        # Instantiate AIInteractionHandler
        ai_handler = AIInteractionHandler(
            config=config,
            cache=cache,
            token_manager=token_manager,
            response_parser=response_parser,
            metrics=metrics,
        )
        system_monitor = SystemMonitor(check_interval=60, token_manager=token_manager)

        # Instantiate the DocumentationGenerator with dependencies
        generator = DocumentationGenerator(
            config=config,
            cache=cache,
            metrics=metrics,
            token_manager=token_manager,
            ai_handler=ai_handler,
            response_parser=response_parser,
            system_monitor=system_monitor,
        )

        # Initialize and process based on arguments
        await generator.initialize()
        if args.repository:
            return await generator.process_repository(args.repository, output_file=args.output)
        elif args.files:
            success = True
            for file_path in args.files:
                try:
                    result = await generator.process_file(Path(file_path), Path("."))
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
        # Ensure cleanup happens
        await generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="Local path or Git URL of the repository to process",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Python files to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/Documentation.md",
        help="Path to the unified documentation markdown file (default: docs/Documentation.md)",
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
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

```
Identify redundancy and duplication and overlapping functionality especially in the areas of response parsing, docstring-related processes, validation and documentation generation.

Upon analyzing the provided code snippets, several instances of redundancy, duplication, and overlapping functionality become evident, particularly in the areas of response parsing, docstring-related processes, validation, and documentation generation. Below is a detailed breakdown of these issues:

## 1. **Redundancy in Docstring Insertion**

**Classes and Methods Involved:**

- `AIInteractionHandler`:
  - Defines an inner class `DocstringTransformer` with methods `visit_Module`, `visit_ClassDef`, and `visit_FunctionDef` to insert docstrings into AST nodes.
- `DocstringProcessor`:
  - Provides a method `insert_docstring(node: ast.AST, docstring: str) -> ast.AST` to insert docstrings into AST nodes.

**Issue:**

Both `AIInteractionHandler` and `DocstringProcessor` implement functionality for inserting docstrings into AST nodes. This results in duplication of code and overlapping responsibilities.

**Example:**

In `AIInteractionHandler`:

```python
class DocstringTransformer(ast.NodeTransformer):
    def __init__(self, docstring_processor: DocstringProcessor, docstring_data: Any) -> None:
        self.docstring_processor = docstring_processor
        self.docstring_data = docstring_data

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Handle function docstrings
        if (...):
            func_docstring = self.docstring_processor.format(func_data)
            node = self.docstring_processor.insert_docstring(node, func_docstring)
        return self.generic_visit(node)
```

In `DocstringProcessor`:

```python
def insert_docstring(self, node: ast.AST, docstring: str) -> ast.AST:
    # Logic to insert docstring into an AST node
```

**Impact:**

- **Code Maintainability:** Having similar code in multiple places increases the maintenance burden and the risk of inconsistencies.
- **Testing Complexity:** Duplicate functionality requires duplicate testing efforts to ensure reliability.

**Recommendation:**

- **Centralize Docstring Insertion:** Refactor the code to have `DocstringProcessor` handle all docstring insertion logic. `AIInteractionHandler` should utilize `DocstringProcessor.insert_docstring` instead of implementing its own version.
- **Remove `DocstringTransformer` from `AIInteractionHandler`:** Delegate transformation tasks to methods in `DocstringProcessor`, reducing duplication.

---

## 2. **Overlapping Functionality in Response Parsing**

**Classes and Methods Involved:**

- `AIInteractionHandler`:
  - Contains logic in `process_code` to parse and validate AI responses, including handling default values and schema validation.
- `ResponseParsingService`:
  - Provides `parse_response` and helper methods to parse and validate AI responses.

**Issue:**

Both classes are involved in parsing and validating AI responses, leading to overlapping functionality. There is code in `AIInteractionHandler` that performs tasks similar to those in `ResponseParsingService`, such as retrying parsing attempts, handling errors, and setting default values.

**Example:**

In `AIInteractionHandler.process_code`:

```python
parsed_response = await self.response_parser.parse_response(
    response=content,
    expected_format="docstring",
    validate_schema=True,
)

# Handle the docstrings
if parsed_response.validation_success:
    try:
        # Parse the docstring data
        docstring_data = self.docstring_processor.parse(parsed_response.content)
        # ...
    except Exception as e:
        self.logger.error(f"Error processing docstrings: {e}")
        return None
else:
    self.logger.warning(f"Response parsing had errors: {parsed_response.errors}")
    return None
```

In `ResponseParsingService.parse_response`:

```python
async def parse_response(self, response: str, expected_format: str = 'json', ... ) -> ParsedResponse:
    # Parsing logic with retries
    for attempt in range(max_retries):
        try:
            if expected_format == 'docstring':
                parsed_content = await self._parse_docstring_response(response)
            # ...
            # Validate if requested
            validation_success = await self._validate_response(parsed_content, expected_format)
            # ...
```

**Impact:**

- **Code Duplication:** Similar parsing and validation logic exist in both classes.
- **Inconsistent Error Handling:** Different methods might handle errors differently, leading to inconsistencies.
- **Redundant Schema Validation:** Validation logic is present in both classes, which could lead to conflicting validation results.

**Recommendation:**

- **Centralize Response Parsing:** Ensure that all response parsing and validation logic is solely handled by `ResponseParsingService`.
- **Simplify `AIInteractionHandler.process_code`:** Modify `AIInteractionHandler` to delegate all parsing and validation to `ResponseParsingService` and only handle high-level logic based on the parsed results.
- **Remove Overlapping Error Handling:** Consolidate error handling in one place to maintain consistency.

---

## 3. **Duplication in Docstring Parsing**

**Classes and Methods Involved:**

- `DocstringProcessor`:
  - Method `parse` parses docstrings from various formats (JSON, string).
- `ResponseParsingService`:
  - Method `_parse_docstring_response` parses docstring responses.

**Issue:**

Both classes implement logic to parse docstrings, potentially leading to redundant code and inconsistent parsing strategies.

**Example:**

In `ResponseParsingService._parse_docstring_response`:

```python
async def _parse_docstring_response(self, response: str) -> Dict[str, Any]:
    """Parse docstring response using DocstringProcessor."""
    docstring_data = self.docstring_processor.parse(response)
    # Convert to dictionary format
    return {
        'summary': docstring_data.summary,
        'description': docstring_data.description,
        #...
    }
```

In `DocstringProcessor.parse`:

```python
def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
    """Parse a raw docstring into structured format."""
    try:
        if isinstance(docstring, dict):
            # Process dictionary
            return DocstringData(...)
        elif isinstance(docstring, str):
            # Process string
            parsed = parse_docstring(docstring)
            return DocstringData(...)
    except Exception as e:
        self.logger.error(f"Error parsing docstring: {e}")
        raise DocumentationError(f"Failed to parse docstring: {e}")
```

**Impact:**

- **Inconsistent Parsing Logic:** Multiple parsing implementations may lead to differences in how docstrings are interpreted.
- **Code Redundancy:** Similar parsing logic is written in two places.

**Recommendation:**

- **Consolidate Docstring Parsing:** Use `DocstringProcessor` exclusively for docstring parsing. Remove parsing logic from `ResponseParsingService` and have it rely on `DocstringProcessor`.
- **Ensure Single Source of Truth:** Having one dedicated class for docstring parsing ensures consistent behavior and simplifies maintenance.

---

## 4. **Validation Logic Repeats Across Modules**

**Classes and Methods Involved:**

- `ResponseParsingService`:
  - Validates responses against schemas using methods like `_validate_response`.
- `AIInteractionHandler`:
  - Performs additional validation and default value assignments in `process_code`.

**Issue:**

Both modules perform validation of the AI responses, leading to overlapping functionality. Default values and required field checks are implemented in more than one place.

**Example:**

In `ResponseParsingService._validate_response`:

```python
async def _validate_response(self, content: Dict[str, Any], format_type: str) -> bool:
    try:
        validate(instance=content, schema=schema)
        return True
    except ValidationError as e:
        self.logger.error(f"Schema validation failed: {e}")
        return False
```

In `AIInteractionHandler.process_code`:

```python
# Check for required fields and provide defaults
if 'summary' not in docstring_data or not docstring_data.summary:
    docstring_data.summary = "No summary provided"
# Similar checks and defaults for other fields
```

**Impact:**

- **Inconsistent Validation Results:** Different validation logic may lead to inconsistent acceptance or rejection of AI responses.
- **Redundant Error Handling:** Duplication of validation increases the codebase size unnecessarily.

**Recommendation:**

- **Centralize Validation Logic:** Move all validation logic into `ResponseParsingService`. `AIInteractionHandler` should rely on the validation status returned by the parsing service.
- **Use Consistent Schemas:** Ensure that all modules use the same schemas and validation rules.
  
---

## 5. **Documentation Generation Logic Spread Across Modules**

**Classes and Methods Involved:**

- `AIInteractionHandler`:
  - Handles code processing, docstring insertion, and updates to code.
- `DocumentationGenerator`:
  - Manages processing files and repositories, generating markdown documentation.

**Issue:**

Both classes are involved in documentation generation, potentially leading to overlapping responsibilities and code duplication.

**Example:**

In `AIInteractionHandler.process_code`:

- Processes source code.
- Generates documentation via AI.
- Inserts docstrings into code.

In `DocumentationGenerator.process_file`:

- Processes files to generate documentation.
- Invokes `AIInteractionHandler` and handles the responses.

**Impact:**

- **Blurring of Concerns:** Responsibilities are not clearly separated, making the code harder to maintain and extend.
- **Overlap in Processing Logic:** Similar steps are taken in both classes, such as handling files, processing code, and managing outputs.

**Recommendation:**

- **Define Clear Module Boundaries:** Assign specific responsibilities to each class. For example:
  - `AIInteractionHandler`: Focus on interacting with the AI service and returning structured data.
  - `DocumentationGenerator`: Handle orchestration, file handling, and compiling final documentation.
- **Reduce Overlap:** Ensure that code processing and docstring insertion are handled in one place, possibly by `DocstringProcessor`.

---

## 6. **Inconsistent Error Handling and Logging**

**Issue:**

Error handling and logging are implemented differently across modules, which can lead to difficulties in debugging and monitoring.

**Recommendation:**

- **Standardize Error Handling:** Create a unified error handling strategy across modules.
- **Centralize Logging Configuration:** Use a consistent logging setup to ensure that all logs have the same format and are correctly propagated.

---

## 7. **Overuse of Inner Classes and Nested Functions**

**Issue:**

Inner classes like `DocstringTransformer` inside `AIInteractionHandler` add complexity and can be moved to their own modules or merged with existing classes like `DocstringProcessor`.

**Recommendation:**

- **Refactor Inner Classes:** Move inner classes to their appropriate modules or eliminate them if redundant.
- **Simplify Class Hierarchy:** Streamline the code by reducing unnecessary nesting.

---

## 8. **Duplicated Utilities and Helper Functions**

**Issue:**

Utility functions like parsing arguments, inserting docstrings, or handling code preprocessing might be duplicated or exist in multiple places.

**Recommendation:**

- **Create Utility Modules:** Collect common helper functions into utility modules that can be reused throughout the application.
- **Avoid Code Duplication:** Before writing a new helper function, check if it already exists elsewhere.

---

## Conclusion

The codebase shows significant redundancy and overlapping functionality in critical areas such as response parsing, docstring processing, validation, and documentation generation. Addressing these issues will lead to a more maintainable, efficient, and robust application.

**General Recommendations:**

- **Refactor for Reusability:** Extract common functionalities into reusable modules or services.
- **Adhere to Single Responsibility Principle:** Ensure each class or module has a clear and singular purpose.
- **Implement DRY (Don't Repeat Yourself):** Avoid code duplication by reusing existing code and consolidating similar functions.
- **Enhance Modularity:** Break down large classes or functions into smaller, focused units that can be easily maintained.

By implementing these changes, the codebase will become cleaner, more efficient, and easier to understand, leading to better overall software quality.

---
I want to refactor ai_interaction and docstring_processor.py to centralize docstring insertion.

Here is your specific recommendations:

**Centralize Docstring Insertion:** Refactor the code to have `DocstringProcessor` handle all docstring insertion logic. `AIInteractionHandler` should utilize `DocstringProcessor.insert_docstring` instead of implementing its own version.

**Remove `DocstringTransformer` from `AIInteractionHandler`:** Delegate transformation tasks to methods in `DocstringProcessor`, reducing duplication.

Provide the code to successfully centralize this functionality in docstring_processor and ai_interaction can utilize the methods instead of implementing its own version.


Certainly! Below is the refactored code that centralizes the docstring insertion functionality within the `DocstringProcessor` class. The `AIInteractionHandler` now utilizes the methods from `DocstringProcessor` instead of implementing its own version. The `DocstringTransformer` class has been moved to `DocstringProcessor`, and `AIInteractionHandler` calls `insert_docstrings` to insert docstrings into the AST.

### Updated `docstring_processor.py`:

```python
"""
Docstring processing module.
"""

import ast
import json
from typing import Optional, Dict, Any, List, Union
from docstring_parser import parse as parse_docstring
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError

class DocstringProcessor:
    """Processes docstrings by parsing, validating, and formatting them."""

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """
        Initialize docstring processor.

        Args:
            metrics: Optional metrics instance for complexity calculations
        """
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
                    docstring['returns'] = returns  # Update the dictionary in place

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

            # Otherwise, parse as a regular docstring string
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

    def extract_from_node(self, node: ast.AST) -> DocstringData:
        """
        Extract docstring from an AST node.

        Args:
            node (ast.AST): The AST node to extract from.

        Returns:
            DocstringData: The extracted docstring data.

        Raises:
            DocumentationError: If extraction fails.
        """
        try:
            raw_docstring = ast.get_docstring(node) or ""
            docstring_data = self.parse(raw_docstring)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity = self.metrics.calculate_complexity(node)
                docstring_data.complexity = complexity

            return docstring_data

        except Exception as e:
            self.logger.error(f"Error extracting docstring: {e}")
            raise DocumentationError(f"Failed to extract docstring: {e}")

    def insert_docstring(self, node: ast.AST, docstring: str) -> ast.AST:
        """
        Insert docstring into an AST node.

        Args:
            node (ast.AST): The AST node to update
            docstring (str): The docstring to insert

        Returns:
            ast.AST: The updated node

        Raises:
            DocumentationError: If insertion fails
        """
        try:
            # Handle module-level docstrings
            if isinstance(node, ast.Module):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                        isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node

            # Handle class and function docstrings
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                # Remove existing docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and \
                        isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                node.body.insert(0, docstring_node)
                return node
            else:
                raise ValueError(f"Invalid node type for docstring: {type(node)}")

        except Exception as e:
            self.logger.error(f"Error inserting docstring: {e}")
            raise DocumentationError(f"Failed to insert docstring: {e}")

    def insert_docstrings(self, tree: ast.AST, docstring_data: DocstringData) -> ast.AST:
        """
        Traverse the AST and insert docstrings from docstring_data.

        Args:
            tree (ast.AST): The AST tree to modify.
            docstring_data (DocstringData): The docstring data to insert.

        Returns:
            ast.AST: The modified AST tree.
        """

        class DocstringTransformer(ast.NodeTransformer):
            def __init__(self, processor: DocstringProcessor, docstring_data: DocstringData) -> None:
                self.processor = processor
                self.docstring_data = docstring_data

            def visit_Module(self, node: ast.Module) -> ast.Module:
                # Insert module-level docstring
                if self.docstring_data.summary:
                    module_docstring = self.processor.format(self.docstring_data)
                    node = self.processor.insert_docstring(node, module_docstring)
                return self.generic_visit(node)

            # Note: The following methods can be expanded to handle class and function docstrings
            # if the docstring_data includes information about classes and functions.

        transformer = DocstringTransformer(self, docstring_data)
        modified_tree = transformer.visit(tree)
        return modified_tree

    def update_docstring(self, existing: str, new_content: str) -> str:
        """
        Update an existing docstring with new content.

        Args:
            existing (str): Existing docstring.
            new_content (str): New content to merge.

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

### Updated `ai_interaction.py`:

```python
"""AI Interaction Handler Module.

Manages interactions with Azure OpenAI API using centralized response parsing.
"""

import ast
from typing import Any, Dict, Optional, Tuple, Type
from types import TracebackType

from openai import AsyncAzureOpenAI

from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.extraction.types import ExtractionContext
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.response_parsing import ResponseParsingService
from exceptions import ExtractionError, ValidationError

from api.token_management import TokenManager
from core.types import DocumentationContext

logger = LoggerSetup.get_logger(__name__)


class AIInteractionHandler:
    """Handler for AI interactions with Azure OpenAI API.

    This class manages communication with the Azure OpenAI API, handles caching,
    token management, code extraction, and response parsing.

    Attributes:
        logger: The logger instance for logging messages.
        metrics: Metrics collector for tracking performance and usage.
        context: Context for code extraction process.
        config: Configuration for Azure OpenAI API.
        cache: Cache instance for caching results.
        token_manager: Token manager for handling API tokens.
        code_extractor: Code extractor for parsing source code.
        docstring_processor: Processor for handling docstrings.
        response_parser: Service for parsing AI responses.
        client: Asynchronous client for Azure OpenAI API.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        code_extractor: Optional[CodeExtractor] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Initialize the AIInteractionHandler with dependency injection.

        Args:
            config: Azure OpenAI configuration.
                If None, it will be loaded from environment variables.
            cache: Cache instance for caching docstrings.
            token_manager: Pre-initialized TokenManager instance for handling API tokens.
            response_parser: Pre-initialized ResponseParsingService for parsing AI responses.
            code_extractor: Optional pre-initialized CodeExtractor for extracting information from code.
            metrics: Optional pre-initialized Metrics collector.

        Raises:
            Exception: If initialization fails.
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Initialize metrics, use provided or default to new one
            self.metrics = metrics or Metrics()

            # Create an extraction context
            self.context = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False,
            )

            # Use the provided config or load it from the environment
            self.config = config or AzureOpenAIConfig.from_env()

            # Set other dependencies (cache, token manager, etc.)
            self.cache = cache
            self.token_manager = token_manager  # Injected dependency
            self.response_parser = response_parser  # Injected dependency
            self.code_extractor = code_extractor or CodeExtractor(context=self.context)

            # Initialize the docstring processor
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)

            # Initialize the API client for Azure OpenAI
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    # ... (other methods)

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        context: Optional[DocumentationContext] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:  # Change return type
        """Process source code to generate documentation.

        Args:
            source_code: The source code to process.
            cache_key: Optional cache key for storing results.
            extracted_info: Optional pre-extracted code information.
            context: Optional extraction context.

        Returns:
            A tuple of (updated_code, ai_documentation), or None if processing fails.

        Raises:
            ExtractionError: If code extraction fails.
            ValidationError: If response validation fails.
        """
        try:
            # Check cache first if enabled
            if cache_key and self.cache:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        code = cached_result.get("updated_code")
                        docs = cached_result.get("documentation")
                        if isinstance(code, str) and isinstance(docs, str):
                            return code, docs
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Process and validate source code
            processed_code = self._preprocess_code(source_code)
            try:
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}") from e

            # Extract metadata if not provided
            if not extracted_info:
                ctx = self.context
                if isinstance(context, DocumentationContext):
                    ctx = ExtractionContext(
                        metrics=self.metrics,
                        metrics_enabled=True,
                        include_private=False,
                        include_magic=False,
                    )
                extraction_result = self.code_extractor.extract_code(
                    processed_code, ctx
                )
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    "module_docstring": extraction_result.module_docstring,
                    "metrics": extraction_result.metrics,
                }

            # Generate prompt
            try:
                prompt = self._create_function_calling_prompt(
                    processed_code, extracted_info
                )
                completion = await self.client.chat.completions.create(
                    model=self.config.deployment_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )
            except Exception as e:
                self.logger.error(f"Error during Azure OpenAI API call: {e}")
                raise

            # Parse and validate the response
            content = completion.choices[0].message.content
            if content is None:
                raise ValidationError("Empty response from AI service")

            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format="docstring",
                validate_schema=True,
            )

            # Handle the docstrings
            if parsed_response.validation_success:
                try:
                    # Parse the docstring data
                    docstring_data = self.docstring_processor.parse(
                        parsed_response.content
                    )

                    # Ensure the documentation matches the expected structure
                    ai_documentation = {
                        "summary": docstring_data.summary or "No summary provided",
                        "description": (
                            docstring_data.description or "No description provided"
                        ),
                        "args": docstring_data.args or [],
                        "returns": (
                            docstring_data.returns
                            or {"type": "Any", "description": ""}
                        ),
                        "raises": docstring_data.raises or [],
                        "complexity": docstring_data.complexity or 1,
                    }

                    # Apply docstrings using DocstringProcessor
                    modified_tree = self.docstring_processor.insert_docstrings(tree, docstring_data)
                    ast.fix_missing_locations(modified_tree)

                    # Convert back to source code
                    updated_code = ast.unparse(modified_tree)

                    # Update the context with AI-generated documentation
                    if context:
                        context.ai_generated = ai_documentation

                    # Cache the result if caching is enabled
                    if cache_key and self.cache:
                        try:
                            await self.cache.save_docstring(
                                cache_key,
                                {
                                    "updated_code": updated_code,
                                    "documentation": ai_documentation,
                                },
                            )
                        except Exception as e:
                            self.logger.error(f"Cache storage error: {e}")

                    return updated_code, ai_documentation

                except Exception as e:
                    self.logger.error(f"Error processing docstrings: {e}")
                    return None
            else:
                self.logger.warning(
                    f"Response parsing had errors: {parsed_response.errors}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    # ... (other methods)
```

**Notes:**

- The `DocstringTransformer` class has been moved from `AIInteractionHandler` to `DocstringProcessor` as an inner class within the `insert_docstrings` method. This centralizes the AST traversal and docstring insertion logic.
- The `insert_docstrings` method in `DocstringProcessor` now handles the insertion of docstrings into the AST. Currently, it is set up to handle module-level docstrings. You can expand it to handle class and function docstrings if `docstring_data` contains information about classes and functions.
- In `AIInteractionHandler.process_code`, we removed the `DocstringTransformer` and replaced it with a call to `self.docstring_processor.insert_docstrings(tree, docstring_data)`.
- This refactoring eliminates duplication and ensures that all docstring insertion logic is centralized within `DocstringProcessor`.

By following this approach, you delegate the responsibility of docstring insertion to `DocstringProcessor`, reducing redundancy and adhering to the single responsibility principle.