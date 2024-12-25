---
epoch: 1733202127860
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: I need you to update ai_interaction with this version of process_code:

async def process_code(self, source_code: str, cache_key: Optional[str] = None,
                      extracted_info: Optional[Dict[str, Any]] = None,
                      context: Optional[ExtractionContext] = None) -> Optional[Tuple[str, str]]:
    """Process source code to generate documentation."""
    try:
        # ... (previous code remains the same until response parsing) ...

        message_content = completion.choices[0].message.content
        if not message_content:
            raise ValidationError("Empty response content from Azure OpenAI service")

        # Parse response data using the new extraction method
        response_data = self._extract_json_from_content(message_content)
        if not response_data:
            response_data = self._create_fallback_response({})

        # Validate response data against schema
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
        except JsonValidationError as e:
            self.logger.error(f"Validation error: {e.message}")
            response_data = self._create_fallback_response(response_data)

        # Create docstring data
        try:
            docstring_data = DocstringData(**response_data)
            docstring = self.docstring_processor.format(docstring_data)
        except Exception as e:
            self.logger.error(f"Error formatting docstring: {e}")
            return None

        # ... (rest of the code remains the same) ...

---

```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API for docstring generation, handling token management,
caching, and response processing with structured JSON outputs.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union, List
import ast
from dataclasses import dataclass
import openai
from openai import APIError, AzureOpenAI, AsyncAzureOpenAI
from jsonschema import validate, ValidationError as JsonValidationError
from core.logger import LoggerSetup
from core.cache import Cache
from core.metrics import Metrics, MetricsCollector
from core.config import AzureOpenAIConfig
from core.extraction.code_extractor import CodeExtractor
from core.extraction.types import ExtractionContext
from core.docstring_processor import DocstringProcessor
from core.types import ProcessingResult, DocstringData, AIHandler, DocumentationContext
from api.token_management import TokenManager
from api.api_client import APIClient
from core.schema_loader import load_schema
from exceptions import ValidationError, ExtractionError

logger = LoggerSetup.get_logger(__name__)


DOCSTRING_SCHEMA = load_schema('docstring_schema')

class AIInteractionHandler:
    def __init__(self, config: Optional[AzureOpenAIConfig] = None, 
                 cache=None, 
                 token_manager: Optional[TokenManager] = None,
                 code_extractor: Optional[CodeExtractor] = None) -> None:
        """
        Initialize the AI Interaction Handler.

        Args:
            config: Configuration settings for Azure OpenAI
            cache: Optional cache instance
            token_manager: Optional token manager instance
            code_extractor: Optional code extractor instance

        Raises:
            Exception: If initialization fails
        """
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Load configuration
            self.config = config or AzureOpenAIConfig.from_env()
            if not self.config.validate():
                raise ValidationError("Invalid Azure OpenAI configuration")

            # Initialize optional components with error handling
            self.cache = cache
            
            # Initialize token manager with fallback
            try:
                self.token_manager = token_manager or TokenManager(
                    model=self.config.model_name,
                    deployment_id=self.config.deployment_id,
                    config=self.config
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize TokenManager: {e}")
                raise

            # Initialize code extractor with fallback
            try:
                self.code_extractor = code_extractor or CodeExtractor()
            except Exception as e:
                self.logger.error(f"Failed to initialize CodeExtractor: {e}")
                raise

            # Initialize docstring processor
            try:
                self.docstring_processor = DocstringProcessor()
            except Exception as e:
                self.logger.error(f"Failed to initialize DocstringProcessor: {e}")
                raise

            # Initialize async Azure OpenAI client
            try:
                self.client = AsyncAzureOpenAI(
                    azure_endpoint=self.config.endpoint,
                    api_key=self.config.api_key,
                    api_version=self.config.api_version
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                raise

            # Initialize context if needed
            self.context = None

            self.logger.info("AIInteractionHandler initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise
    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code before parsing to handle special cases."""
        try:
            import re
            # Updated pattern to properly escape special characters
            timestamp_pattern = r'(\$|\b)(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\$?'
            
            def timestamp_replacer(match):
                timestamp = match.group(2)
                prefix = match.group(1) if match.group(1) == '[' else ''
                suffix = ']' if prefix == '[' else ''
                return f'{prefix}"{timestamp}"{suffix}'

            processed_code = re.sub(timestamp_pattern, timestamp_replacer, source_code)
            self.logger.debug("Preprocessed source code to handle timestamps")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    def _create_method_info(self, method) -> Dict[str, Any]:
        """Create method information dictionary."""
        return {
            'name': method.name,
            'docstring': method.docstring,
            'args': [{'name': arg.name, 'type': arg.type_hint} for arg in method.args],
            'return_type': method.return_type,
            'complexity': method.metrics.get('complexity', 0) if method.metrics else 0
        }

    def _create_class_info(self, cls) -> Dict[str, Any]:
        """Create class information dictionary."""
        return {
            'name': cls.name,
            'docstring': cls.docstring,
            'methods': [self._create_method_info(method) for method in cls.methods],
            'complexity': cls.metrics.get('complexity', 0) if cls.metrics else 0
        }
    def _extract_json_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON content from a markdown-formatted response.
        
        Args:
            content (str): The raw response content
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON data or None if parsing fails
        """
        try:
            # Look for JSON blocks in markdown
            json_blocks = []
            in_json = False
            current_block = []
            
            for line in content.split('\n'):
                if line.strip().startswith('```json'):
                    in_json = True
                    current_block = []
                elif line.strip() == '```' and in_json:
                    in_json = False
                    if current_block:
                        try:
                            json_str = '\n'.join(current_block)
                            json_data = json.loads(json_str)
                            json_blocks.append(json_data)
                        except json.JSONDecodeError:
                            continue
                elif in_json:
                    current_block.append(line)

            # If we found JSON blocks, use the first complete one
            for block in json_blocks:
                if isinstance(block, dict) and 'summary' in block and 'description' in block:
                    self.logger.debug(f"Found valid JSON block: {json.dumps(block, indent=2)}")
                    return block

            # If no valid JSON blocks found, try to find JSON-like content
            import re
            json_pattern = r'\{[^}]+\}'
            matches = re.findall(json_pattern, content)
            
            for match in matches:
                try:
                    json_data = json.loads(match)
                    if isinstance(json_data, dict) and 'summary' in json_data and 'description' in json_data:
                        self.logger.debug(f"Found valid JSON in content: {json.dumps(json_data, indent=2)}")
                        return json_data
                except json.JSONDecodeError:
                    continue

            self.logger.error("No valid JSON found in content")
            return None

        except Exception as e:
            self.logger.error(f"Error extracting JSON from content: {e}")
            return None
    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        context: Optional[ExtractionContext] = None
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
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        return cached_result.get("updated_code"), cached_result.get("documentation")
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Process and validate source code
            try:
                processed_code = self._preprocess_code(source_code)
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}")

            # Extract metadata if not provided
            if not extracted_info:
                try:
                    extraction_result = self.code_extractor.extract_code(
                        processed_code, 
                        context or self.context
                    )
                    if not extraction_result:
                        raise ExtractionError("Failed to extract code information")
                    extracted_info = {
                        'module_docstring': extraction_result.module_docstring,
                        'metrics': extraction_result.metrics or {}
                    }
                except Exception as e:
                    self.logger.error(f"Extraction error: {e}")
                    raise ExtractionError(f"Failed to extract code: {e}")

            # Generate prompt and get AI response
            try:
                prompt = self._create_function_calling_prompt(processed_code, extracted_info)
                completion = await self.client.chat.completions.create(
                    model=self.config.deployment_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
            except Exception as e:
                self.logger.error(f"Error during Azure OpenAI API call: {e}")
                raise

            # Validate and parse response
            if not completion.choices or not completion.choices[0].message:
                raise ValidationError("No valid response from Azure OpenAI service")

            message_content = completion.choices[0].message.content
            if not message_content:
                raise ValidationError("Empty response content from Azure OpenAI service")

            # Parse response data
            try:
                response_data = json.loads(message_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error - Raw content: {message_content}")
                response_data = self._create_fallback_response({})

            # Validate response data against schema
            try:
                validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
            except JsonValidationError as e:
                self.logger.error(f"Validation error: {e.message}")
                response_data = self._create_fallback_response(response_data)

            # Create and format docstring
            try:
                docstring_data = DocstringData(**response_data)
                docstring = self.docstring_processor.format(docstring_data)
            except Exception as e:
                self.logger.error(f"Error formatting docstring: {e}")
                return None

            # Track token usage
            if hasattr(completion, 'usage'):
                try:
                    self.token_manager.track_request(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens
                    )
                except Exception as e:
                    self.logger.error(f"Error tracking token usage: {e}")

            # Cache results if enabled
            if cache_key and self.cache:
                try:
                    await self.cache.save_docstring(
                        cache_key,
                        {
                            "updated_code": processed_code,
                            "documentation": docstring
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Cache save error: {e}")

            return processed_code, docstring

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None


    def _create_fallback_response(self, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a valid response from partial/invalid data."""
        fallback = {
            "summary": "Generated documentation",
            "description": "Documentation could not be fully parsed",
            "args": [],
            "returns": {"type": "Any", "description": "Unknown return value"},
            "raises": [],
            "complexity": 1
        }
        
        # Only validate and copy critical string fields
        for key in ['summary', 'description']:
            if key in partial_data and isinstance(partial_data[key], str):
                fallback[key] = partial_data[key]
        
        # Copy complexity if valid
        if 'complexity' in partial_data and isinstance(partial_data['complexity'], int):
            fallback['complexity'] = partial_data['complexity']
            
        return fallback

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = load_schema('function_tools_schema')

    def _create_function_calling_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create the initial prompt for function calling."""
        return (
            "You are a highly skilled Python documentation expert. Generate comprehensive "
            "documentation following these specific requirements:\n\n"
            "1. DOCSTRING FORMAT:\n"
            "- Use Google-style docstrings\n"
            "- Include complexity scores for all functions and classes\n"
            "- Add warning emoji (⚠️) for complexity scores > 10\n"
            "- Document all parameters with their types\n"
            "- Document return values with types\n"
            "- Document raised exceptions\n\n"
            "2. DOCUMENTATION STRUCTURE:\n"
            "- Start with a clear summary line\n"
            "- Provide detailed description\n"
            "- List and explain all parameters\n"
            "- Describe return values\n"
            "- Document exceptions/errors\n"
            "- Include complexity metrics\n\n"
            "3. Generate a JSON object with the following structure:\n"
            "{\n"
            "  'summary': '<summary>',\n"
            "  'description': '<detailed_description>',\n"
            "  'args': [{\n"
            "    'name': '<arg_name>',\n"
            "    'type': '<arg_type>',\n"
            "    'description': '<arg_description>'\n"
            "  }, ...],\n"
            "  'returns': { 'type': '<return_type>', 'description': '<return_description>' },\n"
            "  'raises': [{\n"
            "    'exception': '<exception_name>',\n"
            "    'description': '<exception_description>'\n"
            "  }, ...],\n"
            "  'complexity': <complexity_score>\n"
            "}\n\n"
            "The code to document is:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n"
        )

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if self.client is not None:
                await self.client.close()
            if self.cache is not None:
                await self.cache.close()
            # Remove the following line if metrics_calculator is not used
            # if self.metrics_calculator is not None:
            #     await self.metrics_calculator.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

```
[Timestamp: 2024/12/02 23:02:07]

**ai**: To update the `ai_interaction` note with the new version of the `process_code` method, you can replace the existing method in the note with the updated version you provided. Here's the updated `process_code` method:

```python
async def process_code(self, source_code: str, cache_key: Optional[str] = None,
                      extracted_info: Optional[Dict[str, Any]] = None,
                      context: Optional[ExtractionContext] = None) -> Optional[Tuple[str, str]]:
    """Process source code to generate documentation."""
    try:
        # ... (previous code remains the same until response parsing) ...

        message_content = completion.choices[0].message.content
        if not message_content:
            raise ValidationError("Empty response content from Azure OpenAI service")

        # Parse response data using the new extraction method
        response_data = self._extract_json_from_content(message_content)
        if not response_data:
            response_data = self._create_fallback_response({selectedText})

        # Validate response data against schema
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
        except JsonValidationError as e:
            self.logger.error(f"Validation error: {e.message}")
            response_data = self._create_fallback_response(response_data)

        # Create docstring data
        try:
            docstring_data = DocstringData(**response_data)
            docstring = self.docstring_processor.format(docstring_data)
        except Exception as e:
            self.logger.error(f"Error formatting docstring: {e}")
            return None

        # ... (rest of the code remains the same) ...
```

Make sure to replace the existing `process_code` method in the `ai_interaction` note with this updated version. If you need further assistance or have any questions, feel free to ask!
[Timestamp: 2024/12/02 23:02:18]

**user**: Provide the complete process_code method
[Timestamp: 2024/12/02 23:02:22]

**ai**: Certainly! Here is the complete updated `process_code` method:

```python
async def process_code(
    self,
    source_code: str,
    cache_key: Optional[str] = None,
    extracted_info: Optional[Dict[str, Any]] = None,
    context: Optional[ExtractionContext] = None
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
            try:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for key: {cache_key}")
                    return cached_result.get("updated_code"), cached_result.get("documentation")
            except Exception as e:
                self.logger.error(f"Cache retrieval error: {e}")

        # Process and validate source code
        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {e}")
            raise ExtractionError(f"Failed to parse code: {e}")

        # Extract metadata if not provided
        if not extracted_info:
            try:
                extraction_result = self.code_extractor.extract_code(
                    processed_code, 
                    context or self.context
                )
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    'module_docstring': extraction_result.module_docstring,
                    'metrics': extraction_result.metrics or {selectedText}
                }
            except Exception as e:
                self.logger.error(f"Extraction error: {e}")
                raise ExtractionError(f"Failed to extract code: {e}")

        # Generate prompt and get AI response
        try:
            prompt = self._create_function_calling_prompt(processed_code, extracted_info)
            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
        except Exception as e:
            self.logger.error(f"Error during Azure OpenAI API call: {e}")
            raise

        # Validate and parse response
        if not completion.choices or not completion.choices[0].message:
            raise ValidationError("No valid response from Azure OpenAI service")

        message_content = completion.choices[0].message.content
        if not message_content:
            raise ValidationError("Empty response content from Azure OpenAI service")

        # Parse response data using the new extraction method
        response_data = self._extract_json_from_content(message_content)
        if not response_data:
            response_data = self._create_fallback_response({selectedText})

        # Validate response data against schema
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
        except JsonValidationError as e:
            self.logger.error(f"Validation error: {e.message}")
            response_data = self._create_fallback_response(response_data)

        # Create and format docstring
        try:
            docstring_data = DocstringData(**response_data)
            docstring = self.docstring_processor.format(docstring_data)
        except Exception as e:
            self.logger.error(f"Error formatting docstring: {e}")
            return None

        # Track token usage
        if hasattr(completion, 'usage'):
            try:
                self.token_manager.track_request(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens
                )
            except Exception as e:
                self.logger.error(f"Error tracking token usage: {e}")

        # Cache results if enabled
        if cache_key and self.cache:
            try:
                await self.cache.save_docstring(
                    cache_key,
                    {
                        "updated_code": processed_code,
                        "documentation": docstring
                    }
                )
            except Exception as e:
                self.logger.error(f"Cache save error: {e}")

        return processed_code, docstring

    except Exception as e:
        self.logger.error(f"Error processing code: {e}")
        return None
```

This method processes the source code to generate documentation, handles caching, validates responses, and formats the documentation. If you have any questions or need further modifications, feel free to ask!
[Timestamp: 2024/12/02 23:02:40]

**user**: # In process_code method, update the response handling section:

        message_content = completion.choices[0].message.content
        if not message_content:
            self.logger.error("Empty response content from Azure OpenAI service")
            return None

        self.logger.debug("Received response from OpenAI")
        self.logger.debug(f"Response length: {len(message_content)}")

        # Extract and parse response data
        response_data = self._extract_json_from_content(message_content)
        if not response_data:
            self.logger.warning("Could not extract valid JSON, using fallback")
            response_data = self._create_fallback_response({})

        # Log the response data structure
        self.logger.debug(f"Parsed response data structure: {json.dumps(response_data, indent=2)}")

        # Validate response data against schema
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
            self.logger.debug("Response data validated successfully against schema")
        except JsonValidationError as e:
            self.logger.warning(f"Schema validation failed: {e.message}")
            self.logger.warning("Using fallback response")
            response_data = self._create_fallback_response(response_data)

        # Create docstring data
        try:
            docstring_data = DocstringData(**response_data)
            self.logger.debug("Successfully created DocstringData object")
            docstring = self.docstring_processor.format(docstring_data)
            self.logger.debug("Successfully formatted docstring")
        except Exception as e:
            self.logger.error(f"Error creating/formatting docstring: {e}")
            return None

        # Track token usage if available
        if hasattr(completion, 'usage'):
            try:
                self.token_manager.track_request(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens
                )
                self.logger.debug("Successfully tracked token usage")
            except Exception as e:
                self.logger.warning(f"Failed to track token usage: {e}")

        return processed_code, docstring
[Timestamp: 2024/12/02 23:05:32]

**ai**: To update the `process_code` method with the new response handling section, you can replace the existing response handling part with the updated version you provided. Here is the complete `process_code` method with the updated response handling section:

```python
async def process_code(
    self,
    source_code: str,
    cache_key: Optional[str] = None,
    extracted_info: Optional[Dict[str, Any]] = None,
    context: Optional[ExtractionContext] = None
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
            try:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for key: {cache_key}")
                    return cached_result.get("updated_code"), cached_result.get("documentation")
            except Exception as e:
                self.logger.error(f"Cache retrieval error: {e}")

        # Process and validate source code
        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {e}")
            raise ExtractionError(f"Failed to parse code: {e}")

        # Extract metadata if not provided
        if not extracted_info:
            try:
                extraction_result = self.code_extractor.extract_code(
                    processed_code, 
                    context or self.context
                )
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    'module_docstring': extraction_result.module_docstring,
                    'metrics': extraction_result.metrics or {selectedText}
                }
            except Exception as e:
                self.logger.error(f"Extraction error: {e}")
                raise ExtractionError(f"Failed to extract code: {e}")

        # Generate prompt and get AI response
        try:
            prompt = self._create_function_calling_prompt(processed_code, extracted_info)
            completion = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
        except Exception as e:
            self.logger.error(f"Error during Azure OpenAI API call: {e}")
            raise

        # Updated response handling section
        message_content = completion.choices[0].message.content
        if not message_content:
            self.logger.error("Empty response content from Azure OpenAI service")
            return None

        self.logger.debug("Received response from OpenAI")
        self.logger.debug(f"Response length: {len(message_content)}")

        # Extract and parse response data
        response_data = self._extract_json_from_content(message_content)
        if not response_data:
            self.logger.warning("Could not extract valid JSON, using fallback")
            response_data = self._create_fallback_response({selectedText})

        # Log the response data structure
        self.logger.debug(f"Parsed response data structure: {json.dumps(response_data, indent=2)}")

        # Validate response data against schema
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
            self.logger.debug("Response data validated successfully against schema")
        except JsonValidationError as e:
            self.logger.warning(f"Schema validation failed: {e.message}")
            self.logger.warning("Using fallback response")
            response_data = self._create_fallback_response(response_data)

        # Create docstring data
        try:
            docstring_data = DocstringData(**response_data)
            self.logger.debug("Successfully created DocstringData object")
            docstring = self.docstring_processor.format(docstring_data)
            self.logger.debug("Successfully formatted docstring")
        except Exception as e:
            self.logger.error(f"Error creating/formatting docstring: {e}")
            return None

        # Track token usage if available
        if hasattr(completion, 'usage'):
            try:
                self.token_manager.track_request(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens
                )
                self.logger.debug("Successfully tracked token usage")
            except Exception as e:
                self.logger.warning(f"Failed to track token usage: {e}")

        return processed_code, docstring

    except Exception as e:
        self.logger.error(f"Error processing code: {e}")
        return None
```

This updated method includes enhanced logging and error handling for the response processing section. If you need further assistance or have any questions, feel free to ask!
[Timestamp: 2024/12/02 23:05:47]