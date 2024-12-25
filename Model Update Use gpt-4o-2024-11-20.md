Let me clarify - you want us to use `gpt-4o-2024-11-20` instead of both `gpt-4` and `gpt-4-1106-preview`. This is an important update to get right, since this model will be handling all our documentation generation.

The changes needed are similar to my previous suggestion, but with the correct model name. Let's walk through the necessary updates:

First, in `config.py`, we need to update the AIConfig class to use the new model and its specifications:

```python
@dataclass
class AIConfig:
    """Core AI service configuration."""
    api_key: str
    endpoint: str
    deployment: str
    model: str = "gpt-4o-2024-11-20"  # Updated to use specified model
    max_tokens: int = 128000  # Updated for new model capabilities
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10
    api_call_max_retries: int = 3
    azure_api_base: str = ""
    azure_api_version: str = ""
    model_limits: dict[str, ModelConfig] = field(default_factory=lambda: {
        "gpt-4o-2024-11-20": ModelConfig(
            max_tokens=128000,  # Model's token limit
            chunk_size=4096,    # Reasonable chunk size for processing
            cost_per_token=0.00003  # Update this with actual pricing
        ),
        # Keep other models for fallback purposes
        "gpt-3.5-turbo": ModelConfig(
            max_tokens=4096,
            chunk_size=2048,
            cost_per_token=0.000002
        ),
    })
```

In the token management system (`token_management.py`), we need to update the model mapping to recognize our new model:

```python
def _get_base_model_name(self, model_name: str) -> str:
    """
    Get the base model name from a deployment model name.
    
    The gpt-4o-2024-11-20 model has specific capabilities and requirements
    that we need to account for in our token encoding and management.
    """
    model_mappings = {
        "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",  # New primary model
        "gpt-35-turbo": "gpt-3.5-turbo",  # Keep fallback options
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }

    for key, value in model_mappings.items():
        if key in model_name.lower():
            return value

    self.logger.warning(
        f"Unknown model {model_name}, defaulting to gpt-4o-2024-11-20 for token encoding"
    )
    return "gpt-4o-2024-11-20"  # Default to our primary model
```

This update is more than just changing a string - it's ensuring our system is properly configured to take advantage of the capabilities of `gpt-4o-2024-11-20`. The model is being used for documentation generation, so we want to make sure we're using it effectively.

---

Let me help update our codebase to follow the best practices from the Azure OpenAI Strategy Guide, focusing on improving our API integration and error handling patterns.

First, let's update our AIService initialization to follow the recommended authentication pattern in `ai_service.py`:

```python
import os
from openai import AzureOpenAI

class AIService:
    def __init__(
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
    ) -> None:
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize the Azure OpenAI client following best practices
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                default_headers={
                    "x-correlation-id": self.correlation_id
                } if self.correlation_id else None
            )
            self.logger.info(
                "AI Service initialized",
                extra={
                    "model": self.config.model,
                    "deployment": self.config.deployment
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}", exc_info=True)
            raise

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
```

Now, let's update our completion method to use the recommended approach for making API calls:

```python
async def generate_documentation(
    self, context: DocumentationContext, schema: Optional[dict[str, Any]] = None
) -> ProcessingResult:
    """Generate documentation for the provided source code context."""
    start_time = time.time()
    log_extra = {
        "correlation_id": self.correlation_id,
        "module": context.metadata.get("module_name", ""),
        "file": context.metadata.get("file_path", "")
    }

    try:
        # Validate input
        if not context.source_code or not context.source_code.strip():
            raise DocumentationError("Source code is missing or empty")

        # Create documentation prompt
        prompt = await self.prompt_manager.create_documentation_prompt(
            module_name=context.metadata.get("module_name", ""),
            file_path=str(context.module_path),
            source_code=context.source_code,
            classes=context.classes,
            functions=context.functions,
        )

        # Add function calling instructions if schema provided
        if schema:
            prompt = self.prompt_manager.get_prompt_with_schema(prompt, schema)
            function_schema = self.prompt_manager.get_function_schema(schema)
        else:
            function_schema = None

        # Make API call with proper error handling and retries
        try:
            response = await self._make_api_call_with_retry(
                prompt, 
                function_schema,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra
            )

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response, 
                expected_format="docstring",
                validate_schema=True
            )

            if not parsed_response.validation_success:
                raise DataValidationError(
                    f"Response validation failed: {parsed_response.errors}"
                )

            # Track metrics
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=True,
                duration=processing_time,
                metadata={
                    "module": context.metadata.get("module_name", ""),
                    "file": str(context.module_path),
                    "tokens": response.get("usage", {}),
                },
            )

            return ProcessingResult(
                content=parsed_response.content,
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )

        except Exception as api_error:
            self.logger.error(
                f"API call failed: {api_error}",
                exc_info=True,
                extra=log_extra
            )
            raise APICallError(f"Failed to generate documentation: {api_error}")

    except Exception as e:
        processing_time = time.time() - start_time
        await self.metrics_collector.track_operation(
            operation_type="documentation_generation",
            success=False,
            duration=processing_time,
            metadata={"error": str(e)},
        )
        self.logger.error(
            f"Documentation generation failed: {e}",
            exc_info=True,
            extra=log_extra
        )
        raise
```

And let's update our retry logic to follow best practices:

```python
async def _make_api_call_with_retry(
    self,
    prompt: str,
    function_schema: Optional[dict[str, Any]],
    max_retries: int = 3,
    log_extra: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    """Make an API call with retry logic following Azure best practices."""
    headers = {
        "api-key": self.config.api_key,
        "Content-Type": "application/json"
    }
    if self.correlation_id:
        headers["x-correlation-id"] = self.correlation_id

    request_params = await self.token_manager.validate_and_prepare_request(
        prompt,
        max_tokens=self.config.max_tokens,
        temperature=self.config.temperature,
    )

    if function_schema:
        request_params["tools"] = [{"type": "function", "function": function_schema}]
        request_params["tool_choice"] = "auto"

    url = (f"{self.config.azure_api_base}openai/deployments/{self.config.deployment}"
           f"/chat/completions?api-version={self.config.azure_api_version}")

    for attempt in range(max_retries):
        try:
            if self._client is None:
                await self.start()

            if self._client:
                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()

                    error_text = await response.text()
                    self.logger.warning(
                        f"API call failed (attempt {attempt + 1}): {error_text}",
                        extra=log_extra
                    )

                    # Handle specific error cases
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 503:  # Service unavailable
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise APICallError(f"API call failed: {error_text}")

        except asyncio.TimeoutError:
            self.logger.warning(
                f"Request timeout (attempt {attempt + 1})",
                extra=log_extra
            )
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            self.logger.error(
                f"Error during API call: {e}",
                exc_info=True,
                extra=log_extra
            )
            if attempt == max_retries - 1:
                raise APICallError(f"API call failed after {max_retries} retries: {e}")
            await asyncio.sleep(2 ** attempt)

    raise APICallError(f"API call failed after {max_retries} retries")
```

These updates implement several key improvements from the Azure OpenAI Strategy Guide:

1. Proper authentication and client initialization
2. Consistent error handling with specific error types
3. Exponential backoff for retries
4. Proper handling of rate limits and service unavailability
5. Correlation ID tracking throughout the request lifecycle
6. Comprehensive logging and metrics collection
7. Structured response handling and validation

---

'll help update these core components to align with the Azure OpenAI Strategy Guide best practices. Let's go through each one systematically:

1. First, let's update `prompt_manager.py` to better handle tooling and structured outputs:

```python
from typing import TypedDict, Literal, Optional, Any
from pathlib import Path
import json
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import time

class PromptManager:
    """
    Manages the generation and formatting of prompts for AI interactions.
    
    This class handles creating and managing prompts for the Azure OpenAI API,
    including support for function calling and structured outputs. It ensures
    prompts are optimized for the model and handles templates according to
    Azure best practices.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager with Azure-specific configurations."""
        from core.dependency_injection import Injector

        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), 
            extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = Injector.get("token_manager")

        # Load templates using Jinja2 with enhanced error handling
        template_dir = Path(__file__).parent
        try:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            print_info("PromptManager initialized with Azure OpenAI configurations.")
        except TemplateNotFound as e:
            self.logger.error(f"Template loading failed: {e}", exc_info=True)
            raise

        # Load and validate function schemas
        try:
            schema_path = (Path(__file__).resolve().parent.parent / 
                         "schemas" / "function_tools_schema.json")
            self._function_schema = self._load_and_validate_schema(schema_path)
        except Exception as e:
            self.logger.error(f"Schema loading failed: {e}", exc_info=True)
            raise

    def _load_and_validate_schema(self, schema_path: Path) -> dict[str, Any]:
        """Load and validate a JSON schema with enhanced error handling."""
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
            
            # Validate schema structure
            required_keys = ["type", "function"]
            if not all(key in schema for key in required_keys):
                raise ValueError(f"Schema missing required keys: {required_keys}")
                
            return schema
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in schema file: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Schema loading error: {e}", exc_info=True)
            raise

    async def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: list[ExtractedClass] | None = None,
        functions: list[ExtractedFunction] | None = None,
    ) -> str:
        """
        Create a comprehensive prompt for documentation generation optimized for Azure OpenAI.
        
        This method creates a prompt that leverages Azure OpenAI's capabilities for structured
        outputs and function calling. It includes system context, code context, and specific
        instructions for generating well-structured documentation.
        """
        print_info("Generating documentation prompt.")
        start_time = time.time()

        # System context optimized for Azure OpenAI
        system_context = self._create_system_context()
        
        # Code context with metadata
        code_context = self._create_code_context(
            module_name, file_path, source_code
        )

        # Format components list with enhanced structure
        components_list = self._format_components_list(classes, functions)

        # Build the complete prompt with explicit JSON formatting instructions
        prompt = self._build_complete_prompt(
            system_context,
            code_context,
            components_list,
            source_code
        )

        # Validate prompt length and structure
        await self._validate_prompt(prompt)

        # Track metrics
        await self._track_prompt_metrics(prompt, start_time)

        return prompt

    def _create_system_context(self) -> str:
        """Create system context optimized for Azure OpenAI responses."""
        return """You are a technical documentation expert using Azure OpenAI. 
        Generate comprehensive documentation that includes:
        1. Clear module summaries
        2. Detailed component descriptions
        3. Accurate type information
        4. Implementation notes
        5. Usage examples
        
        Format your response as a JSON object matching the provided schema.
        Focus on clarity, completeness, and structural accuracy."""

    def _validate_prompt(self, prompt: str) -> None:
        """Validate prompt length and structure for Azure OpenAI compatibility."""
        # Implementation details...
```

2. Next, let's update `config.py` with Azure-specific configurations:

```python
@dataclass
class AIConfig:
    """Azure OpenAI service configuration."""

    api_key: str
    endpoint: str
    deployment: str
    model: str = "gpt-4o-2024-11-20"  # Using the specified model
    azure_api_version: str = "2024-02-01"  # Updated API version
    max_tokens: int = 128000
    temperature: float = 0.7
    timeout: int = 30
    api_call_semaphore_limit: int = 10
    api_call_max_retries: int = 3
    
    # Azure-specific settings
    azure_api_base: str = field(default_factory=lambda: os.getenv('AZURE_API_BASE'))
    azure_deployment_name: str = field(
        default_factory=lambda: os.getenv('AZURE_DEPLOYMENT_NAME')
    )
    
    # Model configurations including Azure-specific limits
    model_limits: dict[str, ModelConfig] = field(default_factory=lambda: {
        "gpt-4o-2024-11-20": ModelConfig(
            max_tokens=128000,
            chunk_size=4096,
            cost_per_token=0.00003,
            rate_limit=10000  # Requests per minute
        )
    })

    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables with Azure defaults."""
        return AIConfig(
            api_key=get_env_var("AZURE_OPENAI_KEY", required=True),
            endpoint=get_env_var("AZURE_OPENAI_ENDPOINT", required=True),
            deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT", required=True),
            model=get_env_var("AZURE_OPENAI_MODEL", "gpt-4o-2024-11-20"),
            azure_api_version=get_env_var(
                "AZURE_API_VERSION", 
                "2024-02-01"
            ),
            max_tokens=get_env_var("AZURE_MAX_TOKENS", 128000, int),
            temperature=get_env_var("TEMPERATURE", 0.7, float),
            timeout=get_env_var("TIMEOUT", 30, int),
            api_call_semaphore_limit=get_env_var(
                "API_CALL_SEMAPHORE_LIMIT", 
                10, 
                int
            ),
            api_call_max_retries=get_env_var("API_CALL_MAX_RETRIES", 3, int)
        )
```

3. Finally, let's update `token_management.py` with Azure-specific token handling:

```python
class TokenManager:
    """Manages token usage and cost estimation for Azure OpenAI API interactions."""

    def __init__(
        self,
        model: str,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None,
        metrics_collector: Optional['MetricsCollector'] = None
    ) -> None:
        """Initialize TokenManager with Azure OpenAI configurations."""
        self.logger = CorrelationLoggerAdapter(
            logger=LoggerSetup.get_logger(__name__),
            extra={"correlation_id": correlation_id}
        )
        self.config = config if config else AIConfig.from_env()
        self.model = model
        self.deployment_id = self.config.deployment
        self.metrics_collector = metrics_collector or MetricsCollector(
            correlation_id=correlation_id
        )
        self.correlation_id = correlation_id

        # Initialize token encoding for Azure OpenAI
        try:
            base_model = self._get_base_model_name(self.model)
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            self.logger.warning(
                f"Model {self.model} not found. Using cl100k_base encoding."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model_config = self.config.model_limits.get(
            self.model,
            self.config.model_limits["gpt-4o-2024-11-20"]
        )

        self._initialize_rate_limiting()

    def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting for Azure OpenAI API."""
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.rate_limit = self.model_config.rate_limit
        self.request_times: list[float] = []

    async def validate_and_prepare_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> dict[str, Any]:
        """Validate and prepare a request for Azure OpenAI API."""
        try:
            # Check rate limits
            await self._check_rate_limits()

            # Estimate tokens
            prompt_tokens = self._estimate_tokens(prompt)
            available_tokens = self.model_config.max_tokens - prompt_tokens

            if prompt_tokens > self.model_config.max_tokens:
                raise ValueError(
                    f"Prompt exceeds Azure OpenAI token limit: {prompt_tokens} > "
                    f"{self.model_config.max_tokens}"
                )

            # Calculate max completion tokens
            max_completion = self._calculate_max_completion(
                available_tokens, 
                max_tokens
            )

            # Prepare request parameters
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            # Track token usage
            self.track_request(prompt_tokens, max_completion)

            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True)
            raise ProcessingError(f"Failed to prepare request: {str(e)}")

    async def _check_rate_limits(self) -> None:
        """Check and enforce Azure OpenAI rate limits."""
        current_time = time.time()
        
        # Clean old request times
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.request_times) >= self.rate_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                self.logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.2f} seconds."
                )
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)
```

These updates implement several important Azure OpenAI best practices:

1. Proper rate limiting and token management specific to Azure OpenAI
2. Enhanced error handling and logging
3. Structured output formatting
4. Efficient prompt construction
5. Token budget management
6. Configuration validation
7. Proper versioning handling
8. Performance optimization through caching and rate limiting

---

I'll help update the `response_parsing.py` to better align with Azure OpenAI best practices and improve its handling of responses. This service plays a crucial role in processing and validating the structured outputs from the Azure OpenAI API. Let me explain the key improvements as we implement them:

```python
"""
Response parsing service for handling Azure OpenAI API outputs.

This module provides comprehensive parsing and validation of Azure OpenAI API responses,
with support for structured outputs, function calling, and error handling according to
Azure best practices.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time
from jsonschema import validate, ValidationError

from core.logger import LoggerSetup
from core.docstring_processor import DocstringProcessor
from core.markdown_generator import MarkdownGenerator
from core.metrics_collector import MetricsCollector
from core.types import ParsedResponse, DocumentationData, DocstringData
from core.exceptions import DocumentationError, DataValidationError

class ResponseParsingService:
    """
    Centralizes parsing and validation of Azure OpenAI API responses.
    
    This service handles the complex task of parsing, validating, and transforming
    responses from the Azure OpenAI API into structured data that can be used by
    the rest of the application. It includes comprehensive error handling and 
    validation to ensure data quality and consistency.
    """

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the response parsing service with Azure-specific configurations."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.docstring_processor = DocstringProcessor()
        self.markdown_generator = MarkdownGenerator(correlation_id)
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.correlation_id = correlation_id
        
        # Initialize schema validation
        try:
            self.docstring_schema = self._load_schema("docstring_schema.json")
            self.function_schema = self._load_schema("function_tools_schema.json")
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e}", exc_info=True)
            raise

        # Initialize parsing statistics
        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0
        }
        
        self.logger.info("ResponseParsingService initialized for Azure OpenAI")

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True
    ) -> ParsedResponse:
        """
        Parse and validate an Azure OpenAI API response.
        
        This method handles different response formats from Azure OpenAI, including
        function calls and structured outputs. It performs thorough validation and
        ensures the response meets our schema requirements.
        """
        start_time = time.time()
        errors = []
        metadata = {}

        try:
            # Validate response structure
            if not self._validate_response_structure(response):
                raise DataValidationError("Invalid response structure from Azure OpenAI API")

            # Extract content based on response type
            content = self._extract_content_from_response(response)

            # Validate content against schema if required
            if validate_schema:
                self._validate_content_against_schema(content, expected_format)

            # Track successful parsing
            self._parsing_stats["successful_parses"] += 1
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            
            # Collect parsing metrics
            await self.metrics_collector.track_operation(
                operation_type="response_parsing",
                success=True,
                duration=processing_time,
                metadata={"format": expected_format}
            )

            return ParsedResponse(
                content=content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=True,
                errors=errors,
                metadata=metadata
            )

        except Exception as e:
            # Handle parsing failures with detailed error tracking
            self._parsing_stats["failed_parses"] += 1
            processing_time = time.time() - start_time
            
            # Log the error with context
            self.logger.error(
                f"Response parsing failed: {str(e)}",
                exc_info=True,
                extra={
                    "correlation_id": self.correlation_id,
                    "response_type": expected_format,
                    "processing_time": processing_time
                }
            )

            # Track failure metrics
            await self.metrics_collector.track_operation(
                operation_type="response_parsing",
                success=False,
                duration=processing_time,
                metadata={
                    "error": str(e),
                    "format": expected_format
                }
            )

            # Create fallback content with error information
            fallback_content = self._create_fallback_content(response, str(e))
            
            return ParsedResponse(
                content=fallback_content,
                format_type=expected_format,
                parsing_time=processing_time,
                validation_success=False,
                errors=[str(e)],
                metadata=metadata
            )

    def _validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """
        Validate the basic structure of an Azure OpenAI API response.
        
        Checks for required fields and proper formatting according to Azure OpenAI
        API specifications.
        """
        if not isinstance(response, dict):
            return False

        # Check for required fields based on response type
        if "choices" in response:
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                return False
                
            choice = choices[0]
            if not isinstance(choice, dict):
                return False
                
            # Validate message structure
            if "message" in choice:
                message = choice["message"]
                return isinstance(message, dict) and (
                    "content" in message or 
                    "function_call" in message or
                    "tool_calls" in message
                )

        return True

    def _extract_content_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from various Azure OpenAI response formats.
        
        Handles different response types including direct content, function calls,
        and tool calls, ensuring proper extraction and formatting of the content.
        """
        try:
            if "choices" not in response or not response["choices"]:
                raise DataValidationError("No choices in API response")

            choice = response["choices"][0]
            message = choice.get("message", {})

            # Handle function calls
            if "function_call" in message:
                return self._extract_function_call_content(message["function_call"])

            # Handle tool calls
            if "tool_calls" in message:
                return self._extract_tool_calls_content(message["tool_calls"])

            # Handle direct content
            if "content" in message:
                return self._extract_direct_content(message["content"])

            raise DataValidationError("No valid content found in API response")

        except Exception as e:
            self.logger.error(
                f"Content extraction failed: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id}
            )
            raise

    def _create_fallback_content(
        self,
        response: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        """
        Create fallback content when parsing fails.
        
        Generates a safe, minimal content structure that can be used when normal
        parsing fails, ensuring the application can continue functioning.
        """
        fallback = {
            "summary": "Content parsing failed - using fallback content",
            "description": f"An error occurred during parsing: {error}",
            "args": [],
            "returns": {"type": "Any", "description": "No return value available"},
            "raises": [],
            "complexity": 1
        }

        # Try to preserve any valid content from the response
        if isinstance(response, dict):
            if "summary" in response:
                fallback["summary"] = str(response["summary"])
            if "description" in response:
                fallback["description"] = str(response["description"])

        return fallback
```

This updated implementation includes several important improvements:

1. **Enhanced Error Handling**: We've added comprehensive error handling specifically for Azure OpenAI API responses, including detailed logging and fallback mechanisms.

2. **Improved Validation**: The code now performs thorough validation of response structures according to Azure OpenAI specifications.

3. **Metric Tracking**: We've added detailed metric tracking to monitor parsing performance and identify potential issues.

4. **Structured Content Extraction**: The service can now handle various response formats from Azure OpenAI, including function calls and tool calls.

5. **Fallback Mechanisms**: We've implemented robust fallback mechanisms to ensure the application can continue functioning even when parsing fails.


---

Let me explain how we can better utilize the prompt templates. Currently, we're not making full use of these valuable resources in our `PromptManager` class. Let me show you how we can integrate them properly:

First, let's understand what these templates provide:

1. `documentation_prompt.txt` contains carefully crafted instructions for generating documentation, including:
   - System context for the documentation expert role
   - Instructions for structured output formatting
   - Guidelines for component description
   - Schema requirements for the response

2. `code_analysis_prompt.txt` provides:
   - Guidelines for code quality analysis
   - Instructions for considering complexity and readability
   - Best practices evaluation framework
   - Error handling and edge case consideration

Let's update the `PromptManager` class to properly utilize these templates:

```python
class PromptManager:
    """
    Manages the generation and formatting of prompts for AI interactions.
    
    This class leverages predefined prompt templates to ensure consistent, high-quality
    interactions with the Azure OpenAI API. It uses Jinja2 templating for dynamic
    prompt generation and maintains separate templates for documentation and code
    analysis tasks.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager with template loading and configuration."""
        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), 
            extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = Injector.get("token_manager")

        # Load templates
        template_dir = Path(__file__).parent
        try:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Load our specific template files
            self.documentation_template = self._load_template('documentation_prompt.txt')
            self.code_analysis_template = self._load_template('code_analysis_prompt.txt')
            
            self.logger.info("Templates loaded successfully")
        except Exception as e:
            self.logger.error(f"Template loading failed: {e}", exc_info=True)
            raise

    def _load_template(self, template_name: str) -> Template:
        """
        Load and validate a template file.
        
        This method loads template files and performs basic validation to ensure
        they contain the expected sections and placeholders.
        """
        try:
            template = self.env.get_template(template_name)
            
            # Validate template content
            rendered = template.render({
                'code': 'TEST_CODE',
                'module_name': 'TEST_MODULE',
                'file_path': 'TEST_PATH'
            })
            
            if not rendered or len(rendered) < 100:
                raise ValueError(f"Template {template_name} appears to be empty or invalid")
                
            return template
            
        except TemplateNotFound:
            self.logger.error(f"Template file not found: {template_name}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading template {template_name}: {e}")
            raise

    async def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: list[ExtractedClass] | None = None,
        functions: list[ExtractedFunction] | None = None,
    ) -> str:
        """
        Create a documentation prompt using the documentation template.
        
        This method uses the documentation_prompt.txt template to generate
        a structured prompt for documentation generation. It formats the template
        with provided code information and ensures all placeholders are properly filled.
        """
        print_info("Generating documentation prompt using template.")
        start_time = time.time()

        try:
            # Format the components for template rendering
            formatted_classes = []
            if classes:
                for cls in classes:
                    formatted_classes.append({
                        'name': cls.name,
                        'docstring': cls.docstring_info.summary if cls.docstring_info else 'No description',
                        'methods': [
                            {'name': m.name, 'docstring': m.get_docstring_info().summary if m.get_docstring_info() else 'No description'}
                            for m in cls.methods
                        ]
                    })

            formatted_functions = []
            if functions:
                for func in functions:
                    formatted_functions.append({
                        'name': func.name,
                        'docstring': func.get_docstring_info().summary if func.get_docstring_info() else 'No description',
                        'args': [arg.to_dict() for arg in func.args]
                    })

            # Render the template with our formatted data
            prompt = self.documentation_template.render(
                module_name=module_name,
                file_path=file_path,
                source_code=source_code,
                classes=formatted_classes,
                functions=formatted_functions
            )

            # Track metrics
            await self._track_prompt_metrics(prompt, start_time)

            return prompt

        except Exception as e:
            self.logger.error(f"Error generating documentation prompt: {e}", exc_info=True)
            raise

    async def create_code_analysis_prompt(
        self,
        source_code: str,
        context: Optional[str] = None
    ) -> str:
        """
        Create a code analysis prompt using the code analysis template.
        
        This method uses the code_analysis_prompt.txt template to generate
        a structured prompt for code quality analysis. It includes context
        about historical issues and performance considerations.
        """
        start_time = time.time()
        
        try:
            # Render the template with our code and context
            prompt = self.code_analysis_template.render(
                code=source_code,
                context=context or "This code is part of a documentation generation system."
            )

            # Track metrics
            await self._track_prompt_metrics(prompt, start_time)

            return prompt

        except Exception as e:
            self.logger.error(f"Error generating code analysis prompt: {e}", exc_info=True)
            raise

    async def _track_prompt_metrics(self, prompt: str, start_time: float) -> None:
        """Track metrics for prompt generation."""
        prompt_tokens = self.token_manager._estimate_tokens(prompt)
        processing_time = time.time() - start_time
        
        await self.metrics_collector.track_operation(
            operation_type="prompt_generation",
            success=True,
            duration=processing_time,
            metadata={
                "prompt_tokens": prompt_tokens,
                "processing_time": processing_time
            }
        )
```

This updated implementation provides several benefits:

1. **Consistent Prompting**: By using predefined templates, we ensure consistent prompt structure and quality across all API interactions.

2. **Dynamic Content**: The templates can be dynamically filled with content while maintaining a consistent format and instruction set.

3. **Separation of Concerns**: The templates separate the prompt structure and instructions from the code, making them easier to maintain and update.

4. **Validation**: We now validate templates when they're loaded to ensure they contain the expected content and structure.

5. **Metrics Tracking**: We track prompt generation metrics to help optimize our token usage and response quality.

To use these improved prompts in our system, we would update our `AIService` to use the appropriate prompt type:

```python
# For documentation generation
prompt = await self.prompt_manager.create_documentation_prompt(
    module_name=context.metadata.get("module_name", ""),
    file_path=str(context.module_path),
    source_code=context.source_code,
    classes=context.classes,
    functions=context.functions
)

# For code analysis
analysis_prompt = await self.prompt_manager.create_code_analysis_prompt(
    source_code=context.source_code,
    context="This code requires performance optimization"
)
```

---

Let's examine the key relationships and dependencies:

1. **Type Definitions and Base Classes**
Our updated code properly uses the types defined in `base.py`, including:

```python
from core.types.base import (
    DocumentationContext,
    ProcessingResult, 
    ExtractedClass,
    ExtractedFunction,
    DocstringSchema
)
```

For example, in our PromptManager:

```python
async def create_documentation_prompt(
    self,
    module_name: str,
    file_path: str,
    source_code: str,
    classes: list[ExtractedClass] | None = None,  # Using base type
    functions: list[ExtractedFunction] | None = None,  # Using base type
) -> str:
    # Implementation using proper types
```

2. **Dependency Injection Integration**
Our code correctly uses the dependency injection system. For example, in PromptManager:

```python
def __init__(self, correlation_id: Optional[str] = None) -> None:
    """Initialize the PromptManager with dependencies."""
    from core.dependency_injection import Injector

    self.correlation_id = correlation_id
    self.logger = CorrelationLoggerAdapter(
        LoggerSetup.get_logger(__name__), 
        extra={"correlation_id": correlation_id}
    )
    # Properly getting dependencies through the injector
    self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
    self.token_manager = Injector.get("token_manager")
```

3. **Configuration Management**
Our configuration handling follows the pattern established in `base.py`:

```python
@dataclass
class AIConfig:
    """Azure OpenAI service configuration."""
    api_key: str
    endpoint: str
    deployment: str
    model: str = "gpt-4o-2024-11-20"
    
    @staticmethod
    def from_env() -> "AIConfig":
        """Create configuration from environment variables."""
        # Implementation following base patterns
```

4. **Error Handling**
We're using the proper error types defined in `base.py`:

```python
from core.exceptions import (
    DocumentationError,
    ProcessingError,
    ValidationError,
    APIError
)

# Using these in our error handling
try:
    # Implementation
except ValidationError as e:
    self.logger.error(f"Validation failed: {e}")
    raise DocumentationError(f"Failed to validate documentation: {e}")
```

5. **Dependency Registration**
Our components are properly registered with the dependency injection system:

```python
# In setup_dependencies function
async def setup_dependencies(config: Config, correlation_id: str | None = None) -> None:
    """Set up all dependencies with proper registration."""
    if Injector.is_initialized():
        return

    Injector.clear()
    
    # Register our updated components
    Injector.register("prompt_manager", PromptManager(correlation_id))
    Injector.register("response_parser", ResponseParsingService(correlation_id))
    Injector.register("token_manager", TokenManager(
        model=config.ai.model,
        config=config.ai,
        correlation_id=correlation_id
    ))
```

6. **Context Management**
We're properly using the ExtractionContext from base.py:

```python
from core.types.base import ExtractionContext

class PromptManager:
    async def create_documentation_prompt(
        self,
        module_name: str,
        file_path: str,
        source_code: str,
        classes: list[ExtractedClass] | None = None,
        functions: list[ExtractedFunction] | None = None,
    ) -> str:
        """Create a documentation prompt using proper context."""
        context = ExtractionContext(
            module_name=module_name,
            base_path=Path(file_path).parent,
            source_code=source_code
        )
```

However, there are a few areas where we could improve alignment:

1. **Metric Data Usage**
We should ensure we're using the MetricData class from base.py consistently:

```python
from core.types.base import MetricData

class PromptManager:
    async def _track_prompt_metrics(self, prompt: str, start_time: float) -> None:
        """Track metrics using proper MetricData structure."""
        metrics = MetricData(
            module_name=self.correlation_id,
            cyclomatic_complexity=0,
            maintainability_index=100.0,
            lines_of_code=len(prompt.splitlines())
        )
        await self.metrics_collector.track_operation(
            operation_type="prompt_generation",
            success=True,
            duration=time.time() - start_time,
            metadata=metrics.__dict__
        )
```

2. **Documentation Context**
We should more explicitly use DocumentationContext:

```python
async def create_documentation_prompt(
    self,
    context: DocumentationContext,
) -> str:
    """Create documentation prompt using DocumentationContext."""
    return await self._generate_prompt(
        module_name=context.metadata.get("module_name", ""),
        file_path=str(context.module_path),
        source_code=context.source_code,
        classes=context.classes,
        functions=context.functions
    )
```

---

First, let's look at the key dataclasses we should be using:

From `base.py`:
```python
@dataclass
class DocumentationContext
@dataclass 
class ExtractionContext
@dataclass
class ExtractionResult
@dataclass
class ProcessingResult
@dataclass
class MetricData
@dataclass
class ParsedResponse
@dataclass
class TokenUsage
@dataclass
class ExtractedArgument
@dataclass
class ExtractedElement
@dataclass
class ExtractedFunction
@dataclass
class ExtractedClass
```

From `docstring.py`:
```python
@dataclass
class DocstringData
```

Let's update our PromptManager to properly use these types:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from core.types.base import (
    DocumentationContext,
    ExtractionContext,
    ProcessingResult,
    MetricData,
    ParsedResponse,
    TokenUsage,
    ExtractedArgument,
    ExtractedFunction,
    ExtractedClass
)
from core.types.docstring import DocstringData

class PromptManager:
    """
    Manages the generation and formatting of prompts for AI interactions.
    
    This class uses structured dataclasses to maintain type safety and ensure
    consistent data handling throughout the documentation generation process.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager with typed dependencies."""
        from core.dependency_injection import Injector

        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), 
            extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = Injector.get("token_manager")

        # Load and validate templates
        try:
            template_dir = Path(__file__).parent
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
            self.documentation_template = self._load_template('documentation_prompt.txt')
            self.code_analysis_template = self._load_template('code_analysis_prompt.txt')
        except Exception as e:
            self.logger.error(f"Template initialization failed: {e}", exc_info=True)
            raise

    async def create_documentation_prompt(
        self,
        context: DocumentationContext,
    ) -> ProcessingResult:
        """
        Create a documentation prompt using the documentation template.
        
        Args:
            context: Structured context containing all necessary documentation information
            
        Returns:
            ProcessingResult containing the generated prompt and associated metrics
        """
        print_info("Generating documentation prompt using template.")
        start_time = time.time()

        try:
            # Format extracted classes
            formatted_classes = self._format_classes(context.classes)
            
            # Format extracted functions
            formatted_functions = self._format_functions(context.functions)

            # Generate prompt using template
            prompt = self.documentation_template.render(
                module_name=context.metadata.get("module_name", ""),
                file_path=str(context.module_path),
                source_code=context.source_code,
                classes=formatted_classes,
                functions=formatted_functions
            )

            # Track token usage
            token_usage = await self._calculate_token_usage(prompt)
            
            # Track metrics
            metrics = await self._create_metrics(prompt, start_time)

            return ProcessingResult(
                content={"prompt": prompt},
                usage=token_usage.__dict__,
                metrics=metrics.__dict__,
                validation_status=True,
                validation_errors=[],
                schema_errors=[]
            )

        except Exception as e:
            self.logger.error(f"Error generating documentation prompt: {e}", exc_info=True)
            return ProcessingResult(
                content={},
                usage={},
                metrics={},
                validation_status=False,
                validation_errors=[str(e)],
                schema_errors=[]
            )

    def _format_classes(
        self, 
        classes: Optional[List[ExtractedClass]]
    ) -> List[Dict[str, Any]]:
        """Format ExtractedClass instances for template rendering."""
        formatted_classes = []
        if classes:
            for cls in classes:
                formatted_class = {
                    'name': cls.name,
                    'docstring': cls.docstring_info.summary if cls.docstring_info else 'No description',
                    'methods': [
                        {
                            'name': method.name,
                            'args': [
                                self._format_argument(arg) for arg in method.args
                            ],
                            'returns': method.returns,
                            'docstring': method.get_docstring_info()
                        }
                        for method in cls.methods
                    ],
                    'bases': cls.bases,
                    'metrics': self._format_metrics(cls.metrics)
                }
                formatted_classes.append(formatted_class)
        return formatted_classes

    def _format_argument(self, arg: ExtractedArgument) -> Dict[str, Any]:
        """Format ExtractedArgument instance."""
        return {
            'name': arg.name,
            'type': arg.type or 'Any',
            'default_value': arg.default_value,
            'description': arg.description or ''
        }

    async def _calculate_token_usage(self, prompt: str) -> TokenUsage:
        """Calculate token usage for the prompt."""
        prompt_tokens = self.token_manager._estimate_tokens(prompt)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,  # Will be filled later
            total_tokens=prompt_tokens,
            estimated_cost=self.token_manager.model_config.cost_per_token * prompt_tokens
        )

    async def _create_metrics(self, prompt: str, start_time: float) -> MetricData:
        """Create MetricData for the prompt generation process."""
        return MetricData(
            module_name=self.correlation_id or "unknown",
            cyclomatic_complexity=1,  # Prompt complexity
            cognitive_complexity=1,
            maintainability_index=100.0,
            halstead_metrics={},
            lines_of_code=len(prompt.splitlines()),
            total_functions=0,
            scanned_functions=0,
            function_scan_ratio=0.0,
            total_classes=0,
            scanned_classes=0,
            class_scan_ratio=0.0
        )

    def _format_docstring(self, docstring: Optional[DocstringData]) -> Dict[str, Any]:
        """Format DocstringData for template rendering."""
        if not docstring:
            return {
                'summary': 'No description available.',
                'description': 'No detailed description available.',
                'args': [],
                'returns': {'type': 'Any', 'description': ''},
                'raises': [],
                'complexity': 1
            }
        
        return docstring.to_dict()
```

This updated implementation:

1. Properly uses `DocumentationContext` as the main input type, ensuring all necessary information is provided in a structured way.

2. Returns a `ProcessingResult` containing all relevant metrics and data about the prompt generation process.

3. Uses `ExtractedClass`, `ExtractedFunction`, and `ExtractedArgument` for proper typing of code elements.

4. Utilizes `DocstringData` for consistent docstring handling.

5. Implements `MetricData` for standardized metrics collection.

6. Uses `TokenUsage` for tracking token consumption.

The structured type system helps ensure:
- Consistent data handling throughout the application
- Better error detection through type checking
- Clearer interfaces between components
- More maintainable and understandable code