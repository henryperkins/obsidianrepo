---
epoch: 1734590788067
modelKey: gemini-2.0-flash-exp|google
tags:
  - copilot-conversation
---

**user**: """AI service module for handling Azure OpenAI API interactions."""

import asyncio
import json
import time
from typing import Any, Dict, Optional, TypeVar

from aiohttp.client import ClientSession  # type: ignore
from aiohttp.client import ClientTimeout  # type: ignore
from openai import AzureOpenAI  # type: ignore

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import APICallError, DocumentationError
from core.logger import LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import ProcessingResult, DocumentationContext

T = TypeVar('T')  # For generic type hints


class AIService:
    """
    Manages interactions with the Azure OpenAI API.

    This service handles API calls, response formatting, and error handling
    for the Azure OpenAI API. It uses asynchronous operations, retry logic,
    and structured data handling to ensure reliable and efficient communication
    with the AI model.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize the AI Service with Azure OpenAI configurations.

        :param config: Optional AIConfig object with Azure OpenAI details.
        :param correlation_id: Optional correlation ID for logging context.
        """
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config: AIConfig = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize the Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                default_headers=(
                    {"x-correlation-id": self.correlation_id}
                    if self.correlation_id
                    else None
                ),
            )
            self.logger.info(
                "AI Service initialized",
                extra={
                    "model": self.config.model,
                    "deployment": self.config.deployment,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}", exc_info=True)
            raise

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")  # type: ignore
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

        # Initialize docstring processor
        try:
            self.docstring_processor = Injector.get("docstring_processor")
        except KeyError:
            self.logger.warning(
                "Docstring processor not registered, using default",
                extra={
                    "status": "warning",
                    "type": "fallback_processor",
                },
            )
            self.docstring_processor = DocstringProcessor()
            Injector.register("docstring_processor", self.docstring_processor)

        self.token_manager = TokenManager(
            model=self.config.model,
            config=self.config,
            correlation_id=correlation_id,
        )
        self.semaphore = asyncio.Semaphore(10)  # Default concurrency limit
        self._client: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self._client is None:
            self._client = ClientSession()
            self.logger.info("AI Service client session initialized")

    def _add_source_code_to_content(
        self, content: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to content fields."""
        content["source_code"] = source_code
        content.setdefault("code_metadata", {})["source_code"] = source_code
        return content

    def _add_source_code_to_function_call(
        self, function_call: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to the function call arguments if possible."""
        if "arguments" in function_call:
            try:
                args = json.loads(function_call["arguments"])
                if isinstance(args, dict):
                    args["source_code"] = source_code
                    function_call["arguments"] = json.dumps(args)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing function call arguments: {e}")
        return function_call

    def _add_source_code_to_message(
        self, message: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to message content or function/tool calls."""
        if "content" in message and message["content"] is not None:
            try:
                content = json.loads(message["content"])
                if isinstance(content, dict):
                    content = self._add_source_code_to_content(content, source_code)
                    message["content"] = json.dumps(content)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(f"Error parsing response content: {e}")
                fallback_content = {
                    "summary": "Error parsing content",
                    "description": str(message.get("content", "")),
                    "args": [],
                    "returns": {
                        "type": "Any",
                        "description": "No return value description available"
                    },
                    "raises": [],
                    "complexity": 1,
                    "source_code": source_code
                }
                message["content"] = json.dumps(fallback_content)

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    tool_call["function"] = self._add_source_code_to_function_call(
                        tool_call["function"], source_code
                    )
        elif "function_call" in message:
            message["function_call"] = self._add_source_code_to_function_call(
                message["function_call"], source_code
            )
        return message

    def _add_source_code_to_response(
        self, response: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to the entire response structure."""
        response["source_code"] = source_code
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice:
                    choice["message"] = self._add_source_code_to_message(
                        choice["message"], source_code
                    )
        return response

    def _format_fallback_response(
        self, response: Dict[str, Any], log_extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Format a fallback response for invalid structures."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response},
        )

        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "args": [],
            "returns": {
                "type": "Any",
                "description": "No return value description provided."
            },
            "raises": [],
            "complexity": 1
        }

        fallback_response = {
            "choices": [
                {"message": {"content": json.dumps(fallback_content)}}
            ],
            "usage": {}
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}", extra=log_extra
        )
        return fallback_response

    async def _make_api_call_with_retry(
        self,
        prompt: str,
        function_schema: Optional[dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
            tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
            parallel_tool_calls=self.config.parallel_tool_calls,
            response_format=str(self.config.response_format) if self.config.response_format else None,
            stream_options=self.config.stream_options,
        )

        if function_schema:
            request_params["tools"] = [
                {"type": "function", "function": function_schema}
            ]
            if not request_params.get("tool_choice"):
                request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
            f"/chat/completions?api-version={self.config.azure_api_version}"
        )

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    self.logger.info(
                        "Making API call",
                        extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                    )

                    async with self._client.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=ClientTimeout(total=self.config.timeout),
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            self.logger.info(
                                "API call succeeded",
                                extra={"status_code": response.status, "correlation_id": self.correlation_id},
                            )
                            # Check if the response is valid JSON
                            try:
                                json.dumps(response_json)
                                return response_json
                            except (TypeError, ValueError) as e:
                                self.logger.warning(
                                    f"Invalid JSON received from AI response (attempt {attempt + 1}), retrying: {e}",
                                    extra=log_extra
                                )
                                await asyncio.sleep(2**attempt)
                                continue
                        error_text = await response.text()
                        self.logger.error(
                            "API call failed",
                            extra={
                                "status_code": response.status,
                                "error_text": error_text[:200],  # Limit error text length
                                "correlation_id": self.correlation_id,
                            },
                        )

                        # Handle specific error cases
                        if response.status == 429:  # Rate limit
                            retry_after = int(
                                response.headers.get("Retry-After", 2**attempt)
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status == 503:  # Service unavailable
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})", extra=log_extra
                )
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.error(
                    f"Error during API call: {e}", exc_info=True, extra=log_extra
                )
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries: {e}"
                    )
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries")

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Generate documentation for the provided source code context.

        :param context: A DocumentationContext object containing source code and metadata.
        :param schema: Optional function schema to influence the AI's response format.
        :return: A ProcessingResult with parsed and validated documentation content.
        """
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Validate input
            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is missing or empty")

            # Create documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(context=context)

            # Add function calling instructions if schema is provided
            if schema:
                base_prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                prompt = self.prompt_manager.get_prompt_with_schema(base_prompt, schema)
                function_schema = self.prompt_manager.get_function_schema(schema)
            else:
                prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                function_schema = None

            # Make API call with retry logic
            response = await self._make_api_call_with_retry(
                prompt,
                function_schema,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra,
            )

            # Log the raw response before validation
            self.logger.debug(f"Raw AI response: {response}", extra=log_extra)

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring", validate_schema=True
            )

            # Track metrics based on validation success
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=parsed_response.validation_success,
                duration=processing_time,
                metadata={
                    "module": context.metadata.get("module_name", ""),
                    "file": str(context.module_path),
                    "tokens": response.get("usage", {}),
                    "validation_success": parsed_response.validation_success,
                    "errors": parsed_response.errors if not parsed_response.validation_success else None
                },
            )

            # Return ProcessingResult with validation status and any errors
            return ProcessingResult(
                content=parsed_response.content,  # Use the content even if validation failed
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=parsed_response.validation_success,
                validation_errors=parsed_response.errors or [],
                schema_errors=[],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            self.logger.error(
                f"Documentation generation failed: {e}", exc_info=True, extra=log_extra
            )
            raise

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None
            self.logger.info("AI Service client session closed")

    async def __aenter__(self) -> "AIService":
        """Enter the async context manager by starting the client session."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the async context manager by closing the client session."""
        await self.close()

[Timestamp: 2024/12/19 00:46:28]

**ai**: Okay, I will analyze the provided code and generate documentation for it.
```json
{
  "summary": "Console utilities for clean output formatting.",
  "description": "This module provides a set of utility functions to format and display output in the console. It includes functions for printing errors, status messages, metrics, code snippets, and other information in a structured and readable way. It also includes placeholders for UI updates.",
  "args": [],
  "returns": {
    "type": "None",
    "description": "This module does not return any value."
  },
  "raises": [],
  "complexity": 1,
  "functions": [
    {
      "name": "setup_live_layout",
      "summary": "Placeholder for setup_live_layout.",
      "description": "This function is a placeholder for setting up a live layout in the console.",
      "args": [],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "stop_live_layout",
      "summary": "Placeholder for stop_live_layout.",
      "description": "This function is a placeholder for stopping a live layout in the console.",
      "args": [],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "format_error_output",
      "summary": "Format error messages for clean console output.",
      "description": "This function takes an error message as input, removes excessive whitespace, and formats it for clean console output by adding an indent.",
      "args": [
        {
          "name": "error_message",
          "type": "str",
          "description": "The error message to format."
        }
      ],
      "returns": {
        "type": "str",
        "description": "The formatted error message."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_section_break",
      "summary": "Print a visual section break.",
      "description": "This function prints a visual section break using a line of dashes.",
      "args": [],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_error",
      "summary": "Display formatted error messages.",
      "description": "This function displays formatted error messages, including an optional correlation ID.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The error message to display."
        },
        {
          "name": "correlation_id",
          "type": "str | None",
          "description": "An optional correlation ID to include in the output."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_status",
      "summary": "Display formatted status messages with optional details.",
      "description": "This function displays formatted status messages with optional details provided as a dictionary.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The status message to display."
        },
        {
          "name": "details",
          "type": "dict[str, Any] | None",
          "description": "An optional dictionary of details to include in the output."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_metrics",
      "summary": "Display metrics in a formatted table.",
      "description": "This function displays metrics in a formatted table, handling both float and other data types.",
      "args": [
        {
          "name": "metrics",
          "type": "dict[str, Any]",
          "description": "A dictionary of metrics to display."
        },
        {
          "name": "title",
          "type": "str",
          "description": "An optional title for the metrics table."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_success",
      "summary": "Display success messages.",
      "description": "This function displays success messages, formatting them for clean console output.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The success message to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_info",
      "summary": "Display info messages with optional details.",
      "description": "This function displays info messages with optional details, which can be a dictionary or any other value.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The info message to display."
        },
        {
          "name": "details",
          "type": "Any",
          "description": "Optional details to include in the output."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "update_header",
      "summary": "Placeholder for update_header.",
      "description": "This function is a placeholder for updating the header of the console.",
      "args": [
        {
          "name": "text",
          "type": "str",
          "description": "The text to display in the header."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "update_footer",
      "summary": "Placeholder for update_footer.",
      "description": "This function is a placeholder for updating the footer of the console.",
      "args": [
        {
          "name": "text",
          "type": "str",
          "description": "The text to display in the footer."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "update_left_panel",
      "summary": "Placeholder for update_left_panel.",
      "description": "This function is a placeholder for updating the left panel of the console.",
      "args": [
        {
          "name": "renderable",
          "type": "Any",
          "description": "The content to display in the left panel."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "update_right_panel",
      "summary": "Placeholder for update_right_panel.",
      "description": "This function is a placeholder for updating the right panel of the console.",
      "args": [
        {
          "name": "renderable",
          "type": "Any",
          "description": "The content to display in the right panel."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_progress",
      "summary": "Placeholder for display_progress.",
      "description": "This function is a placeholder for displaying progress information in the console.",
      "args": [
        {
          "name": "task_description",
          "type": "str",
          "description": "The description of the task in progress."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_code_snippet",
      "summary": "Display a code snippet.",
      "description": "This function displays a code snippet with optional language, theme, and line number settings.",
      "args": [
        {
          "name": "code",
          "type": "str",
          "description": "The code snippet to display."
        },
        {
          "name": "language",
          "type": "str",
          "description": "The language of the code snippet."
        },
        {
          "name": "theme",
          "type": "str",
          "description": "The theme to use for syntax highlighting."
        },
        {
          "name": "line_numbers",
          "type": "bool",
          "description": "Whether to display line numbers."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_warning",
      "summary": "Print a warning message.",
      "description": "This function prints a warning message to the console.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The warning message to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_debug",
      "summary": "Print a debug message.",
      "description": "This function prints a debug message to the console.",
      "args": [
        {
          "name": "message",
          "type": "str",
          "description": "The debug message to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
     {
      "name": "display_metrics_report",
      "summary": "Display a formatted metrics report.",
      "description": "This function displays a formatted metrics report to the console.",
      "args": [
        {
          "name": "metrics",
          "type": "dict[str, Any]",
          "description": "The metrics to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "create_progress",
      "summary": "Create a progress object.",
      "description": "This function creates and returns a progress object.",
      "args": [],
      "returns": {
        "type": "Progress",
        "description": "A progress object."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "print_phase_header",
      "summary": "Print a section header with formatting.",
      "description": "This function prints a section header with a title, formatted with dashes.",
      "args": [
        {
          "name": "title",
          "type": "str",
          "description": "The title of the section header."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "create_status_table",
      "summary": "Create and display a status table.",
      "description": "This function creates and displays a status table with a title and data.",
      "args": [
        {
          "name": "title",
          "type": "str",
          "description": "The title of the status table."
        },
        {
          "name": "data",
          "type": "dict[str, Any]",
          "description": "The data to display in the status table."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "format_validation_status",
      "summary": "Display validation status with optional errors.",
      "description": "This function displays the validation status (passed or failed) and any optional errors.",
      "args": [
        {
          "name": "success",
          "type": "bool",
          "description": "The validation status (True for passed, False for failed)."
        },
        {
          "name": "errors",
          "type": "list[str] | None",
          "description": "An optional list of error messages."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_processing_phase",
      "summary": "Display a processing phase with formatted content.",
      "description": "This function displays a processing phase with a title and formatted content.",
      "args": [
        {
          "name": "title",
          "type": "str",
          "description": "The title of the processing phase."
        },
        {
          "name": "content",
          "type": "dict[str, Any]",
          "description": "The content to display for the processing phase."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_api_metrics",
      "summary": "Display API response metrics in a structured format.",
      "description": "This function displays API response metrics in a structured format.",
      "args": [
        {
          "name": "response_data",
          "type": "dict[str, Any]",
          "description": "The API response metrics to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_validation_results",
      "summary": "Display validation results with details.",
      "description": "This function displays validation results with optional details for each result.",
      "args": [
        {
          "name": "results",
          "type": "dict[str, bool]",
          "description": "A dictionary of validation results (True for passed, False for failed)."
        },
        {
          "name": "details",
          "type": "Optional[dict[str, Any]]",
          "description": "An optional dictionary of details for each validation result."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "display_progress_summary",
      "summary": "Display a summary of the processing progress.",
      "description": "This function displays a summary of the processing progress.",
      "args": [
        {
          "name": "summary",
          "type": "dict[str, Any]",
          "description": "The summary of the processing progress to display."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This function does not return any value."
      },
      "raises": [],
      "complexity": 1
    }
  ]
}
```

[Timestamp: 2024/12/19 00:46:57]

**user**: """AI service module for handling Azure OpenAI API interactions."""

import asyncio
import json
import time
from typing import Any, Dict, Optional, TypeVar

from aiohttp.client import ClientSession  # type: ignore
from aiohttp.client import ClientTimeout  # type: ignore
from openai import AzureOpenAI  # type: ignore

from api.token_management import TokenManager
from core.config import AIConfig
from core.docstring_processor import DocstringProcessor
from core.exceptions import APICallError, DocumentationError
from core.logger import LoggerSetup
from core.metrics_collector import MetricsCollector
from core.prompt_manager import PromptManager
from core.types.base import ProcessingResult, DocumentationContext

T = TypeVar('T')  # For generic type hints


class AIService:
    """
    Manages interactions with the Azure OpenAI API.

    This service handles API calls, response formatting, and error handling
    for the Azure OpenAI API. It uses asynchronous operations, retry logic,
    and structured data handling to ensure reliable and efficient communication
    with the AI model.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Initialize the AI Service with Azure OpenAI configurations.

        :param config: Optional AIConfig object with Azure OpenAI details.
        :param correlation_id: Optional correlation ID for logging context.
        """
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config: AIConfig = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize the Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                default_headers=(
                    {"x-correlation-id": self.correlation_id}
                    if self.correlation_id
                    else None
                ),
            )
            self.logger.info(
                "AI Service initialized",
                extra={
                    "model": self.config.model,
                    "deployment": self.config.deployment,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}", exc_info=True)
            raise

        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = Injector.get("response_parser")  # type: ignore
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)

        # Initialize docstring processor
        try:
            self.docstring_processor = Injector.get("docstring_processor")
        except KeyError:
            self.logger.warning(
                "Docstring processor not registered, using default",
                extra={
                    "status": "warning",
                    "type": "fallback_processor",
                },
            )
            self.docstring_processor = DocstringProcessor()
            Injector.register("docstring_processor", self.docstring_processor)

        self.token_manager = TokenManager(
            model=self.config.model,
            config=self.config,
            correlation_id=correlation_id,
        )
        self.semaphore = asyncio.Semaphore(10)  # Default concurrency limit
        self._client: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self._client is None:
            self._client = ClientSession()
            self.logger.info("AI Service client session initialized")

    def _add_source_code_to_content(
        self, content: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to content fields."""
        content["source_code"] = source_code
        content.setdefault("code_metadata", {})["source_code"] = source_code
        return content

    def _add_source_code_to_function_call(
        self, function_call: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to the function call arguments if possible."""
        if "arguments" in function_call:
            try:
                args = json.loads(function_call["arguments"])
                if isinstance(args, dict):
                    args["source_code"] = source_code
                    function_call["arguments"] = json.dumps(args)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing function call arguments: {e}")
        return function_call

    def _add_source_code_to_message(
        self, message: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code to message content or function/tool calls."""
        if "content" in message and message["content"] is not None:
            try:
                content = json.loads(message["content"])
                if isinstance(content, dict):
                    content = self._add_source_code_to_content(content, source_code)
                    message["content"] = json.dumps(content)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(f"Error parsing response content: {e}")
                fallback_content = {
                    "summary": "Error parsing content",
                    "description": str(message.get("content", "")),
                    "args": [],
                    "returns": {
                        "type": "Any",
                        "description": "No return value description available"
                    },
                    "raises": [],
                    "complexity": 1,
                    "source_code": source_code
                }
                message["content"] = json.dumps(fallback_content)

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    tool_call["function"] = self._add_source_code_to_function_call(
                        tool_call["function"], source_code
                    )
        elif "function_call" in message:
            message["function_call"] = self._add_source_code_to_function_call(
                message["function_call"], source_code
            )
        return message

    def _add_source_code_to_response(
        self, response: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Add source code metadata to the entire response structure."""
        response["source_code"] = source_code
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice:
                    choice["message"] = self._add_source_code_to_message(
                        choice["message"], source_code
                    )
        return response

    def _format_fallback_response(
        self, response: Dict[str, Any], log_extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Format a fallback response for invalid structures."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response},
        )

        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "args": [],
            "returns": {
                "type": "Any",
                "description": "No return value description provided."
            },
            "raises": [],
            "complexity": 1
        }

        fallback_response = {
            "choices": [
                {"message": {"content": json.dumps(fallback_content)}}
            ],
            "usage": {}
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}", extra=log_extra
        )
        return fallback_response

    async def _make_api_call_with_retry(
        self,
        prompt: str,
        function_schema: Optional[dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
            tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
            parallel_tool_calls=self.config.parallel_tool_calls,
            response_format=str(self.config.response_format) if self.config.response_format else None,
            stream_options=self.config.stream_options,
        )

        if function_schema:
            request_params["tools"] = [
                {"type": "function", "function": function_schema}
            ]
            if not request_params.get("tool_choice"):
                request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
            f"/chat/completions?api-version={self.config.azure_api_version}"
        )

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    self.logger.info(
                        "Making API call",
                        extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                    )

                    async with self._client.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=ClientTimeout(total=self.config.timeout),
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            self.logger.info(
                                "API call succeeded",
                                extra={"status_code": response.status, "correlation_id": self.correlation_id},
                            )
                            # Check if the response is valid JSON
                            try:
                                json.dumps(response_json)
                                return response_json
                            except (TypeError, ValueError) as e:
                                self.logger.warning(
                                    f"Invalid JSON received from AI response (attempt {attempt + 1}), retrying: {e}",
                                    extra=log_extra
                                )
                                await asyncio.sleep(2**attempt)
                                continue
                        error_text = await response.text()
                        self.logger.error(
                            "API call failed",
                            extra={
                                "status_code": response.status,
                                "error_text": error_text[:200],  # Limit error text length
                                "correlation_id": self.correlation_id,
                            },
                        )

                        # Handle specific error cases
                        if response.status == 429:  # Rate limit
                            retry_after = int(
                                response.headers.get("Retry-After", 2**attempt)
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status == 503:  # Service unavailable
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})", extra=log_extra
                )
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.error(
                    f"Error during API call: {e}", exc_info=True, extra=log_extra
                )
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries: {e}"
                    )
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries")

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Generate documentation for the provided source code context.

        :param context: A DocumentationContext object containing source code and metadata.
        :param schema: Optional function schema to influence the AI's response format.
        :return: A ProcessingResult with parsed and validated documentation content.
        """
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Validate input
            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is missing or empty")

            # Create documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(context=context)

            # Add function calling instructions if schema is provided
            if schema:
                base_prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                prompt = self.prompt_manager.get_prompt_with_schema(base_prompt, schema)
                function_schema = self.prompt_manager.get_function_schema(schema)
            else:
                prompt = prompt_result.content["prompt"] if isinstance(prompt_result.content, dict) else str(prompt_result.content)
                function_schema = None

            # Make API call with retry logic
            response = await self._make_api_call_with_retry(
                prompt,
                function_schema,
                max_retries=self.config.api_call_max_retries,
                log_extra=log_extra,
            )

            # Log the raw response before validation
            self.logger.debug(f"Raw AI response: {response}", extra=log_extra)

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring", validate_schema=True
            )

            # Track metrics based on validation success
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=parsed_response.validation_success,
                duration=processing_time,
                metadata={
                    "module": context.metadata.get("module_name", ""),
                    "file": str(context.module_path),
                    "tokens": response.get("usage", {}),
                    "validation_success": parsed_response.validation_success,
                    "errors": parsed_response.errors if not parsed_response.validation_success else None
                },
            )

            # Return ProcessingResult with validation status and any errors
            return ProcessingResult(
                content=parsed_response.content,  # Use the content even if validation failed
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=parsed_response.validation_success,
                validation_errors=parsed_response.errors or [],
                schema_errors=[],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            self.logger.error(
                f"Documentation generation failed: {e}", exc_info=True, extra=log_extra
            )
            raise

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self._client:
            await self._client.close()
            self._client = None
            self.logger.info("AI Service client session closed")

    async def __aenter__(self) -> "AIService":
        """Enter the async context manager by starting the client session."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the async context manager by closing the client session."""
        await self.close()

[Timestamp: 2024/12/19 00:47:01]

**ai**: ```json
{
  "summary": "AI service module for handling Azure OpenAI API interactions.",
  "description": "This module provides an AIService class that manages interactions with the Azure OpenAI API. It handles API calls, response formatting, and error handling, using asynchronous operations, retry logic, and structured data handling to ensure reliable and efficient communication with the AI model.",
  "args": [],
  "returns": {
    "type": "None",
    "description": "This module does not return any value."
  },
  "raises": [],
  "complexity": 1,
  "functions": [
    {
      "name": "__init__",
      "summary": "Initialize the AI Service with Azure OpenAI configurations.",
      "description": "This method initializes the AI Service with Azure OpenAI configurations, including the endpoint, API key, and API version. It also sets up logging, prompt management, response parsing, and metrics collection.",
      "args": [
        {
          "name": "config",
          "type": "Optional[AIConfig]",
          "description": "Optional AIConfig object with Azure OpenAI details."
        },
        {
          "name": "correlation_id",
          "type": "Optional[str]",
          "description": "Optional correlation ID for logging context."
        }
      ],
      "returns": {
        "type": "None",
        "description": "This method does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "start",
      "summary": "Start the aiohttp client session if not already started.",
      "description": "This method starts the aiohttp client session if it is not already started, allowing for asynchronous API calls.",
      "args": [],
      "returns": {
        "type": "None",
        "description": "This method does not return any value."
      },
      "raises": [],
      "complexity": 1
    },
    {
      "name": "_add_source_code_to_content",
      "summary": "Add source code metadata to content fields.",
      "description": "This method adds source code metadata to the content fields of a dictionary, including a 'source_code' key and a '
[Timestamp: 2024/12/19 00:47:06]