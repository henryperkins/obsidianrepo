---
epoch: 1734638925781
modelKey: gemini-2.0-flash-exp|google
tags:
  - copilot-conversation
---

**user**: compare [response_parsing updated](response_parsing%20updated.md) with [response_parsing](./response_parsing.md)
[Timestamp: 2024/12/19 14:08:45]

**ai**: Okay, I will compare the two versions of the `response_parsing` code and highlight the differences.

**Key Differences:**

The primary difference between `[[response_parsing updated]]` and `[[response_parsing]]` lies in the way they handle response parsing, particularly in the structure of the `ParsedResponse` object and the way they process the content of the response. The updated version simplifies the structure and improves the parsing logic.

Here's a breakdown of the specific changes:

1.  **`ParsedResponse` Dataclass:**
    *   **`[[response_parsing]]`**: Includes a `markdown: str = ""` field in the `ParsedResponse` dataclass.
    *   **`[[response_parsing updated]]`**:  Removes the `markdown` field from the `ParsedResponse` dataclass, simplifying its structure.

2.  **Schema Loading and Handling:**
    *   Both versions load schemas from a directory, but `[[response_parsing updated]]` has more robust error handling when the schema directory or files are not found.
    *   `[[response_parsing updated]]` uses `Path` objects more consistently for file paths.

3.  **Schema Selection Logic:**
    *   Both versions use the `_select_schema` method to dynamically choose between `function_schema`, `docstring_schema`, or `_FALLBACK_SCHEMA`.
    *   The logic is similar, prioritizing `function_schema` if "parameters" or "examples" are present, then `docstring_schema` if "summary" and "description" are present.
    *   Both versions track schema usage metrics.

4.  **Response Formatting:**
    *   `[[response_parsing]]` has separate methods for formatting different types of responses (`_format_summary_description_response`, `_format_function_call_response`, `_format_tool_calls_response`).
    *   `[[response_parsing updated]]` uses a single `_format_fallback_response` method to handle all fallback cases, making it more streamlined.
    *   `[[response_parsing]]` has a `_standardize_response_format` method that attempts to convert various response formats to a standard "choices" format. `[[response_parsing updated]]` does not have this method, and relies on the validation of the response structure.

5.  **Content Extraction:**
    *   `[[response_parsing]]` has a more complex `_extract_content` method that handles various cases (direct content, function calls, tool calls) and includes methods like `_extract_content_from_function_call`, `_extract_content_from_tool_calls`, and `_extract_content_from_direct_content`.
    *   `[[response_parsing updated]]` simplifies this process by directly parsing the content within the `_parse_message_content` method.

6.  **Validation:**
    *   Both versions use `jsonschema` for validation.
    *   `[[response_parsing updated]]` has a more direct approach to validating the content against the selected schema within the `_parse_message_content` method.
    *   `[[response_parsing]]` uses `_validate_content` to validate the content against the selected schema.

7.  **Error Handling:**
    *   Both versions use the logger to report errors.
    *   `[[response_parsing updated]]` has a more streamlined approach to creating error responses using the `_create_error_response` method.
    *   `[[response_parsing updated]]` has more robust checks for empty or invalid responses.

8.  **`_ensure_required_fields` Method:**
    *   Both versions have this method, which ensures that the parsed content has the required fields (`summary`, `description`, `args`, `returns`, `raises`, `complexity`).
    *   The logic is similar in both versions.

9.  **`_parse_message_content` Method:**
    *   `[[response_parsing updated]]` has a more concise `_parse_message_content` method that directly parses the JSON content, validates it against the schema (if `validate_schema` is `True`), ensures required fields, and creates a `ParsedResponse`.
    *   `[[response_parsing]]` has a more complex `_parse_message_content` method that includes more checks and error handling.

10. **`parse_response` Method:**
    *   Both versions have a `parse_response` method that serves as the main entry point for parsing.
    *   `[[response_parsing updated]]` has a more streamlined approach, first validating the response structure and then parsing the content.
    *   `[[response_parsing]]` has a similar structure but relies on the `_extract_content` method for content extraction.

**Summary of Improvements in `[[response_parsing updated]]`:**

*   **Simplified Structure:** The `ParsedResponse` dataclass is simplified by removing the `markdown` field.
*   **More Direct Parsing:** The parsing logic is more direct, with content parsing and validation happening within the `_parse_message_content` method.
*   **Streamlined Error Handling:** The use of `_create_error_response` and a single `_format_fallback_response` method makes error handling more consistent.
*   **Improved Validation:** The validation process is more integrated into the parsing flow.
*   **Better Readability:** The code is generally more readable and easier to follow.
*   **More Robust:** The updated version includes more checks for invalid or empty responses.

In essence, `[[response_parsing updated]]` represents a refactored version of `[[response_parsing]]` with a focus on simplification, improved error handling, and more direct parsing logic. The updated version is more robust and easier to maintain.

[Timestamp: 2024/12/19 14:08:55]

**user**: Modify response_parsing updated to work with [ai_service](./ai_service.md) [prompt_manager](./prompt_manager.md) and [base](./base.md) [docstring.py](docstring.py.md)
[Timestamp: 2024/12/19 14:09:36]

**ai**: Okay, I'll modify `response_parsing updated` to work with the provided `ai_service`, `prompt_manager`, and `base` and `docstring.py` files. This will involve adjusting imports, using the defined data structures, and ensuring compatibility with the overall architecture.

Here's the modified `response_parsing` code:

```python
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import jsonschema
import os

from core.logger import LoggerSetup  # Assuming you have a logger setup
from core.types.base import ParsedResponse
from core.types.docstring import DocstringData
from dataclasses import dataclass, asdict


class ResponseParsingService:
    """
    Unified service for formatting, validating, and parsing AI responses.
    """

    class DocstringSchema(TypedDict):
        summary: str
        description: str
        args: List[Dict[str, Any]]
        returns: Dict[str, str]
        raises: List[Dict[str, str]]
        complexity: int
        metadata: Dict[str, Any]
        error: str
        error_type: str

    _FALLBACK_SCHEMA: DocstringSchema = {
        "summary": "No summary available",
        "description": "No description available",
        "args": [],
        "returns": {"type": "Any", "description": "No return description provided"},
        "raises": [],
        "complexity": 1,
        "metadata": {},
        "error": "",
        "error_type": "none",
    }

    def __init__(
        self, correlation_id: Optional[str] = None, schema_dir: Optional[str] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

        # Flexible Schema Directory Handling
        if schema_dir:
            self.schema_dir = Path(schema_dir)
        else:
            default_schema_dir = Path(__file__).resolve().parent.parent / "schemas"
            self.schema_dir = Path(os.environ.get("SCHEMA_DIR", default_schema_dir))

        if not self.schema_dir.exists():
            self.logger.error(f"Schema directory does not exist: {self.schema_dir}")
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")

        # Load Schemas
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")

        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        
        self.schema_usage_metrics = {
            "function": 0,
            "docstring": 0,
            "fallback": 0,
        }

        self.logger.info("ResponseParsingService initialized")

    def _validate_schema(
        self, instance: Dict[str, Any], schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate an instance against a schema."""
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, []
        except jsonschema.ValidationError as e:
            self.logger.error(
                f"Schema validation failed: {str(e)}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return False, [str(e)]
        except Exception as e:
            self.logger.error(
                f"Unexpected error during schema validation: {str(e)}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            return False, [f"Unexpected validation error: {str(e)}"]

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a schema by name from the schema directory."""
        schema_path = self.schema_dir / schema_name
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error(
                f"Schema file not found: {schema_name} - {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON schema: {schema_name} - {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise

    def _format_fallback_response(
        self,
        metadata: Dict[str, Any],
        error: str = "",
        error_type: str = "format_error",
        format_type: str = "docstring",
    ) -> Dict[str, Any]:
        """
        Create a standardized fallback response structure.
        """
        self.logger.warning(
            f"{error_type}: {error}. Creating fallback response.",
            extra={"metadata": metadata, "correlation_id": self.correlation_id},
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        fallback_content: ResponseParsingService.DocstringSchema = {
            **self._FALLBACK_SCHEMA,
            "summary": f"Invalid {format_type} format",
            "description": f"The {format_type} response did not match the expected structure.",
            "error": error,
            "error_type": error_type,
            "metadata": {
                **metadata,
                "timestamp": timestamp,
                "correlation_id": self.correlation_id,
            },
        }

        fallback_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": fallback_usage,
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )

        return fallback_response

    def _select_schema(self, content: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Dynamically select the appropriate schema based on the content.
        """
        if "parameters" in content or "examples" in content:
            self.logger.info("Selected schema: function_tools_schema (priority: parameters/examples)")
            self.schema_usage_metrics["function"] += 1
            return self.function_schema, "function"

        if "summary" in content and "description" in content:
            self.logger.info("Selected schema: docstring_schema (priority: summary/description)")
            self.schema_usage_metrics["docstring"] += 1
            return self.docstring_schema, "docstring"

        self.logger.warning("No matching schema found. Using fallback schema.")
        self.schema_usage_metrics["fallback"] += 1
        return self._FALLBACK_SCHEMA, "fallback"

    def _validate_content(self, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate content against the dynamically selected schema.
        """
        schema, schema_type = self._select_schema(content)

        if schema_type == "fallback":
            self.logger.warning("Using fallback schema. Skipping validation.")
            return True, []

        self.logger.info(f"Validating content against schema type: {schema_type}")
        return self._validate_schema(content, schema)

    def _ensure_required_fields(
        self, content: Any
    ) -> Dict[str, Any]:
        """Ensure required fields exist in docstring-like content."""
        if isinstance(content, str):
            return {
                "summary": "Content as string",
                "description": content.strip(),
                "args": [],
                "returns": {"type": "Any", "description": ""},
                "raises": [],
                "complexity": 1,
            }

        result = dict(content)
        defaults = {
            "summary": "No summary provided. Ensure this field is populated.",
            "description": "No description provided.",
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1,
        }

        for key, default in defaults.items():
            if key not in result or not result[key]:
                self.logger.debug(
                    f"Setting default value for field: '{key}'",
                    extra={"correlation_id": self.correlation_id},
                )
                result[key] = default

        return result

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_type: str = "format_error",
    ) -> ParsedResponse:
        """Creates a ParsedResponse indicating an error."""
        if metadata is None:
            metadata = {}
        return ParsedResponse(
            content=self._format_fallback_response(
                metadata,
                error=error,
                error_type=error_type,
                format_type=expected_format,
            ),
            format_type=expected_format,
            parsing_time=time.time() - start_time if start_time else 0.0,
            validation_success=False,
            errors=[error],
            metadata=metadata,
        )

    def _validate_response_structure(
        self,
        response: Optional[Dict[str, Any]],
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """
        Basic structure validation of the AI response.
        """
        if response is None:
            self.logger.error("Response is None.")
            return self._create_error_response(
                "Response is None",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if isinstance(response, str) and not response.strip():
            self.logger.error("Empty response received from AI service.")
            return self._create_error_response(
                "Empty response from AI service",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not isinstance(response, dict):
            self.logger.error(f"Unexpected response type: {type(response)}")
            return self._create_error_response(
                f"Unexpected response type: {type(response)}",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "choices" not in response or not response["choices"]:
            return self._create_error_response(
                "No choices in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not response["choices"][0].get("message"):
            return self._create_error_response(
                "No message in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "content" not in response["choices"][0]["message"]:
            return self._create_error_response(
                "No content field in message",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        return ParsedResponse(
            content=response["choices"][0]["message"]["content"],
            format_type=expected_format,
            parsing_time=time.time() - start_time,
            validation_success=True,
            errors=[],
            metadata=metadata,
        )

    def _create_response(
        self,
        content: Dict[str, Any],
        format_type: str,
        start_time: float,
        success: bool,
        errors: List[str],
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Create a standardized ParsedResponse object."""
        return ParsedResponse(
            content=content,
            format_type=format_type,
            parsing_time=time.time() - start_time,
            validation_success=success,
            errors=errors,
            metadata=metadata,
        )

    async def _parse_message_content(
        self,
        content: str,
        expected_format: str,
        validate_schema: bool,
        start_time: float, 
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Parse and validate message content from AI response."""
        try:
            if not content or not content.strip():
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=["Empty response content"],
                    metadata=metadata,
                )

            try:
                parsed_content = json.loads(content)
                if not isinstance(parsed_content, dict):
                    return self._create_response(
                        content={},
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=["Response content must be a JSON object"],
                        metadata=metadata,
                    )
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in message content: {e}")
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=[f"Invalid JSON: {e}"],
                    metadata=metadata,
                )

            if validate_schema:
                is_valid, validation_errors = self._validate_content(parsed_content)
                if not is_valid:
                    return self._create_response(
                        content=parsed_content,
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=validation_errors,
                        metadata=metadata,
                    )

            parsed_content = self._ensure_required_fields(parsed_content)

            return self._create_response(
                content=parsed_content,
                format_type=expected_format,
                start_time=start_time,
                success=True,
                errors=[],
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Unexpected error parsing message content: {e}", exc_info=True)
            return self._create_response(
                content={},
                format_type=expected_format,
                start_time=start_time,
                success=False,
                errors=[f"Unexpected error: {e}"],
                metadata=metadata,
            )

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedResponse:
        """Parses and validates an AI response."""
        start_time = time.time()
        if metadata is None:
            metadata = {}

        try:
            validated_response = self._validate_response_structure(
                response, expected_format, start_time, metadata
            )
            if validated_response.errors:
                return validated_response

            content = response["choices"][0]["message"].get("content", "")
            
            return await self._parse_message_content(
                content=content,
                expected_format=expected_format,
                validate_schema=validate_schema,
                start_time=start_time,
                metadata=metadata,
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON received from AI response: {e}")
            return self._create_error_response(
                f"Invalid JSON: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}", exc_info=True)
            return self._create_error_response(
                f"Unexpected error during parsing: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
```

**Key Changes Made:**

1.  **Import `ParsedResponse` and `DocstringData`:**
    *   Replaced the local `ParsedResponse` dataclass with the one from `core.types.base`.
    *   Imported `DocstringData` from `core.types.docstring`.

2.  **Removed `markdown` field:**
    *   Removed the `markdown` field from the `ParsedResponse` object, aligning with the `base.py` definition.

3.  **No other changes:**
    *   The core logic of the parsing service remains the same, as it was already well-structured.

**How this integrates with the other components:**

*   **`ai_service`:** The `AIService` will call `response_parser.parse_response` and will receive a `ParsedResponse` object, which it will then use to construct a `ProcessingResult`.
*   **`prompt_manager`:** The `PromptManager` is used to create the prompts, and the `response_parsing` service will process the responses.
*   **`base`:** The `ParsedResponse` dataclass is defined in `base.py`, ensuring consistency across the application.
*   **`docstring.py`:** The `DocstringData` dataclass is used to represent the parsed docstring data, which is part of the response content.

**To use this modified `response_parsing` code:**

1.  Replace the existing `response_parsing` code with this modified version.
2.  Ensure that `core.types.base` and `core.types.docstring` are accessible in your project.
3.  Ensure that the `core.logger` module is correctly configured.

This updated code should now work seamlessly with the other provided components, ensuring a consistent and well-integrated system.

[Timestamp: 2024/12/19 14:19:45]

**user**: integrate this further with [logger](./logger.md)
[Timestamp: 2024/12/19 14:24:29]

**ai**: Okay, I'll integrate the `logger` into the `response_parsing updated` code. This primarily involves using `CorrelationLoggerAdapter` and ensuring that log messages include the correlation ID.

Here's the modified `response_parsing` code:

```python
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import jsonschema
import os

from core.logger import LoggerSetup, CorrelationLoggerAdapter  # Import the logger
from core.types.base import ParsedResponse
from core.types.docstring import DocstringData
from dataclasses import dataclass, asdict


class ResponseParsingService:
    """
    Unified service for formatting, validating, and parsing AI responses.
    """

    class DocstringSchema(TypedDict):
        summary: str
        description: str
        args: List[Dict[str, Any]]
        returns: Dict[str, str]
        raises: List[Dict[str, str]]
        complexity: int
        metadata: Dict[str, Any]
        error: str
        error_type: str

    _FALLBACK_SCHEMA: DocstringSchema = {
        "summary": "No summary available",
        "description": "No description available",
        "args": [],
        "returns": {"type": "Any", "description": "No return description provided"},
        "raises": [],
        "complexity": 1,
        "metadata": {},
        "error": "",
        "error_type": "none",
    }

    def __init__(
        self, correlation_id: Optional[str] = None, schema_dir: Optional[str] = None
    ) -> None:
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(
                f"{__name__}.{self.__class__.__name__}",
            ),
            extra={"correlation_id": correlation_id},
        )
        self.correlation_id = correlation_id

        # Flexible Schema Directory Handling
        if schema_dir:
            self.schema_dir = Path(schema_dir)
        else:
            default_schema_dir = Path(__file__).resolve().parent.parent / "schemas"
            self.schema_dir = Path(os.environ.get("SCHEMA_DIR", default_schema_dir))

        if not self.schema_dir.exists():
            self.logger.error(f"Schema directory does not exist: {self.schema_dir}")
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")

        # Load Schemas
        self.docstring_schema = self._load_schema("docstring_schema.json")
        self.function_schema = self._load_schema("function_tools_schema.json")

        self._parsing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "validation_failures": 0,
        }
        
        self.schema_usage_metrics = {
            "function": 0,
            "docstring": 0,
            "fallback": 0,
        }

        self.logger.info("ResponseParsingService initialized")

    def _validate_schema(
        self, instance: Dict[str, Any], schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate an instance against a schema."""
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, []
        except jsonschema.ValidationError as e:
            self.logger.error(
                f"Schema validation failed: {str(e)}",
                exc_info=True,
            )
            return False, [str(e)]
        except Exception as e:
            self.logger.error(
                f"Unexpected error during schema validation: {str(e)}",
                exc_info=True,
            )
            return False, [f"Unexpected validation error: {str(e)}"]

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a schema by name from the schema directory."""
        schema_path = self.schema_dir / schema_name
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error(
                f"Schema file not found: {schema_name} - {e}",
                exc_info=True,
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON schema: {schema_name} - {e}",
                exc_info=True,
            )
            raise

    def _format_fallback_response(
        self,
        metadata: Dict[str, Any],
        error: str = "",
        error_type: str = "format_error",
        format_type: str = "docstring",
    ) -> Dict[str, Any]:
        """
        Create a standardized fallback response structure.
        """
        self.logger.warning(
            f"{error_type}: {error}. Creating fallback response.",
            extra={"metadata": metadata},
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        fallback_content: ResponseParsingService.DocstringSchema = {
            **self._FALLBACK_SCHEMA,
            "summary": f"Invalid {format_type} format",
            "description": f"The {format_type} response did not match the expected structure.",
            "error": error,
            "error_type": error_type,
            "metadata": {
                **metadata,
                "timestamp": timestamp,
                "correlation_id": self.correlation_id,
            },
        }

        fallback_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": fallback_usage,
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
        )

        return fallback_response

    def _select_schema(self, content: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Dynamically select the appropriate schema based on the content.
        """
        if "parameters" in content or "examples" in content:
            self.logger.info("Selected schema: function_tools_schema (priority: parameters/examples)")
            self.schema_usage_metrics["function"] += 1
            return self.function_schema, "function"

        if "summary" in content and "description" in content:
            self.logger.info("Selected schema: docstring_schema (priority: summary/description)")
            self.schema_usage_metrics["docstring"] += 1
            return self.docstring_schema, "docstring"

        self.logger.warning("No matching schema found. Using fallback schema.")
        self.schema_usage_metrics["fallback"] += 1
        return self._FALLBACK_SCHEMA, "fallback"

    def _validate_content(self, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate content against the dynamically selected schema.
        """
        schema, schema_type = self._select_schema(content)

        if schema_type == "fallback":
            self.logger.warning("Using fallback schema. Skipping validation.")
            return True, []

        self.logger.info(f"Validating content against schema type: {schema_type}")
        return self._validate_schema(content, schema)

    def _ensure_required_fields(
        self, content: Any
    ) -> Dict[str, Any]:
        """Ensure required fields exist in docstring-like content."""
        if isinstance(content, str):
            return {
                "summary": "Content as string",
                "description": content.strip(),
                "args": [],
                "returns": {"type": "Any", "description": ""},
                "raises": [],
                "complexity": 1,
            }

        result = dict(content)
        defaults = {
            "summary": "No summary provided. Ensure this field is populated.",
            "description": "No description provided.",
            "args": [],
            "returns": {"type": "Any", "description": ""},
            "raises": [],
            "complexity": 1,
        }

        for key, default in defaults.items():
            if key not in result or not result[key]:
                self.logger.debug(
                    f"Setting default value for field: '{key}'",
                )
                result[key] = default

        return result

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_type: str = "format_error",
    ) -> ParsedResponse:
        """Creates a ParsedResponse indicating an error."""
        if metadata is None:
            metadata = {}
        return ParsedResponse(
            content=self._format_fallback_response(
                metadata,
                error=error,
                error_type=error_type,
                format_type=expected_format,
            ),
            format_type=expected_format,
            parsing_time=time.time() - start_time if start_time else 0.0,
            validation_success=False,
            errors=[error],
            metadata=metadata,
        )

    def _validate_response_structure(
        self,
        response: Optional[Dict[str, Any]],
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """
        Basic structure validation of the AI response.
        """
        if response is None:
            self.logger.error("Response is None.")
            return self._create_error_response(
                "Response is None",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if isinstance(response, str) and not response.strip():
            self.logger.error("Empty response received from AI service.")
            return self._create_error_response(
                "Empty response from AI service",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not isinstance(response, dict):
            self.logger.error(f"Unexpected response type: {type(response)}")
            return self._create_error_response(
                f"Unexpected response type: {type(response)}",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "choices" not in response or not response["choices"]:
            return self._create_error_response(
                "No choices in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if not response["choices"][0].get("message"):
            return self._create_error_response(
                "No message in response",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        if "content" not in response["choices"][0]["message"]:
            return self._create_error_response(
                "No content field in message",
                expected_format,
                start_time,
                metadata,
                error_type="validation_error",
            )

        return ParsedResponse(
            content=response["choices"][0]["message"]["content"],
            format_type=expected_format,
            parsing_time=time.time() - start_time,
            validation_success=True,
            errors=[],
            metadata=metadata,
        )

    def _create_response(
        self,
        content: Dict[str, Any],
        format_type: str,
        start_time: float,
        success: bool,
        errors: List[str],
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Create a standardized ParsedResponse object."""
        return ParsedResponse(
            content=content,
            format_type=format_type,
            parsing_time=time.time() - start_time,
            validation_success=success,
            errors=errors,
            metadata=metadata,
        )

    async def _parse_message_content(
        self,
        content: str,
        expected_format: str,
        validate_schema: bool,
        start_time: float, 
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """Parse and validate message content from AI response."""
        try:
            if not content or not content.strip():
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=["Empty response content"],
                    metadata=metadata,
                )

            try:
                parsed_content = json.loads(content)
                if not isinstance(parsed_content, dict):
                    return self._create_response(
                        content={},
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=["Response content must be a JSON object"],
                        metadata=metadata,
                    )
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in message content: {e}")
                return self._create_response(
                    content={},
                    format_type=expected_format,
                    start_time=start_time,
                    success=False,
                    errors=[f"Invalid JSON: {e}"],
                    metadata=metadata,
                )

            if validate_schema:
                is_valid, validation_errors = self._validate_content(parsed_content)
                if not is_valid:
                    return self._create_response(
                        content=parsed_content,
                        format_type=expected_format,
                        start_time=start_time,
                        success=False,
                        errors=validation_errors,
                        metadata=metadata,
                    )

            parsed_content = self._ensure_required_fields(parsed_content)

            return self._create_response(
                content=parsed_content,
                format_type=expected_format,
                start_time=start_time,
                success=True,
                errors=[],
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Unexpected error parsing message content: {e}", exc_info=True)
            return self._create_response(
                content={},
                format_type=expected_format,
                start_time=start_time,
                success=False,
                errors=[f"Unexpected error: {e}"],
                metadata=metadata,
            )

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedResponse:
        """Parses and validates an AI response."""
        start_time = time.time()
        if metadata is None:
            metadata = {}

        try:
            validated_response = self._validate_response_structure(
                response, expected_format, start_time, metadata
            )
            if validated_response.errors:
                return validated_response

            content = response["choices"][0]["message"].get("content", "")
            
            return await self._parse_message_content(
                content=content,
                expected_format=expected_format,
                validate_schema=validate_schema,
                start_time=start_time,
                metadata=metadata,
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON received from AI response: {e}")
            return self._create_error_response(
                f"Invalid JSON: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}", exc_info=True)
            return self._create_error_response(
                f"Unexpected error during parsing: {e}",
                expected_format,
                start_time,
                metadata,
                error_type="parse_error",
            )
```

**Key changes:**

1.  **Import `CorrelationLoggerAdapter`:**
    *   Imported `CorrelationLoggerAdapter` from `core.logger`.

2.  **Use `CorrelationLoggerAdapter`:**
    *   The logger is now initialized as a `CorrelationLoggerAdapter` in the `__init__` method, passing the correlation ID:

    ```python
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(
                f"{__name__}.{self.__class__.__name__}",
            ),
            extra={"correlation_id": correlation_id},
        )
    ```

3.  **Removed `extra` from log messages:**
    *   Removed the `extra={"correlation_id": self.correlation_id}` from the log messages, as the `CorrelationLoggerAdapter` now handles it.

With these changes, the `response_parsing` service is now fully integrated with the logging system, ensuring that all log messages include the correlation ID and are formatted consistently.

[Timestamp: 2024/12/19 14:28:37]