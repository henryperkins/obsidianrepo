```python
"""
Enhanced Logging Configuration and Utilities.
Provides structured, contextual, and robust logging across the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
import uuid
from collections.abc import MutableMapping
import re

class SanitizedLogFormatter(logging.Formatter):
    """Custom formatter to sanitize and format log records."""

    def format(self, record):
        # Ensure 'correlation_id' and 'extra' fields are present with default values
        record.correlation_id = getattr(record, 'correlation_id', "N/A")
        record.extra = getattr(record, 'extra', {"info": "[Sanitized]"})

        # Sanitize the message
        record.msg = self.sanitize_message(record.msg)

        # Sanitize arguments if present
        if record.args:
            record.args = self.sanitize_args(record.args)

        # Format the timestamp in ISO8601 format
        if self.usesTime():
            record.asctime = datetime.fromtimestamp(record.created).isoformat() + 'Z'

        # Now format the message using the parent class
        return super().format(record)

    def sanitize_message(self, message: str) -> str:
        """Sanitize sensitive information from the log message."""
        # Implement sanitation logic as needed
        # For example, replace file paths or sensitive patterns
        sanitized_message = message
        # Replace any sensitive data patterns using regex
        # Example: sanitized_message = re.sub(r'secret=\w+', 'secret=[REDACTED]', sanitized_message)
        sanitized_message = re.sub(r'(/[a-zA-Z0-9_\-./]+)', '[Sanitized_Path]', sanitized_message)
        sanitized_message = re.sub(r'secret_key=[^&\s]+', 'secret_key=[REDACTED]', sanitized_message)
        return sanitized_message

    def sanitize_args(self, args: Any) -> Any:
        """Sanitize sensitive information from the log arguments."""
        # Implement sanitation logic for arguments
        # For simplicity, we'll just return the args unmodified here
        # You can add sanitization logic similar to sanitize_message
        return args


class LoggerSetup:
    """Configures and manages application logging."""

    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Path = Path("logs")
    _file_logging_enabled: bool = True
    _configured: bool = False
    _default_level: int = logging.INFO
    _default_format: str = "%(levelname)s: %(message)s"

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Get a configured logger instance."""
        if not cls._configured:
            cls.configure()

        if name is None:
            name = __name__

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        if not logger.hasHandlers():
            logger.setLevel(cls._default_level)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(cls._default_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            if cls._file_logging_enabled:
                try:
                    cls._log_dir.mkdir(parents=True, exist_ok=True)
                    file_handler = RotatingFileHandler(
                        cls._log_dir / f"{name}.log",
                        maxBytes=1024 * 1024,
                        backupCount=5
                    )
                    sanitized_formatter = SanitizedLogFormatter(
                        fmt='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
                            '"message": "%(message)s", "correlation_id": "%(correlation_id)s", '
                            '"extra": "%(extra)s"}',
                        datefmt='%Y-%m-%dT%H:%M:%S'
                    )
                    file_handler.setFormatter(sanitized_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    # Log the exception using the console handler
                    logger.error(f"Failed to set up file handler: {e}")

        cls._loggers[name] = logger
        return logger

    @classmethod
    def configure(cls, level: str = "INFO", format_str: Optional[str] = None,
                  log_dir: Optional[str] = None, file_logging_enabled: bool = True) -> None:
        """Configure global logging settings."""
        if cls._configured:
            return

        cls._default_level = getattr(logging, level.upper(), logging.INFO)
        cls._default_format = format_str or cls._default_format
        cls._file_logging_enabled = file_logging_enabled

        if log_dir:
            cls._log_dir = Path(log_dir)

        cls._configured = True

    @classmethod
    def shutdown(cls) -> None:
        """Cleanup logging handlers and close files."""
        for logger in cls._loggers.values():
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        logging.shutdown()

    @classmethod
    def handle_exception(cls, exc_type: type, exc_value: BaseException, exc_traceback: Any) -> None:
        """Global exception handler."""
        if not issubclass(exc_type, KeyboardInterrupt):
            logger = cls.get_logger("global")
            logger.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
                extra={'correlation_id': 'N/A', 'extra': {}}
            )
        # Call the default excepthook if needed
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter to add a correlation ID to logs."""

    def __init__(self, logger: logging.Logger, correlation_id: Optional[str] = None):
        super().__init__(logger, {})
        self.correlation_id = correlation_id if correlation_id is not None else "N/A"

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        # Avoid mutating the original kwargs
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = self.correlation_id
        kwargs['extra'] = extra
        return msg, kwargs


# Module-level utility functions (optional)
def log_error(msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
    """Log an error message at module level."""
    logger = LoggerSetup.get_logger()
    logger.error(msg, *args, exc_info=exc_info, **kwargs)

def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message at module level."""
    logger = LoggerSetup.get_logger()
    logger.debug(msg, *args, **kwargs)

def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message at module level."""
    logger = LoggerSetup.get_logger()
    logger.info(msg, *args, **kwargs)

def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message at module level."""
    logger = LoggerSetup.get_logger()
    logger.warning(msg, *args, **kwargs)

__all__ = [
    'LoggerSetup',
    'CorrelationLoggerAdapter',
    'log_error',
    'log_debug',
    'log_info',
    'log_warning'
]

# Optional: Set up the root logger if needed
# LoggerSetup.configure()

# Optionally, set the global exception handler
# sys.excepthook = LoggerSetup.handle_exception
```


**Setup**

1. **Import Logger Utilities**: Bring in necessary classes and functions from `logger.py` into your module.

2. **Configure the Logger (Optional)**: Set up the logger using `LoggerSetup.configure()` in your main application file with desired settings like log level, format, and file logging options. This ensures consistency across modules.

3. **Get a Logger Instance**: Use `LoggerSetup.get_logger(__name__)` to obtain a logger specific to your module.

4. **Use the Logger**: Employ standard logging methods (`debug`, `info`, `warning`, `error`, `critical`) to log messages at appropriate levels.

5. **Use Correlation IDs (Optional)**: Utilize `CorrelationLoggerAdapter` to include a correlation ID in logs, aiding tracking across modules/services.

6. **Log Exceptions**: Employ `logger.exception()` to log exceptions with stack traces.

7. **Use Module-Level Utility Functions (Optional)**: Use utility functions like `log_info` and `log_error` which handle logger instances internally.

8. **Handle Uncaught Exceptions (Optional)**: Set `sys.excepthook` to `LoggerSetup.handle_exception` in your application entry point to log unhandled exceptions.

**Key Points**:
- Configure the logger once in the main application file.
- Get module-specific loggers with `__name__`.
- Use correlation IDs where tracing is needed.
- Log exceptions with detailed stack traces.
- Set a global exception handler for uncaught exceptions.

**Additional Tips**:
- Avoid multiple logger configurations.
- Consider data sanitization for sensitive information.
- Choose appropriate log levels based on environment needs.
- Check log outputs for correct data inclusion and formatting.

Update these files to use the logger.py - your response must include the complete, logger.py compatible module.

```python
"""
This module provides classes and functions for handling AI interactions, processing source code,
generating dynamic prompts, and integrating AI-generated documentation back into the source code.

Classes:
    CustomJSONEncoder: Custom JSON encoder that can handle sets and other non-serializable types.
    AIInteractionHandler: Handles AI interactions for generating enriched prompts and managing responses.

Functions:
    serialize_for_logging(obj: Any) -> str: Safely serialize any object for logging purposes.
"""

import ast
import asyncio
import json
import re
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI

from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.metrics import Metrics
from core.response_parsing import ParsedResponse, ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext, ExtractionResult
from exceptions import ConfigurationError, ProcessingError

from logger import LoggerSetup, log_error, log_debug, log_info, log_warning

# Set up the logger
logger = LoggerSetup.get_logger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
        """Override the default JSON encoding.

        Args:
            obj (Any): The object to encode.

        Returns:
            Any: The JSON-encoded version of the object or a string representation.
        """
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, types.ModuleType)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return super().default(obj)


def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging purposes.

    Args:
        obj (Any): The object to serialize.

    Returns:
        str: A JSON string representation of the object.
    """
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"


class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Optional[Cache] = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: Dict[str, Any] = docstring_schema or load_schema()

        try:
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            log_info("AI client initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize AI client: {e}")
            raise

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response (Union[str, Dict[str, Any]]): The response to truncate (either string or dictionary).
            length (int): Maximum length of the truncated response.

        Returns:
            str: Truncated string representation of the response.
        """
        try:
            if isinstance(response, dict):
                json_str = json.dumps(response, indent=2)
                return (json_str[:length] + "...") if len(json_str) > length else json_str
            elif isinstance(response, str):
                return (response[:length] + "...") if len(response) > length else response
            else:
                str_response = str(response)
                return (str_response[:length] + "...") if len(str_response) > length else str_response
        except Exception as e:
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Process the source code to extract metadata, interact with the AI, and integrate responses.

        Args:
            source_code (str): The source code to process.

        Returns:
            Optional[Dict[str, Any]]: The updated code and documentation if successful, otherwise None.
        """
        try:
            tree: ast.AST = ast.parse(source_code)
            context = ExtractionContext()
            context.source_code = source_code
            context.tree = tree

            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                match = re.search(r"Module:?\s*([^\n\.]+)", module_docstring)
                if match:
                    context.module_name = match.group(1).strip()

            extractor = CodeExtractor(context)
            extraction_result: Optional[ExtractionResult] = await extractor.extract_code(source_code)
            if not extraction_result:
                log_error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            log_debug("Generated prompt for AI")

            ai_response: Union[str, Dict[str, Any]] = await self._interact_with_ai(prompt)
            log_debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                log_error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                log_error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            log_error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            log_error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.

        Args:
            ai_response (Dict[str, Any]): The AI-generated response to integrate.
            extraction_result (ExtractionResult): The extraction result containing the source code.

        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            log_debug("Starting AI response integration")
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = self._create_processed_response(ai_response)

            integration_result = self._process_docstrings(processed_response, extraction_result.source_code)
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            if not isinstance(code, str):
                raise ProcessingError("Expected 'code' to be a string in integration result")

            documentation = self._generate_markdown_documentation(ai_response, extraction_result)
            return code, documentation

        except Exception as e:
            log_error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """
        Generate markdown documentation from AI response and extraction result.

        Args:
            ai_response (Dict[str, Any]): The AI response data.
            extraction_result (ExtractionResult): The extraction result containing source code info.

        Returns:
            str: Generated markdown documentation.
        """
        markdown_gen = MarkdownGenerator()
        markdown_context: Dict[str, Any] = {
            "module_name": extraction_result.module_name,
            "file_path": extraction_result.file_path,
            "description": ai_response.get("description", ""),
            "classes": extraction_result.classes,
            "functions": extraction_result.functions,
            "constants": extraction_result.constants,
            "source_code": extraction_result.source_code,
            "ai_documentation": ai_response,
        }
        return markdown_gen.generate(markdown_context)

    def _ensure_required_fields(self, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the AI response has the required fields, and repair if necessary.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            Dict[str, Any]: The AI response with ensured fields.
        """
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            log_error(f"AI response missing required fields: {missing}")

            for field in missing:
                if field == "args":
                    ai_response["args"] = []
                elif field == "returns":
                    ai_response["returns"] = {"type": "None", "description": ""}
                elif field == "raises":
                    ai_response["raises"] = []
                else:
                    ai_response[field] = ""
        return ai_response

    def _create_processed_response(self, ai_response: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Create a list with the processed response.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            List[Dict[str, Union[str, Dict[str, Any]]]]: Processed response for integration.
        """
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """
        Process the docstrings using DocstringProcessor.

        Args:
            processed_response (List[Dict[str, Union[str, Dict[str, Any]]]]): The processed response data.
            source_code (str): The source code to modify.

        Returns:
            Dict[str, Any]: The integration result with updated code and docstrings.
        """
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        log_debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> Union[str, Dict[str, Any]]:
        """
        Interact with the AI model to generate responses.

        Args:
            prompt (str): The prompt to send to the AI model.

        Returns:
            Union[str, Dict[str, Any]]: The AI-generated response.
        """
        try:
            request_params: Dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            log_debug("Sending request to AI")

            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[
                    {"role": "system", "content": "You are a Python documentation expert. Generate complete docstrings in Google format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=request_params.get('temperature', 0.7)
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            log_debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                log_debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse AI response as JSON: {e}")

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        log_debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        log_error(f"Failed to parse extracted JSON: {e2}")

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            log_error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            log_error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise Exception(f"Unexpected error processing request: {str(e)}")

    async def create_dynamic_prompt(self, extracted_info: Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> str:
        """
        Create a dynamic prompt for the AI model.

        Args:
            extracted_info (Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]): The extracted information for the prompt.

        Returns:
            str: The generated prompt.
        """
        try:
            log_debug("Creating dynamic prompt")
            log_debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

            prompt_parts: List[str] = [
                "Generate a complete Python documentation structure as a single JSON object.\n\n",
                "Required JSON structure:\n",
                "{\n",
                '  "summary": "One-line summary of the code",\n',
                '  "description": "Detailed description of functionality",\n',
                '  "args": [{"name": "param1", "type": "str", "description": "param description"}],\n',
                '  "returns": {"type": "ReturnType", "description": "return description"},\n',
                '  "raises": [{"exception": "ValueError", "description": "error description"}]\n',
                "}\n\n",
                "Code Analysis:\n"
            ]

            if extracted_info.get("module_docstring"):
                prompt_parts.append(f"Current Module Documentation:\n{extracted_info['module_docstring']}\n\n")

            if extracted_info.get("classes"):
                prompt_parts.append("Classes:\n")
                for cls in extracted_info["classes"]:
                    prompt_parts.append(f"- {cls['name']}\n")
                    if cls.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {cls['docstring']}\n")
                    if cls.get("methods"):
                        prompt_parts.append("  Methods:\n")
                        for method in cls["methods"]:
                            prompt_parts.append(f"    - {method['name']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("functions"):
                prompt_parts.append("Functions:\n")
                for func in extracted_info["functions"]:
                    prompt_parts.append(f"- {func['name']}\n")
                    if func.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {func['docstring']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("dependencies"):
                prompt_parts.append("Dependencies:\n")
                for dep_type, deps in extracted_info["dependencies"].items():
                    if dep_type == "maintainability_impact":
                        # Handle maintainability_impact separately
                        prompt_parts.append(f"- {dep_type}: {deps}\n")
                        continue

                    if not isinstance(deps, (list, set, tuple)):
                        log_error(f"Non-iterable dependencies for {dep_type}: {deps} ({type(deps)})")
                        continue

                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            log_debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            log_error(f"Error creating prompt: {e}", exc_info=True)
            raise Exception(f"Error creating dynamic prompt: {str(e)}")

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name (str): Name of the function or class.
            is_class (bool): Whether the target is a class.
            params (Optional[List[Dict[str, Any]]]): List of parameters with their types and descriptions.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring to enhance.
            decorators (Optional[List[str]]): List of decorators applied to the function.
            exceptions (Optional[List[Dict[str, str]]]): List of exceptions raised by the function.

        Returns:
            Dict[str, Any]: Generated docstring content.
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info: Dict[str, Any] = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                stop=["END"],
            )
            response_content: Optional[str] = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            log_info(f"Generated docstring for {func_name}")
            return parsed_response.content

        except json.JSONDecodeError as e:
            log_error(f"JSON decoding error while generating docstring for {func_name}: {e}")
            raise
        except ProcessingError as e:
            log_error(f"Processing error while generating docstring for {func_name}: {e}")
            raise
        except Exception as e:
            log_error(f"Unexpected error generating docstring for {func_name}: {e}")
            raise

    async def _verify_deployment(self) -> bool:
        """
        Verify that the configured deployment exists and is accessible.

        Returns:
            bool: True if the deployment exists and is accessible, otherwise False.
        """
        try:
            test_params: Dict[str, Any] = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }
            log_debug(f"Verifying deployment with parameters: {test_params}")
            response = await self.client.chat.completions.create(**test_params)
            log_debug(f"Deployment verification response: {response}")
            return True
        except Exception as e:
            log_error(f"Deployment verification failed: {e}", exc_info=True)
            return False

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Verifies the deployment configuration and raises a ConfigurationError if the deployment is not accessible.

        Returns:
            AIInteractionHandler: Initialized AIInteractionHandler instance.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.
        """
        log_debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            log_debug("Cache resources have been cleaned up")
        log_info("AIInteractionHandler resources have been cleaned up")
```



```python
"""
This module provides classes and functions for handling AI interactions, processing source code,
generating dynamic prompts, and integrating AI-generated documentation back into the source code.

Classes:
    CustomJSONEncoder: Custom JSON encoder that can handle sets and other non-serializable types.
    AIInteractionHandler: Handles AI interactions for generating enriched prompts and managing responses.

Functions:
    serialize_for_logging(obj: Any) -> str: Safely serialize any object for logging purposes.
"""

import ast
import asyncio
import json
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI

from api.api_client import APIClient  # Assumed to be a custom module
from api.token_management import TokenManager  # Assumed to be a custom module
from core.cache import Cache  # Assumed to be a custom module
from core.config import AzureOpenAIConfig  # Assumed to be a custom module
from core.docstring_processor import DocstringProcessor  # Assumed to be a custom module
from core.extraction.code_extractor import CodeExtractor  # Assumed to be a custom module
from core.markdown_generator import MarkdownGenerator  # Assumed to be a custom module
from core.metrics import Metrics  # Assumed to be a custom module
from core.response_parsing import ParsedResponse, ResponseParsingService  # Assumed to be a custom module
from core.schema_loader import load_schema  # Assumed to be a custom module
from core.types import ExtractionContext, ExtractionResult  # Assumed to be a custom module
from exceptions import ConfigurationError, ProcessingError  # Assumed to be a custom module

from logger import LoggerSetup  # Import the logger utilities

# Set up the logger
logger = LoggerSetup.get_logger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
        """Override the default JSON encoding.

        Args:
            obj (Any): The object to encode.

        Returns:
            Any: The JSON-encoded version of the object or a string representation.
        """
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, types.ModuleType)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return super().default(obj)


def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging purposes.

    Args:
        obj (Any): The object to serialize.

    Returns:
        str: A JSON string representation of the object.
    """
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"


class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Optional[Cache] = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: Dict[str, Any] = docstring_schema or load_schema()

        try:
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            self.logger.info("AI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}", exc_info=True)
            raise

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response (Union[str, Dict[str, Any]]): The response to truncate (either string or dictionary).
            length (int): Maximum length of the truncated response.

        Returns:
            str: Truncated string representation of the response.
        """
        try:
            if isinstance(response, dict):
                json_str = json.dumps(response, indent=2)
                return (json_str[:length] + "...") if len(json_str) > length else json_str
            elif isinstance(response, str):
                return (response[:length] + "...") if len(response) > length else response
            else:
                str_response = str(response)
                return (str_response[:length] + "...") if len(str_response) > length else str_response
        except Exception as e:
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Process the source code to extract metadata, interact with the AI, and integrate responses.

        Args:
            source_code (str): The source code to process.

        Returns:
            Optional[Dict[str, Any]]: The updated code and documentation if successful, otherwise None.
        """
        try:
            tree: ast.AST = ast.parse(source_code)
            context = ExtractionContext()
            context.source_code = source_code
            context.tree = tree

            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                match = re.search(r"Module:?\s*([^\n\.]+)", module_docstring)
                if match:
                    context.module_name = match.group(1).strip()

            extractor = CodeExtractor(context)
            extraction_result: Optional[ExtractionResult] = await extractor.extract_code(source_code)
            if not extraction_result:
                self.logger.error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            self.logger.debug("Generated prompt for AI")

            ai_response: Union[str, Dict[str, Any]] = await self._interact_with_ai(prompt)
            self.logger.debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                self.logger.error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                self.logger.error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            self.logger.error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.

        Args:
            ai_response (Dict[str, Any]): The AI-generated response to integrate.
            extraction_result (ExtractionResult): The extraction result containing the source code.

        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            self.logger.debug("Starting AI response integration")
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = self._create_processed_response(ai_response)

            integration_result = self._process_docstrings(processed_response, extraction_result.source_code)
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            if not isinstance(code, str):
                raise ProcessingError("Expected 'code' to be a string in integration result")

            documentation = self._generate_markdown_documentation(ai_response, extraction_result)
            return code, documentation

        except Exception as e:
            self.logger.error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """
        Generate markdown documentation from AI response and extraction result.

        Args:
            ai_response (Dict[str, Any]): The AI response data.
            extraction_result (ExtractionResult): The extraction result containing source code info.

        Returns:
            str: Generated markdown documentation.
        """
        markdown_gen = MarkdownGenerator()
        markdown_context: Dict[str, Any] = {
            "module_name": extraction_result.module_name,
            "file_path": extraction_result.file_path,
            "description": ai_response.get("description", ""),
            "classes": extraction_result.classes,
            "functions": extraction_result.functions,
            "constants": extraction_result.constants,
            "source_code": extraction_result.source_code,
            "ai_documentation": ai_response,
        }
        return markdown_gen.generate(markdown_context)

    def _ensure_required_fields(self, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the AI response has the required fields, and repair if necessary.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            Dict[str, Any]: The AI response with ensured fields.
        """
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            self.logger.error(f"AI response missing required fields: {missing}")

            for field in missing:
                if field == "args":
                    ai_response["args"] = []
                elif field == "returns":
                    ai_response["returns"] = {"type": "None", "description": ""}
                elif field == "raises":
                    ai_response["raises"] = []
                else:
                    ai_response[field] = ""
        return ai_response

    def _create_processed_response(self, ai_response: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Create a list with the processed response.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            List[Dict[str, Union[str, Dict[str, Any]]]]: Processed response for integration.
        """
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """
        Process the docstrings using DocstringProcessor.

        Args:
            processed_response (List[Dict[str, Union[str, Dict[str, Any]]]]): The processed response data.
            source_code (str): The source code to modify.

        Returns:
            Dict[str, Any]: The integration result with updated code and docstrings.
        """
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        self.logger.debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> Union[str, Dict[str, Any]]:
        """
        Interact with the AI model to generate responses.

        Args:
            prompt (str): The prompt to send to the AI model.

        Returns:
            Union[str, Dict[str, Any]]: The AI-generated response.
        """
        try:
            request_params: Dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            self.logger.debug("Sending request to AI")

            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[
                    {"role": "system", "content": "You are a Python documentation expert. Generate complete docstrings in Google format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=request_params.get('temperature', 0.7)
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            self.logger.debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                self.logger.debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse AI response as JSON: {e}", exc_info=True)

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        self.logger.debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        self.logger.error(f"Failed to parse extracted JSON: {e2}", exc_info=True)

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            self.logger.error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise Exception(f"Unexpected error processing request: {str(e)}")

    async def create_dynamic_prompt(self, extracted_info: Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> str:
        """
        Create a dynamic prompt for the AI model.

        Args:
            extracted_info (Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]): The extracted information for the prompt.

        Returns:
            str: The generated prompt.
        """
        try:
            self.logger.debug("Creating dynamic prompt")
            self.logger.debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

            prompt_parts: List[str] = [
                "Generate a complete Python documentation structure as a single JSON object.\n\n",
                "Required JSON structure:\n",
                "{\n",
                '  "summary": "One-line summary of the code",\n',
                '  "description": "Detailed description of functionality",\n',
                '  "args": [{"name": "param1", "type": "str", "description": "param description"}],\n',
                '  "returns": {"type": "ReturnType", "description": "return description"},\n',
                '  "raises": [{"exception": "ValueError", "description": "error description"}]\n',
                "}\n\n",
                "Code Analysis:\n"
            ]

            if extracted_info.get("module_docstring"):
                prompt_parts.append(f"Current Module Documentation:\n{extracted_info['module_docstring']}\n\n")

            if extracted_info.get("classes"):
                prompt_parts.append("Classes:\n")
                for cls in extracted_info["classes"]:
                    prompt_parts.append(f"- {cls['name']}\n")
                    if cls.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {cls['docstring']}\n")
                    if cls.get("methods"):
                        prompt_parts.append("  Methods:\n")
                        for method in cls["methods"]:
                            prompt_parts.append(f"    - {method['name']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("functions"):
                prompt_parts.append("Functions:\n")
                for func in extracted_info["functions"]:
                    prompt_parts.append(f"- {func['name']}\n")
                    if func.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {func['docstring']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("dependencies"):
                prompt_parts.append("Dependencies:\n")
                for dep_type, deps in extracted_info["dependencies"].items():
                    if dep_type == "maintainability_impact":
                        # Handle maintainability_impact separately
                        prompt_parts.append(f"- {dep_type}: {deps}\n")
                        continue

                    if not isinstance(deps, (list, set, tuple)):
                        self.logger.error(f"Non-iterable dependencies for {dep_type}: {deps} ({type(deps)})")
                        continue

                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            self.logger.debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            self.logger.error(f"Error creating prompt: {e}", exc_info=True)
            raise Exception(f"Error creating dynamic prompt: {str(e)}")

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name (str): Name of the function or class.
            is_class (bool): Whether the target is a class.
            params (Optional[List[Dict[str, Any]]]): List of parameters with their types and descriptions.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring to enhance.
            decorators (Optional[List[str]]): List of decorators applied to the function.
            exceptions (Optional[List[Dict[str, str]]]): List of exceptions raised by the function.

        Returns:
            Dict[str, Any]: Generated docstring content.
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info: Dict[str, Any] = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                stop=["END"],
            )
            response_content: Optional[str] = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            self.logger.info(f"Generated docstring for {func_name}")
            return parsed_response.content

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error while generating docstring for {func_name}: {e}", exc_info=True)
            raise
        except ProcessingError as e:
            self.logger.error(f"Processing error while generating docstring for {func_name}: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error generating docstring for {func_name}: {e}", exc_info=True)
            raise

    async def _verify_deployment(self) -> bool:
        """
        Verify that the configured deployment exists and is accessible.

        Returns:
            bool: True if the deployment exists and is accessible, otherwise False.
        """
        try:
            test_params: Dict[str, Any] = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }
            self.logger.debug(f"Verifying deployment with parameters: {test_params}")
            response = await self.client.chat.completions.create(**test_params)
            self.logger.debug(f"Deployment verification response: {response}")
            return True
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {e}", exc_info=True)
            return False

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Verifies the deployment configuration and raises a ConfigurationError if the deployment is not accessible.

        Returns:
            AIInteractionHandler: Initialized AIInteractionHandler instance.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.
        """
        self.logger.debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            self.logger.debug("Cache resources have been cleaned up")
        self.logger.info("AIInteractionHandler resources have been cleaned up")
```