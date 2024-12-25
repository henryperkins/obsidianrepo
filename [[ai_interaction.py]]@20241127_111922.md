---
epoch: 1732727962246
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [ai_service.py](ai_service.py.md)@20241127_111922

**user**: [ai_service.py](ai_service.py.md)  
[Timestamp: 2024/11/27 11:19:22]

**ai**: It seems like you've provided the content of a Python module `ai_interaction.py` and its corresponding documentation file `ai_interaction.py.md`. The module is designed to handle interactions with the Azure OpenAI API for generating documentation from Python source code. It includes features like token management, caching, and metrics collection.

If you need help with a specific part of this module, such as improving its functionality, fixing bugs, or adding new features, please let me know. Additionally, if you have any specific questions or need further explanation about certain parts of the code, feel free to ask!  
[Timestamp: 2024/11/27 11:19:25]

**user**: def **init**(self, cache: Optional[Cache] = None, metrics_collector: Optional[MetricsCollector] = None, token_manager: Optional[TokenManager] = None):  
    """Initialize the AI Interaction Handler."""  
    try:  
        self.cache = cache  
        self.metrics = metrics_collector  
        self.token_manager = token_manager or TokenManager(  
            model=config.model_name,  
            deployment_name=config.deployment_name  
        )

        self.client = AzureOpenAIClient(
            token_manager=self.token_manager,
            metrics_collector=metrics_collector
        )

        self.extraction_manager = ExtractionManager()
        self.response_parser = ResponseParser()
        self.docstring_validator = DocstringValidator()
        
        # Initialize function tools for structured output
        self._initialize_tools()

        logger.info("AI Interaction Handler initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
        raise

def _initialize_tools(self):  
    """Initialize the function tools for structured output."""  
    self.docstring_tool = {  
        "type": "function",  
        "function": {  
            "name": "generate_docstring",  
            "description": "Generate a Python docstring with structured information",  
            "parameters": {  
                "type": "object",  
                "properties": {  
                    "summary": {  
                        "type": "string",  
                        "description": "A concise summary of what the code does"  
                    },  
                    "description": {  
                        "type": "string",  
                        "description": "A detailed description of the functionality"  
                    },  
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
                        },  
                        "description": "List of arguments with their types and descriptions"  
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
                                "type": {"type": "string"},  
                                "description": {"type": "string"}  
                            }  
                        },  
                        "description": "List of exceptions that may be raised"  
                    }  
                },  
                "required": ["summary", "description", "args", "returns"]  
            }  
        }  
    }  
[Timestamp: 2024/11/27 11:19:27]

**ai**: The code snippet you've provided shows an enhancement to the `AIInteractionHandler` class by adding a method `_initialize_tools` that sets up a structured tool for generating Python docstrings. This tool is defined as a dictionary with a specific schema, which includes fields for summary, description, arguments, return types, and exceptions.

Here's a breakdown of the changes and their purpose:

1. **Initialization of Tools**: 
   - The `_initialize_tools` method is introduced to encapsulate the setup of tools used for generating structured outputs, such as docstrings.
   - This method is called within the `__init__` method to ensure that the tool is ready for use when an instance of `AIInteractionHandler` is created.

2. **Structured Docstring Tool**:
   - The `docstring_tool` dictionary defines a schema for creating structured docstrings. This schema includes:
     - `summary`: A brief overview of what the code does.
     - `description`: A detailed explanation of the code's functionality.
     - `args`: A list of arguments, each with a name, type, and description.
     - `returns`: Information about the return type and its description.
     - `raises`: A list of possible exceptions that the code might raise, with their descriptions.

3. **Purpose**:
   - This setup allows for a more organized and consistent way to generate docstrings, which can be particularly useful for automated documentation tools or when integrating with systems that require structured data.

If you have any specific questions about this implementation or need further assistance with integrating or using this tool, feel free to ask!  
[Timestamp: 2024/11/27 11:19:32]

**user**: def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:  
    """Create a dynamic prompt based on the code and metadata."""  
    prompt_parts = [  
        "Generate comprehensive documentation for the following Python code.",  
        "\nCode Analysis:",  
        f"- Complexity Metrics:\n{self._format_metrics(metadata)}",  
    ]

    if metadata.get('docstring'):
        prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

    if metadata.get('class_info'):
        prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

    prompt_parts.extend([
        "\nRequirements:",
        "- Follow Google Style Python docstring format",
        "- Include comprehensive parameter descriptions",
        "- Document return values and types",
        "- List possible exceptions",
        "- Add usage examples for complex functionality",
        
        f"\nSource Code:\n{source_code}",
        
        "\nProvide the documentation in a structured format using the specified function schema."
    ])

    return "\n".join(prompt_parts)

def _format_class_info(self, class_info: Dict[str, Any]) -> str:  
    """Format class information for the prompt."""  
    info_parts = [  
        f"Class Name: {class_info.get('name', 'Unknown')}",  
        f"Base Classes: {', '.join(class_info.get('bases', []))}",  
        f"Methods: {len(class_info.get('methods', []))}",  
    ]  
    return "\n".join(info_parts)  
[Timestamp: 2024/11/27 11:19:40]

**ai**: The code snippet you've provided enhances the `_create_documentation_prompt` method to generate a more detailed and structured prompt for documentation generation. This prompt is designed to be used with an AI model to produce comprehensive documentation for Python code. Here's a detailed breakdown of the changes and their purpose:

## Key Features of the Enhanced Prompt Creation

1. **Dynamic Prompt Construction**:
   - The method constructs a prompt by assembling various parts based on the provided source code and metadata.
   - This dynamic approach allows the prompt to be tailored to the specific characteristics of the code being documented.

2. **Inclusion of Code Analysis and Metadata**:
   - **Complexity Metrics**: The prompt includes a section for complexity metrics, formatted using the `_format_metrics` method. This helps the AI understand the complexity of the code.
   - **Existing Documentation**: If available, existing docstrings are included in the prompt, providing context that can be refined or expanded upon.
   - **Class Information**: The `_format_class_info` method is used to format details about classes, such as their names, base classes, and the number of methods.

3. **Documentation Requirements**:
   - The prompt specifies that the documentation should follow the Google Style Python docstring format, which is a widely used convention.
   - It emphasizes the need for detailed parameter descriptions, return types, exceptions, and usage examples, ensuring that the generated documentation is thorough and useful.

4. **Structured Output**:
   - The prompt ends with a request for the documentation to be provided in a structured format, adhering to a predefined function schema. This ensures consistency and makes it easier to integrate the generated documentation into existing systems.

## `_format_class_info` Method

- This helper method formats class-related metadata into a string that can be included in the prompt. It extracts and formats:
  - The class name.
  - The base classes the class inherits from.
  - The number of methods defined in the class.

## Usage

This approach is particularly useful for generating documentation in environments where code complexity and structure vary significantly. By providing detailed metadata and requirements, the AI model can generate more accurate and contextually relevant documentation.

If you have any further questions or need assistance with implementing or modifying this functionality, feel free to ask!  
[Timestamp: 2024/11/27 11:19:51]

**user**: async def _generate_documentation(  
    self,  
    source_code: str,  
    metadata: Dict[str, Any]  
) -> Optional[ProcessingResult]:  
    """Generate documentation using Azure OpenAI with function calling."""  
    try:  
        prompt = self._create_documentation_prompt(source_code, metadata)

        response, usage = await self.client.process_request(
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            tools=[self.docstring_tool],
            tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
        )

        if not response:
            return None

        # Handle function call response
        if response.get("tool_calls"):
            tool_call = response["tool_calls"][0]
            if tool_call.function.name == "generate_docstring":
                try:
                    docstring_data = json.loads(tool_call.function.arguments)
                    
                    # Validate the structured output
                    is_valid, errors = self.docstring_validator.validate_docstring(docstring_data)
                    if not is_valid:
                        logger.error(f"Docstring validation failed: {errors}")
                        raise ProcessingError(f"Invalid documentation: {errors}")

                    # Generate markdown documentation
                    markdown_generator = MarkdownDocumentationGenerator(
                        source_code,
                        module_path=metadata.get('module_path')
                    )
                    full_documentation = markdown_generator.generate_markdown(docstring_data)

                    return ProcessingResult(
                        content=full_documentation,
                        usage=usage or {},
                        processing_time=(datetime.now() - datetime.now()).total_seconds()
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse function call response: {e}")
                    raise ProcessingError("Invalid JSON in function response")

        raise ProcessingError("No valid function call response received")

    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        raise ProcessingError(f"Failed to generate documentation: {str(e)}")
[Timestamp: 2024/11/27 11:19:55]

**ai**: In the codebase you've described, the validation of generated docstrings is primarily handled by the `DocstringValidator` from the `docstring_utils.py` module. Here's how it fits into the process:

## Validation Process

1. **Docstring Generation**:
   - The `AIInteractionHandler` uses the Azure OpenAI API to generate documentation for the provided source code. This is done through the `_generate_documentation` method.

2. **Response Handling**:
   - After receiving a response from the API, the system checks if the response includes a function call with the name `generate_docstring`. This indicates that the response contains structured docstring data.

3. **Docstring Validation**:
   - The structured docstring data is then validated using the `DocstringValidator`. This component ensures that the generated docstring adheres to the expected format and contains all necessary sections.

   ```python
   # Validate the structured output
   is_valid, errors = self.docstring_validator.validate_docstring(docstring_data)
   if not is_valid:
       logger.error(f"Docstring validation failed: {errors}")
       raise ProcessingError(f"Invalid documentation: {errors}")
   ```

## Role of `ResponseParser`

- The `ResponseParser` is responsible for parsing the raw output from the AI model. It ensures that the response is correctly formatted and extracts the relevant parts needed for further processing.
- While it plays a crucial role in preparing the data for validation, the actual validation logic is handled by the `DocstringValidator`.

## Role of `DocstringValidator`

- The `DocstringValidator` checks the structure and content of the docstring to ensure it meets predefined criteria. This might include checking for required fields, ensuring correct formatting, and validating content consistency.

## Conclusion

In summary, while the `ResponseParser` prepares the data by parsing the AI's output, the `DocstringValidator` is the component that performs the actual validation of the docstring's structure and content. This separation of concerns allows each component to focus on its specific task, ensuring a robust and maintainable system.

If you need more details on how these components interact or wish to see specific implementations, feel free to ask!  
[Timestamp: 2024/11/27 11:20:02]

**user**: def _format_docstring_from_json(self, docstring_data: Dict[str, Any]) -> str:  
    """Format the JSON docstring data into Google Style format."""  
    docstring_parts = []
    
    # Add summary and description
    docstring_parts.append(docstring_data["summary"])
    if docstring_data.get("description"):
        docstring_parts.extend(["", docstring_data["description"]])

    # Add arguments
    if docstring_data.get("args"):
        docstring_parts.append("\nArgs:")
        for arg in docstring_data["args"]:
            docstring_parts.append(f"    {arg['name']} ({arg['type']}): {arg['description']}")

    # Add returns section
    if docstring_data.get("returns"):
        docstring_parts.append("\nReturns:")
        docstring_parts.append(f"    {docstring_data['returns']['type']}: {docstring_data['returns']['description']}")

    # Add raises section
    if docstring_data.get("raises"):
        docstring_parts.append("\nRaises:")
        for error in docstring_data["raises"]:
            docstring_parts.append(f"    {error['type']}: {error['description']}")

    return "\n".join(docstring_parts)
[Timestamp: 2024/11/27 11:20:36]

**ai**: The `_format_docstring_from_json` method is designed to convert structured JSON data into a Google Style Python docstring. This method takes a dictionary containing docstring components and formats it into a human-readable string that follows the Google Style guide. Here's a detailed explanation of how it works:

## Method Breakdown

1. **Initialization**:
   - A list `docstring_parts` is initialized to collect different parts of the docstring.

2. **Summary and Description**:
   - The method starts by appending the `summary` from the `docstring_data` to `docstring_parts`.
   - If a `description` is provided, it is added after a blank line for separation.

3. **Arguments Section**:
   - If there are any `args` in the `docstring_data`, the method adds an "Args:" section.
   - Each argument is formatted with its name, type, and description, following the Google Style format:

     ```plaintext
     Args:
         arg_name (arg_type): Description of the argument.
     ```

4. **Returns Section**:
   - If there is a `returns` entry, a "Returns:" section is added.
   - The return type and its description are formatted as follows:

     ```plaintext
     Returns:
         return_type: Description of the return value.
     ```

5. **Raises Section**:
   - If there are any `raises` entries, a "Raises:" section is included.
   - Each exception is listed with its type and description:

     ```plaintext
     Raises:
         ExceptionType: Description of the exception.
     ```

6. **Final Assembly**:
   - All parts collected in `docstring_parts` are joined with newline characters to form the final docstring.

## Example

Given a JSON input like this:

```json
{
  "summary": "Calculates the factorial of a number.",
  "description": "This function uses a recursive approach to calculate the factorial.",
  "args": [
    {"name": "n", "type": "int", "description": "The number to calculate the factorial for."}
  ],
  "returns": {"type": "int", "description": "The factorial of the number."},
  "raises": [
    {"type": "ValueError", "description": "If n is negative."}
  ]
}
```

The formatted docstring would be:

```plaintext
Calculates the factorial of a number.

This function uses a recursive approach to calculate the factorial.

Args:
    n (int): The number to calculate the factorial for.

Returns:
    int: The factorial of the number.

Raises:
    ValueError: If n is negative.
```

## Usage

This method is particularly useful when you have structured data (like JSON) that needs to be converted into a standard docstring format. It ensures consistency and readability, which are crucial for maintaining high-quality documentation.

If you have any further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:20:46]

**user**: async def process_code(  
    self,  
    source_code: str,  
    cache_key: Optional[str] = None  
) -> Tuple[str, str]:  
    """Process source code to generate and embed documentation."""  
    operation_start = datetime.now()

    try:
        # Validate input
        if not source_code or not source_code.strip():
            raise ValidationError("Empty source code provided")

        # Check cache if enabled
        if self.cache and cache_key:
            try:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    return cached_result
            except CacheError as e:
                logger.warning(f"Cache error, proceeding without cache: {str(e)}")

        # Extract metadata
        metadata = self._extract_metadata(source_code)

        # Generate documentation
        result = await self._generate_documentation(source_code, metadata)
        if not result or not result.content:
            raise ProcessingError("Documentation generation failed")

        # Update code with documentation
        updated_code = await self._update_code(source_code, result.content)
        if not updated_code:
            raise ProcessingError("Code update failed")

        # Cache result if enabled
        if self.cache and cache_key:
            await self._cache_result(cache_key, updated_code, result.content)

        # Track metrics
        await self._track_metrics(operation_start, True, result.usage)

        return updated_code, result.content

    except Exception as e:
        await self._track_error("ProcessingError", e, operation_start)
        logger.error(f"Process code failed: {str(e)}")
        raise
[Timestamp: 2024/11/27 11:20:54]

**ai**: The `process_code` method is an asynchronous function designed to process Python source code by generating and embedding documentation using the Azure OpenAI API. Here's a detailed breakdown of its functionality:

## Key Steps in `process_code`

1. **Initialization**:
   - The method begins by recording the start time of the operation to track processing duration.

2. **Input Validation**:
   - It checks if the `source_code` is non-empty and contains meaningful content. If not, it raises a `ValidationError`.

3. **Cache Check**:
   - If caching is enabled and a `cache_key` is provided, the method attempts to retrieve a cached result using `_check_cache`.
   - If a cache hit occurs, it returns the cached result immediately.
   - If a `CacheError` occurs, it logs a warning and proceeds without using the cache.

4. **Metadata Extraction**:
   - The method extracts metadata from the `source_code` using `_extract_metadata`. This metadata includes information about classes, functions, and complexity metrics.

5. **Documentation Generation**:
   - It calls `_generate_documentation` to generate documentation based on the source code and extracted metadata.
   - If the documentation generation fails (i.e., `result` is `None` or `result.content` is empty), it raises a `ProcessingError`.

6. **Code Update**:
   - The method updates the original source code with the generated documentation using `_update_code`.
   - If the code update fails, it raises a `ProcessingError`.

7. **Cache Result**:
   - If caching is enabled, it stores the updated code and documentation in the cache using `_cache_result`.

8. **Metrics Tracking**:
   - It tracks the operation's metrics, including success status and usage statistics, using `_track_metrics`.

9. **Error Handling**:
   - If any exception occurs during the process, it logs the error, tracks the error metrics using `_track_error`, and re-raises the exception as a `ProcessingError`.

## Return Value

- The method returns a tuple containing the `updated_code` (source code with embedded documentation) and the `result.content` (generated documentation).

## Usage

This method is central to the functionality of the `AIInteractionHandler`, as it orchestrates the entire process of generating and embedding documentation. It handles caching, error tracking, and metrics collection, ensuring a robust and efficient workflow.

## Considerations

- **Asynchronous Design**: The use of `async` and `await` allows for non-blocking operations, making it suitable for environments where multiple requests might be processed concurrently.
- **Error Management**: Comprehensive error handling ensures that issues are logged and tracked, providing insights into potential failures.

If you have any further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:21:03]

**user**: Take all the code snippets i just shared to you and update ai_interaction  
[Timestamp: 2024/11/27 11:21:26]

**ai**: To update the `ai_interaction.py` module with the code snippets you've shared, we'll integrate the new methods and enhancements into the existing structure of the module. Here's how you can incorporate these changes:

## Updated `ai_interaction.py`

```python
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json
from dataclasses import dataclass

from core.logger import LoggerSetup, log_error, log_info, log_debug
from core.cache import Cache
from core.monitoring import MetricsCollector
from config import AzureOpenAIConfig
from api.token_management import TokenManager
from api.api_client import AzureOpenAIClient
from exceptions import ValidationError, ProcessingError, CacheError
from extract.extraction_manager import ExtractionManager
from response_parser import ResponseParser
from markdown_generator import MarkdownDocumentationGenerator
from docstring_utils import DocstringValidator

# Initialize logger and config
logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """Handles AI interactions for documentation generation via Azure OpenAI API."""

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.cache = cache
            self.metrics = metrics_collector
            self.token_manager = token_manager or TokenManager(
                model=config.model_name,
                deployment_name=config.deployment_name
            )

            self.client = AzureOpenAIClient(
                token_manager=self.token_manager,
                metrics_collector=metrics_collector
            )

            self.extraction_manager = ExtractionManager()
            self.response_parser = ResponseParser()
            self.docstring_validator = DocstringValidator()
            
            # Initialize function tools for structured output
            self._initialize_tools()

            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self):
        """Initialize the function tools for structured output."""
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does"
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality"
                        },
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
                            },
                            "description": "List of arguments with their types and descriptions"
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
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            },
                            "description": "List of exceptions that may be raised"
                        }
                    },
                    "required": ["summary", "description", "args", "returns"]
                }
            }
        }

    def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create a dynamic prompt based on the code and metadata."""
        prompt_parts = [
            "Generate comprehensive documentation for the following Python code.",
            "\nCode Analysis:",
            f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        ]

        if metadata.get('docstring'):
            prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

        if metadata.get('class_info'):
            prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

        prompt_parts.extend([
            "\nRequirements:",
            "- Follow Google Style Python docstring format",
            "- Include comprehensive parameter descriptions",
            "- Document return values and types",
            "- List possible exceptions",
            "- Add usage examples for complex functionality",
            
            f"\nSource Code:\n{source_code}",
            
            "\nProvide the documentation in a structured format using the specified function schema."
        ])

        return "\n".join(prompt_parts)

    def _format_class_info(self, class_info: Dict[str, Any]) -> str:
        """Format class information for the prompt."""
        info_parts = [
            f"Class Name: {class_info.get('name', 'Unknown')}",
            f"Base Classes: {', '.join(class_info.get('bases', []))}",
            f"Methods: {len(class_info.get('methods', []))}",
        ]
        return "\n".join(info_parts)

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any]
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata)

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                return None

            # Handle function call response
            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.function.name == "generate_docstring":
                    try:
                        docstring_data = json.loads(tool_call.function.arguments)
                        
                        # Validate the structured output
                        is_valid, errors = self.docstring_validator.validate_docstring(docstring_data)
                        if not is_valid:
                            logger.error(f"Docstring validation failed: {errors}")
                            raise ProcessingError(f"Invalid documentation: {errors}")

                        # Generate markdown documentation
                        markdown_generator = MarkdownDocumentationGenerator(
                            source_code,
                            module_path=metadata.get('module_path')
                        )
                        full_documentation = markdown_generator.generate_markdown(docstring_data)

                        return ProcessingResult(
                            content=full_documentation,
                            usage=usage or {},
                            processing_time=(datetime.now() - datetime.now()).total_seconds()
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse function call response: {e}")
                        raise ProcessingError("Invalid JSON in function response")

            raise ProcessingError("No valid function call response received")

        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    def _format_docstring_from_json(self, docstring_data: Dict[str, Any]) -> str:
        """Format the JSON docstring data into Google Style format."""
        docstring_parts = []
        
        # Add summary and description
        docstring_parts.append(docstring_data["summary"])
        if docstring_data.get("description"):
            docstring_parts.extend(["", docstring_data["description"]])

        # Add arguments
        if docstring_data.get("args"):
            docstring_parts.append("\nArgs:")
            for arg in docstring_data["args"]:
                docstring_parts.append(f"    {arg['name']} ({arg['type']}): {arg['description']}")

        # Add returns section
        if docstring_data.get("returns"):
            docstring_parts.append("\nReturns:")
            docstring_parts.append(f"    {docstring_data['returns']['type']}: {docstring_data['returns']['description']}")

        # Add raises section
        if docstring_data.get("raises"):
            docstring_parts.append("\nRaises:")
            for error in docstring_data["raises"]:
                docstring_parts.append(f"    {error['type']}: {error['description']}")

        return "\n".join(docstring_parts)

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None
    ) -> Tuple[str, str]:
        """Process source code to generate and embed documentation."""
        operation_start = datetime.now()

        try:
            # Validate input
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Extract metadata
            metadata = self._extract_metadata(source_code)

            # Generate documentation
            result = await self._generate_documentation(source_code, metadata)
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = await self._update_code(source_code, result.content)
            if not updated_code:
                raise ProcessingError("Code update failed")

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            # Track metrics
            await self._track_metrics(operation_start, True, result.usage)

            return updated_code, result.content

        except Exception as e:
            await self._track_error("ProcessingError", e, operation_start)
            logger.error(f"Process code failed: {str(e)}")
            raise

    # Additional methods like _extract_metadata, _update_code, _cache_result, etc., would be here.
```

## Key Changes Integrated

1. **Initialization**: Added `_initialize_tools` to set up structured output tools.
2. **Dynamic Prompt Generation**: Updated `_create_documentation_prompt` to create prompts based on code and metadata.
3. **Documentation Generation**: Enhanced `_generate_documentation` to use function calling for generating structured outputs.
4. **Docstring Formatting**: Implemented `_format_docstring_from_json` to convert JSON data into Google Style docstrings.
5. **Process Code**: Updated `process_code` to incorporate caching, error handling, and metrics tracking.

This updated module now includes the functionality to dynamically generate prompts, validate structured outputs, and format docstrings according to the Google Style guide. If you have any further questions or need additional modifications, feel free to ask!  
[Timestamp: 2024/11/27 11:24:22]

**user**: Complete the whole module  
[Timestamp: 2024/11/27 11:24:34]

**ai**: Certainly! Below is the complete `ai_interaction.py` module, incorporating all the enhancements and features discussed, including dynamic prompt generation, structured output handling, and integration with caching and metrics.

```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json
from dataclasses import dataclass

from core.logger import LoggerSetup, log_error, log_info, log_debug
from core.cache import Cache
from core.monitoring import MetricsCollector
from config import AzureOpenAIConfig
from api.token_management import TokenManager
from api.api_client import AzureOpenAIClient
from exceptions import ValidationError, ProcessingError, CacheError
from extract.extraction_manager import ExtractionManager
from response_parser import ResponseParser
from markdown_generator import MarkdownDocumentationGenerator
from docstring_utils import DocstringValidator

# Initialize logger and config
logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """
    Handles AI interactions for documentation generation via Azure OpenAI API.

    Manages token limits, caching mechanisms, and metrics collection for robust processing.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.cache = cache
            self.metrics = metrics_collector
            self.token_manager = token_manager or TokenManager(
                model=config.model_name,
                deployment_name=config.deployment_name
            )

            self.client = AzureOpenAIClient(
                token_manager=self.token_manager,
                metrics_collector=metrics_collector
            )

            self.extraction_manager = ExtractionManager()
            self.response_parser = ResponseParser()
            self.docstring_validator = DocstringValidator()
            
            # Initialize function tools for structured output
            self._initialize_tools()

            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self):
        """Initialize the function tools for structured output."""
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does"
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality"
                        },
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
                            },
                            "description": "List of arguments with their types and descriptions"
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
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            },
                            "description": "List of exceptions that may be raised"
                        }
                    },
                    "required": ["summary", "description", "args", "returns"]
                }
            }
        }

    def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create a dynamic prompt based on the code and metadata."""
        prompt_parts = [
            "Generate comprehensive documentation for the following Python code.",
            "\nCode Analysis:",
            f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        ]

        if metadata.get('docstring'):
            prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

        if metadata.get('class_info'):
            prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

        prompt_parts.extend([
            "\nRequirements:",
            "- Follow Google Style Python docstring format",
            "- Include comprehensive parameter descriptions",
            "- Document return values and types",
            "- List possible exceptions",
            "- Add usage examples for complex functionality",
            
            f"\nSource Code:\n{source_code}",
            
            "\nProvide the documentation in a structured format using the specified function schema."
        ])

        return "\n".join(prompt_parts)

    def _format_class_info(self, class_info: Dict[str, Any]) -> str:
        """Format class information for the prompt."""
        info_parts = [
            f"Class Name: {class_info.get('name', 'Unknown')}",
            f"Base Classes: {', '.join(class_info.get('bases', []))}",
            f"Methods: {len(class_info.get('methods', []))}",
        ]
        return "\n".join(info_parts)

    def _format_metrics(self, metadata: Dict[str, Any]) -> str:
        """Format the complexity metrics into a string to include in the prompt."""
        lines = []
        for func in metadata.get('functions', []):
            metrics = func.get('metrics', {})
            lines.append(
                f"Function {func['name']} - Cyclomatic Complexity: {metrics.get('cyclomatic_complexity')}, "
                f"Cognitive Complexity: {metrics.get('cognitive_complexity')}, "
                f"Maintainability Index: {metrics.get('maintainability_index')}"
            )
        for cls in metadata.get('classes', []):
            lines.append(f"Class {cls['name']}:")
            for method in cls.get('methods', []):
                method_metrics = method.get('metrics', {})
                lines.append(
                    f"  Method {method['name']} - Cyclomatic Complexity: {method_metrics.get('cyclomatic_complexity')}, "
                    f"Cognitive Complexity: {method_metrics.get('cognitive_complexity')}, "
                    f"Maintainability Index: {method_metrics.get('maintainability_index')}"
                )
        return '\n'.join(lines)

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any]
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata)

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                return None

            # Handle function call response
            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.function.name == "generate_docstring":
                    try:
                        docstring_data = json.loads(tool_call.function.arguments)
                        
                        # Validate the structured output
                        is_valid, errors = self.docstring_validator.validate_docstring(docstring_data)
                        if not is_valid:
                            logger.error(f"Docstring validation failed: {errors}")
                            raise ProcessingError(f"Invalid documentation: {errors}")

                        # Generate markdown documentation
                        markdown_generator = MarkdownDocumentationGenerator(
                            source_code,
                            module_path=metadata.get('module_path')
                        )
                        full_documentation = markdown_generator.generate_markdown(docstring_data)

                        return ProcessingResult(
                            content=full_documentation,
                            usage=usage or {},
                            processing_time=(datetime.now() - datetime.now()).total_seconds()
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse function call response: {e}")
                        raise ProcessingError("Invalid JSON in function response")

            raise ProcessingError("No valid function call response received")

        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    def _format_docstring_from_json(self, docstring_data: Dict[str, Any]) -> str:
        """Format the JSON docstring data into Google Style format."""
        docstring_parts = []
        
        # Add summary and description
        docstring_parts.append(docstring_data["summary"])
        if docstring_data.get("description"):
            docstring_parts.extend(["", docstring_data["description"]])

        # Add arguments
        if docstring_data.get("args"):
            docstring_parts.append("\nArgs:")
            for arg in docstring_data["args"]:
                docstring_parts.append(f"    {arg['name']} ({arg['type']}): {arg['description']}")

        # Add returns section
        if docstring_data.get("returns"):
            docstring_parts.append("\nReturns:")
            docstring_parts.append(f"    {docstring_data['returns']['type']}: {docstring_data['returns']['description']}")

        # Add raises section
        if docstring_data.get("raises"):
            docstring_parts.append("\nRaises:")
            for error in docstring_data["raises"]:
                docstring_parts.append(f"    {error['type']}: {error['description']}")

        return "\n".join(docstring_parts)

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None
    ) -> Tuple[str, str]:
        """Process source code to generate and embed documentation."""
        operation_start = datetime.now()

        try:
            # Validate input
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Extract metadata
            metadata = self._extract_metadata(source_code)

            # Generate documentation
            result = await self._generate_documentation(source_code, metadata)
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = await self._update_code(source_code, result.content)
            if not updated_code:
                raise ProcessingError("Code update failed")

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            # Track metrics
            await self._track_metrics(operation_start, True, result.usage)

            return updated_code, result.content

        except Exception as e:
            await self._track_error("ProcessingError", e, operation_start)
            logger.error(f"Process code failed: {str(e)}")
            raise

    def _extract_metadata(self, source_code: str) -> Dict[str, Any]:
        """Extract metadata using ExtractionManager."""
        metadata = self.extraction_manager.extract_metadata(source_code)
        return metadata

    async def _update_code(self, source_code: str, documentation: str) -> Optional[str]:
        """Update source code with generated documentation."""
        try:
            # Insert the documentation at the top of the source code
            updated_code = f'"""\n{documentation}\n"""\n\n{source_code}'
            return updated_code
        except Exception as e:
            logger.error(f"Code update failed: {str(e)}")
            raise ProcessingError(f"Failed to update code: {str(e)}")

    async def _cache_result(self, cache_key: str, code: str, documentation: str) -> None:
        """Cache the processing result."""
        try:
            await self.cache.save_docstring(
                cache_key,
                {
                    'code': code,
                    'docs': documentation
                }
            )
            logger.debug(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.error(f"Caching failed: {str(e)}")
            # Don't raise - caching failure shouldn't stop processing

    async def _track_metrics(self, start_time: datetime, success: bool, usage: Dict[str, Any]) -> None:
        """Track operation metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=success,
                duration=duration,
                usage=usage
            )
        # Optionally log metrics
        log_info(f"Operation metrics: success={success}, duration={duration}, usage={usage}")

    async def _track_error(self, error_type: str, error: Exception, start_time: datetime) -> None:
        """Track and log error metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=False,
                duration=duration,
                error=f"{error_type}: {str(error)}"
            )

    async def _check_cache(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """Check if the result is cached."""
        try:
            cached_data = await self.cache.get_cached_docstring(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data['code'], cached_data['docs']
            else:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            raise CacheError(f"Cache check failed: {str(e)}")

    async def close(self) -> None:
        """Close the AI interaction handler and cleanup resources."""
        try:
            await self.client.close()
            logger.info("AI Interaction Handler closed successfully")
        except Exception as e:
            logger.error(f"Error closing AI handler: {str(e)}")

    async def __aenter__(self) -> 'AIInteractionHandler':
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Async context manager exit."""
        await self.close()
```

## Key Features

- **Initialization**: Sets up caching, metrics collection, and token management.
- **Dynamic Prompt Generation**: Constructs prompts based on code and metadata.
- **Structured Output Handling**: Uses function calling to generate structured docstrings.
- **Docstring Formatting**: Converts JSON data into Google Style docstrings.
- **Caching and Metrics**: Integrates caching and tracks performance metrics.
- **Error Handling**: Comprehensive error tracking and logging.

This module is designed to be robust and efficient, handling various aspects of documentation generation using Azure OpenAI. If you have any further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:25:53]

**user**: How to enable the dynamic prompt to receive information on nodes in other modules  
[Timestamp: 2024/11/27 11:33:49]

**ai**: To enable the dynamic prompt to receive information on nodes in other modules, you can enhance the metadata extraction process to include cross-module analysis. This involves analyzing the dependencies and relationships between modules to gather comprehensive information about the codebase. Here's how you can implement this:

## Steps to Enable Cross-Module Analysis

1. **Dependency Analysis**: Identify and analyze the dependencies between modules to understand how they interact with each other.
2. **AST Traversal**: Traverse the Abstract Syntax Tree (AST) of each module to extract relevant information about classes, functions, and their relationships.
3. **Metadata Aggregation**: Collect and aggregate metadata from multiple modules to provide a holistic view of the codebase.
4. **Dynamic Prompt Enhancement**: Modify the prompt generation logic to incorporate cross-module metadata, allowing the AI to generate more contextually aware documentation.

## Implementation Details

### 1. Dependency Analysis

You can use static analysis tools to identify module dependencies. For example, you can parse import statements to determine which modules are imported by others.

```python
def _extract_imports(self, source_code: str) -> List[str]:
    """Extract import statements from the source code."""
    tree = ast.parse(source_code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return imports
```

### 2. AST Traversal

Extend the `ExtractionManager` to traverse the AST of each module and extract metadata.

```python
def extract_cross_module_metadata(self, module_paths: List[str]) -> Dict[str, Any]:
    """Extract metadata across multiple modules."""
    metadata = {}
    for path in module_paths:
        with open(path, 'r') as file:
            source_code = file.read()
            module_metadata = self.extract_metadata(source_code)
            metadata[path] = module_metadata
    return metadata
```

### 3. Metadata Aggregation

Combine metadata from different modules to provide a comprehensive view.

```python
def aggregate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate metadata from multiple modules."""
    aggregated = {
        'classes': [],
        'functions': [],
        'imports': []
    }
    for module_data in metadata.values():
        aggregated['classes'].extend(module_data.get('classes', []))
        aggregated['functions'].extend(module_data.get('functions', []))
        aggregated['imports'].extend(module_data.get('imports', []))
    return aggregated
```

### 4. Dynamic Prompt Enhancement

Modify the prompt generation logic to include cross-module information.

```python
def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
    """Create a dynamic prompt based on the code and metadata."""
    prompt_parts = [
        "Generate comprehensive documentation for the following Python code.",
        "\nCode Analysis:",
        f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        f"- Imports: {', '.join(metadata.get('imports', []))}"
    ]

    if metadata.get('docstring'):
        prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

    if metadata.get('class_info'):
        prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

    prompt_parts.extend([
        "\nRequirements:",
        "- Follow Google Style Python docstring format",
        "- Include comprehensive parameter descriptions",
        "- Document return values and types",
        "- List possible exceptions",
        "- Add usage examples for complex functionality",
        
        f"\nSource Code:\n{source_code}",
        
        "\nProvide the documentation in a structured format using the specified function schema."
    ])

    return "\n".join(prompt_parts)
```

## Considerations

- **Performance**: Analyzing multiple modules can be resource-intensive. Consider optimizing the analysis process and using caching to improve performance.
- **Scalability**: Ensure that the system can handle large codebases by efficiently managing memory and processing resources.
- **Accuracy**: Validate the extracted metadata to ensure accuracy and completeness, especially when dealing with complex module interactions.

By implementing these steps, you can enhance the dynamic prompt to include cross-module information, enabling the AI to generate more comprehensive and contextually aware documentation. If you need further assistance or have specific questions about the implementation, feel free to ask!  
[Timestamp: 2024/11/27 11:34:03]

**user**: [ast_analysis.py](ast_analysis.py.md) [extraction_manager.py](extraction_manager.py.md)  
[Timestamp: 2024/11/27 11:34:16]

**ai**: To enable dynamic prompts to receive information on nodes in other modules, you can enhance the `ASTAnalyzer` and `ExtractionManager` classes to perform cross-module analysis. This involves analyzing dependencies and aggregating metadata across multiple modules. Here's how you can implement this:

## Enhancements to `ASTAnalyzer` and `ExtractionManager`

1. **Dependency Analysis**: Identify and analyze dependencies between modules to understand their interactions.
2. **Cross-Module Metadata Extraction**: Extend the existing methods to extract and aggregate metadata from multiple modules.
3. **Dynamic Prompt Enhancement**: Modify the prompt generation logic to incorporate cross-module metadata.

## Implementation

### `ASTAnalyzer` Enhancements

Add a method to extract import statements, which will help in identifying dependencies:

```python
class ASTAnalyzer:
    # Existing methods...

    @staticmethod
    def extract_imports(tree: ast.AST) -> List[str]:
        """
        Extract import statements from an AST.

        Args:
            tree (ast.AST): The AST from which to extract import statements.

        Returns:
            List[str]: A list of import module names.
        """
        logger.debug("Extracting import statements from AST.")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
```

### `ExtractionManager` Enhancements

Add methods to handle cross-module metadata extraction and aggregation:

```python
class ExtractionManager:
    # Existing methods...

    def extract_cross_module_metadata(self, module_paths: List[str]) -> Dict[str, Any]:
        """
        Extract metadata across multiple modules.

        Args:
            module_paths (List[str]): List of file paths to the modules.

        Returns:
            Dict[str, Any]: Aggregated metadata from all modules.
        """
        logger.debug("Extracting metadata across multiple modules.")
        aggregated_metadata = {
            'classes': [],
            'functions': [],
            'imports': []
        }
        for path in module_paths:
            with open(path, 'r') as file:
                source_code = file.read()
                module_metadata = self.extract_metadata(source_code)
                aggregated_metadata['classes'].extend(module_metadata.get('classes', []))
                aggregated_metadata['functions'].extend(module_metadata.get('functions', []))
                aggregated_metadata['imports'].extend(module_metadata.get('imports', []))
        return aggregated_metadata

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code, including information about classes and functions.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata categorized into 'classes', 'functions', and 'imports'.
        """
        logger.debug("Starting metadata extraction.")
        try:
            tree = self.analyzer.parse_source_code(source_code)
            self.analyzer.add_parent_info(tree)
            classes, functions = [], []

            for node in ast.walk(tree):
                logger.debug(f"Processing node type: {type(node).__name__}")
                if isinstance(node, ast.ClassDef):
                    try:
                        logger.debug(f"Processing class definition: {node.name}")
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            # Calculate metrics for the class
                            metrics = self.analyzer.calculate_class_metrics(node)
                            metadata.update({'metrics': metrics})
                            classes.append(metadata)
                        logger.debug(
                            f"Extracted {'exception ' if is_exception else ''}class: "
                            f"{node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting class metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )
                elif isinstance(node, ast.FunctionDef):
                    try:
                        logger.debug(f"Processing function definition: {node.name}")
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            # Calculate metrics for the function
                            metrics = self.analyzer.calculate_function_metrics(node)
                            metadata.update({'metrics': metrics})
                            functions.append(metadata)
                        logger.debug(
                            f"Extracted function: {node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting function metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )

            imports = self.analyzer.extract_imports(tree)

            logger.info(
                f"Extraction complete. Found {len(classes)} classes, "
                f"{len(functions)} functions, and {len(imports)} imports"
            )
            return {'classes': classes, 'functions': functions, 'imports': imports}

        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': [], 'imports': []}
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': [], 'imports': []}
```

## Dynamic Prompt Enhancement

Incorporate cross-module metadata into the prompt generation logic:

```python
def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
    """Create a dynamic prompt based on the code and metadata."""
    prompt_parts = [
        "Generate comprehensive documentation for the following Python code.",
        "\nCode Analysis:",
        f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        f"- Imports: {', '.join(metadata.get('imports', []))}"
    ]

    if metadata.get('docstring'):
        prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

    if metadata.get('class_info'):
        prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

    prompt_parts.extend([
        "\nRequirements:",
        "- Follow Google Style Python docstring format",
        "- Include comprehensive parameter descriptions",
        "- Document return values and types",
        "- List possible exceptions",
        "- Add usage examples for complex functionality",
        
        f"\nSource Code:\n{source_code}",
        
        "\nProvide the documentation in a structured format using the specified function schema."
    ])

    return "\n".join(prompt_parts)
```

## Considerations

- **Performance**: Analyzing multiple modules can be resource-intensive. Consider optimizing the analysis process and using caching to improve performance.
- **Scalability**: Ensure that the system can handle large codebases by efficiently managing memory and processing resources.
- **Accuracy**: Validate the extracted metadata to ensure accuracy and completeness, especially when dealing with complex module interactions.

By implementing these enhancements, you can enable the dynamic prompt to include cross-module information, allowing the AI to generate more comprehensive and contextually aware documentation. If you need further assistance or have specific questions about the implementation, feel free to ask!  
[Timestamp: 2024/11/27 11:34:56]

**user**: Produce the complete ast_analysis.py  
[Timestamp: 2024/11/27 11:35:16]

**ai**: Certainly! Below is the complete `ast_analysis.py` module, incorporating the functionality for analyzing Python Abstract Syntax Trees (ASTs). This module includes methods for parsing source code, extracting function and class definitions, handling docstrings and annotations, and calculating complexity metrics.

```python
"""AST Analysis Module - Provides utilities for analyzing Python AST nodes."""

import ast
from typing import List, Optional, Union, Dict
from core.logger import LoggerSetup
from core.metrics import Metrics  # Importing the Metrics class

logger = LoggerSetup.get_logger(__name__)

class ASTAnalyzer:
    """Provides utilities for analyzing Python AST nodes, including parsing,
    extracting classes and functions, and handling docstrings and annotations.
    """

    def __init__(self) -> None:
        """Initialize AST analyzer."""
        self.metrics_calculator = Metrics()

    def parse_source_code(self, source_code: str) -> ast.AST:
        """
        Parse source code into an Abstract Syntax Tree (AST).

        Args:
            source_code (str): Source code to parse.

        Returns:
            ast.AST: Parsed Abstract Syntax Tree.

        Raises:
            SyntaxError: If the source code contains syntax errors.
        """
        try:
            logger.debug("Parsing source code into AST.")
            return ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise

    @staticmethod
    def extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:
        """
        Extract function definitions from an AST.

        Args:
            tree (ast.AST): The AST from which to extract function definitions.

        Returns:
            List[ast.FunctionDef]: A list of function definition nodes.
        """
        logger.debug("Extracting function definitions from AST.")
        return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    @staticmethod
    def extract_classes(tree: ast.AST) -> List[ast.ClassDef]:
        """
        Extract class definitions from an AST.

        Args:
            tree (ast.AST): The AST from which to extract class definitions.

        Returns:
            List[ast.ClassDef]: A list of class definition nodes.
        """
        logger.debug("Extracting class definitions from AST.")
        return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    @staticmethod
    def extract_docstring(node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """
        Extract the docstring from an AST node.

        Args:
            node (Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]): Node from which to extract the docstring.

        Returns:
            Optional[str]: The extracted docstring, or None if no docstring is present.
        """
        try:
            logger.debug(f"Extracting docstring from node: {type(node).__name__}")
            return ast.get_docstring(node)
        except Exception as e:
            logger.error(f"Failed to extract docstring: {e}")
            return None

    @staticmethod
    def get_annotation(node: Optional[ast.AST]) -> str:
        """
        Get the annotation of an AST node.

        Args:
            node (Optional[ast.AST]): The AST node from which to get the annotation.

        Returns:
            str: The annotation as a string. Returns "Any" if no annotation is present.
        """
        if node is None:
            return "Any"
        try:
            logger.debug(f"Getting annotation for node: {ast.dump(node)}")
            return ast.unparse(node)
        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
            return "Any"

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """
        Add parent node information to each node in an AST.

        This function traverses the AST and adds a 'parent' attribute to each child node,
        pointing to its parent node.

        Args:
            tree (ast.AST): The AST to which parent information is to be added.
        """
        logger.debug("Adding parent information to AST nodes.")
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)
        logger.info("Parent information added to AST nodes.")

    def calculate_function_metrics(self, function_node: ast.FunctionDef) -> Dict[str, Union[int, float]]:
        """
        Calculate complexity metrics for a function.

        Args:
            function_node (ast.FunctionDef): AST node of the function.

        Returns:
            Dict[str, Union[int, float]]: Dictionary containing complexity metrics.
        """
        try:
            cyclomatic_complexity = self.metrics_calculator.calculate_cyclomatic_complexity(function_node)
            cognitive_complexity = self.metrics_calculator.calculate_cognitive_complexity(function_node)
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(function_node)
            halstead_metrics = self.metrics_calculator.calculate_halstead_metrics(function_node)

            metrics = {
                'cyclomatic_complexity': cyclomatic_complexity,
                'cognitive_complexity': cognitive_complexity,
                'maintainability_index': maintainability_index,
                'halstead_metrics': halstead_metrics
            }
            logger.debug(f"Calculated metrics for function '{function_node.name}': {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for function '{function_node.name}': {e}")
            return {}

    def calculate_class_metrics(self, class_node: ast.ClassDef) -> Dict[str, Union[int, float]]:
        """
        Calculate complexity metrics for a class.

        Args:
            class_node (ast.ClassDef): AST node of the class.

        Returns:
            Dict[str, Union[int, float]]: Dictionary containing aggregated complexity metrics for the class methods.
        """
        try:
            method_metrics = []
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    metrics = self.calculate_function_metrics(node)
                    method_metrics.append(metrics)

            # Aggregate metrics if needed, for example, average complexities
            # For simplicity, we'll just collect metrics per method
            metrics = {
                'methods_metrics': method_metrics
            }
            logger.debug(f"Calculated metrics for class '{class_node.name}': {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for class '{class_node.name}': {e}")
            return {}

    @staticmethod
    def extract_imports(tree: ast.AST) -> List[str]:
        """
        Extract import statements from an AST.

        Args:
            tree (ast.AST): The AST from which to extract import statements.

        Returns:
            List[str]: A list of import module names.
        """
        logger.debug("Extracting import statements from AST.")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
```

## Key Features

- **Parsing**: Converts source code into an AST using the `parse_source_code` method.
- **Extraction**: Provides methods to extract functions, classes, docstrings, and import statements from an AST.
- **Annotations**: Retrieves type annotations from AST nodes.
- **Parent Information**: Adds parent node information to each node in the AST.
- **Complexity Metrics**: Calculates complexity metrics for functions and classes, including cyclomatic complexity, cognitive complexity, maintainability index, and Halstead metrics.

This module is designed to facilitate the analysis of Python source code by providing a comprehensive set of utilities for working with ASTs. If you have any further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:35:44]

**user**: Generate the complete extraction_manager.py  
[Timestamp: 2024/11/27 11:35:49]

**ai**: Below is the complete `extraction_manager.py` module, which is designed to extract metadata from Python source code. This module focuses on extracting information about class and function definitions, handling exception classes, and providing robust error handling.

```python
import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup
from core.utils import handle_exceptions
from extract.ast_analysis import ASTAnalyzer  # Assuming ASTAnalyzer is in ast_analysis.py

logger = LoggerSetup.get_logger(__name__)

class ExtractionManager:
    """
    A manager for extracting metadata from Python source code, particularly focusing on class and function definitions.
    It provides special handling for exception classes and robust error handling.
    """

    def __init__(self) -> None:
        """Initialize the ExtractionManager with an AST analyzer."""
        logger.debug("Initializing ExtractionManager.")
        self.analyzer = ASTAnalyzer()

    @handle_exceptions(logger.error)
    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code, including information about classes and functions.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata categorized into 'classes', 'functions', and 'imports'.
        """
        logger.debug("Starting metadata extraction.")
        try:
            tree = self.analyzer.parse_source_code(source_code)
            self.analyzer.add_parent_info(tree)
            classes, functions = [], []

            for node in ast.walk(tree):
                logger.debug(f"Processing node type: {type(node).__name__}")
                if isinstance(node, ast.ClassDef):
                    try:
                        logger.debug(f"Processing class definition: {node.name}")
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            # Calculate metrics for the class
                            metrics = self.analyzer.calculate_class_metrics(node)
                            metadata.update({'metrics': metrics})
                            classes.append(metadata)
                        logger.debug(
                            f"Extracted {'exception ' if is_exception else ''}class: "
                            f"{node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting class metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )
                elif isinstance(node, ast.FunctionDef):
                    try:
                        logger.debug(f"Processing function definition: {node.name}")
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            # Calculate metrics for the function
                            metrics = self.analyzer.calculate_function_metrics(node)
                            metadata.update({'metrics': metrics})
                            functions.append(metadata)
                        logger.debug(
                            f"Extracted function: {node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting function metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )

            imports = self.analyzer.extract_imports(tree)

            logger.info(
                f"Extraction complete. Found {len(classes)} classes, "
                f"{len(functions)} functions, and {len(imports)} imports"
            )
            return {'classes': classes, 'functions': functions, 'imports': imports}

        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': [], 'imports': []}
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': [], 'imports': []}

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Determine if a class node represents an exception class.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class is an exception class, otherwise False.
        """
        logger.debug(f"Checking if class {node.name} is an exception class.")
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                logger.debug(f"Class {node.name} is an exception class.")
                return True
            elif isinstance(base, ast.Attribute) and base.attr in {'Exception', 'BaseException'}:
                logger.debug(f"Class {node.name} is an exception class.")
                return True
        logger.debug(f"Class {node.name} is not an exception class.")
        return False

    def _extract_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract metadata from a class definition node.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Indicates if this class is an exception class.

        Returns:
            Dict[str, Any]: A dictionary containing metadata of the class.
        """
        logger.debug(f"Extracting metadata for class: {node.name}")
        try:
            metadata = {
                'name': node.name,
                'docstring': self.analyzer.extract_docstring(node) or '',
                'lineno': node.lineno,
                'is_exception': is_exception,
                'type': 'exception_class' if is_exception else 'class'
            }

            additional_metadata = self._additional_class_metadata(node, is_exception)
            metadata.update(additional_metadata)
            logger.debug(f"Extracted metadata for class {node.name}: {metadata}")
            return metadata

        except (AttributeError, TypeError) as e:
            logger.error(f"Error in class metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'exception_class' if is_exception else 'class',
                'error': str(e)
            }

    def _additional_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract additional metadata specific to normal or exception classes.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Indicates if this class is an exception class.

        Returns:
            Dict[str, Any]: Additional metadata based on class type.
        """
        if not is_exception:
            return {
                'methods': self._extract_methods(node),
                'bases': [self.analyzer.get_annotation(base) for base in node.bases],
                'decorators': self._extract_decorators(node)
            }
        else:
            return {
                'bases': [self.analyzer.get_annotation(base) for base in node.bases],
                'error_code': self._extract_error_code(node)
            }

    def _extract_error_code(self, node: ast.ClassDef) -> Optional[str]:
        """
        Extracts an error code from an exception class if defined.

        Args:
            node (ast.ClassDef): The exception class node.

        Returns:
            Optional[str]: The error code if present, otherwise None.
        """
        logger.debug(f"Extracting error code for exception class: {node.name}")
        error_code = None
        try:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == 'code':
                            value = item.value
                            if isinstance(value, ast.Str):
                                error_code = value.s
                            elif isinstance(value, ast.Constant):
                                error_code = str(value.value)
                            logger.debug(f"Extracted error code for {node.name}: {error_code}")
                            return error_code
            logger.debug(f"No error code found for {node.name}")
            return error_code
        except (AttributeError, TypeError) as e:
            logger.error(f"Error extracting error code for {node.name}: {e}")
            return error_code

    def _extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract metadata for methods within a class definition.

        Args:
            node (ast.ClassDef): The class node to extract methods from.

        Returns:
            List[Dict[str, Any]]: A list containing metadata for each method.
        """
        logger.debug(f"Extracting methods for class: {node.name}")
        methods = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                method_metadata = {
                    'name': stmt.name,
                    'docstring': self.analyzer.extract_docstring(stmt) or '',
                    'lineno': stmt.lineno
                }
                methods.append(method_metadata)
                logger.debug(f"Extracted method: {method_metadata}")
        return methods

    def _extract_decorators(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> List[str]:
        """
        Extract decorators applied to a class or method.

        Args:
            node (Union[ast.ClassDef, ast.FunctionDef]): The node to extract decorators from.

        Returns:
            List[str]: A list of decorator names as strings.
        """
        logger.debug(f"Extracting decorators for {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        logger.debug(f"Extracted decorators: {decorators}")
        return decorators

    def _extract_function_metadata(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract metadata from a function definition node.

        Args:
            node (ast.FunctionDef): The function node to extract metadata from.

        Returns:
            Dict[str, Any]: Metadata for the function, or information about any errors encountered.
        """
        logger.debug(f"Extracting metadata for function: {node.name}")
        try:
            function_metadata = {
                'name': node.name,
                'docstring': self.analyzer.extract_docstring(node) or '',
                'args': [(arg.arg, self.analyzer.get_annotation(arg.annotation)) for arg in node.args.args],
                'return_type': self.analyzer.get_annotation(node.returns),
                'decorators': self._extract_decorators(node),
                'lineno': node.lineno
            }
            logger.debug(f"Extracted function metadata: {function_metadata}")
            return function_metadata
        except Exception as e:
            logger.error(f"Error in function metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'function',
                'error': str(e)
            }
```

## Key Features

- **Metadata Extraction**: Extracts metadata from Python source code, focusing on classes, functions, and imports.
- **Exception Handling**: Identifies and handles exception classes, extracting specific metadata like error codes.
- **Method and Function Analysis**: Extracts metadata for methods within classes and standalone functions, including docstrings, arguments, return types, and decorators.
- **Robust Error Handling**: Uses decorators and logging to handle and report errors during metadata extraction.

This module is designed to provide comprehensive metadata extraction capabilities for Python source code, facilitating further analysis and documentation generation. If you have any further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:37:07]

**user**: Can my script do cross module aware documentation now  
[Timestamp: 2024/11/27 11:37:24]

**ai**: To enable cross-module aware documentation in your script, you need to ensure that your system can analyze and understand the relationships and dependencies between different modules. The current setup with `ASTAnalyzer` and `ExtractionManager` provides a strong foundation for analyzing individual modules, but to achieve cross-module awareness, you should consider the following enhancements:

## Steps to Enable Cross-Module Awareness

1. **Dependency Analysis**: Implement functionality to parse and understand import statements across modules. This allows the system to track which modules are being used and how they are interconnected.
2. **Metadata Aggregation**: Extend the metadata extraction process to aggregate information from multiple modules, providing a comprehensive view of the entire codebase.
3. **Contextual Prompt Generation**: Modify the prompt generation logic to incorporate cross-module metadata, allowing the AI to generate documentation that considers the broader context of the codebase.

## Implementation Enhancements

### 1. Dependency Analysis

Enhance the `ASTAnalyzer` to extract import statements, which will help in identifying module dependencies:

```python
class ASTAnalyzer:
    # Existing methods...

    @staticmethod
    def extract_imports(tree: ast.AST) -> List[str]:
        """
        Extract import statements from an AST.

        Args:
            tree (ast.AST): The AST from which to extract import statements.

        Returns:
            List[str]: A list of import module names.
        """
        logger.debug("Extracting import statements from AST.")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
```

### 2. Metadata Aggregation

Modify the `ExtractionManager` to handle cross-module metadata extraction and aggregation:

```python
class ExtractionManager:
    # Existing methods...

    def extract_cross_module_metadata(self, module_paths: List[str]) -> Dict[str, Any]:
        """
        Extract metadata across multiple modules.

        Args:
            module_paths (List[str]): List of file paths to the modules.

        Returns:
            Dict[str, Any]: Aggregated metadata from all modules.
        """
        logger.debug("Extracting metadata across multiple modules.")
        aggregated_metadata = {
            'classes': [],
            'functions': [],
            'imports': []
        }
        for path in module_paths:
            with open(path, 'r') as file:
                source_code = file.read()
                module_metadata = self.extract_metadata(source_code)
                aggregated_metadata['classes'].extend(module_metadata.get('classes', []))
                aggregated_metadata['functions'].extend(module_metadata.get('functions', []))
                aggregated_metadata['imports'].extend(module_metadata.get('imports', []))
        return aggregated_metadata
```

### 3. Contextual Prompt Generation

Incorporate cross-module metadata into the prompt generation logic:

```python
def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
    """Create a dynamic prompt based on the code and metadata."""
    prompt_parts = [
        "Generate comprehensive documentation for the following Python code.",
        "\nCode Analysis:",
        f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
        f"- Imports: {', '.join(metadata.get('imports', []))}"
    ]

    if metadata.get('docstring'):
        prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

    if metadata.get('class_info'):
        prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

    prompt_parts.extend([
        "\nRequirements:",
        "- Follow Google Style Python docstring format",
        "- Include comprehensive parameter descriptions",
        "- Document return values and types",
        "- List possible exceptions",
        "- Add usage examples for complex functionality",
        
        f"\nSource Code:\n{source_code}",
        
        "\nProvide the documentation in a structured format using the specified function schema."
    ])

    return "\n".join(prompt_parts)
```

## Considerations

- **Performance**: Analyzing multiple modules can be resource-intensive. Consider optimizing the analysis process and using caching to improve performance.
- **Scalability**: Ensure that the system can handle large codebases by efficiently managing memory and processing resources.
- **Accuracy**: Validate the extracted metadata to ensure accuracy and completeness, especially when dealing with complex module interactions.

By implementing these enhancements, your script can achieve cross-module awareness, allowing it to generate more comprehensive and contextually relevant documentation. If you need further assistance or have specific questions about the implementation, feel free to ask!  
[Timestamp: 2024/11/27 11:37:53]

**user**: Update extraction_manager and ast_analysis  
[Timestamp: 2024/11/27 11:38:15]

**ai**: To update the `extraction_manager.py` and `ast_analysis.py` modules to support cross-module awareness, we'll implement enhancements for dependency analysis and metadata aggregation. Below are the updated versions of these modules:

## Updated `ast_analysis.py`

```python
"""AST Analysis Module - Provides utilities for analyzing Python AST nodes."""

import ast
from typing import List, Optional, Union, Dict
from core.logger import LoggerSetup
from core.metrics import Metrics  # Importing the Metrics class

logger = LoggerSetup.get_logger(__name__)

class ASTAnalyzer:
    """Provides utilities for analyzing Python AST nodes, including parsing,
    extracting classes and functions, and handling docstrings and annotations.
    """

    def __init__(self) -> None:
        """Initialize AST analyzer."""
        self.metrics_calculator = Metrics()

    def parse_source_code(self, source_code: str) -> ast.AST:
        """
        Parse source code into an Abstract Syntax Tree (AST).

        Args:
            source_code (str): Source code to parse.

        Returns:
            ast.AST: Parsed Abstract Syntax Tree.

        Raises:
            SyntaxError: If the source code contains syntax errors.
        """
        try:
            logger.debug("Parsing source code into AST.")
            return ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise

    @staticmethod
    def extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:
        """
        Extract function definitions from an AST.

        Args:
            tree (ast.AST): The AST from which to extract function definitions.

        Returns:
            List[ast.FunctionDef]: A list of function definition nodes.
        """
        logger.debug("Extracting function definitions from AST.")
        return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    @staticmethod
    def extract_classes(tree: ast.AST) -> List[ast.ClassDef]:
        """
        Extract class definitions from an AST.

        Args:
            tree (ast.AST): The AST from which to extract class definitions.

        Returns:
            List[ast.ClassDef]: A list of class definition nodes.
        """
        logger.debug("Extracting class definitions from AST.")
        return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    @staticmethod
    def extract_docstring(node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """
        Extract the docstring from an AST node.

        Args:
            node (Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]): Node from which to extract the docstring.

        Returns:
            Optional[str]: The extracted docstring, or None if no docstring is present.
        """
        try:
            logger.debug(f"Extracting docstring from node: {type(node).__name__}")
            return ast.get_docstring(node)
        except Exception as e:
            logger.error(f"Failed to extract docstring: {e}")
            return None

    @staticmethod
    def get_annotation(node: Optional[ast.AST]) -> str:
        """
        Get the annotation of an AST node.

        Args:
            node (Optional[ast.AST]): The AST node from which to get the annotation.

        Returns:
            str: The annotation as a string. Returns "Any" if no annotation is present.
        """
        if node is None:
            return "Any"
        try:
            logger.debug(f"Getting annotation for node: {ast.dump(node)}")
            return ast.unparse(node)
        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
            return "Any"

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """
        Add parent node information to each node in an AST.

        This function traverses the AST and adds a 'parent' attribute to each child node,
        pointing to its parent node.

        Args:
            tree (ast.AST): The AST to which parent information is to be added.
        """
        logger.debug("Adding parent information to AST nodes.")
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)
        logger.info("Parent information added to AST nodes.")

    def calculate_function_metrics(self, function_node: ast.FunctionDef) -> Dict[str, Union[int, float]]:
        """
        Calculate complexity metrics for a function.

        Args:
            function_node (ast.FunctionDef): AST node of the function.

        Returns:
            Dict[str, Union[int, float]]: Dictionary containing complexity metrics.
        """
        try:
            cyclomatic_complexity = self.metrics_calculator.calculate_cyclomatic_complexity(function_node)
            cognitive_complexity = self.metrics_calculator.calculate_cognitive_complexity(function_node)
            maintainability_index = self.metrics_calculator.calculate_maintainability_index(function_node)
            halstead_metrics = self.metrics_calculator.calculate_halstead_metrics(function_node)

            metrics = {
                'cyclomatic_complexity': cyclomatic_complexity,
                'cognitive_complexity': cognitive_complexity,
                'maintainability_index': maintainability_index,
                'halstead_metrics': halstead_metrics
            }
            logger.debug(f"Calculated metrics for function '{function_node.name}': {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for function '{function_node.name}': {e}")
            return {}

    def calculate_class_metrics(self, class_node: ast.ClassDef) -> Dict[str, Union[int, float]]:
        """
        Calculate complexity metrics for a class.

        Args:
            class_node (ast.ClassDef): AST node of the class.

        Returns:
            Dict[str, Union[int, float]]: Dictionary containing aggregated complexity metrics for the class methods.
        """
        try:
            method_metrics = []
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    metrics = self.calculate_function_metrics(node)
                    method_metrics.append(metrics)

            # Aggregate metrics if needed, for example, average complexities
            # For simplicity, we'll just collect metrics per method
            metrics = {
                'methods_metrics': method_metrics
            }
            logger.debug(f"Calculated metrics for class '{class_node.name}': {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for class '{class_node.name}': {e}")
            return {}

    @staticmethod
    def extract_imports(tree: ast.AST) -> List[str]:
        """
        Extract import statements from an AST.

        Args:
            tree (ast.AST): The AST from which to extract import statements.

        Returns:
            List[str]: A list of import module names.
        """
        logger.debug("Extracting import statements from AST.")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
```

## Updated `extraction_manager.py`

```python
import ast
from typing import Dict, Any, List
from core.logger import LoggerSetup
from core.utils import handle_exceptions
from extract.ast_analysis import ASTAnalyzer  # Assuming ASTAnalyzer is in ast_analysis.py

logger = LoggerSetup.get_logger(__name__)

class ExtractionManager:
    """
    A manager for extracting metadata from Python source code, particularly focusing on class and function definitions.
    It provides special handling for exception classes and robust error handling.
    """

    def __init__(self) -> None:
        """Initialize the ExtractionManager with an AST analyzer."""
        logger.debug("Initializing ExtractionManager.")
        self.analyzer = ASTAnalyzer()

    @handle_exceptions(logger.error)
    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code, including information about classes and functions.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata categorized into 'classes', 'functions', and 'imports'.
        """
        logger.debug("Starting metadata extraction.")
        try:
            tree = self.analyzer.parse_source_code(source_code)
            self.analyzer.add_parent_info(tree)
            classes, functions = [], []

            for node in ast.walk(tree):
                logger.debug(f"Processing node type: {type(node).__name__}")
                if isinstance(node, ast.ClassDef):
                    try:
                        logger.debug(f"Processing class definition: {node.name}")
                        is_exception = self._is_exception_class(node)
                        metadata = self._extract_class_metadata(node, is_exception)
                        if metadata:
                            # Calculate metrics for the class
                            metrics = self.analyzer.calculate_class_metrics(node)
                            metadata.update({'metrics': metrics})
                            classes.append(metadata)
                        logger.debug(
                            f"Extracted {'exception ' if is_exception else ''}class: "
                            f"{node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting class metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )
                elif isinstance(node, ast.FunctionDef):
                    try:
                        logger.debug(f"Processing function definition: {node.name}")
                        metadata = self._extract_function_metadata(node)
                        if metadata:
                            # Calculate metrics for the function
                            metrics = self.analyzer.calculate_function_metrics(node)
                            metadata.update({'metrics': metrics})
                            functions.append(metadata)
                        logger.debug(
                            f"Extracted function: {node.name} with metadata: {metadata}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error extracting function metadata for "
                            f"{getattr(node, 'name', '<unknown>')}: {e}"
                        )

            imports = self.analyzer.extract_imports(tree)

            logger.info(
                f"Extraction complete. Found {len(classes)} classes, "
                f"{len(functions)} functions, and {len(imports)} imports"
            )
            return {'classes': classes, 'functions': functions, 'imports': imports}

        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': [], 'imports': []}
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': [], 'imports': []}

    def extract_cross_module_metadata(self, module_paths: List[str]) -> Dict[str, Any]:
        """
        Extract metadata across multiple modules.

        Args:
            module_paths (List[str]): List of file paths to the modules.

        Returns:
            Dict[str, Any]: Aggregated metadata from all modules.
        """
        logger.debug("Extracting metadata across multiple modules.")
        aggregated_metadata = {
            'classes': [],
            'functions': [],
            'imports': []
        }
        for path in module_paths:
            with open(path, 'r') as file:
                source_code = file.read()
                module_metadata = self.extract_metadata(source_code)
                aggregated_metadata['classes'].extend(module_metadata.get('classes', []))
                aggregated_metadata['functions'].extend(module_metadata.get('functions', []))
                aggregated_metadata['imports'].extend(module_metadata.get('imports', []))
        return aggregated_metadata

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """
        Determine if a class node represents an exception class.

        Args:
            node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the class is an exception class, otherwise False.
        """
        logger.debug(f"Checking if class {node.name} is an exception class.")
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Exception', 'BaseException'}:
                logger.debug(f"Class {node.name} is an exception class.")
                return True
            elif isinstance(base, ast.Attribute) and base.attr in {'Exception', 'BaseException'}:
                logger.debug(f"Class {node.name} is an exception class.")
                return True
        logger.debug(f"Class {node.name} is not an exception class.")
        return False

    def _extract_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract metadata from a class definition node.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Indicates if this class is an exception class.

        Returns:
            Dict[str, Any]: A dictionary containing metadata of the class.
        """
        logger.debug(f"Extracting metadata for class: {node.name}")
        try:
            metadata = {
                'name': node.name,
                'docstring': self.analyzer.extract_docstring(node) or '',
                'lineno': node.lineno,
                'is_exception': is_exception,
                'type': 'exception_class' if is_exception else 'class'
            }

            additional_metadata = self._additional_class_metadata(node, is_exception)
            metadata.update(additional_metadata)
            logger.debug(f"Extracted metadata for class {node.name}: {metadata}")
            return metadata

        except (AttributeError, TypeError) as e:
            logger.error(f"Error in class metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'exception_class' if is_exception else 'class',
                'error': str(e)
            }

    def _additional_class_metadata(self, node: ast.ClassDef, is_exception: bool) -> Dict[str, Any]:
        """
        Extract additional metadata specific to normal or exception classes.

        Args:
            node (ast.ClassDef): The class node to extract metadata from.
            is_exception (bool): Indicates if this class is an exception class.

        Returns:
            Dict[str, Any]: Additional metadata based on class type.
        """
        if not is_exception:
            return {
                'methods': self._extract_methods(node),
                'bases': [self.analyzer.get_annotation(base) for base in node.bases],
                'decorators': self._extract_decorators(node)
            }
        else:
            return {
                'bases': [self.analyzer.get_annotation(base) for base in node.bases],
                'error_code': self._extract_error_code(node)
            }

    def _extract_error_code(self, node: ast.ClassDef) -> Optional[str]:
        """
        Extracts an error code from an exception class if defined.

        Args:
            node (ast.ClassDef): The exception class node.

        Returns:
            Optional[str]: The error code if present, otherwise None.
        """
        logger.debug(f"Extracting error code for exception class: {node.name}")
        error_code = None
        try:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == 'code':
                            value = item.value
                            if isinstance(value, ast.Str):
                                error_code = value.s
                            elif isinstance(value, ast.Constant):
                                error_code = str(value.value)
                            logger.debug(f"Extracted error code for {node.name}: {error_code}")
                            return error_code
            logger.debug(f"No error code found for {node.name}")
            return error_code
        except (AttributeError, TypeError) as e:
            logger.error(f"Error extracting error code for {node.name}: {e}")
            return error_code

    def _extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract metadata for methods within a class definition.

        Args:
            node (ast.ClassDef): The class node to extract methods from.

        Returns:
            List[Dict[str, Any]]: A list containing metadata for each method.
        """
        logger.debug(f"Extracting methods for class: {node.name}")
        methods = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                method_metadata = {
                    'name': stmt.name,
                    'docstring': self.analyzer.extract_docstring(stmt) or '',
                    'lineno': stmt.lineno
                }
                methods.append(method_metadata)
                logger.debug(f"Extracted method: {method_metadata}")
        return methods

    def _extract_decorators(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> List[str]:
        """
        Extract decorators applied to a class or method.

        Args:
            node (Union[ast.ClassDef, ast.FunctionDef]): The node to extract decorators from.

        Returns:
            List[str]: A list of decorator names as strings.
        """
        logger.debug(f"Extracting decorators for {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id}.{decorator.attr}")
        logger.debug(f"Extracted decorators: {decorators}")
        return decorators

    def _extract_function_metadata(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract metadata from a function definition node.

        Args:
            node (ast.FunctionDef): The function node to extract metadata from.

        Returns:
            Dict[str, Any]: Metadata for the function, or information about any errors encountered.
        """
        logger.debug(f"Extracting metadata for function: {node.name}")
        try:
            function_metadata = {
                'name': node.name,
                'docstring': self.analyzer.extract_docstring(node) or '',
                'args': [(arg.arg, self.analyzer.get_annotation(arg.annotation)) for arg in node.args.args],
                'return_type': self.analyzer.get_annotation(node.returns),
                'decorators': self._extract_decorators(node),
                'lineno': node.lineno
            }
            logger.debug(f"Extracted function metadata: {function_metadata}")
            return function_metadata
        except Exception as e:
            logger.error(f"Error in function metadata extraction for {node.name}: {e}")
            return {
                'name': node.name,
                'type': 'function',
                'error': str(e)
            }
```

## Key Enhancements

- **Cross-Module Metadata Extraction**: Added `extract_cross_module_metadata` to aggregate metadata from multiple modules.
- **Dependency Analysis**: Enhanced `ASTAnalyzer` to extract import statements, aiding in understanding module dependencies.
- **Comprehensive Metadata**: Extracts metadata for classes, functions, and imports, providing a holistic view of the codebase.
- **Robust Error Handling**: Continues to use decorators and logging for error handling during metadata extraction.

These updates enable the system to perform cross-module analysis, allowing for more comprehensive and contextually aware documentation generation. If you have further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:39:51]

**user**: Update ai_interaction.py  
[Timestamp: 2024/11/27 11:40:14]

**ai**: To update the `ai_interaction.py` module to incorporate cross-module awareness and enhance its functionality, we'll integrate the changes made to `ASTAnalyzer` and `ExtractionManager`. This will allow the module to handle more comprehensive metadata extraction and generate documentation with a broader context.

Here's the updated `ai_interaction.py`:

```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import json
from dataclasses import dataclass

from core.logger import LoggerSetup, log_error, log_info, log_debug
from core.cache import Cache
from core.monitoring import MetricsCollector
from config import AzureOpenAIConfig
from api.token_management import TokenManager
from api.api_client import AzureOpenAIClient
from exceptions import ValidationError, ProcessingError, CacheError
from extract.extraction_manager import ExtractionManager
from response_parser import ResponseParser
from markdown_generator import MarkdownDocumentationGenerator
from docstring_utils import DocstringValidator

# Initialize logger and config
logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """
    Handles AI interactions for documentation generation via Azure OpenAI API.

    Manages token limits, caching mechanisms, and metrics collection for robust processing.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.cache = cache
            self.metrics = metrics_collector
            self.token_manager = token_manager or TokenManager(
                model=config.model_name,
                deployment_name=config.deployment_name
            )

            self.client = AzureOpenAIClient(
                token_manager=self.token_manager,
                metrics_collector=metrics_collector
            )

            self.extraction_manager = ExtractionManager()
            self.response_parser = ResponseParser()
            self.docstring_validator = DocstringValidator()
            
            # Initialize function tools for structured output
            self._initialize_tools()

            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self):
        """Initialize the function tools for structured output."""
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does"
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality"
                        },
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
                            },
                            "description": "List of arguments with their types and descriptions"
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
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            },
                            "description": "List of exceptions that may be raised"
                        }
                    },
                    "required": ["summary", "description", "args", "returns"]
                }
            }
        }

    def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create a dynamic prompt based on the code and metadata."""
        prompt_parts = [
            "Generate comprehensive documentation for the following Python code.",
            "\nCode Analysis:",
            f"- Complexity Metrics:\n{self._format_metrics(metadata)}",
            f"- Imports: {', '.join(metadata.get('imports', []))}"
        ]

        if metadata.get('docstring'):
            prompt_parts.append(f"\nExisting Documentation:\n{metadata['docstring']}")

        if metadata.get('class_info'):
            prompt_parts.append(f"\nClass Information:\n{self._format_class_info(metadata['class_info'])}")

        prompt_parts.extend([
            "\nRequirements:",
            "- Follow Google Style Python docstring format",
            "- Include comprehensive parameter descriptions",
            "- Document return values and types",
            "- List possible exceptions",
            "- Add usage examples for complex functionality",
            
            f"\nSource Code:\n{source_code}",
            
            "\nProvide the documentation in a structured format using the specified function schema."
        ])

        return "\n".join(prompt_parts)

    def _format_class_info(self, class_info: Dict[str, Any]) -> str:
        """Format class information for the prompt."""
        info_parts = [
            f"Class Name: {class_info.get('name', 'Unknown')}",
            f"Base Classes: {', '.join(class_info.get('bases', []))}",
            f"Methods: {len(class_info.get('methods', []))}",
        ]
        return "\n".join(info_parts)

    def _format_metrics(self, metadata: Dict[str, Any]) -> str:
        """Format the complexity metrics into a string to include in the prompt."""
        lines = []
        for func in metadata.get('functions', []):
            metrics = func.get('metrics', {})
            lines.append(
                f"Function {func['name']} - Cyclomatic Complexity: {metrics.get('cyclomatic_complexity')}, "
                f"Cognitive Complexity: {metrics.get('cognitive_complexity')}, "
                f"Maintainability Index: {metrics.get('maintainability_index')}"
            )
        for cls in metadata.get('classes', []):
            lines.append(f"Class {cls['name']}:")
            for method in cls.get('methods', []):
                method_metrics = method.get('metrics', {})
                lines.append(
                    f"  Method {method['name']} - Cyclomatic Complexity: {method_metrics.get('cyclomatic_complexity')}, "
                    f"Cognitive Complexity: {method_metrics.get('cognitive_complexity')}, "
                    f"Maintainability Index: {method_metrics.get('maintainability_index')}"
                )
        return '\n'.join(lines)

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any]
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata)

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                return None

            # Handle function call response
            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.function.name == "generate_docstring":
                    try:
                        docstring_data = json.loads(tool_call.function.arguments)
                        
                        # Validate the structured output
                        is_valid, errors = self.docstring_validator.validate_docstring(docstring_data)
                        if not is_valid:
                            logger.error(f"Docstring validation failed: {errors}")
                            raise ProcessingError(f"Invalid documentation: {errors}")

                        # Generate markdown documentation
                        markdown_generator = MarkdownDocumentationGenerator(
                            source_code,
                            module_path=metadata.get('module_path')
                        )
                        full_documentation = markdown_generator.generate_markdown(docstring_data)

                        return ProcessingResult(
                            content=full_documentation,
                            usage=usage or {},
                            processing_time=(datetime.now() - datetime.now()).total_seconds()
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse function call response: {e}")
                        raise ProcessingError("Invalid JSON in function response")

            raise ProcessingError("No valid function call response received")

        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    def _format_docstring_from_json(self, docstring_data: Dict[str, Any]) -> str:
        """Format the JSON docstring data into Google Style format."""
        docstring_parts = []
        
        # Add summary and description
        docstring_parts.append(docstring_data["summary"])
        if docstring_data.get("description"):
            docstring_parts.extend(["", docstring_data["description"]])

        # Add arguments
        if docstring_data.get("args"):
            docstring_parts.append("\nArgs:")
            for arg in docstring_data["args"]:
                docstring_parts.append(f"    {arg['name']} ({arg['type']}): {arg['description']}")

        # Add returns section
        if docstring_data.get("returns"):
            docstring_parts.append("\nReturns:")
            docstring_parts.append(f"    {docstring_data['returns']['type']}: {docstring_data['returns']['description']}")

        # Add raises section
        if docstring_data.get("raises"):
            docstring_parts.append("\nRaises:")
            for error in docstring_data["raises"]:
                docstring_parts.append(f"    {error['type']}: {error['description']}")

        return "\n".join(docstring_parts)

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        module_paths: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Process source code to generate and embed documentation."""
        operation_start = datetime.now()

        try:
            # Validate input
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Extract metadata
            if module_paths:
                metadata = self.extraction_manager.extract_cross_module_metadata(module_paths)
            else:
                metadata = self.extraction_manager.extract_metadata(source_code)

            # Generate documentation
            result = await self._generate_documentation(source_code, metadata)
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = await self._update_code(source_code, result.content)
            if not updated_code:
                raise ProcessingError("Code update failed")

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            # Track metrics
            await self._track_metrics(operation_start, True, result.usage)

            return updated_code, result.content

        except Exception as e:
            await self._track_error("ProcessingError", e, operation_start)
            logger.error(f"Process code failed: {str(e)}")
            raise

    def _extract_metadata(self, source_code: str) -> Dict[str, Any]:
        """Extract metadata using ExtractionManager."""
        metadata = self.extraction_manager.extract_metadata(source_code)
        return metadata

    async def _update_code(self, source_code: str, documentation: str) -> Optional[str]:
        """Update source code with generated documentation."""
        try:
            # Insert the documentation at the top of the source code
            updated_code = f'"""\n{documentation}\n"""\n\n{source_code}'
            return updated_code
        except Exception as e:
            logger.error(f"Code update failed: {str(e)}")
            raise ProcessingError(f"Failed to update code: {str(e)}")

    async def _cache_result(self, cache_key: str, code: str, documentation: str) -> None:
        """Cache the processing result."""
        try:
            await self.cache.save_docstring(
                cache_key,
                {
                    'code': code,
                    'docs': documentation
                }
            )
            logger.debug(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.error(f"Caching failed: {str(e)}")
            # Don't raise - caching failure shouldn't stop processing

    async def _track_metrics(self, start_time: datetime, success: bool, usage: Dict[str, Any]) -> None:
        """Track operation metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=success,
                duration=duration,
                usage=usage
            )
        # Optionally log metrics
        log_info(f"Operation metrics: success={success}, duration={duration}, usage={usage}")

    async def _track_error(self, error_type: str, error: Exception, start_time: datetime) -> None:
        """Track and log error metrics."""
        if self.metrics:
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='documentation_generation',
                success=False,
                duration=duration,
                error=f"{error_type}: {str(error)}"
            )

    async def _check_cache(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """Check if the result is cached."""
        try:
            cached_data = await self.cache.get_cached_docstring(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data['code'], cached_data['docs']
            else:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            raise CacheError(f"Cache check failed: {str(e)}")

    async def close(self) -> None:
        """Close the AI interaction handler and cleanup resources."""
        try:
            await self.client.close()
            logger.info("AI Interaction Handler closed successfully")
        except Exception as e:
            logger.error(f"Error closing AI handler: {str(e)}")

    async def __aenter__(self) -> 'AIInteractionHandler':
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Async context manager exit."""
        await self.close()
```

## Key Enhancements

- **Cross-Module Metadata Extraction**: The `process_code` method now supports extracting metadata from multiple modules using `extract_cross_module_metadata`.
- **Dynamic Prompt Generation**: Prompts are generated based on comprehensive metadata, including imports and cross-module information.
- **Error Handling and Caching**: Improved error handling and caching mechanisms ensure robust processing.
- **Structured Output**: Utilizes structured output for generating and validating docstrings.

These updates enable the `AIInteractionHandler` to generate more comprehensive and contextually aware documentation by considering the broader context of the codebase. If you have further questions or need additional features, feel free to ask!  
[Timestamp: 2024/11/27 11:41:53]
