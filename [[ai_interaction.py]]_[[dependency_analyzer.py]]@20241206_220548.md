---
epoch: 1733544348272
modelKey: o1-mini|openai
tags:
  - copilot-conversation
---

**user**: [ai_service.py](ai_service.py.md) [dependency_analyzer.py](dependency_analyzer.py.md)
[Timestamp: 2024/12/06 22:05:48]

**ai**: It looks like you've provided the same content for `[[ai_interaction.py]]`, `[[dependency_analyzer.py]]`, and `[[ai_interaction.py.md]]`. The `[[dependency_analyzer.py.md]]` file seems to be a duplicate of `[[dependency_analyzer.py]]` as well. 

If you intended to provide different content for each file, could you please clarify or provide the correct content for each? Otherwise, if you just need information or analysis on these files, let me know what you need help with!

For now, I'll assume you want to know more about the functionality provided by these files:

### `ai_interaction.py`

This module is designed to handle interactions with AI models, particularly for generating and integrating documentation into Python source code. It includes:

- **CustomJSONEncoder**: A custom JSON encoder that can handle sets and other non-serializable types.
- **AIInteractionHandler**: A class that manages the entire process of extracting code metadata, generating prompts, interacting with the AI, parsing responses, and integrating the AI-generated documentation back into the source code.

### `dependency_analyzer.py`

This module focuses on analyzing dependencies within Python source code using the Abstract Syntax Tree (AST). It includes:

- **extract_dependencies_from_node**: A function that extracts dependencies from a given AST node.
- **DependencyAnalyzer**: A class that analyzes and categorizes dependencies into standard library, third-party, and local modules. It also detects circular dependencies and can generate a visual dependency graph using Graphviz.

If you have a specific question or need further details about these modules, feel free to ask!
[Timestamp: 2024/12/06 22:06:01]

**user**: Enhance 
[Timestamp: 2024/12/06 22:06:12]

**ai**: Certainly! To enhance the `ai_interaction.py` and `dependency_analyzer.py` modules, we can focus on improving their functionality, readability, and maintainability. Here are some suggestions for each module:

### Enhancements for `ai_interaction.py`

1. **Error Handling and Logging**:
   - Improve error messages and logging to provide more context and help with debugging.
   - Add more specific logging levels (e.g., `info`, `debug`, `warning`, `error`).

2. **Code Modularity**:
   - Break down large methods into smaller, more manageable functions.
   - Use helper functions to reduce code duplication.

3. **Configuration and Flexibility**:
   - Allow more configuration options for the AI model (e.g., different models, temperature settings).
   - Make the system prompt more flexible and configurable.

4. **Testing and Validation**:
   - Add unit tests to ensure each component works as expected.
   - Validate input data to prevent unexpected errors.

5. **Documentation**:
   - Improve docstrings for better understanding and maintainability.
   - Add comments where necessary to explain complex logic.

Here is an enhanced version of `ai_interaction.py` with some of these improvements:

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
from core.logger import LoggerSetup
from core.markdown_generator import MarkdownGenerator
from core.metrics import Metrics
from core.response_parsing import ParsedResponse, ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext, ExtractionResult
from exceptions import ConfigurationError, ProcessingError

logger = LoggerSetup.get_logger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
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
    """Safely serialize any object for logging purposes."""
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        logger.error(f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}")
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
        config: AzureOpenAIConfig | None = None,
        cache: Cache | None = None,
        token_manager: TokenManager | None = None,
        response_parser: ResponseParsingService | None = None,
        metrics: Metrics | None = None,
        docstring_schema: dict[str, Any] | None = None,
        ai_model: str = "gpt-4",
        temperature: float = 0.7,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Cache | None = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: dict[str, Any] = docstring_schema or load_schema()
        self.ai_model = ai_model
        self.temperature = temperature

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response: The response to truncate (either string or dictionary)
            length: Maximum length of the truncated response

        Returns:
            str: Truncated string representation of the response
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
            logger.error(f"Error truncating response: {str(e)}")
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """Process the source code to extract metadata, interact with the AI, and integrate responses."""
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
            extraction_result: ExtractionResult | None = await extractor.extract_code(source_code)
            if not extraction_result:
                logger.error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            logger.debug("Generated prompt for AI")

            ai_response: str | dict[str, Any] = await self._interact_with_ai(prompt)
            logger.debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                logger.error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                logger.error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            logger.error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """Integrate the AI response into the source code and update the documentation."""
        try:
            logger.debug("Starting AI response integration")
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
            logger.error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """Generate markdown documentation from AI response and extraction result."""
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
        """Ensure the AI response has the required fields, and repair if necessary."""
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            logger.error(f"AI response missing required fields: {missing}")

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
        """Create a list with the processed response."""
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """Process the docstrings using DocstringProcessor."""
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        logger.debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> str | dict[str, Any]:
        """Interact with the AI model to generate responses."""
        try:
            request_params: dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            logger.debug("Sending request to AI")

            system_prompt = self._get_system_prompt()

            response = await self.client.chat.completions.create(
                model=self.ai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=self.temperature
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            logger.debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                logger.debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        logger.debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        logger.error(f"Failed to parse extracted JSON: {e2}")

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            logger.error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI model."""
        return """You are a Python documentation expert. Generate complete docstrings in Google format.
Return ONLY a JSON object with this structure, no other text:
{
"summary": "Brief one-line summary",
"description": "Detailed multi-line description",
"args": [{"name": "param_name", "type": "param_type", "description": "param
[Timestamp: 2024/12/06 22:07:22]

**user**: Continue 
[Timestamp: 2024/12/06 22:07:27]

**ai**: Certainly! Let's continue with the enhancements for `ai_interaction.py` and then move on to `dependency_analyzer.py`.

### Continued Enhancements for `ai_interaction.py`

#### `_get_system_prompt` Method
Let's complete the `_get_system_prompt` method and add more configuration options.

```python
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI model."""
        return """You are a Python documentation expert. Generate complete docstrings in Google format.
Return ONLY a JSON object with this structure, no other text:
{
"summary": "Brief one-line summary",
"description": "Detailed multi-line description",
"args": [{"name": "param_name", "type": "param_type", "description": "param description"}],
"returns": {"type": "return_type", "description": "what is returned"},
"raises": [{"exception": "ErrorType", "description": "when this error occurs"}]
}
Include only the JSON object in the response without any additional text or formatting."""
```

#### `create_dynamic_prompt` Method
Let's add more detailed logging and handle edge cases better.

```python
    async def create_dynamic_prompt(self, extracted_info: dict[str, str | list[dict[str, Any]] | dict[str, Any]]) -> str:
        """Create a dynamic prompt for the AI model."""
        try:
            logger.debug("Creating dynamic prompt")
            logger.debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

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
                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            logger.debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            logger.error(f"Error creating prompt: {e}", exc_info=True)
            raise
```

#### `generate_docstring` Method
Let's add more detailed logging and handle edge cases better.

```python
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
    ) -> dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name: Name of the function or class
            is_class: Whether the target is a class
            params: List of parameters with their types and descriptions
            return_type: Return type of the function
            complexity_score: Complexity score of the function
            existing_docstring: Existing docstring to enhance
            decorators: List of decorators applied to the function
            exceptions: List of exceptions raised by the function

        Returns:
            dict: Generated docstring content
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
                model=self.ai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=self.temperature,
                stop=["END"],
            )
            response_content: str | None = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            logger.info(f"Generated docstring for {func_name}")
            return parsed_response.content

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decoding error while generating docstring for {func_name}: {e}",
                exc_info=True
            )
            raise
        except ProcessingError as e:
            logger.error(
                f"Processing error while generating docstring for {func_name}: {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating docstring for {func_name}: {e}",
                exc_info=True
            )
            raise
```

#### `close` Method
Ensure proper cleanup and logging.

```python
    async def close(self) -> None:
        """Cleanup resources held by AIInteractionHandler."""
        logger.debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            logger.debug("Cache resources have been cleaned up")
        logger.info("AIInteractionHandler resources have been cleaned up")
```

### Enhancements for `dependency_analyzer.py`

1. **Error Handling and Logging**:
   - Improve error messages and logging to provide more context and help with debugging.
   - Add more specific logging levels (e.g., `info`, `debug`, `warning`, `error`).

2. **Code Modularity**:
   - Break down large methods into smaller, more manageable functions.
   - Use helper functions to reduce code duplication.

3. **Configuration and Flexibility**:
   - Allow more configuration options for dependency analysis (e.g., different module paths).

4. **Testing and Validation**:
   - Add unit tests to ensure each component works as expected.
   - Validate input data to prevent unexpected errors.

5. **Documentation**:
   - Improve docstrings for better understanding and maintainability.
   - Add comments where necessary to explain complex logic.

Here is an enhanced version of `dependency_analyzer.py` with some of these improvements:

```python
"""
Dependency analysis module for Python source code.

This module provides functionality to analyze and categorize dependencies
within Python source code using the Abstract Syntax Tree (AST). It identifies
imports, function calls, and attributes used in the code, detects circular
dependencies, and can generate a visual dependency graph.

Functions:
    extract_dependencies_from_node: Extracts dependencies from an AST node.

Classes:
    DependencyAnalyzer: Analyzes and categorizes code dependencies.

Example usage:
    analyzer = DependencyAnalyzer(context)
    dependencies = analyzer.analyze_dependencies(ast_tree)
"""
import sysconfig
import ast
import importlib.util
import os
import sys
from typing import Dict, Set, Optional, List, Tuple, Union, Any

from core.logger import LoggerSetup
from core.types import ExtractionContext
from core.utils import NodeNameVisitor

logger = LoggerSetup.get_logger(__name__)


def extract_dependencies_from_node(node: ast.AST) -> dict[str, set[str]]:
    """Extract dependencies from an AST node."""
    dependencies: dict[str, set[str]] = {"imports": set(), "calls": set(), "attributes": set()}
    for child in ast.walk(node):
        try:
            if isinstance(child, ast.Import):
                for name in child.names:
                    dependencies["imports"].add(name.name)
            elif isinstance(child, ast.ImportFrom) and child.module:
                dependencies["imports"].add(child.module)
            elif isinstance(child, ast.Call):
                visitor = NodeNameVisitor()
                visitor.visit(child.func)
                dependencies["calls"].add(visitor.name)
            elif isinstance(child, ast.Attribute):
                visitor = NodeNameVisitor()
                visitor.visit(child)
                dependencies["attributes"].add(visitor.name)
        except Exception as e:
            logger.warning(f"Unsupported AST node encountered: {e}")
    return dependencies


class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Attributes:
        context (ExtractionContext): The context for extraction operations.
        module_name (Optional[str]): The name of the module being analyzed.
        _function_errors (List[str]): List of errors encountered during function metadata extraction.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The context for extraction operations.
        """
        self.logger = logger
        self.context = context
        self.module_name = context.module_name
        self._function_errors: List[str] = []
        logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Analyze module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node to analyze.
            module_name (Optional[str]): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies into 'stdlib', 'third_party', and 'local'.
        """
        logger.info("Starting dependency analysis")
        
        # Set module name with validation
        if module_name:
            self.module_name = module_name
        elif not self.module_name:
            logger.warning("No module name provided for dependency analysis")
            self.module_name = "<unknown_module>"  # Set a default value

        try:
            raw_deps = extract_dependencies_from_node(node)
            deps = self._categorize_dependencies(raw_deps)
            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                logger.warning(f"Circular dependencies detected: {circular_deps}")
            
            logger.info(f"Dependency analysis completed: {len(deps)} categories found")
            return deps

        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {"stdlib": set(), "third_party": set(), "local": set()}

    def extract_function_metadata(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract function metadata including raises information.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract metadata from.

        Returns:
            Dict[str, Any]: A dictionary containing function metadata such as name, docstring, raises, args, and returns.
        """
        try:
            metadata = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "raises": self._extract_raises(node),
                "args": self._extract_args(node),
                "returns": self._extract_returns(node)
            }
            return metadata
        except Exception as e:
            self._function_errors.append(f"Error extracting metadata for {getattr(node, 'name', 'unknown')}: {e}")
            logger.error(f"Error extracting metadata for {getattr(node, 'name', 'unknown')}: {e}", exc_info=True)
            return {}

    def _extract_raises(self, node: ast.AST) -> List[Dict[str, str]]:
        raises: List[Dict[str, str]] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                visitor = NodeNameVisitor()
                visitor.visit(child.exc)
                if visitor.name:
                    description = "Exception raised in function execution"
                    # Try to extract message from raise statement if it's a Call
                    if isinstance(child.exc, ast.Call) and child.exc.args:
                        msg_visitor = NodeNameVisitor()
                        msg_visitor.visit(child.exc.args[0])
                        if msg_visitor.name:
                            description = msg_visitor.name.strip("'\"")
                    raises.append({
                        "exception": visitor.name,
                        "description": description
                    })
        return raises

    def _extract_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, str]]:
        """Extract arguments from function definition.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract arguments from.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing argument names, types, and descriptions.
        """
        args: List[Dict[str, str]] = []
        for arg in node.args.args:
            args.append({
                "name": arg.arg,
                "type": "Unknown",  # Type inference can be added if needed
                "description": "No description available"
            })
        return args

    def _extract_returns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, str]:
        """Extract return type from function definition.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node to extract return type from.

        Returns:
            Dict[str, str]: A dictionary containing the return type and description.
        """
        return_type = "Any"  # Default return type
        description = "No description available"

        try:
            # Check for return annotation
            if node.returns:
                visitor = NodeNameVisitor()
                visitor.visit(node.returns)
                if visitor.name:
                    return_type = visitor.name

            # Try to extract return description from docstring
            docstring = ast.get_docstring(node)
            if docstring:
                # Look for return/returns section in docstring
                import re
                returns_match = re.search(r'(?:Returns?|->):\s*(.+?)(?:\n\n|\Z)', 
                                        docstring, re.DOTALL)
                if returns_match:
                    description = returns_match.group(1).strip()

        except Exception as e:
            logger.debug(f"Error extracting return type: {e}")

        return {
            "type": return_type,
            "description": description
        }

    def _categorize_dependencies(self, raw_deps: dict[str, set[str]]) -> dict[str, set[str]]:
        """Categorize raw dependencies into stdlib, third-party, or local.

        Args:
            raw_deps (Dict[str, Set[str]]): The raw dependencies extracted from the AST.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies into 'stdlib', 'third_party', and 'local'.
        """
        categorized_deps: Dict[str, Set[str]] = {"stdlib": set(), "third_party": set(), "local": set()}

        for module_name in raw_deps.get("imports", []):
            self._categorize_import(module_name, categorized_deps)

        return categorized_deps

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module to categorize.
            deps (Dict[str, Set[str]]): The dictionary to store categorized dependencies.
        """
        logger.debug(f"Categorizing import: {module_name}")

        try:
            if module_name in sys.builtin_module_names or module_name in self._get_stdlib_modules():
                deps["stdlib"].add(module_name)
                return

            if self.module_name:
                module_parts = self.module_name.split(".")
                if module_name.startswith(".") or module_name.startswith(self.module_name):
                    deps["local"].add(module_name)
                    return
                for i in range(len(module_parts)):
                    potential_module = ".".join(module_parts[:i] + [module_name])
                    if self._module_exists(potential_module):
                        deps["local"].add(module_name)
                        return

            deps["third_party"].add(module_name)

        except Exception as e:
            logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps["third_party"].add(module_name)

    def _module_exists(self, module_name: str) -> bool:
        """Check if a module exists.

        Args:
            module_name (str): The name of the module to check.

        Returns:
            bool: True if the module exists, False otherwise.
        """
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def _get_stdlib_modules(self) -> Set[str]:
        """Get a set of standard library module names.

        Returns:
            Set[str]: A set of standard library module names.
        """
        # For Python 3.10+, use built-in stdlib_module_names
        if hasattr(sys, "stdlib_module_names"):
            return set(sys.stdlib_module_names)
        
        try:
            paths = sysconfig.get_paths()
            stdlib_dir = paths.get("stdlib")
            if not stdlib_dir:
                return set()
                
            # Explicitly type-annotate the set
            modules: Set[str] = set()
            
            # Walk through standard library directory
            for _, _, files in os.walk(stdlib_dir):
                for file in files:
                    if file.endswith(".py"):
                        module = os.path.splitext(file)[0]
                        modules.add(module)
            return modules
            
        except Exception as e:
            logger.error("Error getting stdlib modules: %s", e)
            return set()

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """Detect direct and indirect circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): The dependencies to check for circular references.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        circular_dependencies: List[Tuple[str, str]] = []
        visited: Set[str] = set()
        path: Set[str] = set()
        
        # Guard against None module_name
        if self.module_name is None:
            logger.warning("No module name set for dependency analysis")
            return circular_dependencies

        def visit(module: str) -> None:
            if module in path:
                # Now we know self.module_name is str, not None
                if self.module_name is not None:
                    circular_dependencies.append((module, self.module_name))
                return
            if module in visited:
                return
                
            visited.add(module)
            path.add(module)
            
            for dep in dependencies.get(module, set()):
                visit(dep)
                
            path.remove(module)

        try:
            visit(self.module_name)
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)

        return circular_dependencies

    def generate_dependency_graph(self, dependencies: Dict[str, Set[str]], output_file: Union[str, os.PathLike]) -> None:
        """Generates a visual dependency graph.

        Args:
            dependencies (Dict[str, Set[str]]): The dependencies to visualize.
            output_file (str): The file path to save the generated graph.
        """
        logger.debug("Generating dependency graph.")
        try:
            from graphviz import Digraph
        except ImportError:
            logger.warning("Graphviz not installed. Skipping dependency graph generation.")
            return

        try:
            dot = Digraph(comment="Module Dependencies")

            for category, modules in dependencies.items():
                sub = Digraph(name=f"cluster_{category}")
                sub.attr(label=str(category))
                for module in modules:
                    sub.node(module, label=module, _attributes=None)
                dot.subgraph(sub)  # type: ignore

            dot.render(output_file, view=False, cleanup=True)
            logger.info(f"Dependency graph saved to {output_file}")

        except Exception as e:
            logger.error(f"Error generating dependency graph: {e}")
            logger.error("Make sure Graphviz is installed and in your system PATH")

    def extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract dependencies from AST.

        Args:
            tree (ast.AST): The AST to analyze.

        Returns:
            List[str]: A list of dependency strings.
        """
        deps = self.analyze_dependencies(tree)
        return list({dep for deps in deps.values() for dep in deps})
```

### Summary of Enhancements

- **Improved Logging**: Added more detailed and specific logging to help with debugging and understanding the flow of the program.
- **Modular Code**: Broke down large methods into smaller, more manageable functions.
- **Configuration Flexibility**: Added configuration options for AI model and temperature settings.
- **Error Handling**: Improved error handling to catch and log more specific exceptions.
- **Documentation**: Improved docstrings for better understanding and maintainability.

These enhancements should make the modules more robust, maintainable, and easier to understand. If you have specific areas or functionalities you want to focus on, feel free to let me know!
[Timestamp: 2024/12/06 22:10:22]