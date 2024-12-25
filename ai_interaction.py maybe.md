```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import ast
from dataclasses import dataclass
import openai
from openai import APIError
from jsonschema import validate, ValidationError as JsonValidationError
from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor, DocstringData
from core.code_extraction import CodeExtractor
from api.token_management import TokenManager
from api.api_client import APIClient
from exceptions import ValidationError, ProcessingError, CacheError

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """
    Handles AI interactions for documentation generation via Azure OpenAI API.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.metrics_collector = metrics_collector or MetricsCollector()
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config
            )
            self.client = APIClient(self.config)
            self.docstring_processor = DocstringProcessor()
            self.code_extractor = CodeExtractor()
            self._initialize_tools()
            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = {
            "name": "generate_docstring",
            "description": "Generate a Python docstring with structured information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of what the code does."
                    },
                    "description": {
                        "type": "string",
                        "description": "A detailed description of the functionality."
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
                    }
                },
                "required": ["summary", "description", "args", "returns"]
            }
        }

    def _create_function_calling_prompt(
        self, 
        source_code: str, 
        metadata: Dict[str, Any], 
        node: Optional[ast.AST] = None
    ) -> str:
        """Creates the initial prompt for function calling."""
        # Get code extraction data
        extraction_data = self.code_extractor.extract_code(source_code)
        
        # Format extracted information
        complexity_info = self._format_complexity_info(extraction_data)
        dependencies_info = self._format_dependencies_info(extraction_data)
        type_info = self._format_type_info(extraction_data)
        
        prompt = (
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
            
            "3. CODE CONTEXT:\n"
            f"{complexity_info}\n"
            f"{dependencies_info}\n"
            f"{type_info}\n\n"
            
            "4. SPECIFIC REQUIREMENTS:\n"
            "- Use exact parameter names and types from the code\n"
            "- Include all type hints in documentation\n"
            "- Document class inheritance where applicable\n"
            "- Note async/generator functions appropriately\n"
            "- Include property decorators in documentation\n\n"
            
            "The code to document is:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n"
        )
        return prompt

    def _format_complexity_info(self, extraction_data: Any) -> str:
        """Format complexity information for the prompt."""
        complexity_lines = ["Complexity Information:"]
        
        # Add class complexities
        for cls in extraction_data.classes:
            score = cls.metrics.get('complexity', 0)
            complexity_lines.append(f"- Class '{cls.name}' complexity: {score}")
            for method in cls.methods:
                m_score = method.metrics.get('complexity', 0)
                complexity_lines.append(f"  - Method '{method.name}' complexity: {m_score}")
        
        # Add function complexities
        for func in extraction_data.functions:
            score = func.metrics.get('complexity', 0)
            complexity_lines.append(f"- Function '{func.name}' complexity: {score}")
        
        return "\n".join(complexity_lines)

    def _format_dependencies_info(self, extraction_data: Any) -> str:
        """Format dependency information for the prompt."""
        dep_lines = ["Dependencies:"]
        
        for name, deps in extraction_data.imports.items():
            dep_lines.append(f"- {name}: {', '.join(deps)}")
        
        return "\n".join(dep_lines)

    def _format_type_info(self, extraction_data: Any) -> str:
        """Format type information for the prompt."""
        type_lines = ["Type Information:"]
        
        # Add class attribute types
        for cls in extraction_data.classes:
            type_lines.append(f"Class '{cls.name}':")
            for attr in cls.attributes:
                type_lines.append(f"- {attr['name']}: {attr['type']}")
            for method in cls.methods:
                args_info = [f"{arg.name}: {arg.type_hint}" for arg in method.args]
                type_lines.append(f"- Method '{method.name}({', '.join(args_info)}) -> {method.return_type}'")
        
        # Add function types
        for func in extraction_data.functions:
            args_info = [f"{arg.name}: {arg.type_hint}" for arg in func.args]
            type_lines.append(f"Function '{func.name}({', '.join(args_info)}) -> {func.return_type}'")
        
        return "\n".join(type_lines)

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        max_attempts = 3
        attempts = 0
        start_time = datetime.now()
        prompt = self._create_function_calling_prompt(source_code, metadata, node)

        while attempts < max_attempts:
            try:
                # Make API call
                response = await self.client.chat.completions.create(
                    model=self.config.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[self.docstring_function],
                    tool_choice={"type": "function", "function": {"name": "generate_docstring"}},
                    max_tokens=2000,
                    temperature=0.3
                )

                # Process the response
                message = response.choices[0].message
                if message.tool_calls and message.tool_calls[0].function:
                    function_args = message.tool_calls[0].function.arguments
                    try:
                        # Parse the AI's response
                        response_data = json.loads(function_args)
                        
                        # Convert to DocstringData
                        docstring_data = self._parse_ai_response(response_data)
                        
                        # Format docstring
                        formatted_docstring = self.docstring_processor.format(docstring_data)
                        
                        # Store in metadata for further use
                        if metadata is not None:
                            metadata['ai_generated_docs'] = {
                                'docstring': formatted_docstring,
                                'docstring_data': docstring_data.dict()
                            }
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        return ProcessingResult(
                            content=formatted_docstring,
                            usage=response.usage.model_dump(),
                            processing_time=processing_time
                        )

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from AI: {e}. Arguments: {function_args}")
                        prompt = self._create_refinement_prompt(prompt, "Invalid JSON returned", {})
                        attempts += 1

                else:
                    logger.warning("No tool calls in response")
                    attempts += 1

            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                attempts += 1
                await asyncio.sleep(2 ** attempts)

        logger.error(f"Failed after {max_attempts} attempts")
        return None

    def _create_refinement_prompt(self, original_prompt: str, error_message: str, previous_response: dict) -> str:
        """Creates a refinement prompt, handling previous responses and errors."""
        formatted_response = json.dumps(previous_response, indent=4) if previous_response else ""

        prompt = (
            f"{error_message}\n\n"
            + "Previous Response (if any):\n"
            + f"```json\n{formatted_response}\n```\n\n"
            + original_prompt
        )
        return prompt

    def _parse_ai_response(self, response_data: Dict[str, Any]) -> DocstringData:
        """Parse the AI response into DocstringData."""
        try:
            return DocstringData(
                summary=response_data.get("summary", ""),
                description=response_data.get("description", ""),
                args=response_data.get("args", []),
                returns=response_data.get("returns", {"type": "None", "description": ""}),
                raises=response_data.get("raises", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise

    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Process source code to generate docstrings."""
        try:
            # Check cache first if enabled
            if self.cache and cache_key:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return cached_result['code'], cached_result['docs']

            # Parse code and generate documentation
            tree = ast.parse(source_code)
            metadata = {}
            result = await self._generate_documentation(source_code, metadata, tree)
            
            if not result:
                return None

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, source_code, result.content)

            return source_code, result.content

        except Exception as e:
            logger.error(f"Failed to process code: {e}")
            return None

    async def _cache_result(self, cache_key: str, code: str, docs: str) -> None:
        """Cache the processing result."""
        try:
            if self.cache:
                await self.cache.save_docstring(
                    cache_key,
                    {'code': code, 'docs': docs}
                )
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if hasattr(self, 'client'):
                await self.client.close()
            if self.cache:
                await self.cache.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.token_manager:
                await self.token_manager.close()
        except Exception as e:
            logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
```