```python
# ai_interaction.py  
"""  
AI Interaction Handler Module  
   
Manages interactions with Azure OpenAI API, handling token management,  
caching, and response processing for documentation generation.  
"""  
   
import asyncio  
import json  
from datetime import datetime  
from typing import Optional, Tuple, Dict, Any, List  
import ast  
from dataclasses import dataclass, asdict  
import openai  
from core.logger import LoggerSetup  
from core.cache import Cache  
from core.monitoring import MetricsCollector  
from core.config import AzureOpenAIConfig  
from core.docstring_processor import DocstringProcessor, DocstringData  
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
   
# The Docstring schema remains the same  
DOCSTRING_SCHEMA = {  
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
        }  
    },  
    "required": ["summary", "description", "args", "returns"]  
}  
   
class AIInteractionHandler:  
    """  
    Handles AI interactions for documentation generation via Azure OpenAI API.  
  
    Manages token limits, caching mechanisms, and metrics collection for robust processing.  
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
            self._initialize_tools()  
            logger.info("AI Interaction Handler initialized successfully")  
  
            # Set up OpenAI API configuration  
            openai.api_type = "azure"  
            openai.api_key = self.config.api_key  
            openai.api_base = self.config.endpoint  
            openai.api_version = self.config.api_version  
  
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
                                "name": {"type": "string", "description": "The name of the argument."},  
                                "type": {"type": "string", "description": "The type of the argument."},  
                                "description": {"type": "string", "description": "A description of the argument."}  
                            },  
                            "required": ["name", "type", "description"]  
                        },  
                        "description": "A list of arguments, each with a name, type, and description."  
                    },  
                    "returns": {  
                        "type": "object",  
                        "properties": {  
                            "type": {"type": "string", "description": "The return type."},  
                            "description": {"type": "string", "description": "A description of the return value."}  
                        },  
                        "required": ["type", "description"],  
                        "description": "An object describing the return value, including its type and a description."  
                    },  
                    "raises": {  
                        "type": "array",  
                        "items": {  
                            "type": "object",  
                            "properties": {  
                                "exception": {"type": "string", "description": "The type of exception raised."},  
                                "description": {"type": "string", "description": "A description of when the exception is raised."}  
                            },  
                            "required": ["exception", "description"]  
                        },  
                        "description": "A list of exceptions that may be raised, each with a type and description."  
                    }  
                },  
                "required": ["summary", "description", "args", "returns"]  
            }  
        }  
  
    async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling and robust error handling."""
        max_attempts = 3
        attempts = 0
        start_time = datetime.now()
        prompt = self._create_function_calling_prompt(source_code, metadata, node)
        previous_response = {} # Initialize for refinement prompts

        while attempts < max_attempts:
            try:
                response = await openai.ChatCompletion.acreate(
                    engine=self.config.deployment_name,  # or deployment_id
                    messages=[{"role": "user", "content": prompt}],
                    functions=[self.docstring_function],
                    function_call={"name": "generate_docstring"},
                    max_tokens=2000,
                    temperature=0.3
                )

                assistant_message = response['choices'][0]['message']
                if assistant_message.get("function_call"):
                    function_args = assistant_message["function_call"]["arguments"]

                    try:
                        response_data = json.loads(function_args)
                        previous_response = response_data # Store for refinement
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON returned from function call: {e}. Arguments: {function_args}")
                        prompt = self._create_refinement_prompt(prompt, "Invalid JSON returned. Please ensure the 'generate_docstring' function returns valid JSON.", previous_response)
                        attempts += 1
                        continue

                    docstring_data = self._parse_ai_response(response_data)
                    is_valid, errors = self.docstring_processor.validate(docstring_data)

                    if is_valid:
                        formatted_docstring = self.docstring_processor.format(docstring_data)
                        processing_time = (datetime.now() - start_time).total_seconds()
                        return ProcessingResult(
                            content=formatted_docstring,
                            usage=response.get('usage', {}),
                            processing_time=processing_time
                        )
                    else:
                        prompt = self._create_refinement_prompt(prompt, f"The generated docstring has validation errors:\n{errors}", previous_response)
                        attempts += 1
                else:
                    logger.warning(f"OpenAI did not call the expected function. Response: {assistant_message}")
                    prompt = self._create_refinement_prompt(prompt, "Please call the 'generate_docstring' function to provide structured output.", previous_response)
                    attempts += 1

            except openai.error.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                attempts += 1
                await asyncio.sleep(2 ** attempts)  # Exponential backoff
            except Exception as e:
                logger.exception(f"Unexpected error during documentation generation: {e}")
                return None

        logger.error(f"Failed to generate valid docstring after {max_attempts} attempts.")
        return None
  
    def _create_function_calling_prompt(source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:
    """Creates the initial prompt, handling Markdown indentation correctly."""

    indented_source_code = "\n".join(["    " + line for line in source_code.split("\n")])

    prompt = (
        "You are a highly skilled Python documentation expert. Your task is to generate a comprehensive docstring for the given Python code by calling the 'generate_docstring' function.\n\n"
        + "The code to document is:\n"
        + "```python\n"
        + indented_source_code + "\n"
        + "```\n\n"
        + "Please analyze the code and provide the docstring by calling the 'generate_docstring' function with the appropriate arguments. Ensure that the function arguments adhere to the specified JSON schema. Provide the JSON in a single line."
    )
    return prompt

    def _create_refinement_prompt(self, original_prompt: str, error_message: str, previous_response: dict) -> str:
        """Creates a refinement prompt, handling previous responses and errors."""
    
        formatted_response = json.dumps(previous_response, indent=4) if previous_response else ""  # Format previous response if available
    
        prompt = (
            f"{error_message}\n\n"  # Include the specific error message
            + "Previous Response (if any):\n"
            + f"```json\n{formatted_response}\n```\n\n"  # Show previous JSON response
            + original_prompt  # Append the original prompt
        )
        return prompt
  
    def _parse_ai_response(self, response_data: Dict[str, Any]) -> DocstringData:
        """Parse the AI response into DocstringData. Handles schema validation."""
        try:
            validate(instance=response_data, schema=DOCSTRING_SCHEMA)
            return DocstringData(**response_data) # Use **kwargs for more concise initialization
        except JsonValidationError as e:
            error_path = "/".join(map(str, e.path))
            logger.error(f"Error validating AI response: {e.message} - Path: {error_path}, Instance: {response_data}")
            return DocstringData() # Return default on validation error
  
    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:  
        """Process source code to generate documentation."""  
        try:  
            if not source_code or not source_code.strip():  
                raise ValidationError("Empty source code provided")  
  
            if self.cache and cache_key:  
                try:  
                    cached_result = await self._check_cache(cache_key)  
                    if cached_result:  
                        return cached_result  
                except CacheError as e:  
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")  
  
            result = await self._generate_documentation(source_code, {})  
            if not result:  
                raise ProcessingError("Documentation generation failed")  
  
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'  
  
            if self.cache and cache_key:  
                await self._cache_result(cache_key, updated_code, result.content)  
  
            return updated_code, result.content  
  
        except ValidationError as ve:  
            logger.error(f"Validation error: {str(ve)}")  
            raise  
        except ProcessingError as pe:  
            logger.error(f"Processing error: {str(pe)}")  
            raise  
        except Exception as e:  
            logger.error(f"Process code failed: {str(e)}")  
            raise  
  
    async def _check_cache(self, cache_key: str) -> Optional[Tuple[str, str]]:  
        """  
        Check if result is available in cache.  
  
        Args:  
            cache_key: The cache key to check  
  
        Returns:  
            Optional[Tuple[str, str]]: Cached code and documentation if available  
        """  
        try:  
            if not self.cache:  
                return None  
  
            logger.debug(f"Checking cache for key: {cache_key}")  
            cached_data = await self.cache.get_cached_docstring(cache_key)  
  
            if cached_data and isinstance(cached_data, dict):  
                code = cached_data.get('code')  
                docs = cached_data.get('docs')  
                if code and docs:  
                    logger.info(f"Cache hit for key: {cache_key}")  
                    return code, docs  
  
            logger.debug(f"Cache miss for key: {cache_key}")  
            return None  
  
        except Exception as e:  
            logger.error(f"Cache check failed: {str(e)}")  
            raise CacheError(f"Failed to check cache: {str(e)}")  
  
    async def _cache_result(  
        self,  
        cache_key: str,  
        code: str,  
        documentation: str  
    ) -> None:  
        """  
        Cache the processing result.  
  
        Args:  
            cache_key: The cache key  
            code: The processed code  
            documentation: The generated documentation  
        """  
        try:  
            if not self.cache:  
                return  
  
            await self.cache.save_docstring(  
                cache_key,  
                {  
                    'code': code,  
                    'docs': documentation  
                }  
            )  
            logger.debug(f"Cached result for key: {cache_key}")  
  
        except Exception as e:  
            logger.error(f"Failed to cache result: {str(e)}")  
            logger.warning("Continuing without caching")  
```