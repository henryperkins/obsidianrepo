```python
"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API for docstring generation, handling token management,
caching, and raw response processing.
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
from api.token_management import TokenManager
from api.api_client import APIClient
from exceptions import ValidationError, ProcessingError, CacheError

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """Handles AI interactions for docstring generation via Azure OpenAI API."""

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

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = {
            "name": "generate_docstring",
            "description": "Generate structured Python docstrings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief one-line summary of the code element."
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of functionality."
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

    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """
        Process source code to generate docstrings.

        Args:
            source_code (str): The source code to process
            cache_key (Optional[str]): Cache key for storing results

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_source_code, raw_generated_content)
        """
        try:
            # Check cache first if enabled
            if self.cache and cache_key:
                cached_result = await self.cache.get_cached_docstring(cache_key)
                if cached_result:
                    return cached_result['code'], cached_result['docs']

            # Parse the code
            tree = ast.parse(source_code)
            self._add_parents(tree)

            # Process module-level docstring
            updated_source = source_code
            module_docs = await self._generate_docstring(source_code, tree)
            
            if module_docs:
                # Process each class and function
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        node_source = ast.unparse(node)
                        docs = await self._generate_docstring(node_source, node)
                        if docs:
                            updated_source = self._insert_docstring_in_source(updated_source, node, docs.content)
                            
            if module_docs and self.cache and cache_key:
                await self._cache_result(cache_key, updated_source, module_docs.content)
                
            return updated_source, module_docs.content if module_docs else ""

        except Exception as e:
            logger.error(f"Failed to process code: {e}")
            return None

    async def _generate_docstring(self, source_code: str, node: ast.AST) -> Optional[ProcessingResult]:
        """Generate a docstring for the given code."""
        max_attempts = 3
        attempts = 0
        start_time = datetime.now()

        while attempts < max_attempts:
            try:
                # Create the prompt
                prompt = self._create_docstring_prompt(source_code, node)
                
                # Make API call
                response = await self.client.process_request(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=1000,
                    functions=[self.docstring_function],
                    function_call={"name": "generate_docstring"}
                )

                if not response or not response[0]:
                    attempts += 1
                    continue

                # Extract the function call arguments
                function_args = response[0]['choices'][0]['message']['function_call']['arguments']
                
                try:
                    # Parse and validate the response
                    docstring_data = self._parse_response(function_args)
                    formatted_docstring = self.docstring_processor.format(docstring_data)
                    
                    # Calculate processing metrics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    return ProcessingResult(
                        content=formatted_docstring,
                        usage=response[1],
                        processing_time=processing_time
                    )

                except (json.JSONDecodeError, JsonValidationError) as e:
                    logger.error(f"Response validation failed: {e}")
                    attempts += 1

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                attempts += 1
                await asyncio.sleep(2 ** attempts)

        return None

    def _create_docstring_prompt(self, source_code: str, node: ast.AST) -> str:
        """Create the prompt for docstring generation."""
        node_type = type(node).__name__
        context = self._extract_context(node)
        
        prompt = (
            "Generate a comprehensive Python docstring following Google style guidelines. "
            f"This is a {node_type}. Include:\n"
            "1. A clear one-line summary\n"
            "2. Detailed description\n"
            "3. All parameters with types and descriptions\n"
            "4. Return value with type and description\n"
            "5. Exceptions that may be raised\n\n"
            f"Context:\n{context}\n\n"
            "Code to document:\n"
            "```python\n"
            f"{source_code}\n"
            "```"
        )
        return prompt

    def _extract_context(self, node: ast.AST) -> str:
        """Extract relevant context from the AST node."""
        context_parts = []
        
        if isinstance(node, ast.ClassDef):
            # Add base classes
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if bases:
                context_parts.append(f"Inherits from: {', '.join(bases)}")
                
            # Add class attributes
            attrs = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
            if attrs:
                context_parts.append(f"Class attributes: {', '.join(attrs)}")
                
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Add async status
            if isinstance(node, ast.AsyncFunctionDef):
                context_parts.append("This is an async function")
                
            # Add decorators
            decorators = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
            if decorators:
                context_parts.append(f"Decorators: {', '.join(decorators)}")
                
            # Add return type if present
            if node.returns:
                context_parts.append(f"Return type annotation: {ast.unparse(node.returns)}")
                
        return "\n".join(context_parts)

    def _parse_response(self, function_args: str) -> DocstringData:
        """Parse the AI response into DocstringData."""
        try:
            data = json.loads(function_args)
            return DocstringData(
                summary=data.get("summary", ""),
                description=data.get("description", ""),
                args=data.get("args", []),
                returns=data.get("returns", {"type": "None", "description": ""}),
                raises=data.get("raises", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise

    def _insert_docstring_in_source(self, source_code: str, node: ast.AST, docstring: str) -> str:
        """Insert or update a docstring in the source code."""
        try:
            lines = source_code.splitlines()
            start_line = node.lineno - 1
            
            # Find actual start of the definition (after decorators)
            while start_line < len(lines) and not lines[start_line].lstrip().startswith(('def ', 'class ')):
                start_line += 1
                
            if start_line >= len(lines):
                return source_code
                
            # Calculate indentation
            indentation = len(lines[start_line]) - len(lines[start_line].lstrip())
            indent = ' ' * indentation
            
            # Format docstring with proper indentation
            docstring_lines = [f'{indent}"""'] + [f'{indent}{line}' for line in docstring.splitlines()] + [f'{indent}"""']
            
            # Find and replace existing docstring or insert new one
            next_line = start_line + 1
            
            # Check for existing docstring
            if (next_line < len(lines) and lines[next_line].lstrip().startswith('"""')):
                # Find end of existing docstring
                end_line = next_line + 1
                while end_line < len(lines) and '"""' not in lines[end_line]:
                    end_line += 1
                if end_line < len(lines):
                    end_line += 1
                # Replace existing docstring
                lines[next_line:end_line] = docstring_lines
            else:
                # Insert new docstring
                lines = lines[:next_line] + docstring_lines + lines[next_line:]
                
            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"Failed to insert docstring: {e}")
            return source_code

    async def _cache_result(self, cache_key: str, code: str, docs: str) -> None:
        """Cache the processing result."""
        if not self.cache:
            return

        try:
            await self.cache.save_docstring(cache_key, {
                'code': code,
                'docs': docs
            })
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")

    def _add_parents(self, tree: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            logger.info("AI Interaction Handler closed successfully")
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