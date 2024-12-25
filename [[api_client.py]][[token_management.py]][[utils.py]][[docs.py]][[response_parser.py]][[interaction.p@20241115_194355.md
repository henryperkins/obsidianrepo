---
epoch: 1731721435117
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [api_client.py](api_client.py.md)[token_management.py](token_management.py.md)[utils.py](./utils.py.md)[docs.py](docs.py.md)[response_parser.py](response_parser.py.md)[[interaction.p@20241115_194355

**user**: [api_client.py](simple2/api_client.py.md)[token_management.py](simple2/token_management.py.md)[utils.py](simple2/utils.py.md)[docs.py](simple2/docs.py.md)[response_parser.py](simple2/response_parser.py.md)[interaction.py](interaction.py.md)[config.py](simple2/config.py.md)[monitoring.py](simple2/monitoring.py.md)[logger.py](simple2/logger.py.md) -- api_client and docs are too long, i need to refactor them into additional modules.  
[Timestamp: 2024/11/15 19:43:55]

**ai**: To refactor the `api_client.py` and `docs.py` modules into additional modules, we can break them down into smaller, more focused modules. This will improve maintainability and readability. Here's a suggested refactoring:

## Refactoring `api_client.py`

1. **api_client.py**: Keep the core client logic and initialization.
2. **prompt_builder.py**: Separate the logic for building prompts.
3. **response_handler.py**: Handle response parsing and validation.
4. **cache_manager.py**: Manage caching logic separately.

## Refactoring `docs.py`

1. **docstring_manager.py**: Manage docstring operations and updates.
2. **docstring_parser.py**: Handle parsing and extraction of docstrings.
3. **markdown_generator.py**: Generate markdown documentation.
4. **documentation_manager.py**: Manage the overall documentation generation process.

## Refactored Code

### `api_client.py`

```python
"""
Azure OpenAI API Client Module

This module provides a client for interacting with the Azure OpenAI API to generate
docstrings for Python functions. It handles API requests, retries, and error management.

Version: 1.2.0
Author: Development Team
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Tuple
from openai import AzureOpenAI, APIError
from logger import log_info, log_error, log_debug
from monitoring import SystemMonitor
from token_management import TokenManager
from cache_manager import CacheManager
from config import AzureOpenAIConfig, default_config
from prompt_builder import PromptBuilder
from response_handler import ResponseHandler

class TooManyRetriesError(Exception):
    """Raised when maximum retry attempts are exceeded."""
    pass

class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI to generate docstrings.
    """
    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        self.config = config or default_config
        if not self.config.validate():
            raise ValueError("Invalid Azure OpenAI configuration")
        
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.endpoint
        )

        self.token_manager = TokenManager(
            model="gpt-4o-2024-08-06",
            deployment_name=self.config.deployment_name
        )
        self.monitor = SystemMonitor()
        self.cache = CacheManager()
        self.prompt_builder = PromptBuilder()
        self.response_handler = ResponseHandler()
        self.current_retry = 0

        log_info("Azure OpenAI client initialized successfully")

    async def get_docstring(self, func_name: str, params: List[Tuple[str, str]], return_type: str, complexity_score: int, existing_docstring: str, decorators: List[str] = None, exceptions: List[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"{func_name}:{hash(str(params))}{hash(return_type)}"
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                token_usage = self.token_manager.calculate_usage(cached_response.get('usage', {}).get('prompt_tokens', 0), cached_response.get('usage', {}).get('completion_tokens', 0), cached=True)
                self.monitor.log_request(func_name, status="cache_hit", response_time=0, tokens=token_usage.total_tokens, estimated_cost=token_usage.estimated_cost)
                cached_response['usage']['estimated_cost'] = token_usage.estimated_cost
                return cached_response

            prompt = self.prompt_builder.create_enhanced_json_schema_prompt(func_name, params, return_type, complexity_score, existing_docstring, decorators, exceptions)
            initial_tokens = self.token_manager.estimate_tokens(prompt)
            log_debug(f"Initial token count for {func_name}: {initial_tokens}")

            is_valid, metrics, message = self.token_manager.validate_request(prompt, max_completion_tokens=max_tokens or self.config.max_tokens)
            if not is_valid:
                log_error(f"Token validation failed for {func_name}: {message}")
                return None

            optimized_prompt, token_usage = self.token_manager.optimize_prompt(prompt, max_tokens=max_tokens or self.config.max_tokens)
            log_debug(f"Token usage after optimization: {token_usage}")

            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.deployment_name,
                        messages=[
                            {"role": "system", "content": "You are a documentation expert. Generate clear, comprehensive docstrings following Google style guide."},
                            {"role": "user", "content": optimized_prompt}
                        ],
                        max_tokens=max_tokens or self.config.max_tokens,
                        temperature=temperature or self.config.temperature,
                        functions=[
                            {
                                "name": "generate_docstring",
                                "description": "Generate a structured docstring",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "docstring": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "complexity_score": {"type": "integer"},
                                        "changelog": {"type": "string"},
                                    },
                                    "required": ["docstring", "summary"],
                                },
                            }
                        ],
                        function_call={"name": "generate_docstring"},
                    )

                    final_token_usage = self.token_manager.calculate_usage(response.usage.prompt_tokens, response.usage.completion_tokens, cached=False)
                    parsed_response = self.response_handler.parse_json_response(response.choices[0].message.function_call.arguments)

                    if not parsed_response:
                        log_error(f"Failed to parse response for {func_name} (attempt {attempt + 1})")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay ** attempt)
                            continue
                        return None

                    if not self.response_handler.is_docstring_complete(parsed_response['docstring']):
                        log_error(f"Generated docstring incomplete for {func_name} (attempt {attempt + 1})")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay ** attempt)
                            continue
                        return None

                    final_response = {
                        "content": parsed_response,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "estimated_cost": final_token_usage.estimated_cost
                        }
                    }

                    await self.cache.save_docstring(cache_key, final_response, tags=[f"func:{func_name}", f"model:{self.config.deployment_name}", f"complexity:{complexity_score//10}0"])
                    self.monitor.log_request(func_name, status="success", response_time=time.time(), tokens=response.usage.total_tokens, estimated_cost=final_token_usage.estimated_cost)
                    return final_response

                except APIError as e:
                    log_error(f"API error for {func_name} (attempt {attempt + 1}): {e}")
                    if e.status_code == 429:
                        retry_after = int(e.headers.get('retry-after', self.config.retry_delay ** attempt))
                        await asyncio.sleep(retry_after)
                    elif attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay ** attempt)
                    else:
                        raise

                except Exception as e:
                    log_error(f"Unexpected error for {func_name} (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay ** attempt)
                    else:
                        raise

            return None

        except Exception as e:
            log_error(f"Error in get_docstring for {func_name}: {e}")
            self.monitor.log_request(func_name, status="error", response_time=time.time(), tokens=0, estimated_cost=0.0, error=str(e))
            return None

        finally:
            self.token_manager.reset_retry_counter()

    # Other methods remain unchanged
```

### `prompt_builder.py`

```python
"""
Prompt Builder Module

This module provides functionality to construct prompts for Azure OpenAI API requests.
"""

from typing import List, Tuple

class PromptBuilder:
    """Builds prompts for API requests."""

    def create_enhanced_json_schema_prompt(self, func_name: str, params: List[Tuple[str, str]], return_type: str, complexity_score: int, existing_docstring: str, decorators: List[str] = None, exceptions: List[str] = None) -> str:
        func_name = func_name.strip()
        param_details = ", ".join([f"{name}: {ptype}" for name, ptype in params]) if params else "None"
        return_type = return_type.strip() if return_type else "Any"
        complexity_score = max(0, min(complexity_score, 100))
        existing_docstring = existing_docstring.strip().replace('"', "'") if existing_docstring else "None"
        decorators_info = ", ".join(decorators) if decorators else "None"
        exceptions_info = ", ".join(exceptions) if exceptions else "None"

        prompt = f"""
        Generate a JSON object with the following fields:
        {{
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Google-style docstring including a Complexity section and examples.",
            "complexity_score": {complexity_score}
        }}

        Function: {func_name}
        Parameters: {param_details}
        Returns: {return_type}
        Decorators: {decorators_info}
        Exceptions: {exceptions_info}
        Existing docstring: {existing_docstring}
        """
        return prompt.strip()
```

### `response_handler.py`

```python
"""
Response Handler Module

This module provides functionality to parse and validate responses from Azure OpenAI.
"""

import json
from typing import Optional, Dict, Any
from logger import log_info, log_error, log_debug
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from schema import JSON_SCHEMA

class ResponseHandler:
    """Handles response parsing and validation."""

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        log_debug("Parsing JSON response.")
        try:
            response_json = json.loads(response)
            log_info("Successfully parsed Azure OpenAI response.")
            log_debug(f"Parsed JSON response: {response_json}")

            validate(instance=response_json, schema=JSON_SCHEMA)

            docstring = response_json.get("docstring", "")
            summary = response_json.get("summary", "")
            changelog = response_json.get("changelog", "Initial documentation")
            complexity_score = response_json.get("complexity_score", 0)

            log_debug(f"Extracted docstring: {docstring}")
            log_debug(f"Extracted summary: {summary}")
            log_debug(f"Extracted changelog: {changelog}")
            log_debug(f"Extracted complexity score: {complexity_score}")

            return {
                "docstring": docstring,
                "summary": summary,
                "changelog": changelog,
                "complexity_score": complexity_score
            }
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse response as JSON: {e}")
        except ValidationError as e:
            log_error(f"Response validation error: {e.message}")
        except Exception as e:
            log_error(f"Unexpected error during JSON response parsing: {e}")
        return None

    def is_docstring_complete(self, docstring: str) -> bool:
        return bool(docstring and docstring.strip())
```

### `cache_manager.py`

```python
"""
Cache Manager Module

This module provides functionality to manage caching for Azure OpenAI API requests.
"""

from typing import Dict, Any

class CacheManager:
    """Manages caching of API responses."""

    def get_cached_docstring(self, cache_key: str) -> Optional[Dict[str, Any]]:
        # Implement cache retrieval logic
        pass

    def save_docstring(self, cache_key: str, response: Dict[str, Any], tags: List[str]) -> None:
        # Implement cache saving logic
        pass
```

### `docstring_manager.py`

```python
"""
DocString Manager Module

This module manages docstring operations for source code files.
"""

import ast
from typing import List, Dict
import logging

class DocStringManager:
    """
    Manages docstring operations for source code files.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        logging.debug("DocStringManager initialized.")

    def insert_docstring(self, node: ast.FunctionDef, docstring: str) -> None:
        logging.debug(f"Inserting docstring into function '{node.name}'.")
        node.body.insert(0, ast.Expr(value=ast.Str(s=docstring)))

    def update_source_code(self, documentation_entries: List[Dict]) -> str:
        logging.debug("Updating source code with new docstrings.")
        for entry in documentation_entries:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.FunctionDef) and node.name == entry['function_name']:
                    self.insert_docstring(node, entry['docstring'])

        updated_code = ast.unparse(self.tree)
        logging.info("Source code updated with new docstrings.")
        return updated_code
```

### `docstring_parser.py`

```python
"""
DocString Parser Module

This module handles parsing and extraction of docstrings from Python source code.
"""

import ast
import logging
from typing import Optional, Dict, Any

class DocStringParser:
    """
    Handles parsing and extraction of docstrings from Python source code.
    """
    
    @staticmethod
    def extract_docstring(source_code: str) -> Optional[str]:
        logging.debug("Extracting module-level docstring.")
        try:
            tree = ast.parse(source_code)
            docstring = ast.get_docstring(tree)
            logging.debug(f"Extracted docstring: {docstring}")
            return docstring
        except Exception as e:
            logging.error(f"Failed to parse source code: {e}")
            return None

    @staticmethod
    def parse_function_docstring(func) -> Dict[str, Any]:
        logging.debug(f"Parsing docstring for function: {func.__name__}")
        doc = inspect.getdoc(func)
        if not doc:
            logging.debug("No docstring found.")
            return {}

        sections = {
            'description': '',
            'args': {},
            'returns': '',
            'raises': [],
            'examples': []
        }

        current_section = 'description'
        lines = doc.split('\n')

        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:'):
                current_section = 'args'
                continue
            elif line.lower().startswith('returns:'):
                current_section = 'returns'
                continue
            elif line.lower().startswith('raises:'):
                current_section = 'raises'
                continue
            elif line.lower().startswith('example'):
                current_section = 'examples'
                continue

            if current_section == 'description' and line:
                sections['description'] += line + ' '
            elif current_section == 'args' and line:
                if ':' in line:
                    param, desc = line.split(':', 1)
                    sections['args'][param.strip()] = desc.strip()
            elif current_section == 'returns' and line:
                sections['returns'] += line + ' '
            elif current_section == 'raises' and line:
                sections['raises'].append(line)
            elif current_section == 'examples' and line:
                sections['examples'].append(line)

        logging.debug(f"Parsed docstring sections: {sections}")
        return sections
```

### `markdown_generator.py`

```python
"""
Markdown Generator Module

This module generates markdown documentation from Python code elements.
"""

import logging
from typing import List

class MarkdownGenerator:
    """
    Generates markdown documentation from Python code elements.
    """

    def __init__(self):
        self.output = []
        logging.debug("MarkdownGenerator initialized.")

    def add_header(self, text: str, level: int = 1) -> None:
        logging.debug(f"Adding header: {text}")
        self.output.append(f"{'#' * level} {text}\n")

    def add_code_block(self, code: str, language: str = "python") -> None:
        logging.debug("Adding code block.")
        self.output.append(f"```{language}\n{code}\n```\n")

    def add_section(self, title: str, content: str) -> None:
        logging.debug(f"Adding section: {title}")
        self.output.append(f"### {title}\n\n{content}\n")

    def generate_markdown(self) -> str:
        logging.debug("Generating final markdown document.")
        return "\n".join(self.output)
```

### `documentation_manager.py`

```python
"""
Documentation Manager Module

This module manages the overall documentation generation process.
"""

import logging
from typing import Union, Dict, Optional
from pathlib import Path
from docstring_parser import DocStringParser
from docstring_manager import DocStringManager
from markdown_generator import MarkdownGenerator
from datetime import datetime

class DocumentationManager:
    """
    Manages the overall documentation generation process.
    """

    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parser = DocStringParser()
        self.manager = DocStringManager()
        self.markdown = MarkdownGenerator()
        self.logger = self._setup_logging()
        logging.debug("DocumentationManager initialized.")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('documentation_manager')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        logging.debug(f"Processing file: {file_path}")
        try:
            file_path = Path(file_path)
            if not file_path.exists() or file_path.suffix != '.py':
                self.logger.error(f"Invalid Python file: {file_path}")
                return None

            with open(file_path, 'r') as f:
                source = f.read()

            module_doc = self.parser.extract_docstring(source)
            
            self.markdown.add_header(f"Documentation for {file_path.name}")
            if module_doc:
                self.markdown.add_section("Module Description", module_doc)

            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._process_class(node)
                elif isinstance(node, ast.FunctionDef):
                    self._process_function(node)

            markdown = self.markdown.generate_markdown()
            logging.info(f"Generated markdown for file: {file_path}")
            return markdown

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _process_class(self, node: ast.ClassDef) -> None:
        logging.debug(f"Processing class: {node.name}")
        try:
            class_doc = ast.get_docstring(node)
            self.markdown.add_section(f"Class: {node.name}", class_doc if class_doc else "No documentation available")
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self._process_function(item, is_method=True, class_name=node.name)
        except Exception as e:
            self.logger.error(f"Error processing class {node.name}: {e}")

    def _process_function(self, node: ast.FunctionDef, is_method: bool = False, class_name: str = None) -> None:
        logging.debug(f"Processing function: {node.name}")
        try:
            func_doc = ast.get_docstring(node)
            section_title = f"{'Method' if is_method else 'Function'}: {node.name}"
            if is_method:
                section_title = f"Method: {class_name}.{node.name}"

            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"

            content = [
                f"```python\n{signature}\n```\n",
                func_doc if func_doc else "No documentation available"
            ]
            
            self.markdown.add_section(section_title, "\n".join(content))
        except Exception as e:
            self.logger.error(f"Error processing function {node.name}: {e}")

    def process_directory(self, directory_path: Union[str, Path]) -> Dict[str, str]:
        logging.debug(f"Processing directory: {directory_path}")
        directory_path = Path(directory_path)
        results = {}

        if not directory_path.is_dir():
            self.logger.error(f"Invalid directory path: {directory_path}")
            return results

        for file_path in directory_path.rglob("*.py"):
            try:
                doc_content = self.process_file(file_path)
                if doc_content:
                    results[str(file_path)] = doc_content
            except Exception as e:
                self.logger.error(f"Error processing directory {directory_path}: {e}")

        return results

    def save_documentation(self, content: str, output_file: Union[str, Path]) -> bool:
        logging.debug(f"Saving documentation to: {output_file}")
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Documentation saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving documentation: {e}")
            return False

    def generate_index(self, docs_map: Dict[str, str]) -> str:
        logging.debug("Generating documentation index.")
        index_content = [
            "# Documentation Index\n",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Files\n"
        ]

        for file_path in sorted(docs_map.keys()):
            rel_path = Path(file_path).name
            doc_path = Path(file_path).with_suffix('.md').name
            index_content.append(f"- [{rel_path}]({doc_path})")

        logging.info("Documentation index generated.")
        return "\n".join(index_content)
```

This refactoring breaks down the large modules into smaller, more manageable pieces, each responsible for a specific part of the functionality. This approach enhances readability and maintainability, making it easier to test and extend the code.  
[Timestamp: 2024/11/15 19:45:17]
