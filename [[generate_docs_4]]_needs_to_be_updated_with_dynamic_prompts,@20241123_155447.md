---
epoch: 1732398887487
modelKey: anthropic/claude-3.5-sonnet:beta|openrouterai
tags:
  - copilot-conversation
---

# [generate_docs_4](generate_docs_4.md)_needs_to_be_updated_with_dynamic_prompts,@20241123_155447

**user**: [generate docs 4](generate%20docs%204.md) needs to be updated with dynamic prompts, structured outputs and function calling This implementation is outlined in great detail here: [dynamic prompts with structured outputs](./dynamic%20prompts%20with%20structured%20outputs.md)  
[Timestamp: 2024/11/23 15:54:47]

**ai**: I'll help you update the `generate_docs_4.py` script with dynamic prompts, structured outputs, and function calling based on the implementation details provided. Here are the key changes needed:

1. First, let's define the function schema for docstring generation:

```python
DOCSTRING_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_docstring",
        "description": "Generate a Python docstring with structured information",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A brief summary of what the function does"},
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
                    "description": "List of arguments with their name, type, and description"
                },
                "returns": {"type": "string", "description": "Description of what the function returns"}
            },
            "required": ["summary", "args", "returns"]
        }
    }
}
```

2. Update the `generate_docstring` method in the `DocstringGenerator` class:

```python
async def generate_docstring(self, func_info: FunctionInfo, prompt: str, semaphore: asyncio.Semaphore):
    max_retries = 3
    backoff = 1

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            async with semaphore:
                response = await self.client.chat.completions.create(
                    model=self.deployment_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in Google Style."},
                        {"role": "user", "content": prompt}
                    ],
                    tools=[DOCSTRING_FUNCTION_SCHEMA],
                    tool_choice={"type": "function", "function": {"name": "generate_docstring"}},
                    max_tokens=self.args.max_tokens,
                    temperature=0.5
                )

                api_duration = time.time() - start_time

                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    if tool_call.function.name == "generate_docstring":
                        docstring_json = json.loads(tool_call.function.arguments)
                    else:
                        raise ValueError(f"Unexpected function call: {tool_call.function.name}")
                else:
                    raise ValueError("No function call in response")

                if not self.validate_docstring_json(docstring_json):
                    raise ValueError("Invalid docstring JSON format.")

                formatted_docstring = self.format_docstring_from_json(docstring_json)
                original_docstring = func_info.docstring
                func_info.docstring = formatted_docstring

                # Rest of the method remains the same...
```

3. Add a validation method for the docstring JSON:

```python
def validate_docstring_json(self, docstring_json: Dict[str, Any]) -> bool:
    """Validate that the docstring JSON matches our schema requirements."""
    try:
        required_keys = ["summary", "args", "returns"]
        if not all(key in docstring_json for key in required_keys):
            return False
            
        if not isinstance(docstring_json["args"], list):
            return False
            
        for arg in docstring_json["args"]:
            if not all(key in arg for key in ["name", "type", "description"]):
                return False
                
        return True
    except (KeyError, TypeError):
        return False
```

4. Update the prompt templates to be more dynamic:

```python
def _initialize_prompt_manager(self):
    prompt_manager = PromptManager()

    generate_template = PromptTemplate(
        name="generate_docstring",
        template=(
            "Generate a Python docstring for the following function:\n\n"
            "{function_signature}\n\n"
            "{additional_context}\n\n"
            "Follow these guidelines:\n"
            "1. Use Google Python Style Guide format\n"
            "2. Include a clear summary of the function's purpose\n"
            "3. Document all parameters with types and descriptions\n"
            "4. Specify the return value and type\n\n"
            "The response must be a JSON object matching the provided schema."
        ),
        variables=["function_signature", "additional_context"]
    )

    improve_template = PromptTemplate(
        name="improve_docstring",
        template=(
            "Improve the existing Python docstring for:\n\n"
            "Function: {function_signature}\n\n"
            "Current docstring:\n{existing_docstring}\n\n"
            "Additional context: {additional_context}\n\n"
            "Follow the same guidelines as before and return a JSON object."
        ),
        variables=["function_signature", "existing_docstring", "additional_context"]
    )

    prompt_manager.add_template(generate_template)
    prompt_manager.add_template(improve_template)
    return prompt_manager
```

5. Update the process_functions method to include additional context:

```python
async def process_functions(self, functions: List[FunctionInfo]):
    semaphore = asyncio.Semaphore(5)
    tasks = []

    for func in functions:
        # Get additional context from the function's source code or other metadata
        additional_context = self.extract_additional_context(func)
        
        if not self.args.force and func.docstring and is_docstring_complete(func.docstring):
            self.monitor.stats['cached'] += 1
            continue

        prompt_name = 'generate_docstring' if not func.docstring else 'improve_docstring'
        variables = {
            'function_signature': generate_function_signature(func),
            'additional_context': additional_context
        }
        
        if prompt_name == 'improve_docstring':
            variables['existing_docstring'] = func.docstring

        prompt = self.prompt_manager.get_prompt(prompt_name, **variables)
        tasks.append(self.generate_docstring(func, prompt, semaphore))

    if tasks:
        await asyncio.gather(*tasks)
```

6. Add a method to extract additional context:

```python
def extract_additional_context(self, func_info: FunctionInfo) -> str:
    """Extract additional context about the function to improve docstring generation."""
    context_parts = []
    
    # Add class context if it's a method
    if func_info.is_method:
        context_parts.append(f"This is a method of class '{func_info.parent_class}'")
    
    # Add decorator information
    if func_info.decorators:
        context_parts.append(f"Decorators: {', '.join(func_info.decorators)}")
    
    # Add complexity information
    if func_info.complexity > 0:
        context_parts.append(f"Function complexity score: {func_info.complexity}")
    
    # Add async information
    if func_info.is_async:
        context_parts.append("This is an async function")
    
    return " | ".join(context_parts) if context_parts else "No additional context available."
```

These changes implement:
- Structured output using function calling
- Dynamic prompt generation with additional context
- Improved validation of API responses
- Better error handling for malformed responses

The script will now generate more reliable and structured docstrings while maintaining the existing functionality. The function calling approach ensures that the API returns properly formatted JSON, reducing the likelihood of parsing errors.  
[Timestamp: 2024/11/23 15:57:27]

**user**: Provide the updated, complete module.  
[Timestamp: 2024/11/23 15:58:12]

**ai**: Here's the updated module with dynamic prompts, structured outputs, and function calling integrated:

```python
import ast
import sys
import os
import logging
import asyncio
import tiktoken
import time
import argparse
import json
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from hashlib import md5
from dotenv import load_dotenv

import diskcache
from radon.complexity import cc_visit
from openai import AsyncAzureOpenAI, OpenAIError
import pathspec
import tempfile
import shutil
import subprocess

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

@dataclass
class FunctionInfo:
    name: str
    parameters: List[Tuple[str, Optional[str]]]
    return_type: Optional[str]
    docstring: Optional[str]
    node: ast.AST
    complexity: int = field(default=0)
    code_hash: str = field(default='')
    is_method: bool = field(default=False)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = field(default=False)
    parent_class: Optional[str] = field(default=None)
    file_path: str = field(default='')

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[FunctionInfo]
    docstring: Optional[str] = ''
    complexity: int = field(default=0)

@dataclass
class VariableInfo:
    name: str
    type: Optional[str]
    value: Optional[str]

@dataclass
class ModuleInfo:
    name: str
    file_path: str
    description: Optional[str] = ''
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)

@dataclass
class PromptTemplate:
    name: str
    template: str
    variables: List[str]

class PromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def add_template(self, template: PromptTemplate):
        self.templates[template.name] = template

    def get_prompt(self, name: str, **kwargs) -> str:
        template = self.templates.get(name)
        if not template:
            raise ValueError(f"Template {name} not found")
        missing_vars = [var for var in template.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing variables for template {name}: {missing_vars}")
        return template.template.format(**kwargs)

class DocstringMonitor:
    def __init__(self):
        self.stats = {
            'generated': 0,
            'improved': 0,
            'cached': 0,
            'failed': 0,
            'api_call_count': 0,
            'api_call_duration': 0.0,
            'token_usage': 0,
            'processing_times': [],
        }

    def log_event(self, event_type: str, data: Dict[str, Any]):
        log_data = {'event': event_type, **data}
        logger.info(json.dumps(log_data))
        if event_type in self.stats:
            self.stats[event_type] += 1

class ContentFilter:
    """Class to handle content safety checks (currently a placeholder)."""

    async def is_content_safe(self, content: str) -> bool:
        """Checks if the content is safe (always returns True for now)."""
        return True

class DocstringGenerator:
    def __init__(self, args, monitor: DocstringMonitor):
        self.args = args
        self.monitor = monitor
        self.cache = diskcache.Cache(args.cache_location)
        self.prompt_manager = self._initialize_prompt_manager()
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.content_filter = ContentFilter()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        azure_endpoint = self.args.api_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = self.args.api_key or os.getenv('AZURE_OPENAI_KEY')
        api_version = self.args.api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        deployment_id = self.args.api_deployment_id or os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')
        model_name = os.getenv('AZURE_OPENAI_MODEL_NAME', 'gpt-4')

        if not api_key or not azure_endpoint or not deployment_id:
            logger.error(json.dumps({'event': 'error', 'error': 'Azure OpenAI API key, endpoint, and deployment ID must be provided.'}))
            sys.exit(1)

        try:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
        except Exception as e:
            logger.error(json.dumps({'event': 'error', 'error': f'Failed to initialize Azure OpenAI client: {e}'}))
            sys.exit(1)

        self.deployment_id = deployment_id
        self.model_name = model_name

    def _initialize_prompt_manager(self):
        prompt_manager = PromptManager()

        generate_template = PromptTemplate(
            name="generate_docstring",
            template=(
                "Generate a Python docstring for the following function:\n\n"
                "{function_signature}\n\n"
                "{additional_context}\n\n"
                "Follow these guidelines:\n"
                "1. Use Google Python Style Guide format\n"
                "2. Include a clear summary of the function's purpose\n"
                "3. Document all parameters with types and descriptions\n"
                "4. Specify the return value and type\n\n"
                "The response must be a JSON object matching the provided schema."
            ),
            variables=["function_signature", "additional_context"]
        )

        improve_template = PromptTemplate(
            name="improve_docstring",
            template=(
                "Improve the existing Python docstring for:\n\n"
                "Function: {function_signature}\n\n"
                "Current docstring:\n{existing_docstring}\n\n"
                "Additional context: {additional_context}\n\n"
                "Follow the same guidelines as before and return a JSON object."
            ),
            variables=["function_signature", "existing_docstring", "additional_context"]
        )

        prompt_manager.add_template(generate_template)
        prompt_manager.add_template(improve_template)
        return prompt_manager

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(prompt)
        if len(tokens) <= max_tokens:
            return prompt
        return self.encoding.decode(tokens[:max_tokens])

    async def generate_docstring(self, func_info: FunctionInfo, prompt: str, semaphore: asyncio.Semaphore):
        max_retries = 3
        backoff = 1

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                async with semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.deployment_id,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in Google Style."},
                            {"role": "user", "content": prompt}
                        ],
                        tools=[DOCSTRING_FUNCTION_SCHEMA],
                        tool_choice={"type": "function", "function": {"name": "generate_docstring"}},
                        max_tokens=self.args.max_tokens,
                        temperature=0.5
                    )

                    api_duration = time.time() - start_time

                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        if tool_call.function.name == "generate_docstring":
                            docstring_json = json.loads(tool_call.function.arguments)
                        else:
                            raise ValueError(f"Unexpected function call: {tool_call.function.name}")
                    else:
                        raise ValueError("No function call in response")

                    if not self.validate_docstring_json(docstring_json):
                        raise ValueError("Invalid docstring JSON format.")

                    formatted_docstring = self.format_docstring_from_json(docstring_json)
                    original_docstring = func_info.docstring
                    func_info.docstring = formatted_docstring

                    if original_docstring is None:
                        event_type = 'generated'
                        self.monitor.stats['generated'] += 1
                    else:
                        event_type = 'improved'
                        self.monitor.stats['improved'] += 1
                    self.monitor.log_event(event_type, {'function': func_info.name})

                    if not await self.content_filter.is_content_safe(formatted_docstring):
                        logger.warning(json.dumps({'event': 'warning', 'message': f"Potentially unsafe content detected for {func_info.name}. Docstring discarded."}))
                        func_info.docstring = ''
                        self.monitor.stats['failed'] += 1
                        return

                    total_tokens = response.usage.total_tokens
                    self.monitor.stats['api_call_count'] += 1
                    self.monitor.stats['api_call_duration'] += api_duration
                    self.monitor.stats['token_usage'] += total_tokens
                    self.monitor.log_event('api_call', {'function': func_info.name, 'duration': api_duration, 'tokens': total_tokens})
                    return

            except (OpenAIError, ValueError) as e:
                self.monitor.log_event('failed', {'function': func_info.name, 'error': str(e)})
                self.monitor.stats['failed'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    func_info.docstring = ''
                    return

    def validate_docstring_json(self, docstring_json: Dict[str, Any]) -> bool:
        """Validate that the docstring JSON matches our schema requirements."""
        try:
            required_keys = ["summary", "args", "returns"]
            if not all(key in docstring_json for key in required_keys):
                return False
                
            if not isinstance(docstring_json["args"], list):
                return False
                
            for arg in docstring_json["args"]:
                if not all(key in arg for key in ["name", "type", "description"]):
                    return False
                    
            return True
        except (KeyError, TypeError):
            return False

    def format_docstring_from_json(self, docstring_json: Dict[str, Any]) -> str:
        lines = [docstring_json.get('summary', '').strip(), '']
        if docstring_json.get('args'):
            lines.append("Args:")
            for arg in docstring_json['args']:
                arg_name = arg.get('name', 'arg')
                arg_type = arg.get('type', 'Any')
                arg_desc = arg.get('description', '')
                lines.append(f"    {arg_name} ({arg_type}): {arg_desc}")
            lines.append('')
        if docstring_json.get('returns'):
            lines.append("Returns:")
            lines.append(f"    {docstring_json['returns']}")
        return '\n'.join(lines)

    async def process_functions(self, functions: List[FunctionInfo]):
        semaphore = asyncio.Semaphore(5)
        tasks = []

        for func in functions:
            additional_context = self.extract_additional_context(func)
            
            if not self.args.force and func.docstring and is_docstring_complete(func.docstring):
                self.monitor.stats['cached'] += 1
                continue

            prompt_name = 'generate_docstring' if not func.docstring else 'improve_docstring'
            variables = {
                'function_signature': generate_function_signature(func),
                'additional_context': additional_context
            }
            
            if prompt_name == 'improve_docstring':
                variables['existing_docstring'] = func.docstring

            prompt = self.prompt_manager.get_prompt(prompt_name, **variables)
            tasks.append(self.generate_docstring(func, prompt, semaphore))

        if tasks:
            await asyncio.gather(*tasks)

    def extract_additional_context(self, func_info: FunctionInfo) -> str:
        """Extract additional context about the function to improve docstring generation."""
        context_parts = []
        
        if func_info.is_method:
            context_parts.append(f"This is a method of class '{func_info.parent_class}'")
        
        if func_info.decorators:
            context_parts.append(f"Decorators: {', '.join(func_info.decorators)}")
        
        if func_info.complexity > 0:
            context_parts.append(f"Function complexity score: {func_info.complexity}")
        
        if func_info.is_async:
            context_parts.append("This is an async function")
        
        return " | ".join(context_parts) if context_parts else "No additional context available."

def is_docstring_complete(docstring: str) -> bool:
    """Checks for basic docstring elements."""
    if not docstring:
        return False
    required_sections = ["Args:", "Returns:"]
    return all(section in docstring for section in required_sections)

def generate_function_signature(func_info: FunctionInfo) -> str:
    signature = f"{'async ' if func_info.is_async else ''}def {func_info.name}("
    params = [f"{name}: {type_}" if type_ else name for name, type_ in func_info.parameters]
    signature += ", ".join(params) + ")"
    if func_info.return_type:
        signature += f" -> {func_info.return_type}"
    return signature + ":"

def extract_functions(tree: ast.AST, source_code: str, file_path: str) -> List[FunctionInfo]:
    collected_functions = []

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self, source_code: str, file_path: str):
            self.source_code = source_code
            self.current_class = None
            self.file_path = file_path
            super().__init__()

        def visit_ClassDef(self, node: ast.ClassDef):
            previous_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = previous_class

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._collect_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._collect_function(node)

        def _collect_function(self, node):
            is_method = self.current_class is not None
            decorators = [ast.unparse(dec) for dec in node.decorator_list]
            is_async = isinstance(node, ast.AsyncFunctionDef)
            parameters = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else None) for arg in node.args.args]
            return_type = ast.unparse(node.returns) if node.returns else None
            docstring = ast.get_docstring(node)
            code_segment = ast.get_source_segment(self.source_code, node)
            code_hash = md5(code_segment.encode('utf-8')).hexdigest() if code_segment else ''

            func_info = FunctionInfo(
                name=node.name,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                node=node,
                complexity=0,
                code_hash=code_hash,
                is_method=is_method,
                decorators=decorators,
                is_async=is_async,
                parent_class=self.current_class,
                file_path=self.file_path
            )
            collected_functions.append(func_info)

    visitor = FunctionVisitor(source_code, file_path)
    visitor.visit(tree)
    return collected_functions

def calculate_complexity(node: ast.AST, metric: str = 'basic') -> int:
    if metric == 'basic':
        complexity = 0
        control_flow_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)
        for child in ast.walk(node):
            if isinstance(child, control_flow_nodes):
                complexity += 1
        max_depth = [0]

        def _visit(node, depth):
            if isinstance(node, control_flow_nodes):
                depth += 1
                max_depth[0] = max(max_depth[0], depth)
            for child in ast.iter_child_nodes(node):
                _visit(child, depth)

        _visit(node, 0)
        complexity += max_depth[0]
        return complexity
    elif metric == 'cyclomatic':
        code = ast.unparse(node)
        try:
            return cc_visit(code)[0].complexity
        except (IndexError, AttributeError):
            return 0
    else:
        logger.warning(json.dumps({'event': 'warning', 'message': f"Unknown complexity metric '{metric}'. Using 'basic'."}))
        return calculate_complexity(node, 'basic')

def extract_module_info(python_file: str, tree: ast.AST, source_code: str) -> ModuleInfo:
    module_docstring = ast.get_docstring(tree)
    module_name = os.path.basename(python_file)
    module_info = ModuleInfo(
        name=module_name,
        file_path=python_file,
        description=module_docstring or '',
    )

    class_defs = []
    function_defs = []
    variable_defs = []

    class ClassAndFunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None

        def visit_ClassDef(self, node: ast.ClassDef):
            class_name = node.name
            bases = [ast.unparse(base) for base in node.bases]
            previous_class = self.current_class
            self.current_class = class_name

            class_methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = extract_function_info(child, source_code, python_file, self.current_class)
                    func_info.complexity = calculate_complexity(child)
                    class_methods.append(func_info)

            class_docstring = ast.get_docstring(node)
            class_info = ClassInfo(
                name=class_name,
                bases=bases,
                methods=class_methods,
                docstring=class_docstring or '',
            )
            module_info.classes.append(class_info)

            self.current_class = previous_class
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if self.current_class is None:
                func_info = extract_function_info(node, source_code, python_file)
                func_info.complexity = calculate_complexity(node)
                module_info.functions.append(func_info)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            if self.current_class is None:
                func_info = extract_function_info(node, source_code, python_file)
                func_info.is_async = True
                func_info.complexity = calculate_complexity(node)
                module_info.functions.append(func_info)

        def visit_Assign(self, node: ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                var_value = ast.unparse(node.value)
                var_type = None
                variable_info = VariableInfo(
                    name=var_name,
                    type=var_type,
                    value=var_value,
                )
                module_info.variables.append(variable_info)

    visitor = ClassAndFunctionVisitor()
    visitor.visit(tree)
    return module_info

def extract_function_info(node, source_code: str, file_path: str, parent_class: Optional[str] = None) -> FunctionInfo:
    is_method = parent_class is not None
    decorators = [ast.unparse(dec) for dec in node.decorator_list]
    is_async = isinstance(node, ast.AsyncFunctionDef)
    parameters = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else None) for arg in node.args.args]
    return_type = ast.unparse(node.returns) if node.returns else None
    docstring = ast.get_docstring(node)
    code_segment = ast.get_source_segment(source_code, node)
    code_hash = md5(code_segment.encode('utf-8')).hexdigest() if code_segment else ''

    func_info = FunctionInfo(
        name=node.name,
        parameters=parameters,
        return_type=return_type,
        docstring=docstring,
        node=node,
        complexity=0,
        code_hash=code_hash,
        is_method=is_method,
        decorators=decorators,
        is_async=is_async,
        parent_class=parent_class,
        file_path=file_path,
    )
    return func_info

def generate_markdown(modules: List[ModuleInfo], output_file: str):
    md_lines = []
    for module in modules:
        md_lines.append(f"# Module: {module.name}")
        md_lines.append("")
        md_lines.append("## Overview")
        md_lines.append(f"**File:** `{module.file_path}`")
        module_description = module.description.strip().split('\n')[0] if module.description else ''
        md_lines.append(f"**Description:** {module_description}")
        md_lines.append("")

        if module.classes:
            md_lines.append("## Classes")
            md_lines.append("")
            md_lines.append("| Class | Inherits From | Complexity Score* |")
            md_lines.append("|-------|---------------|------------------|")
            for cls in module.classes:
                bases = ', '.join(cls.bases) if cls.bases else '-'
                complexity = '-'  # Complexity at class level can be defined if needed
                md_lines.append(f"| `{cls.name}` | `{bases}` | {complexity} |")
            md_lines.append("")

            md_lines.append("### Class Methods")
            md_lines.append("")
            md_lines.append("| Class | Method | Parameters | Returns | Complexity Score* |")
            md_lines.append("|-------|--------|------------|---------|------------------|")
            for cls in module.classes:
                for method in cls.methods:
                    params = f"({', '.join([f'{name}: {type_}' if type_ else name for name, type_ in method.parameters])})"
                    returns = method.return_type or '-'
                    complexity = f"{method.complexity} {'⚠️' if method.complexity > 10 else ''}"
                    md_lines.append(f"| `{cls.name}` | `{method.name}` | `{params}` | `{returns}` | {complexity} |")
            md_lines.append("")

        if module.functions:
            md_lines.append("## Functions")
            md_lines.append("")
            md_lines.append("| Function | Parameters | Returns | Complexity Score* |")
            md_lines.append("|----------|------------|---------|------------------|")
            for func in module.functions:
                params = f"({', '.join([f'{name}: {type_}' if type_ else name for name, type_ in func.parameters])})"
                returns = func.return_type or '-'
                complexity = f"{func.complexity} {'⚠️' if func.complexity > 10 else ''}"
                md_lines.append(f"| `{func.name}` | `{params}` | `{returns}` | {complexity} |")
            md_lines.append("")

        if module.variables:
            md_lines.append("## Constants and Variables")
            md_lines.append("")
            md_lines.append("| Name | Type | Value |")
            md_lines.append("|------|------|-------|")
            for var in module.variables:
                var_type = var.type or '-'
                var_value = var.value or '-'
                md_lines.append(f"| `{var.name}` | `{var_type}` | `{var_value}` |")
            md_lines.append("")

        md_lines.append("## Recent Changes")
        md_lines.append("- [YYYY-MM-DD] Added feature X")
        md_lines.append("- [YYYY-MM-DD] Modified method Y")
        md_lines.append("- [YYYY-MM-DD] Fixed bug Z")
        md_lines.append("")

        md_lines.append("## Source Code")
        md_lines.append("```python")
        try:
            with open(module.file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            md_lines.append(source_code)
        except Exception as e:
            logger.error(json.dumps({'event': 'error', 'error': f"Error reading {module.file_path}: {e}"}))
        md_lines.append("```")
        md_lines.append("")

    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        logger.info(json.dumps({'event': 'file_written', 'file': output_file}))
    except Exception as e:
        logger.error(json.dumps({'event': 'error', 'error': f"Error writing to {output_file}: {e}"}))

def read_file_with_fallback_encoding(file_path: str) -> Optional[str]:
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            pass
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Markdown documentation for Python code.")
    parser.add_argument('input', nargs='+', help="Python file(s), directory, or GitHub repository URL(s).")
    parser.add_argument('-o', '--output', default='documentation.md', help="Output Markdown file (default: documentation.md).")
    parser.add_argument('--api-key', help="Azure OpenAI API key.")
    parser.add_argument('--api-endpoint', help="Azure OpenAI API endpoint.")
    parser.add_argument('--api-deployment-id', help="Azure OpenAI deployment ID.")
    parser.add_argument('--api-version', default='2024-02-15-preview', help="Azure OpenAI API version (default: 2024-02-15-preview).")
    parser.add_argument('--max-tokens', type=int, default=300, help="Max tokens per API call (default: 300).")
    parser.add_argument('--complexity-metric', choices=['basic', 'cyclomatic'], default='basic', help="Complexity metric (basic or cyclomatic).")
    parser.add_argument('--cache-location', default='.docstring_cache', help="Cache directory (default: .docstring_cache).")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity level (-v, -vv).")
    parser.add_argument('--force', action='store_true', help="Force regeneration, bypassing cache.")
    return parser.parse_args()

def setup_logging(verbosity_level: int):
    log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    index = min(len(log_levels) - 1, max(0, 2 - verbosity_level))
    log_level = log_levels[index]
    logging.getLogger().setLevel(log_level)

def load_gitignore(repo_path: str) -> pathspec.PathSpec:
    gitignore_path = os.path.join(repo_path, '.gitignore')
    if not os.path.isfile(gitignore_path):
        logger.info(json.dumps({'event': 'info', 'message': "No .gitignore file found."}))
        return pathspec.PathSpec.from_lines('gitwildmatch', [])
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        gitignore_content = f.readlines()
    return pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content)

def is_github_url(url):
    return url.startswith('https://github.com/') or url.startswith('git@github.com:')

def find_repo_root(path):
    current_path = os.path.abspath(path)
    while True:
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            break
        current_path = parent
    return None

def process_path(path, args, monitor, temp_dirs):
    modules_info = []
    if is_github_url(path):
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        try:
            subprocess.run(['git', 'clone', path, temp_dir], check=True)
            repo_root = temp_dir
            path = temp_dir
        except subprocess.CalledProcessError as e:
            logger.error(json.dumps({'event': 'error', 'error': f"Failed to clone {path}: {e}"}))
            return []
    else:
        repo_root = find_repo_root(path)
        if not repo_root:
            repo_root = os.getcwd()
            logger.info(json.dumps({'event': 'info', 'message': "Repository root not found. Using current working directory as root."}))

    gitignore_spec = load_gitignore(repo_root)
    excluded_dirs = {'venv/', 'env/', '.venv/', '.env/', 'virtualenv/', 'venv3/', 'pyenv/', 'pipenv/'}

    python_files = []
    if os.path.isfile(path) and path.endswith('.py'):
        full_path = os.path.abspath(path)
        rel_path = os.path.relpath(full_path, repo_root)
        if not gitignore_spec.match_file(rel_path):
            if not any(rel_path.startswith(excl_dir) for excl_dir in excluded_dirs):
                python_files.append(full_path)
                logger.debug(json.dumps({'event': 'info', 'message': f"Added {full_path} to processing list."}))
            else:
                logger.debug(json.dumps({'event': 'info', 'message': f"Ignored {full_path} as it's within an excluded directory."}))
        else:
            logger.debug(json.dumps({'event': 'info', 'message': f"Ignored {full_path} as per .gitignore."}))
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            rel_root = os.path.relpath(root, repo_root)
            dirs[:] = [d for d in dirs if not any(os.path.join(rel_root, d).startswith(excl_dir) for excl_dir in excluded_dirs)]
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_root)
                    if not gitignore_spec.match_file(rel_path):
                        if not any(rel_path.startswith(excl_dir) for excl_dir in excluded_dirs):
                            python_files.append(full_path)
                            logger.debug(json.dumps({'event': 'info', 'message': f"Added {full_path} to processing list."}))
                        else:
                            logger.debug(json.dumps({'event': 'info', 'message': f"Ignored {full_path} as it's within an excluded directory."}))
                    else:
                        logger.debug(json.dumps({'event': 'info', 'message': f"Ignored {full_path} as per .gitignore."}))
    else:
        logger.warning(json.dumps({'event': 'warning', 'message': f"Invalid path: {path}"}))
        return []

    if not python_files:
        logger.warning(json.dumps({'event': 'warning', 'message': f"No Python files found to process in {path}."}))
        return []

    for python_file in python_files:
        source_code = read_file_with_fallback_encoding(python_file)
        if not source_code:
            logger.error(json.dumps({'event': 'error', 'error': f"Could not decode {python_file}"}))
            continue

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(json.dumps({'event': 'error', 'error': f"Syntax error in {python_file}: {e}"}))
            continue

        module_info = extract_module_info(python_file, tree, source_code)
        for cls in module_info.classes:
            for method in cls.methods:
                method.complexity = calculate_complexity(method.node, args.complexity_metric)
        for func in module_info.functions:
            func.complexity = calculate_complexity(func.node, args.complexity_metric)
        modules_info.append(module_info)
        monitor.log_event('file_processed', {'file': python_file, 'functions': len(module_info.functions)})

    return modules_info

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    monitor = DocstringMonitor()

    temp_dirs = []
    modules_info = []

    try:
        for path in args.input:
            modules = process_path(path, args, monitor, temp_dirs)
            modules_info.extend(modules)
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir)

    if not modules_info:
        logger.warning(json.dumps({'event': 'warning', 'message': "No modules found to process."}))
        return

    all_functions = []
    for module in modules_info:
        for cls in module.classes:
            all_functions.extend(cls.methods)
        all_functions.extend(module.functions)

    if not all_functions:
        logger.warning(json.dumps({'event': 'warning', 'message': "No functions found to process."}))
        return

    docstring_generator = DocstringGenerator(args, monitor)
    try:
        asyncio.run(docstring_generator.process_functions(all_functions))
    except KeyboardInterrupt:
        logger.error(json.dumps({'event': 'error', 'error': "Script interrupted by user."}))
        sys.exit(1)

    generate_markdown(modules_info, args.output)

    monitor.stats['total_duration'] = sum(monitor.stats.get('processing_times', []))
    monitor.log_event('summary', {'stats': monitor.stats})
    docstring_generator.cache.close()

if __name__ == "__main__":
    main()
```

## Key Changes Implemented

- **Dynamic Prompts**: The prompts are now dynamically generated with additional context about the function, including class membership, decorators, complexity, and async status.
- **Structured Outputs**: The API calls now use a structured output schema to ensure the responses are in JSON format, which is validated and parsed.
- **Function Calling**: The script uses function calling to ensure the API returns data in the specified structure.

This updated module should provide more reliable and structured docstring generation using the Azure OpenAI API.  
[Timestamp: 2024/11/23 15:59:53]

**user**: {"event": "improved", "function": "add_change"}  
{"event": "api_call", "function": "add_change", "duration": 53.874749183654785, "tokens": 212}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/ai_interaction.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/ai_interaction.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/exceptions.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/exceptions.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/main.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/main.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/extract/extraction_manager.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/extract/extraction_manager.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/extract/functions.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/extract/functions.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/extract/base.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/extract/base.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/extract/**init**.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/extract/**init**.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/extract/classes.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/extract/classes.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/api/response_parser.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/api/response_parser.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/api/api_client.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/api/api_client.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/api/token_management.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/api/token_management.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/api/**init**.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/api/**init**.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/api/base_client.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/api/base_client.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/logger.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/logger.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/utils.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/utils.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/metrics.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/metrics.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/config.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/config.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/cache.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/cache.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/**init**.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/**init**.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/core/monitoring.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/core/monitoring.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/docs/docs.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/docs/docs.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/docs/docstring_utils.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/docs/docstring_utils.py'"}  
{"event": "error", "error": "Error reading /tmp/tmptiz1cu7l/docs/markdown_generator.py: [Errno 2] No such file or directory: '/tmp/tmptiz1cu7l/docs/markdown_generator.py'"}  
{"event": "file_written", "file": "documentation.md"}  
{"event": "summary", "stats": {"generated": 28, "improved": 492, "cached": 135, "failed": 0, "api_call_count": 260, "api_call_duration": 7164.9632205963135, "token_usage": 65571, "processing_times": [], "total_duration": 0}}  
[Timestamp: 2024/11/23 16:02:53]

**ai**: It looks like there are some errors occurring when trying to read files from a temporary directory (`/tmp/tmptiz1cu7l/`). This suggests that either:

1. The GitHub repository clone operation failed
2. The temporary directory was cleaned up before file processing completed
3. The file paths are incorrect

Let's fix these issues by:

1. Adding better error handling for the GitHub clone operation:

```python
def process_path(path, args, monitor, temp_dirs):
    modules_info = []
    if is_github_url(path):
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        try:
            logger.info(json.dumps({'event': 'info', 'message': f"Cloning repository {path} to {temp_dir}"}))
            result = subprocess.run(
                ['git', 'clone', path, temp_dir], 
                check=True,
                capture_output=True,
                text=True
            )
            repo_root = temp_dir
            path = temp_dir
            logger.info(json.dumps({'event': 'success', 'message': "Repository cloned successfully"}))
        except subprocess.CalledProcessError as e:
            logger.error(json.dumps({
                'event': 'error', 
                'error': f"Failed to clone {path}: {e}",
                'stdout': e.stdout,
                'stderr': e.stderr
            }))
            return []
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Unexpected error while cloning {path}: {str(e)}"
            }))
            return []
```

2. Improving the file reading process:

```python
def read_file_with_fallback_encoding(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        logger.error(json.dumps({
            'event': 'error',
            'error': f"File not found: {file_path}"
        }))
        return None
        
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.debug(json.dumps({
                    'event': 'info',
                    'message': f"Successfully read {file_path} with {encoding} encoding"
                }))
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Error reading {file_path}: {str(e)}"
            }))
            return None
    return None
```

3. Adding cleanup protection:

```python
def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    monitor = DocstringMonitor()

    temp_dirs = []
    modules_info = []

    try:
        for path in args.input:
            modules = process_path(path, args, monitor, temp_dirs)
            if modules:  # Only process if we got valid modules
                modules_info.extend(modules)
            else:
                logger.warning(json.dumps({
                    'event': 'warning',
                    'message': f"No valid modules found in {path}"
                }))
    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Error processing paths: {str(e)}"
        }))
    finally:
        # Clean up temp directories
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.debug(json.dumps({
                        'event': 'cleanup',
                        'message': f"Removed temporary directory: {temp_dir}"
                    }))
            except Exception as e:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Failed to cleanup temporary directory {temp_dir}: {str(e)}"
                }))
```

4. Add a function to verify repository structure:

```python
def verify_repository_structure(repo_path: str) -> bool:
    """Verify that the cloned repository has the expected structure."""
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(repo_path):
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Repository path {repo_path} is not a directory"
            }))
            return False

        # List all files and directories
        contents = os.listdir(repo_path)
        logger.debug(json.dumps({
            'event': 'info',
            'message': f"Repository contents: {contents}"
        }))

        # Check for Python files
        python_files = [f for f in contents if f.endswith('.py')]
        if not python_files:
            logger.warning(json.dumps({
                'event': 'warning',
                'message': f"No Python files found in repository root"
            }))

        return True
    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Error verifying repository structure: {str(e)}"
        }))
        return False
```

5. Update the process_path function to use the verification:

```python
def process_path(path, args, monitor, temp_dirs):
    modules_info = []
    if is_github_url(path):
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        try:
            logger.info(json.dumps({'event': 'info', 'message': f"Cloning repository {path} to {temp_dir}"}))
            result = subprocess.run(['git', 'clone', path, temp_dir], check=True, capture_output=True, text=True)
            
            if not verify_repository_structure(temp_dir):
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Invalid repository structure after cloning {path}"
                }))
                return []
                
            repo_root = temp_dir
            path = temp_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(json.dumps({
                'event': 'error', 
                'error': f"Failed to clone {path}: {e}",
                'stdout': e.stdout,
                'stderr': e.stderr
            }))
            return []
```

These changes will:
- Provide better error handling and logging
- Verify repository structure after cloning
- Ensure proper cleanup of temporary directories
- Give more detailed information about file reading failures

The errors you're seeing suggest that the files weren't properly cloned or were cleaned up too early. These changes should help identify exactly where the process is failing and provide better error recovery.  
[Timestamp: 2024/11/23 16:03:29]

**user**: Provide the full function or method that you modify to implement Adding better error handling for the GitHub clone operation, Improving the file reading process, Adding cleanup protection:Add a function to verify repository structure, Update the process_path function to use the verification:

[Timestamp: 2024/11/23 16:05:18]

**ai**: Here are the complete, updated functions with all the improvements:

```python
def read_file_with_fallback_encoding(file_path: str) -> Optional[str]:
    """
    Read a file with multiple encoding attempts and improved error handling.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        Optional[str]: File contents if successful, None otherwise
    """
    if not os.path.exists(file_path):
        logger.error(json.dumps({
            'event': 'error',
            'error': f"File not found: {file_path}",
            'path': file_path
        }))
        return None
        
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.debug(json.dumps({
                    'event': 'info',
                    'message': f"Successfully read {file_path} with {encoding} encoding",
                    'encoding': encoding,
                    'path': file_path
                }))
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Error reading {file_path}: {str(e)}",
                'exception_type': type(e).__name__,
                'path': file_path
            }))
    return None

def verify_repository_structure(repo_path: str) -> bool:
    """
    Verify that the cloned repository has the expected structure.
    
    Args:
        repo_path (str): Path to the repository root
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    try:
        if not os.path.isdir(repo_path):
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Repository path {repo_path} is not a directory",
                'path': repo_path
            }))
            return False

        # List all files and directories
        contents = os.listdir(repo_path)
        logger.debug(json.dumps({
            'event': 'info',
            'message': f"Repository contents: {contents}",
            'path': repo_path,
            'file_count': len(contents)
        }))

        # Check for Python files (including in subdirectories)
        python_files = []
        for root, _, files in os.walk(repo_path):
            python_files.extend([
                os.path.join(root, f) 
                for f in files 
                if f.endswith('.py')
            ])

        if not python_files:
            logger.warning(json.dumps({
                'event': 'warning',
                'message': "No Python files found in repository",
                'path': repo_path
            }))
            return False

        logger.info(json.dumps({
            'event': 'info',
            'message': f"Found {len(python_files)} Python files",
            'python_file_count': len(python_files),
            'path': repo_path
        }))

        return True

    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Error verifying repository structure: {str(e)}",
            'exception_type': type(e).__name__,
            'path': repo_path
        }))
        return False

def process_path(path: str, args: argparse.Namespace, monitor: DocstringMonitor, temp_dirs: List[str]) -> List[ModuleInfo]:
    """
    Process a path (file, directory, or repository URL) and extract module information.
    
    Args:
        path (str): Path to process
        args (argparse.Namespace): Command line arguments
        monitor (DocstringMonitor): Monitoring instance
        temp_dirs (List[str]): List of temporary directories to track
        
    Returns:
        List[ModuleInfo]: List of extracted module information
    """
    modules_info = []
    
    # Handle GitHub repository URLs
    if is_github_url(path):
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        try:
            logger.info(json.dumps({
                'event': 'clone_start',
                'repository': path,
                'temp_dir': temp_dir
            }))
            
            result = subprocess.run(
                ['git', 'clone', path, temp_dir],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if not verify_repository_structure(temp_dir):
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Invalid repository structure after cloning {path}",
                    'repository': path,
                    'temp_dir': temp_dir
                }))
                return []
                
            repo_root = temp_dir
            path = temp_dir
            
            logger.info(json.dumps({
                'event': 'clone_success',
                'repository': path,
                'temp_dir': temp_dir
            }))
            
        except subprocess.CalledProcessError as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Failed to clone {path}",
                'repository': path,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'return_code': e.returncode
            }))
            return []
        except subprocess.TimeoutExpired:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Timeout while cloning {path}",
                'repository': path
            }))
            return []
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Unexpected error while cloning {path}: {str(e)}",
                'repository': path,
                'exception_type': type(e).__name__
            }))
            return []
    else:
        repo_root = find_repo_root(path)
        if not repo_root:
            repo_root = os.getcwd()
            logger.info(json.dumps({
                'event': 'info',
                'message': "Repository root not found. Using current working directory.",
                'path': path,
                'cwd': repo_root
            }))

    # Load .gitignore patterns
    gitignore_spec = load_gitignore(repo_root)
    excluded_dirs = {'venv/', 'env/', '.venv/', '.env/', 'virtualenv/', 'venv3/', 'pyenv/', 'pipenv/'}

    # Collect Python files
    python_files = []
    if os.path.isfile(path) and path.endswith('.py'):
        full_path = os.path.abspath(path)
        rel_path = os.path.relpath(full_path, repo_root)
        if not gitignore_spec.match_file(rel_path):
            if not any(rel_path.startswith(excl_dir) for excl_dir in excluded_dirs):
                python_files.append(full_path)
                logger.debug(json.dumps({
                    'event': 'file_added',
                    'file': full_path,
                    'relative_path': rel_path
                }))
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            rel_root = os.path.relpath(root, repo_root)
            
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                os.path.join(rel_root, d).startswith(excl_dir) 
                for excl_dir in excluded_dirs
            )]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.
[Timestamp: 2024/11/23 16:05:57]

**user**: Continue
[Timestamp: 2024/11/23 16:06:00]

**ai**: Here's the continuation and completion of the `process_path` function, along with the updated `main` function that includes the improved cleanup protection:

```python
def process_path(path: str, args: argparse.Namespace, monitor: DocstringMonitor, temp_dirs: List[str]) -> List[ModuleInfo]:
    # ... (previous part remains the same until the directory walking loop)
    
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            rel_root = os.path.relpath(root, repo_root)
            
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                os.path.join(rel_root, d).startswith(excl_dir) 
                for excl_dir in excluded_dirs
            )]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_root)
                    
                    if not gitignore_spec.match_file(rel_path):
                        if not any(rel_path.startswith(excl_dir) for excl_dir in excluded_dirs):
                            python_files.append(full_path)
                            logger.debug(json.dumps({
                                'event': 'file_added',
                                'file': full_path,
                                'relative_path': rel_path
                            }))
                        else:
                            logger.debug(json.dumps({
                                'event': 'file_excluded',
                                'file': full_path,
                                'reason': 'excluded_directory'
                            }))
                    else:
                        logger.debug(json.dumps({
                            'event': 'file_excluded',
                            'file': full_path,
                            'reason': 'gitignore'
                        }))
    else:
        logger.warning(json.dumps({
            'event': 'warning',
            'message': f"Invalid path: {path}",
            'path': path,
            'path_type': 'unknown'
        }))
        return []

    if not python_files:
        logger.warning(json.dumps({
            'event': 'warning',
            'message': f"No Python files found to process",
            'path': path
        }))
        return []

    # Process each Python file
    for python_file in python_files:
        try:
            source_code = read_file_with_fallback_encoding(python_file)
            if not source_code:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Could not read file contents",
                    'file': python_file
                }))
                continue

            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Syntax error in file",
                    'file': python_file,
                    'line': e.lineno,
                    'offset': e.offset,
                    'details': str(e)
                }))
                continue
            except Exception as e:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Failed to parse file",
                    'file': python_file,
                    'exception_type': type(e).__name__,
                    'details': str(e)
                }))
                continue

            # Extract module information
            try:
                module_info = extract_module_info(python_file, tree, source_code)
                
                # Calculate complexity for methods and functions
                for cls in module_info.classes:
                    for method in cls.methods:
                        method.complexity = calculate_complexity(
                            method.node, 
                            args.complexity_metric
                        )
                
                for func in module_info.functions:
                    func.complexity = calculate_complexity(
                        func.node,
                        args.complexity_metric
                    )
                
                modules_info.append(module_info)
                
                logger.info(json.dumps({
                    'event': 'module_processed',
                    'file': python_file,
                    'classes': len(module_info.classes),
                    'functions': len(module_info.functions)
                }))
                
                monitor.log_event('file_processed', {
                    'file': python_file,
                    'functions': len(module_info.functions)
                })
                
            except Exception as e:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Failed to extract module info",
                    'file': python_file,
                    'exception_type': type(e).__name__,
                    'details': str(e)
                }))
                continue

        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Unexpected error processing file",
                'file': python_file,
                'exception_type': type(e).__name__,
                'details': str(e)
            }))
            continue

    return modules_info

def main():
    """
    Main entry point for the documentation generator.
    """
    args = parse_arguments()
    setup_logging(args.verbose)
    monitor = DocstringMonitor()

    temp_dirs = []
    modules_info = []

    try:
        for path in args.input:
            try:
                logger.info(json.dumps({
                    'event': 'processing_start',
                    'path': path
                }))
                
                modules = process_path(path, args, monitor, temp_dirs)
                if modules:
                    modules_info.extend(modules)
                    logger.info(json.dumps({
                        'event': 'processing_success',
                        'path': path,
                        'modules_count': len(modules)
                    }))
                else:
                    logger.warning(json.dumps({
                        'event': 'warning',
                        'message': f"No valid modules found",
                        'path': path
                    }))
                    
            except Exception as e:
                logger.error(json.dumps({
                    'event': 'error',
                    'error': f"Failed to process path",
                    'path': path,
                    'exception_type': type(e).__name__,
                    'details': str(e)
                }))
                continue

        if not modules_info:
            logger.warning(json.dumps({
                'event': 'warning',
                'message': "No modules found to process"
            }))
            return

        # Process functions
        all_functions = []
        for module in modules_info:
            for cls in module.classes:
                all_functions.extend(cls.methods)
            all_functions.extend(module.functions)

        if not all_functions:
            logger.warning(json.dumps({
                'event': 'warning',
                'message': "No functions found to process"
            }))
            return

        # Generate docstrings
        docstring_generator = DocstringGenerator(args, monitor)
        try:
            asyncio.run(docstring_generator.process_functions(all_functions))
        except KeyboardInterrupt:
            logger.error(json.dumps({
                'event': 'error',
                'error': "Script interrupted by user"
            }))
            sys.exit(1)
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Failed to generate docstrings",
                'exception_type': type(e).__name__,
                'details': str(e)
            }))
            sys.exit(1)

        # Generate markdown documentation
        try:
            generate_markdown(modules_info, args.output)
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Failed to generate markdown",
                'exception_type': type(e).__name__,
                'details':
[Timestamp: 2024/11/23 16:06:43]

**user**: Continue
[Timestamp: 2024/11/23 16:06:48]

**ai**: Here's the completion of the `main` function and the additional helper functions:

```python
def main():
    """
    Main entry point for the documentation generator.
    """
    # ... (previous part remains the same until the markdown generation)

        # Generate markdown documentation
        try:
            generate_markdown(modules_info, args.output)
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Failed to generate markdown",
                'exception_type': type(e).__name__,
                'details': str(e)
            }))
            sys.exit(1)

        # Update statistics
        monitor.stats['total_duration'] = sum(monitor.stats.get('processing_times', []))
        monitor.log_event('summary', {'stats': monitor.stats})

    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Unexpected error in main execution",
            'exception_type': type(e).__name__,
            'details': str(e)
        }))
        sys.exit(1)
        
    finally:
        # Cleanup temporary directories
        cleanup_temp_directories(temp_dirs)
        
        # Close resources
        try:
            if 'docstring_generator' in locals():
                docstring_generator.cache.close()
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Error closing cache",
                'exception_type': type(e).__name__,
                'details': str(e)
            }))

def cleanup_temp_directories(temp_dirs: List[str]) -> None:
    """
    Safely clean up temporary directories.
    
    Args:
        temp_dirs (List[str]): List of temporary directory paths to clean up
    """
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                logger.debug(json.dumps({
                    'event': 'cleanup_start',
                    'temp_dir': temp_dir
                }))
                
                # Remove read-only attributes if necessary
                def remove_readonly(func, path, excinfo):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                shutil.rmtree(temp_dir, onerror=remove_readonly)
                
                logger.debug(json.dumps({
                    'event': 'cleanup_success',
                    'temp_dir': temp_dir
                }))
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Failed to cleanup temporary directory",
                'temp_dir': temp_dir,
                'exception_type': type(e).__name__,
                'details': str(e)
            }))

def safe_file_operations(func):
    """
    Decorator for safe file operations with proper error handling.
    
    Args:
        func: The function to wrap
        
    Returns:
        The wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"File operation error in {func.__name__}",
                'exception_type': type(e).__name__,
                'details': str(e),
                'errno': e.errno
            }))
            return None
        except Exception as e:
            logger.error(json.dumps({
                'event': 'error',
                'error': f"Unexpected error in {func.__name__}",
                'exception_type': type(e).__name__,
                'details': str(e)
            }))
            return None
    return wrapper

@safe_file_operations
def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to check/create
        
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(json.dumps({
            'event': 'directory_check',
            'directory': directory,
            'status': 'exists_or_created'
        }))
        return True
    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Failed to create directory",
            'directory': directory,
            'exception_type': type(e).__name__,
            'details': str(e)
        }))
        return False

@safe_file_operations
def safe_write_file(file_path: str, content: str) -> bool:
    """
    Safely write content to a file with proper error handling.
    
    Args:
        file_path (str): Path to the file to write
        content (str): Content to write to the file
        
    Returns:
        bool: True if write was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not ensure_directory_exists(directory):
            return False

        # Write to temporary file first
        temp_file = file_path + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Rename temporary file to target file
        os.replace(temp_file, file_path)
        
        logger.debug(json.dumps({
            'event': 'file_write_success',
            'file': file_path,
            'size': len(content)
        }))
        return True
        
    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Failed to write file",
            'file': file_path,
            'exception_type': type(e).__name__,
            'details': str(e)
        }))
        
        # Cleanup temporary file if it exists
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error(json.dumps({
            'event': 'error',
            'error': "Script interrupted by user"
        }))
        sys.exit(1)
    except Exception as e:
        logger.error(json.dumps({
            'event': 'error',
            'error': f"Unhandled exception in main",
            'exception_type': type(e).__name__,
            'details': str(e)
        }))
        sys.exit(1)
```

This completes the implementation with:
1. Robust error handling throughout the entire process
2. Safe file operations with proper cleanup
3. Detailed logging of all operations and errors
4. Proper resource cleanup in all cases
5. Helper functions for common operations
6. Safe directory and file operations with atomic writes
7. Comprehensive error reporting and status tracking

The code now handles:
- Temporary directory cleanup
- File operation errors
- Permission issues
- Interrupted operations
- Resource cleanup
- Atomic file writes
- Directory creation
- Detailed error logging

All operations are now safer and more robust, with proper error handling and logging throughout the entire process.  
[Timestamp: 2024/11/23 16:07:29]

**user**: @functools.wraps(func)  
?  
[Timestamp: 2024/11/23 16:15:41]

**user**: @functools.wraps(func)  
?  
[Timestamp: 2024/11/23 16:15:52]
