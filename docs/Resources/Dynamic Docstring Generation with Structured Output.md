---
tags:
  - "#code-analysis"
  - "#documentation-generation"
  - "#code-documentation-system"
  - "#code-implementation"
  - "#best-practices"
  - "#dynamic-prompt-generation"
  - "#structured-output-handling"
  - code-analysis
  - documentation-generation
  - code-documentation-system
  - code-implementation
  - best-practices
  - dynamic-prompt-generation
  - structured-output-handling
---
Based on the provided information and requirements, I'll help update the DocstringGenerator class in generate_docs_4.py to incorporate dynamic prompts, structured outputs, and function calling. Here's the implementation:

```python
class DocstringGenerator:
    def __init__(self, args, monitor: DocstringMonitor):
        self.args = args
        self.monitor = monitor
        self.cache = diskcache.Cache(args.cache_location)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.content_filter = ContentFilter()
        self.client = None
        self._initialize_client()
        self._initialize_tools()

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
                            "description": "A concise summary of what the function does"
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
                            "type": "string",
                            "description": "Description of what the function returns"
                        }
                    },
                    "required": ["summary", "args", "returns"]
                }
            }
        }

    def create_dynamic_prompt(self, func_info: FunctionInfo, context: str = "") -> str:
        """Create a dynamic prompt based on function information and context."""
        prompt = f"Generate a detailed Python docstring for:\n\n{generate_function_signature(func_info)}\n\n"
        
        if func_info.docstring:
            prompt += f"Existing docstring:\n{func_info.docstring}\n\n"
        
        if context:
            prompt += f"Additional context: {context}\n\n"
            
        if func_info.is_method:
            prompt += f"This is a method of class '{func_info.parent_class}'\n"
            
        if func_info.decorators:
            prompt += f"Decorators: {', '.join(func_info.decorators)}\n"
            
        prompt += "\nProvide a comprehensive docstring following Google Style guidelines."
        return prompt

    async def generate_docstring(self, func_info: FunctionInfo, prompt: str, semaphore: asyncio.Semaphore):
        """Generate a docstring using the Azure OpenAI API with function calling."""
        max_retries = 3
        backoff = 1

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                async with semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.deployment_id,
                        messages=[
                            {"role": "system", "content": "You are a specialized Python documentation assistant that generates precise, detailed docstrings."},
                            {"role": "user", "content": prompt}
                        ],
                        tools=[self.docstring_tool],
                        tool_choice={"type": "function", "function": {"name": "generate_docstring"}},
                        max_tokens=self.args.max_tokens,
                        temperature=0.5
                    )

                    api_duration = time.time() - start_time

                    # Handle function call response
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        if tool_call.function.name == "generate_docstring":
                            docstring_json = json.loads(tool_call.function.arguments)
                            formatted_docstring = self.format_docstring_from_json(docstring_json)
                            
                            # Update function info and stats
                            original_docstring = func_info.docstring
                            func_info.docstring = formatted_docstring

                            if original_docstring is None:
                                self.monitor.stats['generated'] += 1
                                event_type = 'generated'
                            else:
                                self.monitor.stats['improved'] += 1
                                event_type = 'improved'

                            self.monitor.log_event(event_type, {
                                'function': func_info.name,
                                'duration': api_duration,
                                'tokens': response.usage.total_tokens
                            })

                            # Cache the result
                            self.cache.set(func_info.code_hash, formatted_docstring)
                            return
                    
                    raise ValueError("No valid function call response received")

            except (OpenAIError, ValueError, json.JSONDecodeError) as e:
                self.monitor.log_event('error', {
                    'function': func_info.name,
                    'attempt': attempt + 1,
                    'error': str(e)
                })
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    func_info.docstring = ''
                    self.monitor.stats['failed'] += 1
                    return

    async def process_functions(self, functions: List[FunctionInfo]):
        """Process multiple functions concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(5)
        tasks = []

        for func in functions:
            if not self.args.force and func.code_hash in self.cache:
                func.docstring = self.cache.get(func.code_hash)
                self.monitor.stats['cached'] += 1
                self.monitor.log_event('cache_hit', {'function': func.name})
                continue

            if func.docstring is None or not is_docstring_complete(func.docstring):
                prompt = self.create_dynamic_prompt(func)
                
                # Handle token limits
                max_tokens = 4000 - self.args.max_tokens
                if self.count_tokens(prompt) > max_tokens:
                    prompt = self.truncate_prompt(prompt, max_tokens)
                    self.monitor.log_event('prompt_truncated', {'function': func.name})

                tasks.append(self.generate_docstring(func, prompt, semaphore))

        if tasks:
            await asyncio.gather(*tasks)
```

Key changes and improvements:

1. Added `_initialize_tools()` to define the function schema for structured output

2. Implemented `create_dynamic_prompt()` that generates context-aware prompts based on function information

3. Updated `generate_docstring()` to use function calling and handle structured responses

4. Enhanced error handling and monitoring

5. Improved token management and rate limiting

6. Added better caching logic

To use this updated implementation:

```python
# Example usage
async def main():
    args = parse_arguments()
    monitor = DocstringMonitor()
    generator = DocstringGenerator(args, monitor)
    
    # Process your functions
    functions = [...] # Your list of FunctionInfo objects
    await generator.process_functions(functions)
```

This implementation provides:

- Structured JSON output via function calling
- Dynamic prompt generation based on function context
- Better error handling and retry logic
- Improved monitoring and logging
- Rate limiting and token management
- Efficient caching

The code maintains compatibility with the existing codebase while adding the requested functionality for dynamic prompts and structured outputs.