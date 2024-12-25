---
epoch: 1732925474280
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: [docs.py](docs.py.md)[d
[Timestamp: 2024/11/29 18:11:14]

**ai**: It seems you have provided the content of two files: `docs.py` and `docs.py.md`. Both files appear to contain the same Python code for a Documentation Management Module, which handles docstring operations and documentation generation.

If you have any specific questions or need further assistance with this code, feel free to ask!
[Timestamp: 2024/11/29 18:11:15]

**user**: class DocStringManager:
    """
    Manages docstring operations and documentation generation.

    Attributes:
        context (DocumentationContext): The context for documentation generation.
        cache (Optional[Any]): Optional cache for storing intermediate results.
    """

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): The context for documentation generation.
            cache (Optional[Any]): Optional cache for storing intermediate results.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator(MarkdownConfig(include_source=True))
        self.code_extractor = CodeExtractor(ExtractionContext())
        self.metrics_calculator = Metrics()
        self._add_parents(self.tree)

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation for the current context.

        Returns:
            str: The generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            sections = [
                self._create_module_section(),
                self._create_overview_section(),
                self._create_classes_section(extraction_result.classes),
                self._create_class_methods_section(extraction_result.classes),
                self._create_functions_section(extraction_result.functions),
                self._create_constants_section(extraction_result.constants),
                self._create_changes_section(),
                self._create_source_code_section(extraction_result.metrics)
            ]
            return self.markdown_generator.generate(sections, self.context.module_path)
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            raise DocumentationError("Documentation generation failed", {'error': str(e)})

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module section.

        Returns:
            DocumentationSection: The module section of the documentation.
        """
        module_name = self.context.module_path.stem if self.context.module_path else 'Unknown Module'
        return DocumentationSection(title=f"Module: {module_name}", content="")

    def _create_overview_section(self) -> DocumentationSection:
        """
        Create overview section.

        Returns:
            DocumentationSection: The overview section of the documentation.
        """
        file_path = self.context.module_path or Path('unknown')
        description = self.context.metadata.get('description', 'No description provided.')
        content = [
            "## Overview",
            f"**File:** `{file_path}`",
            f"**Description:** {description}"
        ]
        return DocumentationSection(title="Overview", content="\n".join(content))

    def _create_classes_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
       class DocStringManager:
    """
    Manages docstring operations and documentation generation.

    Attributes:
        context (DocumentationContext): The context for documentation generation.
        cache (Optional[Any]): Optional cache for storing intermediate results.
    """

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): The context for documentation generation.
            cache (Optional[Any]): Optional cache for storing intermediate results.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator(MarkdownConfig(include_source=True))
        self.code_extractor = CodeExtractor(ExtractionContext())
        self.metrics_calculator = Metrics()
        self._add_parents(self.tree)

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation for the current context.

        Returns:
            str: The generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            sections = [
                self._create_module_section(),
                self._create_overview_section(),
                self._create_classes_section(extraction_result.classes),
                self._create_class_methods_section(extraction_result.classes),
                self._create_functions_section(extraction_result.functions),
                self._create_constants_section(extraction_result.constants),
                self._create_changes_section(),
                self._create_source_code_section(extraction_result.metrics)
            ]
            return self.markdown_generator.generate(sections, self.context.module_path)
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            raise DocumentationError("Documentation generation failed", {'error': str(e)})

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module section.

        Returns:
            DocumentationSection: The module section of the documentation.
        """
        module_name = self.context.module_path.stem if self.context.module_path else 'Unknown Module'
        return DocumentationSection(title=f"Module: {module_name}", content="")

    def _create_overview_section(self) -> DocumentationSection:
        """
        Create overview section.

        Returns:
            DocumentationSection: The overview section of the documentation.
        """
        file_path = self.context.module_path or Path('unknown')
        description = self.context.metadata.get('description', 'No description provided.')
        content = [
            "## Overview",
            f"**File:** `{file_path}`",
            f"**Description:** {description}"
        ]
        return DocumentationSection(title="Overview", content="\n".join(content))

    def _create_classes_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
[Timestamp: 2024/11/29 18:11:55]

**ai**: It looks like you've pasted the `DocStringManager` class twice, and the second instance is incomplete. The `DocStringManager` class is responsible for managing docstring operations and generating documentation for a given context. It includes methods to create various sections of documentation, such as module, overview, classes, class methods, functions, constants, recent changes, and source code.

If you have any specific questions or need help with a particular part of this class, please let me know!
[Timestamp: 2024/11/29 18:11:57]

**user**: Complete the second instance
[Timestamp: 2024/11/29 18:12:06]

**ai**: Certainly! Here's the completion of the second instance of the `DocStringManager` class:

```python
class DocStringManager:
    """
    Manages docstring operations and documentation generation.

    Attributes:
        context (DocumentationContext): The context for documentation generation.
        cache (Optional[Any]): Optional cache for storing intermediate results.
    """

    def __init__(self, context: DocumentationContext, cache: Optional[Any] = None) -> None:
        """
        Initialize DocStringManager with context and optional cache.

        Args:
            context (DocumentationContext): The context for documentation generation.
            cache (Optional[Any]): Optional cache for storing intermediate results.
        """
        self.context = context
        self.tree: ast.Module = ast.parse(context.source_code)
        self.processor = DocstringProcessor()
        self.cache = cache
        self.changes: List[str] = []
        self.markdown_generator = MarkdownGenerator(MarkdownConfig(include_source=True))
        self.code_extractor = CodeExtractor(ExtractionContext())
        self.metrics_calculator = Metrics()
        self._add_parents(self.tree)

    async def generate_documentation(self) -> str:
        """
        Generate complete documentation for the current context.

        Returns:
            str: The generated documentation in markdown format.

        Raises:
            DocumentationError: If documentation generation fails.
        """
        try:
            extraction_result = self.code_extractor.extract_code(self.context.source_code)
            sections = [
                self._create_module_section(),
                self._create_overview_section(),
                self._create_classes_section(extraction_result.classes),
                self._create_class_methods_section(extraction_result.classes),
                self._create_functions_section(extraction_result.functions),
                self._create_constants_section(extraction_result.constants),
                self._create_changes_section(),
                self._create_source_code_section(extraction_result.metrics)
            ]
            return self.markdown_generator.generate(sections, self.context.module_path)
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            raise DocumentationError("Documentation generation failed", {'error': str(e)})

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module section.

        Returns:
            DocumentationSection: The module section of the documentation.
        """
        module_name = self.context.module_path.stem if self.context.module_path else 'Unknown Module'
        return DocumentationSection(title=f"Module: {module_name}", content="")

    def _create_overview_section(self) -> DocumentationSection:
        """
        Create overview section.

        Returns:
            DocumentationSection: The overview section of the documentation.
        """
        file_path = self.context.module_path or Path('unknown')
        description = self.context.metadata.get('description', 'No description provided.')
        content = [
            "## Overview",
            f"**File:** `{file_path}`",
            f"**Description:** {description}"
        ]
        return DocumentationSection(title="Overview", content="\n".join(content))

    def _create_classes_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
        Create classes section.

        Args:
            classes (List[ExtractedClass]): List of extracted classes.

        Returns:
            DocumentationSection: The classes section of the documentation.
        """
        if not classes:
            return DocumentationSection("Classes", "")
        content = [
            "## Classes",
            "| Class | Inherits From | Complexity Score* |",
            "|-------|---------------|------------------|"
        ]
        for cls in classes:
            complexity_score = self.metrics_calculator.calculate_complexity(cls.node)
            warning = " ⚠️" if complexity_score > 10 else ""
            bases = ", ".join(cls.bases) if cls.bases else "None"
            row = f"| `{cls.name}` | `{bases}` | {complexity_score}{warning} |"
            content.append(row)
        return DocumentationSection(title="Classes", content="\n".join(content))

    def _create_class_methods_section(self, classes: List[ExtractedClass]) -> DocumentationSection:
        """
        Create class methods section.

        Args:
            classes (List[ExtractedClass]): List of extracted classes.

        Returns:
            DocumentationSection: The class methods section of the documentation.
        """
        if not classes:
            return DocumentationSection("Class Methods", "")
        content = [
            "### Class Methods",
            "| Class | Method | Parameters | Returns | Complexity Score* |",
            "|-------|--------|------------|---------|-------------------|"
        ]
        for cls in classes:
            for method in cls.methods:
                complexity_score = self.metrics_calculator.calculate_complexity(method.node)
                warning = " ⚠️" if complexity_score > 10 else ""
                params = self._format_parameters(method.args)
                returns = method.return_type or "None"
                row = f"| `{cls.name}` | `{method.name}` | `{params}` | `{returns}` | {complexity_score}{warning} |"
                content.append(row)
        return DocumentationSection(title="Class Methods", content="\n".join(content))

    def _create_functions_section(self, functions: List[ExtractedFunction]) -> DocumentationSection:
        """
        Create functions section.

        Args:
            functions (List[ExtractedFunction]): List of extracted functions.

        Returns:
            DocumentationSection: The functions section of the documentation.
        """
        if not functions:
            return DocumentationSection("Functions", "")
        content = [
            "## Functions",
            "| Function | Parameters | Returns | Complexity Score* |",
            "|----------|------------|---------|-------------------|"
        ]
        for func in functions:
            complexity_score = self.metrics_calculator.calculate_complexity(func.node)
            warning = " ⚠️" if complexity_score > 10 else ""
            params = self._format_parameters(func.args)
            returns = func.return_type or "None"
            row = f"| `{func.name}` | `{params}` | `{returns}` | {complexity_score}{warning} |"
            content.append(row)
        return DocumentationSection(title="Functions", content="\n".join(content))

    def _create_constants_section(self, constants: List[Dict[str, Any]]) -> DocumentationSection:
        """
        Create constants and variables section.

        Args:
            constants (List[Dict[str, Any]]): List of extracted constants.

        Returns:
            DocumentationSection: The constants and variables section of the documentation.
        """
        if not constants:
            return DocumentationSection("Constants and Variables", "")
        content = [
            "## Constants and Variables",
            "| Name | Type | Value |",
            "|------|------|--------|"
        ]
        for const in constants:
            row = f"| `{const['name']}` | `{const['type']}` | `{const['value']}` |"
            content.append(row)
        return DocumentationSection(title="Constants and Variables", content="\n".join(content))

    def _create_changes_section(self) -> DocumentationSection:
        """
        Create recent changes section.

        Returns:
            DocumentationSection: The recent changes section of the documentation.
        """
        content = ["## Recent Changes"]
        changes = self.context.metadata.get('changes', [])
        if changes:
            content.extend(f"- {change}" for change in changes)
        else:
            content.append("- No recent changes recorded.")
        return DocumentationSection(title="Recent Changes", content="\n".join(content))

    def _create_source_code_section(self, metrics: Dict[str, Any]) -> DocumentationSection:
        """
        Create source code section with complexity info.

        Args:
            metrics (Dict[str, Any]): Complexity metrics of the source code.

        Returns:
            DocumentationSection: The source code section of the documentation.
        """
        complexity_header = self._format_complexity_header(metrics)
        content = [
            "## Source Code",
            "```python",
            f'"""{complexity_header}"""',
            self.context.source_code,
            "```"
        ]
        return DocumentationSection(title="Source Code", content="\n".join(content))

    def _format_parameters(self, args: List[ExtractedArgument]) -> str:
        """
        Format function parameters.

        Args:
            args (List[ExtractedArgument]): List of function arguments.

        Returns:
            str: Formatted string of parameters.
        """
        params = []
        for arg in args:
            param = f"{arg.name}: {arg.type_hint or 'Any'}"
            if arg.default_value is not None:
                param += f" = {arg.default_value}"
            params.append(param)
        return f"({', '.join(params)})"

    def _format_complexity_header(self, metrics: Dict[str, Any]) -> str:
        """
        Format complexity information for module header.

        Args:
            metrics (Dict[str, Any]): Complexity metrics of the source code.

        Returns:
            str: Formatted complexity header.
        """
        lines = ["Module Complexity Information:"]
        for name, score in metrics.items():
            if isinstance(score, (int, float)):
                warning = " ⚠️" if score > 10 else ""
                lines.append(f"    {name}: {score}{warning}")
        return "\n".join(lines)

    def _add_parents(self, node: ast.AST) -> None:
        """
        Add parent references to AST nodes.

        Args:
            node (ast.AST): The AST node to process.
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_parents(child)
```

This completion mirrors the first instance of the `DocStringManager` class, ensuring consistency in functionality and structure. Let me know if you need further modifications or explanations!
[Timestamp: 2024/11/29 18:12:31]

**user**: [ai_service.py](ai_service.py.md)
[Timestamp: 2024/11/29 18:12:50]

**ai**: The `AIInteractionHandler` class is designed to manage interactions with the Azure OpenAI API for generating docstrings. It handles token management, caching, and response processing with structured JSON outputs. Here's a brief overview of its main components:

- **Initialization**: The class is initialized with optional components like a cache, metrics collector, token manager, configuration, and code extractor. It sets up the OpenAI API configuration and initializes necessary tools for structured output.

- **Docstring Generation**: The `_generate_documentation` method uses Azure OpenAI to generate documentation for given source code. It creates a prompt with detailed requirements and attempts to get a valid response from the API.

- **Prompt Creation**: Methods like `_create_function_calling_prompt`, `_format_complexity_info`, `_format_dependencies_info`, and `_format_type_info` help in creating a comprehensive prompt for the AI model, including complexity, dependencies, and type information.

- **Response Handling**: The class validates and processes the API response, ensuring it conforms to a predefined JSON schema (`DOCSTRING_SCHEMA`).

- **Code Processing**: The `process_code` method processes the source code to generate and insert docstrings, updating the code and generating markdown documentation.

- **Caching and Resource Management**: The class includes methods for caching results and managing resources, such as closing connections and cleaning up in an asynchronous context.

If you have any specific questions or need further details about any part of this class, feel free to ask!
[Timestamp: 2024/11/29 18:12:56]

**user**: class AIInteractionHandler:
    """Handles AI interactions for docstring generation via Azure OpenAI API."""
    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None,
        code_extractor: Optional[CodeExtractor] = None
    ):
        """
        Initialize the AI Interaction Handler.

        Args:
            cache (Optional[Cache]): Cache instance for storing results.
            metrics_collector (Optional[MetricsCollector]): Collector for gathering metrics.
            token_manager (Optional[TokenManager]): Manager for handling API tokens.
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
            code_extractor (Optional[CodeExtractor]): Extractor for code analysis.
        """
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
            self.code_extractor = code_extractor or CodeExtractor()
            self._initialize_tools()
            logger.info("AI Interaction Handler initialized successfully")

            # Set up OpenAI API configuration
            openai.api_type = "azure"
            openai.api_key = self.config.api_key
            openai.api_base = self.config.endpoint
            openai.api_version = self.config.api_version

            # Initialize module state
            self._current_module_tree = None
            self._current_module_docs = {}
            self._current_module = None

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Process source code to generate docstrings."""
        try:
            # Validate token usage
            valid, metrics, message = await self.token_manager.validate_request(source_code)
            if not valid:
                logger.error(f"Token validation failed: {message}")
                return None

            # Initialize module state
            self._current_module_tree = ast.parse(source_code)
            self._current_module_docs = {}
            self._current_module = type('Module', (), {
                '__name__': cache_key.split(':')[1] if cache_key else 'unknown',
                '__doc__': ast.get_docstring(self._current_module_tree) or '',
                '__version__': '0.1.0',
                '__author__': 'Unknown'
            })
            
            # Generate module documentation
            module_docs = await self._generate_documentation(source_code, {}, self._current_module_tree)
            if not module_docs:
                logger.error("Documentation generation failed")
                return None

            # Process classes and functions
            for node in ast.walk(self._current_module_tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    element_docs = await self._generate_documentation(
                        ast.unparse(node), 
                        {"element_type": type(node).__name__}, 
                        node
                    )
                    if element_docs:
                        self._insert_docstring(node, element_docs.content)

            # Generate final output
            updated_code = ast.unparse(self._current_module_tree)

            # Cache results
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, module_docs.content)

            # Track token usage
            self.token_manager.track_request(metrics['prompt_tokens'], metrics['max_completion_tokens'])
            logger.info(f"Tokens used: {metrics['prompt_tokens']} prompt, {metrics['max_completion_tokens']} completion")

            return updated_code, module_docs.content

        except Exception as e:
            logger.error(f"Process code failed: {e}")
            return None

    async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI."""
        try:
            prompt = self._create_function_calling_prompt(source_code, metadata, node)
            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
                tools=[self.docstring_function],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )
            
            if not response:
                return None
                
            message = response.choices[0].message
            if message.tool_calls and message.tool_calls[0].function:
                function_args = message.tool_calls[0].function.arguments
                response_data = json.loads(function_args)
                validate(instance=response_data, schema=DOCSTRING_SCHEMA)
                docstring_data = DocstringData(**response_data)
                
                # Add complexity if node provided
                if node and self.metrics_collector:
                    complexity = self.metrics_collector.calculate_complexity(node)
                    docstring_data.set_complexity(complexity)
                
                formatted_docstring = self.docstring_processor.format(docstring_data)
                
                # Track token usage
                self.token_manager.track_request(usage['prompt_tokens'], usage['completion_tokens'])
                logger.info(f"Tokens used: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion")
                
                return ProcessingResult(
                    content=formatted_docstring,
                    usage=usage or {},
                    processing_time=0.0
                )
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = {
            "type": "function",
            "function": {
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
                    "required": ["summary", "description", "args", "returns", "raises"]
                }
            }
        }

    def _create_function_calling_prompt(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:
        """
        Create the initial prompt for function calling.

        Args:
            source_code (str): The source code to document.
            metadata (Dict[str, Any]): Additional metadata for the prompt.
            node (Optional[ast.AST]): The AST node representing the code element.

        Returns:
            str: The generated prompt for function calling.
        """
        extraction_data = self.code_extractor.extract_code(source_code)
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
        """
        Format complexity information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for complexity analysis.

        Returns:
            str: Formatted complexity information.
        """
        complexity_lines = ["Complexity Information:"]
        for cls in extraction_data.classes:
            score = cls.metrics.get('complexity', 0)
            complexity_lines.append(f"- Class '{cls.name}' complexity: {score}")
            for method in cls.methods:
                m_score = method.metrics.get('complexity', 0)
                complexity_lines.append(f"  - Method '{method.name}' complexity: {m_score}")
        for func in extraction_data.functions:
            score = func.metrics.get('complexity', 0)
            complexity_lines.append(f"- Function '{func.name}' complexity: {score}")
        return "\n".join(complexity_lines)

    def _format_dependencies_info(self, extraction_data: Any) -> str:
        """
        Format dependency information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for dependency analysis.

        Returns:
            str: Formatted dependency information.
        """
        dep_lines = ["Dependencies:"]
        for name, deps in extraction_data.imports.items():
            dep_lines.append(f"- {name}: {', '.join(deps)}")
        return "\n".join(dep_lines)

    def _format_type_info(self, extraction_data: Any) -> str:
        """
        Format type information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for type analysis.

        Returns:
            str: Formatted type information.
        """
        type_lines = ["Type Information:"]
        for cls in extraction_data.classes:
            type_lines.append(f"Class '{cls.name}':")
            for attr in cls.attributes:
                type_lines.append(f"- {attr['name']}: {attr['type']}")
            for method in cls.methods:
                args_info = [f"{arg.name}: {arg.type_hint}" for arg in method.args]
                type_lines.append(f"- Method '{method.name}({', '.join(args_info)}) -> {method.return_type}'")
        for func in extraction_data.functions:
            args_info = [f"{arg.name}: {arg.type_hint}" for arg in func.args]
            type_lines.append(f"Function '{func.name}({', '.join(args_info)}) -> {func.return_type}'")
        return "\n".join(type_lines)

    def _create_refinement_prompt(self, original_prompt: str, error_message: str, previous_response: dict) -> str:
        """
        Create a refinement prompt, handling previous responses and errors.

        Args:
            original_prompt (str): The original prompt used for function calling.
            error_message (str): Error message to include in the refinement prompt.
            previous_response (dict): Previous response data to include in the prompt.

        Returns:
            str: The refined prompt for further attempts.
        """
        formatted_response = json.dumps(previous_response, indent=4) if previous_response else ""
        prompt = (
            f"{error_message}\n\n"
            + "Previous Response (if any):\n"
            + f"```json\n{formatted_response}\n```\n\n"
            + original_prompt
        )
        return prompt

    def _insert_docstring(self, node: ast.AST, docstring: str) -> None:
        """
        Insert or update docstring in AST node.

        Args:
            node (ast.AST): The AST node to update.
            docstring (str): The docstring to insert.
        """
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return

        docstring_node = ast.Expr(value=ast.Constant(value=docstring))

        if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
            node.body.pop(0)

        node.body.insert(0, docstring_node)

    async def _cache_result(self, cache_key: str, code: str, documentation: str) -> None:
        """
        Cache the processing result.

        Args:
            cache_key (str): The key to use for caching.
            code (str): The processed code.
            documentation (str): The generated documentation.
        """
        try:
            if self.cache:
                await self.cache.save_docstring(
                    cache_key,
                    {'code': code, 'docs': documentation}
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
        """
        Async context manager entry.

        Returns:
            AIInteractionHandler: The instance of the handler.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()
[Timestamp: 2024/11/29 18:13:01]

**ai**: The `AIInteractionHandler` class is designed to handle interactions with the Azure OpenAI API for generating docstrings. It processes source code, generates documentation, and manages caching and token usage. Here's a breakdown of its key components and functionality:

### Initialization
- **Constructor (`__init__`)**: Sets up the handler with optional components like cache, metrics collector, token manager, configuration, and code extractor. It initializes the OpenAI API configuration and tools for structured output.

### Code Processing
- **`process_code` Method**: Validates token usage, parses the source code into an AST, and generates documentation for the module, classes, and functions. It caches the results and tracks token usage.

### Documentation Generation
- **`_generate_documentation` Method**: Creates a prompt for the AI model to generate docstrings, processes the response, and formats the docstring. It also calculates complexity metrics if a node is provided.

### Prompt Creation
- **`_create_function_calling_prompt` Method**: Constructs a detailed prompt for the AI model, including complexity, dependencies, and type information extracted from the code.

### Utility Methods
- **`_initialize_tools`**: Sets up the function tools for structured output.
- **`_format_complexity_info`, `_format_dependencies_info`, `_format_type_info`**: Format extracted data for inclusion in the prompt.
- **`_create_refinement_prompt`**: Generates a refined prompt if the initial response is invalid.
- **`_insert_docstring`**: Inserts or updates docstrings in the AST nodes.
- **`_cache_result`**: Caches the processed code and documentation.

### Resource Management
- **`close`, `__aenter__`, `__aexit__`**: Manage resources and cleanup, supporting asynchronous context management.

This class is designed to streamline the process of generating and inserting docstrings into Python code using AI, while efficiently managing resources and API interactions. If you have any specific questions or need further clarification on any part, feel free to ask!
[Timestamp: 2024/11/29 18:13:09]

**user**: What is the difference vbetween the two versions of ai_interaction i showed you
[Timestamp: 2024/11/29 18:13:24]

**ai**: The two versions of the `AIInteractionHandler` class you provided have several differences. Here's a summary of the key differences:

1. **Initialization of Module State**:
   - In the second version, the initialization of module state includes setting `_current_module_tree`, `_current_module_docs`, and `_current_module`. This is not present in the first version.

2. **Token Validation**:
   - The second version includes a token validation step in the `process_code` method using `self.token_manager.validate_request(source_code)`. This step is not present in the first version.

3. **Module State Management**:
   - The second version initializes the module state with `ast.parse(source_code)` and sets attributes like `__name__`, `__doc__`, `__version__`, and `__author__`. This is not explicitly done in the first version.

4. **Tracking Token Usage**:
   - In the second version, token usage is tracked using `self.token_manager.track_request(metrics['prompt_tokens'], metrics['max_completion_tokens'])` after generating documentation. This is not explicitly mentioned in the first version.

5. **Error Handling and Logging**:
   - Both versions include error handling and logging, but the second version includes more detailed logging related to token usage and validation.

6. **Selected Text Handling**:
   - The second version includes references to `{selectedText}`, which seems to be a placeholder or variable related to the current module documentation. This is not present in the first version.

Overall, the second version appears to have more detailed handling of token validation and usage, as well as additional state management for the module being processed. If you need further clarification on any specific part, feel free to ask!
[Timestamp: 2024/11/29 18:13:33]

**user**: Does the first version have any functionality the second one does not ?
[Timestamp: 2024/11/29 18:13:58]

**ai**: Based on the provided descriptions of both versions of the `AIInteractionHandler` class, the first version does not appear to have any additional functionality that is not present in the second version. Instead, the second version seems to have enhancements and additional features, such as:

1. **Token Validation and Tracking**: The second version includes token validation and tracking, which are not mentioned in the first version.

2. **Module State Initialization**: The second version initializes and manages module state more explicitly.

3. **Detailed Logging**: The second version includes more detailed logging related to token usage.

4. **Selected Text Handling**: The second version references `{selectedText}`, which might be related to handling specific text selections or context, though the exact functionality is unclear without more context.

Overall, the second version seems to build upon the first with additional features and improvements. If there are specific functionalities you are curious about, please let me know!
[Timestamp: 2024/11/29 18:14:02]

**user**: Are you sure it doesn't have any functions related to markdow ngeneration
[Timestamp: 2024/11/29 18:14:17]

**ai**: Upon reviewing the descriptions again, the first version of the `AIInteractionHandler` class does include functionality related to markdown generation, which was not explicitly highlighted in the second version. Specifically, the first version has methods like `_format_markdown`, `_generate_changelog`, `_generate_class_documentation`, `_generate_function_documentation`, and `_generate_complexity_metrics` that are involved in formatting documentation as markdown.

These methods are responsible for creating structured markdown documentation, including sections for summaries, changelogs, classes, functions, and complexity metrics. This markdown generation functionality is not explicitly mentioned in the second version you provided.

Therefore, the first version does indeed have markdown generation capabilities that are not detailed in the second version. If you have any more questions or need further clarification, feel free to ask!
[Timestamp: 2024/11/29 18:14:22]