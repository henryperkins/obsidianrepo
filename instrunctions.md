Absolutely, enhancing the markdown output generated by your documentation tool is a great way to improve the readability and comprehensiveness of your project's documentation. Below, I'll provide detailed guidance on how to implement each of the strategies you've outlined, referencing the relevant parts of your codebase.

---

### **1. Thorough Docstring Extraction 📝**

**Goal:** Ensure all docstrings are extracted and processed from the source code to include every documented element in the markdown output.

**Implementation Steps:**

- **Update `code_extraction.py`:**
    
    - **Enhance Class and Function Extraction:**
        
        Ensure that the `_extract_classes` and `_extract_functions` methods in the `CodeExtractor` class are correctly parsing all classes and functions, including nested ones.
        
        ```python
        # code_extraction.py
        
        class CodeExtractor:
            # ...
        
            def _extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
                """Extract all classes from the AST, including nested classes."""
                classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Skip private classes if not included
                        if not self.context.include_private and node.name.startswith('_'):
                            continue
                        try:
                            classes.append(self._process_class(node))
                        except Exception as e:
                            self.errors.append(f"Failed to extract class {node.name}: {str(e)}")
                return classes
        
            # Similar enhancements for functions
        ```
        
- **Handle Nested Definitions:**
    
    Modify the extractor to handle nested classes and functions by recursively processing child nodes.
    
    ```python
    # code_extraction.py
    
    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        # ...
        # Process methods, including nested classes and functions
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(self._process_function(child))
            elif isinstance(child, ast.ClassDef):
                # Handle nested classes if needed
                pass
        # ...
    ```
    

### **2. Detailed Docstring Parsing 🔍**

**Goal:** Parse and format docstrings into a structured format for detailed markdown generation.

**Implementation Steps:**

- **Update `docstring_processor.py`:**
    
    - **Improve Parsing Logic:**
        
        Enhance the `parse` method in the `DocstringProcessor` class to handle different docstring formats (e.g., Google style, NumPy style).
        
        ```python
        # docstring_processor.py
        
        class DocstringProcessor:
            # ...
        
            def parse(self, docstring: str) -> DocstringData:
                """
                Parse a raw docstring into a structured format.
                Supports multiple docstring styles.
                """
                # Use a third-party library like docstring_parser
                from docstring_parser import parse
        
                try:
                    parsed = parse(docstring, style='google')
                    args = [
                        {'name': param.arg_name, 'type': param.type_name, 'description': param.description}
                        for param in parsed.params
                    ]
                    returns = {
                        'type': parsed.returns.type_name if parsed.returns else None,
                        'description': parsed.returns.description if parsed.returns else None
                    }
                    return DocstringData(
                        summary=parsed.short_description or '',
                        description=parsed.long_description or '',
                        args=args,
                        returns=returns,
                        raises=[{'exception': e.type_name, 'description': e.description} for e in parsed.raises] if parsed.raises else []
                    )
                except Exception as e:
                    logger.error(f"Error parsing docstring: {e}")
                    return DocstringData("", "", [], {}, [])
        ```
        
- **Handle Different Styles:**
    
    Add support for different docstring styles by parameterizing the parsing method.
    
    ```python
    # docstring_processor.py
    
    class DocstringProcessor:
        # ...
    
        def parse(self, docstring: str, style: str = 'google') -> DocstringData:
            """
            Parse a raw docstring into a structured format.
            Supports multiple docstring styles.
            """
            # Modify the parse method to accept a style parameter
            parsed = parse(docstring, style=style)
            # ...
    ```
    

### **3. Comprehensive Section Generation 📄**

**Goal:** Include all sections (module overviews, class descriptions, function details) in the markdown documentation for a complete codebase overview.

**Implementation Steps:**

- **Update `markdown_generator.py`:**
    
    - **Enhance Section Generation Logic:**
        
        Modify the `_generate_section` method to include module overviews, class hierarchies, and detailed function documentation.
        
        ```python
        # markdown_generator.py
        
        class MarkdownGenerator:
            # ...
        
            def generate(
                self,
                sections: List[DocumentationSection],
                include_source: bool = True,
                source_code: Optional[str] = None,
                module_path: Optional[str] = None
            ) -> str:
                # ...
                # Generate module overview
                if module_path:
                    md_lines.append(f"# Module `{module_path}`\n")
                    md_lines.append(f"{self._module_docstring}\n")
        
                # Generate sections
                for section in sections:
                    md_lines.extend(self._generate_section(section))
                # ...
        ```
        
- **Include Class and Function Details:**
    
    Ensure that class methods and attributes are documented under each class section.
    
    ```python
    # docs.py
    
    class DocStringManager:
        # ...
    
        async def _create_class_section(self, node: ast.ClassDef) -> DocumentationSection:
            # ...
            attributes_section = DocumentationSection(
                title="Attributes",
                content="",
                subsections=[
                    DocumentationSection(
                        title=attr['name'],
                        content=attr.get('docstring', '')
                    ) for attr in extracted_info.attributes
                ]
            )
    
            return DocumentationSection(
                title=f"Class: {node.name}",
                content=docstring_data.description,
                subsections=[attributes_section, methods_section]
            )
    ```
    

### **4. Validation and Error Handling ✅**

**Goal:** Implement validation checks to ensure docstrings meet required standards and handle errors gracefully.

**Implementation Steps:**

- **Update `docstring_processor.py`:**
    
    - **Implement Validation Logic:**
        
        Add a `validate` method to check for required sections and content length.
        
        ```python
        # docstring_processor.py
        
        class DocstringProcessor:
            # ...
        
            def validate(self, docstring_data: DocstringData) -> Tuple[bool, List[str]]:
                """
                Validate the structured docstring data.
                """
                errors = []
                if len(docstring_data.summary) < self.min_length['summary']:
                    errors.append("Summary is too short.")
                if len(docstring_data.description) < self.min_length['description']:
                    errors.append("Description is too short.")
                # Additional validation rules
                is_valid = not errors
                return is_valid, errors
        ```
        
- **Handle Validation in `docs.py`:**
    
    Modify the `process_docstring` method to handle validation errors.
    
    ```python
    # docs.py
    
    class DocStringManager:
        # ...
    
        async def process_docstring(
            self,
            node: ast.AST,
            docstring_data: DocstringData
        ) -> bool:
            # ...
            is_valid, errors = self.processor.validate(docstring_data)
            if not is_valid:
                logger.warning(f"Validation errors: {errors}")
                # Handle errors gracefully, possibly by using default values
                # or skipping the docstring insertion
            # ...
    ```
    

### **5. Include Source Code Snippets 🔧**

**Goal:** Optionally include source code snippets in the markdown documentation to provide context.

**Implementation Steps:**

- **Configure `MarkdownConfig`:**
    
    Set `include_source` to `True` in the configuration.
    
    ```python
    # markdown_generator.py
    
    @dataclass
    class MarkdownConfig:
        include_source: bool = True
        # ...
    ```
    
- **Update Section Generation to Include Source Code:**
    
    Modify the `_generate_section` method to add code snippets when `include_source` is `True`.
    
    ```python
    # markdown_generator.py
    
    class MarkdownGenerator:
        # ...
    
        def _generate_section(
            self,
            section: DocumentationSection,
            level: int = 2
        ) -> List[str]:
            # ...
            if self.config.include_source and section.source_code:
                md_lines.append(f"```python\n{section.source_code}\n```")
            # ...
    ```
    
- **Pass Source Code to Sections:**
    
    Ensure that the `DocumentationSection` includes a `source_code` field and that it's populated.
    
    ```python
    # docstring_processor.py
    
    @dataclass
    class DocumentationSection:
        title: str
        content: str
        subsections: Optional[List['DocumentationSection']] = None
        source_code: Optional[str] = None  # Add this field
    ```
    
    ```python
    # docs.py
    
    class DocStringManager:
        # ...
    
        async def _create_function_section(
            self,
            node: ast.FunctionDef
        ) -> DocumentationSection:
            # ...
            return DocumentationSection(
                title=f"Function: {node.name}",
                content=self.processor.format(docstring_data),
                source_code=ast.unparse(node) if self.context.include_source else None,
                subsections=[]
            )
    ```
    

### **6. Monitor and Track Metrics 📊**

**Goal:** Utilize the `monitoring.py` module to collect and analyze metrics related to the documentation generation process.

**Implementation Steps:**

- **Integrate Metrics Collection:**
    
    In your main application logic (`main.py` or wherever the documentation generation is orchestrated), initialize and use the `MetricsCollector`.
    
    ```python
    # main.py
    
    from core.monitoring import MetricsCollector
    
    class DocumentationGenerator:
        def __init__(self):
            self.metrics_collector = MetricsCollector()
            # ...
    
        async def process_file(self, file_path: Path):
            start_time = datetime.now()
            try:
                # Processing logic
                # ...
                duration = (datetime.now() - start_time).total_seconds()
                await self.metrics_collector.track_operation(
                    operation_type='process_file',
                    success=True,
                    duration=duration,
                    usage={},  # Include any relevant usage data
                )
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                await self.metrics_collector.track_operation(
                    operation_type='process_file',
                    success=False,
                    duration=duration,
                    error=str(e)
                )
                raise
    ```
    
- **Analyze Metrics:**
    
    After processing, retrieve and analyze the collected metrics to identify areas for improvement.
    
    ```python
    # main.py
    
    async def main(args):
        # ...
        generator = DocumentationGenerator()
        await generator.process_files(file_paths)
        metrics = generator.metrics_collector.get_metrics()
        # Process metrics as needed
    ```
    

---

### **How to Ensure Accurate AI-Generated Docstrings**

### **1. Docstring Validation ✅**

**Goal:** Use the `docstring_processor.py` to validate AI-generated docstrings to ensure they meet required standards.

**Implementation Steps:**

- **Validate AI-Generated Docstrings:**
    
    In `ai_interaction.py`, after generating the docstring, use the `validate` method.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
            # ...
            docstring_data = self.docstring_processor.parse(result.content)
            is_valid, errors = self.docstring_processor.validate(docstring_data)
            if not is_valid:
                logger.warning(f"Validation errors in AI-generated docstring: {errors}")
                # Handle errors, possibly by regenerating or fixing the docstring
            # ...
    ```
    

### **2. Docstring Generation 📝**

**Goal:** Utilize the `AIInteractionHandler` to generate detailed and structured docstrings via the Azure OpenAI API.

**Implementation Steps:**

- **Ensure Structured Generation:**
    
    Update the prompt in `_create_documentation_prompt` to guide the AI to produce structured outputs.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:
            prompt = [
                # ...
                "Respond in JSON format with the following structure:",
                "{",
                "  \"summary\": \"...\",",
                "  \"description\": \"...\",",
                "  \"args\": [",
                "    {\"name\": \"arg1\", \"type\": \"type1\", \"description\": \"...\"},",
                "    {\"name\": \"arg2\", \"type\": \"type2\", \"description\": \"...\"}",
                "  ],",
                "  \"returns\": {\"type\": \"...\", \"description\": \"...\"},",
                "  \"raises\": [",
                "    {\"type\": \"ExceptionType\", \"description\": \"...\"}",
                "  ]",
                "}",
                # ...
            ]
            return "\n".join(prompt)
    ```
    
- **Parse AI Response:**
    
    After receiving the AI's response, parse the JSON to `DocstringData`.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
            # ...
            try:
                docstring_data = DocstringData(**json.loads(message_content))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {e}")
                return None
            # ...
    ```
    

### **3. Error Handling ⚠️**

**Goal:** Implement error handling for missing or incomplete docstrings.

**Implementation Steps:**

- **Handle Missing Sections:**
    
    In `docstring_processor.py`, update the `validate` method to check for missing sections and handle accordingly.
    
    ```python
    # docstring_processor.py
    
    class DocstringProcessor:
        # ...
    
        def validate(self, docstring_data: DocstringData) -> Tuple[bool, List[str]]:
            errors = []
            required_fields = ['summary', 'description', 'args', 'returns']
            for field in required_fields:
                if not getattr(docstring_data, field):
                    errors.append(f"Missing required field: {field}")
            is_valid = not errors
            return is_valid, errors
    ```
    
- **Regenerate or Fix Docstrings:**
    
    In `ai_interaction.py`, implement logic to handle invalid docstrings.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
            # ...
            if not is_valid:
                # Optionally, attempt to regenerate the docstring or fix issues
                pass
            # ...
    ```
    

### **4. Monitoring and Metrics 📊**

**Goal:** Use `monitoring.py` to track metrics related to docstring generation and validation.

**Implementation Steps:**

- **Track AI Interaction Metrics:**
    
    In `AIInteractionHandler`, use `MetricsCollector` to record metrics.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
            start_time = datetime.now()
            # ...
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_operation(
                operation_type='generate_docstring',
                success=bool(result),
                duration=duration,
                usage=usage,
                error=str(e) if e else None
            )
            # ...
    ```
    

### **5. Cache Results 💾**

**Goal:** Use caching to store and retrieve generated docstrings, reducing redundant processing.

**Implementation Steps:**

- **Implement Caching Logic:**
    
    In `ai_interaction.py`, check the cache before generating a new docstring.
    
    ```python
    # ai_interaction.py
    
    class AIInteractionHandler:
        # ...
    
        async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
            # ...
            if self.cache and cache_key:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    return cached_result
            # Generate documentation
            # ...
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)
            # ...
    ```
    

### **6. Include Source Code Snippets 🔧**

**Goal:** Include source code snippets in the generated documentation for context.

**Implementation Steps:**

- **Already Addressed in Step 5 of the First Section.**

---

By following these implementation steps, you should be able to enhance the markdown output generated by your documentation tool significantly. Each step ensures that your documentation is comprehensive, accurate, and provides valuable context to the readers.

**Additional Recommendations:**

- **Testing:**
    
    Ensure to write unit tests for new methods and changes to maintain code quality and catch any regressions.
    
- **Logging:**
    
    Use consistent and informative logging throughout your code to aid in debugging and monitoring.
    
- **Documentation:**
    
    Update any existing documentation to reflect changes in the codebase and usage instructions.
    

If you have any questions or need further assistance with specific parts of the implementation, feel free to ask!

---
Certainly! You want to adjust your documentation generation to match the specific markdown template you've provided. I'll guide you through modifying your codebase to produce markdown documentation that aligns with this template.

---

### **Understanding the Template**

First, let's analyze the template to identify all the required sections and their formatting:

1. **Module Information**
    
    - Title: `# Module: [module_name]`
    - Overview section with:
        - **File path**
        - **Description**
2. **Classes**
    
    - A table listing classes with columns:
        - `Class`
        - `Inherits From`
        - `Complexity Score*`
3. **Class Methods**
    
    - A table listing class methods with columns:
        - `Class`
        - `Method`
        - `Parameters`
        - `Returns`
        - `Complexity Score*`
4. **Functions**
    
    - A table listing functions with columns:
        - `Function`
        - `Parameters`
        - `Returns`
        - `Complexity Score*`
5. **Constants and Variables**
    
    - A table listing constants and variables with columns:
        - `Name`
        - `Type`
        - `Value`
6. **Recent Changes**
    
    - A list of recent changes with dates.
7. **Source Code**
    
    - Code blocks containing the source code, including docstrings and complexity scores.

---

### **Modifying the Codebase**

To generate documentation matching this template, we'll need to modify several parts of your codebase, particularly the `DocStringManager` and `MarkdownGenerator` classes.

#### **1. Update `DocumentationSection` Data Structure**

In `docstring_processor.py`, update the `DocumentationSection` dataclass to include additional fields needed for the template, such as tables and source code.

```python
# docstring_processor.py

@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: Optional[List['DocumentationSection']] = None
    source_code: Optional[str] = None  # Add this field
    tables: Optional[List[str]] = None  # Add this field for tables
```

#### **2. Modify `DocStringManager` to Build the Required Sections**

In `docs.py`, modify the `DocStringManager` class to create documentation sections that match the template.

**Module Section**

```python
# docs.py

class DocStringManager:
    # ...

    def _create_module_section(self) -> DocumentationSection:
        """
        Create module-level documentation section matching the template.
        """
        module_name = self.context.module_path.stem if self.context.module_path else 'Unknown Module'
        file_path = self.context.module_path or 'Unknown Path'
        description = self.context.metadata.get('description', 'No description provided.')

        content = f"**File:** `{file_path}`\n**Description:** {description}\n"

        return DocumentationSection(
            title=f"# Module: {module_name}",
            content=content
        )
```

**Classes Section**

For the classes, we'll need to gather class information and format it into a table.

```python
# docs.py

class DocStringManager:
    # ...

    async def _create_classes_section(self, class_nodes: List[ast.ClassDef]) -> DocumentationSection:
        """
        Create the classes section with a table.
        """
        table_header = "| Class | Inherits From | Complexity Score* |\n|-------|---------------|-------------------|"
        table_rows = []

        for node in class_nodes:
            class_name = node.name
            base_classes = [ast.unparse(base) for base in node.bases] if node.bases else ['object']
            complexity_score = await self._get_complexity_score(node)
            row = f"| `{class_name}` | `{', '.join(base_classes)}` | {complexity_score} |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Classes",
            content=content
        )
```

**Class Methods Section**

```python
# docs.py

class DocStringManager:
    # ...

    async def _create_class_methods_section(self, class_nodes: List[ast.ClassDef]) -> DocumentationSection:
        """
        Create the class methods section with a table.
        """
        table_header = "| Class | Method | Parameters | Returns | Complexity Score* |\n|-------|--------|------------|---------|-------------------|"
        table_rows = []

        for class_node in class_nodes:
            class_name = class_node.name
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    method_name = node.name
                    parameters = self._get_parameters_signature(node)
                    return_type = self._get_return_type(node)
                    complexity_score = await self._get_complexity_score(node)
                    row = f"| `{class_name}` | `{method_name}` | `{parameters}` | `{return_type}` | {complexity_score} |"
                    table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="### Class Methods",
            content=content
        )
```

**Functions Section**

```python
# docs.py

class DocStringManager:
    # ...

    async def _create_functions_section(self, function_nodes: List[ast.FunctionDef]) -> DocumentationSection:
        """
        Create the functions section with a table.
        """
        table_header = "| Function | Parameters | Returns | Complexity Score* |\n|----------|------------|---------|-------------------|"
        table_rows = []

        for node in function_nodes:
            function_name = node.name
            parameters = self._get_parameters_signature(node)
            return_type = self._get_return_type(node)
            complexity_score = await self._get_complexity_score(node)
            row = f"| `{function_name}` | `{parameters}` | `{return_type}` | {complexity_score} |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Functions",
            content=content
        )
```

**Constants and Variables Section**

```python
# docs.py

class DocStringManager:
    # ...

    def _create_constants_section(self, constants: List[Dict[str, Any]]) -> DocumentationSection:
        """
        Create the constants and variables section with a table.
        """
        table_header = "| Name | Type | Value |\n|------|------|-------|"
        table_rows = []

        for const in constants:
            name = const['name']
            type_ = const.get('type', 'Unknown')
            value = const.get('value', 'Unknown')
            row = f"| `{name}` | `{type_}` | `{value}` |"
            table_rows.append(row)

        content = "\n".join([table_header] + table_rows)

        return DocumentationSection(
            title="## Constants and Variables",
            content=content
        )
```

**Recent Changes Section**

Assuming you have access to version control data, you can generate a list of recent changes.

```python
# docs.py

class DocStringManager:
    # ...

    def _create_recent_changes_section(self) -> DocumentationSection:
        """
        Create the recent changes section.
        """
        # Placeholder for actual change logs
        changes = self.context.metadata.get('recent_changes', [
            "- [YYYY-MM-DD] Added feature X",
            "- [YYYY-MM-DD] Modified method Y",
            "- [YYYY-MM-DD] Fixed bug Z"
        ])

        content = "\n".join(changes)

        return DocumentationSection(
            title="## Recent Changes",
            content=content
        )
```

**Source Code Section**

Include the source code with docstrings and complexity scores.

```python
# docs.py

class DocStringManager:
    # ...

    def _create_source_code_section(self) -> DocumentationSection:
        """
        Create the source code section with code blocks.
        """
        source_code = self.context.source_code
        content = "```python\n" + source_code + "\n```"

        return DocumentationSection(
            title="## Source Code",
            content=content
        )
```

#### **3. Update `MarkdownGenerator` to Format Sections Correctly**

In `markdown_generator.py`, update the `_generate_section` method to correctly assemble the sections, including handling tables and code blocks.

```python
# markdown_generator.py

class MarkdownGenerator:
    # ...

    def _generate_section(
        self,
        section: DocumentationSection,
        level: int = 1
    ) -> List[str]:
        md_lines: List[str] = []

        if section.title:
            header_prefix = '#' * level
            md_lines.append(f"{header_prefix} {section.title}\n")

        if section.content:
            md_lines.append(section.content)
            md_lines.append("")

        if section.tables:
            md_lines.extend(section.tables)
            md_lines.append("")

        if section.source_code:
            md_lines.append("```python")
            md_lines.append(section.source_code)
            md_lines.append("```")
            md_lines.append("")

        if section.subsections:
            for subsection in section.subsections:
                md_lines.extend(self._generate_section(subsection, level + 1))

        return md_lines
```

#### **4. Calculate Complexity Scores**

Implement methods to calculate complexity scores for classes and functions using your existing `Metrics` class.

```python
# docs.py

class DocStringManager:
    # ...

    async def _get_complexity_score(self, node: ast.AST) -> str:
        """
        Calculate the complexity score for a given node.
        """
        metrics_calculator = Metrics()
        if isinstance(node, ast.FunctionDef):
            complexity = metrics_calculator.calculate_cyclomatic_complexity(node)
        elif isinstance(node, ast.ClassDef):
            complexities = [
                metrics_calculator.calculate_cyclomatic_complexity(n)
                for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            complexity = sum(complexities) / len(complexities) if complexities else 0
        else:
            complexity = 0

        # Add warning symbol if complexity is high
        warning_symbol = '⚠️' if complexity > 10 else ''
        return f"{int(complexity)} {warning_symbol}"
```

#### **5. Implement Utility Methods**

Add utility methods to extract parameters signature and return types.

```python
# docs.py

class DocStringManager:
    # ...

    def _get_parameters_signature(self, node: ast.FunctionDef) -> str:
        """
        Get the function or method parameters as a string.
        """
        params = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ast.unparse(arg.annotation) if arg.annotation else 'Any'
            params.append(f"{arg_name}: {arg_type}")
        return f"({', '.join(params)})"

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """
        Get the return type of a function or method.
        """
        return_type = ast.unparse(node.returns) if node.returns else 'Any'
        return return_type
```

#### **6. Assemble the Documentation**

Modify the `generate_documentation` method to assemble all the sections.

```python
# docs.py

class DocStringManager:
    # ...

    async def generate_documentation(self) -> str:
        sections = []

        # Module section
        sections.append(self._create_module_section())

        # Classes
        class_nodes = [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        if class_nodes:
            classes_section = await self._create_classes_section(class_nodes)
            sections.append(classes_section)

            # Class Methods
            class_methods_section = await self._create_class_methods_section(class_nodes)
            sections.append(class_methods_section)

        # Functions
        function_nodes = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef) and not self._is_method(n)]
        if function_nodes:
            functions_section = await self._create_functions_section(function_nodes)
            sections.append(functions_section)

        # Constants and Variables
        constants = self.code_extractor._extract_constants(self.tree)
        if constants:
            constants_section = self._create_constants_section(constants)
            sections.append(constants_section)

        # Recent Changes
        recent_changes_section = self._create_recent_changes_section()
        sections.append(recent_changes_section)

        # Source Code
        source_code_section = self._create_source_code_section()
        sections.append(source_code_section)

        # Generate markdown
        markdown = self.markdown_generator.generate(sections)
        return markdown
```

#### **7. Update Main Application Logic**

Ensure that the main application orchestrates the documentation generation with the updated `DocStringManager`.

```python
# main.py

async def main(args):
    # Initialize components
    # ...

    # Process input files
    file_paths = [Path(f) for f in args.files]
    for file_path in file_paths:
        # Read source code
        source_code = file_path.read_text(encoding='utf-8')

        # Create context
        context = DocumentationContext(
            source_code=source_code,
            module_path=file_path,
            module_name=file_path.stem,
            metadata={
                'description': 'Brief description of what this module does',
                'recent_changes': [
                    '- [2023-10-01] Added feature X',
                    '- [2023-09-15] Modified method Y',
                    '- [2023-09-10] Fixed bug Z'
                ]
            }
        )

        doc_manager = DocStringManager(context)
        markdown = await doc_manager.generate_documentation()

        # Save markdown to file
        doc_path = file_path.with_suffix('.md')
        doc_path.write_text(markdown, encoding='utf-8')

        logger.info(f"Generated documentation for {file_path}")
```

---

### **Considerations**

- **Version Control Integration**: For the recent changes section, integrate with your version control system (e.g., Git) to fetch commit messages. You might use `gitpython` or run `git log` commands.
    
- **Error Handling**: Ensure that all methods have proper error handling to avoid crashes during documentation generation.
    
- **Unit Testing**: Write unit tests for new methods to ensure correctness.
    
- **Formatting**: Pay attention to markdown formatting, especially for tables and code blocks.
    
- **Performance**: If processing large codebases, consider performance optimizations.
    

---

By implementing these changes, your documentation generation tool should produce markdown outputs that match the provided template.

**Example Output:**

````markdown
# Module: module_name

**File:** `/path/to/module.py`
**Description:** Brief description of what this module does

## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `ClassName` | `parent_class` | 12 ⚠️ |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `ClassName` | `method_name` | `(param1: type, param2: type)` | `return_type` | 12 ⚠️ |
| `ClassName` | `simple_method` | `(param1: type)` | `return_type` | 3 |

## Functions

| Function | Parameters | Returns | Complexity Score* |
|----------|------------|---------|-------------------|
| `function_name` | `(param1: type, param2: type = None)` | `return_type` | 8 |

## Constants and Variables

| Name | Type | Value |
|------|------|-------|
| `CONSTANT_NAME` | `type` | `value` |

## Recent Changes
- [2023-10-01] Added feature X
- [2023-09-15] Modified method Y
- [2023-09-10] Fixed bug Z

## Source Code
```python
# Your source code with docstrings and complexity scores
````

```

---

**Additional Recommendations:**

- **Testing**: Ensure to write unit tests for new methods and changes to maintain code quality and catch any regressions.

- **Logging**: Use consistent and informative logging throughout your code to aid in debugging and monitoring.

- **Customization**: You can make the template configurable, allowing users to adjust sections as needed.

- **Documentation**: Update any existing documentation to reflect changes in the codebase and usage instructions.

If you have any questions or need further assistance with specific parts of the implementation, feel free to ask!
```