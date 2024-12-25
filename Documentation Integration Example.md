```python
class DocumentationManager:
    """
    Enhanced documentation manager with integrated docstring handling.
    """
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Initialize docstring manager with default style
        self.docstring_manager = DocStringManager()
        
    async def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a single Python file with enhanced docstring handling.
        """
        log_debug(f"Processing file: {file_path}")
        start_time = time.time()

        try:
            file_path = Path(file_path)
            if not file_path.exists() or file_path.suffix != '.py':
                self.logger.error(f"Invalid Python file: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the source code
            tree = ast.parse(source)
            
            # Process classes and functions with docstring preservation
            documentation_entries = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    try:
                        # Generate unique identifier
                        identifier = f"{file_path.stem}:{node.name}"
                        
                        # Get existing docstring
                        existing_docstring = ast.get_docstring(node)
                        
                        if existing_docstring:
                            # Process existing docstring
                            processed_docstring = self.docstring_manager.process_docstring(
                                existing_docstring,
                                identifier=identifier,
                                preserve=True
                            )
                        else:
                            processed_docstring = None

                        # Create documentation entry
                        entry = {
                            'name': node.name,
                            'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                            'docstring': processed_docstring or "No documentation available.",
                            'lineno': node.lineno
                        }

                        # Extract additional metadata
                        if isinstance(node, ast.ClassDef):
                            entry.update(self._process_class(node))
                        else:
                            entry.update(self._process_function(node))

                        documentation_entries.append(entry)
                        
                    except Exception as e:
                        log_error(f"Error processing {node.name}: {e}")
                        continue

            # Generate markdown documentation
            markdown_content = self._generate_markdown(file_path, documentation_entries)
            
            # Save documentation
            output_path = self.output_dir / f"{file_path.stem}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Report results
            monitor.log_operation_complete(
                str(file_path),
                time.time() - start_time,
                len(documentation_entries)
            )

            return markdown_content

        except Exception as e:
            log_error(f"Error processing file {file_path}: {e}")
            monitor.log_request(str(file_path), "error", time.time() - start_time, error=str(e))
            return None

    def _process_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Process a class definition with docstring preservation."""
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                identifier = f"{node.name}.{item.name}"
                docstring = ast.get_docstring(item)
                if docstring:
                    processed = self.docstring_manager.process_docstring(
                        docstring,
                        identifier=identifier,
                        preserve=True
                    )
                    methods[item.name] = processed

        return {
            'methods': methods,
            'bases': [ast.unparse(base) for base in node.bases],
            'decorators': [ast.unparse(dec) for dec in node.decorator_list]
        }

    def _process_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Process a function definition with docstring preservation."""
        return {
            'params': [(arg.arg, self._get_type_annotation(arg)) for arg in node.args.args],
            'returns': ast.unparse(node.returns) if node.returns else None,
            'decorators': [ast.unparse(dec) for dec in node.decorator_list],
            'is_method': isinstance(node.parent, ast.ClassDef) if hasattr(node, 'parent') else False
        }

    def _generate_markdown(self, file_path: Path, entries: List[Dict[str, Any]]) -> str:
        """Generate markdown documentation with enhanced formatting."""
        markdown_gen = MarkdownGenerator()
        
        # Add header
        markdown_gen.add_header(f"Documentation for {file_path.name}")
        
        # Process module docstring if present
        module_doc = self.docstring_manager.extract_sections(
            entries[0].get('docstring', '') if entries else ''
        ).get('Description', '')
        
        if module_doc:
            markdown_gen.add_section("Module Description", module_doc)

        # Group entries by type
        classes = [e for e in entries if e['type'] == 'class']
        functions = [e for e in entries if e['type'] == 'function']

        # Document classes
        if classes:
            markdown_gen.add_header("Classes", level=2)
            for class_entry in classes:
                markdown_gen.add_header(class_entry['name'], level=3)
                markdown_gen.add_code_block(class_entry['docstring'], language="python")
                
                # Document methods
                if class_entry['methods']:
                    markdown_gen.add_header("Methods", level=4)
                    for method_name, method_doc in class_entry['methods'].items():
                        markdown_gen.add_header(method_name, level=5)
                        markdown_gen.add_code_block(method_doc, language="python")

        # Document functions
        if functions:
            markdown_gen.add_header("Functions", level=2)
            for func_entry in functions:
                markdown_gen.add_header(func_entry['name'], level=3)
                markdown_gen.add_code_block(func_entry['docstring'], language="python")

        return markdown_gen.generate_markdown()

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Get type annotation as a string."""
        return ast.unparse(arg.annotation) if arg.annotation else "Any"
```