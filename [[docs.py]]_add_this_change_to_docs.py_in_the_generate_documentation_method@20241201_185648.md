---
epoch: 1733101008889
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: [docs.py](docs.py.md) add this change to docs.py in the generate_documentation method:
async def generate_documentation(self) -> str:
    """Generate complete documentation."""
    try:
        # ... (other code)

        # Format changes
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', 'No description provided.')
                        })
                    elif isinstance(change, str):
                        changes.append({  # Create a dictionary here!
                            'date': current_date,
                            'description': change
                        })
                    else:
                        self.logger.warning(f"Skipping invalid change entry: {change}")

        # ... (rest of the code)
[Timestamp: 2024/12/01 18:56:48]

**ai**: To incorporate the changes into the `generate_documentation` method of `docs.py`, you need to update the section that formats changes. Here's how you can modify the method:

1. Locate the `generate_documentation` method in your `docs.py` file.
2. Replace the existing section that formats changes with the new code you provided.

Here's the updated section for formatting changes:

```python
# Format changes
changes = []
if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
    raw_changes = self.context.metadata.get('changes', [])
    if raw_changes:
        current_date = datetime.now().strftime('%Y-%m-%d')
        for change in raw_changes:
            if isinstance(change, dict):
                changes.append({
                    'date': change.get('date', current_date),
                    'description': change.get('description', 'No description provided.')
                })
            elif isinstance(change, str):
                changes.append({  # Create a dictionary here!
                    'date': current_date,
                    'description': change
                })
            else:
                self.logger.warning(f"Skipping invalid change entry: {change}")
```

Make sure to replace the existing code in the `generate_documentation` method with this updated version to ensure the changes are applied correctly.
[Timestamp: 2024/12/01 18:56:53]

**user**: show the full method
[Timestamp: 2024/12/01 18:57:01]

**ai**: Certainly! Here's the full `generate_documentation` method with the updated changes section:

```python
async def generate_documentation(self) -> str:
    """
    Generate complete documentation.

    Returns:
        str: The generated documentation in markdown format.

    Raises:
        DocumentationError: If documentation generation fails.
    """
    try:
        # Extract code elements
        extraction_result = self.code_extractor.extract_code(
            self.context.source_code
        )
        if not extraction_result:
            raise DocumentationError("Code extraction failed")

        # Format constants properly
        formatted_constants = []
        if extraction_result.constants:
            for const in extraction_result.constants:
                if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                    formatted_constants.append(const)
                else:
                    try:
                        if isinstance(const, tuple):
                            name, value, const_type = const
                        else:
                            name = const
                            value = const
                            const_type = type(const).__name__
                        
                        formatted_constants.append({
                            'name': str(name),
                            'type': str(const_type),
                            'value': str(value)
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed constant: {const} - {str(e)}")

        # Format changes
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', 'No description provided.')
                        })
                    elif isinstance(change, str):
                        changes.append({  # Create a dictionary here!
                            'date': current_date,
                            'description': change
                        })
                    else:
                        self.logger.warning(f"Skipping invalid change entry: {change}")

        # Format classes
        formatted_classes = []
        if extraction_result.classes:
            for cls in extraction_result.classes:
                if isinstance(cls, dict):
                    formatted_classes.append(cls)
                else:
                    try:
                        formatted_classes.append({
                            'name': cls.name,
                            'docstring': cls.docstring,
                            'methods': cls.methods,
                            'bases': cls.bases if hasattr(cls, 'bases') else [],
                            'metrics': cls.metrics if hasattr(cls, 'metrics') else {},
                            'source': cls.source if hasattr(cls, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed class: {cls} - {str(e)}")

        # Format functions
        formatted_functions = []
        if extraction_result.functions:
            for func in extraction_result.functions:
                if isinstance(func, dict):
                    formatted_functions.append(func)
                else:
                    try:
                        formatted_functions.append({
                            'name': func.name,
                            'docstring': func.docstring,
                            'args': func.args,
                            'return_type': func.return_type if hasattr(func, 'return_type') else None,
                            'metrics': func.metrics if hasattr(func, 'metrics') else {},
                            'source': func.source if hasattr(func, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed function: {func} - {str(e)}")

        # Create documentation context
        doc_context = {
            'module_name': (
                self.context.module_path.stem
                if self.context.module_path else "Unknown"
            ),
            'file_path': (
                str(self.context.module_path)
                if self.context.module_path else ""
            ),
            'description': (
                extraction_result.module_docstring
                or "No description available."
            ),
            'classes': formatted_classes,
            'functions': formatted_functions,
            'constants': formatted_constants,
            'metrics': extraction_result.metrics or {},
            'source_code': (
                self.context.source_code
                if self.context.include_source else None
            ),
            'changes': changes,
            'imports': extraction_result.imports if hasattr(extraction_result, 'imports') else {},
            'ai_docs': (
                self.context.metadata.get('ai_generated')
                if hasattr(self.context, 'metadata') else None
            )
        }

        # Add any additional metadata from context
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            for key, value in self.context.metadata.items():
                if key not in doc_context:
                    doc_context[key] = value

        # Generate markdown using the markdown generator
        return self.markdown_generator.generate(doc_context)

    except Exception as e:
        self.logger.error(f"Documentation generation failed: {e}")
        raise DocumentationError(f"Failed to generate documentation: {e}")
```

This method includes the updated logic for formatting changes as you specified.
[Timestamp: 2024/12/01 18:57:17]

**user**: addd this change to the method as well: async def generate_documentation(self) -> str:
    # ... (other code)

    # Format changes - More robust handling
    changes = []
    if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
        raw_changes = self.context.metadata.get('changes', [])
        if raw_changes:
            current_date = datetime.now().strftime('%Y-%m-%d')

            for change in raw_changes:
                if isinstance(change, (dict, str, list, tuple)):  # Explicitly handle expected types
                    changes.append(change) # Append directly, let _generate_changes_section handle formatting
                else:
                    self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")

    # ... (rest of the code)
[Timestamp: 2024/12/01 18:59:54]

**ai**: Certainly! Here's the updated `generate_documentation` method with the new logic for handling changes:

```python
async def generate_documentation(self) -> str:
    """
    Generate complete documentation.

    Returns:
        str: The generated documentation in markdown format.

    Raises:
        DocumentationError: If documentation generation fails.
    """
    try:
        # Extract code elements
        extraction_result = self.code_extractor.extract_code(
            self.context.source_code
        )
        if not extraction_result:
            raise DocumentationError("Code extraction failed")

        # Format constants properly
        formatted_constants = []
        if extraction_result.constants:
            for const in extraction_result.constants:
                if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                    formatted_constants.append(const)
                else:
                    try:
                        if isinstance(const, tuple):
                            name, value, const_type = const
                        else:
                            name = const
                            value = const
                            const_type = type(const).__name__
                        
                        formatted_constants.append({
                            'name': str(name),
                            'type': str(const_type),
                            'value': str(value)
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed constant: {const} - {str(e)}")

        # Format changes - More robust handling
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')

                for change in raw_changes:
                    if isinstance(change, (dict, str, list, tuple)):  # Explicitly handle expected types
                        changes.append(change)  # Append directly, let _generate_changes_section handle formatting
                    else:
                        self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")

        # Format classes
        formatted_classes = []
        if extraction_result.classes:
            for cls in extraction_result.classes:
                if isinstance(cls, dict):
                    formatted_classes.append(cls)
                else:
                    try:
                        formatted_classes.append({
                            'name': cls.name,
                            'docstring': cls.docstring,
                            'methods': cls.methods,
                            'bases': cls.bases if hasattr(cls, 'bases') else [],
                            'metrics': cls.metrics if hasattr(cls, 'metrics') else {},
                            'source': cls.source if hasattr(cls, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed class: {cls} - {str(e)}")

        # Format functions
        formatted_functions = []
        if extraction_result.functions:
            for func in extraction_result.functions:
                if isinstance(func, dict):
                    formatted_functions.append(func)
                else:
                    try:
                        formatted_functions.append({
                            'name': func.name,
                            'docstring': func.docstring,
                            'args': func.args,
                            'return_type': func.return_type if hasattr(func, 'return_type') else None,
                            'metrics': func.metrics if hasattr(func, 'metrics') else {},
                            'source': func.source if hasattr(func, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed function: {func} - {str(e)}")

        # Create documentation context
        doc_context = {
            'module_name': (
                self.context.module_path.stem
                if self.context.module_path else "Unknown"
            ),
            'file_path': (
                str(self.context.module_path)
                if self.context.module_path else ""
            ),
            'description': (
                extraction_result.module_docstring
                or "No description available."
            ),
            'classes': formatted_classes,
            'functions': formatted_functions,
            'constants': formatted_constants,
            'metrics': extraction_result.metrics or {},
            'source_code': (
                self.context.source_code
                if self.context.include_source else None
            ),
            'changes': changes,
            'imports': extraction_result.imports if hasattr(extraction_result, 'imports') else {},
            'ai_docs': (
                self.context.metadata.get('ai_generated')
                if hasattr(self.context, 'metadata') else None
            )
        }

        # Add any additional metadata from context
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            for key, value in self.context.metadata.items():
                if key not in doc_context:
                    doc_context[key] = value

        # Generate markdown using the markdown generator
        return self.markdown_generator.generate(doc_context)

    except Exception as e:
        self.logger.error(f"Documentation generation failed: {e}")
        raise DocumentationError(f"Failed to generate documentation: {e}")
```

This version includes the updated logic for more robust handling of changes, allowing for different types of change entries.
[Timestamp: 2024/12/01 19:00:04]

**user**: add this modification to [markdown_generator.py](markdown_generator.py.md) def _generate_changes_section(self, context: Dict[str, Any]) -> str:
    """Generate the recent changes section, with robust handling of different data types."""
    if 'changes' not in context or not context['changes']:
        return ""

    changes_data = context['changes']
    lines = ["## Recent Changes"]

    for change in changes_data:
        # Provide default values if keys are missing
        date = change.get('date', datetime.now().strftime('%Y-%m-%d'))
        description = change.get('description', "")

        # Ensure date and description are strings
        date = str(date)
        description = str(description)

        lines.append(f"- [{date}] {description}")

    return "\n".join(lines) if len(lines) > 1 else ""
[Timestamp: 2024/12/01 19:04:46]

**ai**: To incorporate the changes into the `_generate_changes_section` method of `markdown_generator.py`, you need to update the method with the provided robust handling logic. Here's the updated method:

```python
def _generate_changes_section(self, context: Dict[str, Any]) -> str:
    """Generate the recent changes section, with robust handling of different data types."""
    if 'changes' not in context or not context['changes']:
        return ""

    changes_data = context['changes']
    lines = ["## Recent Changes"]

    for change in changes_data:
        # Provide default values if keys are missing
        date = change.get('date', datetime.now().strftime('%Y-%m-%d'))
        description = change.get('description', "")

        # Ensure date and description are strings
        date = str(date)
        description = str(description)

        lines.append(f"- [{date}] {description}")

    return "\n".join(lines) if len(lines) > 1 else ""
```

Make sure to replace the existing `_generate_changes_section` method in your `markdown_generator.py` file with this updated version to ensure the changes are applied correctly.
[Timestamp: 2024/12/01 19:04:50]

**user**: add this change to generate_documentation in docs.py: # Format changes (robust handling)
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        # Ensure 'date' and 'description' keys are present, even if empty
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', "")
                        })
                    elif isinstance(change, str):
                        changes.append({'date': current_date, 'description': change})
                    elif isinstance(change, (list, tuple)):  # Handle potential lists/tuples
                        try:
                            changes.append({'date': change[0], 'description': change[1]})
                        except IndexError:  # Handle incorrect list/tuple structure
                            self.logger.warning(f"Invalid change format (list/tuple): {change}")
                            changes.append({'date': current_date, 'description': str(change)}) # Fallback
                    else:
                        self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")
[Timestamp: 2024/12/01 19:05:22]

**ai**: Here's the updated `generate_documentation` method in `docs.py` with the new robust handling for changes:

```python
async def generate_documentation(self) -> str:
    """
    Generate complete documentation.

    Returns:
        str: The generated documentation in markdown format.

    Raises:
        DocumentationError: If documentation generation fails.
    """
    try:
        # Extract code elements
        extraction_result = self.code_extractor.extract_code(
            self.context.source_code
        )
        if not extraction_result:
            raise DocumentationError("Code extraction failed")

        # Format constants properly
        formatted_constants = []
        if extraction_result.constants:
            for const in extraction_result.constants:
                if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                    formatted_constants.append(const)
                else:
                    try:
                        if isinstance(const, tuple):
                            name, value, const_type = const
                        else:
                            name = const
                            value = const
                            const_type = type(const).__name__
                        
                        formatted_constants.append({
                            'name': str(name),
                            'type': str(const_type),
                            'value': str(value)
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed constant: {const} - {str(e)}")

        # Format changes (robust handling)
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        # Ensure 'date' and 'description' keys are present, even if empty
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', "")
                        })
                    elif isinstance(change, str):
                        changes.append({'date': current_date, 'description': change})
                    elif isinstance(change, (list, tuple)):  # Handle potential lists/tuples
                        try:
                            changes.append({'date': change[0], 'description': change[1]})
                        except IndexError:  # Handle incorrect list/tuple structure
                            self.logger.warning(f"Invalid change format (list/tuple): {change}")
                            changes.append({'date': current_date, 'description': str(change)})  # Fallback
                    else:
                        self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")

        # Format classes
        formatted_classes = []
        if extraction_result.classes:
            for cls in extraction_result.classes:
                if isinstance(cls, dict):
                    formatted_classes.append(cls)
                else:
                    try:
                        formatted_classes.append({
                            'name': cls.name,
                            'docstring': cls.docstring,
                            'methods': cls.methods,
                            'bases': cls.bases if hasattr(cls, 'bases') else [],
                            'metrics': cls.metrics if hasattr(cls, 'metrics') else {},
                            'source': cls.source if hasattr(cls, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed class: {cls} - {str(e)}")

        # Format functions
        formatted_functions = []
        if extraction_result.functions:
            for func in extraction_result.functions:
                if isinstance(func, dict):
                    formatted_functions.append(func)
                else:
                    try:
                        formatted_functions.append({
                            'name': func.name,
                            'docstring': func.docstring,
                            'args': func.args,
                            'return_type': func.return_type if hasattr(func, 'return_type') else None,
                            'metrics': func.metrics if hasattr(func, 'metrics') else {},
                            'source': func.source if hasattr(func, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed function: {func} - {str(e)}")

        # Create documentation context
        doc_context = {
            'module_name': (
                self.context.module_path.stem
                if self.context.module_path else "Unknown"
            ),
            'file_path': (
                str(self.context.module_path)
                if self.context.module_path else ""
            ),
            'description': (
                extraction_result.module_docstring
                or "No description available."
            ),
            'classes': formatted_classes,
            'functions': formatted_functions,
            'constants': formatted_constants,
            'metrics': extraction_result.metrics or {},
            'source_code': (
                self.context.source_code
                if self.context.include_source else None
            ),
            'changes': changes,
            'imports': extraction_result.imports if hasattr(extraction_result, 'imports') else {},
            'ai_docs': (
                self.context.metadata.get('ai_generated')
                if hasattr(self.context, 'metadata') else None
            )
        }

        # Add any additional metadata from context
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            for key, value in self.context.metadata.items():
                if key not in doc_context:
                    doc_context[key] = value

        # Generate markdown using the markdown generator
        return self.markdown_generator.generate(doc_context)

    except Exception as e:
        self.logger.error(f"Documentation generation failed: {e}")
        raise DocumentationError(f"Failed to generate documentation: {e}")
```

This version includes the updated logic for robust handling of changes, ensuring different types of change entries are processed appropriately.
[Timestamp: 2024/12/01 19:05:45]

**user**: # In core/docs.py (DocStringManager.generate_documentation)
import json

# ... other code ...

        # Format changes (EXTREMELY robust handling)
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', "")
                        })
                    elif isinstance(change, str):
                        try:  # Attempt to parse as JSON
                            change_dict = json.loads(change)
                            if isinstance(change_dict, dict):
                                changes.append({
                                    'date': change_dict.get('date', current_date),
                                    'description': change_dict.get('description', "")
                                })
                            else:
                                self.logger.warning(f"String change is not a valid dictionary: {change}")
                                changes.append({'date': current_date, 'description': change})
                        except json.JSONDecodeError:
                            self.logger.warning(f"String change is not valid JSON: {change}")
                            changes.append({'date': current_date, 'description': change}) # Fallback
                    elif isinstance(change, (list, tuple)):
                        # ... (handle list/tuple as before)
                    else:
                        self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")

# In core/markdown_generator.py (_generate_changes_section) - No changes needed here
[Timestamp: 2024/12/01 19:08:28]

**ai**: Here's the updated `generate_documentation` method in `core/docs.py` with the extremely robust handling for changes, including JSON parsing for string entries:

```python
import json
from datetime import datetime

# ... other code ...

async def generate_documentation(self) -> str:
    """
    Generate complete documentation.

    Returns:
        str: The generated documentation in markdown format.

    Raises:
        DocumentationError: If documentation generation fails.
    """
    try:
        # Extract code elements
        extraction_result = self.code_extractor.extract_code(
            self.context.source_code
        )
        if not extraction_result:
            raise DocumentationError("Code extraction failed")

        # Format constants properly
        formatted_constants = []
        if extraction_result.constants:
            for const in extraction_result.constants:
                if isinstance(const, dict) and all(key in const for key in ['name', 'type', 'value']):
                    formatted_constants.append(const)
                else:
                    try:
                        if isinstance(const, tuple):
                            name, value, const_type = const
                        else:
                            name = const
                            value = const
                            const_type = type(const).__name__
                        
                        formatted_constants.append({
                            'name': str(name),
                            'type': str(const_type),
                            'value': str(value)
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed constant: {const} - {str(e)}")

        # Format changes (EXTREMELY robust handling)
        changes = []
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            raw_changes = self.context.metadata.get('changes', [])
            if raw_changes:
                current_date = datetime.now().strftime('%Y-%m-%d')
                for change in raw_changes:
                    if isinstance(change, dict):
                        changes.append({
                            'date': change.get('date', current_date),
                            'description': change.get('description', "")
                        })
                    elif isinstance(change, str):
                        try:  # Attempt to parse as JSON
                            change_dict = json.loads(change)
                            if isinstance(change_dict, dict):
                                changes.append({
                                    'date': change_dict.get('date', current_date),
                                    'description': change_dict.get('description', "")
                                })
                            else:
                                self.logger.warning(f"String change is not a valid dictionary: {change}")
                                changes.append({'date': current_date, 'description': change})
                        except json.JSONDecodeError:
                            self.logger.warning(f"String change is not valid JSON: {change}")
                            changes.append({'date': current_date, 'description': change})  # Fallback
                    elif isinstance(change, (list, tuple)):
                        try:
                            changes.append({'date': change[0], 'description': change[1]})
                        except IndexError:
                            self.logger.warning(f"Invalid change format (list/tuple): {change}")
                            changes.append({'date': current_date, 'description': str(change)})  # Fallback
                    else:
                        self.logger.warning(f"Skipping invalid change entry of type {type(change)}: {change}")

        # Format classes
        formatted_classes = []
        if extraction_result.classes:
            for cls in extraction_result.classes:
                if isinstance(cls, dict):
                    formatted_classes.append(cls)
                else:
                    try:
                        formatted_classes.append({
                            'name': cls.name,
                            'docstring': cls.docstring,
                            'methods': cls.methods,
                            'bases': cls.bases if hasattr(cls, 'bases') else [],
                            'metrics': cls.metrics if hasattr(cls, 'metrics') else {},
                            'source': cls.source if hasattr(cls, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed class: {cls} - {str(e)}")

        # Format functions
        formatted_functions = []
        if extraction_result.functions:
            for func in extraction_result.functions:
                if isinstance(func, dict):
                    formatted_functions.append(func)
                else:
                    try:
                        formatted_functions.append({
                            'name': func.name,
                            'docstring': func.docstring,
                            'args': func.args,
                            'return_type': func.return_type if hasattr(func, 'return_type') else None,
                            'metrics': func.metrics if hasattr(func, 'metrics') else {},
                            'source': func.source if hasattr(func, 'source') else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed function: {func} - {str(e)}")

        # Create documentation context
        doc_context = {
            'module_name': (
                self.context.module_path.stem
                if self.context.module_path else "Unknown"
            ),
            'file_path': (
                str(self.context.module_path)
                if self.context.module_path else ""
            ),
            'description': (
                extraction_result.module_docstring
                or "No description available."
            ),
            'classes': formatted_classes,
            'functions': formatted_functions,
            'constants': formatted_constants,
            'metrics': extraction_result.metrics or {},
            'source_code': (
                self.context.source_code
                if self.context.include_source else None
            ),
            'changes': changes,
            'imports': extraction_result.imports if hasattr(extraction_result, 'imports') else {},
            'ai_docs': (
                self.context.metadata.get('ai_generated')
                if hasattr(self.context, 'metadata') else None
            )
        }

        # Add any additional metadata from context
        if hasattr(self.context, 'metadata') and isinstance(self.context.metadata, dict):
            for key, value in self.context.metadata.items():
                if key not in doc_context:
                    doc_context[key] = value

        # Generate markdown using the markdown generator
        return self.markdown_generator.generate(doc_context)

    except Exception as e:
        self.logger.error(f"Documentation generation failed: {e}")
        raise DocumentationError(f"Failed to generate documentation: {e}")
```

This version includes the extremely robust handling for changes, with JSON parsing for string entries and handling of lists/tuples.
[Timestamp: 2024/12/01 19:08:42]