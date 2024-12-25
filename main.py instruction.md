# Main.py Integration Instructions

## 1. Required Modifications

### Update Imports
```python
from docstring.manager import DocStringManager
from docstring.config import DocStringConfig
```

### Add Configuration Loading
```python
def load_docstring_config(args):
    """Load DocString configuration."""
    config = DocStringConfig()
    if args.docstring_config:
        config.load_config(Path(args.docstring_config))
    return config

def setup_components(args):
    """Setup all required components."""
    docstring_config = load_docstring_config(args)
    client = AzureOpenAIClient()
    docstring_manager = DocStringManager(
        style_guide=docstring_config.get_style_guide(),
        preservation_dir=Path(args.output_dir) / '.docstring_preservation'
    )
    return client, docstring_manager
```

### Modify File Processing
```python
async def process_file(file_path: str, args: argparse.Namespace, client: AzureOpenAIClient, docstring_manager: DocStringManager) -> None:
    """Process a single Python file with enhanced docstring handling."""
    try:
        # Load and process source code
        source_code = load_source_file(file_path)
        
        # Extract metadata with docstring preservation
        extraction_manager = ExtractionManager()
        metadata = extraction_manager.extract_metadata(source_code)
        
        # Process with docstring preservation
        for class_data in metadata['classes']:
            identifier = f"{file_path}:class:{class_data['name']}"
            if docstring_manager.validate(class_data.get('docstring', '')):
                # Preserve existing docstring
                docstring_manager.process_docstring(
                    class_data['docstring'],
                    identifier=identifier,
                    preserve=True
                )
        
        for function_data in metadata['functions']:
            identifier = f"{file_path}:function:{function_data['name']}"
            if function_data.get('docstring'):
                # Preserve existing docstring
                docstring_manager.process_docstring(
                    function_data['docstring'],
                    identifier=identifier,
                    preserve=True
                )
        
        # Process with API if needed
        interaction_handler = InteractionHandler(
            client=client,
            docstring_manager=docstring_manager
        )
        updated_code, documentation = await interaction_handler.process_all_functions(source_code)
        
        if updated_code:
            # Save updated source
            save_updated_source(file_path, updated_code)
            
        if documentation:
            # Save documentation
            save_documentation(documentation, args.output_dir, file_path)
            
    except Exception as e:
        log_error(f"Error processing file {file_path}: {e}")
        raise
```

### Add Command Line Arguments
```python
def add_docstring_arguments(parser: argparse.ArgumentParser):
    """Add DocString-related command line arguments."""
    parser.add_argument(
        '--docstring-config',
        help='Path to docstring configuration file'
    )
    parser.add_argument(
        '--preserve-docstrings',
        action='store_true',
        help='Preserve existing docstring content'
    )
    parser.add_argument(
        '--docstring-style',
        choices=['google', 'numpy', 'sphinx'],
        default='google',
        help='Docstring style to use'
    )
```

## 2. Configuration File Example

Create `docstring_config.yaml`:
```yaml
format:
  style: google
  indentation: 4
  line_length: 80
  blank_lines_between_sections: 1
  section_order:
    - Description
    - Args
    - Returns
    - Raises
    - Examples
    - Notes

preservation:
  enabled: true
  storage_dir: .docstring_preservation
  preserve_custom_sections: true
  preserve_decorators: true
  preserve_examples: true
  ttl_days: 30

validation:
  enforce_style: true
  require_description: true
  require_param_description: true
  require_return_description: true
  require_examples: false
  max_description_length: 1000
  min_description_length: 10
```

## 3. Usage Examples

### Basic Usage
```bash
python main.py path/to/source.py --docstring-config config.yaml
```

### With Style Override
```bash
python main.py path/to/source.py --docstring-style numpy
```

### Preserve Existing Content
```bash
python main.py path/to/source.py --preserve-docstrings
```

## 4. Error Handling

Add specific error handling for docstring-related issues:
```python
class DocStringError(Exception):
    """Base exception for docstring-related errors."""
    pass

class DocStringValidationError(DocStringError):
    """Raised when docstring validation fails."""
    pass

class DocStringPreservationError(DocStringError):
    """Raised when docstring preservation fails."""
    pass
```

## 5. Testing

Add test cases for docstring handling:
```python
def test_docstring_preservation():
    """Test docstring preservation functionality."""
    # Test implementation

def test_docstring_validation():
    """Test docstring validation rules."""
    # Test implementation

def test_docstring_migration():
    """Test migration of existing docstrings."""
    # Test implementation
```