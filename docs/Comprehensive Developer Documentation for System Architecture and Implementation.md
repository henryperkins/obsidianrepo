Here's an updated version of the developer documentation for the system, incorporating the latest code structure and features:

---

# Developer Documentation

## System Architecture & Technical Specifications

### 1. Core Modules

#### 1.1 Code Extraction (`code_extraction.py`)
```python
def extract_classes_and_functions_from_ast(tree, content) -> dict
def extract_class_details(node, content) -> dict
def extract_function_details(node, content) -> dict
def calculate_complexity(node) -> int
```

**Purpose:** Handles AST-based code analysis and extraction.
- Utilizes Python's `ast` module for parsing.
- Extracts classes, methods, and functions.
- Calculates cyclomatic complexity.
- Preserves source code context.

**Usage Example:**
```python
with open('example.py', 'r') as file:
    content = file.read()
    tree = ast.parse(content)
    extracted_data = extract_classes_and_functions_from_ast(tree, content)
```

#### 1.2 File Processing (`file_processing.py`)
```python
async def process_file(filepath) -> ProcessingResult
async def clone_repo(repo_url, clone_dir) -> ProcessingResult
def get_all_files(directory, exclude_dirs=None) -> list
def load_gitignore_patterns(repo_dir) -> list
```

**Key Features:**
- Asynchronous file processing.
- Git repository handling.
- File system operations.
- Code formatting with `black`.
- Error handling with Sentry integration.

### 2. API Integration

#### 2.1 API Interaction (`api_interaction.py`)
```python
async def make_openai_request(model_name, messages, functions, service) -> dict
async def analyze_function_with_openai(function_details, service) -> dict
```

**Configuration Requirements:**
```python
# Required environment variables
OPENAI_API_KEY=<your_key>
AZURE_OPENAI_API_KEY=<your_azure_key>
AZURE_OPENAI_ENDPOINT=<your_endpoint>
```

**Rate Limiting & Retries:**
```python
async def exponential_backoff_with_jitter(
    func, 
    max_retries=5,
    base_delay=1,
    max_delay=60
)
```

#### 2.2 Caching System (`cache.py`)
```python
class ThreadSafeCache:
    def __init__(self, max_size_mb: int = 500)
    def get(self, key: str) -> Optional[Dict[str, Any]]
    def set(self, key: str, value: Dict[str, Any]) -> None
    def get_stats(self) -> Dict[str, Any]
```

**Cache Configuration:**
- Default max size: 500MB.
- LRU (Least Recently Used) eviction policy.
- Thread-safe operations.
- Performance monitoring.

### 3. Implementation Details

#### 3.1 AST Processing
```python
def add_parent_info(node, parent=None):
    """
    Adds parent references to AST nodes recursively.
    
    Args:
        node: Current AST node
        parent: Parent node reference
    """
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_info(child, node)
```

#### 3.2 Complexity Calculation
```python
def calculate_complexity(node):
    """
    Calculates cyclomatic complexity.
    
    Complexity increases for:
    - If statements
    - Loops
    - Exception handling
    - Boolean operations
    """
    complexity = 1  # Base complexity
    for subnode in ast.walk(node):
        if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try)):
            complexity += 1
        elif isinstance(subnode, ast.BoolOp):
            complexity += len(subnode.values) - 1
    return complexity
```

### 4. Error Handling & Logging

#### 4.1 Sentry Integration
```python
def initialize_sentry():
    sentry_sdk.init(
        dsn="your-sentry-dsn",
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
```

#### 4.2 Error Capture
```python
def capture_openai_error(error, context):
    with sentry_sdk.push_scope() as scope:
        scope.set_context("api_call", context)
        sentry_sdk.capture_exception(error)
```

### 5. Command Line Interface

```python
async def main():
    parser = argparse.ArgumentParser(description="Analyze code and generate documentation.")
    parser.add_argument("input_path", help="Path to input directory or repository URL")
    parser.add_argument("output_file", help="Path to output markdown file")
    parser.add_argument("--service", help="Service to use ('openai' or 'azure')", required=True)
```

### 6. Output Format

#### 6.1 Markdown Generation
```python
def write_analysis_to_markdown(results, output_file_path, repo_dir):
    """
    Generates structured markdown documentation:
    
    # Code Documentation and Analysis
    ## Table of Contents
    ## [File Path]
    ### Summary
    ### Recent Changes
    ### Function Analysis
    ### Source Code
    """
```

### 7. Development Guidelines

#### 7.1 Adding New Features
1. Create feature branch: `feature/feature-name`.
2. Implement changes.
3. Add tests.
4. Update documentation.
5. Submit pull request.

#### 7.2 Code Style
- Follow PEP 8.
- Use type hints.
- Document all public functions.
- Add logging for important operations.

#### 7.3 Testing
```python
# Example test structure
async def test_process_file():
    filepath = "test_files/example.py"
    result = await process_file(filepath)
    assert result.success
    assert "functions" in result.data
    assert "classes" in result.data
```

### 8. Performance Considerations

#### 8.1 Optimization Tips
- Use caching for repeated API calls.
- Process files concurrently when possible.
- Monitor memory usage with large repositories.
- Use streaming for large file operations.

#### 8.2 Resource Management
```python
async def process_files_concurrently(files_list):
    chunk_size = 10  # Process files in chunks
    for i in range(0, len(files_list), chunk_size):
        chunk = files_list[i:i + chunk_size]
        tasks = [process_file(filepath) for filepath in chunk]
        await asyncio.gather(*tasks)
```

### 9. Troubleshooting

#### Common Issues and Solutions:
1. **Rate Limiting:**
   - Implement exponential backoff.
   - Use multiple API keys.
   - Monitor usage patterns.

2. **Memory Issues:**
   - Adjust cache size.
   - Process files in chunks.
   - Clean up temporary files.

3. **API Errors:**
   - Check API keys.
   - Verify endpoint configuration.
   - Monitor response status codes.

### 10. Security Considerations

- Store API keys securely.
- Validate input paths.
- Sanitize repository URLs.
- Handle sensitive code sections.
- Implement access controls.

This documentation provides a comprehensive guide for developers working with the system. For specific implementation details, refer to the individual module documentation and inline comments.