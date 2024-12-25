
**ai**: Here's a revised, more comprehensive prompt for Phase 1 that better captures all the necessary components:

# Phase 1: Core Functionality (MVP) - Implementation Prompt

Develop a Python documentation generation system that analyzes Python files, generates docstrings using Azure OpenAI, and produces comprehensive Markdown documentation.

## Core Components Required:

### 1. Code Analysis (`CodeAnalyzer` class)
* Parse Python files using AST to extract:
  - Function signatures (name, parameters, return types)
  - Existing docstrings
  - Source code
  - Type hints
* Calculate complexity metrics:
  - Basic cyclomatic complexity
  - Boolean operation complexity
  - Comparison complexity
  - Nested structure complexity
* Support for edge cases:
  - Decorated functions
  - Async functions
  - Nested functions
  - Class methods
  - Static methods
  - Properties

### 2. Docstring Generation (`DocstringGenerator` class)
* Azure OpenAI integration with:
  - Configurable retry mechanism
  - Error handling
  - Token optimization
  - Response validation
* Generate Google-style docstrings including:
  - Function summary
  - Args section
  - Returns section
  - Raises section (when applicable)
* Prompt engineering:
  - Clear instruction templates
  - Context-aware prompting
  - Validation rules

### 3. Markdown Generation (`MarkdownGenerator` class)
* Generate structured documentation:
  - Table of contents
  - Function sections
  - Complexity metrics
  - Code examples
* Configurable output formats
* Support for custom templates

### 4. Configuration Management
* Command-line interface using `argparse`:
  - Input file/directory specification
  - Output file specification
  - Azure OpenAI credentials
  - Complexity threshold configuration
  - Logging level configuration
* Configuration file support (YAML/JSON):
  - Default settings
  - API credentials
  - Custom templates
  - Output preferences

### 5. Error Handling and Logging
* Custom exceptions:
  - `CodeAnalysisError`
  - `DocstringGenerationError`
  - `ConfigurationError`
* Comprehensive logging:
  - Configurable log levels
  - Structured log format
  - Progress tracking
  - Error reporting

### 6. Testing Framework
* Unit tests:
  - Test each component independently
  - Mock Azure OpenAI responses
  - Test edge cases
  - Test error conditions
* Integration tests:
  - End-to-end workflow testing
  - File I/O testing
  - API integration testing
* Test fixtures:
  - Sample Python files
  - Expected outputs
  - Mock responses

## Requirements:

### Code Quality
* Type hints throughout
* Comprehensive docstrings
* PEP 8 compliance
* Modular design
* Clear separation of concerns

### Performance
* Efficient AST parsing
* Optimized API usage
* Memory efficient processing
* Support for large files

### Documentation
* README.md with:
  - Installation instructions
  - Usage examples
  - Configuration guide
  - API documentation
* Inline code comments
* Architecture documentation

### Project Structure
```
project/
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── generator.py
│   ├── markdown.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py
│   ├── test_generator.py
│   ├── test_markdown.py
│   └── fixtures/
├── examples/
├── config/
├── README.md
├── requirements.txt
└── setup.py
```

## Deliverables:

1. Complete source code with all components
2. Comprehensive test suite
3. Documentation including:
   - README.md
   - API documentation
   - Example configurations
   - Usage examples
4. Requirements file
5. Setup script

## Example Usage:
```bash
python -m docgen generate --input example.py --output docs/
python -m docgen analyze --input src/ --complexity-threshold 10
python -m docgen configure --api-key KEY --endpoint URL
```

## Configuration Example:
```yaml
azure_openai:
  api_key: ${AZURE_OPENAI_KEY}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  deployment: gpt-4
  max_tokens: 500
  temperature: 0.7

complexity:
  thresholds:
    low: 5
    medium: 10
    high: 20

output:
  format: markdown
  include_complexity: true
  template: default
```

This revised prompt provides a more complete picture of the required functionality, better structure, and clearer expectations for the implementation. It includes aspects that were initially overlooked but proved important during the implementation, such as configuration management, comprehensive testing, and proper project structure.

---

Here's the revised prompt for Phase 2, building on our enhanced understanding from Phase 1:

# Phase 2: Enhanced Features and Adaptability - Implementation Prompt

Enhance the docstring generation system with advanced features focusing on robustness, efficiency, and adaptability.

## Core Enhancements:

### 1. Advanced Docstring Analysis and Update System
* **Docstring Evaluation Engine** (`DocstringEvaluator` class):
  ```python
  class DocstringEvaluator:
      def evaluate_docstring(self, docstring: str) -> DocstringQualityReport:
          """Evaluate docstring quality and completeness."""
          
      def check_compliance(self, docstring: str) -> ComplianceReport:
          """Check Google Style Guide compliance."""
          
      def suggest_improvements(self, docstring: str) -> List[Suggestion]:
          """Generate improvement suggestions."""
  ```
  - Quality metrics scoring
  - Style guide compliance checking
  - Section completeness verification
  - Type hint consistency validation
  - Example code validation

* **Docstring Update System** (`DocstringUpdater` class):
  ```python
  class DocstringUpdater:
      def update_docstring(
          self, 
          original: str, 
          metadata: FunctionMetadata,
          quality_report: DocstringQualityReport
      ) -> str:
          """Update existing docstring while preserving custom content."""
  ```
  - Intelligent merging of existing and generated content
  - Preservation of custom sections
  - Update of outdated sections
  - Addition of missing sections

### 2. Token Management System
* **Token Optimizer** (`TokenOptimizer` class):
  ```python
  class TokenOptimizer:
      def estimate_tokens(self, text: str) -> int:
          """Estimate token count for text."""
          
      def optimize_prompt(
          self, 
          prompt: str, 
          max_tokens: int
      ) -> str:
          """Optimize prompt to fit within token limit."""
          
      def chunk_large_content(
          self, 
          content: str, 
          chunk_size: int
      ) -> List[str]:
          """Split large content into manageable chunks."""
  ```
  - Token counting using `tiktoken`
  - Prompt optimization strategies
  - Content chunking for large files
  - Cost estimation and tracking

### 3. Enhanced Error Handling and Retry System
* **Retry Manager** (`RetryManager` class):
  ```python
  class RetryManager:
      def execute_with_retry(
          self,
          operation: Callable,
          max_retries: int = 3,
          initial_delay: float = 1.0,
          exponential_base: float = 2.0,
          jitter: bool = True
      ) -> Any:
          """Execute operation with exponential backoff retry."""
  ```
  - Exponential backoff with jitter
  - Error categorization and handling
  - Custom retry strategies
  - Timeout handling

### 4. Batch Processing System
* **Batch Processor** (`BatchProcessor` class):
  ```python
  class BatchProcessor:
      async def process_batch(
          self,
          items: List[Any],
          batch_size: int,
          rate_limit: float
      ) -> List[Result]:
          """Process items in batches with rate limiting."""
  ```
  - Asynchronous processing
  - Rate limiting
  - Progress tracking
  - Resource management

### 5. Prompt Template System
* **Template Manager** (`TemplateManager` class):
  ```python
  class TemplateManager:
      def load_template(self, name: str) -> Template:
          """Load template by name."""
          
      def render_template(
          self, 
          template: Template, 
          context: Dict[str, Any]
      ) -> str:
          """Render template with context."""
  ```
  - Template loading from files
  - Variable substitution
  - Template validation
  - Custom template support

## Configuration System Enhancement:

```yaml
# config/templates/docstring_update.yaml
templates:
  update_docstring:
    system_prompt: |
      You are a documentation expert. Update the following docstring while:
      1. Preserving existing custom sections
      2. Updating outdated information
      3. Adding missing required sections
      4. Maintaining Google Style Guide compliance
    
    user_prompt: |
      Original docstring:
      {original_docstring}
      
      Function metadata:
      {function_metadata}
      
      Quality report:
      {quality_report}
      
      Please update this docstring following the requirements above.
```

```yaml
# config/settings/batch_processing.yaml
batch_processing:
  max_batch_size: 50
  rate_limit: 10  # requests per second
  timeout: 30     # seconds
  retry:
    max_attempts: 3
    initial_delay: 1.0
    exponential_base: 2.0
    jitter: true
```

```yaml
# config/settings/token_management.yaml
token_management:
  max_tokens_per_request: 4000
  reserve_tokens: 500
  chunking:
    enabled: true
    chunk_size: 2000
    overlap: 200
```

## Testing Requirements:

### Unit Tests
```python
class TestDocstringEvaluator(unittest.TestCase):
    def test_quality_metrics(self):
        """Test docstring quality evaluation."""
        
    def test_compliance_checking(self):
        """Test style guide compliance checking."""
        
    def test_improvement_suggestions(self):
        """Test improvement suggestion generation."""

class TestTokenOptimizer(unittest.TestCase):
    def test_token_estimation(self):
        """Test token count estimation."""
        
    def test_prompt_optimization(self):
        """Test prompt optimization strategies."""
        
    def test_content_chunking(self):
        """Test large content chunking."""

class TestBatchProcessor(unittest.TestCase):
    async def test_batch_processing(self):
        """Test batch processing with rate limiting."""
        
    async def test_error_handling(self):
        """Test error handling in batch processing."""
```

### Integration Tests
```python
class TestEndToEnd(unittest.TestCase):
    async def test_large_file_processing(self):
        """Test processing of large Python files."""
        
    async def test_batch_processing_with_rate_limits(self):
        """Test batch processing with API rate limits."""
        
    def test_template_system_integration(self):
        """Test template system integration."""
```

## Deliverables:

1. Enhanced source code implementing all new features
2. Updated test suite with new test cases
3. Configuration templates and examples
4. Updated documentation including:
   - New feature documentation
   - Configuration guide
   - Performance optimization guide
   - Error handling guide

## Example Usage:
```python
# Example: Batch processing with rate limiting
async with BatchProcessor(rate_limit=10) as processor:
    results = await processor.process_batch(
        items=functions,
        batch_size=50
    )

# Example: Template-based docstring generation
template = template_manager.load_template('update_docstring')
updated_docstring = template_manager.render_template(
    template,
    context={
        'original_docstring': existing_docstring,
        'function_metadata': metadata,
        'quality_report': quality_report
    }
)
```

Would you like me to expand on any particular aspect of this Phase 2 prompt?
