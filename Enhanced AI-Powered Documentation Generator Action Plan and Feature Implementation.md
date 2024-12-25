This roadmap integrates the proposed features for context management, leveraging the provided codebase and research on context window limitations.

### Phase 1: Context-Aware Chunking and Hierarchical Documentation

**Objective:** Intelligently segment code into manageable chunks while preserving structural relationships for coherent documentation.

**Actions:**

1. **Semantic Chunking Implementation:**  In `utils.py`, create a `chunk_code` function:
    - **Language-Specific Chunking:** Implement language-specific logic within `chunk_code` using abstract syntax trees (ASTs).  For Python, use the `ast` module.  For other languages, leverage language-specific parsers (e.g., Tree-sitter).  This ensures chunks respect code structure (functions, classes, blocks).
    - **Token Counting:** Integrate `tiktoken` for precise token counting within each chunk.  Add a `max_tokens` parameter to `chunk_code` to control chunk size.
    - **Chunk Metadata:**  Each chunk should store metadata (e.g., file path, starting line number, ending line number, function/class name if applicable).  This metadata is crucial for linking chunks and reconstructing context.

2. **Hierarchical Chunk Representation:**  In `context_manager.py`:
    - **Hierarchical Structure:**  The `ContextManager` should store chunks in a hierarchical structure (e.g., a tree) reflecting the code's organization (project, module, class, function).
    - **Chunk Linking:**  Store links between chunks based on their relationships (e.g., a function call in one chunk to a function definition in another).
    - **Methods:**  Add methods to `ContextManager` to retrieve context based on this hierarchy (e.g., `get_context_for_function(function_name)`).

**Resources:**

- Tree-sitter: [https://tree-sitter.github.io/tree-sitter/](https://tree-sitter.github.io/tree-sitter/)
- Python's `ast` Module: [https://docs.python.org/3/library/ast.html](https://docs.python.org/3/library/ast.html)

### Phase 2: Token Limit Management and Dynamic Context

**Objective:** Optimize token usage through contextual compression and dynamic context window adjustment.

**Actions:**

1. **Contextual Compression:** In `utils.py`:
    - **Summarization:** Implement a `summarize_code` function using NLP techniques (e.g., NLTK, Transformers) to condense less critical code sections (e.g., lengthy comments, less complex functions).  This function should be called during chunking if a chunk exceeds the token limit.
    - **Prioritization for Summarization:**  Prioritize summarization of comments and less complex code over core logic.  Use code complexity metrics (e.g., cyclomatic complexity from `radon`) to guide this prioritization.

2. **Dynamic Window Adjustment:** In `file_handlers.py`:
    - **Context-Aware Adjustment:**  Modify `fetch_documentation_rest` to dynamically adjust the context window size based on the current chunk's complexity and token usage.  Use a combination of token counts and complexity metrics to determine the optimal window size.
    - **AI Feedback Integration:**  If the AI signals `INSUFFICIENT_CONTEXT`, attempt to increase the context window size (up to a reasonable limit) before retrying the API call.

**Resources:**

- NLTK: [https://www.nltk.org/](https://www.nltk.org/)
- Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)

### Phase 3: Incremental Context and Multi-Language Support

**Objective:**  Enable efficient context updates and handle multi-language codebases.

**Actions:**

1. **Incremental Updates:** In `file_handlers.py`:
    - **Change Detection:**  Before processing a file, check if it has been modified since the last run (e.g., using file modification timestamps or Git diffs).  Only re-chunk and re-process modified files.
    - **ContextManager Updates:**  Implement methods in `ContextManager` to update specific chunks and their associated context, rather than rebuilding the entire context from scratch.

2. **Multi-Language Support:** In `language_functions` directory:
    - **Language-Specific Handlers:**  Enhance existing language handlers and add new ones as needed.  Each handler should implement chunking, summarization, and context extraction logic specific to the language.
    - **Cross-Language Linking:**  If necessary, implement mechanisms to link context across different languages (e.g., if a Python function calls a JavaScript function).  This might involve creating a unified representation of code elements across languages.

**Resources:**

- GitPython: [https://gitpython.readthedocs.io/en/stable/](https://gitpython.readthedocs.io/en/stable/) (for Git diff integration)


### Phase 4: Caching, Prioritization, and Cross-Context Integration

**Objective:** Optimize performance and enhance AI understanding through caching, selective context retention, and cross-context summarization.

**Actions:**

1. **Contextual Caching:** In `file_handlers.py`:
    - **Cache Implementation:**  Implement a caching mechanism (e.g., using `functools.lru_cache`, Redis, or Memcached) to store frequently accessed chunks and their associated context.
    - **Cache Invalidation:**  Implement a strategy to invalidate cached chunks when the corresponding code is modified.

2. **Selective Retention and Prioritization:** In `context_manager.py`:
    - **Relevance Scoring:**  Develop a scoring function that considers code complexity, usage frequency (if available), and AI feedback to prioritize context entries.
    - **Context Filtering:**  When retrieving context, filter out low-scoring entries to stay within token limits.

3. **Cross-Chunk and Cross-Context Summarization:** In `utils.py`:
    - **Cross-Chunk Linking:**  Enhance chunk metadata to include information about related chunks (e.g., function calls, class inheritance).  Use this information to retrieve related chunks when building context for a specific chunk.
    - **Cross-Context Summarization:**  Implement summarization across different types of context (e.g., code, documentation, comments).  This could involve creating a unified representation of context and then applying summarization techniques.

**Resources:**

- `functools.lru_cache`: [https://docs.python.org/3/library/functools.html#functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- Redis: [https://redis.io/](https://redis.io/)
- Memcached: [https://memcached.org/](https://memcached.org/)

### Phase 5: Advanced Features and Integration

**Objective:** Implement advanced features like code growth prediction, error handling, and a user interface for context management.

**Actions:**

1. **Code Growth Prediction:** In a new module (e.g., `code_growth_predictor.py`):
    - **Metrics Tracking:**  Collect code metrics over time (e.g., lines of code, number of functions, dependencies).
    - **Prediction Model:**  Implement a simple prediction model (e.g., linear regression) or integrate a more advanced forecasting library.

2. **Robust Error Handling and Logging:** Throughout the codebase:
    - **Error Handling:**  Implement `try...except` blocks to handle potential errors during file processing, chunking, API calls, and other operations.
    - **Logging:**  Use the `logging` module to log events, errors, and performance metrics.

3. **User Interface (Optional):**  In `documentation-viewer`:
    - **Context Visualization:**  Display the current context window and allow users to explore the hierarchical context structure.
    - **Manual Context Control:**  Provide options for users to manually add or remove context entries, override prioritization scores, or adjust the context window size.

**Resources:**

- Scikit-learn (for prediction models): [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Streamlit (for building a simple UI): [https://streamlit.io/](https://streamlit.io/)

This enhanced roadmap provides a more concrete and actionable plan, incorporating the specific features and considerations you outlined, along with relevant code examples and resources.  It emphasizes a phased approach, starting with the most critical context management features and progressively adding more advanced functionalities.  This structure allows for iterative development, testing, and refinement of the tool.


---

Your revised prompts are significantly improved and demonstrate a good understanding of the task and the desired functionality. Here are a few refinements and suggestions to make them even better, incorporating best practices and addressing potential edge cases:

**Prompt 1: Define CodeChunk Dataclass for Context-Aware Chunking**

Create a Python dataclass named `CodeChunk` to store metadata about a code chunk. It should have the following fields:

* `file_path`: str (Path to the source file)
* `start_line`: int (Starting line number of the chunk, inclusive)
* `end_line`: int (Ending line number of the chunk, inclusive)
* `function_name`: Optional[str] (Name of the function if the chunk represents a function, otherwise None)
* `class_name`: Optional[str] (Name of the class if the chunk represents a class, otherwise None)
* `chunk_id`: str (A unique identifier for the chunk, generated using the uuid library)
* `tokens`: List[str] (The token strings of the code in the chunk.  Store the actual tokens, not token IDs.)
* `token_count`: int (The number of tokens in the chunk)
* `language`: str (The programming language of the chunk, e.g., "python")

Include type hints and a method called `get_context_string()` that returns a concise string representation of the chunk's context, including file path, line numbers, and function/class name if applicable. This string will be used for logging and debugging. Also, include a `__repr__` method for easy printing of the dataclass. The `chunk_id` should be automatically generated using the `uuid` library. Import necessary libraries.  Ensure the `get_context_string()` method handles cases where `function_name` or `class_name` are None.  The output of  `get_context_string()` should be something like:  "File: path/to/file.py, Lines: 10-20, Function: my_function" or "File: path/to/file.py, Lines 30-40, Class: MyClass" or "File: path/to/file.py, Lines: 50-60" (if no function or class).

* **Changes:**
    - Clarified that `start_line` and `end_line` are inclusive.
    - Changed `tokens` type to `List[str]` to store the actual token strings, which is more useful for context and documentation generation.
    - Added `language` field to store the programming language of the chunk.
    - Specified the desired output format for `get_context_string()`.

---

**Prompt 2: Implement Context-Aware Code Chunking**

Write a Python function called `chunk_code(code: str, file_path: str, language: str, max_tokens: int = 512) -> List[CodeChunk]` that takes a string of code, the file path, the programming language, and a maximum number of tokens as input and returns a list of `CodeChunk` objects.

The function should:

1. Use the appropriate parsing mechanism for the given `language`.  For Python, use the `ast` module.  For other languages, you might use a different parsing library (e.g., Tree-sitter) or a regular expression-based approach if the language structure is simple.
2. Split the code into chunks based on these definitions, ensuring that each chunk contains at most `max_tokens` tokens.  Prioritize keeping function/class definitions within a single chunk. If a function or class is too large to fit in one chunk, split it into multiple chunks, adding a suffix like "_part1", "_part2" to the `function_name` or `class_name` in the `CodeChunk` metadata.
3. Use the `tiktoken` library for Python code. For other languages, use an appropriate tokenizer or a reasonable approximation for token counting. If a chunk exceeds the token limit, raise a custom `ChunkTooLargeError` (create a custom exception for this).  Include the chunk's context string in the exception message.
4. Create a `CodeChunk` object for each chunk, populating all fields, including `language`.
5. Include comprehensive error handling and logging. Log the context string of each created chunk.
6. Include a detailed docstring.

* **Changes:**
    - Added `language` parameter to handle different programming languages.
    - Added instructions for handling large functions/classes that exceed the token limit.
    - Specified using appropriate tokenizers for different languages.
    - Added requirement to include the context string in the `ChunkTooLargeError`.

---

**Prompt 3: Implement Hierarchical Context Manager**

Write a Python class called `HierarchicalContextManager` that manages code chunks in a hierarchical structure: Project -> Module -> Class -> Function -> Code Chunk.  Use nested dictionaries to represent this hierarchy.  The keys at each level should be strings.  The values at the lowest level (Code Chunk) should be lists of `CodeChunk` objects, as a function/class could be split into multiple chunks.

The class should have the following methods:

* `add_chunk(chunk: CodeChunk)`: Adds a code chunk to the correct location in the hierarchy.  Create missing levels dynamically.  If multiple chunks exist for the same function/class, append the new chunk to the existing list.
* `get_context_for_function(module_path: str, function_name: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves chunks related to a function, limited to `max_tokens`. Prioritize the function's chunks, then the class's chunks (if applicable), then the module's chunks.  If the total tokens exceed `max_tokens`, truncate the least prioritized chunks (module-level chunks first).
* `get_context_for_module(module_path: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves chunks within a module, limited to `max_tokens`.  If the total tokens exceed `max_tokens`, truncate the chunks to fit.
* `get_context_for_project(project_path: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves chunks within a project, limited to `max_tokens`.  Truncate if necessary.
* `update_chunk(chunk: CodeChunk)`: Updates a chunk. Raise `ChunkNotFoundError` if not found.
* `remove_chunk(chunk_id: str)`: Removes a chunk by ID. Raise `ChunkNotFoundError` if not found.
* `clear_context()`: Clears the entire context.

Include docstrings, comprehensive error handling, logging, and type hints.  The paths should be relative to the project root.  The `get_context_*` methods should return an empty list if no chunks are found for the given entity.

* **Changes:**
    - Clarified the use of lists for code chunks at the lowest level of the hierarchy.
    - Added `max_tokens` parameter and truncation logic to the `get_context_*` methods.
    - Specified the prioritization order for context retrieval.
    - Added clarification about returning an empty list if no chunks are found.

---

**Prompt 4: Unit Tests for Chunking and Context Manager** -  This prompt is already quite good.  No significant changes needed.


**Prompt 5: Integration with `file_handlers.py`**

Modify the `process_file` and `process_all_files` functions in `file_handlers.py` to integrate the `chunk_code` function and the `HierarchicalContextManager`.

1. Instantiate the `HierarchicalContextManager` once at the beginning of the `process_all_files` function.
2. In `process_file`, after successfully reading the file content and determining the language, call `chunk_code` with the `code`, `file_path`, and `language` arguments to get the chunks.
3. Add each chunk to the `HierarchicalContextManager` using `add_chunk`.  Handle the `ChunkTooLargeError` gracefully. If this error occurs, log a warning including the context string of the problematic chunk and skip processing of that file.
4. Modify the `generate_documentation_prompt` function in `write_documentation_report.py` to retrieve the relevant context.  Determine the module path and function/class name (if applicable) from the current `file_path`.  Then, call the appropriate `get_context_*` method of the `HierarchicalContextManager` with the `module_path`, `function_name`, and `max_tokens` arguments.  Use a `max_tokens` value that leaves enough room in the prompt for the instructions and the code structure itself.
5. Include comprehensive error handling, logging, and type hints for all new code.

* **Changes:**
    - Specified passing the `language` argument to `chunk_code`.
    - Added instructions for handling the `ChunkTooLargeError`.
    - Clarified how to determine the arguments for the `get_context_*` methods.
    - Emphasized leaving enough room in the prompt for instructions and code structure.

---

These refined prompts should provide even clearer instructions to the AI and help you generate more robust and efficient code for your project.  Remember to review and adapt the generated code to your specific needs and coding style.  Thorough testing is essential to ensure the correct functionality and handling of edge cases.