Okay, let's expand the roadmap to an 8-phase approach, allowing for a more granular implementation of the features and better integration of advanced techniques. This will also provide more flexibility for prioritizing and scheduling development efforts.

**8-Phase Action Plan for Enhanced AutoDocs with Context Window Management:**

**Phase 1: Foundation - Code Chunking for Python**

* Focus: Implement robust code chunking specifically for Python, laying the groundwork for context management.
* Tasks:
    * Implement `CodeChunk` dataclass (immutable, with `chunk_content`).
    * Implement `chunk_code` function for Python (using `ast`, `overlap_tokens`, handling `ChunkTooLargeError`).
    * Integrate `chunk_code` into `process_file` (iterate through chunks, adapt prompt generation).

**Phase 2: Basic Context Management for Python**

* Focus: Implement the hierarchical context manager and integrate it with the Python processing pipeline.
* Tasks:
    * Implement `HierarchicalContextManager` (relative paths, language filtering, documentation retrieval methods).
    * Instantiate `HierarchicalContextManager` in `process_all_files`.
    * Modify `generate_documentation_prompt` to retrieve context.

**Phase 3: Contextual Summarization for Python**

* Focus: Implement basic summarization to condense lengthy code comments and docstrings in Python.
* Tasks:
    * Implement `summarize_code` function in `utils.py` (start with a simple NLTK-based approach).
    * Integrate `summarize_code` into `chunk_code` (summarize before raising `ChunkTooLargeError`).

**Phase 4: Contextual Compression for Python**

* Focus: Implement basic compression techniques to reduce token usage in Python code.
* Tasks:
    * Implement comment removal/shortening in `chunk_code` (prioritize removing less relevant comments).
    * Explore other basic compression techniques (e.g., removing blank lines, simplifying code structure where possible).

**Phase 5: Dynamic Context Adjustment and Prioritization for Python**

* Focus: Implement dynamic context window adjustment and prioritization based on chunk characteristics.
* Tasks:
    * Implement basic relevance scoring in `HierarchicalContextManager`.
    * Implement dynamic `max_tokens` adjustment in `fetch_documentation_rest`.
    * Implement prioritized context retrieval in `get_context_*` methods.

**Phase 6: Multi-Language Support - Chunking and Summarization**

* Focus: Extend chunking and summarization support to other languages (JavaScript, Java, etc.).
* Tasks:
    * Implement `chunk_code` and `summarize_code` for each supported language (using appropriate parsers and tokenizers).
    * Update language-specific handlers (`language_functions` package) to integrate with chunking and summarization.

**Phase 7: Cross-Language Context Linking and Advanced Summarization**

* Focus: Implement advanced features for linking context across languages and improving summarization quality.
* Tasks:
    * Implement cross-language context linking (if needed).  This might involve creating a unified representation of code elements across languages.
    * Explore and integrate more advanced summarization techniques (e.g., Transformers-based models).

**Phase 8: Caching, Feedback Loops, and User Interface**

* Focus: Optimize performance, implement feedback mechanisms, and provide a user interface for context management.
* Tasks:
    * Implement contextual caching in `file_handlers.py`.
    * Implement contextual feedback loops to refine context management strategies.
    * Develop a user interface for context visualization and control (optional, but highly recommended).

**Advantages of the 8-Phase Approach:**

* **Granularity:**  Allows for more focused development efforts in each phase.
* **Prioritization:**  Makes it easier to prioritize features and adjust the roadmap based on project needs and available resources.
* **Incremental Improvement:**  Facilitates iterative development and testing, allowing for continuous refinement of the context management system.
* **Flexibility:**  Provides more flexibility for incorporating advanced techniques (like cross-language linking and advanced summarization) in later phases.

This expanded roadmap provides a more structured and manageable approach to implementing context window management in AutoDocs.  By breaking down the development into smaller, well-defined phases, you can ensure a more robust and efficient implementation. Remember to thoroughly test each phase before moving on to the next, and to adapt the roadmap as needed based on your specific project requirements and priorities.

---

Let's break down Phase 1: Foundation - Code Chunking for Python, into actionable steps and corresponding AI prompts.

**Phase 1: Foundation - Code Chunking for Python**

* **Objective:** Implement robust code chunking specifically for Python, laying the groundwork for context management.  This phase focuses solely on Python to establish a solid base before expanding to other languages.

* **Outline:**
    1. **Define the `CodeChunk` dataclass:** Create a dataclass to store metadata and content for each code chunk.
    2. **Implement the `chunk_code` function:**  This function will take Python code as input and return a list of `CodeChunk` objects.
    3. **Integrate `chunk_code` into `file_handlers.py`:** Modify the `process_file` function to use chunking.  This involves adapting the prompt generation and AI API interaction to work with chunks.

* **AI Prompts and Implementation Details:**

**Prompt 1:  `CodeChunk` Dataclass Definition**


Create a Python dataclass named `CodeChunk` to store metadata about a code chunk. It should have the following fields, including type hints:

* `file_path`: str (Path to the source file)
* `start_line`: int (Starting line number of the chunk, inclusive)
* `end_line`: int (Ending line number of the chunk, inclusive)
* `function_name`: Optional[str] (Name of the function if the chunk represents a function, otherwise None)
* `class_name`: Optional[str] (Name of the class if the chunk represents a class, otherwise None)
* `chunk_id`: str (A unique identifier for the chunk, generated using the uuid library)
* `chunk_content`: str (The actual code string of the chunk)
* `tokens`: List[str] (The token strings of the code in the chunk)
* `token_count`: int (The number of tokens in the chunk)
* `language`: str (The programming language of the chunk, e.g., "python")

The dataclass should be immutable (use `@dataclass(frozen=True)`).  Include a method called `get_context_string()` that returns a concise string representation of the chunk's context, including file path, line numbers, and function/class name if applicable (handle cases where function_name or class_name are None).  The output of `get_context_string()` should be something like: "File: path/to/file.py, Lines: 10-20, Function: my_function", or "File: path/to/file.py, Lines: 30-40, Class: MyClass", or "File: path/to/file.py, Lines: 50-60" (if no function or class).  Also include a `__repr__` method for easy printing of the dataclass.  Import necessary libraries (uuid, dataclasses, typing).

* **Implementation Note:** Place this `CodeChunk` definition in a new file, `code_chunk.py`, in your project.  This will make it easier to import and use in other modules.

**Prompt 2: `chunk_code` Function Implementation**

Write a Python function called `chunk_code(code: str, file_path: str, language: str, max_tokens: int = 512, overlap_tokens: int = 10) -> List[CodeChunk]` that takes a string of code, the file path, the programming language, a maximum number of tokens, and an overlap token count as input and returns a list of `CodeChunk` objects.  Import necessary libraries, including `ast`, `tiktoken`, and the `CodeChunk` dataclass you defined earlier.

The function should:

1. Check if the language is Python. If not, raise a `NotImplementedError` with a message indicating that chunking is currently only implemented for Python.
2. Use the `ast` module to parse the Python code.
3. Traverse the AST and create chunks based on function and class definitions. Prioritize keeping entire function/class definitions within a single chunk.
4. If a function or class is too large to fit in `max_tokens`, split it into multiple chunks, adding a suffix like "_part1", "_part2" to the `function_name` or `class_name` in the `CodeChunk` metadata.  Split at logical boundaries like the end of a statement or block.
5. For code outside of function/class definitions, create chunks of at most `max_tokens` tokens.
6. Use `tiktoken` for token counting.
7. Implement overlapping chunks using the `overlap_tokens` parameter.  The last `overlap_tokens` tokens of a chunk should be included at the beginning of the next chunk.
8. If, after attempting summarization (if implemented in a later phase), a chunk *still* exceeds `max_tokens`, raise a custom `ChunkTooLargeError` (create a custom exception for this). Include the chunk's context string in the exception message.
9. Create a `CodeChunk` object for each chunk, populating all fields, including `chunk_content`, `tokens`, `token_count`, and `language`.
10. Include comprehensive error handling (try...except blocks) and logging. Log the context string of each created chunk at the DEBUG level.
11. Include a detailed docstring explaining the function's purpose, arguments, return value, and how it handles different cases (e.g., large functions, code outside of functions/classes).
	* **Implementation Note:** Place this `chunk_code` function in the `utils.py` file.

**Prompt 3: Integrating `chunk_code` into `file_handlers.py`**

Modify the `process_file` function in `file_handlers.py` to integrate the `chunk_code` function from `utils.py`. Import the necessary classes (CodeChunk, ChunkTooLargeError).

1. After successfully reading the file content and determining the language, call `chunk_code` with the `code`, `file_path`, and `language` arguments to get a list of `CodeChunk` objects.
2. Iterate through the list of `CodeChunk` objects. For each chunk:
	    a.  Modify the `generate_documentation_prompt` function in `write_documentation_report.py` to accept a `CodeChunk` object as an argument instead of the entire `code_structure`.  The prompt should include the `chunk_content` and relevant metadata from the `CodeChunk` object.  The prompt should also include context from previous chunks of the same file (if available).  You'll need to manage this chunk history within `process_file`.
    b. Call `fetch_documentation_rest` with the generated prompt.
    c.  Store the generated documentation for each chunk.
3. After processing all chunks in a file, combine the documentation for each chunk into a single documentation object for the entire file.  Then, proceed with the existing code to insert the documentation and write the report.
4. Handle the `ChunkTooLargeError` gracefully. If this error occurs, log a warning including the context string of the problematic chunk and skip processing of that file.  Return a `FileProcessingResult` object with `success=False` and an appropriate error message.
5. Include comprehensive error handling, logging, and type hints for all new and modified code.  Add a docstring explaining the changes made to `process_file`.

	* **Implementation Note:** These changes will be made directly in `file_handlers.py` and `write_documentation_report.py`.

These prompts and implementation notes provide a detailed guide for completing Phase 1 of the enhanced AutoDocs project.  Remember to test thoroughly at each step to ensure correctness and address any unexpected behavior.  This focused approach will establish a strong foundation for the subsequent phases of context management.

---

**Phase 2: Basic Context Management for Python**

* **Objective:** Implement the hierarchical context manager and integrate it with the Python processing pipeline. This phase focuses on establishing the core context management functionality for Python, building upon the code chunking implemented in Phase 1.

* **Outline:**
    1. **Implement the `HierarchicalContextManager` class:** This class will store and retrieve `CodeChunk` objects in a hierarchical structure.
    2. **Integrate `HierarchicalContextManager` into `file_handlers.py`:** Modify the `process_all_files` and `process_file` functions to use the context manager.
    3. **Adapt prompt generation:** Modify `generate_documentation_prompt` to retrieve context from the `HierarchicalContextManager`.

* **AI Prompts and Implementation Details:**

**Prompt 1: `HierarchicalContextManager` Class Implementation**

Write a Python class called `HierarchicalContextManager` that manages code chunks and their associated documentation in a hierarchical structure: Project -> Module -> Class -> Function -> CodeChunk/DocChunk. Use nested dictionaries to represent this hierarchy. The keys at each level should be strings (representing project paths, module names, class names, and function names). The values at the lowest level (Code Chunk/Doc Chunk) should be lists of `CodeChunk` objects (for code) or dictionaries (for documentation), as a function/class could be split into multiple chunks and each chunk will have its own documentation.

The class should have the following methods, including type hints, docstrings, and comprehensive error handling:

* `add_code_chunk(chunk: CodeChunk)`: Adds a code chunk to the correct location in the hierarchy based on its `file_path`, `class_name`, and `function_name`. Create missing levels dynamically. If multiple chunks exist for the same function/class, append the new chunk to the existing list.  Handle cases where a chunk might not belong to a function or class.
* `add_doc_chunk(chunk_id: str, documentation: Dict[str, Any])`: Adds documentation for a specific code chunk, identified by its `chunk_id`.  The documentation should be stored as a dictionary.  Raise a `ChunkNotFoundError` if the `chunk_id` doesn't exist in the code chunk hierarchy.
* `get_context_for_function(module_path: str, function_name: str, language: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves code chunks related to a function, limited to `max_tokens`.  The `module_path` should be a relative path from the project root. Prioritize the function's chunks, then the class's chunks (if applicable), then the module's chunks. If the total `token_count` of the chunks exceeds `max_tokens`, truncate the least prioritized chunks (module-level chunks first). Filter chunks by `language`. Return an empty list if no chunks are found.
* `get_context_for_class(module_path: str, class_name: str, language: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves code chunks related to a class, limited to `max_tokens`. Prioritize the class's chunks, then the module's chunks.  Filter by language. Return an empty list if no chunks are found.
* `get_context_for_module(module_path: str, language: str, max_tokens: int = 4096) -> List[CodeChunk]`: Retrieves chunks within a module, limited to `max_tokens`. Filter by language. Return an empty list if no chunks are found.
* `get_documentation_for_chunk(chunk_id: str) -> Optional[Dict[str, Any]]`: Retrieves the documentation associated with a specific code chunk ID. Returns None if no documentation is found for the given ID.
* `update_code_chunk(chunk: CodeChunk)`: Updates a code chunk. Raise `ChunkNotFoundError` if not found.
* `remove_code_chunk(chunk_id: str)`: Removes a code chunk by ID. Raise `ChunkNotFoundError` if not found.  Also remove the associated documentation.
* `clear_context()`: Clears the entire context.

Import necessary libraries, including `CodeChunk` from your `code_chunk.py` file and `typing`.


* **Implementation Note:** Place the `HierarchicalContextManager` class in the `context_manager.py` file, replacing the existing `ContextManager` class.

**Prompt 2: Integrating `HierarchicalContextManager` into `file_handlers.py`**

Modify the `process_all_files` and `process_file` functions in `file_handlers.py` to integrate the `HierarchicalContextManager`. Import the necessary classes (`HierarchicalContextManager`, `CodeChunk`).

1. In `process_all_files`, instantiate the `HierarchicalContextManager` *once* at the beginning of the function, before the loop that processes each file.
2. In `process_file`, after chunking the code with `chunk_code`:
    a. Add each `CodeChunk` to the `HierarchicalContextManager` using `add_code_chunk`.
    b. After generating documentation for a chunk with `fetch_documentation_rest`, add the documentation to the `HierarchicalContextManager` using `add_doc_chunk`, associating it with the chunk's ID.
3. Modify the `generate_documentation_prompt` function in `write_documentation_report.py` to retrieve context from the `HierarchicalContextManager`.  Use the chunk's `file_path` to determine the `module_path`, `class_name`, and `function_name`.  Then, call the appropriate `get_context_*` method (e.g., `get_context_for_function`) of the `HierarchicalContextManager` with the `module_path`, `function_name`, `language`, and `max_tokens` arguments.  Use a `max_tokens` value that leaves enough room in the prompt for the instructions and the code structure itself.  Also retrieve the documentation for any related chunks using `get_documentation_for_chunk` and include it in the prompt.
4. Include comprehensive error handling, logging, and type hints for all new and modified code.  Add docstrings explaining the changes.

These prompts and implementation notes provide a detailed guide for completing Phase 2.  Testing is crucial at each step.  This phase sets up the core context management structure, enabling more advanced context-aware features in later phases.

---

**Phase 3: Contextual Summarization for Python**

* **Objective:** Implement basic summarization to condense lengthy code comments and docstrings in Python, reducing token consumption within the context window.  This phase focuses on Python and lays the groundwork for more advanced summarization techniques in later phases.

* **Outline:**
    1. **Implement the `summarize_code` function:** This function will take Python code as input and return a summarized version.
    2. **Integrate `summarize_code` into `chunk_code`:**  Call the summarization function if a chunk exceeds the token limit.

* **AI Prompts and Implementation Details:**

**Prompt 1: `summarize_code` Function Implementation**

Implement a Python function called `summarize_code(code: str, language: str, max_tokens: int = 50) -> str` in the `utils.py` file.  This function should take a string of code, the programming language, and a maximum number of tokens for the summary as input, and return a summarized string.

1. Check if the language is Python. If not, return the original code unchanged (we'll handle other languages in Phase 6).
2. For Python code, use the following approach for summarization:
    a. Extract all docstrings and comments from the code.  Use the `ast` module to parse the code and identify docstrings.  For comments, use a regular expression to find lines starting with `#`.
    b. Combine the extracted docstrings and comments into a single string.
    c. Use the `NLTK` library (or a similar text summarization technique) to summarize the combined string. Limit the summary to `max_tokens` using `tiktoken` for token counting.  You can use the `summarize_text` function from the previous examples as a starting point, but improve it by handling edge cases (e.g., empty input, input shorter than the maximum length) and adding error handling.
    d. If the summarized string is shorter than the original combined docstrings and comments, return the summarized string. Otherwise, return the original combined string truncated to `max_tokens`.  This ensures the function always returns a shorter or equal length string.
3. Include a docstring explaining the function's purpose, arguments, return value, and the summarization approach.  Include type hints.  Handle any potential errors during summarization gracefully by logging the error and returning the original (or truncated) string.


**Prompt 2: Integrating `summarize_code` into `chunk_code`**


Modify the `chunk_code` function in `utils.py` to integrate the `summarize_code` function.

1. After creating a chunk and before checking if it exceeds `max_tokens`, call `summarize_code` with the chunk's content and language.  Pass the current `max_tokens` value to `summarize_code` to ensure the summarized chunk doesn't exceed the limit.
2. Replace the original chunk content with the summarized content returned by `summarize_code`.  Update the `tokens` and `token_count` fields of the `CodeChunk` accordingly.
3. *Only after summarization*, check if the chunk still exceeds `max_tokens`.  If it does, raise the `ChunkTooLargeError` as before.  This gives summarization a chance to reduce the chunk size before resorting to skipping the file.
4. Add a docstring explaining the integration of summarization.

**Implementation Notes:**

* Ensure that `NLTK` is installed and the necessary NLTK data (e.g., `punkt`, `stopwords`) are downloaded.
* The summarization logic in `summarize_code` can be improved in later phases by using more advanced techniques (e.g., Transformers-based models).
* Consider adding a configuration option to enable/disable summarization, or to choose different summarization methods.

By completing Phase 3, AutoDocs will be able to handle lengthy comments and docstrings more effectively, optimizing token usage and improving the AI's ability to focus on the most relevant code within the context window.  Remember to test thoroughly after each change.


To implement Gemini in your AutoDocumentation script, let's go through a more detailed setup and integration process. This will cover setting up the environment, creating a client, and modifying your script to use Gemini.

### Step-by-Step Implementation

#### 1. Environment Setup

**Install Required Libraries:**

```bash
pip install google-cloud-aiplatform
```

**Set Up Authentication:**

- Ensure your `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to your service account key file.

#### 2. Create a Gemini Client (`gemini_client.py`)

This client will handle interactions with the Gemini API.

```python
import vertexai
from vertexai.preview.language_models import TextGenerationModel
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, project, location, model_name="gemini-1.5-pro-002"):
        vertexai.init(project=project, location=location)
        self.model = TextGenerationModel.from_pretrained(model_name)

    def generate_text(self, prompt: str, max_output_tokens=1500, temperature=0.7) -> str:
        try:
            response = self.model.predict(prompt, max_output_tokens=max_output_tokens, temperature=temperature)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ""
```

#### 3. Modify `file_handlers.py`

Adapt your file handling to use the Gemini client.

```python
import json
import logging
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)

async def fetch_documentation_rest(prompt, llm, retry=3):
    for attempt in range(retry):
        try:
            if isinstance(llm, GeminiClient):
                prompt_text = prompt[-1]["content"]
                documentation_json = llm.generate_text(prompt_text)
                try:
                    documentation = json.loads(documentation_json)
                    return documentation
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from Gemini: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error during API request (Attempt {attempt+1}): {e}")
            if attempt < retry - 1:
                continue
            else:
                raise
    logger.error("All retry attempts to fetch documentation failed.")
    return None
```

#### 4. Modify `process_manager.py`

Update the process manager to handle Gemini as a model option.

```python
from gemini_client import GeminiClient

class DocumentationProcessManager:
    def __init__(self, project, location, model="azure"):
        self.model = model
        if self.model == "gemini":
            self.llm = GeminiClient(project=project, location=location)
        else:
            self.llm = None

    async def process_files(self, file_paths):
        tasks = [
            self._process_single_file(file_path)
            for file_path in file_paths
        ]
        for future in asyncio.as_completed(tasks):
            result = await future
            # Handle result

    async def _process_single_file(self, file_path):
        # Process each file
        documentation = await fetch_documentation_rest(prompt, self.llm)
        # Handle documentation
```

#### 5. Modify `main.py`

Set up the main script to choose between models and start processing.

```python
import argparse
import asyncio
from process_manager import DocumentationProcessManager

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate documentation using Gemini.")
    parser.add_argument("--model", choices=["azure", "gemini"], default="gemini", help="Choose the language model")
    return parser.parse_args()

async def main():
    args = parse_arguments()
    project = "inbound-lattice-435123-g7"
    location = "us-central1"

    manager = DocumentationProcessManager(project=project, location=location, model=args.model)
    file_paths = ["path/to/your/files"]  # Replace with actual file paths
    await manager.process_files(file_paths)

if __name__ == "__main__":
    asyncio.run(main())
```

### Explanation

- **Gemini Client:** A dedicated class to manage API interactions with Gemini. It initializes the model and provides a method to generate text.
  
- **Model Selection:** The script allows switching between Azure and Gemini using a command-line argument. This flexibility lets you choose the best model for your needs.

- **API Integration:** The `fetch_documentation_rest` function is adapted to use the Gemini client, handling retries and JSON parsing.

- **Process Manager:** The `DocumentationProcessManager` is updated to manage the lifecycle of processing files using the selected model.

- **Main Script:** The main script sets up the environment and starts the documentation generation process, allowing you to specify the model via command-line arguments.

This setup provides a comprehensive integration of the Gemini API into your script, allowing for flexible model selection and robust error handling. Make sure to replace placeholders with actual values and test thoroughly.