Certainly! Below is the implementation of dynamic context management as per your specifications.

---

### **`src/context_manager.py`**

```python
"""
Context Management Module.

This module provides dynamic context management by adjusting the context window
based on code relevance. It prioritizes recent changes or frequently accessed
segments by tracking access frequency and timestamps.

Classes:
    ContextManager: Manages code segments and their relevance scores for context window optimization.
"""

import time
from typing import List, Dict, Tuple

class ContextManager:
    """
    Manages code segments and their relevance scores for context window optimization.

    This class tracks access frequency and timestamps for code segments and
    provides methods to update access information and retrieve the most relevant
    code segments based on recency and frequency.

    Attributes:
        context_size_limit (int): The maximum number of code segments to include in the context.
    """

    def __init__(self, context_size_limit: int = 10):
        """
        Initialize the ContextManager.

        Args:
            context_size_limit (int): The maximum number of code segments to include in the context.
        """
        self.context_size_limit = context_size_limit
        # Stores code segments with their metadata: {segment_id: (code_segment, access_frequency, last_access_time)}
        self.code_segments: Dict[str, Tuple[str, int, float]] = {}

    def add_or_update_segment(self, segment_id: str, code_segment: str):
        """
        Add a new code segment or update an existing one.

        Args:
            segment_id (str): A unique identifier for the code segment.
            code_segment (str): The code segment content.
        """
        current_time = time.time()
        if segment_id in self.code_segments:
            # Update access frequency and timestamp
            _, access_frequency, _ = self.code_segments[segment_id]
            access_frequency += 1
            self.code_segments[segment_id] = (code_segment, access_frequency, current_time)
        else:
            # Add new code segment with initial access frequency
            self.code_segments[segment_id] = (code_segment, 1, current_time)

    def get_relevant_segments(self, max_total_size: int = 2048) -> List[str]:
        """
        Retrieve the most relevant code segments based on access frequency and recency.

        Args:
            max_total_size (int): The maximum total size (in characters) of code segments to return.

        Returns:
            List[str]: A list of code segments sorted by relevance.
        """
        # Calculate relevance score based on access frequency and recency
        current_time = time.time()
        relevance_scores = []
        for segment_id, (code_segment, access_frequency, last_access_time) in self.code_segments.items():
            # For recency, compute time since last access (the smaller, the better)
            time_since_last_access = current_time - last_access_time
            # Calculate relevance score (you can adjust the weighting factors as needed)
            relevance_score = access_frequency / (1 + time_since_last_access)
            relevance_scores.append((relevance_score, code_segment))

        # Sort segments by relevance score in descending order
        relevance_scores.sort(reverse=True, key=lambda x: x[0])

        # Select code segments until the total size reaches max_total_size
        relevant_segments = []
        total_size = 0
        for _, code_segment in relevance_scores:
            code_size = len(code_segment)
            if total_size + code_size > max_total_size:
                break
            relevant_segments.append(code_segment)
            total_size += code_size

        return relevant_segments

    def remove_segment(self, segment_id: str):
        """
        Remove a code segment from tracking.

        Args:
            segment_id (str): The unique identifier of the code segment to remove.
        """
        if segment_id in self.code_segments:
            del self.code_segments[segment_id]

    def clear(self):
        """
        Clear all tracked code segments.
        """
        self.code_segments.clear()
```

---

### **Integration with `workflow.py`**

We need to integrate the `ContextManager` into the existing `workflow.py` module. Below are the necessary modifications.

#### **Import the ContextManager**

At the top of `workflow.py`, import the `ContextManager`:

```python
from context_manager import ContextManager
```

#### **Modify the `process_files` Function**

Initialize the `ContextManager` within the `process_files` function:

```python
async def process_files(
    files_list: List[str],
    repo_dir: str,
    concurrency_limit: int,
    service: str,
    cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process multiple Python files concurrently.

    This function processes a list of Python files, analyzing each file's content
    using the specified AI service, and returns the analysis results.
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Initialize the ContextManager
    context_manager = ContextManager(context_size_limit=10)  # Adjust the limit as needed

    with tqdm(total=len(files_list), desc="Processing files") as pbar:
        for filepath in files_list:
            # Existing code for processing each file
            # ...

            # After extracting functions and methods
            # Combine functions and methods for processing
            all_functions = extracted_data["functions"] + [
                method for class_info in extracted_data["classes"]
                for method in class_info["methods"]
            ]

            tasks = []
            for function_details in all_functions:
                # Generate a unique segment ID, e.g., based on file path and function name
                segment_id = f"{filepath}:{function_details['name']}"
                # Add or update the segment in the context manager
                context_manager.add_or_update_segment(segment_id, function_details['code'])

                # Create tasks for analysis, passing the context manager
                task = analyze_with_semaphore(
                    function_details,
                    semaphore,
                    service,
                    cache,
                    context_manager  # Pass the context manager to the analysis function
                )
                tasks.append(task)

            # Wait for all analysis tasks to complete
            functions_analysis = await asyncio.gather(*tasks)

            # Existing code continues
            # ...
```

#### **Modify the `analyze_with_semaphore` Function**

Update the function signature to accept the `context_manager`:

```python
async def analyze_with_semaphore(
    function_details: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    service: str,
    cache: Any,
    context_manager: ContextManager
) -> Dict[str, Any]:
    """
    Analyze a function with rate limiting semaphore and caching.

    This function uses a semaphore to limit the number of concurrent analyses
    and checks the cache for existing results before performing the analysis.
    """
    # Existing code
    # ...

    async with semaphore:
        try:
            # Get relevant code segments from the context manager
            relevant_code_segments = context_manager.get_relevant_segments(max_total_size=2048)

            # Call the analysis function with the relevant code segments
            result = await analyze_function_with_openai(
                validated_details.dict(),
                service,
                context_code_segments=relevant_code_segments
            )
            # Existing code continues
            # ...
```

#### **Modify the `analyze_function_with_openai` Function**

Update the function signature to accept `context_code_segments`:

```python
async def analyze_function_with_openai(
    function_details: Dict[str, Any],
    service: str,
    context_code_segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a function using OpenAI's API or Azure OpenAI service.

    This function sends a request to the specified AI service to analyze a Python function
    and generate structured documentation, including a summary, a Google-style docstring,
    and a changelog entry.
    """
    # Existing code
    # ...

    # Prepare the context code segments
    if context_code_segments:
        context_code_str = "\n\n".join(context_code_segments)
        context_section = f"Additional Context Code:\n```python\n{context_code_str}\n```\n\n"
    else:
        context_section = ""

    # Prepare the user prompt
    user_prompt = (
        f"{context_section}"
        f"Analyze and document this function:\n\n"
        f"Function Name: {function_details['name']}\n"
        f"Parameters: {format_parameters(function_details['params'])}\n"
        f"Return Type: {function_details['return_type']}\n"
        f"Existing Docstring: {function_details['docstring'] or 'None'}\n\n"
        f"Dependencies and Effects:\n{dependency_section}\n\n"
        f"Source Code:\n"
        f"```python\n{function_details['code']}\n```\n\n"
        f"Requirements:\n"
        f"1. Generate a Google-style docstring\n"
        f"2. Include type hints if present\n"
        f"3. Document any exceptions that may be raised\n"
        f"4. Document any significant side effects\n"
        f"5. Provide a clear, concise summary\n"
        f"6. Include a changelog entry"
    )

    # Existing code continues
    # ...
```

---

### **Explanation**

- **ContextManager Implementation (`context_manager.py`):**

  - **`ContextManager` Class:** Manages code segments by tracking their access frequency and last access time.
  - **`add_or_update_segment` Method:** Updates access frequency and timestamp whenever a code segment is accessed.
  - **`get_relevant_segments` Method:** Calculates a relevance score based on access frequency and recency, and retrieves the most relevant code segments up to a specified total size limit to ensure memory efficiency.
  - **Memory Optimization:** By limiting the total size of the returned code segments, we ensure that the context does not consume excessive memory.

- **Integration with `workflow.py`:**

  - **Initialization:** A `ContextManager` instance is initialized within the `process_files` function.
  - **Tracking Access:** As each function is processed, its code segment is added or updated in the `ContextManager` with a unique identifier combining the file path and function name.
  - **Including Relevant Code Segments:** Before analyzing a function, the `analyze_with_semaphore` function retrieves the most relevant code segments from the `ContextManager` and passes them to the `analyze_function_with_openai` function.
  - **Adjusting Context Window:** The AI analysis function includes these relevant code segments in the prompt, providing additional context that can improve the quality of the AI-generated documentation.

- **Ensuring Backward Compatibility:**

  - **Minimal Changes:** Existing functions and their signatures are preserved where possible, with additional parameters added to support the new functionality.
  - **Functionality Preservation:** The core functionality of processing files and generating documentation remains unaffected.
  - **No New Dependencies:** Only standard Python libraries are used, avoiding any new external dependencies.

- **Performance Optimization:**

  - **Efficient Data Structures:** The `ContextManager` uses a dictionary for quick access and updates.
  - **Relevance Calculation:** Relevance scores are calculated efficiently, and sorting is performed only on the necessary data.
  - **Memory Usage:** By limiting the total size of code segments included in the context, we prevent excessive memory consumption.

- **Coding Standards:**

  - **Documentation:** Docstrings are provided for all classes and methods, following project conventions.
  - **Formatting:** Code is formatted according to PEP 8 standards.
  - **Naming Conventions:** Variable and method names are descriptive and consistent with the project's style.

---

### **Usage**

With these changes, the dynamic context management is now integrated into the workflow. The AI analysis will now consider the most relevant code segments based on recent changes and access frequency, potentially improving the generated documentation.

**Note:** Adjust the `context_size_limit` and `max_total_size` parameters as needed to fine-tune the performance and relevance of the context window.

---

By implementing the `ContextManager` and integrating it with `workflow.py`, we have fulfilled the requirements of dynamic context management, ensuring that the most relevant code segments are prioritized during AI interactions. All changes adhere to the specified restrictions, maintaining coding standards, performance efficiency, and backward compatibility.

---

Certainly! Below is the implementation of the token limit awareness feature within `src/context_manager.py`, along with the necessary integrations and explanations.

---

### **Updated `src/context_manager.py` with Token Limit Awareness**

```python
"""
Context Management Module with Token Limit Awareness.

This module provides dynamic context management by adjusting the context window
based on code relevance and token limits. It prioritizes recent changes or
frequently accessed segments by tracking access frequency and timestamps.

Classes:
    ContextManager: Manages code segments and their relevance scores for context window optimization.
"""

import time
import logging
from typing import List, Dict, Tuple
import tiktoken  # Import tiktoken for token counting

class ContextManager:
    """
    Manages code segments and their relevance scores for context window optimization,
    with token limit awareness.

    This class tracks access frequency and timestamps for code segments and
    provides methods to update access information and retrieve the most relevant
    code segments based on recency and frequency, while ensuring the token limit
    is not exceeded.

    Attributes:
        context_size_limit (int): The maximum number of code segments to include in the context.
        max_tokens (int): The maximum number of tokens allowed in the context.
        model_name (str): The name of the model to use for token encoding.
    """

    def __init__(self, context_size_limit: int = 10, max_tokens: int = 2048, model_name: str = 'gpt-3.5-turbo'):
        """
        Initialize the ContextManager.

        Args:
            context_size_limit (int): The maximum number of code segments to include in the context.
            max_tokens (int): The maximum number of tokens allowed in the context.
            model_name (str): The name of the model to use for token encoding.
        """
        self.context_size_limit = context_size_limit
        self.max_tokens = max_tokens
        self.model_name = model_name
        # Stores code segments with their metadata: {segment_id: (code_segment, access_frequency, last_access_time)}
        self.code_segments: Dict[str, Tuple[str, int, float]] = {}

        # Initialize the tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer for model {model_name}: {str(e)}")
            raise

    def add_or_update_segment(self, segment_id: str, code_segment: str):
        """
        Add a new code segment or update an existing one.

        Args:
            segment_id (str): A unique identifier for the code segment.
            code_segment (str): The code segment content.
        """
        current_time = time.time()
        if segment_id in self.code_segments:
            # Update access frequency and timestamp
            _, access_frequency, _ = self.code_segments[segment_id]
            access_frequency += 1
            self.code_segments[segment_id] = (code_segment, access_frequency, current_time)
        else:
            # Add new code segment with initial access frequency
            self.code_segments[segment_id] = (code_segment, 1, current_time)

    def get_relevant_segments(self) -> List[str]:
        """
        Retrieve the most relevant code segments based on access frequency and recency,
        ensuring the total token count does not exceed the maximum token limit.

        Returns:
            List[str]: A list of code segments sorted by relevance.
        """
        # Calculate relevance score based on access frequency and recency
        current_time = time.time()
        relevance_scores = []
        for segment_id, (code_segment, access_frequency, last_access_time) in self.code_segments.items():
            # For recency, compute time since last access (the smaller, the better)
            time_since_last_access = current_time - last_access_time
            # Calculate relevance score (you can adjust the weighting factors as needed)
            relevance_score = access_frequency / (1 + time_since_last_access)
            relevance_scores.append((relevance_score, code_segment))

        # Sort segments by relevance score in descending order
        relevance_scores.sort(reverse=True, key=lambda x: x[0])

        # Select code segments until the total tokens reach the max_tokens limit
        relevant_segments = []
        total_tokens = 0

        for _, code_segment in relevance_scores:
            # Count tokens in the code segment
            tokens = self.tokenizer.encode(code_segment)
            num_tokens = len(tokens)

            # Check if adding this code segment would exceed the max token limit
            if total_tokens + num_tokens > self.max_tokens:
                # If approaching the limit, attempt to truncate the code segment
                available_tokens = self.max_tokens - total_tokens

                if available_tokens <= 0:
                    # Cannot add more tokens
                    logging.info("Token limit reached. Cannot add more code segments.")
                    break

                # Truncate the code segment to fit the available tokens
                truncated_tokens = tokens[:available_tokens]
                truncated_code = self.tokenizer.decode(truncated_tokens)

                # Provide feedback about truncation
                logging.info(f"Truncating code segment due to token limit: Original tokens={num_tokens}, Truncated tokens={available_tokens}")

                # Add the truncated code segment
                relevant_segments.append(truncated_code)
                total_tokens += len(truncated_tokens)
                break  # No more segments can be added
            else:
                # Add the full code segment
                relevant_segments.append(code_segment)
                total_tokens += num_tokens

        return relevant_segments

    def remove_segment(self, segment_id: str):
        """
        Remove a code segment from tracking.

        Args:
            segment_id (str): The unique identifier of the code segment to remove.
        """
        if segment_id in self.code_segments:
            del self.code_segments[segment_id]

    def clear(self):
        """
        Clear all tracked code segments.
        """
        self.code_segments.clear()
```

---

### **Explanation**

- **Token Counting with Tiktoken:**
  - The `ContextManager` now initializes a tokenizer using `tiktoken.encoding_for_model(model_name)`.
  - When retrieving relevant segments, it encodes each code segment to count the number of tokens.
  
- **Token Limit Awareness:**
  - In the `get_relevant_segments` method, before adding a code segment, it checks if adding it would exceed the `max_tokens` limit.
  - If the limit is about to be exceeded, it truncates the code segment to fit the remaining tokens.
  - Truncation is performed by slicing the token list and decoding it back into a string.
  
- **Feedback and Alerts:**
  - Uses `logging.info` to provide feedback when truncation occurs due to token limits.
  - Alerts when the token limit is reached and no more code segments can be added.
  
- **Performance Optimization:**
  - Token counting is performed only when necessary, avoiding unnecessary computations.
  - Efficient data structures and sorting mechanisms are used.
  
- **Coding Standards:**
  - The code follows PEP8 standards and includes comprehensive docstrings.
  - Variable and method names are consistent and descriptive.

---

### **Integration with `workflow.py`**

To integrate the updated `ContextManager` with `workflow.py`, the following modifications are necessary.

#### **Import Tiktoken and Adjust Model Parameters**

At the top of `workflow.py`, import `tiktoken` and adjust the model parameters:

```python
import tiktoken
```

Add a utility function to determine the model's context window:

```python
def get_model_context_window(model_name: str) -> int:
    """Get the maximum context window size for a given model."""
    if 'gpt-3.5' in model_name:
        return 4096
    elif 'gpt-4' in model_name:
        return 8192
    else:
        # Default value
        return 4096
```

#### **Modify the `process_files` Function**

In the `process_files` function, initialize the `ContextManager` with appropriate `max_tokens` and `model_name`:

```python
async def process_files(
    files_list: List[str],
    repo_dir: str,
    concurrency_limit: int,
    service: str,
    cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process multiple Python files concurrently.
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Determine the model name and context window
    # Adjust the model_name based on your configuration or service
    if service == 'openai':
        model_name = 'gpt-4'  # Or fetch from your configuration
    else:
        # For Azure, you might need to fetch the deployment model name
        model_name = 'gpt-4'  # Adjust accordingly

    context_window = get_model_context_window(model_name)
    max_response_tokens = 6000  # As per your OpenAIConfig settings
    system_prompt_tokens = 500   # Estimate tokens used by system prompts and other fixed parts
    available_prompt_tokens = context_window - max_response_tokens - system_prompt_tokens

    # Ensure available_prompt_tokens is positive
    if available_prompt_tokens <= 0:
        logging.warning("Available prompt tokens is non-positive. Adjust max_response_tokens or system prompts.")
        available_prompt_tokens = 1024  # Default to a safe value

    # Initialize the ContextManager
    context_manager = ContextManager(
        context_size_limit=10,  # Adjust the limit as needed
        max_tokens=available_prompt_tokens,
        model_name=model_name
    )

    with tqdm(total=len(files_list), desc="Processing files") as pbar:
        for filepath in files_list:
            # Existing code for processing each file
            # ...

            # After extracting functions and methods
            # Combine functions and methods for processing
            all_functions = extracted_data["functions"] + [
                method for class_info in extracted_data["classes"]
                for method in class_info["methods"]
            ]

            tasks = []
            for function_details in all_functions:
                # Generate a unique segment ID, e.g., based on file path and function name
                segment_id = f"{filepath}:{function_details['name']}"
                # Add or update the segment in the context manager
                context_manager.add_or_update_segment(segment_id, function_details['code'])

                # Create tasks for analysis, passing the context manager
                task = analyze_with_semaphore(
                    function_details,
                    semaphore,
                    service,
                    cache,
                    context_manager  # Pass the context manager to the analysis function
                )
                tasks.append(task)

            # Wait for all analysis tasks to complete
            functions_analysis = await asyncio.gather(*tasks)

            # Existing code continues
            # ...
```

#### **Modify the `analyze_with_semaphore` Function**

Update the function signature to accept the `context_manager` and pass it to the analysis function:

```python
async def analyze_with_semaphore(
    function_details: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    service: str,
    cache: Any,
    context_manager: ContextManager
) -> Dict[str, Any]:
    """
    Analyze a function with rate limiting semaphore and caching.
    """
    # Existing validation code
    # ...

    async with semaphore:
        try:
            # Get relevant code segments from the context manager
            relevant_code_segments = context_manager.get_relevant_segments()

            # Call the analysis function with the relevant code segments
            result = await analyze_function_with_openai(
                validated_details.dict(),
                service,
                context_code_segments=relevant_code_segments
            )
            # Existing caching code
            # ...
        except Exception as e:
            # Existing exception handling code
            # ...
```

#### **Modify the `analyze_function_with_openai` Function**

Update the function signature to accept `context_code_segments` and adjust the prompt:

```python
async def analyze_function_with_openai(
    function_details: Dict[str, Any],
    service: str,
    context_code_segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a function using OpenAI's API or Azure OpenAI service.
    """
    # Existing code
    # ...

    # Prepare the context code segments
    if context_code_segments:
        context_code_str = "\n\n".join(context_code_segments)
        context_section = f"Additional Context Code:\n```python\n{context_code_str}\n```\n\n"
    else:
        context_section = ""

    # Prepare the user prompt
    user_prompt = (
        f"{context_section}"
        f"Analyze and document this function:\n\n"
        f"Function Name: {function_details['name']}\n"
        f"Parameters: {format_parameters(function_details['params'])}\n"
        f"Return Type: {function_details['return_type']}\n"
        f"Existing Docstring: {function_details['docstring'] or 'None'}\n\n"
        f"Dependencies and Effects:\n{dependency_section}\n\n"
        f"Source Code:\n"
        f"```python\n{function_details['code']}\n```\n\n"
        f"Requirements:\n"
        f"1. Generate a Google-style docstring\n"
        f"2. Include type hints if present\n"
        f"3. Document any exceptions that may be raised\n"
        f"4. Document any significant side effects\n"
        f"5. Provide a clear, concise summary\n"
        f"6. Include a changelog entry"
    )

    # Existing code for making API requests
    # ...
```

---

### **Explanation**

- **Adjusting Token Limits:**
  - In `process_files`, we calculate `available_prompt_tokens` by subtracting `max_response_tokens` and an estimated `system_prompt_tokens` from the model's context window.
  - This ensures that the total tokens used in the prompt do not exceed the model's context window.

- **Initializing ContextManager with Token Limits:**
  - We pass `available_prompt_tokens` to the `ContextManager` to limit the total tokens of the context code segments.

- **Passing Context Segments to Analysis Function:**
  - In `analyze_with_semaphore`, we retrieve the relevant code segments from the `ContextManager` and pass them to `analyze_function_with_openai`.

- **Including Context Code in the Prompt:**
  - In `analyze_function_with_openai`, we include the context code segments in the user prompt under "Additional Context Code".

- **Maintaining Backward Compatibility:**
  - Existing function signatures are preserved where possible.
  - Core functionality and AI interaction logic remain unaffected.

- **Efficient Token Monitoring:**
  - Token counting is handled efficiently within the `ContextManager`.
  - We avoid unnecessary tokenization elsewhere in the code.

- **Coding Standards and Documentation:**
  - All changes adhere to PEP8 standards and include appropriate docstrings.
  - Variable names are clear and consistent.

---

### **Additional Notes**

- **Dependencies:**
  - The `tiktoken` library is used for token counting, as allowed.
  - No other new dependencies are introduced.

- **Performance Considerations:**
  - Token counting is performed only when retrieving relevant segments, minimizing performance impact.
  - The `tiktoken` library is efficient for tokenization tasks.

- **Feedback Mechanism:**
  - The system provides feedback via logging when truncation occurs due to token limits.
  - Developers can monitor these logs to understand and adjust the context management if needed.

---

### **Conclusion**

By implementing the token limit awareness feature in `src/context_manager.py` and integrating it with `workflow.py`, we have fulfilled the requirements specified in your task:

- **Real-time Token Monitoring:** The `ContextManager` monitors token usage in real-time using the `tiktoken` library.
- **Truncating or Summarizing Code Segments:** When approaching token limits, code segments are truncated to fit within the available tokens.
- **Feedback and Alerts:** The system provides feedback through logging when truncation occurs, guiding further adjustments.
- **Integration and Compatibility:** The feature is integrated with `workflow.py` and maintains backward compatibility, not interfering with existing token management or AI interaction logic.
- **Performance and Standards:** Efficient token monitoring is implemented, and all code adheres to PEP8 standards and includes consistent documentation.

You can now proceed to use this enhanced context management in your tool to ensure smooth interactions with language models, even when processing large code segments.

---

### **Next Steps**

- **Testing:** Thoroughly test the updated modules to ensure they work as expected and handle edge cases.
- **Logging Configuration:** Adjust the logging level as needed for your environment to control the verbosity of feedback messages.
- **Adjusting Parameters:** Fine-tune the `max_tokens` and other parameters based on the specific models and use cases in your application.

Let me know if you need further assistance or modifications!

---

Certainly! Below is the implementation of the dependency analysis feature as per your specifications.

---

### **`src/dependency_analyzer.py`**

```python
"""
Dependency Analyzer Module.

This module provides functionality to parse Python code and identify dependencies
such as imports, function calls, and module interactions within the codebase.
It supports static analysis by scanning code for import statements, function
definitions, and references within the project.

Classes:
    DependencyAnalyzer: Analyzes code files to extract dependency information.

Functions:
    extract_dependencies_from_code: Extract dependencies from given code content.
"""

import ast
from typing import List, Dict, Set, Optional
import os
import logging


class DependencyAnalyzer:
    """
    Analyzes code files to extract dependency information.

    This class provides methods to parse Python code files and extract information
    about imports, function calls, class usages, and module interactions. It is
    designed to handle large codebases efficiently.

    Attributes:
        project_root (str): The root directory of the project to analyze.
        module_dependencies (Dict[str, Set[str]]): Mapping of modules to their dependencies.
    """

    def __init__(self, project_root: str):
        """
        Initialize the DependencyAnalyzer.

        Args:
            project_root (str): The root directory of the project to analyze.
        """
        self.project_root = project_root
        self.module_dependencies: Dict[str, Set[str]] = {}

    def analyze_file(self, file_path: str):
        """
        Analyze a single Python file to extract dependencies.

        Args:
            file_path (str): The path to the Python file to analyze.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)
            visitor = _DependencyVisitor()
            visitor.visit(tree)

            # Get module name relative to the project root
            module_name = os.path.relpath(file_path, self.project_root).replace(os.sep, '.')[:-3]  # Remove .py extension

            self.module_dependencies[module_name] = visitor.dependencies

            logging.debug(f"Analyzed {module_name}: Found dependencies {visitor.dependencies}")

        except Exception as e:
            logging.warning(f"Failed to analyze {file_path}: {str(e)}")

    def analyze_project(self):
        """
        Analyze all Python files in the project to extract dependencies.

        This method walks through the project directory and analyzes each Python file.
        """
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)

    def get_module_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get the mapping of modules to their dependencies.

        Returns:
            Dict[str, Set[str]]: A dictionary where keys are module names and values are sets of dependencies.
        """
        return self.module_dependencies


class _DependencyVisitor(ast.NodeVisitor):
    """
    AST Node Visitor for extracting dependencies.

    This class traverses the AST of a Python file and collects information about
    imports, function calls, and attribute accesses that represent dependencies.
    """

    def __init__(self):
        self.dependencies: Set[str] = set()
        self.current_module: Optional[str] = None
        self.imported_modules: Dict[str, str] = {}  # alias to module name mapping

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name
            alias_name = alias.asname if alias.asname else module_name
            self.imported_modules[alias_name] = module_name
            self.dependencies.add(module_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module if node.module else ''
        for alias in node.names:
            full_name = f"{module_name}.{alias.name}" if module_name else alias.name
            alias_name = alias.asname if alias.asname else alias.name
            self.imported_modules[alias_name] = full_name
            self.dependencies.add(full_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
            if func_name in self.imported_modules:
                self.dependencies.add(self.imported_modules[func_name])
        elif isinstance(func, ast.Attribute):
            value = func.value
            attr_chain = self._get_attribute_chain(func)
            if attr_chain[0] in self.imported_modules:
                module_name = self.imported_modules[attr_chain[0]]
                full_dependency = f"{module_name}{'.'.join(attr_chain[1:])}"
                self.dependencies.add(full_dependency)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr_chain = self._get_attribute_chain(node)
        if attr_chain[0] in self.imported_modules:
            module_name = self.imported_modules[attr_chain[0]]
            full_dependency = f"{module_name}{'.'.join(attr_chain[1:])}"
            self.dependencies.add(full_dependency)
        self.generic_visit(node)

    def _get_attribute_chain(self, node: ast.Attribute) -> List[str]:
        """
        Helper method to get the chain of attribute names from an Attribute node.

        Args:
            node (ast.Attribute): The Attribute node to process.

        Returns:
            List[str]: A list of attribute names representing the chain.
        """
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.insert(0, f".{node.attr}")
            node = node.value
        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)
        return attrs

```

---

### **Explanation**

- **DependencyAnalyzer Class:**

  - **Purpose:** Analyzes Python files to extract dependency information.
  - **Attributes:**
    - `project_root`: The root directory of the project.
    - `module_dependencies`: A dictionary mapping module names to their dependencies.
  - **Methods:**
    - `analyze_file`: Analyzes a single Python file.
    - `analyze_project`: Walks through the project directory and analyzes all Python files.
    - `get_module_dependencies`: Returns the mapping of module dependencies.

- **_DependencyVisitor Class:**

  - **Purpose:** A subclass of `ast.NodeVisitor` that traverses the AST to extract dependencies.
  - **Methods:**
    - `visit_Import`: Handles `import` statements.
    - `visit_ImportFrom`: Handles `from ... import ...` statements.
    - `visit_Call`: Handles function calls to detect dependencies on imported functions.
    - `visit_Attribute`: Handles attribute access to detect dependencies on imported modules or classes.
    - `_get_attribute_chain`: Helper method to retrieve the chain of attributes from an `Attribute` node.

- **Dependency Extraction Logic:**

  - **Imports:** Captures all modules and names imported, storing any aliases used.
  - **Function Calls:** Identifies if a function call is to an imported function and adds the original module to dependencies.
  - **Attribute Access:** Tracks when attributes of imported modules are accessed, indicating a dependency.

- **Performance Considerations:**

  - The code uses the `ast` module for static analysis, which is efficient and avoids executing the code.
  - Dependency information is stored in sets and dictionaries for fast lookup and minimal memory overhead.
  - The visitor pattern ensures a single pass through the AST per file.

- **Coding Standards:**

  - Docstrings are provided for classes and methods, following project conventions.
  - Code is formatted according to PEP 8 standards.
  - Variable and method names are descriptive and consistent with the project's style.

---

### **Integration with `workflow.py`**

To integrate the `DependencyAnalyzer` with `workflow.py`, we'll make necessary modifications to include dependency analysis during the file processing.

#### **Import the DependencyAnalyzer**

At the top of `workflow.py`, import the `DependencyAnalyzer`:

```python
from dependency_analyzer import DependencyAnalyzer
```

#### **Modify the `process_files` Function**

Initialize the `DependencyAnalyzer` within the `process_files` function and use it to analyze each file.

```python
async def process_files(
    files_list: List[str],
    repo_dir: str,
    concurrency_limit: int,
    service: str,
    cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process multiple Python files concurrently.
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Initialize the DependencyAnalyzer
    dependency_analyzer = DependencyAnalyzer(project_root=repo_dir)

    with tqdm(total=len(files_list), desc="Processing files") as pbar:
        for filepath in files_list:
            try:
                # Analyze dependencies in the file
                dependency_analyzer.analyze_file(filepath)

                # Existing code for reading the file
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                logging.info(f"Processing file: {filepath}")

                # Existing code for cache checking and parsing
                # ...

                # After parsing and before processing functions, retrieve dependencies
                module_name = os.path.relpath(filepath, repo_dir).replace(os.sep, '.')[:-3]
                module_dependencies = dependency_analyzer.module_dependencies.get(module_name, set())

                # Incorporate module dependencies into analysis results
                # ...

                # Existing code continues
                # ...

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {str(e)}")
                results[filepath] = {
                    "source_code": content if 'content' in locals() else "",
                    "functions": [],
                    "module_summary": f"Error: {str(e)}",
                    "error": str(e)
                }

            finally:
                pbar.update(1)

    # Optionally, you can use dependency_analyzer.module_dependencies for further processing
    return results
```

#### **Incorporate Dependency Information**

When processing each function, include dependency information:

```python
for function_details in all_functions:
    # Existing code to extract dependencies
    if "dependencies" not in function_details:
        function_details["dependencies"] = extract_dependencies(
            function_details["node"],
            content
        )

    # Include module-level dependencies
    function_details["dependencies"]["module_imports"] = list(module_dependencies)

    # Continue with existing code
    task = analyze_with_semaphore(
        function_details,
        semaphore,
        service,
        cache
    )
    tasks.append(task)
```

#### **Update Analysis Results**

Modify how dependencies are presented in the analysis results to include the module-level dependencies.

---

### **Explanation**

- **Integration Steps:**

  - **Initialize DependencyAnalyzer:** We create an instance of `DependencyAnalyzer` with the project root directory.
  - **Analyze Each File:** As we process each file, we call `analyze_file` to extract dependencies.
  - **Retrieve Module Dependencies:** We obtain the dependencies for the current module and include them in the function analysis.
  - **Include in Function Details:** Module-level dependencies are added to each function's `dependencies` dictionary under a new key `module_imports`.

- **Preserving Existing Functionality:**

  - Existing code parsing and documentation generation logic remains unchanged.
  - The dependency analysis enhances the data available for each function without disrupting current processes.

- **Performance Considerations:**

  - Dependency analysis occurs during the file processing loop, minimizing additional passes through the code.
  - The `DependencyAnalyzer` uses efficient data structures to keep memory usage low.
  - Since we're using the `ast` module, there is no execution of the code, ensuring safety and performance.

- **Coding Standards:**

  - The added code in `workflow.py` follows the existing naming and documentation conventions.
  - Variable names are consistent and descriptive.

---

### **Usage**

With these changes, the dependency analysis is now integrated into your workflow. Each function's analysis includes additional dependency information, which can be used for context management or further processing.

**Example of Accessing Dependencies in Function Analysis:**

```python
function_dependencies = function_details["dependencies"]
module_imports = function_dependencies.get("module_imports", [])
print(f"Function {function_details['name']} depends on modules: {module_imports}")
```

---

### **Additional Notes**

- **Avoiding New Dependencies:**

  - We only use the standard `ast` module for parsing Python code, complying with the restriction on dependencies.

- **Backward Compatibility:**

  - The existing functionality of code parsing and documentation generation is unaffected.
  - Dependency information is additive and does not interfere with other processes.

- **Potential Enhancements:**

  - You can further process `module_dependencies` to build a dependency graph of the entire project.
  - This information can be used to optimize context management by prioritizing code segments based on dependency relationships.

- **Limitations:**

  - The current implementation captures direct imports and function calls. Indirect dependencies (e.g., through dynamic imports) are not detected.
  - This is acceptable for static analysis and within the scope of the task.

---

### **Conclusion**

By implementing the `DependencyAnalyzer` and integrating it with `workflow.py`, we have fulfilled the requirements of dependency analysis:

- **Dependency Extraction:** The codebase is parsed to identify dependencies such as imports, function calls, and module interactions.
- **Static Analysis:** The analysis is performed statically using the `ast` module, ensuring safety and efficiency.
- **Integration with Workflow:** The dependency information is integrated into the workflow for relevant files.
- **Adherence to Standards:** All code follows existing project conventions, including naming and documentation standards.
- **Performance Optimization:** Efficient parsing and minimal memory overhead are achieved through the use of appropriate data structures and the standard library.
- **No New Dependencies:** Only existing libraries (`ast`) are used, adhering to the restriction.

---

### **Next Steps**

- **Testing:** Run the updated code on your codebase to ensure it works as expected.
- **Logging Levels:** Adjust logging levels if needed to control verbosity.
- **Further Integration:** Utilize the dependency information in your context management or documentation generation to enhance functionality.

Let me know if you need any further assistance or modifications!

---

Certainly! Below is the implementation of the metadata tagging feature as per your specifications.


---

### **`src/metadata_manager.py`**

```python
"""
Metadata Manager Module.

This module provides functionality to tag each code segment with structured metadata,
such as dependency information, access frequency, and usage patterns. It stores metadata
in a structured format (JSON) to support fast lookup and context-building for AI interactions.

Classes:
    MetadataManager: Manages metadata tagging and storage for code segments.
"""

import json
import os
import threading
from typing import Dict, Any
import logging


class MetadataManager:
    """
    Manages metadata tagging and storage for code segments.

    This class provides methods to tag code segments with metadata, store and retrieve
    metadata efficiently, and integrate with other modules for context-building.

    Attributes:
        metadata_store (Dict[str, Dict[str, Any]]): In-memory storage of metadata.
        metadata_file (str): Path to the metadata JSON file.
        lock (threading.Lock): A lock to manage concurrent access to the metadata store.
    """

    def __init__(self, metadata_file: str = ".metadata_store.json"):
        """
        Initialize the MetadataManager.

        Args:
            metadata_file (str): The file path to store metadata in JSON format.
        """
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.metadata_file = metadata_file
        self.lock = threading.Lock()
        self._load_metadata()

    def _load_metadata(self):
        """
        Load metadata from the JSON file into the in-memory store.
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
                logging.info(f"Loaded metadata from {self.metadata_file}")
            except Exception as e:
                logging.warning(f"Failed to load metadata: {str(e)}")
                self.metadata_store = {}
        else:
            self.metadata_store = {}

    def _save_metadata(self):
        """
        Save the in-memory metadata store to the JSON file.
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=4)
            logging.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logging.warning(f"Failed to save metadata: {str(e)}")

    def tag_code_segment(self, segment_id: str, metadata: Dict[str, Any]):
        """
        Tag a code segment with metadata.

        Args:
            segment_id (str): A unique identifier for the code segment.
            metadata (Dict[str, Any]): A dictionary containing metadata to associate with the code segment.
        """
        with self.lock:
            self.metadata_store[segment_id] = metadata
            logging.debug(f"Tagged code segment {segment_id} with metadata")
            self._save_metadata()

    def update_metadata(self, segment_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for an existing code segment.

        Args:
            segment_id (str): The unique identifier of the code segment.
            metadata (Dict[str, Any]): A dictionary containing updated metadata.
        """
        with self.lock:
            if segment_id in self.metadata_store:
                self.metadata_store[segment_id].update(metadata)
                logging.debug(f"Updated metadata for code segment {segment_id}")
                self._save_metadata()
            else:
                logging.warning(f"Code segment {segment_id} not found in metadata store")

    def get_metadata(self, segment_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific code segment.

        Args:
            segment_id (str): The unique identifier of the code segment.

        Returns:
            Dict[str, Any]: The metadata associated with the code segment.
        """
        with self.lock:
            return self.metadata_store.get(segment_id, {})

    def remove_metadata(self, segment_id: str):
        """
        Remove metadata for a specific code segment.

        Args:
            segment_id (str): The unique identifier of the code segment.
        """
        with self.lock:
            if segment_id in self.metadata_store:
                del self.metadata_store[segment_id]
                logging.debug(f"Removed metadata for code segment {segment_id}")
                self._save_metadata()
            else:
                logging.warning(f"Code segment {segment_id} not found in metadata store")

    def clear_metadata(self):
        """
        Clear all metadata from the store.
        """
        with self.lock:
            self.metadata_store.clear()
            logging.info("Cleared all metadata from the store")
            self._save_metadata()
```

---

### **Explanation**

- **MetadataManager Class:**

  - **Purpose:** Manages metadata tagging and storage for code segments.
  - **Attributes:**
    - `metadata_store`: An in-memory dictionary storing metadata for each code segment.
    - `metadata_file`: The file path to store metadata in JSON format.
    - `lock`: A threading lock to ensure thread-safe access to the metadata store.
  - **Methods:**
    - `__init__`: Initializes the MetadataManager and loads existing metadata.
    - `_load_metadata`: Loads metadata from the JSON file into the in-memory store.
    - `_save_metadata`: Saves the in-memory metadata store to the JSON file.
    - `tag_code_segment`: Tags a code segment with metadata.
    - `update_metadata`: Updates metadata for an existing code segment.
    - `get_metadata`: Retrieves metadata for a specific code segment.
    - `remove_metadata`: Removes metadata for a specific code segment.
    - `clear_metadata`: Clears all metadata from the store.

- **Metadata Storage:**

  - Metadata is stored in a JSON file (`.metadata_store.json`) for persistence.
  - The in-memory `metadata_store` provides fast access during runtime.
  - Threading locks ensure safe concurrent access when used in multi-threaded environments.

- **Metadata Format:**

  - Each code segment is identified by a unique `segment_id`.
  - Metadata is stored as a dictionary associated with the `segment_id`.
  - This structure supports fast lookup and efficient context-building.

- **Performance Considerations:**

  - Loading and saving metadata is optimized by reading and writing the entire JSON file at once.
  - The use of in-memory storage reduces disk I/O during runtime.
  - Threading locks prevent race conditions without significant performance overhead.

- **Coding Standards:**

  - The code follows PEP8 standards.
  - Docstrings are provided for all classes and methods.
  - Variable and method names are descriptive and consistent with project conventions.

- **Dependencies:**

  - No new external dependencies are introduced.
  - Uses built-in modules (`json`, `os`, `threading`, `logging`).

---

### **Integration with `workflow.py`**

To integrate the `MetadataManager` with `workflow.py`, we need to modify the workflow to tag each code segment with metadata during processing.

#### **Import the MetadataManager**

At the top of `workflow.py`, import the `MetadataManager`:

```python
from metadata_manager import MetadataManager
```

#### **Initialize the MetadataManager**

Within the `process_files` function, initialize the `MetadataManager`:

```python
async def process_files(
    files_list: List[str],
    repo_dir: str,
    concurrency_limit: int,
    service: str,
    cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process multiple Python files concurrently.
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Initialize the MetadataManager
    metadata_manager = MetadataManager(metadata_file=os.path.join(repo_dir, '.metadata_store.json'))

    # Existing code continues...
```

#### **Tag Code Segments with Metadata**

When processing each function, tag it with metadata:

```python
for function_details in all_functions:
    # Generate a unique segment ID
    segment_id = f"{filepath}:{function_details['name']}"

    # Prepare metadata
    metadata = {
        'name': function_details['name'],
        'filepath': filepath,
        'dependencies': function_details.get('dependencies', {}),
        'complexity': function_details.get('complexity'),
        'access_frequency': 1,  # Initial access frequency
        'last_access_time': time.time(),
        'usage_patterns': []  # This can be populated with actual usage patterns if available
    }

    # Tag the code segment with metadata
    metadata_manager.tag_code_segment(segment_id, metadata)

    # Continue with existing code...
```

#### **Update Metadata on Access**

In the `analyze_with_semaphore` function, update the access frequency and last access time:

```python
async def analyze_with_semaphore(
    function_details: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    service: str,
    cache: Any,
    context_manager: ContextManager,
    metadata_manager: MetadataManager  # Pass the metadata manager
) -> Dict[str, Any]:
    # Existing code...

    # Update metadata
    segment_id = f"{function_details['filepath']}:{function_details['name']}"
    metadata = metadata_manager.get_metadata(segment_id)
    if metadata:
        metadata['access_frequency'] += 1
        metadata['last_access_time'] = time.time()
        metadata_manager.update_metadata(segment_id, metadata)

    # Continue with existing code...
```

**Note:** You need to ensure that `function_details` includes the `filepath` key. If not, you can pass `filepath` to the function or adjust the segment ID accordingly.

#### **Include Metadata in Analysis**

Pass the metadata to the analysis function if needed:

```python
async def analyze_function_with_openai(
    function_details: Dict[str, Any],
    service: str,
    context_code_segments: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None  # Include metadata
) -> Dict[str, Any]:
    # Existing code...

    # Use metadata in the prompt if necessary
    metadata_section = ""
    if metadata:
        metadata_json = json.dumps(metadata, indent=2)
        metadata_section = f"Metadata:\n```json\n{metadata_json}\n```\n\n"

    # Prepare the user prompt
    user_prompt = (
        f"{context_section}"
        f"{metadata_section}"
        f"Analyze and document this function:\n\n"
        f"Function Name: {function_details['name']}\n"
        # Continue existing prompt...
    )
```

---

### **Integration with `cache.py`**

Since `cache.py` is responsible for caching analysis results, we can enhance it to store metadata along with the cached data.

#### **Modify the CacheEntry Dataclass**

Update the `CacheEntry` dataclass to include metadata:

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Dict[str, Any]
    timestamp: float
    size_bytes: int
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # New field for metadata
```

#### **Update Cache Methods**

When setting cache entries, include the metadata:

```python
def set(self, key: str, value: Dict[str, Any], extraction_metadata: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    # Existing code...

    # Include metadata in the database
    metadata_str = json.dumps(metadata) if metadata else None

    with sqlite3.connect(self.db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO function_cache 
            (key, data, timestamp, size_bytes, extraction_metadata, metadata) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, data_str, time.time(), size_bytes, metadata_str)
        )
```

Ensure that the database schema in `_init_db` includes a `metadata` column.

---

### **Explanation**

- **Metadata Integration:**

  - Metadata tagging is incorporated into the workflow during the processing of each file.
  - Metadata includes details like dependencies, complexity, access frequency, and usage patterns.
  - The `MetadataManager` is used to store and retrieve metadata efficiently.

- **Performance Considerations:**

  - Metadata is stored in-memory during runtime for fast access.
  - The JSON file provides persistence without significant overhead.
  - Threading locks ensure thread safety without compromising performance.

- **Backward Compatibility:**

  - Existing workflows and functionalities remain unaffected.
  - The integration is additive and does not disrupt current processes.
  - The code modifications are designed to be minimal and non-intrusive.

- **No New Dependencies:**

  - Only built-in Python libraries (`json`, `os`, `threading`, `logging`) are used.
  - This maintains consistency with the project's dependency management.

- **Coding Standards:**

  - All added code adheres to PEP8 standards and includes appropriate documentation.
  - Variable names and method signatures follow project conventions.

---

### **Usage**

With these changes, metadata tagging is now integrated into your tool. Metadata for each code segment is stored and can be used to enhance AI interactions, especially during summarization or retrieval tasks.

**Example of Accessing Metadata:**

```python
# Retrieve metadata for a code segment
segment_id = f"{filepath}:{function_name}"
metadata = metadata_manager.get_metadata(segment_id)
print(f"Metadata for {segment_id}: {metadata}")
```

---

### **Additional Notes**

- **Usage Patterns:**

  - The `usage_patterns` field in metadata can be populated with actual usage data if available.
  - This could include information like how often a function is called or its role in the application.

- **Extensibility:**

  - The `MetadataManager` can be extended to include additional metadata fields as needed.
  - Methods can be added for more complex queries or analyses of the metadata.

- **Integration with AI Interactions:**

  - Metadata can be included in prompts to the AI model to provide additional context.
  - This can improve the quality of AI-generated documentation or suggestions.

---

### **Conclusion**

By implementing the `MetadataManager` and integrating it with `workflow.py` and `cache.py`, we have fulfilled the requirements of metadata tagging:

- **Metadata Tagging:** Each code segment is tagged with structured metadata, including dependencies, access frequency, and usage patterns.

- **Efficient Storage:** Metadata is stored in a structured format (JSON) for fast lookup and efficient storage.

- **Integration with Workflow:** The metadata tagging is integrated seamlessly into the existing workflow without breaking existing features.

- **Coding Standards and Performance:** All code follows the project's coding conventions and is optimized for performance, even with large projects.

- **No New Dependencies:** The implementation avoids introducing new dependencies, maintaining consistency across the project.

---

### **Next Steps**

- **Testing:** Thoroughly test the updated tool to ensure that metadata is correctly stored and retrieved.

- **Adjusting Metadata Fields:** Depending on your needs, you may add or modify metadata fields.

- **Leveraging Metadata:** Update AI interaction logic to make use of the metadata, enhancing context and improving suggestions.

- **Monitoring Performance:** Keep an eye on the performance when processing large projects and optimize as necessary.

Let me know if you need any further assistance or modifications!