---
epoch: 1733318185933
modelKey: gpt-4o-2024-11-20|azure openai
tags:
  - copilot-conversation
---

**user**: Only do prompt 1 [Untitled 4](./Untitled%204.md)
[Timestamp: 2024/12/04 07:16:25]

**ai**: It seems like you're referencing a note titled "Untitled 4" in your Obsidian vault. However, I don't have the content of that note. If you could provide more details or the content of the note, I can assist you further with your request.
[Timestamp: 2024/12/04 07:16:28]

**user**: [Bulletproof prompts](./Bulletproof%20prompts.md) only do prompt 1
[Timestamp: 2024/12/04 07:17:03]

**ai**: ### Prompt 1: Initial Steps (Code Extraction and Prompt Generation)

**Target Audience:** Experienced Python developers and AI engineers.

**Task:** Draft the initial part of a comprehensive workflow for an AI-Assisted Docstring Enrichment System in Python. Focus on the initial steps, ensuring to include the purpose, detailed steps, code examples, appropriate tools or libraries, and visualizations where helpful. The workflow should cover:

#### 1. Comprehensive Code Extraction with `CodeExtractor`

- **Purpose:** Explain the need to extract detailed metadata from Python source code to provide the AI with rich context for generating accurate and informative docstrings.

- **Target Audience Considerations:** Assume familiarity with Python's AST module, code analysis techniques, and data serialization formats.

- **Steps:**

  1. **Parsing Source Code:**
     - Use `ast.parse` to convert Python source code into an Abstract Syntax Tree (AST) without executing the code.

  2. **Traversing the AST:**
     - Identify modules, classes, functions, methods, and other code elements.
     - Utilize `ast.NodeVisitor` or `ast.walk` for efficient traversal.

  3. **Extracting Metadata:**
     - **Functions and Methods:**
       - Name, arguments, type hints (using the `typing` module), default values.
       - Return type annotations.
       - Decorators and their effects (e.g., `@staticmethod`, `@classmethod`).
       - Existing docstrings.
       - Complexity indicators (e.g., recursion, algorithmic complexity).
     - **Classes:**
       - Name, base classes, inheritance hierarchy.
       - Attributes and their types.
       - Methods and their metadata.
       - Existing docstrings.
       - Inherited members from parent classes.
     - **Modules:**
       - Name, description.
       - Dependencies (imports and external libraries).
       - Global variables and constants.
       - Existing docstrings.
     - **Comments and Annotations:**
       - Inline comments that provide additional context.
       - Annotations for special behaviors.

  4. **Handling Edge Cases:**
     - Nested classes, inner functions, lambda expressions.
     - Asynchronous functions (`async def`).
     - Generator functions.

  5. **Packaging Metadata:**
     - Organize the extracted information into an `ExtractionResult` object.
     - **Specify Data Format:**
       - Use a Python data class (`@dataclass`) to define `ExtractionResult`.
       - Alternatively, define a JSON schema for serialization.
     - **Suggested Libraries:**
       - `dataclasses` for structured data.
       - `pydantic` for data validation.
       - `json` or `yaml` for serialization.

- **Output Example:**

  ```python
  from dataclasses import dataclass, field
  from typing import List, Dict, Optional

  @dataclass
  class FunctionMetadata:
      name: str
      args: List[Dict[str, str]]
      return_type: Optional[str]
      decorators: List[str]
      docstring: Optional[str]
      complexity: Optional[str]

  @dataclass
  class ClassMetadata:
      name: str
      base_classes: List[str]
      attributes: List[Dict[str, str]]
      methods: List[FunctionMetadata]
      docstring: Optional[str]
      inherited_members: List[str]

  @dataclass
  class ModuleMetadata:
      name: str
      description: Optional[str]
      dependencies: List[str]
      docstring: Optional[str]

  @dataclass
  class ExtractionResult:
      module_metadata: ModuleMetadata
      classes: List[ClassMetadata]
      functions: List[FunctionMetadata]
  ```

- **Visualization:**
  - Include diagrams such as class hierarchies or module dependency graphs to illustrate relationships.

#### 2. Optimized Prompt Generation with `PromptGenerator`

- **Purpose:** Describe how to create concise, informative, and context-rich prompts using the extracted metadata to guide the AI in generating high-quality docstrings that adhere to specified documentation standards.

- **Target Audience Considerations:** Assume knowledge of AI prompt engineering, natural language processing, and familiarity with documentation standards.

- **Steps:**

  1. **Initializing `PromptGenerator`:**
     - Provide module-level context, including the overall purpose and key functionalities.

  2. **Specifying Documentation Standards:**
     - Define the desired docstring format (e.g., Google, NumPy, Sphinx).
     - Include style guidelines and examples within the prompts.
     - **Suggested Tools:**
       - `pydocstyle` or `flake8-docstrings` for enforcing standards.

  3. **Providing Examples:**
     - Offer examples of well-written docstrings in the specified style to guide the AI.

  4. **Generating Tailored Prompts for Each Code Element:**
     - **Include in the Prompt:**
       - Element type (function, method, class, module).
       - Name and signature.
       - Arguments with type hints and default values.
       - Return type annotations.
       - Decorators and their implications.
       - Existing docstrings and comments.
       - Relevant code snippets or pseudocode (if necessary).
       - Dependencies and relationships with other code elements.
       - Complexity notes.
       - Inherited members for classes.
       - Global context when applicable.
     - **Specify Expected AI Response Format:**
       - Request the AI to output docstrings in a structured format (e.g., JSON with specific fields).
     - **Emphasize Critical Information:**
       - Purpose, usage scenarios, important implementation details, edge cases, and error handling.

  5. **Optimizing Prompt Length:**
     - Keep prompts within the AI model's token limits (e.g., GPT-4's context window).
     - Prioritize essential information to avoid overwhelming the AI.

- **Example Prompt:**

  ```plaintext
  You are an AI assistant tasked with generating a Python docstring in the Google style for the following function. Provide the docstring in JSON format with fields: "description", "args", "returns", "raises", and "examples".

  **Function Name:** calculate_average
  **Signature:** calculate_average(numbers: List[float]) -> float
  **Description:** Calculates the average of a list of numbers.
  **Arguments:**
  - **numbers** (*List[float]*): A list of floating-point numbers.
  **Returns:**
  - (*float*): The average of the numbers.
  **Current Docstring:** None
  **Module Context:** Part of the statistics module providing basic statistical functions.
  **Example Usage:**
  ```python
  average = calculate_average([1.5, 2.5, 3.5])
  ```

  Please generate a complete docstring in JSON format adhering to the Google Python Style Guide.
  ```

- **Visualization:**
  - If helpful, include flowcharts to illustrate how the `PromptGenerator` processes input and generates prompts.
[Timestamp: 2024/12/04 07:17:38]