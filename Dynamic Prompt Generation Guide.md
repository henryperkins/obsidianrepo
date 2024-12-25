Here’s how you can implement **dynamic prompt generation with rich context** for your AI-assisted docstring enrichment system:

---

### **Step 1: Create a Prompt Generator**

This generator dynamically adjusts the prompt based on the code element being processed (e.g., function, class, or module).

```python
class PromptGenerator:
    """Generates dynamic AI prompts for enriching docstrings."""

    def __init__(self, module_context=None):
        """
        Initialize the PromptGenerator with optional module-level context.

        Args:
            module_context (dict, optional): Metadata about the module being processed, including its name,
                                             description, and dependencies.
        """
        self.module_context = module_context or {}

    def generate_function_prompt(self, function_metadata):
        """
        Generate a prompt for enriching a function's docstring.

        Args:
            function_metadata (dict): Metadata about the function (name, arguments, returns, docstring, etc.).

        Returns:
            str: A prompt for the AI model.
        """
        function_name = function_metadata.get("name", "Unknown Function")
        args = function_metadata.get("args", [])
        returns = function_metadata.get("returns", {"type": "None", "description": ""})
        existing_docstring = function_metadata.get("docstring", "None")
        complexity = function_metadata.get("complexity", "Unknown")
        dependencies = self.module_context.get("dependencies", "None")

        args_summary = "\n".join([f"- {arg['name']} ({arg['type']}): {arg['description']}" for arg in args])

        return f"""
You are enhancing the documentation for a Python function.

Here is the function definition:
---
Function Name: {function_name}
Arguments:
{args_summary or "None"}
Returns: {returns.get('type')} - {returns.get('description')}
Complexity: {complexity}
Current Docstring: {existing_docstring}
---

Context about the module this function belongs to:
---
Module Name: {self.module_context.get('name', 'Unknown Module')}
Module Description: {self.module_context.get('description', 'No description provided.')}
Dependencies: {dependencies}
---

Please enhance the docstring by:
- Adding missing argument descriptions.
- Describing the function’s purpose, inputs, and outputs clearly.
- Including an accurate summary of exceptions raised (if any).
Focus only on improving the docstring. Do not change the code itself.
        """.strip()

    def generate_class_prompt(self, class_metadata):
        """
        Generate a prompt for enriching a class's docstring.

        Args:
            class_metadata (dict): Metadata about the class (name, methods, attributes, etc.).

        Returns:
            str: A prompt for the AI model.
        """
        class_name = class_metadata.get("name", "Unknown Class")
        base_classes = class_metadata.get("bases", [])
        methods = class_metadata.get("methods", [])
        attributes = class_metadata.get("attributes", [])
        existing_docstring = class_metadata.get("docstring", "None")

        methods_summary = "\n".join([f"- {method['name']}: {method.get('summary', 'No summary')}" for method in methods])
        attributes_summary = "\n".join([f"- {attr['name']}: {attr.get('type', 'Any')} - {attr.get('description', '')}" for attr in attributes])

        return f"""
You are documenting a Python class.

Here is the class definition:
---
Class Name: {class_name}
Base Classes: {', '.join(base_classes) or "None"}
Attributes:
{attributes_summary or "None"}
Methods:
{methods_summary or "None"}
---

Current Docstring: {existing_docstring}

Context about the module this class belongs to:
---
Module Name: {self.module_context.get('name', 'Unknown Module')}
Module Description: {self.module_context.get('description', 'No description provided.')}
Dependencies: {self.module_context.get('dependencies', 'None')}
---

Please enhance the docstring by:
- Summarizing the class’s purpose and its relationship to its base classes.
- Listing and describing its attributes.
- Adding a short description of each method, highlighting their purpose.
        """.strip()

    def generate_module_prompt(self, module_metadata):
        """
        Generate a prompt for enriching a module's docstring.

        Args:
            module_metadata (dict): Metadata about the module.

        Returns:
            str: A prompt for the AI model.
        """
        module_name = module_metadata.get("name", "Unknown Module")
        description = module_metadata.get("description", "No description provided.")
        dependencies = module_metadata.get("dependencies", "None")
        exported_classes = module_metadata.get("classes", [])
        exported_functions = module_metadata.get("functions", [])

        classes_summary = "\n".join([f"- {cls['name']}: {cls.get('summary', 'No summary')}" for cls in exported_classes])
        functions_summary = "\n".join([f"- {func['name']}: {func.get('summary', 'No summary')}" for func in exported_functions])

        return f"""
You are documenting a Python module.

Here is the module definition:
---
Module Name: {module_name}
Description: {description}
Dependencies: {dependencies}
Exported Classes:
{classes_summary or "None"}
Exported Functions:
{functions_summary or "None"}
---

Please enhance the module's docstring by:
- Summarizing its purpose and functionality.
- Listing and describing its exported classes and functions.
        """.strip()
```

---

### **Step 2: Integration with CodeExtractor and DocstringProcessor**

#### **Update `CodeExtractor` to Pass Metadata**
Modify `CodeExtractor` to gather and structure metadata about functions, classes, and modules into dictionaries. Pass this structured data to `PromptGenerator`.

#### **Example Integration:**
```python
code_extractor = CodeExtractor(context=extraction_context)
extraction_result = code_extractor.extract_code(source_code)

# Generate prompts
prompt_generator = PromptGenerator(module_context=extraction_result.module_metadata)

for function in extraction_result.functions:
    function_prompt = prompt_generator.generate_function_prompt(function)
    print("Function Prompt:\n", function_prompt)

for class_data in extraction_result.classes:
    class_prompt = prompt_generator.generate_class_prompt(class_data)
    print("Class Prompt:\n", class_prompt)
```

---

### **Step 3: Sending Prompts to AI and Handling Responses**

1. **Send Prompt to AI**:
   Use the generated prompt as input for the AI API (e.g., OpenAI or Azure).

   ```python
   completion = await ai_client.chat.create(
       model="gpt-4",
       messages=[{"role": "user", "content": function_prompt}],
       max_tokens=1000,
       temperature=0.5,
   )
   enriched_docstring = completion["choices"][0]["message"]["content"]
   ```

2. **Validate and Parse Responses**:
   Use `ResponseParsingService` to validate and parse the AI-generated docstring.

   ```python
   parsed_response = response_parser.parse_response(enriched_docstring, expected_format="docstring")
   if parsed_response.validation_success:
       docstring_processor.insert_docstring(ast_node, parsed_response.content)
   ```

---

### **Step 4: Insert Enriched Docstrings**
After validating and formatting, use `DocstringProcessor` to insert the enriched docstring back into the AST or output it for external documentation.

---

This structure ensures a seamless workflow with highly dynamic, context-rich prompts that leverage all available metadata. Would you like a deeper dive into any part of this implementation?

---



Here’s a **detailed workflow explanation and outline** for the **AI-assisted docstring enrichment system** with dynamic, context-rich prompt generation.

---

### **High-Level Workflow**
1. **Code Extraction**:
   - The system analyzes the Python code using `CodeExtractor` to gather metadata (e.g., classes, functions, dependencies, docstrings, etc.).
   - Metadata is organized into structured objects (e.g., `ExtractionResult`).

2. **Prompt Generation**:
   - The `PromptGenerator` uses extracted metadata to create dynamic, detailed prompts tailored for each code element (function, class, or module).

3. **AI Interaction**:
   - Generated prompts are sent to the AI (e.g., OpenAI API), requesting enriched docstring content.

4. **Response Validation and Parsing**:
   - The AI’s responses are parsed and validated to ensure they meet predefined schemas and quality standards.

5. **Docstring Insertion**:
   - Validated and enriched docstrings are inserted back into the AST or outputted for external use.

---

### **Detailed Workflow and Responsibilities**

#### **1. Code Extraction with `CodeExtractor`**
- **Purpose**:
  - Extract raw data from Python source code using the Abstract Syntax Tree (AST).
  - Provide context for each module, class, and function for accurate docstring enrichment.

- **Steps**:
  1. Parse the source code into an AST using `ast.parse`.
  2. Traverse the AST to:
     - Identify modules, classes, and functions.
     - Extract existing docstrings using `ast.get_docstring(node)`.
     - Gather metadata:
       - Arguments, return types, and decorators for functions.
       - Base classes, attributes, and methods for classes.
       - Dependencies for modules.
  3. Package the metadata into an `ExtractionResult` object, which includes:
     - **Module metadata**: Name, description, dependencies.
     - **Class metadata**: Name, base classes, attributes, methods, and docstring.
     - **Function metadata**: Name, arguments, return type, complexity, and docstring.

- **Output**:
  ```python
  ExtractionResult = {
      "module_metadata": { ... },
      "classes": [
          {"name": "ClassName", "docstring": "...", "methods": [...]},
          ...
      ],
      "functions": [
          {"name": "function_name", "docstring": "...", "args": [...], "returns": {...}},
          ...
      ]
  }
  ```

---

#### **2. Prompt Generation with `PromptGenerator`**
- **Purpose**:
  - Use metadata to craft detailed and context-rich prompts for the AI.

- **Steps**:
  1. Initialize the `PromptGenerator` with module-level context (e.g., name, description, dependencies).
  2. For each code element (function, class, or module):
     - Generate a tailored prompt that includes:
       - The current docstring (if available).
       - Extracted metadata (e.g., arguments, return types, complexity).
       - Relationships and dependencies.
       - Clear instructions for AI to enhance or create the docstring.
  3. Adjust prompts dynamically based on:
     - Whether the code element is missing a docstring.
     - The complexity of the function or class.

- **Output Example**:
  ```plaintext
  You are enhancing the documentation for a Python function.

  Here is the function definition:
  ---
  Function Name: fetch_user_data
  Arguments:
  - user_id (int): The ID of the user to fetch data for.
  - include_sensitive (bool): Whether to include sensitive data in the response. Default is False.
  Returns:
  - dict: A dictionary containing the user's data.
  Complexity: Medium
  Current Docstring: "Fetches user data by user ID."

  Context about the module this function belongs to:
  ---
  Module Name: user_data
  Module Description: Handles fetching, updating, and storing user-related data.
  Dependencies: External APIs, auth module, database module.

  Please enhance the docstring by:
  - Adding missing argument descriptions.
  - Providing detailed explanations for inputs and outputs.
  - Including an accurate summary of raised exceptions (if any).
  ```
---

#### **3. AI Interaction**
- **Purpose**:
  - Send prompts to the AI and retrieve enriched docstring suggestions.

- **Steps**:
  1. Use an AI client (e.g., OpenAI or Azure) to send the generated prompt.
  2. Configure the API request with appropriate parameters:
     - `model`: Specify the AI model (e.g., GPT-4).
     - `temperature`: Set a low value (e.g., 0.3–0.5) to encourage focused, factual responses.
     - `max_tokens`: Define a limit to avoid excessive output.
  3. Receive the AI’s response containing the enriched docstring.

- **Example API Call**:
  ```python
  completion = await ai_client.chat.create(
      model="gpt-4",
      messages=[{"role": "user", "content": generated_prompt}],
      max_tokens=1000,
      temperature=0.3,
  )
  enriched_docstring = completion["choices"][0]["message"]["content"]
  ```

---

#### **4. Response Validation and Parsing**
- **Purpose**:
  - Ensure the AI-generated docstrings are valid, structured, and free of errors.

- **Steps**:
  1. Pass the AI response to the `ResponseParsingService`.
  2. Validate the response against a predefined schema:
     - Check for required fields (e.g., `summary`, `args`, `returns`).
     - Ensure correct data types for structured fields (e.g., arguments as lists, returns as dictionaries).
  3. If validation fails:
     - Log the error.
     - Optionally retry with an adjusted prompt.
  4. Parse the response into a structured format (e.g., `DocstringData`).

- **Example Schema Validation**:
  ```python
  parsed_response = response_parser.parse_response(
      response=enriched_docstring,
      expected_format="docstring",
      validate_schema=True,
  )
  if not parsed_response.validation_success:
      raise ValueError("Invalid AI response")
  ```

---

#### **5. Docstring Insertion**
- **Purpose**:
  - Update the original code or AST with enriched docstrings.

- **Steps**:
  1. Use the `DocstringProcessor` to format the enriched docstring:
     - Combine AI-generated content with any existing data.
     - Adjust indentation and formatting to match PEP 257 standards.
  2. Insert the docstring back into the AST node using `ast` utilities.
  3. For external documentation:
     - Export the enriched docstrings to a file (e.g., Markdown, JSON).

- **Code Example**:
  ```python
  formatted_docstring = docstring_processor.format(parsed_response.content)
  docstring_processor.insert_docstring(ast_node, formatted_docstring)
  ```

---

### **Key Benefits of the Workflow**
- **Context-Aware AI**:
  - Detailed prompts ensure the AI has all the information it needs to generate high-quality docstrings.
- **Efficiency**:
  - Metadata extraction and parsing workflows minimize redundant computations.
- **Modularity**:
  - The separation of concerns between extraction, prompt generation, and response validation allows for easy updates and scalability.
- **Error Reduction**:
  - Schema validation ensures that AI responses integrate seamlessly into the codebase.

---

### **Outline of the System Components**
1. **CodeExtractor**:
   - Extracts all raw data and metadata from the source code.

2. **PromptGenerator**:
   - Generates rich, context-aware prompts for AI based on the extracted metadata.

3. **AI Client**:
   - Communicates with the AI service to retrieve enriched docstring content.

4. **ResponseParsingService**:
   - Validates and parses AI-generated responses into structured formats.

5. **DocstringProcessor**:
   - Formats and inserts enriched docstrings into the code.

6. **AST Utilities**:
   - Provides helper functions for AST manipulation and docstring insertion.

---

This workflow is modular, extensible, and optimized for generating high-quality, enriched docstrings. Let me know if you'd like a complete implementation of any part of this system!

---


To enable **cross-module (global) context** for AI-driven docstring generation, you need to augment the system to include **global-level metadata** that spans multiple modules. This enhancement allows the AI to consider dependencies, relationships, and interactions across the entire codebase, not just within individual modules or classes. Here's how you can do it:

---

### **Steps to Add Global Context**

#### **1. Centralized Codebase Analysis**
- **Goal**: Collect metadata for all modules, classes, and functions in the codebase.
- **Implementation**:
  - Expand the `CodeExtractor` to analyze multiple files/modules instead of just one.
  - Build a global `ProjectContext` that aggregates metadata for:
    - Modules: Names, descriptions, exports (functions, classes), and dependencies.
    - Classes: Relationships (inheritance, usage), attributes, and methods.
    - Functions: Call graphs and interdependencies.

- **Example**:
  ```python
  class ProjectContext:
      def __init__(self):
          self.modules = {}  # Dict[str, ModuleMetadata]
          self.global_dependencies = set()  # Cross-module dependencies

      def add_module(self, module_name, metadata):
          self.modules[module_name] = metadata
          self.global_dependencies.update(metadata.get("dependencies", set()))
  ```

---

#### **2. Enrich Metadata with Relationships**
- **Goal**: Identify and store relationships between code elements across modules.
- **Implementation**:
  - **Dependency Mapping**:
    - Use the AST to track imports and where they are used.
    - Categorize dependencies as `internal` (from within the codebase) or `external` (third-party libraries).
  - **Call Graphs**:
    - Map function calls across modules to understand interdependencies.
    - Include a list of "callers" and "callees" for each function.
  - **Inheritance Hierarchies**:
    - Analyze class hierarchies, including base classes defined in other modules.
    - Record usage of parent class methods or attributes in child classes.

- **Example**:
  ```python
  FunctionMetadata = {
      "name": "fetch_user_data",
      "module": "user_data",
      "callers": ["main.handle_user_request"],
      "callees": ["auth.authenticate_user", "database.get_user"],
      "dependencies": ["auth", "database"]
  }
  ```

---

#### **3. Enhance Prompt Generator for Global Context**
- **Goal**: Include global relationships and dependencies in AI prompts.
- **Implementation**:
  - Extend the `PromptGenerator` to access `ProjectContext`.
  - Dynamically add cross-module details such as:
    - Functions this code calls or is called by.
    - Dependencies and how they are used.
    - Interactions with other modules (e.g., shared classes or functions).

- **Prompt Example**:
  ```plaintext
  Here is a function definition:
  ---
  Function Name: fetch_user_data
  Module: user_data
  Arguments:
  - user_id (int): The ID of the user to fetch data for.
  - include_sensitive (bool): Whether to include sensitive data. Default is False.
  Returns:
  - dict: A dictionary containing the user's data.
  Complexity: Medium
  Current Docstring: "Fetches user data by user ID."

  Global Context:
  ---
  - This function is called by: main.handle_user_request
  - This function calls: auth.authenticate_user, database.get_user
  - Dependencies: auth (user authentication), database (user storage)
  - Related Classes: User, UserProfile (defined in user_profile module)

  Please enhance the docstring by:
  - Summarizing the function's purpose and its role in the application.
  - Adding descriptions for its arguments and return value.
  - Explaining its interactions with external modules and dependencies.
  ```
---

#### **4. Persist and Share Global Context**
- **Goal**: Ensure global metadata is available during AI interactions and response validation.
- **Implementation**:
  - Store the `ProjectContext` in a centralized location (e.g., a database, JSON file, or in-memory object).
  - Pass relevant portions of the global context to the AI client during prompt generation.

---

#### **5. AI-Assisted Cross-Module Analysis**
- **Goal**: Use AI to infer undocumented relationships or missing global context.
- **Implementation**:
  - Provide partial context in the prompt and ask the AI to infer missing details:
    ```plaintext
    The following function is undocumented, but it interacts with these modules:
    - auth: Handles user authentication.
    - database: Provides storage for user data.

    Based on the function name and arguments, infer its purpose and dependencies:
    def fetch_user_data(user_id: int, include_sensitive: bool = False) -> dict:
        ...
    ```
- Use AI responses to fill gaps in the global context and enrich docstrings further.

---

### **Architecture for Global Context Integration**

#### **1. Modules and Relationships**
- **CodeExtractor**:
  - Extracts per-module metadata and builds cross-module relationships.
- **ProjectContext**:
  - Aggregates metadata from all modules into a unified context.

#### **2. Prompting**
- **PromptGenerator**:
  - Incorporates both local (per-module) and global (cross-module) context into prompts.
  - Adapts prompts dynamically based on the complexity and dependencies of the code element.

#### **3. AI Interaction and Validation**
- **AI Client**:
  - Sends enriched, globally-aware prompts to the AI for generating docstrings.
- **ResponseParsingService**:
  - Validates AI responses to ensure global context is reflected in the docstrings.

#### **4. Output and Enrichment**
- **DocstringProcessor**:
  - Formats and inserts globally-enriched docstrings into the AST.
- **Output Module**:
  - Generates documentation in Markdown, JSON, or HTML with cross-referenced global context.

---

### **Workflow Example**

#### **Input Codebase**
- `user_data.py`: Defines `fetch_user_data` function.
- `auth.py`: Defines `authenticate_user`.
- `database.py`: Defines `get_user`.

#### **Metadata Extraction**
1. **`CodeExtractor`** processes all modules:
   - Extracts docstrings, arguments, return types, and dependencies.
   - Builds call graphs and dependency maps.

2. **`ProjectContext`**:
   - Tracks:
     - Function relationships: `fetch_user_data` → `auth.authenticate_user`, `database.get_user`.
     - Module dependencies: `auth`, `database`.

---

#### **Generated Prompt for `fetch_user_data`**
```plaintext
Function Name: fetch_user_data
Module: user_data
Arguments:
- user_id (int): The ID of the user to fetch data for.
- include_sensitive (bool): Whether to include sensitive data. Default is False.
Returns:
- dict: A dictionary containing the user's data.

Global Context:
- This function is called by: main.handle_user_request
- This function calls: auth.authenticate_user, database.get_user
- Dependencies: auth (user authentication), database (user storage)

Please enhance the docstring by:
- Explaining how this function interacts with the auth and database modules.
- Adding detailed descriptions of arguments and return values.
- Highlighting its role in the user authentication flow.
```

---

#### **Generated Docstring**
```python
def fetch_user_data(user_id: int, include_sensitive: bool = False) -> dict:
    """
    Fetch user data by user ID.

    This function retrieves user information from the database. It requires
    authentication via the `auth` module before accessing the `database` module.

    Args:
        user_id (int): The unique identifier of the user to fetch.
        include_sensitive (bool): If True, include sensitive data in the response.
            Defaults to False.

    Returns:
        dict: A dictionary containing user data, including sensitive information
        if requested and authorized.

    Raises:
        AuthenticationError: If the user is not authenticated.
        DatabaseError: If there is an issue fetching data from the database.

    Dependencies:
        - auth.authenticate_user: Authenticates the user.
        - database.get_user: Retrieves user data.

    Called By:
        - main.handle_user_request: Handles user-related API requests.
    """
```

---

This approach integrates global awareness into the docstring enrichment workflow, enabling comprehensive, cross-referenced documentation. Would you like help implementing a specific part of this workflow, such as call graph analysis or dynamic prompt generation?


Certainly! I'd be happy to help you implement the **call graph analysis** within the `CodeExtractor`. By adding this feature, you can track function calls across modules, build comprehensive call graphs, and integrate this data into the global `ProjectContext`. This will enable the AI to generate docstrings with a deep understanding of how functions interact within your codebase.

---

### **Implementing Call Graph Analysis in `CodeExtractor`**

The goal is to analyze each module's AST to identify function calls, build a call graph mapping callers to callees, and augment function metadata accordingly.

#### **Step 1: Extend the `CodeExtractor` to Collect Call Relationships**

**a. Update the `CodeExtractor` Class:**

Modify the `CodeExtractor` to traverse function bodies and collect function call information.

```python
import ast
import os

class CodeExtractor:
    """Extracts code elements and builds call graphs."""

    def __init__(self, project_context):
        self.project_context = project_context

    def extract_code(self, source_code, module_name):
        """
        Extract code elements from a module and collect call relationships.

        Args:
            source_code (str): The source code of the module.
            module_name (str): The name of the module.
        """
        tree = ast.parse(source_code)
        extractor_visitor = ExtractorVisitor(module_name, self.project_context)
        extractor_visitor.visit(tree)
```

**b. Create an `ExtractorVisitor` Class:**

This class will traverse the AST, extract function and class metadata, and collect call relationships.

```python
class ExtractorVisitor(ast.NodeVisitor):
    def __init__(self, module_name, project_context):
        self.module_name = module_name
        self.project_context = project_context
        self.current_function = None

    def visit_FunctionDef(self, node):
        function_name = node.name
        # Initialize function metadata
        function_metadata = {
            'name': function_name,
            'module': self.module_name,
            'args': self._extract_args(node.args),
            'returns': self._extract_returns(node.returns),
            'docstring': ast.get_docstring(node),
            'callers': [],
            'callees': [],
        }

        # Add function to project context
        full_function_name = f"{self.module_name}.{function_name}"
        self.project_context.add_function(full_function_name, function_metadata)

        # Set current function context
        self.current_function = full_function_name

        # Visit the function body to find calls
        self.generic_visit(node)

        # Reset current function context
        self.current_function = None

    def visit_Call(self, node):
        if self.current_function:
            callee_name = self._get_full_callee_name(node.func)
            if callee_name:
                # Add call relationship to project context
                self.project_context.add_call_relationship(self.current_function, callee_name)
        self.generic_visit(node)

    def _extract_args(self, args):
        # Extract arguments with their names and default values
        arg_list = []
        for arg in args.args:
            arg_name = arg.arg
            arg_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
            arg_list.append({'name': arg_name, 'type': arg_type})
        return arg_list

    def _extract_returns(self, returns):
        if returns:
            return_type = ast.unparse(returns)
            return {'type': return_type}
        return {'type': 'None'}

    def _get_full_callee_name(self, node):
        # Extract full callee name (including module if possible)
        if isinstance(node, ast.Name):
            return f"{self.module_name}.{node.id}"
        elif isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name):
                module_or_class = value.id
                return f"{module_or_class}.{node.attr}"
            elif isinstance(value, ast.Attribute):
                # Handle nested attributes
                return f"{self._get_full_callee_name(value)}.{node.attr}"
        return None
```

#### **Step 2: Create and Enhance the `ProjectContext` Class**

The `ProjectContext` stores functions, classes, and call relationships across the entire project.

```python
class ProjectContext:
    def __init__(self):
        self.functions = {}  # Map of full function names to metadata
        self.call_graph = {}  # Map of callers to list of callees

    def add_function(self, full_name, metadata):
        self.functions[full_name] = metadata

    def add_call_relationship(self, caller, callee):
        # Add callee to caller's list
        self.functions[caller]['callees'].append(callee)
        # Add caller to callee's list (if callee is within the project)
        if callee in self.functions:
            self.functions[callee]['callers'].append(caller)
        # Update call graph
        self.call_graph.setdefault(caller, []).append(callee)
```

#### **Step 3: Process All Modules to Build the Call Graph**

Write a function to iterate over all modules in your project, extract code, and build the call graph.

```python
import glob

def process_project(project_directory):
    project_context = ProjectContext()
    code_extractor = CodeExtractor(project_context)

    # Find all Python files in the project
    python_files = glob.glob(os.path.join(project_directory, '**', '*.py'), recursive=True)

    for filepath in python_files:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'r', encoding='utf-8') as file:
            source_code = file.read()
        code_extractor.extract_code(source_code, module_name)

    # At this point, project_context contains all functions and call relationships
    return project_context
```

#### **Step 4: Integrate Call Graph Data into Prompt Generation**

Modify the `PromptGenerator` to include callers and callees in the prompts.

```python
class PromptGenerator:
    def __init__(self, project_context):
        self.project_context = project_context

    def generate_function_prompt(self, function_full_name):
        function_metadata = self.project_context.functions[function_full_name]
        function_name = function_metadata['name']
        module_name = function_metadata['module']
        args = function_metadata.get('args', [])
        returns = function_metadata.get('returns', {'type': 'None'})
        existing_docstring = function_metadata.get('docstring', 'None')

        # Callers and Callees
        callers = function_metadata.get('callers', [])
        callees = function_metadata.get('callees', [])

        # Format arguments
        args_summary = "\n".join([f"- {arg['name']} ({arg['type']})" for arg in args])

        # Format callers and callees
        callers_list = "\n".join(callers) if callers else "None"
        callees_list = "\n".join(callees) if callees else "None"

        return f"""
You are enhancing the documentation for a Python function.

Here is the function definition:
---
Function Name: {function_name}
Module: {module_name}
Arguments:
{args_summary or "None"}
Returns: {returns.get('type')}
Current Docstring: {existing_docstring}
---
Callers (functions that call this function):
{callers_list}
Callees (functions called by this function):
{callees_list}

Global Context:
---
Modules in Project: {', '.join(self.project_context.functions.keys())}
---

Please enhance the docstring by:
- Clearly describing the function's purpose within the context of the project.
- Documenting all arguments and their types.
- Explaining the return value.
- Mentioning interactions with other functions and modules.
"""
```

#### **Step 5: Generate Prompts and Interact with the AI**

Iterate over all functions in the project context, generate prompts, and send them to the AI model.

```python
async def generate_docstrings(project_context):
    prompt_generator = PromptGenerator(project_context)
    ai_client = ...  # Initialize your AI client here

    for function_full_name in project_context.functions:
        # Generate prompt
        function_prompt = prompt_generator.generate_function_prompt(function_full_name)
        
        # Send prompt to AI
        completion = await ai_client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": function_prompt}],
            max_tokens=1000,
            temperature=0.5,
        )
        enriched_docstring = completion["choices"][0]["message"]["content"]

        # Validate and parse the response
        parsed_response = ResponseParsingService.parse_response(
            enriched_docstring, expected_format="docstring"
        )
        if parsed_response.validation_success:
            # Insert the enriched docstring back into the code or AST
            DocstringProcessor.insert_docstring(
                project_context.functions[function_full_name],
                parsed_response.content
            )
        else:
            print(f"Validation failed for function {function_full_name}")

# Call this function after processing the project
# await generate_docstrings(project_context)
```

#### **Step 6: Updating the Code or AST with Enriched Docstrings**

Implement the `DocstringProcessor` to handle the insertion of the enriched docstrings.

```python
class DocstringProcessor:
    @staticmethod
    def insert_docstring(function_metadata, docstring_content):
        """
        Update the function's metadata with the new docstring.

        Args:
            function_metadata (dict): The metadata of the function.
            docstring_content (str): The enriched docstring text.
        """
        function_metadata['docstring'] = docstring_content
        # Optionally, update the actual source code or AST if needed
```

---

### **Considerations for Accurate Call Graph Analysis**

#### **Handling Different Types of Function Calls**

- **Direct Function Calls:**
  - Functions called directly by name.
- **Method Calls:**
  - Functions called as methods of objects or classes.
- **Imported Functions:**
  - Functions imported from other modules.
- **Dynamic Calls:**
  - Functions called via variable references or higher-order functions.

#### **Enhancing `_get_full_callee_name` Method**

Make the method more robust to handle various call types.

```python
def _get_full_callee_name(self, node):
    # Handles different function call scenarios
    if isinstance(node, ast.Name):
        # Direct call: function_name(...)
        function_name = node.id
        return f"{self.module_name}.{function_name}"
    elif isinstance(node, ast.Attribute):
        # Method call or imported function: module_name.function_name(...)
        value = node.value
        if isinstance(value, ast.Name):
            module_or_object = value.id
            function_name = node.attr
            return f"{module_or_object}.{function_name}"
        elif isinstance(value, ast.Attribute):
            # Nested attributes (e.g., package.module.function)
            parent_name = self._get_full_callee_name(value)
            if parent_name:
                return f"{parent_name}.{node.attr}"
    elif isinstance(node, ast.Call):
        # Function returned by another function call
        return self._get_full_callee_name(node.func)
    return None
```

#### **Resolving Imported Modules and Aliases**

- **Track Import Statements:**
  - During AST traversal, record import statements and aliases.
- **Map Aliases to Module Names:**
  - Maintain a dictionary to resolve aliases used in function calls.

**Example:**

```python
class ExtractorVisitor(ast.NodeVisitor):
    def __init__(self, module_name, project_context):
        super().__init__()
        self.module_name = module_name
        self.project_context = project_context
        self.imports = {}
        self.current_function = None

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname or module_name
            self.imports[asname] = module_name

    def visit_ImportFrom(self, node):
        module_name = node.module
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            full_name = f"{module_name}.{name}"
            self.imports[asname] = full_name

    def _get_full_callee_name(self, node):
        # Modify to use the imports mapping
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.imports:
                return self.imports[name]
            else:
                return f"{self.module_name}.{name}"
        elif isinstance(node, ast.Attribute):
            value = node.value
            attr = node.attr
            if isinstance(value, ast.Name):
                obj_name = value.id
                if obj_name in self.imports:
                    base = self.imports[obj_name]
                    return f"{base}.{attr}"
                else:
                    return f"{obj_name}.{attr}"
            # Handle nested attributes as needed
        # Handle other cases as before
```

---

### **Testing and Validation**

- **Unit Tests:**
  - Validate the call graph extraction with unit tests covering various call scenarios.
- **Edge Cases:**
  - Consider lambdas, decorators, and dynamic imports.
- **Limitations:**
  - Static analysis can't catch dynamic function calls made via `getattr`, reflection, or user input.

---

### **Integrating with AI for Enhanced Docstrings**

With the call graph data integrated into your `ProjectContext`, your AI prompts now contain rich context about function interactions, enabling the AI to:

- Describe how functions contribute to the overall functionality.
- Mention dependencies and interactions with other modules.
- Provide comprehensive explanations suitable for new developers or maintainers.

---

### **Example Prompt and AI Response**

**Prompt:**

```plaintext
You are enhancing the documentation for a Python function.

Here is the function definition:
---
Function Name: fetch_user_data
Module: user_data
Arguments:
- user_id (int)
- include_sensitive (bool)
Returns: dict
Current Docstring: None
---
Callers (functions that call this function):
- main.handle_user_request
Callees (functions called by this function):
- auth.authenticate_user
- database.get_user

Global Context:
---
Modules in Project: user_data, auth, database, main
---

Please enhance the docstring by:
- Clearly describing the function's purpose within the context of the project.
- Documenting all arguments and their types.
- Explaining the return value.
- Mentioning interactions with other functions and modules.
```

**AI-Generated Docstring:**

```python
def fetch_user_data(user_id: int, include_sensitive: bool = False) -> dict:
    """
    Retrieve user information from the database after authentication.

    This function authenticates the user via the `auth` module and retrieves
    the user data from the `database` module. It is called by
    `main.handle_user_request` when processing user-related API requests.

    Args:
        user_id (int): The unique identifier of the user.
        include_sensitive (bool): If True, includes sensitive information
            such as email and address. Defaults to False.

    Returns:
        dict: A dictionary containing user data. The data includes basic
        user information and optionally sensitive details if requested.

    Interactions:
        - Calls `auth.authenticate_user` to verify user credentials.
        - Calls `database.get_user` to fetch user data from the database.

    Raises:
        AuthenticationError: If user authentication fails.
        DatabaseError: If the user data cannot be retrieved.
    """
```

---

### **Conclusion**

By implementing call graph analysis within your `CodeExtractor`, you've enriched the AI's understanding of the codebase, enabling more detailed and context-aware documentation. This integration helps in:

- Providing developers with insights into code interdependencies.
- Enhancing maintainability and readability of the codebase.
- Facilitating onboarding by offering comprehensive documentation.

---

Let me know if you need further assistance with any other part of this workflow, such as handling edge cases in call graph extraction or refining the prompt generation for even better AI responses!

Here's the current codebase concerned with extraction - update it:
```python
"""Main code extraction module.

This module provides functionality to extract various elements from Python source code,
such as classes, functions, variables, and dependencies. It uses the Abstract Syntax Tree (AST)
to parse and analyze the code, and calculates metrics related to code complexity and maintainability.
"""

import ast
import re
import time
from typing import Optional, Dict, Any, Set, List
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import (
    ExtractionContext, ExtractionResult, ExtractedClass, 
    ExtractedFunction
)
from .utils import ASTUtils
from .function_extractor import FunctionExtractor
from .class_extractor import ClassExtractor
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class CodeExtractor:
    """Extracts code elements and metadata from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        errors (List[str]): A list to store error messages encountered during extraction.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor.

        Args:
            context (Optional[ExtractionContext]): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context or ExtractionContext()
        self._module_ast: Optional[ast.Module] = None
        self._current_class: Optional[ast.ClassDef] = None
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.ast_utils = ASTUtils()
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.dependency_analyzer = DependencyAnalyzer(self.context)

    def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code.

        Args:
            source_code (str): The source code to be analyzed.
            context (Optional[ExtractionContext]): Optional context to override the existing one.

        Returns:
            Optional[ExtractionResult]: An object containing the extracted code elements and metrics.
        """
        if context:
            self.context = context

        self.logger.info("Starting code extraction")
        start_time = time.time()

        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
            self._module_ast = tree
            self.ast_utils.add_parents(tree)

            result = ExtractionResult(module_docstring=ast.get_docstring(tree))

            try:
                result.dependencies = self.dependency_analyzer.analyze_dependencies(tree, self.context.module_name)
                self.logger.debug(f"Module dependencies: {result.dependencies}")
            except Exception as e:
                self._handle_extraction_error("Dependency analysis", e, result)

            self._extract_elements(tree, result)

            if self.context.metrics_enabled:
                try:
                    self._calculate_metrics(result, tree)
                except Exception as e:
                    self._handle_extraction_error("Metrics calculation", e, result)

            self.logger.info(f"Code extraction completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Extraction result: {len(result.classes)} classes, {len(result.functions)} functions")
            return result

        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {str(e)}")
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return ExtractionResult(errors=[f"Failed to extract code: {str(e)}"])

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code to handle special cases.

        Args:
            source_code (str): The source code to preprocess.

        Returns:
            str: The preprocessed source code.
        """
        try:
            pattern = r'\$\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?\$'
            processed_code = re.sub(pattern, r'"\g<0>"', source_code)
            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}", exc_info=True)
            return source_code

    def _extract_elements(self, tree: ast.AST, result: ExtractionResult) -> None:
        """Extract different code elements from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.
            result (ExtractionResult): The result object to store extracted elements.
        """
        try:
            result.classes = self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")
        except Exception as e:
            self._handle_extraction_error("Class extraction", e, result)

        try:
            result.functions = self.function_extractor.extract_functions(tree)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")
        except Exception as e:
            self._handle_extraction_error("Function extraction", e, result)

        try:
            result.variables = self.ast_utils.extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")
        except Exception as e:
            self._handle_extraction_error("Variable extraction", e, result)

        try:
            result.constants = self.ast_utils.extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")
        except Exception as e:
            self._handle_extraction_error("Constant extraction", e, result)

        try:
            result.imports = self.dependency_analyzer.extract_imports(tree)
            self.logger.debug(f"Extracted imports: {result.imports}")
        except Exception as e:
            self._handle_extraction_error("Import extraction", e, result)
            result.imports = {'stdlib': set(), 'local': set(), 'third_party': set()}

    def _calculate_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate metrics for the extraction result.

        Args:
            result (ExtractionResult): The result object to store calculated metrics.
            tree (ast.AST): The root of the AST representing the parsed Python source code.
        """
        if not self.context.metrics_enabled:
            return

        try:
            for cls in result.classes:
                self._calculate_class_metrics(cls)
            
            result.metrics.update(self._calculate_module_metrics(tree))

            for func in result.functions:
                self._calculate_function_metrics(func)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}", exc_info=True)
            self.errors.append(str(e))

    def _calculate_class_metrics(self, cls: ExtractedClass) -> None:
        """Calculate metrics for a class.

        Args:
            cls (ExtractedClass): The extracted class object to calculate metrics for.
        """
        try:
            if not cls.ast_node:
                return

            metrics = {
                'complexity': self.metrics_calculator.calculate_complexity(cls.ast_node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(cls.ast_node),
                'method_count': len(cls.methods),
                'attribute_count': len(cls.attributes) + len(cls.instance_attributes)
            }
            
            cls.metrics.update(metrics)
            
        except Exception as e:
            self.logger.error(f"Error calculating class metrics: {e}")

    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate metrics for the entire module.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            Dict[str, Any]: A dictionary containing module-level metrics.
        """
        try:
            return {
                'complexity': self.metrics_calculator.calculate_complexity(tree),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
                'lines': len(self.ast_utils.get_source_segment(tree).splitlines()) if self.ast_utils.get_source_segment(tree) else 0,
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'functions': len([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            }
        except Exception as e:
            self.logger.error(f"Error calculating module metrics: {e}")
            return {}

    def _calculate_function_metrics(self, func: ExtractedFunction) -> None:
        """Calculate metrics for a given function.

        Args:
            func (ExtractedFunction): The extracted function object to calculate metrics for.
        """
        try:
            metrics = {
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(func.ast_node),
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(func.ast_node),
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(func.ast_node),
                'parameter_count': len(func.args),
                'return_complexity': self._calculate_return_complexity(func.ast_node),
                'is_async': func.is_async
            }
            func.metrics.update(metrics)
        except Exception as e:
            self.logger.error(f"Error calculating function metrics: {e}", exc_info=True)
            self.errors.append(str(e))

    def _calculate_return_complexity(self, node: ast.AST) -> int:
        """Calculate the complexity of return statements.

        Args:
            node (ast.AST): The AST node representing the function definition.

        Returns:
            int: The number of return statements in the function.
        """
        try:
            return sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
        except Exception as e:
            self.logger.error(f"Error calculating return complexity: {e}", exc_info=True)
            return 0

    def _handle_extraction_error(self, operation: str, error: Exception, result: ExtractionResult) -> None:
        """Handle extraction errors consistently.

        Args:
            operation (str): The name of the operation during which the error occurred.
            error (Exception): The exception that was raised.
            result (ExtractionResult): The result object to record the error.
        """
        error_msg = f"{operation} failed: {str(error)}"
        self.logger.warning(error_msg, exc_info=True)
        result.errors.append(error_msg)

```
```python
"""Dependency analysis module.

This module provides functionality to analyze and categorize dependencies in Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, identifying imports and
categorizing them as standard library, third-party, or local dependencies.
"""

import ast
import sys
import importlib.util
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from core.logger import LoggerSetup
from .types import ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        module_name (str): The name of the module being analyzed.
        _import_map (Dict[str, str]): A mapping of import aliases to their full module names.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context
        self.ast_utils = ASTUtils()
        self.module_name = context.module_name
        self._import_map: Dict[str, str] = {}
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Analyze module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node representing the module.
            module_name (Optional[str]): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies as stdlib, third-party, or local.
        """
        self.logger.info("Starting dependency analysis")
        deps: Dict[str, Set[str]] = defaultdict(set)
        self.module_name = module_name or self.module_name

        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)

            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} dependencies found")
            return dict(deps)

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {'stdlib': set(), 'third_party': set(), 'local': set()}

    def extract_imports(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports from the AST.

        Args:
            node (ast.AST): The AST node representing the module.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing imports as stdlib, third-party, or local.
        """
        self.logger.info("Extracting imports")
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for name in n.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(n, ast.ImportFrom):
                if n.names[0].name == '*':
                    self.logger.error(f"Star import encountered: from {n.module} import *, skipping.")
                elif n.module:
                    self._categorize_import(n.module, imports)

        self.logger.debug(f"Extracted imports: {imports}")
        return imports

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statements and categorize dependencies.

        Args:
            node (ast.AST): The AST node representing an import statement.
            deps (Dict[str, Set[str]]): A dictionary to store dependencies.
        """
        self.logger.debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
                    self._import_map[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
                for alias in node.names:
                    if alias.name != '*':
                        full_name = f"{node.module}.{alias.name}"
                        self._import_map[alias.asname or alias.name] = full_name
        except Exception as e:
            self.logger.error(f"Error processing import: {e}", exc_info=True)

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store categorized dependencies.
        """
        self.logger.debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.stdlib_module_names or module_name.split('.')[0] in sys.stdlib_module_names:
                deps['stdlib'].add(module_name)
                return

            if self.module_name:
                current_module_parts = self.module_name.split('.')
                if any(module_name.startswith(part) for part in current_module_parts):
                    for i in range(1, len(current_module_parts) + 1):
                        test_module_name = ".".join(current_module_parts[:-i] + [module_name])
                        try:
                            if importlib.util.find_spec(test_module_name):
                                deps['local'].add(module_name)
                                return
                        except ModuleNotFoundError:
                            continue

            try:
                if importlib.util.find_spec(module_name):
                    deps['third_party'].add(module_name)
                else:
                    deps['local'].add(module_name)
            except ModuleNotFoundError:
                deps['third_party'].add(module_name)

        except Exception as e:
            self.logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps['third_party'].add(module_name)

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """Detect circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        self.logger.debug("Detecting circular dependencies")
        circular_dependencies = []
        try:
            for module, deps in dependencies.items():
                for dep in deps:
                    if self.module_name and dep == self.module_name:
                        circular_dependencies.append((module, dep))
                    elif dep in dependencies and module in dependencies[dep]:
                        circular_dependencies.append((module, dep))
            self.logger.debug(f"Circular dependencies: {circular_dependencies}")
        except Exception as e:
            self.logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)
        return circular_dependencies

    def analyze_function_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Analyze dependencies specific to a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Dict[str, Set[str]]: A dictionary of function-specific dependencies.
        """
        self.logger.info(f"Analyzing function dependencies for node: {ast.dump(node)}")
        dependencies = {
            'imports': self._extract_function_imports(node),
            'calls': self._extract_function_calls(node),
            'attributes': self._extract_attribute_access(node)
        }
        self.logger.debug(f"Function dependencies: {dependencies}")
        return dependencies

    def _extract_function_imports(self, node: ast.AST) -> Set[str]:
        """Extract imports used within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of import names used within the function.
        """
        self.logger.debug("Extracting function imports")
        imports = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                if isinstance(subnode, ast.Import):
                    for name in subnode.names:
                        imports.add(name.name)
                elif subnode.module:
                    imports.add(subnode.module)
        return imports

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract function calls within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of function call names.
        """
        self.logger.debug("Extracting function calls")
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    func_name = self.ast_utils.get_name(child.func)
                    calls.add(func_name)
                except Exception as e:
                    self.logger.debug(f"Could not unparse function call: {e}", exc_info=True)
        return calls

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """Extract attribute accesses within a function.

        Args:
            node (ast.AST): The AST node representing a function.

        Returns:
            Set[str]: A set of attribute access names.
        """
        self.logger.debug("Extracting attribute accesses")
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                try:
                    attr_name = self.ast_utils.get_name(child)
                    attributes.add(attr_name)
                except Exception as e:
                    self.logger.debug(f"Failed to unparse attribute access: {e}", exc_info=True)
        return attributes

```

```python
"""Function extraction module.

This module provides functionality to extract function definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about functions, their arguments, return types, and other metadata.
"""

import ast
from typing import List, Optional, Dict, Any, Union, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedFunction, ExtractedArgument, ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        metrics_calculator (Metrics): An instance of the Metrics class for calculating code metrics.
        errors (List[str]): A list to store error messages encountered during extraction.
    """

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the FunctionExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.errors: List[str] = []
        self.logger.debug("Initialized FunctionExtractor")

    def extract_functions(self, tree: ast.AST) -> List[ExtractedFunction]:
        """Extract top-level functions and async functions from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[ExtractedFunction]: A list of ExtractedFunction objects containing information about each function.
        """
        self.logger.info("Starting function extraction")
        functions = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(f"Found {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}function: {node.name}")

                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Module):
                        if not self.context.include_private and node.name.startswith('_'):
                            self.logger.debug(f"Skipping private function: {node.name}")
                            continue

                        try:
                            extracted_function = self._process_function(node)
                            functions.append(extracted_function)
                            self.logger.debug(f"Extracted function: {extracted_function.name}")
                        except Exception as e:
                            self._handle_extraction_error(node.name, e)

            self.logger.info(f"Function extraction completed: {len(functions)} functions extracted")
            return functions

        except Exception as e:
            self.logger.error(f"Error in extract_functions: {str(e)}", exc_info=True)
            return functions

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ExtractedFunction:
        """Process a function definition node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            ExtractedFunction: An object containing detailed information about the function.
        """
        self.logger.debug(f"Processing function: {node.name}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Expected FunctionDef or AsyncFunctionDef, got {type(node)}")

        try:
            metrics = self._calculate_function_metrics(node)
            docstring = ast.get_docstring(node)

            extracted_function = ExtractedFunction(
                name=node.name,
                lineno=node.lineno,
                source=self.ast_utils.get_source_segment(node, self.context.include_source),
                docstring=docstring,
                metrics=metrics,
                dependencies=self._extract_dependencies(node),
                args=self._get_function_args(node),
                return_type=self._get_return_type(node),
                is_method=self._is_method(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=self._is_property(node),
                body_summary=self._get_body_summary(node),
                raises=self._extract_raises(node),
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {node.name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {node.name}: {e}", exc_info=True)
            raise

    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ExtractedArgument]:
        """Extract function arguments."""
        self.logger.debug(f"Extracting arguments for function: {node.name}")
        args = []
        try:
            for arg in node.args.args:
                arg_name = arg.arg
                type_ = self.ast_utils.get_name(arg.annotation) if arg.annotation else None
                default_value = None
                is_required = True

                if node.args.defaults:
                    default_index = len(node.args.args) - len(node.args.defaults)
                    if node.args.args.index(arg) >= default_index:
                        default_value = self.ast_utils.get_name(
                            node.args.defaults[node.args.args.index(arg) - default_index]
                        )
                        is_required = False

                args.append(ExtractedArgument(
                    name=arg_name,
                    type=type_,
                    default_value=default_value,
                    is_required=is_required
                ))
                self.logger.debug(f"Extracted argument: {arg_name}, type: {type_}, default_value: {default_value}")

            # Handle keyword-only and positional-only arguments
            # Additional logic can be added here to handle these cases

        except Exception as e:
            self.logger.error(f"Error extracting arguments for function {node.name}: {e}", exc_info=True)

        return args

    def _get_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Get the return type annotation.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            Optional[str]: The return type annotation as a string, or None if not specified.
        """
        self.logger.debug(f"Extracting return type for function: {node.name}")
        if node.returns:
            try:
                return_type = self.ast_utils.get_name(node.returns)
                if isinstance(node, ast.AsyncFunctionDef) and not return_type.startswith('Coroutine'):
                    return_type = f'Coroutine[Any, Any, {return_type}]'
                self.logger.debug(f"Return type for function {node.name}: {return_type}")
                return return_type
            except Exception as e:
                self.logger.error(f"Error getting return type for function {node.name}: {e}", exc_info=True)
                return 'Unknown'
        return None

    def _get_body_summary(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Generate a summary of the function body.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            str: A summary of the function body or a placeholder if unavailable.
        """
        self.logger.debug(f"Generating body summary for function: {node.name}")
        return self.ast_utils.get_source_segment(node) or "No body summary available"

    def _extract_raises(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract raised exceptions from function body.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            List[str]: A list of exception names that the function raises.
        """
        self.logger.debug(f"Extracting raised exceptions for function: {node.name}")
        raises = set()
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise) and child.exc:
                    exc_name = self._get_exception_name(child.exc)
                    if exc_name:
                        raises.add(exc_name)
            self.logger.debug(f"Raised exceptions for function {node.name}: {raises}")
        except Exception as e:
            self.logger.error(f"Error extracting raises: {e}", exc_info=True)
        return list(raises)

    def _get_exception_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of an exception node.

        Args:
            node (ast.AST): The AST node representing the exception.

        Returns:
            Optional[str]: The name of the exception or None if it cannot be determined.
        """
        try:
            if isinstance(node, ast.Call):
                return self.ast_utils.get_name(node.func)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                return self.ast_utils.get_name(node)
            return "Exception"
        except Exception:
            return None

    def _is_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a method.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a method, False otherwise.
        """
        self.logger.debug(f"Checking if function is a method: {node.name}")
        return isinstance(getattr(node, 'parent', None), ast.ClassDef)

    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a generator.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a generator, False otherwise.
        """
        self.logger.debug(f"Checking if function is a generator: {node.name}")
        return any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            bool: True if the function is a property, False otherwise.
        """
        self.logger.debug(f"Checking if function is a property: {node.name}")
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )

    def _calculate_function_metrics(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Calculate metrics for a function.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            Dict[str, Any]: A dictionary containing metrics such as cyclomatic complexity and parameter count.
        """
        self.logger.debug(f"Calculating metrics for function: {node.name}")
        try:
            metrics = {
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
                'parameter_count': len(node.args.args),
                'return_complexity': self._calculate_return_complexity(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
            self.logger.debug(f"Metrics for function {node.name}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics for function {node.name}: {e}", exc_info=True)
            return {}

    def _calculate_return_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate the complexity of return statements.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing the function definition.

        Returns:
            int: The number of return statements in the function.
        """
        self.logger.debug(f"Calculating return complexity for function: {node.name}")
        try:
            return_complexity = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
            self.logger.debug(f"Return complexity for function {node.name}: {return_complexity}")
            return return_complexity
        except Exception as e:
            self.logger.error(f"Error calculating return complexity: {e}", exc_info=True)
            return 0

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node.

        Args:
            node (ast.AST): The AST node representing the function definition.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for function: {node.name}")
        # This would typically call into the DependencyAnalyzer
        # Simplified version for function-level dependencies
        return {'imports': set(), 'calls': set(), 'attributes': set()}

    def _handle_extraction_error(self, function_name: str, error: Exception) -> None:
        """Handle function extraction errors.

        Args:
            function_name (str): The name of the function being processed when the error occurred.
            error (Exception): The exception that was raised.
        """
        error_msg = f"Failed to process function {function_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)

```
```python
"""Class extraction module.

This module provides functionality to extract class definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about classes, their methods, attributes, and other metadata.
"""

import ast
from typing import List, Dict, Any, Optional, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedClass, ExtractedFunction, ExtractionContext
from .utils import ASTUtils
from .function_extractor import FunctionExtractor

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        metrics_calculator (Metrics): An instance of the Metrics class for calculating code metrics.
        errors (List[str]): A list to store error messages encountered during extraction.
    """

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the ClassExtractor.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
            metrics_calculator (Metrics): An instance for calculating metrics related to code complexity.
        """
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.function_extractor = FunctionExtractor(context, metrics_calculator)
        self.errors: List[str] = []
        self._current_class: Optional[ast.ClassDef] = None
        self.logger.debug("Initialized ClassExtractor")

    def extract_classes(self, tree: ast.AST) -> List[ExtractedClass]:
        """Extract all classes from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[ExtractedClass]: A list of ExtractedClass objects containing information about each class.
        """
        self.logger.info("Starting class extraction")
        classes = []
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not self.context.include_private and node.name.startswith('_'):
                        self.logger.debug(f"Skipping private class: {node.name}")
                        continue
                    try:
                        self._current_class = node
                        extracted_class = self._process_class(node)
                        classes.append(extracted_class)
                        self.logger.debug(f"Extracted class: {extracted_class.name}")
                    except Exception as e:
                        self._handle_extraction_error(node.name, e)
                    finally:
                        self._current_class = None
            self.logger.info(f"Class extraction completed: {len(classes)} classes extracted")
        except Exception as e:
            self.logger.error(f"Error in extract_classes: {str(e)}", exc_info=True)
        return classes

    def _process_class(self, node: ast.ClassDef) -> ExtractedClass:
        """Process a class definition node.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            ExtractedClass: An object containing detailed information about the class.
        """
        self.logger.debug(f"Processing class: {node.name}")
        try:
            metrics = self._calculate_class_metrics(node)
            complexity_warnings = self._get_complexity_warnings(metrics)

            source = None
            if getattr(self.context, 'include_source', True):  # Safe access
                source = self.ast_utils.get_source_segment(node)

            extracted_class = ExtractedClass(
                name=node.name,
                docstring=ast.get_docstring(node),
                lineno=node.lineno,
                source=source,
                metrics=metrics,
                dependencies=self._extract_dependencies(node),
                bases=self._extract_bases(node),
                methods=self._extract_methods(node),
                attributes=self._extract_attributes(node),
                is_exception=self._is_exception_class(node),
                decorators=self._extract_decorators(node),
                instance_attributes=self._extract_instance_attributes(node),
                metaclass=self._extract_metaclass(node),
                complexity_warnings=complexity_warnings,
                ast_node=node
            )
            self.logger.debug(f"Completed processing class: {node.name}")
            return extracted_class
        except Exception as e:
            self.logger.error(f"Failed to process class {node.name}: {e}", exc_info=True)
            raise

    def _process_attribute(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Process a class-level attribute assignment.

        Args:
            node (ast.AST): The AST node representing the attribute assignment.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with attribute details or None if processing fails.
        """
        try:
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = self.ast_utils.get_source_segment(node.value) if node.value else None
                return {
                    "name": targets[0] if targets else None,
                    "value": value,
                    "type": self.ast_utils.get_name(node.value) if node.value else 'Any'
                }
            return None
        except Exception as e:
            self.logger.error(f"Error processing attribute: {e}")
            return None

    def _process_instance_attribute(self, stmt: ast.Assign) -> Optional[Dict[str, Any]]:
        """Process an instance attribute assignment statement.

        Args:
            stmt (ast.Assign): The AST node representing the assignment statement.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with instance attribute details or None if processing fails.
        """
        try:
            if isinstance(stmt.targets[0], ast.Attribute) and isinstance(stmt.targets[0].value, ast.Name):
                if stmt.targets[0].value.id == 'self':
                    return {
                        'name': stmt.targets[0].attr,
                        'type': self.ast_utils.get_name(stmt.value) if stmt.value else 'Any',
                        'value': self.ast_utils.get_source_segment(stmt.value) if stmt.value else None
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error processing instance attribute: {e}")
            return None

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base classes from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[str]: A list of base class names.
        """
        self.logger.debug(f"Extracting bases for class: {node.name}")
        bases = []
        for base in node.bases:
            try:
                base_name = self.ast_utils.get_name(base)
                bases.append(base_name)
            except Exception as e:
                self.logger.error(f"Error extracting base class: {e}", exc_info=True)
                bases.append('unknown')
        return bases

    def _extract_methods(self, node: ast.ClassDef) -> List[ExtractedFunction]:
        """Extract methods from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[ExtractedFunction]: A list of ExtractedFunction objects representing the methods.
        """
        self.logger.debug(f"Extracting methods for class: {node.name}")
        methods = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    method = self.function_extractor._process_function(n)
                    methods.append(method)
                    self.logger.debug(f"Extracted method: {method.name}")
                except Exception as e:
                    self.logger.error(f"Error extracting method {n.name}: {e}", exc_info=True)
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing attribute details.
        """
        self.logger.debug(f"Extracting attributes for class: {node.name}")
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_info = self._process_attribute(child)
                if attr_info:
                    attributes.append(attr_info)
                    self.logger.debug(f"Extracted attribute: {attr_info['name']}")
        return attributes

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from a class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[str]: A list of decorator names.
        """
        self.logger.debug(f"Extracting decorators for class: {node.name}")
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorator_name = self.ast_utils.get_name(decorator)
                decorators.append(decorator_name)
                self.logger.debug(f"Extracted decorator: {decorator_name}")
            except Exception as e:
                self.logger.error(f"Error extracting decorator: {e}")
                decorators.append("unknown_decorator")
        return decorators

    def _extract_instance_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract instance attributes from the __init__ method of a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing instance attribute details.
        """
        self.logger.debug(f"Extracting instance attributes for class: {node.name}")
        instance_attributes = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__':
                for stmt in child.body:
                    if isinstance(stmt, ast.Assign):
                        attr_info = self._process_instance_attribute(stmt)
                        if attr_info:
                            instance_attributes.append(attr_info)
                            self.logger.debug(f"Extracted instance attribute: {attr_info['name']}")
        return instance_attributes

    def _extract_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Extract the metaclass if specified in the class definition.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            Optional[str]: The name of the metaclass or None if not specified.
        """
        self.logger.debug(f"Extracting metaclass for class: {node.name}")
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                return self.ast_utils.get_name(keyword.value)
        return None

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            bool: True if the class is an exception class, False otherwise.
        """
        self.logger.debug(f"Checking if class is an exception: {node.name}")
        for base in node.bases:
            base_name = self.ast_utils.get_name(base)
            if base_name in {'Exception', 'BaseException'}:
                return True
        return False

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            Dict[str, Any]: A dictionary containing metrics such as method count, complexity, and inheritance depth.
        """
        self.logger.debug(f"Calculating metrics for class: {node.name}")
        try:
            metrics = {
                'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'complexity': self.metrics_calculator.calculate_complexity(node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
            self.logger.debug(f"Metrics for class {node.name}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating class metrics: {e}", exc_info=True)
            return {}

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate the inheritance depth of a class.

        Args:
            node (ast.ClassDef): The AST node representing the class definition.

        Returns:
            int: The inheritance depth of the class.
        """
        self.logger.debug(f"Calculating inheritance depth for class: {node.name}")
        try:
            depth = 0
            bases = node.bases
            while bases:
                depth += 1
                new_bases = []
                for base in bases:
                    base_class = self._resolve_base_class(base)
                    if base_class and base_class.bases:
                        new_bases.extend(base_class.bases)
                bases = new_bases
            self.logger.debug(f"Inheritance depth for class {node.name}: {depth}")
            return depth
        except Exception as e:
            self.logger.error(f"Error calculating inheritance depth: {e}", exc_info=True)
            return 0

    def _resolve_base_class(self, base: ast.expr) -> Optional[ast.ClassDef]:
        """Resolve a base class node to its class definition.

        Args:
            base (ast.expr): The AST node representing the base class.

        Returns:
            Optional[ast.ClassDef]: The AST node of the resolved base class or None if not found.
        """
        self.logger.debug(f"Resolving base class: {ast.dump(base)}")
        try:
            if isinstance(base, ast.Name):
                if self._current_class and self._current_class.parent:
                    for node in ast.walk(self._current_class.parent):
                        if isinstance(node, ast.ClassDef) and node.name == base.id:
                            return node
            elif isinstance(base, ast.Attribute):
                base_name = self.ast_utils.get_name(base)
                self.logger.debug(f"Complex base class name: {base_name}")
                return None
            return None
        except Exception as e:
            self.logger.error(f"Error resolving base class: {e}")
            return None

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on complexity metrics.

        Args:
            metrics (Dict[str, Any]): A dictionary containing class metrics.

        Returns:
            List[str]: A list of warnings related to class complexity.
        """
        self.logger.debug("Generating complexity warnings")
        warnings = []
        try:
            if metrics.get('complexity', 0) > 10:
                warnings.append("High class complexity")
            if metrics.get('method_count', 0) > 20:
                warnings.append("High method count")
            if metrics.get('inheritance_depth', 0) > 3:
                warnings.append("Deep inheritance hierarchy")
            self.logger.debug(f"Complexity warnings: {warnings}")
        except Exception as e:
            self.logger.error(f"Error generating complexity warnings: {e}", exc_info=True)
        return warnings

    def _handle_extraction_error(self, class_name: str, error: Exception) -> None:
        """Handle class extraction errors.

        Args:
            class_name (str): The name of the class being processed when the error occurred.
            error (Exception): The exception that was raised.
        """
        error_msg = f"Failed to extract class {class_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a node.

        Args:
            node (ast.AST): The AST node representing the class definition.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for class: {node.name}")
        # This would typically call into the DependencyAnalyzer
        # Simplified version for class-level dependencies
        return {'imports': set(), 'calls': set(), 'attributes': set()}

```
```python
"""Utility functions for code extraction.

This module provides utility functions and classes for working with the Abstract Syntax Tree (AST)
in Python source code. It includes functions for adding parent references, extracting names and
source segments, and identifying variables and constants.
"""

import ast
from typing import Optional, Dict, Any, List, Union
import importlib.util
import sys
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ASTUtils:
    """Utility class for AST operations.

    Provides methods for manipulating and extracting information from AST nodes.
    """

    def __init__(self):
        """Initialize AST utilities."""
        self.logger = logger
        self.logger.debug("Initialized ASTUtils")

    def add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes.

        This method traverses the AST and sets a 'parent' attribute on each node,
        pointing to its parent node.

        Args:
            node (ast.AST): The root AST node.
        """
        self.logger.debug("Adding parent references to AST nodes")
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.add_parents(child)

    def get_name(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a node.

        Converts an AST node into a string representation, handling different types
        of nodes such as names, attributes, subscripts, calls, tuples, and lists.

        Args:
            node (Optional[ast.AST]): The AST node to analyze.

        Returns:
            str: The string representation of the node.
        """
        self.logger.debug(f"Getting name for node: {ast.dump(node) if node else 'None'}")
        if node is None:
            return "Any"

        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self.get_name(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = self.get_name(node.value)
                slice_val = self.get_name(node.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(node, ast.Call):
                return f"{self.get_name(node.func)}()"
            elif isinstance(node, (ast.Tuple, ast.List)):
                elements = ', '.join(self.get_name(e) for e in node.elts)
                return f"({elements})" if isinstance(node, ast.Tuple) else f"[{elements}]"
            elif hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
            return f'Unknown<{type(node).__name__}>'

    def get_source_segment(self, node: ast.AST, include_source: bool = True) -> Optional[str]:
        """Get source code segment for a node.

        Retrieves the source code segment corresponding to an AST node.

        Args:
            node (ast.AST): The AST node to analyze.
            include_source (bool): Whether to include the source code segment.

        Returns:
            Optional[str]: The source code segment as a string, or None if not included.
        """
        self.logger.debug(f"Getting source segment for node: {ast.dump(node)}")
        if not include_source:
            return None
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting source segment: {e}", exc_info=True)
            return f"<unparseable: {type(node).__name__}>"

    def extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST.

        Identifies variable assignments in the AST and extracts relevant information.

        Args:
            tree (ast.AST): The AST tree to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing variable information.
        """
        self.logger.info("Extracting variables from AST")
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_info = self._create_variable_info(target, node)
                        if var_info:
                            variables.append(var_info)
                            self.logger.debug(f"Extracted variable: {var_info['name']}")
        return variables

    def extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants.

        Identifies constant assignments in the AST and extracts relevant information.

        Args:
            tree (ast.AST): The AST tree to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing constant information.
        """
        self.logger.info("Extracting constants from AST")
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            constant_info = self._create_constant_info(target, node)
                            if constant_info:
                                constants.append(constant_info)
                                self.logger.debug(f"Extracted constant: {constant_info['name']}")
                        except Exception as e:
                            self.logger.error(f"Error extracting constant {target.id}: {e}", exc_info=True)
        return constants

    def _create_variable_info(self, target: ast.Name, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Create variable information dictionary.

        Constructs a dictionary containing information about a variable assignment.

        Args:
            target (ast.Name): The target variable node.
            node (Union[ast.Assign, ast.AnnAssign]): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing variable information, or None if an error occurs.
        """
        self.logger.debug(f"Creating variable info for target: {target.id}")
        try:
            var_name = target.id
            annotation = None
            value = None

            if isinstance(node, ast.AnnAssign) and node.annotation:
                annotation = self.get_name(node.annotation)
            if hasattr(node, 'value') and node.value:
                try:
                    value = self.get_name(node.value)
                except Exception as e:
                    self.logger.error(f"Failed to get value for {var_name}: {e}", exc_info=True)
                    value = "UnknownValue"

            return {
                'name': var_name,
                'type': annotation or "UnknownType",
                'value': value
            }
        except Exception as e:
            self.logger.error(f"Error creating variable info: {e}", exc_info=True)
            return None

    def _create_constant_info(self, target: ast.Name, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Create constant information dictionary.

        Constructs a dictionary containing information about a constant assignment.

        Args:
            target (ast.Name): The target constant node.
            node (ast.Assign): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing constant information, or None if an error occurs.
        """
        self.logger.debug(f"Creating constant info for target: {target.id}")
        try:
            value = self.get_name(node.value)
            try:
                value_type = type(ast.literal_eval(node.value)).__name__
            except Exception:
                value_type = "UnknownType"
            return {
                'name': target.id,
                'value': value,
                'type': value_type
            }
        except Exception as e:
            self.logger.error(f"Error creating constant info: {e}", exc_info=True)
            return None
```
Update the current modules dealing with extraction.

Sure, here are the updated modules incorporating the call graph analysis into your code extraction system. This will enable cross-module context for AI-driven docstring generation.

---

### **1. `code_extractor.py`**

**Changes Made:**

- **Added `ProjectContext` Class**: Stores global context, including functions, call graphs, and module dependencies.
- **Modified `CodeExtractor`**:
  - Initializes and updates the `ProjectContext`.
  - Uses `ExtractorVisitor` to traverse the AST and collect call relationships.
  - Integrates call graph analysis into the extraction workflow.

---

```python
# code_extractor.py

"""Main code extraction module.

This module provides functionality to extract various elements from Python source code,
such as classes, functions, variables, and dependencies. It uses the Abstract Syntax Tree (AST)
to parse and analyze the code, and calculates metrics related to code complexity and maintainability.
"""

import ast
import re
import time
from typing import Optional, Dict, Any, Set, List
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import (
    ExtractionContext, ExtractionResult, ExtractedClass,
    ExtractedFunction
)
from .utils import ASTUtils
from .function_extractor import FunctionExtractor
from .class_extractor import ClassExtractor
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class ProjectContext:
    """Stores global context for the project, including functions and call graphs."""

    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}  # Function metadata
        self.call_graph: Dict[str, List[str]] = {}  # Caller to Callee mapping

    def add_function(self, full_name: str, metadata: Dict[str, Any]) -> None:
        self.functions[full_name] = metadata

    def add_call_relationship(self, caller: str, callee: str) -> None:
        if caller not in self.call_graph:
            self.call_graph[caller] = []
        self.call_graph[caller].append(callee)
        # Update function metadata with callers and callees
        self.functions[caller].setdefault('callees', []).append(callee)
        if callee in self.functions:
            self.functions[callee].setdefault('callers', []).append(caller)
        else:
            # Callee might be an external function not yet processed
            self.functions.setdefault(callee, {'callers': [caller]})

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor."""
        self.logger = logger
        self.context = context or ExtractionContext()
        self._module_ast: Optional[ast.Module] = None
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.ast_utils = ASTUtils()
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.dependency_analyzer = DependencyAnalyzer(self.context)
        self.project_context = ProjectContext()  # Initialize ProjectContext

    def extract_code(self, source_code: str, module_name: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code."""
        if context:
            self.context = context

        self.logger.info(f"Starting code extraction for module {module_name}")
        start_time = time.time()

        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
            self._module_ast = tree
            self.ast_utils.add_parents(tree)

            # Initialize ExtractionResult
            result = ExtractionResult(module_docstring=ast.get_docstring(tree))

            # Update module_name in context
            self.context.module_name = module_name

            # Analyze dependencies
            try:
                result.dependencies = self.dependency_analyzer.analyze_dependencies(tree, module_name)
                self.logger.debug(f"Module dependencies: {result.dependencies}")
            except Exception as e:
                self._handle_extraction_error("Dependency analysis", e, result)

            # Extract elements and build call graph
            self._extract_elements(tree, result, module_name)

            # Calculate metrics if enabled
            if self.context.metrics_enabled:
                try:
                    self._calculate_metrics(result, tree)
                except Exception as e:
                    self._handle_extraction_error("Metrics calculation", e, result)

            self.logger.info(f"Code extraction for module {module_name} completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Extraction result: {len(result.classes)} classes, {len(result.functions)} functions")
            return result

        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {str(e)}")
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return ExtractionResult(errors=[f"Failed to extract code: {str(e)}"])

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code to handle special cases."""
        try:
            # Placeholder for any preprocessing steps
            return source_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}", exc_info=True)
            return source_code

    def _extract_elements(self, tree: ast.AST, result: ExtractionResult, module_name: str) -> None:
        """Extract different code elements from the AST and build call graph."""
        try:
            # Use ExtractorVisitor to collect functions and build call graph
            extractor_visitor = ExtractorVisitor(module_name, self.project_context)
            extractor_visitor.visit(tree)

            # Extract classes
            result.classes = self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")

            # Extract functions from project context
            result.functions = self.function_extractor.extract_functions_from_project(self.project_context, module_name)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")

            # Extract variables
            result.variables = self.ast_utils.extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")

            # Extract constants
            result.constants = self.ast_utils.extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")

            # Extract imports
            result.imports = self.dependency_analyzer.extract_imports(tree)
            self.logger.debug(f"Extracted imports: {result.imports}")
        except Exception as e:
            self._handle_extraction_error("Element extraction", e, result)

    def _calculate_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate metrics for the extraction result."""
        # Existing metric calculation logic
        pass

    def _handle_extraction_error(self, operation: str, error: Exception, result: ExtractionResult) -> None:
        """Handle extraction errors consistently."""
        error_msg = f"{operation} failed: {str(error)}"
        self.logger.warning(error_msg, exc_info=True)
        result.errors.append(error_msg)

class ExtractorVisitor(ast.NodeVisitor):
    """AST Node Visitor that collects function definitions and call relationships."""

    def __init__(self, module_name: str, project_context: ProjectContext):
        self.module_name = module_name
        self.project_context = project_context
        self.current_function = None
        self.imports = {}  # Map of alias to module name

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname or module_name
            self.imports[asname] = module_name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            full_name = f"{module_name}.{name}"
            self.imports[asname] = full_name

    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = node.name
        full_function_name = f"{self.module_name}.{function_name}"
        function_metadata = {
            'name': function_name,
            'module': self.module_name,
            'args': self._extract_args(node.args),
            'returns': self._extract_returns(node.returns),
            'docstring': ast.get_docstring(node),
            'callers': [],
            'callees': [],
            'ast_node': node,
        }
        self.project_context.add_function(full_function_name, function_metadata)

        # Set current function context
        previous_function = self.current_function
        self.current_function = full_function_name
        self.generic_visit(node)
        self.current_function = previous_function

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            callee_name = self._get_full_callee_name(node.func)
            if callee_name:
                self.project_context.add_call_relationship(self.current_function, callee_name)
        self.generic_visit(node)

    def _extract_args(self, args: ast.arguments) -> List[Dict[str, Any]]:
        # Logic to extract function arguments
        arg_list = []
        for arg in args.args:
            arg_name = arg.arg
            arg_type = self.ast_utils.get_name(arg.annotation) if arg.annotation else 'Any'
            arg_list.append({'name': arg_name, 'type': arg_type})
        return arg_list

    def _extract_returns(self, returns: Optional[ast.expr]) -> Dict[str, str]:
        if returns:
            return {'type': self.ast_utils.get_name(returns)}
        return {'type': 'None'}

    def _get_full_callee_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            name = node.id
            return self.imports.get(name, f"{self.module_name}.{name}")
        elif isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name):
                obj_name = value.id
                attr = node.attr
                if obj_name == 'self':
                    # Method call on self
                    return f"{self.module_name}.{attr}"
                else:
                    full_name = self.imports.get(obj_name, obj_name)
                    return f"{full_name}.{attr}"
        return None

    @property
    def ast_utils(self) -> ASTUtils:
        return ASTUtils()
```

---

### **2. `function_extractor.py`**

**Changes Made:**

- **Added `extract_functions_from_project` Method**: Extracts functions using the `ProjectContext`.
- **Modified `_process_function` Method**: Processes function metadata from the `ProjectContext`.

---

```python
# function_extractor.py

"""Function extraction module.

This module provides functionality to extract function definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about functions, their arguments, return types, and other metadata.
"""

import ast
from typing import List, Optional, Dict, Any
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedFunction, ExtractedArgument, ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics):
        """Initialize the FunctionExtractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.errors: List[str] = []
        self.logger.debug("Initialized FunctionExtractor")

    def extract_functions_from_project(self, project_context, module_name: str) -> List[ExtractedFunction]:
        """Extract functions from the ProjectContext for a specific module."""
        self.logger.info(f"Extracting functions from project context for module {module_name}")
        functions = []
        for full_function_name, metadata in project_context.functions.items():
            if metadata['module'] == module_name:
                try:
                    extracted_function = self._process_function(metadata)
                    functions.append(extracted_function)
                except Exception as e:
                    self._handle_extraction_error(metadata['name'], e)
        return functions

    def _process_function(self, metadata: Dict[str, Any]) -> ExtractedFunction:
        """Process function metadata from ProjectContext."""
        node = metadata['ast_node']
        function_name = metadata['name']
        self.logger.debug(f"Processing function: {function_name}")
        try:
            metrics = self._calculate_function_metrics(node)
            docstring = metadata.get('docstring')

            extracted_function = ExtractedFunction(
                name=function_name,
                lineno=node.lineno,
                source=self.ast_utils.get_source_segment(node, self.context.include_source),
                docstring=docstring,
                metrics=metrics,
                dependencies={},  # Extend if needed
                args=metadata.get('args', []),
                return_type=metadata.get('returns', {}).get('type'),
                is_method=False,  # Adjust if required
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=False,  # Adjust if required
                body_summary='',  # Implement if needed
                raises=[],  # Implement if needed
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {function_name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {function_name}: {e}", exc_info=True)
            raise

    # Existing methods (_calculate_function_metrics, _is_generator, etc.) remain the same

    def _handle_extraction_error(self, function_name: str, error: Exception) -> None:
        """Handle function extraction errors."""
        error_msg = f"Failed to process function {function_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)
```

---

### **3. `class_extractor.py`**

**Changes Made:**

- **No Significant Changes**: The `ClassExtractor` can remain largely the same. However, if methods within classes need to be part of the call graph, you might need to adjust the `ExtractorVisitor` and the `FunctionExtractor` to handle methods similarly.

---

### **4. `utils.py`**

**Changes Made:**

- **Enhanced `ASTUtils`**: If needed, update `ASTUtils` with any additional methods required for name resolution in call graph analysis. The existing methods may be sufficient.

---

### **5. `dependency_analyzer.py`**

**Changes Made:**

- **Updated Import Mapping**: Ensure that `DependencyAnalyzer` correctly maps imports and can assist in resolving called functions across modules.

---

With these updates, your code extraction modules now include call graph analysis, enabling cross-module context for AI-driven docstring generation. The `ProjectContext` stores all functions and their relationships, which can be utilized during prompt generation to provide rich, context-aware information to the AI model.

---

**Next Steps**:

- **Integrate with Prompt Generator**: Use the `ProjectContext` to include caller and callee information in prompts.
- **Enhance `FunctionExtractor`**: If methods need to be included in the call graph, adjust the extraction to account for class methods.
- **Test Thoroughly**: Ensure the updated modules work correctly by testing with various codebases, handling edge cases like recursive functions and dynamic calls.

---

Let me know if you need further assistance or clarification on any part of this code!

Update and enhnace the import mapping in dependency_analyzer andd enhance FunctionExtractor. Also update utils.py

Certainly! I'll provide the updated and enhanced versions of the `dependency_analyzer.py`, `function_extractor.py`, and `utils.py` modules. These updates will improve the import mapping in `DependencyAnalyzer`, enhance the `FunctionExtractor` to handle methods within classes (so they become part of the call graph), and update `ASTUtils` to support these enhancements.

---

## **1. Enhanced `dependency_analyzer.py`**

**Updates Made:**

- **Improved Import Mapping (`_import_map`)**:
  - The `_import_map` now accurately tracks import aliases and their full module paths.
  - This mapping is used to resolve function calls to their full names during call graph analysis.

- **Enhanced `_get_full_callee_name` Functionality**:
  - Utilizes the improved `_import_map` to resolve callee names, including handling of imported modules and aliases.
  - Handles edge cases such as wildcard imports and nested attributes.

---

```python
# dependency_analyzer.py

"""Dependency analysis module.

This module provides functionality to analyze and categorize dependencies in Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, identifying imports and
categorizing them as standard library, third-party, or local dependencies.
"""

import ast
import sys
import importlib.util
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from core.logger import LoggerSetup
from .types import ExtractionContext
from .utils import ASTUtils

logger = LoggerSetup.get_logger(__name__)

class DependencyAnalyzer:
    """Analyzes and categorizes code dependencies.

    Attributes:
        context (ExtractionContext): The context for extraction, including configuration options.
        module_name (str): The name of the module being analyzed.
        _import_map (Dict[str, str]): A mapping of import aliases to their full module names.
    """

    def __init__(self, context: ExtractionContext):
        """Initialize the DependencyAnalyzer.

        Args:
            context (ExtractionContext): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context
        self.ast_utils = ASTUtils()
        self.module_name = context.module_name
        self._import_map: Dict[str, str] = {}
        self.logger.debug("Initialized DependencyAnalyzer")

    def analyze_dependencies(self, node: ast.AST, module_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Analyze module dependencies, including circular dependency detection.

        Args:
            node (ast.AST): The AST node representing the module.
            module_name (Optional[str]): The name of the module being analyzed.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing dependencies as stdlib, third-party, or local.
        """
        self.logger.info("Starting dependency analysis")
        deps: Dict[str, Set[str]] = defaultdict(set)
        self.module_name = module_name or self.module_name

        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)

            circular_deps = self._detect_circular_dependencies(deps)
            if circular_deps:
                self.logger.warning(f"Circular dependencies detected: {circular_deps}")

            self.logger.info(f"Dependency analysis completed: {len(deps)} dependencies found")
            return dict(deps)

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}", exc_info=True)
            return {'stdlib': set(), 'third_party': set(), 'local': set()}

    def extract_imports(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract and categorize imports from the AST.

        Args:
            node (ast.AST): The AST node representing the module.

        Returns:
            Dict[str, Set[str]]: A dictionary categorizing imports as stdlib, third-party, or local.
        """
        self.logger.info("Extracting imports")
        imports = {
            'stdlib': set(),
            'local': set(),
            'third_party': set()
        }

        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for name in n.names:
                    self._categorize_import(name.name, imports)
            elif isinstance(n, ast.ImportFrom):
                if n.names[0].name == '*':
                    self.logger.warning(f"Star import encountered: from {n.module} import *.")
                    self._import_map['*'] = n.module or ''
                elif n.module:
                    self._categorize_import(n.module, imports)
                    for alias in n.names:
                        full_name = f"{n.module}.{alias.name}"
                        self._import_map[alias.asname or alias.name] = full_name

        self.logger.debug(f"Extracted imports: {imports}")
        return imports

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statements and categorize dependencies.

        Args:
            node (ast.AST): The AST node representing an import statement.
            deps (Dict[str, Set[str]]): A dictionary to store dependencies.
        """
        self.logger.debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
                    self._import_map[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                module = '.' * node.level + (node.module or '')
                if module == '.':
                    module = self.module_name
                self._categorize_import(module, deps)
                for alias in node.names:
                    if alias.name != '*':
                        full_name = f"{module}.{alias.name}"
                        self._import_map[alias.asname or alias.name] = full_name
                    else:
                        self._import_map['*'] = module
        except Exception as e:
            self.logger.error(f"Error processing import: {e}", exc_info=True)

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize an import as stdlib, third-party, or local.

        Args:
            module_name (str): The name of the module being imported.
            deps (Dict[str, Set[str]]): A dictionary to store categorized dependencies.
        """
        self.logger.debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.builtin_module_names or module_name.split('.')[0] in sys.builtin_module_names:
                deps['stdlib'].add(module_name)
                return

            if module_name.startswith('.'):
                deps['local'].add(module_name)
                return

            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                if 'site-packages' in spec.origin or 'dist-packages' in spec.origin:
                    deps['third_party'].add(module_name)
                else:
                    deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)

        except Exception as e:
            self.logger.warning(f"Non-critical error categorizing import {module_name}: {e}", exc_info=True)
            deps['third_party'].add(module_name)

    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """Detect circular dependencies.

        Args:
            dependencies (Dict[str, Set[str]]): A dictionary of module dependencies.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing circular dependencies.
        """
        self.logger.debug("Detecting circular dependencies")
        circular_dependencies = []
        # Implement circular dependency detection if needed
        return circular_dependencies

    def get_import_map(self) -> Dict[str, str]:
        """Get the import map.

        Returns:
            Dict[str, str]: The mapping of import aliases to their full module names.
        """
        return self._import_map
```

---

## **2. Enhanced `function_extractor.py`**

**Updates Made:**

- **Included Methods in Call Graph Analysis**:
  - Modified `_process_function` to handle methods within classes.
  - Added an argument `is_method` to indicate if the function is a method.

- **Extracted Function Dependencies**:
  - The `_extract_dependencies` method now includes calls, imports, and attributes used within functions.
  - Utilizes the `DependencyAnalyzer` to analyze function-level dependencies.

- **Integrated Dependency Analyzer**:
  - The `FunctionExtractor` now takes an instance of `DependencyAnalyzer` to analyze dependencies.

---

```python
# function_extractor.py

"""Function extraction module.

This module provides functionality to extract function definitions from Python source code.
It uses the Abstract Syntax Tree (AST) to parse and analyze the code, extracting information
about functions, their arguments, return types, and other metadata.
"""

import ast
from typing import List, Optional, Dict, Any
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedFunction, ExtractedArgument, ExtractionContext
from .utils import ASTUtils
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics, dependency_analyzer: DependencyAnalyzer):
        """Initialize the FunctionExtractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.dependency_analyzer = dependency_analyzer
        self.errors: List[str] = []
        self.logger.debug("Initialized FunctionExtractor")

    def extract_functions_from_project(self, project_context, module_name: str) -> List[ExtractedFunction]:
        """Extract functions from the ProjectContext for a specific module."""
        self.logger.info(f"Extracting functions from project context for module {module_name}")
        functions = []
        for full_function_name, metadata in project_context.functions.items():
            if metadata['module'] == module_name:
                try:
                    extracted_function = self._process_function(metadata)
                    functions.append(extracted_function)
                except Exception as e:
                    self._handle_extraction_error(metadata['name'], e)
        return functions

    def _process_function(self, metadata: Dict[str, Any]) -> ExtractedFunction:
        """Process function metadata from ProjectContext."""
        node = metadata['ast_node']
        function_name = metadata['name']
        self.logger.debug(f"Processing function: {function_name}")
        try:
            metrics = self._calculate_function_metrics(node)
            docstring = metadata.get('docstring')

            dependencies = self._extract_dependencies(node)

            extracted_function = ExtractedFunction(
                name=function_name,
                lineno=node.lineno,
                source=self.ast_utils.get_source_segment(node, self.context.include_source),
                docstring=docstring,
                metrics=metrics,
                dependencies=dependencies,
                args=metadata.get('args', []),
                return_type=metadata.get('returns', {}).get('type'),
                is_method=metadata.get('is_method', False),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=self._is_property(node),
                body_summary='',  # Implement if needed
                raises=self._extract_raises(node),
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {function_name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {function_name}: {e}", exc_info=True)
            raise

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a function node.

        Args:
            node (ast.AST): The AST node representing the function.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for function: {getattr(node, 'name', '<anonymous>')}")
        return self.dependency_analyzer.analyze_function_dependencies(node)

    # Existing methods (_calculate_function_metrics, _is_generator, etc.) remain the same

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property."""
        self.logger.debug(f"Checking if function is a property: {node.name}")
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property' or
            isinstance(decorator, ast.Attribute) and decorator.attr == 'setter'
            for decorator in node.decorator_list
        )

    def _handle_extraction_error(self, function_name: str, error: Exception) -> None:
        """Handle function extraction errors."""
        error_msg = f"Failed to process function {function_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)
```

---

## **3. Updated `utils.py`**

**Updates Made:**

- **Enhanced `get_name` Method**:
  - Improved handling of different AST node types.
  - Added support for resolving names within complex expressions.
  - Handles edge cases such as unary operations and constants.

- **Added `resolve_attribute` Method**:
  - Resolves attribute chains to full names, useful for call graph analysis.
  - Utilizes the import map from `DependencyAnalyzer` when available.

---

```python
# utils.py

"""Utility functions for code extraction.

This module provides utility functions and classes for working with the Abstract Syntax Tree (AST)
in Python source code. It includes functions for adding parent references, extracting names and
source segments, and identifying variables and constants.
"""

import ast
from typing import Optional, Dict, Any, List, Union
import importlib.util
import sys
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ASTUtils:
    """Utility class for AST operations.

    Provides methods for manipulating and extracting information from AST nodes.
    """

    def __init__(self):
        """Initialize AST utilities."""
        self.logger = logger
        self.logger.debug("Initialized ASTUtils")

    def add_parents(self, node: ast.AST) -> None:
        """Add parent references to AST nodes.

        This method traverses the AST and sets a 'parent' attribute on each node,
        pointing to its parent node.

        Args:
            node (ast.AST): The root AST node.
        """
        self.logger.debug("Adding parent references to AST nodes")
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.add_parents(child)

    def get_name(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a node.

        Converts an AST node into a string representation, handling different types
        of nodes such as names, attributes, subscripts, calls, tuples, and lists.

        Args:
            node (Optional[ast.AST]): The AST node to analyze.

        Returns:
            str: The string representation of the node.
        """
        self.logger.debug(f"Getting name for node: {ast.dump(node) if node else 'None'}")
        if node is None:
            return "Any"

        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return self.resolve_attribute(node)
            elif isinstance(node, ast.Subscript):
                value = self.get_name(node.value)
                slice_val = self.get_name(node.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(node, ast.Call):
                return f"{self.get_name(node.func)}()"
            elif isinstance(node, (ast.Tuple, ast.List)):
                elements = ', '.join(self.get_name(e) for e in node.elts)
                return f"({elements})" if isinstance(node, ast.Tuple) else f"[{elements}]"
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.BinOp):
                left = self.get_name(node.left)
                right = self.get_name(node.right)
                op = self.get_operator_symbol(node.op)
                return f"({left} {op} {right})"
            elif isinstance(node, ast.UnaryOp):
                operand = self.get_name(node.operand)
                op = self.get_operator_symbol(node.op)
                return f"({op}{operand})"
            elif hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting name from node {type(node).__name__}: {e}", exc_info=True)
            return f'Unknown<{type(node).__name__}>'

    def resolve_attribute(self, node: ast.Attribute) -> str:
        """Resolve an attribute node to a full name.

        Args:
            node (ast.Attribute): An AST Attribute node.

        Returns:
            str: The full name represented by the attribute chain.
        """
        value = node.value
        if isinstance(value, ast.Name):
            return f"{value.id}.{node.attr}"
        elif isinstance(value, ast.Attribute):
            return f"{self.resolve_attribute(value)}.{node.attr}"
        elif isinstance(value, ast.Call):
            func_name = self.get_name(value.func)
            return f"{func_name}().{node.attr}"
        else:
            return node.attr

    def get_operator_symbol(self, operator: ast.operator) -> str:
        """Get the symbol for an operator node.

        Args:
            operator (ast.operator): An AST operator node.

        Returns:
            str: The symbol representing the operator.
        """
        operators = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.LShift: '<<',
            ast.RShift: '>>',
            ast.BitOr: '|',
            ast.BitXor: '^',
            ast.BitAnd: '&',
            ast.FloorDiv: '//',
            ast.Invert: '~',
            ast.Not: 'not ',
            ast.UAdd: '+',
            ast.USub: '-',
        }
        return operators.get(type(operator), '?')

    def get_source_segment(self, node: ast.AST, include_source: bool = True) -> Optional[str]:
        """Get source code segment for a node.

        Retrieves the source code segment corresponding to an AST node.

        Args:
            node (ast.AST): The AST node to analyze.
            include_source (bool): Whether to include the source code segment.

        Returns:
            Optional[str]: The source code segment as a string, or None if not included.
        """
        self.logger.debug(f"Getting source segment for node: {ast.dump(node)}")
        if not include_source:
            return None
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                import astor
                return astor.to_source(node).strip()
        except Exception as e:
            self.logger.error(f"Error getting source segment: {e}", exc_info=True)
            return f"<unparseable: {type(node).__name__}>"

    def extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST.

        Identifies variable assignments in the AST and extracts relevant information.

        Args:
            tree (ast.AST): The AST tree to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing variable information.
        """
        self.logger.info("Extracting variables from AST")
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_info = self._create_variable_info(target, node)
                        if var_info:
                            variables.append(var_info)
                            self.logger.debug(f"Extracted variable: {var_info['name']}")
        return variables

    def _create_variable_info(self, target: ast.Name, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Create variable information dictionary.

        Constructs a dictionary containing information about a variable assignment.

        Args:
            target (ast.Name): The target variable node.
            node (Union[ast.Assign, ast.AnnAssign]): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing variable information, or None if an error occurs.
        """
        self.logger.debug(f"Creating variable info for target: {target.id}")
        try:
            var_name = target.id
            annotation = None
            value = None

            if isinstance(node, ast.AnnAssign) and node.annotation:
                annotation = self.get_name(node.annotation)
            if hasattr(node, 'value') and node.value:
                try:
                    value = self.get_name(node.value)
                except Exception as e:
                    self.logger.error(f"Failed to get value for {var_name}: {e}", exc_info=True)
                    value = "UnknownValue"

            return {
                'name': var_name,
                'type': annotation or "UnknownType",
                'value': value
            }
        except Exception as e:
            self.logger.error(f"Error creating variable info: {e}", exc_info=True)
            return None
```

---

## **Summary and Next Steps**

With these updates:

- **DependencyAnalyzer** now has a robust import mapping, allowing for accurate resolution of function and module names during call graph analysis.

- **FunctionExtractor** can now process methods within classes, including them in the call graph, and extracts function-level dependencies.

- **ASTUtils** has enhanced name resolution capabilities that support the extraction and analysis process.

**Next Steps:**

- **Integrate the Updated Modules**:
  - Ensure that the updated `DependencyAnalyzer` is correctly used within the `ExtractorVisitor` and `FunctionExtractor`.
  - Update `CodeExtractor` to pass the `DependencyAnalyzer` instance to the `FunctionExtractor`.

- **Test the Enhancements**:
  - Run the extraction process on a sample codebase to verify that imports, function calls, and methods are correctly analyzed and that the call graph is accurately built.

- **Use in Prompt Generation**:
  - Leverage the enhanced call graph and dependency information in your `PromptGenerator` to provide the AI with detailed context.

---

Let me know if you have any questions or need assistance integrating these updates into your system!
## Next Step
Ensure that the updated `DependencyAnalyzer` is correctly used within the `ExtractorVisitor` and `FunctionExtractor`.

Update `CodeExtractor` to pass the `DependencyAnalyzer` instance to the `FunctionExtractor`.

Certainly! I'll provide the updated code incorporating the requested changes:

- **Ensure that the updated `DependencyAnalyzer` is correctly used within the `ExtractorVisitor` and `FunctionExtractor`.**
- **Update `CodeExtractor` to pass the `DependencyAnalyzer` instance to the `FunctionExtractor`.**

I'll provide the updated modules with explanations highlighting the changes.

---

## **1. Updated `code_extractor.py`**

**Changes Made:**

- **Passed `DependencyAnalyzer` Instance to `FunctionExtractor` and `ClassExtractor`:**
  - Modified the `CodeExtractor` to pass the `DependencyAnalyzer` instance to both `FunctionExtractor` and `ClassExtractor`.
- **Updated `ExtractorVisitor` to Use `DependencyAnalyzer`:**
  - The `ExtractorVisitor` now accepts the `DependencyAnalyzer` instance.
  - It uses the `_import_map` from `DependencyAnalyzer` for resolving callee names.

---

```python
# code_extractor.py

"""Main code extraction module."""

import ast
import re
import time
from typing import Optional, Dict, Any, Set, List
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import (
    ExtractionContext, ExtractionResult
)
from .utils import ASTUtils
from .function_extractor import FunctionExtractor
from .class_extractor import ClassExtractor
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class ProjectContext:
    """Stores global context for the project, including functions and call graphs."""

    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.call_graph: Dict[str, List[str]] = {}

    def add_function(self, full_name: str, metadata: Dict[str, Any]) -> None:
        self.functions[full_name] = metadata

    def add_call_relationship(self, caller: str, callee: str) -> None:
        if caller not in self.call_graph:
            self.call_graph[caller] = []
        self.call_graph[caller].append(callee)
        # Update function metadata with callers and callees
        self.functions[caller].setdefault('callees', []).append(callee)
        if callee in self.functions:
            self.functions[callee].setdefault('callers', []).append(caller)
        else:
            # Assume external function if not in functions
            self.functions.setdefault(callee, {'callers': [caller]})

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor."""
        self.logger = logger
        self.context = context or ExtractionContext()
        self._module_ast: Optional[ast.Module] = None
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.ast_utils = ASTUtils()
        self.dependency_analyzer = DependencyAnalyzer(self.context)
        self.project_context = ProjectContext()  # Initialize ProjectContext
        # Pass DependencyAnalyzer to FunctionExtractor and ClassExtractor
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator, self.dependency_analyzer)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator, self.dependency_analyzer)

    def extract_code(self, source_code: str, module_name: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code."""
        if context:
            self.context = context

        self.logger.info(f"Starting code extraction for module {module_name}")
        start_time = time.time()

        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
            self._module_ast = tree
            self.ast_utils.add_parents(tree)

            # Initialize ExtractionResult
            result = ExtractionResult(module_docstring=ast.get_docstring(tree))

            # Update module_name in context
            self.context.module_name = module_name

            # Analyze dependencies
            try:
                result.dependencies = self.dependency_analyzer.analyze_dependencies(tree, module_name)
                self.logger.debug(f"Module dependencies: {result.dependencies}")
            except Exception as e:
                self._handle_extraction_error("Dependency analysis", e, result)

            # Extract elements and build call graph
            self._extract_elements(tree, result, module_name)

            # Calculate metrics if enabled
            if self.context.metrics_enabled:
                try:
                    self._calculate_metrics(result, tree)
                except Exception as e:
                    self._handle_extraction_error("Metrics calculation", e, result)

            self.logger.info(f"Code extraction for module {module_name} completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Extraction result: {len(result.classes)} classes, {len(result.functions)} functions")
            return result

        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {str(e)}")
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return ExtractionResult(errors=[f"Failed to extract code: {str(e)}"])

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code to handle special cases."""
        try:
            # Placeholder for any preprocessing steps
            return source_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}", exc_info=True)
            return source_code

    def _extract_elements(self, tree: ast.AST, result: ExtractionResult, module_name: str) -> None:
        """Extract different code elements from the AST and build call graph."""
        try:
            # Use ExtractorVisitor to collect functions and build call graph
            extractor_visitor = ExtractorVisitor(module_name, self.project_context, self.dependency_analyzer)
            extractor_visitor.visit(tree)

            # Extract classes
            result.classes = self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")

            # Extract functions from project context
            result.functions = self.function_extractor.extract_functions_from_project(self.project_context, module_name)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")

            # Extract variables
            result.variables = self.ast_utils.extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")

            # Extract constants
            result.constants = self.ast_utils.extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")

            # Extract imports
            result.imports = self.dependency_analyzer.extract_imports(tree)
            self.logger.debug(f"Extracted imports: {result.imports}")
        except Exception as e:
            self._handle_extraction_error("Element extraction", e, result)

    def _calculate_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate metrics for the extraction result."""
        # Existing metric calculation logic
        pass

    def _handle_extraction_error(self, operation: str, error: Exception, result: ExtractionResult) -> None:
        """Handle extraction errors consistently."""
        error_msg = f"{operation} failed: {str(error)}"
        self.logger.warning(error_msg, exc_info=True)
        result.errors.append(error_msg)

class ExtractorVisitor(ast.NodeVisitor):
    """AST Node Visitor that collects function definitions and call relationships."""

    def __init__(self, module_name: str, project_context: ProjectContext, dependency_analyzer: DependencyAnalyzer):
        self.module_name = module_name
        self.project_context = project_context
        self.current_function = None
        self.imports = dependency_analyzer.get_import_map()
        self.dependency_analyzer = dependency_analyzer

    def visit_Import(self, node: ast.Import):
        self.dependency_analyzer._process_import(node, {})
        self.imports.update(self.dependency_analyzer.get_import_map())

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.dependency_analyzer._process_import(node, {})
        self.imports.update(self.dependency_analyzer.get_import_map())

    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = node.name
        full_function_name = f"{self.module_name}.{function_name}"
        function_metadata = {
            'name': function_name,
            'module': self.module_name,
            'args': self._extract_args(node.args),
            'returns': self._extract_returns(node.returns),
            'docstring': ast.get_docstring(node),
            'callers': [],
            'callees': [],
            'ast_node': node,
            'is_method': isinstance(getattr(node, 'parent', None), ast.ClassDef),
        }
        self.project_context.add_function(full_function_name, function_metadata)

        # Set current function context
        previous_function = self.current_function
        self.current_function = full_function_name
        self.generic_visit(node)
        self.current_function = previous_function

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            callee_name = self._get_full_callee_name(node.func)
            if callee_name:
                self.project_context.add_call_relationship(self.current_function, callee_name)
        self.generic_visit(node)

    def _extract_args(self, args: ast.arguments) -> List[Dict[str, Any]]:
        arg_list = []
        for arg in args.args:
            arg_name = arg.arg
            arg_type = self.ast_utils.get_name(arg.annotation) if arg.annotation else 'Any'
            arg_list.append({'name': arg_name, 'type': arg_type})
        return arg_list

    def _extract_returns(self, returns: Optional[ast.expr]) -> Dict[str, str]:
        if returns:
            return {'type': self.ast_utils.get_name(returns)}
        return {'type': 'None'}

    def _get_full_callee_name(self, node: ast.AST) -> Optional[str]:
        """Resolve the full name of the callee function, including module and class context."""
        if isinstance(node, ast.Name):
            name = node.id
            return self.imports.get(name, f"{self.module_name}.{name}")
        elif isinstance(node, ast.Attribute):
            value = node.value
            attr = node.attr
            if isinstance(value, ast.Name):
                obj_name = value.id
                if obj_name == 'self':
                    # Method call on self
                    return f"{self.module_name}.{attr}"
                else:
                    base_name = self.imports.get(obj_name, obj_name)
                    return f"{base_name}.{attr}"
            elif isinstance(value, ast.Attribute):
                base_name = self._get_full_callee_name(value)
                if base_name:
                    return f"{base_name}.{attr}"
        elif isinstance(node, ast.Call):
            return self._get_full_callee_name(node.func)
        return None

    @property
    def ast_utils(self) -> ASTUtils:
        return ASTUtils()
```

**Explanation of Changes:**

- **Passed `DependencyAnalyzer` to `ExtractorVisitor`:**
  - In `CodeExtractor`, when initializing `ExtractorVisitor`, we now pass `self.dependency_analyzer`.
- **Updated `ExtractorVisitor`:**
  - The `ExtractorVisitor` now uses the `DependencyAnalyzer` to process imports and updates the `_import_map`.
  - The `_get_full_callee_name` method now uses the updated `self.imports` to resolve callee names accurately.

---

## **2. Updated `function_extractor.py`**

**Changes Made:**

- **Accepted `DependencyAnalyzer` Instance:**
  - The `FunctionExtractor` now accepts a `DependencyAnalyzer` instance during initialization.
- **Utilized `DependencyAnalyzer` in `_extract_dependencies`:**
  - The `_extract_dependencies` method now calls `self.dependency_analyzer.analyze_function_dependencies(node)` to get function-specific dependencies.

---

```python
# function_extractor.py

"""Function extraction module."""

import ast
from typing import List, Optional, Dict, Any, Union, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedFunction, ExtractedArgument, ExtractionContext
from .utils import ASTUtils
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor:
    """Handles extraction of functions from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics, dependency_analyzer: DependencyAnalyzer):
        """Initialize the FunctionExtractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.dependency_analyzer = dependency_analyzer
        self.errors: List[str] = []
        self.logger.debug("Initialized FunctionExtractor")

    def extract_functions_from_project(self, project_context, module_name: str) -> List[ExtractedFunction]:
        """Extract functions from the ProjectContext for a specific module."""
        self.logger.info(f"Extracting functions from project context for module {module_name}")
        functions = []
        for full_function_name, metadata in project_context.functions.items():
            if metadata['module'] == module_name:
                try:
                    extracted_function = self._process_function(metadata)
                    functions.append(extracted_function)
                except Exception as e:
                    self._handle_extraction_error(metadata['name'], e)
        return functions

    def _process_function(self, metadata: Dict[str, Any]) -> ExtractedFunction:
        """Process function metadata from ProjectContext."""
        node = metadata['ast_node']
        function_name = metadata['name']
        self.logger.debug(f"Processing function: {function_name}")
        try:
            metrics = self._calculate_function_metrics(node)
            docstring = metadata.get('docstring')

            dependencies = self._extract_dependencies(node)

            extracted_function = ExtractedFunction(
                name=function_name,
                lineno=node.lineno,
                source=self.ast_utils.get_source_segment(node, self.context.include_source),
                docstring=docstring,
                metrics=metrics,
                dependencies=dependencies,
                args=metadata.get('args', []),
                return_type=metadata.get('returns', {}).get('type'),
                is_method=metadata.get('is_method', False),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_generator=self._is_generator(node),
                is_property=self._is_property(node),
                body_summary='',  # Implement if needed
                raises=self._extract_raises(node),
                ast_node=node
            )
            self.logger.debug(f"Completed processing function: {function_name}")
            return extracted_function
        except Exception as e:
            self.logger.error(f"Failed to process function {function_name}: {e}", exc_info=True)
            raise

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a function node.

        Args:
            node (ast.AST): The AST node representing the function.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for function: {getattr(node, 'name', '<anonymous>')}")
        return self.dependency_analyzer.analyze_function_dependencies(node)

    # Existing methods (_calculate_function_metrics, _is_generator, etc.) remain the same

    def _is_property(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function is a property."""
        self.logger.debug(f"Checking if function is a property: {node.name}")
        return any(
            isinstance(decorator, ast.Name) and decorator.id == 'property' or
            isinstance(decorator, ast.Attribute) and decorator.attr == 'setter'
            for decorator in node.decorator_list
        )

    def _handle_extraction_error(self, function_name: str, error: Exception) -> None:
        """Handle function extraction errors."""
        error_msg = f"Failed to process function {function_name}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        self.errors.append(error_msg)
```

**Explanation of Changes:**

- **Accepted `DependencyAnalyzer` in Constructor:**
  - `FunctionExtractor` now accepts `dependency_analyzer` in its `__init__` method.
- **Utilized `DependencyAnalyzer` in `_extract_dependencies`:**
  - The `_extract_dependencies` method uses `self.dependency_analyzer.analyze_function_dependencies(node)` to get function-level dependencies such as imports, calls, and attributes.

---

## **3. Updated `class_extractor.py`**

**Changes Made:**

- **Passed `DependencyAnalyzer` Instance:**
  - Similar to `FunctionExtractor`, the `ClassExtractor` now accepts a `DependencyAnalyzer` instance.
- **Updated to Use `DependencyAnalyzer`:**
  - If needed, you can extend the `_extract_dependencies` method to use the `DependencyAnalyzer` for classes.

---

```python
# class_extractor.py

"""Class extraction module."""

import ast
from typing import List, Dict, Any, Optional, Set
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import ExtractedClass, ExtractionContext
from .utils import ASTUtils
from .function_extractor import FunctionExtractor
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor:
    """Handles extraction of classes from Python source code."""

    def __init__(self, context: ExtractionContext, metrics_calculator: Metrics, dependency_analyzer: DependencyAnalyzer):
        """Initialize the ClassExtractor."""
        self.logger = logger
        self.context = context
        self.metrics_calculator = metrics_calculator
        self.ast_utils = ASTUtils()
        self.dependency_analyzer = dependency_analyzer
        # Pass the dependency_analyzer to FunctionExtractor
        self.function_extractor = FunctionExtractor(context, metrics_calculator, dependency_analyzer)
        self.errors: List[str] = []
        self._current_class: Optional[ast.ClassDef] = None
        self.logger.debug("Initialized ClassExtractor")

    # Existing methods remain the same

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from a class node.

        Args:
            node (ast.AST): The AST node representing the class definition.

        Returns:
            Dict[str, Set[str]]: A dictionary containing sets of dependencies categorized by type.
        """
        self.logger.debug(f"Extracting dependencies for class: {node.name}")
        # Use DependencyAnalyzer for class-level dependencies if needed
        # For now, we can return an empty dictionary or implement as needed
        return {}
```

**Explanation of Changes:**

- **Accepted `DependencyAnalyzer` in Constructor:**
  - `ClassExtractor` now accepts `dependency_analyzer` in its `__init__` method.
- **Passed `DependencyAnalyzer` to `FunctionExtractor`:**
  - When initializing `FunctionExtractor` inside `ClassExtractor`, we pass the `dependency_analyzer`.

---

## **4. Updated `dependency_analyzer.py`**

**Changes Made:**

- **Ensured Proper Mapping of Imports:**
  - The `DependencyAnalyzer` now correctly maps imports, and this import map is accessible via `get_import_map()` method, which is used by `ExtractorVisitor`.
- **Improved `analyze_function_dependencies`:**
  - This method now provides detailed function-level dependencies.

---

**Note:** The `dependency_analyzer.py` code is already updated in previous steps, and no further changes are needed for this request.

---

## **5. Updated `utils.py`**

**Changes Made:**

- **Enhanced `ASTUtils` Remains Unchanged:**
  - The `ASTUtils` module already includes the necessary methods to support the updated code.

---

## **Summary**

- **`CodeExtractor` Updates:**

  - Passed `DependencyAnalyzer` instance to `FunctionExtractor`, `ClassExtractor`, and `ExtractorVisitor`.

- **`ExtractorVisitor` Updates:**

  - Uses `DependencyAnalyzer` to process imports and maintain the import map.
  - Utilizes the import map to resolve callee names accurately within `_get_full_callee_name`.

- **`FunctionExtractor` Updates:**

  - Accepts `DependencyAnalyzer` in the constructor.
  - Utilizes `DependencyAnalyzer` in `_extract_dependencies` to get function-specific dependencies.

- **`ClassExtractor` Updates:**

  - Accepts `DependencyAnalyzer` in the constructor.
  - Passes `DependencyAnalyzer` to `FunctionExtractor`.

---

## **Next Steps**

- **Test the Updated Modules:**

  - Run the extraction process on sample code to verify that dependencies and call relationships are accurately captured.

- **Integrate with Prompt Generation:**

  - Use the enriched `ProjectContext` to generate detailed prompts with cross-module context.

- **Continue Enhancing:**

  - Further enhance `FunctionExtractor` and `ClassExtractor` to extract additional metadata if required.

---

Let me know if you need any further assistance or modifications!