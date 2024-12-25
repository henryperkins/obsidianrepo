---
epoch: 1731464980323
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: import ast  
import astor  
from logger import log_info, log_error  
from metrics import Metrics

class DocStringManager:  
    """  
    Handles the generation, updating, and insertion of docstrings into Python code using AST.  
    """

    def __init__(self, source_code):
        """
        Initialize the DocStringManager with source code and parse it into an AST.

        :param source_code: The Python source code to manage.
        :raises SyntaxError: If the source code cannot be parsed into an AST.
        """
        self.source_code = source_code
        try:
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed from source code.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code into AST: {e}")
            raise e

    def generate_prompt(self, function_node):
        """
        Create a prompt for generating a docstring using OpenAI based on function metadata.

        :param function_node: The AST node of the function to generate a prompt for.
        :return: A string prompt for generating a docstring.
        """
        func_name = function_node.name
        args = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") for arg in function_node.args.args]
        return_annotation = ast.unparse(function_node.returns) if function_node.returns else "Any"
        complexity_score = Metrics.calculate_cyclomatic_complexity(function_node)

        params_formatted = ", ".join([f"{name}: {ptype}" for name, ptype in args])

        prompt = f"""
        Generate a JSON object with the following fields:
        {{
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Google-style docstring including a Complexity section.",
            "complexity_score": {complexity_score}
        }}

        Function: {func_name}
        Parameters: {params_formatted}
        Returns: {return_annotation}
        Existing docstring: None
        """
        return prompt.strip()

    def insert_docstring(self, function_node, docstring):
        """
        Insert or replace the docstring in the function's AST node.

        :param function_node: The AST node of the function to insert the docstring into.
        :param docstring: The docstring to insert.
        """
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Node is not a function definition, cannot insert docstring.")
            return

        if ast.get_docstring(function_node):
            function_node.body[0].value = ast.Constant(value=docstring)
        else:
            function_node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
        log_info(f"Inserted docstring into function '{function_node.name}'.")

    def update_source_code(self):
        """
        Update the source code with the modified AST containing new docstrings.

        :return: The updated source code as a string, or None if an error occurs.
        """
        try:
            updated_code = astor.to_source(self.tree)
            log_info("Source code updated with new docstrings.")
            return updated_code
        except Exception as e:
            log_error(f"Failed to generate source code from AST: {e}")
            return None

    def generate_markdown_documentation(self, entries):
        """
        Generate a Markdown documentation from the collected entries.

        :param entries: A list of dictionaries containing function metadata and docstrings.
        :return: A string containing the generated Markdown documentation.
        """
        documentation = "# Summary\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['summary']}\n"
        documentation += "\n# Changelog\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['changelog']}\n"
        documentation += "\n# Function/Method/Class Glossary\n\n"
        documentation += "| Function Name | Complexity Score | Docstring |\n"
        documentation += "|---------------|-----------------|-----------|\n"
        for entry in entries:
            docstring = entry["docstring"].replace("\n", " ")
            documentation += f"| {entry['function_name']} | {entry['complexity_score']} | {docstring} |\n"
        return documentation

# Suggested Test Cases

def test_docstring_manager():  
    source_code = """  
def example_function(param1, param2):  
    pass  
"""  
    manager = DocStringManager(source_code)  
    function_node = manager.tree.body[0]  
    prompt = manager.generate_prompt(function_node)  
    assert "example_function" in prompt

    docstring = "This is a test docstring."
    manager.insert_docstring(function_node, docstring)
    updated_code = manager.update_source_code()
    assert docstring in updated_code

    entries = [{
        "function_name": "example_function",
        "summary": "Example function summary.",
        "changelog": "Initial documentation.",
        "docstring": docstring,
        "complexity_score": 1
    }]
    markdown = manager.generate_markdown_documentation(entries)
    assert "Example function summary." in markdown
    assert "Initial documentation." in markdown

# Run Test Cases

test_docstring_manager()  
[Timestamp: 2024/11/12 20:29:40]

**ai**: The provided code for `DocStringManager` is well-structured and includes methods for generating prompts, inserting docstrings, updating source code, and generating Markdown documentation. Below are some improvements and additional test cases to ensure robustness and maintainability:

1. **Error Handling**: Ensure that the `insert_docstring` method handles cases where the function body might be empty.
2. **Type Annotations**: Add type annotations to methods for better clarity.
3. **Logging**: Ensure all critical operations are logged for easier debugging.
4. **Test Cases**: Expand test cases to cover edge cases, such as functions with no parameters or existing docstrings.

Here's the updated code with improvements and additional test cases:

```python
import ast
import astor
from logger import log_info, log_error
from metrics import Metrics

class DocStringManager:
    """
    Handles the generation, updating, and insertion of docstrings into Python code using AST.
    """

    def __init__(self, source_code: str):
        """
        Initialize the DocStringManager with source code and parse it into an AST.

        :param source_code: The Python source code to manage.
        :raises SyntaxError: If the source code cannot be parsed into an AST.
        """
        self.source_code = source_code
        try:
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed from source code.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code into AST: {e}")
            raise e

    def generate_prompt(self, function_node: ast.FunctionDef) -> str:
        """
        Create a prompt for generating a docstring using OpenAI based on function metadata.

        :param function_node: The AST node of the function to generate a prompt for.
        :return: A string prompt for generating a docstring.
        """
        func_name = function_node.name
        args = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") for arg in function_node.args.args]
        return_annotation = ast.unparse(function_node.returns) if function_node.returns else "Any"
        complexity_score = Metrics.calculate_cyclomatic_complexity(function_node)

        params_formatted = ", ".join([f"{name}: {ptype}" for name, ptype in args])

        prompt = f"""
        Generate a JSON object with the following fields:
        {{
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Google-style docstring including a Complexity section.",
            "complexity_score": {complexity_score}
        }}

        Function: {func_name}
        Parameters: {params_formatted}
        Returns: {return_annotation}
        Existing docstring: None
        """
        return prompt.strip()

    def insert_docstring(self, function_node: ast.FunctionDef, docstring: str):
        """
        Insert or replace the docstring in the function's AST node.

        :param function_node: The AST node of the function to insert the docstring into.
        :param docstring: The docstring to insert.
        """
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Node is not a function definition, cannot insert docstring.")
            return

        if function_node.body and isinstance(function_node.body[0], ast.Expr) and isinstance(function_node.body[0].value, ast.Constant):
            function_node.body[0].value = ast.Constant(value=docstring)
        else:
            function_node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
        log_info(f"Inserted docstring into function '{function_node.name}'.")

    def update_source_code(self) -> str:
        """
        Update the source code with the modified AST containing new docstrings.

        :return: The updated source code as a string, or None if an error occurs.
        """
        try:
            updated_code = astor.to_source(self.tree)
            log_info("Source code updated with new docstrings.")
            return updated_code
        except Exception as e:
            log_error(f"Failed to generate source code from AST: {e}")
            return None

    def generate_markdown_documentation(self, entries: list) -> str:
        """
        Generate a Markdown documentation from the collected entries.

        :param entries: A list of dictionaries containing function metadata and docstrings.
        :return: A string containing the generated Markdown documentation.
        """
        documentation = "# Summary\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['summary']}\n"
        documentation += "\n# Changelog\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['changelog']}\n"
        documentation += "\n# Function/Method/Class Glossary\n\n"
        documentation += "| Function Name | Complexity Score | Docstring |\n"
        documentation += "|---------------|-----------------|-----------|\n"
        for entry in entries:
            docstring = entry["docstring"].replace("\n", " ")
            documentation += f"| {entry['function_name']} | {entry['complexity_score']} | {docstring} |\n"
        return documentation

# Suggested test cases
def test_docstring_manager():
    # Test with a simple function
    source_code = """
def example_function(param1, param2):
    pass
"""
    manager = DocStringManager(source_code)
    function_node = manager.tree.body[0]
    prompt = manager.generate_prompt(function_node)
    assert "example_function" in prompt

    docstring = "This is a test docstring."
    manager.insert_docstring(function_node, docstring)
    updated_code = manager.update_source_code()
    assert docstring in updated_code

    entries = [{
        "function_name": "example_function",
        "summary": "Example function summary.",
        "changelog": "Initial documentation.",
        "docstring": docstring,
        "complexity_score": 1
    }]
    markdown = manager.generate_markdown_documentation(entries)
    assert "Example function summary." in markdown
    assert "Initial documentation." in markdown

    # Test with a function with no parameters
    source_code_no_params = """
def no_param_function():
    pass
"""
    manager_no_params = DocStringManager(source_code_no_params)
    function_node_no_params = manager_no_params.tree.body[0]
    prompt_no_params = manager_no_params.generate_prompt(function_node_no_params)
    assert "no_param_function" in prompt_no_params

    docstring_no_params = "This is a test docstring for no_param_function."
    manager_no_params.insert_docstring(function_node_no_params, docstring_no_params)
    updated_code_no_params = manager_no_params.update_source_code()
    assert docstring_no_params in updated_code_no_params

    # Test with a function that already has a docstring
    source_code_with_docstring = '''
def function_with_docstring():
    """Existing docstring."""
    pass
'''
    manager_with_docstring = DocStringManager(source_code_with_docstring)
    function_node_with_docstring = manager_with_docstring.tree.body[0]
    new_docstring = "This is a new docstring."
    manager_with_docstring.insert_docstring(function_node_with_docstring, new_docstring)
    updated_code_with_docstring = manager_with_docstring.update_source_code()
    assert new_docstring in updated_code_with_docstring
    assert "Existing docstring." not in updated_code_with_docstring

# Run test cases
test_docstring_manager()
```

## Key Changes

- **Type Annotations**: Added type annotations to method signatures for clarity.
- **Docstring Insertion**: Improved logic to handle existing docstrings and ensure proper insertion.
- **Additional Test Cases**: Added tests for functions with no parameters and functions with existing docstrings to ensure comprehensive coverage.  
[Timestamp: 2024/11/12 20:30:02]

**user**: Merge this into that: def generate_prompt(self, function_node):  
    """  
    Create a prompt for generating a docstring using Azure OpenAI based on function metadata.

    The prompt is designed to generate a Google-style docstring with additional metadata, including a complexity score.
    It provides the output in a structured JSON format.

    :param function_node: The AST node of the function to generate a prompt for.
    :return: A tuple containing the prompt string and the function definition for structured output, or None if the prompt is too long.

    The generated prompt includes:
    - Function name
    - Parameters with types
    - Return type
    - Complexity score
    - Existing docstring (if any)

    Example prompt format:
    ```
    Function: <function_name>
    Parameters: <param1>: <type1>, <param2>: <type2>, ...
    Returns: <return_type>
    Complexity Score: <complexity_score>
    Existing Docstring: None

    Generate a docstring for the above function using Google style, including a 'Complexity' section.
    Provide the output in the following JSON format:
    {
        "summary": "Brief function overview.",
        "changelog": "Change history or 'Initial documentation.'",
        "docstring": "Complete Google-style docstring.",
        "complexity_score": <complexity_score>
    }
    ```

    Note: If the prompt exceeds 1500 words, it will not be processed, and an error will be logged.
    """
    func_name = function_node.name
    args = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") for arg in function_node.args.args]
    return_annotation = ast.unparse(function_node.returns) if function_node.returns else "Any"
    complexity_score = Metrics.calculate_cyclomatic_complexity(function_node)

    params_formatted = ", ".join([f"{name}: {ptype}" for name, ptype in args])

    # Define function for structured output
    function_definition = {
        "name": "generate_docstring",
        "description": "Generates a Google-style docstring with additional metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "changelog": {"type": "string"},
                "docstring": {"type": "string"},
                "complexity_score": {"type": "integer"}
            },
            "required": ["summary", "docstring"]
        }
    }

    prompt = f"""
    Function: {func_name}
    Parameters: {params_formatted}
    Returns: {return_annotation}
    Complexity Score: {complexity_score}
    Existing Docstring: None

    Generate a docstring for the above function using Google style, including a 'Complexity' section.
    Provide the output in the following JSON format:
    {{
        "summary": "Brief function overview.",
        "changelog": "Change history or 'Initial documentation.'",
        "docstring": "Complete Google-style docstring.",
        "complexity_score": {complexity_score}
    }}
    """

    # Token limit check
    if len(prompt.split()) > 1500:
        log_error(f"Prompt too long for function '{func_name}'. Skipping Azure OpenAI call.")
        return None

    return prompt.strip(), function_definition
[Timestamp: 2024/11/12 20:30:08]

**ai**: To merge the new `generate_prompt` function into the existing `DocStringManager` class, we need to replace the current `generate_prompt` method with the new implementation. This new version includes additional functionality, such as a structured output definition and a check for prompt length. Here's how the updated `DocStringManager` class will look:

```python
import ast
import astor
from logger import log_info, log_error
from metrics import Metrics

class DocStringManager:
    """
    Handles the generation, updating, and insertion of docstrings into Python code using AST.
    """

    def __init__(self, source_code: str):
        """
        Initialize the DocStringManager with source code and parse it into an AST.

        :param source_code: The Python source code to manage.
        :raises SyntaxError: If the source code cannot be parsed into an AST.
        """
        self.source_code = source_code
        try:
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed from source code.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code into AST: {e}")
            raise e

    def generate_prompt(self, function_node: ast.FunctionDef):
        """
        Create a prompt for generating a docstring using Azure OpenAI based on function metadata.

        The prompt is designed to generate a Google-style docstring with additional metadata, including a complexity score.
        It provides the output in a structured JSON format.

        :param function_node: The AST node of the function to generate a prompt for.
        :return: A tuple containing the prompt string and the function definition for structured output, or None if the prompt is too long.

        The generated prompt includes:
        - Function name
        - Parameters with types
        - Return type
        - Complexity score
        - Existing docstring (if any)

        Example prompt format:
        ```
        Function: <function_name>
        Parameters: <param1>: <type1>, <param2>: <type2>, ...
        Returns: <return_type>
        Complexity Score: <complexity_score>
        Existing Docstring: None

        Generate a docstring for the above function using Google style, including a 'Complexity' section.
        Provide the output in the following JSON format:
        {
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Complete Google-style docstring.",
            "complexity_score": <complexity_score>
        }
        ```

        Note: If the prompt exceeds 1500 words, it will not be processed, and an error will be logged.
        """
        func_name = function_node.name
        args = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") for arg in function_node.args.args]
        return_annotation = ast.unparse(function_node.returns) if function_node.returns else "Any"
        complexity_score = Metrics.calculate_cyclomatic_complexity(function_node)

        params_formatted = ", ".join([f"{name}: {ptype}" for name, ptype in args])

        # Define function for structured output
        function_definition = {
            "name": "generate_docstring",
            "description": "Generates a Google-style docstring with additional metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "changelog": {"type": "string"},
                    "docstring": {"type": "string"},
                    "complexity_score": {"type": "integer"}
                },
                "required": ["summary", "docstring"]
            }
        }

        prompt = f"""
        Function: {func_name}
        Parameters: {params_formatted}
        Returns: {return_annotation}
        Complexity Score: {complexity_score}
        Existing Docstring: None

        Generate a docstring for the above function using Google style, including a 'Complexity' section.
        Provide the output in the following JSON format:
        {{
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Complete Google-style docstring.",
            "complexity_score": {complexity_score}
        }}
        """

        # Token limit check
        if len(prompt.split()) > 1500:
            log_error(f"Prompt too long for function '{func_name}'. Skipping Azure OpenAI call.")
            return None

        return prompt.strip(), function_definition

    def insert_docstring(self, function_node: ast.FunctionDef, docstring: str):
        """
        Insert or replace the docstring in the function's AST node.

        :param function_node: The AST node of the function to insert the docstring into.
        :param docstring: The docstring to insert.
        """
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Node is not a function definition, cannot insert docstring.")
            return

        if function_node.body and isinstance(function_node.body[0], ast.Expr) and isinstance(function_node.body[0].value, ast.Constant):
            function_node.body[0].value = ast.Constant(value=docstring)
        else:
            function_node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
        log_info(f"Inserted docstring into function '{function_node.name}'.")

    def update_source_code(self) -> str:
        """
        Update the source code with the modified AST containing new docstrings.

        :return: The updated source code as a string, or None if an error occurs.
        """
        try:
            updated_code = astor.to_source(self.tree)
            log_info("Source code updated with new docstrings.")
            return updated_code
        except Exception as e:
            log_error(f"Failed to generate source code from AST: {e}")
            return None

    def generate_markdown_documentation(self, entries: list) -> str:
        """
        Generate a Markdown documentation from the collected entries.

        :param entries: A list of dictionaries containing function metadata and docstrings.
        :return: A string containing the generated Markdown documentation.
        """
        documentation = "# Summary\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['summary']}\n"
        documentation += "\n# Changelog\n\n"
        for entry in entries:
            documentation += f"- {entry['function_name']}: {entry['changelog']}\n"
        documentation += "\n# Function/Method/Class Glossary\n\n"
        documentation += "| Function Name | Complexity Score | Docstring |\n"
        documentation += "|---------------|-----------------|-----------|\n"
        for entry in entries:
            docstring = entry["docstring"].replace("\n", " ")
            documentation += f"| {entry['function_name']} | {entry['complexity_score']} | {docstring} |\n"
        return documentation

# Suggested test cases
def test_docstring_manager():
    # Test with a simple function
    source_code = """
def example_function(param1, param2):
    pass
"""
    manager = DocStringManager(source_code)
    function_node = manager.tree.body[0]
    prompt, function_definition = manager.generate_prompt(function_node)
    assert "example_function" in prompt

    docstring = "This is a test docstring."
    manager.insert_docstring(function_node, docstring)
    updated_code = manager.update_source_code()
    assert docstring in updated_code

    entries = [{
        "function_name": "example_function",
        "summary": "Example function summary.",
        "changelog": "Initial documentation.",
        "docstring": docstring,
        "complexity_score": 1
    }]
    markdown = manager.generate_markdown_documentation(entries)
    assert "Example function summary." in markdown
    assert "Initial documentation." in markdown

    # Test with a function with no parameters
    source_code_no_params = """
def no_param_function():
    pass
"""
    manager_no_params = DocStringManager(source_code_no_params)
    function_node_no_params = manager_no_params.tree.body[0]
    prompt_no_params, _ = manager_no_params.generate_prompt(function_node_no_params)
    assert "no_param_function" in prompt_no_params

    docstring_no_params = "This is a test docstring for no_param_function."
    manager_no_params.insert_docstring(function_node_no_params, docstring_no_params)
    updated_code_no_params = manager_no_params.update_source_code()
    assert docstring_no_params in updated_code_no_params

    # Test with a function that already has a docstring
    source_code_with_docstring = '''
def function_with_docstring():
    """Existing docstring."""
    pass
'''
    manager_with_docstring = DocStringManager(source_code_with_docstring)
    function_node_with_docstring = manager_with_docstring.tree.body[0]
    new_docstring = "This is a new docstring."
    manager_with_docstring.insert_docstring(function_node_with_docstring, new_docstring)
    updated_code_with_docstring = manager_with_docstring.update_source_code()
    assert new_docstring in updated_code_with_docstring
    assert "Existing docstring." not in updated_code_with_docstring

# Run test cases
test_docstring_manager()
```

## Key Changes

- **Updated `generate_prompt`**: The method now includes a structured output definition and checks for prompt length.
- **Return Type**: The method returns a tuple containing the prompt and the function definition.
- **Test Cases**: Updated to handle the new return type of `generate_prompt`.  
[Timestamp: 2024/11/12 20:30:56]
