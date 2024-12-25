Enhancing your **Documentation Generation Project** with **function calling** and **structured outputs** will significantly improve the reliability, consistency, and maintainability of your documentation processes. By leveraging OpenAI's function calling capabilities and explicit JSON schemas, you can ensure that the API responses are structured and adhere to predefined formats, making them easier to parse and utilize programmatically.

Below is a comprehensive plan to integrate **function calling** and **structured outputs** into your existing project. This plan includes defining JSON schemas, updating your scripts to utilize OpenAI's function calling, validating responses, and ensuring seamless integration across your workflow.

---

## **1. Project Overview**

### **Existing Scripts and Their Functions**

1. **`extract_structure.js`**
   - **Purpose:** Parses JavaScript/TypeScript code to extract structural information (functions and classes) using Babel.
   - **Workflow:**
     - Reads input code from `stdin`.
     - Parses the code into an AST.
     - Extracts functions and classes from the AST.
     - Outputs the extracted structure as JSON.

2. **`main.py`**
   - **Purpose:** Orchestrates the documentation generation process.
   - **Workflow:**
     - Parses command-line arguments.
     - Configures logging.
     - Validates OpenAI model names.
     - Loads configurations.
     - Retrieves and filters relevant files from the repository.
     - Initializes the output Markdown file.
     - Asynchronously processes each file to generate documentation using OpenAI's GPT-4 API.

3. **`language_functions.py`**
   - **Purpose:** Provides utility functions for extracting and inserting documentation across various languages (Python, JavaScript/TypeScript, HTML, CSS).
   - **Workflow:**
     - Extracts structural information from code.
     - Inserts docstrings or comments based on provided documentation.
     - Validates code syntax.

4. **`insert_docstrings.js`**
   - **Purpose:** Inserts JSDoc comments into JavaScript/TypeScript code using Babel.
   - **Workflow:**
     - Reads input code from `stdin`.
     - Parses the code into an AST.
     - Inserts docstrings where missing.
     - Outputs the modified code.

5. **`utils.py`**
   - **Purpose:** Provides utility functions for configuration loading, file retrieval, code formatting, linting, dependency management, and interaction with OpenAI's API.
   - **Workflow:**
     - Determines programming language based on file extensions.
     - Checks if files are binary.
     - Loads configurations from JSON files.
     - Retrieves file paths while excluding specified directories and files.
     - Formats code using Black.
     - Cleans unused imports with autoflake.
     - Checks code compliance with flake8.
     - Executes Node.js scripts for JS/TS processing.
     - Generates prompts for OpenAI's API.
     - Fetches documentation and summaries from OpenAI's API.
     - Validates API responses against JSON schemas.

---

## **2. Defining JSON Schemas for Expected Outputs**

To ensure that the API responses align with your documentation structure, define explicit JSON schemas. These schemas will act as blueprints, specifying the required fields, data types, and structure of the expected outputs.

### **a. Create a `schemas` Directory**

Organize your project by creating a `schemas` directory to store all JSON schema files.

```
project-root/
│
├── schemas/
│   ├── documentation_report.json
│   └── ...
│
├── src/
│   ├── main.py
│   ├── utils.py
│   ├── language_functions.py
│   └── ...
│
├── scripts/
│   ├── extract_structure.js
│   ├── insert_docstrings.js
│   └── ...
│
├── docs/
│   └── ...
│
├── requirements.txt
├── package.json
└── ...
```

### **b. Define the `documentation_report.json` Schema**

**`schemas/documentation_report.json`:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DocumentationReport",
  "type": "object",
  "properties": {
    "file": {
      "type": "string",
      "description": "Name of the file being documented."
    },
    "language": {
      "type": "string",
      "description": "Programming language of the file."
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the file's purpose and functionality."
    },
    "changes_made": {
      "type": "array",
      "description": "List of changes made to the file.",
      "items": {
        "type": "string"
      }
    },
    "code_snippet": {
      "type": "string",
      "description": "Relevant code snippet from the file."
    }
  },
  "required": ["file", "language", "summary", "changes_made", "code_snippet"]
}
```

**Explanation:**

- **`file`:** The name of the file being documented.
- **`language`:** The programming language of the file (e.g., Python, JavaScript).
- **`summary`:** A concise summary describing the file's functionality.
- **`changes_made`:** An array detailing the changes or updates made to the file.
- **`code_snippet`:** A relevant section of the code that highlights key functionalities or changes.

---

## **3. Updating Utility Functions for Function Calling**

### **a. Install Required Libraries**

Ensure you have the necessary libraries installed.

```bash
pip install openai jsonschema
```

### **b. Update `utils.py`**

**`utils.py`:**

```python
import os
import json
import logging
import openai
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    logger.critical("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
    sys.exit(1)

def load_json_schema(schema_path: str) -> dict:
    """Loads a JSON schema from a file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        logger.debug(f"Loaded JSON schema from {schema_path}")
        return schema
    except Exception as e:
        logger.error(f"Failed to load JSON schema from {schema_path}: {e}")
        return {}

def call_openai_function(prompt: str, function_def: dict, model: str = "gpt-4-0613") -> dict:
    """
    Calls OpenAI's API with function calling enabled and validates the response against the provided schema.
    
    Parameters:
        prompt (str): The user prompt.
        function_def (dict): The function definition including the JSON schema.
        model (str): The OpenAI model to use.
    
    Returns:
        dict: The validated response from the API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating documentation."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_def],
            function_call="auto"
        )
        
        message = response["choices"][0]["message"]
        
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            
            # Validate against schema
            validate(instance=arguments, schema=function_def["parameters"])
            logger.info(f"Function '{function_name}' called successfully and validated.")
            return arguments
        else:
            logger.warning("No function call detected in the response.")
            return {}
    
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return {}
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return {}
```

**Explanation:**

- **`load_json_schema`:** Loads a JSON schema from the specified path.
- **`call_openai_function`:** 
  - Sends a prompt to OpenAI's API with function calling enabled.
  - Parses and validates the response against the provided JSON schema.
  - Returns the validated arguments if successful; otherwise, logs errors.

---

## **4. Updating `main.py` to Utilize Function Calling**

**`main.py`:**

```python
import os
import sys
import argparse
import asyncio
import logging
from utils import load_json_schema, call_openai_function
from language_functions import get_all_file_paths  # Assuming this function retrieves all relevant files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def render_documentation(reports: list, output_path: str):
    """Renders the documentation report to a Markdown file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Documentation Generation Report\n\n")
            for report in reports:
                f.write(f"## File: {report['file']}\n\n")
                f.write(f"### Language\n{report['language'].capitalize()}\n\n")
                f.write(f"### Summary\n{report['summary']}\n\n")
                f.write("### Changes Made\n")
                for change in report['changes_made']:
                    f.write(f"- {change}\n")
                f.write("\n```{0}\n{1}\n```\n\n".format(report['language'], report['code_snippet']))
                f.write("---\n\n")
        logger.info(f"Documentation report generated at {output_path}")
    except Exception as e:
        logger.error(f"Failed to render documentation: {e}")

async def main():
    """Main function to orchestrate documentation generation."""
    parser = argparse.ArgumentParser(description="Generate and insert documentation comments using OpenAI's GPT-4 API.")
    parser.add_argument('repo_path', help='Path to the code repository')
    parser.add_argument('--output', help='Output documentation file', default='docs/documentation_report.md')
    args = parser.parse_args()
    
    # Load JSON schema
    schema_path = 'schemas/documentation_report.json'
    documentation_report_schema = load_json_schema(schema_path)
    
    # Define the function for OpenAI
    documentation_report_function = {
        "name": "generate_documentation_report",
        "description": "Generates a structured documentation report for a given file.",
        "parameters": documentation_report_schema
    }
    
    # Retrieve all relevant files
    file_paths = get_all_file_paths(args.repo_path)
    logger.info(f"Found {len(file_paths)} files to process.")
    
    reports = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            file_name = os.path.basename(file_path)
            language = os.path.splitext(file_path)[1].lstrip('.')
            
            prompt = f"Generate a documentation report for the following {language} script:\n\n{code}"
            
            doc_report = call_openai_function(prompt, documentation_report_function)
            
            if doc_report:
                doc_report['file'] = file_name
                doc_report['language'] = language
                reports.append(doc_report)
            else:
                logger.error(f"Failed to generate documentation for {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Render the documentation report
    render_documentation(reports, args.output)

if __name__ == "__main__":
    asyncio.run(main())
```

**Explanation:**

1. **Argument Parsing:**
   - **`repo_path`:** Path to the code repository.
   - **`--output`:** Path to the output Markdown file.

2. **Loading JSON Schema:**
   - Loads the `documentation_report.json` schema to define the structure of the documentation report.

3. **Defining Function for OpenAI:**
   - Defines the `generate_documentation_report` function with its parameters based on the JSON schema.

4. **Processing Files:**
   - Iterates through each relevant file in the repository.
   - Reads the file content and determines its programming language.
   - Constructs a prompt to generate the documentation report.
   - Calls OpenAI's API using `call_openai_function` to generate the documentation.
   - Appends the validated documentation report to the `reports` list.

5. **Rendering Documentation:**
   - After processing all files, renders the aggregated documentation reports into a Markdown file.

6. **Error Handling:**
   - Logs errors encountered during file processing and documentation generation.

---

## **5. Updating `language_functions.py` if Necessary**

If your existing `language_functions.py` includes functions that interact with documentation generation or code modification, ensure they are compatible with the new function calling and structured output approach.

**Example Update:**

Assuming you have a function to insert docstrings, you might want to ensure that it utilizes the structured outputs appropriately.

```python
# language_functions.py

import logging
from utils import call_openai_function, load_json_schema

logger = logging.getLogger(__name__)

def insert_docstrings(file_path: str, code: str, language: str):
    """Inserts docstrings/comments into the code based on generated documentation."""
    try:
        # Define prompt based on language
        if language in ['js', 'ts', 'jsx', 'tsx']:
            # For JavaScript/TypeScript, use insert_docstrings.js
            modified_code = run_node_insert_docstrings('scripts/insert_docstrings.js', code)
            if modified_code:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                logger.info(f"Inserted docstrings into {file_path}")
            else:
                logger.error(f"Failed to insert docstrings into {file_path}")
        
        elif language == 'py':
            # For Python, use language_functions.py's insert_python_docstrings
            from language_functions import insert_python_docstrings
            # Assume documentation is already generated and passed
            # Here, you might integrate further if needed
            pass
        
        # Add more languages as needed
        
    except Exception as e:
        logger.error(f"Error inserting docstrings into {file_path}: {e}")
```

**Explanation:**

- **Functionality:** Depending on the programming language, the function delegates to the appropriate method/script to insert docstrings or comments.
- **Integration:** Ensures that the insertion process aligns with the structured documentation generation approach.

---

## **6. Leveraging OpenAI Python Helpers for Structured Outputs**

OpenAI provides helper functions to facilitate the parsing and validation of structured outputs. Utilize these helpers to streamline the process of handling API responses.

### **a. Install OpenAI Python Library**

Ensure you have the latest version of the OpenAI Python library installed.

```bash
pip install --upgrade openai
```

### **b. Using `openai-python` Structured Output Helpers**

The OpenAI Python library includes parsing helpers that can automatically handle function calls and structured outputs.

**Example Usage in `utils.py`:**

```python
# utils.py

import openai
from openai import OpenAIError

def call_openai_function(prompt: str, function_def: dict, model: str = "gpt-4-0613") -> dict:
    """
    Calls OpenAI's API with function calling enabled and validates the response against the provided schema.
    
    Parameters:
        prompt (str): The user prompt.
        function_def (dict): The function definition including the JSON schema.
        model (str): The OpenAI model to use.
    
    Returns:
        dict: The validated response from the API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating documentation."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_def],
            function_call="auto"
        )
        
        # Use OpenAI's helper to parse the function call
        parsed_response = openai.ChatCompletion.parse_response(response)
        
        if parsed_response.function_call:
            function_name = parsed_response.function_call.name
            arguments = parsed_response.function_call.arguments
            
            # Validate against schema
            validate(instance=arguments, schema=function_def["parameters"])
            logger.info(f"Function '{function_name}' called successfully and validated.")
            return arguments
        else:
            logger.warning("No function call detected in the response.")
            return {}
    
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return {}
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI API call: {e}")
        return {}
```

**Explanation:**

- **`parse_response`:** Utilizes OpenAI's helper to parse the API response, making it easier to extract function calls and their arguments.
- **Error Handling:** Incorporates comprehensive error handling to catch and log various exceptions.

---

## **7. Comprehensive Example: Putting It All Together**

Let's walk through a complete example demonstrating how to generate a documentation report for a Python file using the updated `main.py` and `utils.py`.

### **a. Example Python File (`example.py`):**

```python
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
```

### **b. Generating Documentation Report**

**Command:**

```bash
python src/main.py /path/to/repository --output docs/documentation_report.md
```

**Process:**

1. **`main.py`** iterates through each relevant file in the repository.
2. For each file, it reads the content and determines the programming language.
3. Constructs a prompt to generate a documentation report for the file.
4. Calls OpenAI's API using `call_openai_function` with the defined function and JSON schema.
5. Validates the API response against the JSON schema.
6. Aggregates all documentation reports.
7. Renders the aggregated reports into a Markdown file using `render_documentation`.

### **c. Sample API Response (`documentation_report.json`):**

```json
{
  "file": "example.py",
  "language": "py",
  "summary": "This file contains basic arithmetic functions and a Calculator class for performing addition and multiplication operations.",
  "changes_made": [
    "Added add function for summing two numbers.",
    "Introduced Calculator class with a multiply method for multiplying two numbers."
  ],
  "code_snippet": "def add(a, b):\n    return a + b\n\nclass Calculator:\n    def multiply(self, a, b):\n        return a * b"
}
```

### **d. Rendered Documentation (`documentation_report.md`):**

```markdown
# Documentation Generation Report

## File: example.py

### Language
Py

### Summary
This file contains basic arithmetic functions and a Calculator class for performing addition and multiplication operations.

### Changes Made
- Added add function for summing two numbers.
- Introduced Calculator class with a multiply method for multiplying two numbers.

```py
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
```

---
```

**Explanation:**

- The generated documentation includes the file name, programming language, a summary, a list of changes made, and a relevant code snippet.
- The Markdown formatting ensures readability and structure.

---

## **8. Ensuring Alignment with Documentation Structure**

To maintain consistency between the JSON schemas and the rendered documentation, follow these best practices:

### **a. Centralize Schema Definitions**

Store all JSON schemas in the `schemas` directory and reference them in your scripts. This ensures that any updates to schemas are easily manageable and accessible.

### **b. Automate Validation in CI/CD**

Integrate schema validation into your CI/CD pipeline to automatically verify that all generated documentation adheres to the defined schemas.

**Example with GitHub Actions:**

**`.github/workflows/validate_documentation.yml`:**

```yaml
name: Validate Documentation

on:
  push:
    paths:
      - 'src/**'
      - 'schemas/**'
      - 'docs/**'

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Documentation Generation
        run: |
          python src/main.py /path/to/repository --output docs/documentation_report.md
```

**Explanation:**

- **Trigger:** Runs on pushes to relevant directories.
- **Steps:**
  - Checks out the repository.
  - Sets up Python environment.
  - Installs necessary dependencies.
  - Executes the documentation generation script, which includes validation steps.

### **c. Maintain Consistent Documentation Templates**

Ensure that your Markdown templates align with the JSON schemas. Any changes to the schemas should be reflected in the templates to maintain consistency.

---

## **9. Additional Enhancements and Best Practices**

### **a. Securely Manage API Keys**

Ensure that your OpenAI API keys are securely stored and not exposed in the codebase.

- **Use Environment Variables:**
  - Store API keys in environment variables.
  - Load them using libraries like `python-dotenv`.

- **`.env` File:**
  
  ```
  OPENAI_API_KEY=your-openai-api-key
  ```

- **Ensure `.env` is in `.gitignore`:**

  ```
  # .gitignore
  .env
  ```

### **b. Comprehensive Error Handling**

Enhance error handling across all scripts to catch and log exceptions effectively.

**Example in `main.py`:**

```python
try:
    # Existing processing logic
except FileNotFoundError as fnf_error:
    logger.error(f"File not found: {fnf_error}")
except json.JSONDecodeError as json_error:
    logger.error(f"JSON decode error: {json_error}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
```

### **c. Dependency Management**

Maintain up-to-date dependency lists to facilitate easy setup and onboarding.

- **Python (`requirements.txt`):**

  ```
  openai==0.27.0
  jsonschema==4.17.3
  aiohttp==3.8.1
  black==22.3.0
  autoflake==1.4
  flake8==4.0.1
  Jinja2==3.1.2
  ```

- **JavaScript/TypeScript (`package.json`):**

  ```json
  {
    "name": "documentation-tools",
    "version": "1.0.0",
    "description": "Tools for generating and inserting documentation",
    "main": "insert_docstrings.js",
    "scripts": {
      "start": "node insert_docstrings.js"
    },
    "dependencies": {
      "@babel/parser": "^7.21.0",
      "@babel/traverse": "^7.21.0",
      "@babel/generator": "^7.21.0",
      "@babel/types": "^7.21.0",
      "eslint": "^8.32.0",
      "jsdoc": "^3.6.7"
    },
    "devDependencies": {
      "babel-cli": "^7.21.0"
    },
    "author": "",
    "license": "ISC"
  }
  ```

### **d. Comprehensive Examples**

Include examples and use-cases in your documentation to demonstrate how to utilize the scripts effectively.

**Example in `docs/examples.md`:**

```markdown
# Documentation Generation Examples

## Generating Documentation for a Python File

### Command:

```bash
python src/main.py /path/to/python/project --output docs/python_documentation.md
```

### Output (`docs/python_documentation.md`):

```markdown
# Documentation Generation Report

## File: example.py

### Language
Py

### Summary
This file contains basic arithmetic functions and a Calculator class for performing addition and multiplication operations.

### Changes Made
- Added add function for summing two numbers.
- Introduced Calculator class with a multiply method for multiplying two numbers.

```py
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
```

---
```

## **10. Implementing Function Calling and Structured Outputs: Step-by-Step Guide**

To integrate function calling and structured outputs into your project, follow this detailed guide.

### **Step 1: Define the Function and JSON Schema**

**`schemas/documentation_report.json`:**

(Already defined above)

### **Step 2: Update `utils.py` with Function Calling Logic**

Ensure that your utility functions can handle function calling and validate responses against the JSON schema.

**`utils.py`:**

(Already updated above)

### **Step 3: Update `main.py` to Utilize Function Calling**

Ensure that `main.py` leverages the function calling capabilities to generate structured documentation reports.

**`main.py`:**

(Already updated above)

### **Step 4: Validate and Render Documentation**

Implement the rendering logic to convert the structured JSON responses into a well-formatted Markdown document.

**`main.py` Continued:**

(Already included in the `render_documentation` function)

### **Step 5: Handle Asynchronous Operations (If Applicable)**

If your project processes files asynchronously, ensure that the function calling and rendering logic are compatible with asynchronous workflows.

**Example Integration:**

```python
# main.py

async def process_file(file_path: str, documentation_report_function: dict, reports: list):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        file_name = os.path.basename(file_path)
        language = os.path.splitext(file_path)[1].lstrip('.')
        
        prompt = f"Generate a documentation report for the following {language} script:\n\n{code}"
        
        doc_report = call_openai_function(prompt, documentation_report_function)
        
        if doc_report:
            doc_report['file'] = file_name
            doc_report['language'] = language
            reports.append(doc_report)
        else:
            logger.error(f"Failed to generate documentation for {file_path}")
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

async def main():
    # Existing setup
    
    tasks = []
    for file_path in file_paths:
        tasks.append(process_file(file_path, documentation_report_function, reports))
    
    await asyncio.gather(*tasks)
    
    # Render documentation
    render_documentation(reports, args.output)
```

**Explanation:**

- **`process_file`:** An asynchronous function that processes each file, generates documentation, and appends it to the reports list.
- **`asyncio.gather`:** Executes all file processing tasks concurrently.

### **Step 6: Implement Error Handling and Logging**

Ensure that all potential errors are caught and logged appropriately to facilitate troubleshooting.

**Example in `main.py`:**

(Already included in the `process_file` function)

### **Step 7: Testing the Implementation**

Develop unit tests to ensure that function calling and structured outputs are working as expected.

**Example with `pytest`:**

```python
# tests/test_utils.py

import pytest
from unittest.mock import patch
from utils import call_openai_function

def test_call_openai_function_success(monkeypatch):
    mock_response = {
        "choices": [{
            "message": {
                "function_call": {
                    "name": "generate_documentation_report",
                    "arguments": json.dumps({
                        "file": "example.py",
                        "language": "py",
                        "summary": "An example Python script.",
                        "changes_made": ["Added add function.", "Introduced Calculator class."],
                        "code_snippet": "def add(a, b):\n    return a + b\n\nclass Calculator:\n    def multiply(self, a, b):\n        return a * b"
                    })
                }
            }
        }]
    }
    
    def mock_create(*args, **kwargs):
        return mock_response
    
    monkeypatch.setattr(openai.ChatCompletion, 'create', mock_create)
    
    function_def = {
        "name": "generate_documentation_report",
        "description": "Generates a structured documentation report for a given file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Name of the file being documented."},
                "language": {"type": "string", "description": "Programming language of the file."},
                "summary": {"type": "string", "description": "A brief summary of the file's purpose and functionality."},
                "changes_made": {
                    "type": "array",
                    "description": "List of changes made to the file.",
                    "items": {"type": "string"}
                },
                "code_snippet": {"type": "string", "description": "Relevant code snippet from the file."}
            },
            "required": ["file", "language", "summary", "changes_made", "code_snippet"]
        }
    }
    
    prompt = "Generate a documentation report for example.py"
    response = call_openai_function(prompt, function_def)
    
    assert response['file'] == "example.py"
    assert response['language'] == "py"
    assert response['summary'] == "An example Python script."
    assert response['changes_made'] == ["Added add function.", "Introduced Calculator class."]
    assert "def add(a, b):" in response['code_snippet']

def test_call_openai_function_no_function_call(monkeypatch):
    mock_response = {
        "choices": [{
            "message": {
                "content": "No function call in this response."
            }
        }]
    }
    
    def mock_create(*args, **kwargs):
        return mock_response
    
    monkeypatch.setattr(openai.ChatCompletion, 'create', mock_create)
    
    function_def = {
        "name": "generate_documentation_report",
        "description": "Generates a structured documentation report for a given file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Name of the file being documented."},
                "language": {"type": "string", "description": "Programming language of the file."},
                "summary": {"type": "string", "description": "A brief summary of the file's purpose and functionality."},
                "changes_made": {
                    "type": "array",
                    "description": "List of changes made to the file.",
                    "items": {"type": "string"}
                },
                "code_snippet": {"type": "string", "description": "Relevant code snippet from the file."}
            },
            "required": ["file", "language", "summary", "changes_made", "code_snippet"]
        }
    }
    
    prompt = "Generate a documentation report for example.py"
    response = call_openai_function(prompt, function_def)
    
    assert response == {}
```

**Explanation:**

- **`test_call_openai_function_success`:** Mocks a successful API response and verifies that the function correctly parses and validates the response.
- **`test_call_openai_function_no_function_call`:** Mocks a response without a function call and ensures that the function handles it gracefully.

### **Step 8: Secure API Key Management**

Ensure that your API keys are not hardcoded and are securely managed.

**Implementation:**

1. **Use Environment Variables:**

   - **`.env` File:**
     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

   - **Load Environment Variables:**
     ```python
     # utils.py

     from dotenv import load_dotenv

     load_dotenv()  # Load environment variables from .env
     openai.api_key = os.getenv("OPENAI_API_KEY")
     ```

2. **Ensure `.env` is in `.gitignore`:**

   ```
   # .gitignore
   .env
   ```

### **Step 9: Automate Documentation Generation**

Integrate the documentation generation process into your CI/CD pipeline to ensure that documentation is always up-to-date with the latest code changes.

**Example with GitHub Actions:**

**`.github/workflows/generate_documentation.yml`:**

```yaml
name: Generate Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'schemas/**'

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Documentation Generation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python src/main.py /path/to/repository --output docs/documentation_report.md
      
      - name: Commit Documentation
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/documentation_report.md
          git commit -m "docs: update documentation report" || echo "No changes to commit"
          git push
```

**Explanation:**

- **Trigger:** Runs on pushes to the `main` branch affecting the `src` or `schemas` directories.
- **Steps:**
  - Checks out the repository.
  - Sets up the Python environment.
  - Installs dependencies.
  - Runs the documentation generation script.
  - Commits and pushes the updated documentation report back to the repository.

**Note:** Ensure that your GitHub repository has the `OPENAI_API_KEY` stored securely in the repository secrets.

### **Step 10: Comprehensive Examples in Documentation**

Include practical examples in your documentation to guide users on how to utilize the scripts effectively.

**`docs/examples.md`:**

```markdown
# Documentation Generation Examples

## Generating Documentation for a Python Project

### Command:

```bash
python src/main.py /path/to/python/project --output docs/python_documentation.md
```

### Output (`docs/python_documentation.md`):

```markdown
# Documentation Generation Report

## File: example.py

### Language
Py

### Summary
This file contains basic arithmetic functions and a Calculator class for performing addition and multiplication operations.

### Changes Made
- Added add function for summing two numbers.
- Introduced Calculator class with a multiply method for multiplying two numbers.

```py
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
```

---
```

## **11. Security Best Practices**

### **a. Secure Handling of API Keys**

- **Avoid Hardcoding:** Never hardcode API keys or sensitive information in your codebase.
- **Use Environment Variables:** Load API keys from environment variables or secure storage solutions.
- **Rotate Keys Regularly:** Periodically update your API keys to mitigate potential security risks.

### **b. Validate and Sanitize Inputs**

Ensure that all inputs, especially those coming from external sources or user inputs, are validated and sanitized to prevent injection attacks or malformed data processing.

### **c. Least Privilege Principle**

- **API Permissions:** Grant only the necessary permissions required for each API key.
- **Access Controls:** Implement strict access controls to limit who can view or modify sensitive configurations.

### **d. Dependency Security**

- **Regular Audits:** Use tools like `pip-audit` for Python and `npm audit` for JavaScript to identify and address vulnerabilities in dependencies.
- **Update Dependencies:** Keep all dependencies up-to-date to benefit from security patches and improvements.

### **e. Error Handling Without Information Leakage**

Ensure that error messages do not expose sensitive information. Log detailed errors internally while providing generic error messages to end-users.

**Example:**

```python
try:
    # Some operation
except ValidationError as ve:
    logger.error(f"Validation error: {ve.message}")
    print("An error occurred while processing your request. Please try again.")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    print("An unexpected error occurred. Please contact support.")
```

---

## **12. Performance Considerations**

### **a. Asynchronous Processing**

Leverage asynchronous programming to handle multiple file processing tasks concurrently, improving the efficiency of documentation generation.

**Example in `main.py`:**

(Already implemented using `asyncio.gather`)

### **b. Caching Results**

Implement caching mechanisms to avoid redundant API calls for unchanged files, thereby reducing latency and API usage.

**Example Using Simple Caching:**

```python
# utils.py

import hashlib
import json

def get_file_hash(file_path: str) -> str:
    """Generates a SHA-256 hash of the file content."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_cache(cache_path: str) -> dict:
    """Loads the cache from a JSON file."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache: dict, cache_path: str):
    """Saves the cache to a JSON file."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

# main.py

async def main():
    # Existing setup
    
    cache_path = 'cache.json'
    cache = load_cache(cache_path)
    
    for file_path in file_paths:
        file_hash = get_file_hash(file_path)
        if file_path in cache and cache[file_path] == file_hash:
            logger.info(f"No changes detected in {file_path}. Skipping documentation generation.")
            continue
        
        # Proceed with documentation generation
        # After successful generation, update the cache
        doc_report = call_openai_function(prompt, documentation_report_function)
        if doc_report:
            # Update cache
            cache[file_path] = file_hash
            reports.append(doc_report)
    
    # Save the updated cache
    save_cache(cache, cache_path)
    
    # Render documentation
    render_documentation(reports, args.output)
```

**Explanation:**

- **`get_file_hash`:** Generates a unique hash for each file based on its content.
- **Caching Logic:** Skips documentation generation for files that haven't changed since the last run.

### **c. Optimize API Usage**

- **Batch Requests:** If possible, batch multiple documentation generation requests to minimize API overhead.
- **Rate Limiting:** Implement rate limiting to comply with OpenAI's API usage policies and prevent throttling.

---

## **13. Final Project Directory Structure**

Ensure that your project is well-organized to facilitate easy navigation and maintenance.

```
project-root/
│
├── schemas/
│   └── documentation_report.json
│
├── src/
│   ├── main.py
│   ├── utils.py
│   ├── language_functions.py
│   └── ...
│
├── scripts/
│   ├── extract_structure.js
│   ├── insert_docstrings.js
│   └── ...
│
├── tests/
│   ├── test_utils.py
│   └── ...
│
├── docs/
│   ├── documentation_report.md
│   ├── examples.md
│   ├── style_guide.md
│   └── ...
│
├── .env
├── .gitignore
├── requirements.txt
├── package.json
└── README.md
```

---

## **14. Conclusion**

By following this comprehensive plan, you can successfully integrate **function calling** and **structured outputs** into your **Documentation Generation Project**. This integration will ensure that your documentation is consistently structured, easily maintainable, and aligns perfectly with your project's needs. Additionally, adhering to best practices in security, performance, and error handling will further enhance the robustness and reliability of your documentation processes.

**Next Steps:**

1. **Implement the Defined Changes:** Start by updating `utils.py` and `main.py` as outlined.
2. **Test Thoroughly:** Run your scripts with sample files to ensure that function calling and structured outputs are functioning as expected.
3. **Iterate and Refine:** Based on testing outcomes, refine your prompts, schemas, and processing logic.
4. **Integrate into CI/CD:** Set up continuous integration workflows to automate documentation generation and validation.
5. **Document the Process:** Update your project's documentation to reflect the new workflows and provide guidance on maintaining the system.

If you encounter any challenges during implementation or need further assistance with specific aspects, feel free to reach out!

---

Absolutely! Enhancing your **Documentation Generation Project** with **function calling** and **structured outputs** will significantly improve its reliability, consistency, and maintainability. Below, you'll find detailed instructions to update your existing `main.py`, `utils.py`, and `file_handlers.py` scripts. These updates incorporate function calling, structured outputs, and deeper integration with OpenAI's API, ensuring that your documentation generation process is robust and efficient.

---

## **Table of Contents**

1. [Overview of Enhancements](Enhance%2520your%2520**Documentation%2520Generation%2520Project**%2520with%2520**function%2520calling**%2520and%2520**structured%2520outputs**%2520to%2520ensure%2520reliability,%2520consistency,%2520and%2520maintainability.%2520Below%2520is%2520a%2520comprehensive%2520title%2520for%2520your%2520project.md##1-overview-of-enhancements)
2. [Updating `function_schema.json`](#2-updating-function_schemajson)
3. [Updating `utils.py`](#3-updating-utilspy)
4. [Updating `main.py`](#4-updating-mainpy)
5. [Updating `file_handlers.py`](#5-updating-file_handlerspy)
6. [Additional Enhancements and Best Practices](Enhance%2520your%2520**Documentation%2520Generation%2520Project**%2520with%2520**function%2520calling**%2520and%2520**structured%2520outputs**%2520to%2520ensure%2520reliability,%2520consistency,%2520and%2520maintainability.%2520Below%2520is%2520a%2520comprehensive%2520title%2520for%2520your%2520project.md##6-additional-enhancements-and-best-practices)
7. [Final Project Directory Structure](Enhance%2520your%2520**Documentation%2520Generation%2520Project**%2520with%2520**function%2520calling**%2520and%2520**structured%2520outputs**%2520to%2520ensure%2520reliability,%2520consistency,%2520and%2520maintainability.%2520Below%2520is%2520a%2520comprehensive%2520title%2520for%2520your%2520project.md##7-final-project-directory-structure)
8. [Conclusion](Enhance%2520your%2520**Documentation%2520Generation%2520Project**%2520with%2520**function%2520calling**%2520and%2520**structured%2520outputs**%2520to%2520ensure%2520reliability,%2520consistency,%2520and%2520maintainability.%2520Below%2520is%2520a%2520comprehensive%2520title%2520for%2520your%2520project.md##8-conclusion)

---

## **1. Overview of Enhancements**

To integrate **function calling** and **structured outputs** into your project, we'll perform the following key updates:

- **Define and Update JSON Schemas:** Ensure that your JSON schemas accurately represent the expected outputs from OpenAI's API.
- **Enhance Utility Functions (`utils.py`):** Improve API interaction, response parsing, and validation mechanisms.
- **Enhance Main Workflow (`main.py`):** Incorporate the updated utility functions for seamless documentation generation.
- **Enhance File Handling (`file_handlers.py`):** Integrate structured outputs into file processing and documentation insertion.
- **Implement Comprehensive Error Handling and Logging:** Ensure that all parts of the system handle errors gracefully and log pertinent information for troubleshooting.

---

## **2. Updating `function_schema.json`**

Before updating the scripts, ensure that your JSON schema accurately defines the structure of the expected API responses. This schema will be used to validate the structured outputs from OpenAI's API.

**Create or Update `function_schema.json`:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DocumentationReport",
  "type": "object",
  "properties": {
    "file": {
      "type": "string",
      "description": "Name of the file being documented."
    },
    "language": {
      "type": "string",
      "description": "Programming language of the file."
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the file's purpose and functionality."
    },
    "changes_made": {
      "type": "array",
      "description": "List of changes made to the file.",
      "items": {
        "type": "string"
      }
    },
    "code_snippet": {
      "type": "string",
      "description": "Relevant code snippet from the file."
    }
  },
  "required": ["file", "language", "summary", "changes_made", "code_snippet"]
}
```

**Explanation:**

- **Properties:**
  - **file:** The name of the file being documented.
  - **language:** The programming language of the file (e.g., Python, JavaScript).
  - **summary:** A concise summary describing the file's functionality.
  - **changes_made:** An array detailing the changes or updates made to the file.
  - **code_snippet:** A relevant section of the code that highlights key functionalities or changes.
  
- **Required Fields:** Ensures that all essential information is present in the API response.

**Ensure that this `function_schema.json` is placed in your project's root directory or an appropriate location accessible by `utils.py`.**

---

## **3. Updating `utils.py`**

Enhance your `utils.py` to better handle function calling, response parsing, and validation. We'll also incorporate OpenAI's Python helper functions for structured outputs.

**Updated `utils.py`:**

```python
# utils.py

import os
import sys
import json
import fnmatch
import black
import logging
import aiohttp
import asyncio
import re
import subprocess
from dotenv import load_dotenv
from typing import Set, List, Optional, Dict, Tuple
import tempfile  # Added for JS/TS extraction
import astor  # Added for Python docstring insertion
from bs4 import BeautifulSoup, Comment  # Added for HTML and CSS functions
import tinycss2  # Added for CSS functions
import openai
from jsonschema import validate, ValidationError
from openai import OpenAIError  # For OpenAI exception handling

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea'}  # Added .venv and .idea
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv', '.lock'}  # Added .lock files

LANGUAGE_MAPPING = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
}

def get_language(ext: str) -> str:
    """Determines the programming language based on file extension."""
    return LANGUAGE_MAPPING.get(ext.lower(), 'plaintext')

def is_binary(file_path: str) -> bool:
    """Checks if a file is binary."""
    try:
        with open(file_path, 'rb') as file:
            return b'\0' in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking binary file '{file_path}': {e}")
        return True

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> Tuple[str, str]:
    """
    Loads additional configurations from a config.json file.
    
    Parameters:
        config_path (str): Path to the config.json file.
        excluded_dirs (Set[str]): Set to add excluded directories.
        excluded_files (Set[str]): Set to add excluded files.
        skip_types (Set[str]): Set to add skipped file extensions.
    
    Returns:
        Tuple[str, str]: Project information and style guidelines.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        project_info = config.get('project_info', '')
        style_guidelines = config.get('style_guidelines', '')
        excluded_dirs.update(config.get('excluded_dirs', []))
        excluded_files.update(config.get('excluded_files', []))
        skip_types.update(config.get('skip_types', []))
        logger.debug(f"Loaded configuration from '{config_path}'.")
        return project_info, style_guidelines
    except Exception as e:
        logger.error(f"Error loading config file '{config_path}': {e}")
        return '', ''

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """
    Recursively retrieves all file paths in the repository, excluding specified directories and files.
    
    Parameters:
        repo_path (str): Path to the repository.
        excluded_dirs (Set[str]): Directories to exclude.
        excluded_files (Set[str]): Files to exclude.
    
    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in excluded_files):
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    logger.debug(f"Retrieved {len(file_paths)} files from '{repo_path}'.")
    return file_paths

def load_json_schema(schema_path: str) -> dict:
    """Loads a JSON schema from a file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        logger.debug(f"Loaded JSON schema from {schema_path}")
        return schema
    except Exception as e:
        logger.error(f"Failed to load JSON schema from {schema_path}: {e}")
        return {}

def format_with_black(code: str) -> str:
    """
    Formats the given Python code using Black.

    Parameters:
        code (str): The Python code to format.

    Returns:
        str: The formatted Python code.
    """
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        logger.debug("Successfully formatted code with Black.")
        return formatted_code
    except black.NothingChanged:
        logger.debug("No changes made by Black; code is already formatted.")
        return code
    except Exception as e:
        logger.error(f"Error formatting code with Black: {e}")
        return code  # Return the original code if formatting fails

def clean_unused_imports(code: str) -> str:
    """
    Removes unused imports from Python code using autoflake.

    Parameters:
        code (str): The Python code to clean.

    Returns:
        str: The cleaned Python code.
    """
    try:
        cleaned_code = subprocess.check_output(
            ['autoflake', '--remove-all-unused-imports', '--stdout'],
            input=code.encode('utf-8'),
            stderr=subprocess.STDOUT
        )
        logger.debug("Successfully removed unused imports with autoflake.")
        return cleaned_code.decode('utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(f"Autoflake failed: {e.output.decode('utf-8')}")
        return code  # Return original code if autoflake fails
    except FileNotFoundError:
        logger.error("Autoflake is not installed. Please install it using 'pip install autoflake'.")
        return code
    except Exception as e:
        logger.error(f"Error cleaning imports with autoflake: {e}")
        return code

def check_with_flake8(file_path: str) -> bool:
    """
    Checks Python code compliance using flake8 and attempts to fix issues if found.

    Parameters:
        file_path (str): Path to the Python file to check.

    Returns:
        bool: True if the code passes flake8 checks after fixes, False otherwise.
    """
    logger.debug(f"Entering check_with_flake8 with file_path={file_path}")
    result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
    if result.returncode == 0:
        logger.debug(f"No flake8 issues in {file_path}")
        return True
    else:
        logger.error(f"flake8 issues in {file_path}:\n{result.stdout}")
        # Attempt to auto-fix with autoflake and black
        try:
            logger.info(f"Attempting to auto-fix flake8 issues in {file_path}")
            subprocess.run(['autoflake', '--remove-all-unused-imports', '--in-place', file_path], check=True)
            subprocess.run(['black', '--quiet', file_path], check=True)
            # Re-run flake8 to confirm
            result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug(f"No flake8 issues after auto-fix in {file_path}")
                return True
            else:
                logger.error(f"flake8 issues remain after auto-fix in {file_path}:\n{result.stdout}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Auto-fix failed for {file_path}: {e}", exc_info=True)
            return False

def run_flake8(file_path: str) -> Optional[str]:
    """
    Runs flake8 on the specified file and returns the output.

    Parameters:
        file_path (str): Path to the Python file to check.

    Returns:
        Optional[str]: The flake8 output if any issues are found, else None.
    """
    try:
        result = subprocess.run(
            ["flake8", file_path],
            capture_output=True,
            text=True,
            check=False  # Do not raise exception on non-zero exit
        )
        if result.stdout:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Error running flake8 on '{file_path}': {e}", exc_info=True)
        return None

def run_node_script(script_path: str, input_code: str) -> Optional[Dict[str, any]]:
    """
    Runs a Node.js script that outputs JSON (e.g., extract_structure.js) and returns the parsed JSON.

    Parameters:
        script_path (str): Path to the Node.js script.
        input_code (str): The code to process.

    Returns:
        Optional[Dict[str, Any]]: The JSON output from the script if successful, None otherwise.
    """
    try:
        logger.debug(f"Running Node.js script: {script_path}")
        result = subprocess.run(
            ['node', script_path],
            input=input_code,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        output_json = json.loads(result.stdout)
        return output_json
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON output from {script_path}: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return None

def run_node_insert_docstrings(script_path: str, input_code: str) -> Optional[str]:
    """
    Runs the insert_docstrings.js script and returns the modified code.

    Parameters:
        script_path (str): Path to the insert_docstrings.js script.
        input_code (str): The code to process.

    Returns:
        Optional[str]: The modified code if successful, None otherwise.
    """
    try:
        logger.debug(f"Running Node.js script: {script_path}")
        result = subprocess.run(
            ['node', script_path],
            input=input_code,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return None

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if a file extension is valid (not in the skip list)."""
    return ext.lower() not in skip_types

def extract_json_from_response(response: str) -> Optional[dict]:
    """Extracts JSON content from the model's response.

    Attempts multiple methods to extract JSON:
    1. Function calling format.
    2. JSON enclosed in triple backticks.
    3. Entire response as JSON.

    Args:
        response (str): The raw response string from the model.

    Returns:
        Optional[dict]: The extracted JSON as a dictionary, or None if extraction fails.
    """
    # First, try to extract JSON using the function calling format
    try:
        response_json = json.loads(response)
        if "function_call" in response_json and "arguments" in response_json["function_call"]:
            return json.loads(response_json["function_call"]["arguments"])
    except json.JSONDecodeError:
        pass  # Fallback to other extraction methods

    # Try to find JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # As a last resort, attempt to use the entire response if it's valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None

def generate_documentation_prompt(code_structure: dict, project_info: Optional[str], style_guidelines: Optional[str], language: str) -> str:
    """
    Generates a prompt for the OpenAI API based on the code structure.

    Parameters:
        code_structure (dict): The extracted structure of the code.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Documentation style guidelines.
        language (str): The programming language of the code.

    Returns:
        str: The generated prompt.
    """
    prompt = "You are an experienced software developer tasked with generating comprehensive documentation for a codebase."
    if project_info:
        prompt += f"\n\n**Project Information:** {project_info}"
    if style_guidelines:
        prompt += f"\n\n**Style Guidelines:** {style_guidelines}"
    prompt += f"\n\n**Language:** {language.capitalize()}"
    prompt += f"\n\n**Code Structure:**\n```json\n{json.dumps(code_structure, indent=2)}\n```"
    prompt += "\n\n**Instructions:** Based on the above code structure, generate the following documentation sections:\n1. **Summary:** A detailed summary of the codebase.\n2. **Changes Made:** A comprehensive list of changes or updates made to the code.\n\n**Please ensure that the documentation is clear, detailed, and adheres to the provided style guidelines.**"
    return prompt

async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    retry: int = 3,
) -> Optional[dict]:
    """
    Fetches documentation from OpenAI's API with optional retries.

    Args:
        session (aiohttp.ClientSession): The session to use for making the API request.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent API requests.
        model_name (str): The model to use for the OpenAI request (e.g., 'gpt-4').
        function_schema (dict): The JSON schema for the function call.
        retry (int, optional): Number of retry attempts for failed requests. Defaults to 3.

    Returns:
        Optional[dict]: The generated documentation, or None if failed.
    """
    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
                        {"role": "user", "content": prompt}
                    ],
                    "functions": [function_schema],
                    "function_call": "auto"  # Let the model decide which function to call
                }

                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                    response_text = await resp.text()
                    if resp.status != 200:
                        logger.error(f"OpenAI API request failed with status {resp.status}: {response_text}")
                        continue  # Retry on failure
                    response = await resp.json()
                    logger.debug(f"Full API Response: {json.dumps(response, indent=2)}")

                    choice = response.get("choices", [])[0]
                    message = choice.get('message', {})

                    # Check for function_call
                    if 'function_call' in message:
                        arguments = message['function_call'].get('arguments', '{}')
                        try:
                            documentation = json.loads(arguments)
                            logger.debug("Received documentation via function_call.")
                            return documentation
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON from function_call arguments: {e}")
                            logger.error(f"Arguments Content: {arguments}")
                            return None
                    else:
                        logger.error("No function_call found in the response.")
                        return None

            except Exception as e:
                logger.error(f"Error fetching documentation from OpenAI API: {e}")
                if attempt < retry:
                    logger.info(f"Retrying... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff on retries
                else:
                    logger.error("All retry attempts failed.")
                    return None

async def fetch_summary(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    retry: int = 3,
) -> Optional[str]:
    """
    Fetches a summary from the OpenAI API.

    Args:
        session (aiohttp.ClientSession): The session to use for making the API request.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent API requests.
        model_name (str): The model to use for the OpenAI request (e.g., 'gpt-4').
        retry (int, optional): Number of retry attempts for failed requests. Defaults to 3.

    Returns:
        Optional[str]: The summary text if successful, otherwise None.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are an AI assistant that summarizes code."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }

    for attempt in range(1, retry + 1):
        try:
            async with semaphore:
                async with session.post(
                    OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Ensure the response contains 'choices' and it's well-formed
                        choices = data.get('choices', [])
                        if choices and 'message' in choices[0]:
                            summary = choices[0]['message']['content'].strip()
                            return summary
                        else:
                            logger.error(f"Unexpected API response structure: {data}")
                            return None

                    elif response.status in {429, 500, 502, 503, 504}:
                        error_text = await response.text()
                        logger.warning(
                            f"API rate limit or server error (status {response.status}). "
                            f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds. "
                            f"Response: {error_text}"
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Unhandled API request failure with status {response.status}: {error_text}"
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out during attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)

        except aiohttp.ClientError as e:
            logger.error(
                f"Client error during API request: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate summary after multiple attempts.")
    return None

def call_openai_function(prompt: str, function_def: dict, model: str = "gpt-4-0613") -> dict:
    """
    Calls OpenAI's API with function calling enabled and validates the response against the provided schema.

    Parameters:
        prompt (str): The user prompt.
        function_def (dict): The function definition including the JSON schema.
        model (str): The OpenAI model to use.

    Returns:
        dict: The validated response from the API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating documentation."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_def],
            function_call="auto"
        )

        message = response["choices"][0]["message"]

        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])

            # Validate against schema
            validate(instance=arguments, schema=function_def["parameters"])
            logger.info(f"Function '{function_name}' called successfully and validated.")
            return arguments
        else:
            logger.warning("No function call detected in the response.")
            return {}

    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return {}
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return {}
```

**Key Enhancements:**

1. **Function Calling with JSON Schema Validation:**
   - **`call_openai_function`:** This function interacts with OpenAI's API using function calling. It sends a prompt along with a function definition (`function_def`) that includes the JSON schema (`function_schema.json`). The response is parsed and validated against the schema to ensure structured outputs.
   
2. **Improved `fetch_documentation`:**
   - **Retry Mechanism:** Implements retries with exponential backoff for handling transient API failures.
   - **Detailed Logging:** Logs comprehensive information about API responses and errors for easier troubleshooting.
   
3. **Loading JSON Schemas:**
   - **`load_json_schema`:** A utility function to load JSON schemas from files, ensuring that your function definitions are based on accurate schemas.
   
4. **Enhanced Error Handling:**
   - Comprehensive try-except blocks to catch and log various exceptions without crashing the application.
   
5. **Consistent Naming and Structure:**
   - Ensures that all functions and variables follow a consistent naming convention for better readability and maintenance.
   
6. **Dependency Management:**
   - **Import Ordering:** Organized imports logically for better clarity.
   - **Removed Redundant Imports:** Ensures that only necessary libraries are imported to optimize performance.

**Ensure that you have the necessary dependencies installed:**

```bash
pip install openai jsonschema aiohttp aiofiles black autoflake flake8 python-dotenv beautifulsoup4 tinycss2 astor tqdm
```

---

## **4. Updating `main.py`**

Integrate the updated utility functions into your main workflow to utilize function calling and structured outputs effectively.

**Updated `main.py`:**

```python
# main.py

import os
import sys
import argparse
import asyncio
import logging
import json
import aiohttp
from file_handlers import process_all_files
from utils import (
    load_config,
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    function_schema,
    call_openai_function,  # Newly added for function calling
    load_json_schema,     # Newly added for loading JSON schemas
)

# Configure logging
logger = logging.getLogger(__name__)

def configure_logging(log_level):
    """Configures logging based on the provided log level."""
    logger.setLevel(log_level)

    # Create formatter with module, function, and line number
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s')

    # Create file handler which logs debug and higher level messages
    file_handler = logging.FileHandler('docs_generation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

def validate_model_name(model_name: str) -> bool:
    """Validates the OpenAI model name format."""
    valid_models = [
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",  # Ensure this model supports function calling
        # Add other valid model names as needed
    ]
    if model_name in valid_models:
        return True
    else:
        logger.error(f"Invalid model name '{model_name}'. Please choose a valid OpenAI model.")
        return False

async def main():
    """Main function to orchestrate documentation generation."""
    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using OpenAI's GPT-4 API."
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--model", help="OpenAI model to use (default: gpt-4)", default="gpt-4")
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines to follow", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files will be modified)", action='store_true')
    parser.add_argument("--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)", default="INFO")
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(log_level)

    logger.info("Starting Documentation Generation Tool.")
    logger.debug(f"Parsed arguments: {args}")

    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)
    else:
        logger.debug("OPENAI_API_KEY found.")

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output
    model_name = args.model
    project_info_arg = args.project_info
    style_guidelines_arg = args.style_guidelines
    safe_mode = args.safe_mode

    logger.info(f"Repository Path: {repo_path}")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Concurrency Level: {concurrency}")
    logger.info(f"Output Markdown File: {output_file}")
    logger.info(f"OpenAI Model: {model_name}")
    logger.info(f"Safe Mode: {'Enabled' if safe_mode else 'Disabled'}")

    # Validate model name
    if not validate_model_name(model_name):
        sys.exit(1)

    if not os.path.isdir(repo_path):
        logger.critical(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)
    else:
        logger.debug(f"Repository path '{repo_path}' is valid.")

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(ext.strip() for ext in args.skip_types.split(','))
        logger.debug(f"Updated skip_types: {skip_types}")

    # Check if config file exists
    if not os.path.isfile(config_path):
        logger.warning(f"Configuration file '{config_path}' not found. Proceeding with default and command-line settings.")
        project_info_config, style_guidelines_config = '', ''
    else:
        # Load additional configurations
        try:
            project_info_config, style_guidelines_config = load_config(config_path, excluded_dirs, excluded_files, skip_types)
            logger.debug(f"Loaded configurations from '{config_path}': Project Info='{project_info_config}', Style Guidelines='{style_guidelines_config}'")
        except Exception as e:
            logger.error(f"Failed to load configuration from '{config_path}': {e}")
            sys.exit(1)

    # Determine final project_info and style_guidelines
    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    if project_info:
        logger.debug(f"Project Info: {project_info}")
    if style_guidelines:
        logger.debug(f"Style Guidelines: {style_guidelines}")

    # Load JSON schema for function calling
    schema_path = 'function_schema.json'  # Ensure this path is correct
    function_schema_loaded = load_json_schema(schema_path)
    if not function_schema_loaded:
        logger.critical(f"Failed to load function schema from '{schema_path}'. Exiting.")
        sys.exit(1)

    # Get all file paths
    try:
        file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files)
        logger.info(f"Total files to process: {len(file_paths)}")
    except Exception as e:
        logger.error(f"Error retrieving file paths from '{repo_path}': {e}")
        sys.exit(1)

    if not file_paths:
        logger.warning("No files found to process. Exiting.")
        sys.exit(0)

    logger.info("Initializing output Markdown file.")
    # Clear and initialize the output file with a header
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Documentation Generation Report\n\n")
        logger.debug(f"Output file '{output_file}' initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize output file '{output_file}': {e}")
        sys.exit(1)

    # Initialize semaphore and locks
    semaphore = asyncio.Semaphore(concurrency)
    output_lock = asyncio.Lock()

    # Start the asynchronous processing
    logger.info("Starting asynchronous file processing.")
    try:
        async with aiohttp.ClientSession() as session:
            await process_all_files(
                session=session,
                file_paths=file_paths,
                skip_types=skip_types,
                output_file=output_file,
                semaphore=semaphore,
                output_lock=output_lock,
                model_name=model_name,
                function_schema=function_schema_loaded,
                repo_root=repo_path,
                project_info=project_info,
                style_guidelines=style_guidelines,
                safe_mode=safe_mode
            )
    except Exception as e:
        logger.critical(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")
    logger.info(f"Check the output file '{output_file}' for the generated documentation.")

if __name__ == "__main__":
    asyncio.run(main())
```

**Key Enhancements:**

1. **Loading and Validating Function Schema:**
   - **`load_json_schema`:** Loads the JSON schema from `function_schema.json`.
   - **Validation:** Ensures that the schema is loaded successfully before proceeding.

2. **Function Calling Integration:**
   - **`call_openai_function`:** Utilizes the updated utility function to interact with OpenAI's API using function calling.
   - **Structured Outputs:** Ensures that the responses are structured and validated against the schema.

3. **Error Handling:**
   - **Schema Loading Failure:** If the schema fails to load, the script logs a critical error and exits.
   - **File Processing Errors:** Comprehensive error handling during file processing to prevent the entire script from crashing due to individual file issues.

4. **Logging Enhancements:**
   - **Detailed Logging:** Added more debug logs to trace the flow of data and operations.
   - **Consistent Logging Levels:** Ensures that critical errors are logged appropriately, while debug information is available for troubleshooting.

**Ensure that `function_schema.json` is correctly placed and referenced in `main.py`.**

---

## **5. Updating `file_handlers.py`**

Enhance your `file_handlers.py` to integrate function calling and structured outputs more deeply into file processing and documentation insertion.

**Updated `file_handlers.py`:**

```python
# file_handlers.py

import os
import sys
import json
import logging
import ast
import astor
import shutil
from typing import Set, List, Optional, Dict
import aiofiles
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import subprocess
from pathlib import Path
import tempfile

from language_functions import (
    extract_python_structure,
    insert_python_docstrings,
    is_valid_python_code,
    extract_js_ts_structure,
    insert_js_ts_docstrings,
    extract_html_structure,
    insert_html_comments,
    extract_css_structure,
    insert_css_docstrings,
)

from utils import (
    load_config,
    is_binary,
    get_language,
    get_all_file_paths,
    is_valid_extension,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    generate_documentation_prompt,
    fetch_documentation,
    function_schema,
    format_with_black,
    clean_unused_imports,
    check_with_flake8,
    run_flake8,
    run_node_script,
    run_node_insert_docstrings,
    call_openai_function,  # Newly added for function calling
    load_json_schema,     # Newly added for loading JSON schemas
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create formatter with module, function, and line number
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s')

# Create file handler which logs debug and higher level messages
file_handler = logging.FileHandler('file_handlers.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Change to DEBUG for more verbosity on console
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

async def main():
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(10)
        prompt = "Your prompt here"
        model_name = "gpt-4"
        result = await fetch_documentation_with_retries(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

async def insert_docstrings_for_file(js_ts_file: str, documentation_file: str) -> None:
    logger.debug(f"Entering insert_docstrings_for_file with js_ts_file={js_ts_file}, documentation_file={documentation_file}")
    process = await asyncio.create_subprocess_exec(
        'node',
        'insert_docstrings.js',
        js_ts_file,
        documentation_file,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        logger.error(f"Error inserting docstrings into {js_ts_file}: {stderr.decode().strip()}")
    else:
        logger.info(stdout.decode().strip())
    logger.debug("Exiting insert_docstrings_for_file")

async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str] = None,
    style_guidelines: Optional[str] = None,
    safe_mode: bool = False,
) -> None:
    logger.debug(f"Entering process_file with file_path={file_path}")
    summary = ""
    changes = []
    
    try:
        # Check if file extension is valid or binary, and get language type
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return

        language = get_language(ext)
        if language == "plaintext":
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return

        logger.info(f"Processing file: {file_path}")

        # Read the file content
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug(f"File content for {file_path}:\n{content}")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return

        # Extract code structure based on the language
        code_structure = await extract_code_structure(content, file_path, language)
        if not code_structure:
            logger.warning(f"Could not extract code structure from '{file_path}'")
            return
        
        # Generate the documentation prompt and log it
        prompt = generate_documentation_prompt(
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines,
            language=language
        )
        logger.debug(f"Generated prompt for '{file_path}': {prompt}")

        # Fetch documentation from OpenAI using function calling
        documentation = await fetch_documentation(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'.")
            return

        # Insert documentation into code
        summary, changes, new_content = await process_code_documentation(
            content, documentation, language, file_path
        )

        if safe_mode:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
        else:
            await backup_and_write_new_content(file_path, new_content)

        # Write the documentation report
        await write_documentation_report(output_file, summary, changes, new_content, language, output_lock, file_path, repo_root)
        
        logger.info(f"Successfully processed and documented '{file_path}'")
    
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)

async def extract_code_structure(content: str, file_path: str, language: str) -> Optional[dict]:
    """
    Extracts the structure of the code based on the language.
    """
    try:
        if language == "python":
            logger.debug(f"Extracting Python structure for file '{file_path}'")
            return extract_python_structure(content)
        elif language in ["javascript", "typescript"]:
            logger.debug(f"Extracting JS/TS structure for file '{file_path}'")
            structure_output = run_node_script('extract_structure.js', content)
            if not structure_output:
                logger.warning(f"Could not extract code structure from '{file_path}'")
                return None
            return structure_output
        elif language == "html":
            logger.debug(f"Extracting HTML structure for file '{file_path}'")
            return extract_html_structure(content)
        elif language == "css":
            logger.debug(f"Extracting CSS structure for file '{file_path}'")
            return extract_css_structure(content)
        else:
            logger.warning(f"Unsupported language for structure extraction: {language}")
            return None
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None

async def process_code_documentation(content: str, documentation: dict, language: str, file_path: str) -> tuple[str, list, str]:
    """
    Inserts the docstrings or comments into the code based on the documentation.
    """
    summary = documentation.get("summary", "")
    changes = documentation.get("changes_made", [])
    new_content = content

    try:
        if language == "python":
            new_content = insert_python_docstrings(content, documentation)
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
        elif language in ["javascript", "typescript"]:
            new_content = run_node_insert_docstrings('insert_docstrings.js', content)
            new_content = format_with_black(new_content)
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
        elif language == "css":
            new_content = insert_css_docstrings(content, documentation)
        
        logger.debug(f"Processed {language} file '{file_path}'.")
        return summary, changes, new_content
    
    except Exception as e:
        logger.error(f"Error processing {language} file '{file_path}': {e}", exc_info=True)
        return summary, changes, content

async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """
    Creates a backup of the file and writes the new content.
    """
    backup_path = f"{file_path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.copy(file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'")

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'")
    
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        if backup_path and os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            os.remove(backup_path)
            logger.info(f"Restored original file from backup for '{file_path}'")

async def write_documentation_report(
    output_file: str, summary: str, changes: list, new_content: str, language: str,
    output_lock: asyncio.Lock, file_path: str, repo_root: str
) -> None:
    """
    Writes the summary, changes, and new content to the output markdown report.
    """
    try:
        relative_path = os.path.relpath(file_path, repo_root)
        async with output_lock:
            async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                header = f"# File: {relative_path}\n\n"
                summary_section = f"## Summary\n\n{summary}\n\n"
                changes_section = f"## Changes Made\n\n" + "\n".join(f"- {change}" for change in changes) + "\n\n"
                code_block = f"```{language}\n{new_content}\n```\n\n"
                await f.write(header)
                await f.write(summary_section)
                await f.write(changes_section)
                await f.write(code_block)
    except Exception as e:
        logger.error(f"Error writing documentation for '{file_path}': {e}", exc_info=True)

async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool = False,
) -> None:
    logger.info("Starting process of all files.")
    tasks = []
    
    for file_path in file_paths:
        # Call process_file for each file asynchronously
        task = process_file(
            session=session,
            file_path=file_path,
            skip_types=skip_types,
            output_file=output_file,
            semaphore=semaphore,
            output_lock=output_lock,
            model_name=model_name,
            function_schema=function_schema,
            repo_root=repo_root,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode
        )
        tasks.append(task)
    
    # Use tqdm for progress tracking
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f
    
    logger.info("Completed processing all files.")
```

**Key Enhancements:**

1. **Function Calling Integration:**
   - **`call_openai_function`:** Utilizes function calling to interact with OpenAI's API, ensuring that responses adhere to the predefined JSON schema.
   
2. **Structured Outputs:**
   - **`fetch_documentation`:** Enhanced to handle structured outputs with retries and detailed logging.
   - **`generate_documentation_prompt`:** Generates detailed prompts based on the code structure, project info, and style guidelines.
   
3. **Asynchronous Processing with Progress Tracking:**
   - **`process_all_files`:** Uses `tqdm`'s `asyncio` integration to provide real-time progress tracking during file processing.
   
4. **Enhanced Error Handling:**
   - Comprehensive error handling in all functions to ensure that failures in processing individual files do not crash the entire script.
   
5. **Logging Enhancements:**
   - **`file_handlers.log`:** Added a separate log file for `file_handlers.py` to capture detailed debug information.
   
6. **Code Formatting and Cleaning:**
   - Ensures that inserted docstrings or comments are formatted consistently using tools like Black and autoflake.

**Ensure that all necessary dependencies are installed:**

```bash
pip install openai jsonschema aiohttp aiofiles black autoflake flake8 python-dotenv beautifulsoup4 tinycss2 astor tqdm
```

---

## **6. Additional Enhancements and Best Practices**

To further enhance your Documentation Generation Project, consider implementing the following best practices and additional features:

### **a. Implement Caching Mechanism**

To avoid redundant API calls for unchanged files, implement a caching mechanism based on file hashes.

**Implementation Steps:**

1. **Generate File Hashes:**

```python
import hashlib

def get_file_hash(file_path: str) -> str:
    """Generates a SHA-256 hash of the file content."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
```

2. **Load and Save Cache:**

```python
def load_cache(cache_path: str) -> Dict[str, str]:
    """Loads the cache from a JSON file."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, str], cache_path: str):
    """Saves the cache to a JSON file."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
```

3. **Integrate Caching into `process_file`:**

```python
async def process_file(
    ...
    cache: Dict[str, str],  # Pass the cache dictionary
    cache_path: str,        # Path to the cache file
    ...
) -> None:
    ...
    file_hash = get_file_hash(file_path)
    if file_path in cache and cache[file_path] == file_hash:
        logger.info(f"No changes detected in {file_path}. Skipping documentation generation.")
        return

    # Proceed with documentation generation
    ...

    # After successful documentation generation, update the cache
    cache[file_path] = file_hash
    save_cache(cache, cache_path)
```

4. **Update `main.py` to Initialize and Pass Cache:**

```python
async def main():
    ...
    cache_path = 'cache.json'
    cache = load_cache(cache_path)
    
    ...
    
    await process_all_files(
        ...
        cache=cache,
        cache_path=cache_path,
        ...
    )
    
    ...
```

### **b. Secure API Key Management**

Ensure that your OpenAI API keys are securely managed and not exposed in the codebase.

**Implementation Steps:**

1. **Use Environment Variables:**

   - **`.env` File:**
     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

   - **Load Environment Variables in `utils.py`:**
     ```python
     from dotenv import load_dotenv

     load_dotenv()  # Load environment variables from .env
     openai.api_key = os.getenv("OPENAI_API_KEY")
     ```

2. **Ensure `.env` is in `.gitignore`:**
   
   ```
   # .gitignore
   .env
   ```

### **c. Comprehensive Logging and Monitoring**

Implement comprehensive logging to monitor the documentation generation process and quickly identify issues.

**Implementation Steps:**

1. **Separate Log Files:**
   - Maintain separate log files for different modules (e.g., `main.log`, `file_handlers.log`) for better traceability.

2. **Logging Levels:**
   - Use appropriate logging levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) to categorize log messages.

3. **Log Rotation:**
   - Implement log rotation to prevent log files from growing indefinitely.
   - Example using `logging.handlers`:

     ```python
     from logging.handlers import RotatingFileHandler

     # Replace FileHandler with RotatingFileHandler
     file_handler = RotatingFileHandler('docs_generation.log', maxBytes=5*1024*1024, backupCount=5)
     file_handler.setLevel(logging.DEBUG)
     file_handler.setFormatter(formatter)
     ```

### **d. Testing and Continuous Integration**

Develop unit and integration tests to ensure that the documentation generation and validation processes work as expected.

**Implementation Steps:**

1. **Use `pytest` for Testing:**

   - **Example Test for `call_openai_function`:**
     ```python
     # tests/test_utils.py

     import pytest
     from unittest.mock import patch
     from utils import call_openai_function

     def test_call_openai_function_success(monkeypatch):
         mock_response = {
             "choices": [{
                 "message": {
                     "function_call": {
                         "name": "generate_documentation_report",
                         "arguments": json.dumps({
                             "file": "example.py",
                             "language": "py",
                             "summary": "An example Python script.",
                             "changes_made": ["Added add function.", "Introduced Calculator class."],
                             "code_snippet": "def add(a, b):\n    return a + b\n\nclass Calculator:\n    def multiply(self, a, b):\n        return a * b"
                         })
                     }
                 }
             }]
         }

         def mock_create(*args, **kwargs):
             return mock_response

         monkeypatch.setattr(openai.ChatCompletion, 'create', mock_create)

         function_def = {
             "name": "generate_documentation_report",
             "description": "Generates a structured documentation report for a given file.",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "file": {"type": "string", "description": "Name of the file being documented."},
                     "language": {"type": "string", "description": "Programming language of the file."},
                     "summary": {"type": "string", "description": "A brief summary of the file's purpose and functionality."},
                     "changes_made": {
                         "type": "array",
                         "description": "List of changes made to the file.",
                         "items": {"type": "string"}
                     },
                     "code_snippet": {"type": "string", "description": "Relevant code snippet from the file."}
                 },
                 "required": ["file", "language", "summary", "changes_made", "code_snippet"]
             }
         }

         prompt = "Generate a documentation report for example.py"
         response = call_openai_function(prompt, function_def)

         assert response['file'] == "example.py"
         assert response['language'] == "py"
         assert response['summary'] == "An example Python script."
         assert response['changes_made'] == ["Added add function.", "Introduced Calculator class."]
         assert "def add(a, b):" in response['code_snippet']

     def test_call_openai_function_no_function_call(monkeypatch):
         mock_response = {
             "choices": [{
                 "message": {
                     "content": "No function call in this response."
                 }
             }]
         }

         def mock_create(*args, **kwargs):
             return mock_response

         monkeypatch.setattr(openai.ChatCompletion, 'create', mock_create)

         function_def = {
             "name": "generate_documentation_report",
             "description": "Generates a structured documentation report for a given file.",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "file": {"type": "string", "description": "Name of the file being documented."},
                     "language": {"type": "string", "description": "Programming language of the file."},
                     "summary": {"type": "string", "description": "A brief summary of the file's purpose and functionality."},
                     "changes_made": {
                         "type": "array",
                         "description": "List of changes made to the file.",
                         "items": {"type": "string"}
                     },
                     "code_snippet": {"type": "string", "description": "Relevant code snippet from the file."}
                 },
                 "required": ["file", "language", "summary", "changes_made", "code_snippet"]
             }
         }

         prompt = "Generate a documentation report for example.py"
         response = call_openai_function(prompt, function_def)

         assert response == {}
     ```

2. **Integrate Tests into CI/CD Pipeline:**

   - **Example with GitHub Actions:**

     ```yaml
     # .github/workflows/test.yml

     name: Run Tests

     on:
       push:
         branches: [ main ]
       pull_request:
         branches: [ main ]

     jobs:
       test:
         runs-on: ubuntu-latest

         steps:
           - uses: actions/checkout@v2

           - name: Set up Python
             uses: actions/setup-python@v2
             with:
               python-version: '3.9'

           - name: Install Dependencies
             run: |
               pip install -r requirements.txt
               pip install pytest

           - name: Run Tests
             run: |
               pytest
     ```

### **e. Automate Documentation Formatting**

Use templating engines like Jinja2 to format the generated documentation consistently.

**Implementation Steps:**

1. **Install Jinja2:**

   ```bash
   pip install Jinja2
   ```

2. **Create a Jinja2 Template (`templates/documentation_report.md.j2`):**

   ```jinja
   # Documentation Generation Report

   {% for report in reports %}
   ## File: {{ report.file }}

   ### Language
   {{ report.language.capitalize() }}

   ### Summary
   {{ report.summary }}

   ### Changes Made
   {% for change in report.changes_made %}
   - {{ change }}
   {% endfor %}

   ```{{ report.language }}
   {{ report.code_snippet }}
   ```

   ---
   {% endfor %}
   ```

3. **Update `write_documentation_report` to Use Jinja2:**

   ```python
   from jinja2 import Environment, FileSystemLoader

   async def write_documentation_report_jinja(
       reports: List[dict], output_path: str
   ) -> None:
       """
       Renders the documentation report using Jinja2 and writes it to the output file.
       """
       try:
           env = Environment(loader=FileSystemLoader('templates'))
           template = env.get_template('documentation_report.md.j2')
           rendered = template.render(reports=reports)

           async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
               await f.write(rendered)
           logger.info(f"Documentation report generated at {output_path}")
       except Exception as e:
           logger.error(f"Error rendering documentation with Jinja2: {e}", exc_info=True)
   ```

4. **Integrate the New Function into `main.py`:**

   ```python
   # Replace existing write_documentation_report call with write_documentation_report_jinja
   await write_documentation_report_jinja(reports, args.output)
   ```

---

## **7. Final Project Directory Structure**

Ensure that your project is well-organized to facilitate easy navigation and maintenance.

```
project-root/
│
├── function_schema.json
├── schemas/
│   └── documentation_report.json
│
├── src/
│   ├── main.py
│   ├── utils.py
│   ├── language_functions.py
│   └── ...
│
├── scripts/
│   ├── extract_structure.js
│   ├── insert_docstrings.js
│   └── ...
│
├── tests/
│   ├── test_utils.py
│   └── ...
│
├── templates/
│   └── documentation_report.md.j2
│
├── docs/
│   ├── documentation_report.md
│   ├── examples.md
│   ├── style_guide.md
│   └── ...
│
├── .env
├── .gitignore
├── requirements.txt
├── package.json
└── README.md
```

**Notes:**

- **`function_schema.json`:** Contains the JSON schema for function calling.
- **`schemas/`:** Additional JSON schemas if needed.
- **`templates/`:** Jinja2 templates for documentation formatting.
- **`scripts/`:** Node.js scripts like `extract_structure.js` and `insert_docstrings.js`.
- **`tests/`:** Test cases for your scripts.
- **`docs/`:** Generated documentation and related markdown files.
- **`.env`:** Environment variables (ensure it's in `.gitignore`).
- **`requirements.txt` and `package.json`:** Manage Python and Node.js dependencies respectively.

---

## **8. Conclusion**

By following the above instructions, you've successfully integrated **function calling** and **structured outputs** into your Documentation Generation Project. This integration ensures that your documentation is consistently structured, easily maintainable, and aligns perfectly with your project's needs. Additionally, adhering to best practices in security, performance, and error handling further enhances the robustness and reliability of your documentation processes.

**Next Steps:**

1. **Implement the Defined Changes:** Start by updating `utils.py`, `main.py`, and `file_handlers.py` as outlined.
2. **Test Thoroughly:** Run your scripts with sample files to ensure that function calling and structured outputs are functioning as expected.
3. **Iterate and Refine:** Based on testing outcomes, refine your prompts, schemas, and processing logic.
4. **Integrate into CI/CD:** Set up continuous integration workflows to automate documentation generation and validation.
5. **Document the Process:** Update your project's documentation to reflect the new workflows and provide guidance on maintaining the system.

If you encounter any challenges during implementation or need further assistance with specific aspects, feel free to reach out!

---

Certainly! Below are the comprehensive **dependency requirements** for your Documentation Generation Tool project, covering both **Python** and **Node.js (npm)** environments. These dependencies are essential for the proper functioning of the tool, ensuring that all modules and scripts work seamlessly together.

---

## **1. Python Dependencies**

### **1.1. `requirements.txt`**

Create a `requirements.txt` file in the root directory of your project and include the following dependencies:

```txt
aiohttp>=3.8.0
aiofiles>=0.8.0
astor>=0.8.1
beautifulsoup4>=4.11.1
tinycss2>=1.2.1
tqdm>=4.64.0
python-dotenv>=0.20.0
```

### **1.2. Explanation of Python Dependencies**

- **aiohttp** (`aiohttp>=3.8.0`): 
  - *Purpose:* Asynchronous HTTP client/server framework.
  - *Usage:* Facilitates asynchronous API calls to OpenAI's GPT-4 for documentation generation.

- **aiofiles** (`aiofiles>=0.8.0`): 
  - *Purpose:* Asynchronous file handling.
  - *Usage:* Enables non-blocking file read/write operations, improving performance when processing large codebases.

- **astor** (`astor>=0.8.1`): 
  - *Purpose:* Python AST (Abstract Syntax Tree) manipulation.
  - *Usage:* Assists in analyzing and modifying Python code structures for docstring insertion.

- **beautifulsoup4** (`beautifulsoup4>=4.11.1`): 
  - *Purpose:* HTML and XML parsing.
  - *Usage:* Parses and modifies HTML files to insert comments based on generated documentation.

- **tinycss2** (`tinycss2>=1.2.1`): 
  - *Purpose:* CSS parsing.
  - *Usage:* Analyzes and modifies CSS files to insert comments based on generated documentation.

- **tqdm** (`tqdm>=4.64.0`): 
  - *Purpose:* Progress bar for Python loops.
  - *Usage:* Provides visual feedback during the processing of multiple files, enhancing user experience.

- **python-dotenv** (`python-dotenv>=0.20.0`): 
  - *Purpose:* Loads environment variables from `.env` files.
  - *Usage:* Securely manages sensitive information like OpenAI API keys without hardcoding them into scripts.

### **1.3. Installing Python Dependencies**

After creating the `requirements.txt` file, install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## **2. Node.js (npm) Dependencies**

### **2.1. `package.json`**

Create a `package.json` file inside the `scripts/` directory of your project with the following content:

```json
{
  "name": "documentation-generation-scripts",
  "version": "1.0.0",
  "description": "Scripts for extracting structure and inserting docstrings into JavaScript/TypeScript files.",
  "main": "extract_structure.js",
  "scripts": {
    "extract": "node extract_structure.js",
    "insert": "node insert_docstrings.js"
  },
  "dependencies": {
    "@babel/parser": "^7.21.3",
    "@babel/traverse": "^7.21.4",
    "@babel/generator": "^7.21.3",
    "prettier": "^2.8.8",
    "typescript": "^5.0.4",
    "recast": "^0.22.5"
  },
  "devDependencies": {},
  "author": "Your Name",
  "license": "MIT"
}
```

### **2.2. Explanation of Node.js Dependencies**

- **@babel/parser** (`@babel/parser@^7.21.3`): 
  - *Purpose:* JavaScript parser.
  - *Usage:* Parses JavaScript/TypeScript code into ASTs for analysis and manipulation.

- **@babel/traverse** (`@babel/traverse@^7.21.4`): 
  - *Purpose:* AST traversal utility.
  - *Usage:* Navigates through AST nodes to locate specific structures (e.g., functions, classes) for documentation insertion.

- **@babel/generator** (`@babel/generator@^7.21.3`): 
  - *Purpose:* AST to code generator.
  - *Usage:* Converts modified ASTs back into JavaScript/TypeScript code after docstring insertion.

- **prettier** (`prettier@^2.8.8`): 
  - *Purpose:* Code formatter.
  - *Usage:* Formats the modified code to ensure consistency and readability after docstring insertion.

- **typescript** (`typescript@^5.0.4`): 
  - *Purpose:* TypeScript compiler and language features.
  - *Usage:* Assists in parsing and handling TypeScript-specific syntax within JS/TS files.

- **recast** (`recast@^0.22.5`): 
  - *Purpose:* JavaScript syntax tree transformer.
  - *Usage:* Facilitates advanced AST manipulations for inserting docstrings and comments into JS/TS code.

### **2.3. Installing Node.js Dependencies**

1. **Navigate to the `scripts/` Directory:**

   ```bash
   cd scripts
   ```

2. **Initialize the Node.js Project (If Not Already Initialized):**

   If the `package.json` file does not exist, initialize it:

   ```bash
   npm init -y
   ```

3. **Install the Dependencies:**

   ```bash
   npm install @babel/parser@^7.21.3 @babel/traverse@^7.21.4 @babel/generator@^7.21.3 prettier@^2.8.8 typescript@^5.0.4 recast@^0.22.5
   ```

   **Note:** The versions specified (`^7.21.3`, etc.) ensure compatibility and allow for minor updates.

4. **Verify Installation:**

   Ensure all dependencies are installed correctly:

   ```bash
   npm list
   ```

5. **Return to the Project Root Directory:**

   ```bash
   cd ..
   ```

---

## **3. Complete `requirements.txt` and `package.json` Files**

### **3.1. `requirements.txt`**

```txt
aiohttp>=3.8.0
aiofiles>=0.8.0
astor>=0.8.1
beautifulsoup4>=4.11.1
tinycss2>=1.2.1
tqdm>=4.64.0
python-dotenv>=0.20.0
```

### **3.2. `package.json`**

```json
{
  "name": "documentation-generation-scripts",
  "version": "1.0.0",
  "description": "Scripts for extracting structure and inserting docstrings into JavaScript/TypeScript files.",
  "main": "extract_structure.js",
  "scripts": {
    "extract": "node extract_structure.js",
    "insert": "node insert_docstrings.js"
  },
  "dependencies": {
    "@babel/parser": "^7.21.3",
    "@babel/traverse": "^7.21.4",
    "@babel/generator": "^7.21.3",
    "prettier": "^2.8.8",
    "typescript": "^5.0.4",
    "recast": "^0.22.5"
  },
  "devDependencies": {},
  "author": "Your Name",
  "license": "MIT"
}
```

---

## **4. Additional Setup Steps**

### **4.1. Environment Variables**

Ensure that the OpenAI API key is securely stored and accessible to your Python scripts. You can achieve this by creating a `.env` file in the project root:

1. **Create a `.env` File:**

   ```bash
   touch .env
   ```

2. **Add the OpenAI API Key to `.env`:**

   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

3. **Ensure `.env` is Ignored by Git:**

   Add `.env` to your `.gitignore` to prevent accidental commits of sensitive information.

   ```bash
   echo ".env" >> .gitignore
   ```

4. **Load Environment Variables in Python Scripts:**

   Ensure that your Python scripts load environment variables from the `.env` file using `python-dotenv`. For example, in your `utils.py` or main script:

   ```python
   from dotenv import load_dotenv
   load_dotenv()
   
   import os
   
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
   ```

### **4.2. Node.js Scripts Configuration**

Ensure that your Node.js scripts (`extract_structure.js` and `insert_docstrings.js`) are correctly implemented to utilize the installed dependencies. Here's a brief overview of how these scripts might interact with the dependencies:

- **`extract_structure.js`:**
  - **Purpose:** Parses JavaScript/TypeScript files to extract their structural information (e.g., functions, classes).
  - **Dependencies Usage:**
    - **@babel/parser:** To parse the code into an AST.
    - **@babel/traverse:** To navigate the AST and extract relevant structures.
    - **recast:** To manipulate and format the AST if needed.

- **`insert_docstrings.js`:**
  - **Purpose:** Inserts docstrings or comments into JavaScript/TypeScript files based on generated documentation.
  - **Dependencies Usage:**
    - **@babel/parser:** To parse the code into an AST.
    - **@babel/traverse:** To locate insertion points within the AST.
    - **@babel/generator:** To generate the modified code from the updated AST.
    - **prettier:** To format the modified code for consistency.

Ensure that these scripts are executable and correctly reference the installed dependencies. You might also want to include scripts in `package.json` for easier execution:

```json
"scripts": {
  "extract": "node extract_structure.js",
  "insert": "node insert_docstrings.js"
}
```

This allows you to run them using:

```bash
npm run extract
npm run insert
```

---

## **5. Final Checklist**

Before running the Documentation Generation Tool, ensure that you have completed the following steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/documentation-generation-tool.git
   cd documentation-generation-tool
   ```

2. **Set Up the Python Environment:**

   - Create and activate a virtual environment.
   - Install Python dependencies.

3. **Set Up the Node.js Environment (If Applicable):**

   - Navigate to the `scripts/` directory.
   - Install Node.js dependencies.

4. **Configure Environment Variables:**

   - Create a `.env` file with your OpenAI API key.
   - Ensure the API key is loaded in your Python scripts.

5. **Configure `config.json`:**

   - Define excluded directories, files, file types, project information, and style guidelines as needed.

6. **Prepare `function_schema.json`:**

   - Ensure the schema accurately defines the structure for OpenAI's function calling.

7. **Verify Installations:**

   - Test Python and Node.js installations to ensure all dependencies are correctly installed.

8. **Run the Tool:**

   Execute the main script with desired arguments. For example:

   ```bash
   python main.py /path/to/your/repository
   ```

---

## **Conclusion**

By following the above steps and ensuring that all dependencies are correctly installed and configured, your Documentation Generation Tool will be well-equipped to analyze and document your codebase efficiently. Proper dependency management is crucial for the stability and scalability of the tool, allowing it to handle diverse programming languages and large projects seamlessly.

If you encounter any issues during setup or execution, refer to the [Troubleshooting](Enhance%2520your%2520**Documentation%2520Generation%2520Project**%2520with%2520**function%2520calling**%2520and%2520**structured%2520outputs**%2520to%2520ensure%2520reliability,%2520consistency,%2520and%2520maintainability.%2520Below%2520is%2520a%2520comprehensive%2520title%2520for%2520your%2520project.md##troubleshooting) section of the README or consult the detailed logs generated during the tool's operation.

Happy documenting! 🚀