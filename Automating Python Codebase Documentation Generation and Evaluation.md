```python
import os
import sys
import json
import logging
import shutil
import datetime

# Import functions from individual scripts
from extract import process_directory as extract_ast_info
from evaluate_and_generate_docstrings import evaluate_and_generate_docstrings
from calculate_complexity_and_insert_docstrings import process_files as insert_docstrings_and_calculate_complexity
from compile_documentation import generate_markdown_documentation

def generate_documentation_pipeline(
    codebase_dir,
    output_dir="documentation",
    schemas_file="output_schemas.json",
    markdown_file="documentation.md",
    backup=True
):
    """
    Orchestrates the entire documentation generation workflow.

    Parameters:
        codebase_dir (str): Path to the root directory of the codebase to analyze.
        output_dir (str): Directory where the documentation will be saved.
        schemas_file (str): Filename for the intermediate JSON schemas.
        markdown_file (str): Filename for the generated Markdown documentation.
        backup (bool): Whether to create a backup of the source code before making changes.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("documentation_generation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    start_time = datetime.datetime.now()
    logging.info("Starting documentation generation pipeline.")

    try:
        if backup:
            # Create a backup of the source code
            backup_dir = f"{codebase_dir}_backup_{start_time.strftime('%Y%m%d%H%M%S')}"
            shutil.copytree(codebase_dir, backup_dir)
            logging.info(f"Backup of source code created at: {backup_dir}")

        # Step 1: AST Information Extraction
        logging.info("Extracting AST information from the codebase...")
        schemas = extract_ast_info(codebase_dir)
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(schemas, f, indent=4)
        logging.info(f"AST information saved to: {schemas_file}")

        # Step 2: Docstring Evaluation and Generation
        logging.info("Evaluating existing docstrings and generating new ones if necessary...")
        updated_schemas = evaluate_and_generate_docstrings(schemas)
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(updated_schemas, f, indent=4)
        logging.info("Docstrings updated in schemas.")

        # Step 3: Complexity Calculation and Docstring Insertion
        logging.info("Calculating complexity scores and inserting docstrings into the source code...")
        insert_docstrings_and_calculate_complexity(updated_schemas, codebase_dir)
        logging.info("Complexity scores calculated and docstrings inserted.")

        # Step 4: Markdown Documentation Compilation
        logging.info("Compiling Markdown documentation...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_markdown_path = os.path.join(output_dir, markdown_file)
        generate_markdown_documentation(updated_schemas, codebase_dir, output_file=output_markdown_path)
        logging.info(f"Markdown documentation generated at: {output_markdown_path}")

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logging.info(f"Documentation generation pipeline completed successfully in {elapsed_time:.2f} seconds.")

    except Exception as e:
        logging.exception("An error occurred during the documentation generation pipeline.")
        if backup and os.path.exists(backup_dir):
            # Restore from backup
            logging.info("Restoring source code from backup...")
            if os.path.exists(codebase_dir):
                shutil.rmtree(codebase_dir)
            shutil.copytree(backup_dir, codebase_dir)
            logging.info("Source code restored from backup.")
        sys.exit(1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Automate the documentation generation process for a Python codebase.")
    parser.add_argument('codebase_dir', help='Path to the root directory of the codebase to analyze.')
    parser.add_argument('--output_dir', default='documentation', help='Directory where the documentation will be saved.')
    parser.add_argument('--schemas_file', default='output_schemas.json', help='Filename for the intermediate JSON schemas.')
    parser.add_argument('--markdown_file', default='documentation.md', help='Filename for the generated Markdown documentation.')
    parser.add_argument('--no-backup', action='store_true', help='Do not create a backup of the source code.')
    args = parser.parse_args()

    generate_documentation_pipeline(
        codebase_dir=args.codebase_dir,
        output_dir=args.output_dir,
        schemas_file=args.schemas_file,
        markdown_file=args.markdown_file,
        backup=not args.no_backup
    )

```

```python
import os
import json
import datetime


def generate_markdown_documentation(schemas, codebase_dir, output_file="documentation.md"):
    """
    Compiles a Markdown document from the provided schemas and codebase.

    Parameters:
        schemas (list): List of dictionaries containing extracted information about the codebase.
        codebase_dir (str): Path to the root directory of the codebase.
        output_file (str): Path to the output Markdown file.
    """
    markdown_content = ""

    # --- Summary Section ---
    markdown_content += "# Codebase Documentation\n\n"
    markdown_content += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += (
        "This document provides a comprehensive overview of the Python codebase, "
        "including a summary, changelog, glossary of functions and classes, "
        "and the updated source code with embedded docstrings and complexity scores.\n\n"
    )

    # --- Changelog Section (Placeholder) ---
    markdown_content += "## Changelog\n\n"
    markdown_content += "_(Update this section with details of changes made to the codebase)_\n\n"

    # --- Glossary Section ---
    markdown_content += "## Glossary\n\n"

    for module in schemas:
        module_name = module.get("module_name", "Unknown Module")
        markdown_content += f"### Module: {module_name}\n\n"

        # Module Docstring
        if module.get("docstring"):
            markdown_content += f"**Module Docstring:**\n\n```python\n{module['docstring']}\n```\n\n"

        # Functions
        for func in module.get("functions", []):
            markdown_content += f"#### Function: {func['name']}\n\n"
            markdown_content += f"- **Defined in:** `{module_name}`\n"
            complexity = func.get("complexity", 'N/A')
            markdown_content += f"- **Complexity:** {complexity}\n"
            if func.get("docstring"):
                markdown_content += f"- **Docstring:**\n\n```python\n{func['docstring']}\n```\n"
            markdown_content += "\n"

        # Classes
        for cls in module.get("classes", []):
            markdown_content += f"#### Class: {cls['name']}\n\n"
            markdown_content += f"- **Defined in:** `{module_name}`\n"
            if cls.get("docstring"):
                markdown_content += f"- **Docstring:**\n\n```python\n{cls['docstring']}\n```\n"

            # Class Methods
            for method in cls.get("methods", []):
                markdown_content += f"##### Method: {method['name']}\n\n"
                complexity = method.get("complexity", 'N/A')
                markdown_content += f"- **Complexity:** {complexity}\n"
                if method.get("docstring"):
                    markdown_content += f"- **Docstring:**\n\n```python\n{method['docstring']}\n```\n"
                markdown_content += "\n"
            markdown_content += "\n"

    # --- Updated Source Code Section ---
    markdown_content += "## Updated Source Code\n\n"

    for dirpath, dirnames, filenames in os.walk(codebase_dir):
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path, codebase_dir)
                markdown_content += f"### File: {relative_path}\n\n"
                markdown_content += "```python\n"
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        code_content = code_file.read()
                        markdown_content += code_content
                except Exception as e:
                    markdown_content += f"# Error reading file {relative_path}: {e}"
                markdown_content += "\n```\n\n"

    # --- Write to Markdown file ---
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Markdown documentation generated successfully: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile codebase documentation into a Markdown file.")
    parser.add_argument('--schemas_file', default='output_schemas.json', help='Path to the JSON schemas file.')
    parser.add_argument('--codebase_dir', default='.', help='Path to the codebase directory.')
    parser.add_argument('--output_file', default='documentation.md', help='Path to the output Markdown file.')
    args = parser.parse_args()

    try:
        with open(args.schemas_file, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Schemas file '{args.schemas_file}' not found. Ensure the extraction step has been completed.")
        exit(1)

    generate_markdown_documentation(schemas, args.codebase_dir, args.output_file)

```

```python
import ast
import os
import json
import sys

def calculate_complexity(node):
    """
    Calculates the cyclomatic complexity of a function or method.

    Complexity is incremented for each branching point in the code.
    """
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.AsyncWith, ast.With,
                              ast.Try, ast.ExceptHandler, ast.BoolOp, ast.BinOp)):
            complexity += 1
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id in ('and_', 'or_'):
                complexity += 1
    return complexity

def insert_docstrings_and_calculate_complexity(file_path, schemas):
    """
    Inserts docstrings into functions and classes and calculates complexity.

    Parameters:
        file_path (str): Path to the Python file to process.
        schemas (list): List of schema dictionaries containing docstrings and other details.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return

    # Map to access schema details quickly
    schema_map = {}
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    for module_schema in schemas:
        if module_schema.get('module_name') == module_name:
            for func in module_schema.get('functions', []):
                schema_map[('function', func['name'])] = func
            for cls in module_schema.get('classes', []):
                schema_map[('class', cls['name'])] = cls
                for method in cls.get('methods', []):
                    schema_map[('method', cls['name'], method['name'])] = method

    class ComplexityDocstringInserter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Function or Method
            is_method = isinstance(self.current_parent, ast.ClassDef)
            if is_method:
                parent_name = self.current_parent.name
                schema_key = ('method', parent_name, node.name)
            else:
                schema_key = ('function', node.name)
            schema = schema_map.get(schema_key)

            if schema:
                # Insert docstring if missing
                if not ast.get_docstring(node) and schema.get('docstring'):
                    docstring_node = ast.Expr(value=ast.Constant(value=schema['docstring'], kind=None))
                    node.body.insert(0, docstring_node)

                # Calculate complexity
                complexity = calculate_complexity(node)
                schema['complexity'] = complexity

            # Continue walking the AST
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            schema_key = ('class', node.name)
            schema = schema_map.get(schema_key)

            if schema:
                # Insert docstring if missing
                if not ast.get_docstring(node) and schema.get('docstring'):
                    docstring_node = ast.Expr(value=ast.Constant(value=schema['docstring'], kind=None))
                    node.body.insert(0, docstring_node)

            # Set current parent to this class
            self.current_parent = node
            self.generic_visit(node)
            self.current_parent = None
            return node

        def generic_visit(self, node):
            if not hasattr(self, 'current_parent'):
                self.current_parent = None
            super().generic_visit(node)

    inserter = ComplexityDocstringInserter()
    try:
        new_tree = inserter.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_source = ast.unparse(new_tree)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_source)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_files(schemas, root_dir):
    """
    Processes all Python files in the given directory, inserting docstrings and calculating complexity.

    Parameters:
        schemas (list): List of schema dictionaries containing docstrings and other details.
        root_dir (str): Root directory of the codebase to process.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                insert_docstrings_and_calculate_complexity(file_path, schemas)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate complexity and insert docstrings into source code.')
    parser.add_argument('codebase_dir', help='Path to the root directory of the codebase to process.')
    parser.add_argument('--schemas_file', default='output_schemas.json', help='Path to the JSON schemas file.')
    args = parser.parse_args()

    try:
        with open(args.schemas_file, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Schemas file '{args.schemas_file}' not found. Ensure the previous steps have been completed.")
        sys.exit(1)

    process_files(schemas, args.codebase_dir)
    print("Docstrings inserted and complexity calculated successfully.")

```

```python
import os
import sys
import json
import time
import hashlib
import logging
import openai
import pydocstyle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docstring_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logging.error("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Cache for API responses to avoid redundant calls
api_cache = {}

def generate_docstring_with_gpt4(entity_details, entity_type='function'):
    """
    Generates a Google Style docstring using GPT-4.

    Parameters:
        entity_details (dict): Details of the function or class.
        entity_type (str): 'function' or 'class'.
    
    Returns:
        str: Generated docstring.
    """
    # Create a unique cache key based on entity details
    key_data = json.dumps(entity_details, sort_keys=True)
    cache_key = hashlib.md5(key_data.encode()).hexdigest()

    if cache_key in api_cache:
        logging.info(f"Using cached docstring for {entity_type} '{entity_details['name']}'.")
        return api_cache[cache_key]

    # Build the prompt for GPT-4
    code_snippet = entity_details.get('code_snippet', '')
    prompt = f"Generate a detailed Google Style docstring for the following Python {entity_type}:\n\n"
    if code_snippet:
        prompt += f"```python\n{code_snippet}\n```\n\n"

    if entity_type == 'function':
        if entity_details.get("parameters"):
            prompt += "Parameters:\n"
            for param in entity_details["parameters"]:
                param_type = param.get('type') or 'Any'
                prompt += f"- {param['name']}: {param_type}\n"
        if entity_details.get("return_type"):
            prompt += f"Returns:\n- {entity_details['return_type']}\n"
        if entity_details.get("exceptions"):
            prompt += "Raises:\n"
            for exc in entity_details["exceptions"]:
                prompt += f"- {exc}\n"
    elif entity_type == 'class':
        if entity_details.get("attributes"):
            prompt += "Attributes:\n"
            for attr in entity_details["attributes"]:
                prompt += f"- {attr}\n"
        if entity_details.get("methods"):
            prompt += "Methods:\n"
            for method in entity_details["methods"]:
                prompt += f"- {method['name']}\n"

    # Attempt to generate the docstring with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates Google Style docstrings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            generated_docstring = response.choices[0].message.content.strip()
            api_cache[cache_key] = generated_docstring  # Cache the result
            logging.info(f"Generated docstring for {entity_type} '{entity_details['name']}'.")
            return generated_docstring
        except openai.error.RateLimitError as e:
            wait_time = 2 ** attempt
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error while generating docstring for {entity_type} '{entity_details['name']}': {e}")
            return None
    logging.error(f"Failed to generate docstring for {entity_type} '{entity_details['name']}' after {max_retries} attempts.")
    return None

def is_valid_google_style(docstring):
    """
    Checks if a docstring adheres to Google Style using pydocstyle.

    Parameters:
        docstring (str): The docstring to validate.
    
    Returns:
        bool: True if the docstring adheres to Google Style, False otherwise.
    """
    if not docstring:
        return False

    temp_source = f'''
def temp_function():
    """
{docstring}
    """
    pass
'''
    errors = list(pydocstyle.check_source(temp_source, select=['D']))  # Select all docstring checks
    return not errors

def evaluate_and_generate_docstrings(schemas):
    """
    Evaluates existing docstrings and generates new ones if necessary.

    Parameters:
        schemas (list): List of module schemas containing entities.
    
    Returns:
        list: Updated schemas with generated docstrings.
    """
    updated_schemas = []
    for module in schemas:
        module_name = module.get('module_name', 'Unknown Module')
        logging.info(f"Processing module '{module_name}'.")
        # Process functions
        for func in module.get('functions', []):
            docstring = func.get('docstring', '')
            if not is_valid_google_style(docstring):
                logging.info(f"Generating docstring for function '{func['name']}' in module '{module_name}'.")
                generated_docstring = generate_docstring_with_gpt4(func, entity_type='function')
                if generated_docstring:
                    func['docstring'] = generated_docstring
            else:
                logging.info(f"Existing docstring for function '{func['name']}' is valid.")

        # Process classes
        for cls in module.get('classes', []):
            docstring = cls.get('docstring', '')
            if not is_valid_google_style(docstring):
                logging.info(f"Generating docstring for class '{cls['name']}' in module '{module_name}'.")
                generated_docstring = generate_docstring_with_gpt4(cls, entity_type='class')
                if generated_docstring:
                    cls['docstring'] = generated_docstring
            else:
                logging.info(f"Existing docstring for class '{cls['name']}' is valid.")

            # Process methods
            for method in cls.get('methods', []):
                docstring = method.get('docstring', '')
                if not is_valid_google_style(docstring):
                    logging.info(f"Generating docstring for method '{method['name']}' in class '{cls['name']}'.")
                    generated_docstring = generate_docstring_with_gpt4(method, entity_type='function')
                    if generated_docstring:
                        method['docstring'] = generated_docstring
                else:
                    logging.info(f"Existing docstring for method '{method['name']}' is valid.")

        updated_schemas.append(module)
    return updated_schemas

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate and generate Google Style docstrings for a Python codebase.')
    parser.add_argument('--schemas_file', default='output_schemas.json', help='Path to the JSON schemas file.')
    parser.add_argument('--output_file', default='updated_schemas.json', help='Path to save the updated JSON schemas.')
    args = parser.parse_args()

    # Load schemas
    try:
        with open(args.schemas_file, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
    except FileNotFoundError:
        logging.error(f"Schemas file '{args.schemas_file}' not found. Ensure the extraction step has been completed.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from schemas file '{args.schemas_file}': {e}")
        sys.exit(1)

    # Evaluate and generate docstrings
    updated_schemas = evaluate_and_generate_docstrings(schemas)

    # Save updated schemas
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_schemas, f, indent=4)
    logging.info(f"Updated schemas saved to '{args.output_file}'.")
    logging.info("Docstring evaluation and generation completed successfully.")

```

```python
import os
import ast
import sys
import json
import builtins
import importlib.util
from collections import deque


def get_imports(file_path):
    """Extract import statements from a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
            tree = ast.parse(source, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    if module:
                        imports.append(module)
                    else:
                        # Handle relative imports (module is None)
                        for alias in node.names:
                            imports.append(alias.name)
    except Exception as e:
        print(f"Error parsing imports in {file_path}: {e}")
    return imports


def categorize_imports(imports):
    """Categorize imports into standard library, third-party, and local modules."""
    std_libs = []
    third_party = []
    local_modules = []

    for imp in imports:
        if imp in sys.builtin_module_names:
            std_libs.append(imp)
        else:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                local_modules.append(imp)
            elif 'site-packages' in (spec.origin or ''):
                third_party.append(imp)
            else:
                std_libs.append(imp)
    return std_libs, third_party, local_modules


def extract_exception_names(node):
    """Extract exception names from a function or method node."""
    exceptions = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            exc = child.exc
            if isinstance(exc, ast.Call):
                exc_name = getattr(exc.func, 'id', None)
                if exc_name:
                    exceptions.append(exc_name)
            elif isinstance(exc, ast.Name):
                exceptions.append(exc.id)
            else:
                exceptions.append('UnknownException')
    return exceptions


def extract_function_details(node, module_name, file_path, source):
    """Extract details from a function node."""
    details = {
        "type": "function",
        "name": node.name,
        "parameters": [],
        "return_type": None,
        "decorators": [],
        "docstring": ast.get_docstring(node),
        "body_summary": None,
        "exceptions": [],
        "line_number": node.lineno,
        "module": module_name,
        "cross_module_refs": [],
        "annotations": {},
        "code_snippet": ast.get_source_segment(source, node),
    }

    # Parameters
    for arg in node.args.args:
        arg_type = None
        if arg.annotation:
            arg_type = ast.unparse(arg.annotation)
        param_details = {
            "name": arg.arg,
            "type": arg_type,
            "default": None,
        }
        details["parameters"].append(param_details)

    # Return type
    if node.returns:
        details["return_type"] = ast.unparse(node.returns)

    # Decorators
    for decorator in node.decorator_list:
        details["decorators"].append(ast.unparse(decorator))

    # Body summary
    if node.body:
        first_stmt = node.body[0]
        details["body_summary"] = first_stmt.__class__.__name__

    # Exceptions
    details["exceptions"] = extract_exception_names(node)

    # Cross-module references
    cross_refs = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                if not hasattr(builtins, child.id):
                    cross_refs.add(child.id)
    details["cross_module_refs"] = list(cross_refs)

    # Annotations
    if node.returns or node.args.args:
        details["annotations"] = {}
        if node.returns:
            details["annotations"]["return"] = ast.unparse(node.returns)
        for arg in node.args.args:
            if arg.annotation:
                details["annotations"][arg.arg] = ast.unparse(arg.annotation)

    return details


def extract_class_details(node, module_name, file_path, source):
    """Extract details from a class node."""
    details = {
        "type": "class",
        "name": node.name,
        "attributes": [],
        "methods": [],
        "decorators": [],
        "docstring": ast.get_docstring(node),
        "line_number": node.lineno,
        "module": module_name,
        "cross_module_refs": [],
        "code_snippet": ast.get_source_segment(source, node),
    }

    # Decorators
    for decorator in node.decorator_list:
        details["decorators"].append(ast.unparse(decorator))

    # Class attributes
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    details["attributes"].append(target.id)

    # Methods
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            method_details = extract_function_details(item, module_name, file_path, source)
            details["methods"].append(method_details)

    # Cross-module references
    cross_refs = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                if not hasattr(builtins, child.id):
                    cross_refs.add(child.id)
    details["cross_module_refs"] = list(cross_refs)

    return details


def extract_module_details(file_path, root_dir):
    """Extract details from a Python module."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
            tree = ast.parse(source, filename=file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    module_name = os.path.relpath(file_path, root_dir).replace(os.sep, '.')[:-3]

    details = {
        "module_name": module_name,
        "functions": [],
        "classes": [],
        "imports": {},
        "module_level_variables": [],
        "docstring": ast.get_docstring(tree),
        "code_snippet": source,
    }

    # Imports
    imports = get_imports(file_path)
    std_libs, third_party, local_modules = categorize_imports(imports)
    details["imports"] = {
        "standard_library": std_libs,
        "third_party": third_party,
        "local_modules": local_modules,
    }

    # Module-level variables and classes/functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_details = extract_function_details(node, module_name, file_path, source)
            details["functions"].append(func_details)
        elif isinstance(node, ast.ClassDef):
            class_details = extract_class_details(node, module_name, file_path, source)
            details["classes"].append(class_details)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    details["module_level_variables"].append(target.id)

    return details


def process_directory(directory):
    """Process all Python files in a directory and its subdirectories."""
    results = []
    queue = deque([directory])
    root_dir = directory

    while queue:
        current_dir = queue.popleft()
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                queue.append(item_path)
            elif item.endswith('.py'):
                module_details = extract_module_details(item_path, root_dir)
                if module_details:
                    results.append(module_details)
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract AST information from a Python codebase.')
    parser.add_argument('directory', help='Root directory of the codebase to analyze.')
    parser.add_argument('--output', default='output_schemas.json', help='Output JSON file.')
    args = parser.parse_args()

    codebase_dir = args.directory
    output_file = args.output

    results = process_directory(codebase_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f'Extraction complete. Results saved to {output_file}.')

```