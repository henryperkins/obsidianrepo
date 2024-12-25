---
tags:
  - "#copilot-conversation"
  - "#copilot"
  - "#multimodal"
  - "#ai"
  - "#docstring-generation"
  - "#code-documentation"
  - "#complexity-analysis"
  - copilot-conversation
  - copilot
  - multimodal
  - ai
  - docstring-generation
  - code-documentation
  - complexity-analysis
---
# Goal: Function Calling with Structured Outputs and Prompt Caching for Docstring Generation

The objective is to implement a system where the GPT-4-o-2024-08-06 model can write documentation and insert docstrings into source code. The process involves the following steps:

- The AI model takes docstring information from a Python extraction module that traverses the AST (Abstract Syntax Tree) of the code.
- It then formats the docstring information into Google-style docstrings and inserts them into the source code.
- Using the context of the docstring and source code, the AI creates a summary and additional documentation.
- Finally, a docs module formats this information into a Markdown file, including a summary, changelog, glossary, and complexity calculations for functions, classes, and methods.

[DocString Workflow and Integration](./DocString%20Workflow%20and%20Integration.md):

# [DocString Workflow and Integration](./DocString%20Workflow%20and%20Integration.md)

## Overview

This workflow integrates multiple processes, including AST parsing, complexity calculation, docstring generation, and Markdown documentation generation. It handles various scenarios to ensure that each function is properly documented with a summary, changelog, and complexity score.

## Full Workflow for Documentation Generation

### Step 1: Parse Code with AST

- **Objective**: Extract function details such as name, parameters, return type, existing docstring, and function complexity.

### Step 2: Evaluate the Existing Docstring

- **Scenario A**: **No Docstring**
  - Action: Generate a new docstring using OpenAI with a JSON schema prompt.
- **Scenario B**: **Incomplete/Incorrect Docstring**
  - Action: Request an updated docstring from OpenAI if the existing one is incomplete or not in Google style.
- **Scenario C**: **Correct Docstring**
  - Action: Use the existing docstring if it is complete and follows Google style.

### Step 3: Calculate Complexity Score

- **Objective**: Compute a basic complexity score based on the function’s structure to assess maintainability.

### Step 4: Generate Summary and Changelog

- **Objective**: Include a brief summary and changelog entry in the documentation. Mark the changelog as “Initial documentation” for first-time entries.

### Step 5: Write Documentation to Markdown

- **Objective**: Create a structured Markdown file with sections for Summary, Changelog, and a Glossary, containing each function’s name, complexity score, and docstring.

---

## Scenarios and Workflow in Detail

### Scenario A: No Docstring

1. **Parse Function Details**: Extract function name, parameters, and return type using AST.
2. **Calculate Complexity Score**: Compute the function’s complexity score.
3. **Generate Docstring, Summary, and Changelog**:
   - Use OpenAI to create a Google-style docstring, summary, and changelog.
4. **Insert Generated Docstring**: Add the new docstring to the function.
5. **Store Details for Documentation**: Record the function details in the glossary entries.

### Scenario B: Incomplete or Incorrect Docstring

1. **Parse Function Details**: Extract function name, parameters, return type, and existing docstring.
2. **Calculate Complexity Score**: Compute the function’s complexity score.
3. **Determine Docstring Quality**: Check for completeness and adherence to Google style.
4. **Generate Updated Docstring, Summary, and Changelog**:
   - Request an updated docstring from OpenAI if needed.
5. **Replace Existing Docstring**: Update the function with the new docstring.
6. **Store Details for Documentation**: Record the updated function details in the glossary entries.

### Scenario C: Correct Docstring

1. **Parse Function Details**: Extract function name, parameters, return type, and existing docstring.
2. **Calculate Complexity Score**: Compute the function’s complexity score.
3. **Verify Docstring Completeness**: Ensure the docstring is complete and correctly formatted.
4. **Skip OpenAI Analysis**: Use the existing docstring if it is correct.
5. **Store Details for Documentation**: Record the function details in the glossary entries.

---

## Implementation of Workflow

### Task 1: Extract Function Details

```python
import ast

def get_function_details(function_node):
    """
    Extract function details including name, parameters, return type, and docstring.

    :param function_node: The AST node of the function.
    :return: Tuple containing function name, parameters, return type, and docstring.
    """
    func_name = function_node.name
    params = [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Unknown")
              for arg in function_node.args.args]
    return_type = ast.unparse(function_node.returns) if function_node.returns else "Unknown"
    docstring = ast.get_docstring(function_node) or ""
    return func_name, params, return_type, docstring
```

### Task 2: Calculate Complexity Score

```python
def calculate_complexity_score(function_node):
    """
    Calculate a simple complexity score based on the function's AST.

    :param function_node: The AST node of the function.
    :return: The complexity score.
    """
    score = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)) 
                for node in ast.walk(function_node))
    return score
```

### Task 3: Check Docstring Completeness

```python
def is_incomplete_or_incorrect(docstring, params, return_type):
    """
    Check if a docstring is complete and in Google style.

    :param docstring: The existing docstring.
    :param params: List of function parameters.
    :param return_type: The return type of the function.
    :return: True if the docstring is incomplete or incorrect, False otherwise.
    """
    has_params = all(param[0] in docstring for param in params)
    has_return = "Returns:" in docstring or return_type == "Unknown"
    return not (has_params and has_return)
```

### Task 4: Insert or Update Docstring

```python
def insert_docstring(function_node, docstring):
    """
    Insert a generated or updated docstring into the function node.

    :param function_node: The AST node of the function.
    :param docstring: The docstring to insert.
    """
    function_node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
    ast.fix_missing_locations(function_node)
```

### Task 5: Generate JSON Schema Prompt

```python
def create_prompt(func_name, params, return_type, complexity_score, docstring):
    """
    Generate a JSON schema prompt for OpenAI to generate a docstring.

    :param func_name: The function name.
    :param params: List of function parameters.
    :param return_type: The return type of the function.
    :param complexity_score: The complexity score of the function.
    :param docstring: The existing docstring.
    :return: The JSON schema prompt.
    """
    param_details = ", ".join([f"{name}: {ptype}" for name, ptype in params])

    prompt = f"""
    Generate a JSON object with the following fields:
    {{
        "summary": "Brief function overview.",
        "changelog": "Change history or 'Initial documentation.'",
        "docstring": "Google-style docstring including a Complexity section.",
        "complexity_score": {complexity_score}
    }}

    Function: {func_name}
    Parameters: {param_details}
    Returns: {return_type}
    Existing docstring: {docstring or "None"}
    """
    return prompt.strip()
```

### Task 6: Analyze Function and Store Documentation

```python
import openai
import json
import asyncio

async def analyze_function(client, function_node, glossary_entries):
    """
    Analyze a function, handle different docstring scenarios, and store documentation entries.

    :param client: The OpenAI client.
    :param function_node: The AST node of the function.
    :param glossary_entries: List to store documentation entries.
    """
    func_name, params, return_type, existing_docstring = get_function_details(function_node)
    complexity_score = calculate_complexity_score(function_node)

    if not existing_docstring or is_incomplete_or_incorrect(existing_docstring, params, return_type):
        prompt = create_prompt(func_name, params, return_type, complexity_score, existing_docstring)
        response = await client.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        response_data = json.loads(response.choices[0].message.content.strip())
        generated_docstring = response_data.get("docstring", "")
        summary = response_data.get("summary", "")
        changelog = response_data.get("changelog", "Initial documentation.")
        insert_docstring(function_node, generated_docstring)
    else:
        generated_docstring = existing_docstring
        summary = f"{func_name} is already documented correctly."
        changelog = "No changes needed."

    glossary_entries.append({
        "function_name": func_name,
        "complexity_score": complexity_score,
        "docstring": generated_docstring,
        "summary": summary,
        "changelog": changelog
    })
```

### Task 7: Write Analysis to Markdown

```python
def write_analysis_to_markdown(glossary_entries, output_file_path):
    """
    Write the analysis to a Markdown file with summary, changelog, and glossary.

    :param glossary_entries: List containing documentation entries.
    :param output_file_path: Path to the output Markdown file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as md_file:
        # Summary section
        md_file.write("# Summary\n\n")
        for entry in glossary_entries:
            md_file.write(f"- {entry['function_name']}: {entry['summary']}\n")
        md_file.write("\n")

        # Changelog section
        md_file.write("# Changelog\n\n")
        for entry in glossary_entries:
            md_file.write(f"- {entry['function_name']}: {entry['changelog']}\n")
        md_file.write("\n")

        # Glossary section with table format
        md_file.write("# Function/Method/Class Glossary\n\n")
        md_file.write("| Function Name | Complexity Score | Docstring |\n")
        md_file.write("|---------------|-----------------|-----------|\n")
        
        for entry in glossary_entries:
            func_name = entry["function_name"]
            complexity_score = entry["complexity_score"]
            docstring = entry["docstring"].replace("\n", " ")  # One-line format for table
            md_file.write(f"| {func_name} | {complexity_score} | {docstring} |\n")

        md_file.write("\n")  # End of glossary table
```

---

## Summary of Each Scenario’s Workflow

- **Scenario A (No Docstring)**: Generate a new docstring with OpenAI, insert it, and store the function details for the glossary.
- **Scenario B (Incomplete/Incorrect Docstring)**: Request an updated docstring from OpenAI, replace the old one, and store updated details.
- **Scenario C (Correct Docstring)**: Skip OpenAI analysis and retain the existing docstring, adding it to the glossary as-is.

This workflow ensures consistent and thorough documentation with summaries, changelogs, and complexity insights, resulting in a well-structured and informative codebase.

---

## Additional Notes

1. **Docstring Insertion**: 
   - The generated or updated docstring is inserted directly into the source code using AST, ensuring that each function has inline documentation.

2. **Documentation Generation**:
   - After insertion, the docstring is also included in the Markdown documentation, specifically in the **Function/Method/Class Glossary** section.
   - The generated Markdown documentation provides an external reference with detailed summaries, changelogs, and complexity scores.

### Full Workflow Summary

- The **source code** is updated with docstrings, ensuring inline documentation for each function.
- The **generated Markdown documentation** offers an external overview with additional context, including docstrings, complexity scores, summaries, and changelogs.

This approach provides a comprehensive documentation solution, keeping the codebase well-documented and easily accessible.
