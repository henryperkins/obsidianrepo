# [DocString Workflow and Integration](DocString%20Workflow%20and%20Integration.md)

This workflow integrates AST parsing, complexity calculation, docstring generation, and Markdown documentation generation. It handles various docstring conditions to ensure each function is documented with a summary, changelog, and complexity score.

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
- **Objective**: Create a structured Markdown file with sections for Summary, Changelog, and a Glossary containing each function’s name, complexity score, and docstring.

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
    """Extract function details including name, parameters, return type, and docstring."""
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
    """Calculate a simple complexity score based on the function's AST."""
    score = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)) 
                for node in ast.walk(function_node))
    return score
```

### Task 3: Check Docstring Completeness

```python
def is_incomplete_or_incorrect(docstring, params, return_type):
    """Check if a docstring is complete and in Google style."""
    has_params = all(param[0] in docstring for param in params)
    has_return = "Returns:" in docstring or return_type == "Unknown"
    return not (has_params and has_return)
```

### Task 4: Insert or Update Docstring

```python
def insert_docstring(function_node, docstring):
    """Insert a generated or updated docstring into the function node."""
    function_node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
    ast.fix_missing_locations(function_node)
```

### Task 5: Generate JSON Schema Prompt

```python
def create_prompt(func_name, params, return_type, complexity_score, docstring):
    """Generate JSON schema prompt for OpenAI."""
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
    """Analyze function, handle different scenarios, and store documentation entries."""
    func_name, params, return_type, existing_docstring = get_function_details(function_node)

    # Calculate complexity score
    complexity_score = calculate_complexity_score(function_node)

    # Determine if we need to generate or update a docstring
    if not existing_docstring or is_incomplete_or_incorrect(existing_docstring, params, return_type):
        # Create prompt with JSON schema
        prompt = create_prompt(func_name, params, return_type, complexity_score, existing_docstring)
        response = await client.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        
        # Parse JSON response
        response_data = json.loads(response.choices[0].message.content.strip())
        generated_docstring = response_data.get("docstring", "")
        summary = response_data.get("summary", "")
        changelog = response_data.get("changelog", "Initial documentation.")

        # Insert or replace the generated docstring
        insert_docstring(function_node, generated_docstring)
    else:
        # Use the existing docstring if correct
        generated_docstring = existing_docstring
        summary = f"{func_name} is already documented correctly."
        changelog = "No changes needed."

    # Add the function details to glossary entries
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
    """Write analysis to Markdown with summary, changelog, and glossary."""
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

This workflow ensures consistent documentation with detailed summaries, changelogs, and complexity insights, providing thorough and well-structured documentation for the codebase.

---

## Additional Notes

1. **Docstring Insertion**: 
   - The generated or updated docstring from OpenAI is inserted directly into the source code using AST, ensuring the function itself is documented within the codebase.

2. **Documentation Generation**:
   - After insertion, the docstring is also captured in the Markdown documentation under the AI-generated documentation section.
   - Each function’s docstring is included in the **Function/Method/Class Glossary** section within the generated documentation file.

### Full Workflow Summary

- The **source code** is updated with the docstring, ensuring each function has inline documentation.
- The **generated Markdown documentation** includes the docstring, complexity score, summary, and changelog, providing a detailed, AI-assisted overview for external reference.

This setup provides both inline and external documentation, keeping the codebase informative and accessible.