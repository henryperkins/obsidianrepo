# Reference Document for DocString Workflow Automation

## Overview

This document outlines the workflow for automating the generation and insertion of docstrings in a Python codebase, along with the compilation of comprehensive documentation. The workflow integrates AST parsing, complexity calculation, docstring evaluation and generation, and Markdown documentation generation.

## Workflow Steps

### Step 1: Parse Code with AST

- **Objective:** Extract detailed information about functions and classes, including:
  - Function/Class Name
  - Parameters (name, type, default value)
  - Return Type
  - Decorators
  - Existing Docstring
  - Body Summary
  - Exceptions Raised
  - Class Attributes
  - Module-Level Variables
  - Imports
  - Annotations
  - Line Numbers

- **Tools:** Use Python's `ast` module.
- **Output:** JSON schemas for each function and class.

### Step 2: Evaluate and Generate DocStrings

- **Objective:** Assess existing docstrings and generate new ones if necessary.
  - **Scenario A:** No Docstring
    - Generate a new docstring using GPT-4.
  - **Scenario B:** Incomplete/Incorrect Docstring
    - Update the docstring using GPT-4 if it is incomplete or not in Google style.
  - **Scenario C:** Correct Docstring
    - Use the existing docstring if it is complete and follows Google style.

- **Tools:** GPT-4 API for docstring generation.
- **Output:** Updated JSON schemas with complete docstrings.

### Step 3: Calculate Complexity Score

- **Objective:** Compute a basic complexity score for each function to assess maintainability.
- **Tools:** Use AST to analyze function structure.
- **Output:** Complexity scores included in JSON schemas.

### Step 4: Insert DocStrings into Source Code

- **Objective:** Insert generated or updated docstrings into the source code.
- **Tools:** Modify the AST to insert docstrings.
- **Output:** Updated source code files.

### Step 5: Compile Documentation to Markdown

- **Objective:** Create a structured Markdown document with sections for:
  - Summary
  - Changelog
  - Glossary (function names, complexity scores, and docstrings)

- **Tools:** Use standard Python libraries for file operations.
- **Output:** A comprehensive Markdown document.

## Considerations

- **File Structure:** Organize scripts within a `tools/` directory.
- **Coding Standards:** Adhere to PEP 8 guidelines.
- **Performance:** Optimize for large codebases and efficient API usage.
- **Security:** Implement input validation and error handling.
- **Testing:** Develop unit and integration tests to ensure accuracy and completeness.

## Goals/Checklist

- [ ] Extract and structure AST information into JSON schemas.
- [ ] Evaluate and generate docstrings using GPT-4.
- [ ] Calculate and include complexity scores.
- [ ] Insert docstrings into the source code.
- [ ] Compile a comprehensive Markdown document.
- [ ] Validate each step with appropriate tests.
- [ ] Document scripts and their usage.

## Additional Notes

- **Docstring Insertion:** Ensure that docstrings are inserted in a way that maintains code readability and structure.
- **Documentation Generation:** The Markdown document should be easy to navigate and provide a thorough overview of the codebase.
- **Integration:** Consider integrating the workflow into a CI/CD pipeline for continuous documentation updates.

---

This reference document provides a clear and structured overview of the entire workflow, ensuring that the LLM has all the necessary information to generate the scripts effectively.