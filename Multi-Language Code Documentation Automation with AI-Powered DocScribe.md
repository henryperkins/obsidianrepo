## DocScribe: AI-Powered Multi-Language Code Documentation Script

**High-Level Overview:**

DocScribe is a Python script designed to automate the generation of high-quality, Google-style documentation (docstrings/comments) for codebases written in multiple programming languages. It leverages the power of OpenAI's GPT-4 API to analyze code structure and generate human-readable, informative documentation.

**Key Components:**

1. **Repository Traversal:**
   - DocScribe starts by traversing the specified code repository, identifying all relevant files based on configured file extensions and exclusion patterns.
   - It uses efficient file system operations to navigate directories and collect file paths.

2. **Code Analysis:**
   - For each identified file, DocScribe analyzes the code to extract structural information, such as:
     - Functions and their parameters (including async functions)
     - Classes, methods, and attributes
     - HTML tags and attributes
     - CSS selectors and declarations
   - It uses language-specific parsers (e.g., `ast` for Python, external Node.js scripts for JavaScript/TypeScript) to understand the code structure.

3. **AI-Powered Documentation Generation:**
   - DocScribe interacts with the OpenAI GPT-4 API to generate documentation for each code element.
   - It constructs prompts that include:
     - The code element's name, parameters, and context (e.g., surrounding code)
     - Project-specific information (e.g., purpose, style guidelines)
     - Instructions to generate specific documentation sections (e.g., summary, changes made)
   - It uses the OpenAI API's function calling capabilities to structure the response in a predictable JSON format.

4. **Documentation Insertion:**
   - DocScribe inserts the generated documentation directly into the source code files.
   - It uses language-specific techniques to insert docstrings (e.g., triple quotes in Python) or comments (e.g., `/* */` in JavaScript/CSS, `<!-- -->` in HTML).
   - It ensures that the inserted documentation is formatted correctly and adheres to the language's syntax.

5. **Documentation Report:**
   - DocScribe generates a comprehensive Markdown report summarizing the documentation generated for each file.
   - The report includes:
     - File name and path
     - Summary of the file's purpose
     - List of changes made
     - Detailed documentation for each function, class, method, etc.
   - The report can be used for code review, documentation sharing, or as a standalone reference.

6. **Configuration and Customization:**
   - DocScribe allows users to customize its behavior through a configuration file (`config.json`) and command-line arguments.
   - Configuration options include:
     - OpenAI API key
     - Repository path
     - Excluded directories and files
     - Supported file extensions
     - Project information and style guidelines
     - Concurrency level for API requests
     - Output file for the documentation report

7. **Error Handling and Logging:**
   - DocScribe implements robust error handling to catch exceptions and prevent script termination.
   - It logs all script activity, including API requests, errors, and warnings, to a log file for debugging and monitoring.

**Task Prompt for LLM Assistance:**

```
You are an AI assistant specializing in Python and OpenAI API integration. I am developing a script called DocScribe that automates code documentation using GPT-4. I need your help with the following tasks:

1. **Enhance Error Handling:**
   - Review the existing error handling in the script and identify any potential areas for improvement.
   - Suggest specific exception types that should be caught and handled in different parts of the code.
   - Provide code examples for handling these exceptions gracefully, including logging error details and potential recovery mechanisms.

2. **Improve Concurrency Control:**
   - The script currently uses `asyncio.Semaphore` to limit the number of concurrent API requests. Explore alternative concurrency control mechanisms, such as asynchronous queues or task-based approaches, to optimize performance and handle potential rate limiting from the OpenAI API.
   - Provide code examples and explain the advantages and disadvantages of each approach.

3. **Refactor for Modularity:**
   - The script has grown in size and complexity. Analyze the code and suggest ways to refactor it into smaller, more modular functions and classes.
   - Identify potential areas where code can be reused or abstracted into separate modules.
   - Provide a high-level outline of the proposed refactoring, including module names and function/class responsibilities.

4. **Add Support for Additional Languages:**
   - The script currently supports Python, JavaScript, TypeScript, HTML, and CSS. Research and propose a strategy for adding support for more programming languages, such as Java, C++, and Go.
   - Consider using external tools or libraries for parsing and manipulating code in these languages.
   - Provide a roadmap for implementing this new feature, including any necessary dependencies and code examples.

5. **Improve Documentation Report:**
   - The documentation report is currently a simple Markdown file. Explore ways to enhance its structure and presentation.
   - Consider adding features like:
     - Table of contents
     - Collapsible sections for functions and classes
     - Links to external documentation resources
     - Code syntax highlighting
   - Provide code examples and library suggestions for implementing these enhancements.

Please provide detailed explanations, code examples, and best practices for each suggestion.
```

This task prompt will guide the LLM to provide specific and actionable assistance in improving the DocScribe script.