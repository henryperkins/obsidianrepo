# DocScribe

An AI-powered code documentation generator.

## Overview

**DocScribe** is an automation tool that enhances your codebase by generating comprehensive documentation using OpenAI's GPT-4 API. By parsing your source code, DocScribe extracts the structure and generates clear, detailed docstrings and comments, making your code more understandable and maintainable. 

DocScribe supports multiple programming languages, including:

- Python
- JavaScript
- TypeScript
- HTML
- CSS

## Features

- **Automated Documentation Generation**: Generate summaries, docstrings, and comments for your code automatically.
- **Multi-language Support**: Handles Python, JavaScript, TypeScript, HTML, and CSS files.
- **Direct Code Insertion**: Inserts generated documentation directly into your source files.
- **Comprehensive Reports**: Produces a Markdown report summarizing your codebase.
- **Concurrent Processing**: Supports concurrent file processing for faster execution.
- **Customizable Configuration**: Exclude files or directories, set style guidelines, and provide project-specific information.
- **Safe Mode**: Run in safe mode to prevent modifications to source files.

## Installation

### Prerequisites

- **Python**: Version 3.7 or higher.
- **Node.js**: Required for processing JavaScript/TypeScript code.
- **OpenAI API Key**: Obtain from [OpenAI](https://openai.com/).

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/henryperkins/docscribe.git
    cd docscribe
    ```

2. **Install Python Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install Node.js Dependencies** 
    Navigate to the `scripts` directory (if applicable) and install any required Node.js packages. 

4. **Set Up OpenAI API Key**
    Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

    Or create a `.env` file in the project root:

    ```javascript
    OPENAI_API_KEY=your-api-key-here
    ```

## Usage

Run DocScribe on your code repository:

```bash
python main.py /path/to/your/repository
```

### Command-Line Arguments

- `repo_path`: Path to the code repository to document (positional argument).
- `-c`, `--config`: Path to `config.json` (default: `config.json`).
- `--concurrency`: Number of concurrent requests (default: `5`).
- `-o`, `--output`: Output Markdown file (default: `output.md`).
- `--model`: OpenAI model to use (default: `gpt-4`).
- `--skip-types`: Comma-separated list of file extensions to skip.
- `--project-info`: Information about the project.
- `--style-guidelines`: Documentation style guidelines to follow.
- `--safe-mode`: Run in safe mode (no files will be modified).
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `--schema`: Path to `function_schema.json` (default: `function_schema.json`).

### Example

Generate documentation for your project without modifying files and output the report to `docs.md`:

```bash
python main.py /path/to/your/repository -o docs.md --safe-mode
```

## Configuration

Customize DocScribe's behavior using a `config.json` file.

### Example `config.json`

```json
{
  "project_info": "An innovative project that streamlines code documentation.",
  "style_guidelines": "All docstrings should follow the Google Python Style Guide.",
  "excluded_dirs": ["venv", "node_modules", "tests"],
  "excluded_files": [".gitignore", "README.md"],
  "skip_types": [".md", ".json", ".txt"]
}
```

- **project_info**: General information about your project to help generate relevant documentation.
- **style_guidelines**: Specific guidelines for the style of documentation.
- **excluded_dirs**: Directories to exclude from processing.
- **excluded_files**: Specific files to exclude.
- **skip_types**: File extensions to skip.

## Output

DocScribe produces a Markdown report summarizing your codebase, including:

- **Summary**: A high-level overview of each file.
- **Changes Made**: List of changes or updates made to the code.
- **Functions**: Table listing functions with their arguments and descriptions.
- **Classes**: Details of classes and their methods with descriptions.
- **Code Blocks**: Source code with inserted docstrings and comments.

### Sample Generated Report Excerpt

```markdown
# File: main.py

## Summary

The codebase provides essential utility functions for setting configurations and validating models for projects using OpenAI. It includes functions to configure logging and validate model names, ensuring proper usage within an OpenAI environment.

## Functions

| Function              | Arguments     | Description                                         |
|----------------|---------------|-----------------------------------------------------|
| `configure_logging`   | `log_level`   | Configures logging based on the provided log level. |
| `validate_model_name` | `model_name`  | Validates the OpenAI model name format.             |

## Classes

No classes are defined in this file.
```

## Examples

### Documenting a Python Function

**Before:**

```python
def configure_logging(log_level):
    # Function implementation
    pass
```

**After DocScribe:**

```python
def configure_logging(log_level):
    """Configures logging based on the provided log level."""
    # Function implementation
    pass
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   Click the "Fork" button at the top right of the repository page.
2. **Create a Branch**

    ```bash
    git checkout -b feature/your-feature-name
    ```

3. **Commit Your Changes**

    ```bash
    git commit -am 'Add some feature'
    ```

4. **Push to the Branch**

    ```bash
    git push origin feature/your-feature-name
    ```

5. **Open a Pull Request**
   Submit your pull request on GitHub and provide a detailed description of your changes.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](https://you.com/LICENSE).