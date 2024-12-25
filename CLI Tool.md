To create a complete CLI tool for the Simple Extract project, we'll integrate the functionalities for extracting code, configuring settings, and analyzing code patterns. This comprehensive implementation will provide a cohesive user experience, allowing users to interact with the tool through a single command-line interface.

Here's the complete implementation:

```python
import argparse
import json
import os
import re

CONFIG_FILE_PATH = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            return json.load(config_file)
    else:
        return {}

def save_config(config):
    with open(CONFIG_FILE_PATH, 'w') as config_file:
        json.dump(config, config_file, indent=4)

def prompt_for_input(prompt_text, default=None):
    if default:
        prompt_text += f" [{default}]"
    prompt_text += ": "
    user_input = input(prompt_text)
    return user_input if user_input else default

def extract_code(args):
    config = load_config()
    if not args.file:
        args.file = prompt_for_input("Enter the file or directory to extract from", config.get('default_directory', 'src/'))
    if not args.type:
        args.type = prompt_for_input("Enter the type of code to extract (function/class)", config.get('extraction_type', 'function'))
    if not args.output:
        args.output = prompt_for_input("Enter the output file or directory", 'output/')

    confirm = prompt_for_input(f"Are you sure you want to extract {args.type} from {args.file} to {args.output}? (y/n)", default="y")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files_to_process = []
    if os.path.isfile(args.file):
        files_to_process = [args.file]
    else:
        for root, _, files in os.walk(args.file):
            for file in files:
                if file.endswith('.py'):
                    files_to_process.append(os.path.join(root, file))

    for file_path in files_to_process:
        with open(file_path, 'r') as file:
            content = file.read()

        if args.type == 'function':
            pattern = r'def\s+\w+\(.*?\):'
        elif args.type == 'class':
            pattern = r'class\s+\w+\(.*?\):'
        else:
            print(f"Unknown type: {args.type}")
            return

        matches = re.findall(pattern, content)
        output_file_path = os.path.join(args.output, os.path.basename(file_path))
        with open(output_file_path, 'w') as output_file:
            for match in matches:
                output_file.write(match + '\n')

    print(f"Extraction completed. Extracted {args.type}s are saved in {args.output}.")

def configure_tool(args):
    config = load_config()

    if args.view:
        print("Current Configuration:")
        print(json.dumps(config, indent=4))
        return

    if args.set:
        key, value = args.set.split('=')
        config[key] = value
        save_config(config)
        print(f"Configuration updated: {key} = {value}")
        return

    if args.reset:
        confirm = prompt_for_input("Are you sure you want to reset the configuration to default? (y/n)", default="n")
        if confirm.lower() == 'y':
            config = {}
            save_config(config)
            print("Configuration reset to default.")
        return

    print("Interactive Configuration:")
    default_directory = prompt_for_input("Enter default directory for extraction", config.get('default_directory', 'src/'))
    extraction_type = prompt_for_input("Enter default extraction type (function/class)", config.get('extraction_type', 'function'))

    config['default_directory'] = default_directory
    config['extraction_type'] = extraction_type

    save_config(config)
    print("Configuration saved.")

def analyze_code(args):
    if not args.file:
        args.file = prompt_for_input("Enter the file or directory to analyze", default="src/")
    if not args.pattern:
        args.pattern = prompt_for_input("Enter the pattern to analyze for", default="def ")

    files_to_process = []
    if os.path.isfile(args.file):
        files_to_process = [args.file]
    else:
        for root, _, files in os.walk(args.file):
            for file in files:
                if file.endswith('.py'):
                    files_to_process.append(os.path.join(root, file))

    pattern_count = 0
    total_lines = 0
    function_count = 0

    for file_path in files_to_process:
        with open(file_path, 'r') as file:
            content = file.read()
            total_lines += content.count('\n') + 1
            pattern_count += len(re.findall(args.pattern, content))
            function_count += len(re.findall(r'def\s+\w+\(.*?\):', content))

    print(f"Analysis Report for {args.file}:")
    print(f"Total Python files analyzed: {len(files_to_process)}")
    print(f"Total lines of code: {total_lines}")
    print(f"Occurrences of pattern '{args.pattern}': {pattern_count}")
    print(f"Total functions defined: {function_count}")

def main():
    parser = argparse.ArgumentParser(prog='simple-extract', description='Simple Extract CLI Tool')
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help')

    # Extract subcommand
    parser_extract = subparsers.add_parser('extract', help='Extract code from files')
    parser_extract.add_argument('--file', help='File or directory to extract from')
    parser_extract.add_argument('--type', choices=['function', 'class'], help='Type of code to extract')
    parser_extract.add_argument('--output', help='Output file or directory')
    parser_extract.set_defaults(func=extract_code)

    # Config subcommand
    parser_config = subparsers.add_parser('config', help='Configure the tool')
    parser_config.add_argument('--view', action='store_true', help='View current configuration')
    parser_config.add_argument('--set', help='Set a configuration value (key=value)')
    parser_config.add_argument('--reset', action='store_true', help='Reset configuration to default')
    parser_config.set_defaults(func=configure_tool)

    # Analyze subcommand
    parser_analyze = subparsers.add_parser('analyze', help='Analyze code for patterns and metrics')
    parser_analyze.add_argument('--file', help='File or directory to analyze')
    parser_analyze.add_argument('--pattern', help='Pattern to analyze for')
    parser_analyze.set_defaults(func=analyze_code)

    # Parse arguments and call appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

### Key Features

- **Extract Command**: Extracts functions or classes from specified Python files or directories, saving the results to an output directory.
- **Config Command**: Manages configuration settings, allowing users to view, set, reset, and interactively configure tool preferences.
- **Analyze Command**: Analyzes code for specified patterns and calculates basic metrics, providing a summary report.

### Considerations

- **Error Handling**: The implementation includes basic error handling, such as checking for file existence and handling invalid inputs.
- **Extensibility**: The code is structured to allow easy addition of new features or enhancements.
- **Testing**: Ensure to test the CLI with various inputs and scenarios to validate its functionality.
- **Documentation**: Consider adding detailed documentation and usage examples to help users understand how to use the CLI effectively.

This complete implementation provides a functional and interactive CLI tool for the Simple Extract project, allowing users to extract, configure, and analyze code efficiently.

To ensure users can effectively utilize the Simple Extract CLI tool, comprehensive documentation is essential. This documentation will cover installation, usage, command descriptions, examples, and troubleshooting tips.

### Simple Extract CLI Tool Documentation

---

## Overview

The Simple Extract CLI tool is designed to help developers extract specific code components, configure tool settings, and analyze code patterns within Python projects. It provides a user-friendly command-line interface for efficient code management.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/simple-extract.git
   cd simple-extract
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed. You can install any required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Make the Script Executable** (Optional):
   If you want to run the script directly from the command line:
   ```bash
   chmod +x simple_extract.py
   ```

## Usage

The CLI tool provides three main commands: `extract`, `config`, and `analyze`. Each command has specific options and functionalities.

### Commands

#### Extract Command

Extracts functions or classes from specified Python files or directories.

- **Usage**:
  ```bash
  python simple_extract.py extract --file <path> --type <function/class> --output <output_path>
  ```

- **Options**:
  - `--file`: File or directory to extract from.
  - `--type`: Type of code to extract (`function` or `class`).
  - `--output`: Output file or directory for extracted code.

- **Example**:
  ```bash
  python simple_extract.py extract --file src/ --type function --output extracted/
  ```

#### Config Command

Manages configuration settings for the tool.

- **Usage**:
  ```bash
  python simple_extract.py config [--view] [--set key=value] [--reset]
  ```

- **Options**:
  - `--view`: View current configuration settings.
  - `--set`: Set a configuration value (format: `key=value`).
  - `--reset`: Reset configuration to default settings.

- **Example**:
  ```bash
  python simple_extract.py config --set default_directory=src/
  ```

#### Analyze Command

Analyzes code for specified patterns and calculates basic metrics.

- **Usage**:
  ```bash
  python simple_extract.py analyze --file <path> --pattern <pattern>
  ```

- **Options**:
  - `--file`: File or directory to analyze.
  - `--pattern`: Pattern to analyze for (e.g., `def ` for functions).

- **Example**:
  ```bash
  python simple_extract.py analyze --file src/ --pattern def
  ```

## Configuration

The tool uses a JSON configuration file (`config.json`) to store settings. You can modify this file directly or use the `config` command to update settings interactively.

## Troubleshooting

- **File Not Found**: Ensure the specified file or directory exists and the path is correct.
- **Invalid Command**: Use `python simple_extract.py --help` to view available commands and options.
- **Permission Issues**: Ensure you have the necessary permissions to read/write files in the specified directories.

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This documentation provides a comprehensive guide for users to install, configure, and use the Simple Extract CLI tool effectively. It includes detailed command descriptions, usage examples, and troubleshooting tips to help users navigate any issues they may encounter.