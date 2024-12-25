Creating a comprehensive `README.md` file is crucial for helping users and contributors understand your project. Below is a template for a `README.md` file tailored to your project's context, which involves using the Azure OpenAI API, code extraction, and documentation generation.

# Project Name

## Overview

This project is designed to automate the generation of documentation for Python codebases using the Azure OpenAI API. It includes functionality for extracting function details, calculating complexity scores, and generating structured Markdown documentation.

## Features

- **API Client Integration:** Interacts with the Azure OpenAI API to generate text and embeddings.
- **Code Extraction:** Analyzes Python code to extract function metadata and calculate complexity scores.
- **Documentation Generation:** Automates the creation of Markdown documentation, including summaries, changelogs, and glossaries.
- **Configuration Management:** Centralized handling of environment variables and application settings.
- **Comprehensive Testing:** Ensures functionality and performance through unit and integration tests.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API Key
- Docker (optional, for containerization)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**

   Create a `.env` file in the root directory with your configuration settings:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   AZURE_ENDPOINT=your_azure_endpoint
   ```

### Usage

1. **Run the Application:**

   ```bash
   python main.py
   ```

2. **Generate Documentation:**

   The application will analyze the codebase and generate Markdown documentation in the specified output directory.

### Configuration

The project uses a `Config` class to manage environment variables. Ensure all necessary variables are set in your `.env` file or system environment.

### Testing

To run the test suite, use:

```bash
pytest
```

This will execute all unit and integration tests, ensuring the system functions correctly.

## Documentation

- **[Azure OpenAI API Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/):** Detailed information on using the Azure OpenAI API.
- **[Python Standard Library Documentation](https://docs.python.org/3/library/):** Reference for Python’s built-in libraries.
- **[Docker Documentation](https://docs.docker.com/):** Guides on using Docker for containerization.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

For questions or support, please contact [your.email@example.com](mailto:your.email@example.com).

### Key Sections Explained

- **Overview:** Provides a brief summary of what the project does.
- **Features:** Lists the main functionalities of the project.
- **Getting Started:** Includes instructions on how to set up and run the project.
- **Configuration:** Describes how to manage configuration settings.
- **Testing:** Explains how to run tests to verify the system.
- **Documentation:** Links to relevant external documentation.
- **Contributing:** Offers guidelines for contributing to the project.
- **License:** States the licensing terms.
- **Contact:** Provides contact information for support or inquiries.
