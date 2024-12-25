# Implementing Contextual Compression and Hybrid Prioritization in AutoSomething Project

Here's a series of prompts using the provided template to guide the AI in implementing the contextual compression feature into the Auto Something project:

## **1. Contextual Compression**

### **1.1. Implementing Contextual Compression**

#### **1.1.1. Refined Prompt 1:**

> **Context:**  
> The `AutoSomething` project is a documentation generation system that analyzes Python code and generates structured documentation. The project uses a context management system to handle large codebases efficiently. The current context management system includes modules like `utils.py`, `extract.py`, and `documentation.py`, which handle code parsing, dependency extraction, and documentation generation. The project follows a modular structure, with each module handling specific tasks.
> 
> The `utils.py` module contains utility functions for code formatting, complexity calculation, and dependency extraction. The `extract.py` module is responsible for extracting classes, functions, and their dependencies from the code. The `documentation.py` module generates documentation based on the extracted information.
> 
> **File Structure:** All modules are located in the `src/` directory. The `src/config.py` file contains project-wide configuration settings. The `src/main.py` file is the entry point of the application.
> 
> **Task:**  
> Implement the contextual compression feature into the `AutoSomething` project. This feature aims to maximize token efficiency by summarizing less critical parts of the context, allowing for more context to be included within limited token windows. The system should identify critical and non-critical context components, assign priority scores, and then compress the non-critical parts using summarization techniques.

> **Restrictions:**  
> - **File Structure:** All new code should be added to the existing `src/` directory. Do not modify the file structure or create new directories.
> - **Coding Standards:** Follow the project's existing coding conventions, including naming conventions, documentation standards, and formatting rules.
> - **Performance:** Ensure that the contextual compression does not significantly impact the overall performance of the documentation generation process.
> - **Dependencies:** Use only the existing dependencies and avoid introducing new external libraries for summarization.
> - **Backward Compatibility:** Ensure that the new feature does not break existing functionalities, especially the context management system and documentation generation process.

#### **1.1.2. Refined Prompt 2:**

> **Context:**  
> The testing framework for the `AutoSomething` project is based on the `unittest` library. The project includes a `tests/` directory, where all test cases are located. Each module has a corresponding test file, e.g., `tests/utils_tests.py` for testing the `utils.py` module.
> 
> **Task:**  
> Develop a comprehensive test suite to verify the contextual compression feature. The tests should ensure that the system correctly identifies critical and non-critical context components, assigns appropriate priority scores, and effectively compresses the non-critical parts while retaining the most important information.

> **Restrictions:**  
> - **File Structure:** All new test cases should be added to the existing `tests/` directory. Do not modify the file structure or create new directories.
> - **Test Coverage:** Ensure that the tests cover various scenarios, including different types of context components, complexity levels, and priority scores.
> - **Isolation:** Each test case should be independent and not rely on shared states or data.
> - **Mocking:** Use mocking where necessary to isolate the unit under test, especially when dealing with external dependencies or I/O operations.
> - **Performance Testing:** Include tests to ensure that the contextual compression does not significantly degrade the overall performance of the system.

---

## **2. Hybrid Prioritization**

### **2.1. Implementing Hybrid Prioritization**

#### **2.1.1. Refined Prompt 1:**

> **Context:**  
> The `AutoSomething` project uses a context management system to handle large codebases efficiently. The current system includes a basic prioritization mechanism that assigns priority scores based on component types. However, this approach might not always capture the most critical parts of the context.
> 
> **Task:**  
> Implement a hybrid prioritization approach that combines dynamic prioritization with contextual importance analysis. This approach should assign higher priority scores to critical sections like function definitions, class definitions, or key decision points, while also adjusting scores based on factors like complexity and frequency of use.

> **Restrictions:**  
> - **File Structure:** All new code should be added to the existing `src/` directory. Do not modify the file structure or create new directories.
> - **Coding Standards:** Follow the project's existing coding conventions, including naming conventions, documentation standards, and formatting rules.
> - **Performance:** Ensure that the hybrid prioritization does not significantly impact the overall performance of the context management system.
> - **Dependencies:** Use only the existing dependencies and avoid introducing new external libraries for analysis or prioritization.
> - **Backward Compatibility:** Ensure that the new prioritization approach does not break existing functionalities, especially the context management and documentation generation processes.

#### **2.1.2. Refined Prompt 2:**

> **Context:**  
> The testing framework for the `AutoSomething` project is well-established and includes test cases for various components.
> 
> **Task:**  
> Develop a comprehensive test suite to verify the hybrid prioritization approach. The tests should ensure that the system correctly identifies critical sections, assigns appropriate priority scores, and adjusts these scores based on dynamic factors like complexity and usage.

> **Restrictions:**  
> - **File Structure:** All new test cases should be added to the existing `tests/` directory. Do not modify the file structure or create new directories.
> - **Test Coverage:** Ensure that the tests cover various scenarios, including different types of context components, complexity levels, and usage patterns.
> - **Isolation:** Each test case should be independent and not rely on shared states or data.
> - **Mocking:** Use mocking where necessary to isolate the unit under test, especially when dealing with external dependencies or I/O operations.
> - **Performance Testing:** Include tests to ensure that the hybrid prioritization does not significantly degrade the overall performance of the context management system.

---

## **Guidelines For Developers**

- **Module Structure:**
   - Create new modules for the contextual compression and hybrid prioritization features, e.g., `src/compression.py` and `src/prioritization.py`.
   - Ensure that each module has a clear and focused responsibility.
- **Integration with Existing System:**
   - Modify the `src/main.py` file to integrate the new features into the main workflow.
   - Update the `src/config.py` file to include configuration options for the new features, such as compression thresholds and prioritization strategies.
- **Testing and Validation:**
   - Develop comprehensive test suites for both features, following the provided testing guidelines.
   - Ensure that the tests cover various scenarios and edge cases to validate the effectiveness and accuracy of the new features.
- **Documentation and Training:**
   - Update the project's documentation to include information about the new features, their purpose, and how to use them.
   - Provide training materials or tutorials to help developers understand and utilize the new features effectively.
- **Continuous Improvement:**
   - Encourage developers to provide feedback on the new features, including suggestions for improvement and potential issues.
   - Regularly review and update the features based on user feedback and changing project requirements.
- **Scalability and Performance:**
   - Optimize the new features to handle large-scale codebases and complex contexts efficiently.
   - Conduct performance testing to ensure that the features do not introduce significant overhead.
- **Backward Compatibility:**
   - Ensure that the new features do not break existing functionalities, especially the context management and documentation generation processes.
   - Thoroughly test the integration of the new features with the existing system.
- **Error Handling and Logging:**
   - Implement robust error handling mechanisms to manage potential issues during compression, prioritization, and integration.
   - Use logging to track the performance and behavior of the new features, helping with debugging and optimization.
- **Version Control and Collaboration:**
   - Use version control systems like Git to manage changes and collaborate effectively.
   - Encourage developers to work in branches and create pull requests for code reviews and feedback.

By following these guidelines and prompts, developers can effectively implement the contextual compression and hybrid prioritization features into the Auto Something project, ensuring that the system generates high-quality, relevant, and user-friendly documentation while adapting to the evolving needs of the codebase and its users.

---

Here's a series of prompts using the provided template to guide the AI in implementing the context-aware chunking feature into the Auto Something project:

## **1. Context-Aware Chunking**

### **1.1. Implementing Context-Aware Chunking**

#### **1.1.1. Refined Prompt 1:**

> **Context:**  
> The `AutoSomething` project is a documentation generation system that analyzes Python code and generates structured documentation. The project aims to handle large codebases efficiently by breaking them into manageable chunks while preserving relationships between code elements. The current system includes modules like `utils.py`, `extract.py`, and `documentation.py`, which handle code parsing, dependency extraction, and documentation generation. The project follows a modular structure, with each module handling specific tasks.
> 
> The `utils.py` module contains utility functions for code formatting, complexity calculation, and dependency extraction. The `extract.py` module is responsible for extracting classes, functions, and their dependencies from the code. The `documentation.py` module generates documentation based on the extracted information.
> 
> **File Structure:** All modules are located in the `src/` directory. The `src/config.py` file contains project-wide configuration settings. The `src/main.py` file is the entry point of the application.
> 
> **Task:**  
> Implement the context-aware chunking feature into the `AutoSomething` project. This feature aims to break large codebases into manageable chunks while preserving semantic relationships between code elements. The system should use language-specific parsers to ensure that logical boundaries are maintained, allowing the AI model to handle each chunk independently while understanding its context within the codebase.

> **Restrictions:**  
> - **File Structure:** All new code should be added to the existing `src/` directory. Do not modify the file structure or create new directories.
> - **Coding Standards:** Follow the project's existing coding conventions, including naming conventions, documentation standards, and formatting rules.
> - **Performance:** Ensure that the context-aware chunking process does not significantly impact the overall performance of the documentation generation system.
> - **Dependencies:** Use only the existing dependencies and avoid introducing new external libraries for parsing or chunking.
> - **Backward Compatibility:** Ensure that the new feature does not break existing functionalities, especially the code parsing and documentation generation processes.

#### **1.1.2. Refined Prompt 2:**

> **Context:**  
> The testing framework for the `AutoSomething` project is based on the `unittest` library. The project includes a `tests/` directory, where all test cases are located. Each module has a corresponding test file, e.g., `tests/utils_tests.py` for testing the `utils.py` module.
> 
> **Task:**  
> Develop a comprehensive test suite to verify the context-aware chunking feature. The tests should ensure that the system correctly breaks large codebases into manageable chunks while preserving semantic relationships. The tests should cover various scenarios, including different code structures, complexity levels, and relationship types.

> **Restrictions:**  
> - **File Structure:** All new test cases should be added to the existing `tests/` directory. Do not modify the file structure or create new directories.
> - **Test Coverage:** Ensure that the tests cover a wide range of code structures, complexity levels, and relationship types to validate the effectiveness of the chunking process.
> - **Isolation:** Each test case should be independent and not rely on shared states or data.
> - **Mocking:** Use mocking where necessary to isolate the unit under test, especially when dealing with external dependencies or I/O operations.
> - **Performance Testing:** Include tests to ensure that the context-aware chunking process does not significantly degrade the overall performance of the system.

---

## **2. Language-Specific Parsing**

### **2.1. Implementing Language-Specific Parsing**

#### **2.1.1. Refined Prompt 1:**

> **Context:**  
> The `AutoSomething` project supports multiple programming languages, including Python, JavaScript, and Java. The project aims to handle language-specific code parsing and chunking efficiently. The current system includes a basic language detection mechanism, but it does not handle language-specific parsing and chunking.
> 
> **Task:**  
> Implement a language-specific parsing and chunking system into the `AutoSomething` project. This system should use language-specific parsers to understand the syntax and semantics of each language, allowing for accurate code parsing and chunking. The system should support at least Python, JavaScript, and Java initially, with the ability to add more languages in the future.

> **Restrictions:**  
> - **File Structure:** All new code should be added to the existing `src/` directory. Create a new module, e.g., `src/language_parser.py`, for the language-specific parsing and chunking logic.
> - **Coding Standards:** Follow the project's existing coding conventions, including naming conventions, documentation standards, and formatting rules.
> - **Performance:** Ensure that the language-specific parsing and chunking process is efficient, especially when dealing with large codebases.
> - **Dependencies:** Use existing libraries or tools for language-specific parsing, such as AST parsers for Python, Babel for JavaScript, and JavaParser for Java. Avoid introducing new external dependencies if possible.
> - **Backward Compatibility:** Ensure that the new feature does not break existing functionalities, especially the code parsing and documentation generation processes.

#### **2.1.2. Refined Prompt 2:**

> **Context:**  
> The testing framework for the `AutoSomething` project is well-established and includes test cases for various components.
> 
> **Task:**  
> Develop a comprehensive test suite to verify the language-specific parsing and chunking feature. The tests should ensure that the system correctly parses and chunks code in different languages, preserving the semantic relationships between code elements. The tests should cover various scenarios, including different code structures, language-specific features, and edge cases.

> **Restrictions:**  
> - **File Structure:** All new test cases should be added to the existing `tests/` directory. Do not modify the file structure or create new directories.
> - **Test Coverage:** Ensure that the tests cover a wide range of code structures, language-specific features, and edge cases to validate the effectiveness of the language-specific parsing and chunking process.
> - **Isolation:** Each test case should be independent and not rely on shared states or data.
> - **Mocking:** Use mocking where necessary to isolate the unit under test, especially when dealing with external dependencies or I/O operations.
> - **Performance Testing:** Include tests to ensure that the language-specific parsing and chunking process is efficient, especially when handling large codebases.

---

## **Guidelines For Developers**

- **Module Structure:**
   - Create a new module, e.g., `src/language_parser.py`, for the language-specific parsing and chunking logic.
   - Ensure that the module is well-organized and follows the project's coding standards.
- **Language Support:**
   - Initially, support at least Python, JavaScript, and Java.
   - Use existing libraries or tools for language-specific parsing, such as AST parsers, to ensure accurate and efficient code processing.
- **Integration with Existing System:**
   - Modify the `src/main.py` file to integrate the new language-specific parsing and chunking system into the main workflow.
   - Update the `src/config.py` file to include configuration options for language-specific parsing, such as parser settings and supported languages.
- **Testing and Validation:**
   - Develop comprehensive test suites for the language-specific parsing and chunking feature, following the provided testing guidelines.
   - Ensure that the tests cover various languages, code structures, and edge cases to validate the effectiveness and accuracy of the new feature.
- **Documentation and Training:**
   - Update the project's documentation to include information about the new feature, its purpose, and how it works.
   - Provide training materials or tutorials to help developers understand and utilize the language-specific parsing and chunking system effectively.
- **Continuous Improvement:**
   - Encourage developers to provide feedback on the new feature, including suggestions for supporting additional languages and improving parsing accuracy.
   - Regularly review and update the feature based on user feedback and changing project requirements.
- **Scalability and Performance:**
   - Optimize the language-specific parsing and chunking system to handle large codebases efficiently, especially when dealing with multiple languages.
   - Conduct performance testing to ensure that the system scales well with increasing codebase size and complexity.
- **Backward Compatibility:**
   - Ensure that the new feature does not break existing functionalities, especially the code parsing and documentation generation processes.
   - Thoroughly test the integration of the new feature with the existing system.
- **Error Handling and Logging:**
   - Implement robust error handling mechanisms to manage potential issues during language-specific parsing and chunking, especially when dealing with different languages and code structures.
   - Use logging to track the performance and behavior of the new feature, helping with debugging and optimization.
- **Version Control and Collaboration:**
   - Use version control systems like Git to manage changes and collaborate effectively.
   - Encourage developers to work in branches and create pull requests for code reviews and feedback, especially when adding support for new languages.
