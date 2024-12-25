# **1. Overview of the Project**

## **1.1. Project Description**

> **Context:**  
> The project, named *[Project Name]*, is a *[brief description of the project, e.g., web application, data processing pipeline, etc.]*. It is designed to *[describe the main purpose or functionality of the project]*. The codebase is structured to support *[mention any architectural patterns, e.g., microservices, monolithic, etc.]*.

## **1.2. Key Modules and Components**

> **Context:**  
> The project consists of several key modules, each responsible for specific functionalities:
> - **Module 1:** *[Brief description of Module 1, its purpose, and key functionalities. Include file paths if necessary.]*  
> - **Module 2:** *[Brief description of Module 2, its purpose, and key functionalities. Include file paths if necessary.]*  
> - **Module 3:** *[Brief description of Module 3, its purpose, and key functionalities. Include file paths if necessary.]*  
> - *[Add more modules as necessary]*

# **2. Existing Code Structures and Patterns**

## **2.1. Code Structures**

> **Context:**  
> The codebase follows specific coding patterns and structures:
> - **Design Patterns:** *[List any design patterns used, e.g., Singleton, Factory, Observer, etc.]*  
> - **Coding Standards:** *[Mention any coding standards or guidelines followed, e.g., PEP 8 for Python, etc.]*  
> - **File Organization:** *[Describe how files and directories are organized, e.g., by feature, by layer, etc.]*

## **2.2. Dependencies and Integrations**

> **Context:**  
> The project relies on several external libraries and integrations:
> - **Library 1:** *[Brief description of Library 1 and its purpose in the project.]*  
> - **Library 2:** *[Brief description of Library 2 and its purpose in the project.]*  
> - **Integration 1:** *[Brief description of Integration 1 and its purpose in the project.]*  
> - *[Add more dependencies and integrations as necessary]*

# **3. Current Challenges and Goals**

## **3.1. Challenges**

> **Context:**  
> The project currently faces several challenges that need to be addressed:
> - **Challenge 1:** *[Brief description of Challenge 1 and its impact on the project.]*  
> - **Challenge 2:** *[Brief description of Challenge 2 and its impact on the project.]*  
> - *[Add more challenges as necessary]*

## **3.2. Goals**

> **Context:**  
> The primary goals for the upcoming development phase include:
> - **Goal 1:** *[Brief description of Goal 1 and its significance.]*  
> - **Goal 2:** *[Brief description of Goal 2 and its significance.]*  
> - *[Add more goals as necessary]*

# **4. Specific Areas of Focus for Code Generation**

## **4.1. Target Areas**

> **Context:**  
> The AI should focus on the following areas for code generation:
> - **Area 1:** *[Brief description of Area 1 and why it is a focus for code generation.]*  
> - **Area 2:** *[Brief description of Area 2 and why it is a focus for code generation.]*  
> - *[Add more areas as necessary]*

## **4.2. Desired Outcomes**

> **Context:**  
> The desired outcomes for the code generation process include:
> - **Outcome 1:** *[Brief description of Outcome 1 and its expected impact.]*  
> - **Outcome 2:** *[Brief description of Outcome 2 and its expected impact.]*  
> - *[Add more outcomes as necessary]*

---

# **Guidelines For Using the Prompt**

1. **Provide Comprehensive Context:** Ensure that the AI has a clear understanding of the project, its components, and its goals.
2. **Highlight Key Areas:** Focus the AI's attention on the most relevant parts of the codebase to guide its code generation efforts.
3. **Set Clear Expectations:** Define the desired outcomes and any specific challenges or goals to align the AI's output with project needs.
4. **Iterate and Refine:** Continuously update and refine the prompt as the project evolves and new information becomes available.

This prompt template is designed to equip the AI with the necessary background and context to generate code effectively and align with the project's objectives.

---

# AI Code Generation Prompt Guide

## Overview

This guide provides a structured approach to preparing prompts for AI code generation. The template ensures that AI models receive comprehensive context about the project, existing code patterns, and desired outcomes.

## Template Structure

### 1. Project Overview

```markdown
### 1.1. Project Description

**Context:**  
The project, named [Project Name], is a [brief description]. It is designed to [main purpose/functionality]. The codebase follows [architectural pattern] architecture.

### 1.2. Key Modules and Components

**Context:**  
The project consists of these key modules:
- **[Module Name]:** [Purpose and key functionalities] (File: `path/to/module`)
- **[Additional Modules]:** [Description]
```

### 2. Code Structure and Dependencies

```markdown
### 2.1. Code Structures

**Context:**  
The codebase follows these patterns:
- **Design Patterns:** [List used patterns]
- **Coding Standards:** [Coding guidelines]
- **File Organization:** [Directory structure approach]

### 2.2. Dependencies and Integrations

**Context:**  
Key dependencies:
- **[Library Name]:** [Purpose in project]
- **[Integration Name]:** [Integration details]
```

### 3. Requirements and Goals

```markdown
### 3.1. Challenges

**Context:**  
Current challenges:
- **[Challenge 1]:** [Impact on project]
- **[Challenge 2]:** [Impact description]

### 3.2. Goals

**Context:**  
Primary goals:
- **[Goal 1]:** [Expected outcome]
- **[Goal 2]:** [Expected outcome]
```

### 4. Code Generation Focus

```markdown
### 4.1. Target Areas

**Context:**  
Focus areas for generation:
- **[Area 1]:** [Why it needs generation]
- **[Area 2]:** [Generation requirements]

### 4.2. Desired Outcomes

**Context:**  
Expected outcomes:
- **[Outcome 1]:** [Success criteria]
- **[Outcome 2]:** [Success criteria]
```

## Best Practices

1. **Be Specific with Context:**
   - Provide actual file paths when referencing modules
   - Include version numbers for dependencies
   - Reference specific coding standards (e.g., PEP 8)

2. **Include Code Examples:**

   ```python
   # Example of current implementation:
   def existing_function():
       # Implementation details
       pass

   # Desired pattern for generated code:
   def target_pattern():
       # Target implementation
       pass
   ```

3. **Detail Integration Points:**
   - Specify how generated code should interact with existing systems
   - Define expected input/output formats
   - List required error handling patterns

4. **Define Constraints:**
   - Maximum function/method complexity
   - Code style requirements
   - Documentation format expectations
   - Performance requirements

## Example Usage

Here's a filled-out example for a documentation generator project:

```markdown
### 1.1. Project Description

**Context:**  
The project, named "Auto-Something", is a documentation generation system designed to analyze and document code across multiple programming languages. It uses AI services while maintaining context awareness and hierarchical organization.

### 1.2. Key Modules and Components

**Context:**  
- **Configuration (`config.py`):** Manages settings using Pydantic models
- **Context Management (`context_manager.py`):** Handles dynamic context windows
- **Hierarchy Management (`hierarchy.py`):** Manages documentation structure
- **Multi-language Support (`multilang.py`):** Provides language detection and parsing

### 2.1. Code Structures

**Context:**  
- **Design Patterns:**
  - Factory Pattern for language parser creation
  - Singleton for configuration
  - Observer for context updates
- **Coding Standards:** PEP 8, Google docstring style
- **File Organization:** Core modules with language-specific extensions

### 2.2. Dependencies

**Context:**  
- **Pydantic:** Configuration and validation
- **tiktoken:** Token counting and management
- **aiohttp:** Async API interactions
- **SQLite:** Metadata storage
```

## Guidelines for Completion

1. **Fill All Sections:** Complete each template section with relevant details
2. **Maintain Consistency:** Use consistent terminology throughout
3. **Validate References:** Ensure all referenced files and patterns exist
4. **Review Dependencies:** Verify all listed dependencies are current
5. **Test Constraints:** Confirm all stated constraints are valid

## Validation Checklist

- [ ] All sections completed with project-specific information
- [ ] File paths and module references verified
- [ ] Design patterns and coding standards documented
- [ ] Integration points clearly defined
- [ ] Constraints and requirements specified
- [ ] Examples provided where helpful
- [ ] Dependencies and versions listed
- [ ] Success criteria established for outcomes
