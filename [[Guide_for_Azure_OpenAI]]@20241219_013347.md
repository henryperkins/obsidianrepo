---
epoch: 1734593627591
modelKey: azureml://registries/azure-openai/models/gpt-4o/versions/2024-11-20|azure openai
tags:
  - copilot-conversation
---

**user**:  [Guide for Azure OpenAI](./Guide%20for%20Azure%20OpenAI.md)
[Timestamp: 2024/12/19 01:33:47]

**ai**: It seems you've provided a comprehensive guide for Azure OpenAI, including advanced use cases and implementation strategies. If you have any specific questions or need further assistance with any section of the guide, feel free to ask!
[Timestamp: 2024/12/19 01:33:50]

**user**: [ai_service](./ai_service.md) [prompt_manager](./prompt_manager.md) [response_parsing](./response_parsing.md) [response_formatter.py](response_formatter.py.md) [schema_validator.py](schema_validator.py.md)
[Timestamp: 2024/12/19 01:34:11]

**ai**: It seems you've shared a comprehensive set of modules and documentation for an AI service system that interacts with the Azure OpenAI API. These modules include functionalities for prompt management, response parsing, response formatting, and schema validation. If you have any specific questions, need clarifications, or want to enhance any part of the system, feel free to ask!
[Timestamp: 2024/12/19 01:34:16]

**user**: [docstring_processor](./docstring_processor.md) [docstring_validator.py](docstring_validator.py.md) [base](./base.md) [docstring.py](docstring.py.md)
[Timestamp: 2024/12/19 01:34:30]

**ai**: You've provided a detailed set of modules and documentation for handling docstring processing, validation, and related operations. These modules include:

1. **Docstring Processor**: Responsible for parsing and validating docstrings, collecting metrics, and generating structured data.
2. **Docstring Validator**: Validates docstring data against a predefined schema.
3. **Base Module**: Contains foundational data structures and utility classes for handling extraction, processing, and validation of code elements.
4. **Docstring Data**: A data structure for representing and validating docstring content.

If you need help with any specific part of these modules, such as improving functionality, debugging, or integrating them into a larger system, let me know!
[Timestamp: 2024/12/19 01:34:37]

**user**: [docstring_schema](docstring_schema.md) [function_tools_schema](function_tools_schema.md) [documentation_prompt.txt](./documentation_prompt.txt.md) [code_analysis_prompt.txt](./code_analysis_prompt.txt.md)
[Timestamp: 2024/12/19 01:36:13]

**ai**: Here’s a detailed analysis of the provided schemas and prompt templates:

---

### **1. Docstring Schema**
#### **Purpose**
Defines the structure for validating Google-style docstrings. It ensures that docstrings adhere to a consistent format, including fields like `summary`, `description`, `args`, `returns`, `raises`, and `complexity`.

#### **Strengths**
- **Comprehensive Structure**: Covers all essential components of a docstring, including arguments, return values, exceptions, and complexity.
- **Validation Rules**: Enforces required fields (`summary`, `description`, `returns`) and ensures proper nesting for `args` and `raises`.
- **Default Values**: Provides sensible defaults for optional fields like `args` and `raises`.

#### **Potential Issues**
1. **Complexity Field**: The `complexity` field allows both integers and strings, which might lead to inconsistent data. It should ideally be restricted to integers.
2. **Additional Properties**: The schema disallows additional properties, which might limit flexibility for future extensions.
3. **Default Values for Arrays**: While `args` and `raises` default to empty arrays, this might not always reflect the actual state of the function.

#### **Suggestions for Improvement**
- **Restrict Complexity**: Limit the `complexity` field to integers for consistency.
- **Allow Extensions**: Add a mechanism (e.g., `x-extensions`) to support additional properties without breaking the schema.
- **Dynamic Defaults**: Allow defaults to be dynamically generated based on context (e.g., `args` populated from function signatures).

---

### **2. Function Tools Schema**
#### **Purpose**
Defines the structure for generating Python docstrings with structured information. It is used to guide the AI in creating consistent and detailed docstrings.

#### **Strengths**
- **Alignment with Docstring Schema**: Mirrors the `docstring_schema` structure, ensuring compatibility between generation and validation.
- **Detailed Descriptions**: Provides clear descriptions for each field, aiding both developers and the AI in understanding the schema.
- **Default Values**: Includes defaults for optional fields like `args` and `raises`.

#### **Potential Issues**
1. **Redundancy**: The schema closely resembles the `docstring_schema`, leading to potential duplication.
2. **Complexity Field**: Similar to the `docstring_schema`, the `complexity` field might lead to inconsistencies due to its dual type (`integer` and `string`).
3. **Scalability**: The schema is tightly coupled to the current docstring structure, making it harder to adapt to new styles or formats.

#### **Suggestions for Improvement**
- **Unify Schemas**: Combine the `docstring_schema` and `function_tools_schema` into a single reusable schema to avoid redundancy.
- **Restrict Complexity**: Limit the `complexity` field to integers.
- **Modular Design**: Break the schema into reusable components (e.g., `args` and `raises` as separate schemas) for better maintainability.

---

### **3. Documentation Prompt**
#### **Purpose**
Provides a template for generating comprehensive Google-style documentation for Python modules. It guides the AI in analyzing the code and producing structured JSON responses.

#### **Strengths**
- **Clear Objective**: Clearly states the goal of generating Google-style documentation.
- **Structured Output**: Specifies a JSON schema for the response, ensuring consistency and adherence to the `docstring_schema`.
- **Contextual Information**: Includes module name, file path, and code structure to provide context for the AI.

#### **Potential Issues**
1. **Static Schema**: The JSON schema is hardcoded, making it difficult to adapt to different documentation styles or formats.
2. **Limited Error Handling**: The prompt does not explicitly address how the AI should handle ambiguous or incomplete code.
3. **Verbose Output**: Including the full source code might lead to token limits being exceeded for large modules.

#### **Suggestions for Improvement**
- **Dynamic Schema**: Allow the schema to be dynamically selected based on user preferences or project requirements.
- **Error Handling**: Add instructions for handling ambiguous or incomplete code (e.g., "If the code is ambiguous, provide a best-effort response with a warning.").
- **Token Optimization**: Include only relevant parts of the source code (e.g., function signatures) to reduce token usage.

---

### **4. Code Analysis Prompt**
#### **Purpose**
Guides the AI in analyzing code for quality and providing actionable improvements. It focuses on aspects like complexity, best practices, error handling, performance, and documentation.

#### **Strengths**
- **Comprehensive Analysis**: Covers all critical aspects of code quality, from complexity to documentation.
- **Actionable Insights**: Encourages the AI to provide specific recommendations for improvement.
- **Structured Output**: Specifies a JSON format for the response, ensuring consistency and usability.

#### **Potential Issues**
1. **Ambiguity in Recommendations**: The prompt does not specify how detailed the recommendations should be, which might lead to inconsistent responses.
2. **Performance Bottlenecks**: Analyzing large codebases might exceed token limits or lead to incomplete responses.
3. **Subjectivity**: Terms like "best practices" and "readability" are subjective and might lead to varying interpretations.

#### **Suggestions for Improvement**
- **Granularity**: Specify the desired level of detail for recommendations (e.g., "Provide at least three specific examples for each issue.").
- **Scalability**: Include instructions for handling large codebases (e.g., "Focus on the most critical functions if the code is too large.").
- **Standardization**: Define what constitutes "best practices" and "readability" to reduce subjectivity.

---

### **General Observations**
#### **Strengths**
- **Consistency**: The schemas and prompts are well-aligned, ensuring smooth integration between generation, validation, and analysis.
- **Clarity**: Each schema and prompt is clearly documented, making it easy to understand and use.
- **Focus on Quality**: Emphasizes best practices, structured outputs, and actionable insights.

#### **Areas for Improvement**
1. **Dynamic Adaptability**: Introduce mechanisms to adapt schemas and prompts to different styles, formats, or user preferences.
2. **Error Handling**: Provide explicit instructions for handling errors, ambiguities, or incomplete data.
3. **Scalability**: Optimize for large codebases by focusing on critical sections or summarizing less relevant parts.

---

### **Conclusion**
The provided schemas and prompts form a robust foundation for generating, validating, and analyzing Python documentation and code. By addressing the identified issues and implementing the suggested improvements, you can enhance their flexibility, scalability, and usability. Let me know if you'd like help implementing any of these suggestions!
[Timestamp: 2024/12/19 01:36:41]

**ai**: Here’s a detailed analysis of **code redundancy** and **missed integration opportunities** across the provided modules, schemas, and prompts. This analysis highlights areas where functionality can be reused, streamlined, or better integrated to reduce duplication and improve maintainability.

---

### **1. Code Redundancy**
#### **1.1 Redundant Schema Definitions**
- **Observation**: The `docstring_schema` and `function_tools_schema` are nearly identical in structure, with both defining fields like `summary`, `description`, `args`, `returns`, `raises`, and `complexity`.
- **Impact**: Maintaining two separate schemas increases the risk of inconsistencies and duplication of effort when updates are needed.
- **Solution**:
  - **Unify Schemas**: Create a single reusable schema (e.g., `base_docstring_schema`) that both `docstring_schema` and `function_tools_schema` can extend or reference.
  - **Example**:
    ```json
    {
        "type": "object",
        "properties": {
            "summary": {...},
            "description": {...},
            "args": {...},
            "returns": {...},
            "raises": {...},
            "complexity": {...}
        },
        "required": ["summary", "description", "returns"],
        "additionalProperties": false
    }
    ```
    Then, reference this schema in both `docstring_schema` and `function_tools_schema` using `$ref`.

---

#### **1.2 Redundant Parsing Logic**
- **Observation**: The `DocstringProcessor` and `ResponseParser` both include logic for parsing and validating JSON responses. For example:
  - `DocstringProcessor` parses docstrings into structured data and validates them against `docstring_schema`.
  - `ResponseParser` parses AI responses and validates them against schemas (e.g., `docstring_schema`).
- **Impact**: Duplicated parsing and validation logic increases maintenance overhead and the likelihood of bugs.
- **Solution**:
  - **Centralize Parsing and Validation**: Create a shared utility module (e.g., `schema_utils`) with functions for parsing and validating data against schemas.
  - **Example**:
    ```python
    def parse_and_validate(data: dict, schema: dict) -> tuple[bool, list[str]]:
        """Parse and validate data against a schema."""
        try:
            validate(instance=data, schema=schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
    ```

---

#### **1.3 Redundant Fallback Handling**
- **Observation**: Both `ResponseFormatter` and `DocstringProcessor` include fallback mechanisms for handling invalid or incomplete data:
  - `ResponseFormatter` creates fallback responses for invalid AI outputs.
  - `DocstringProcessor` generates fallback docstring structures when parsing fails.
- **Impact**: Duplicated fallback logic leads to inconsistent handling of errors and increases code complexity.
- **Solution**:
  - **Centralize Fallback Logic**: Create a shared fallback handler that both modules can use.
  - **Example**:
    ```python
    def generate_fallback_response(reason: str, schema: dict) -> dict:
        """Generate a fallback response based on the schema."""
        fallback = {key: "Not available" for key in schema["properties"]}
        fallback["error"] = reason
        return fallback
    ```

---

### **2. Missed Integration Opportunities**
#### **2.1 Schema Reuse**
- **Missed Opportunity**: The `DocstringValidator` and `ResponseParser` both validate data against schemas but do not share schema definitions or validation logic.
- **Solution**:
  - Use the same schema definitions (e.g., `docstring_schema`) across both modules.
  - Share validation logic by integrating `SchemaValidator` into `ResponseParser`.

---

#### **2.2 Shared Metrics Collection**
- **Missed Opportunity**: Both `DocstringProcessor` and `ResponseParser` collect metrics (e.g., processing time, validation success) but do so independently.
- **Solution**:
  - Create a shared `MetricsCollector` utility that both modules can use.
  - **Example**:
    ```python
    class MetricsCollector:
        def track_operation(self, operation_type: str, success: bool, duration: float, metadata: dict):
            # Shared logic for tracking metrics
    ```

---

#### **2.3 Prompt and Schema Integration**
- **Missed Opportunity**: The `documentation_prompt.txt` and `code_analysis_prompt.txt` templates specify JSON schemas for AI responses, but these schemas are not directly linked to the `docstring_schema` or `function_tools_schema`.
- **Solution**:
  - Dynamically generate the JSON schema in the prompts using the actual schema definitions.
  - **Example**:
    ```python
    from json import dumps

    def generate_prompt_with_schema(schema: dict, template: str) -> str:
        schema_json = dumps(schema, indent=4)
        return template.replace("{{ schema }}", schema_json)
    ```

---

#### **2.4 Docstring Data and Response Parsing**
- **Missed Opportunity**: The `DocstringData` class provides a structured representation of docstrings, but it is not used in `ResponseParser` to parse AI-generated responses.
- **Solution**:
  - Use `DocstringData` as the target data structure for parsed AI responses in `ResponseParser`.
  - **Example**:
    ```python
    parsed_docstring = DocstringData.from_dict(response["choices"][0]["message"]["content"])
    ```

---

#### **2.5 Complexity Analysis**
- **Missed Opportunity**: The `complexity` field in both schemas is static and does not leverage actual complexity analysis tools.
- **Solution**:
  - Integrate tools like `radon` to calculate McCabe complexity dynamically during docstring processing or code analysis.
  - **Example**:
    ```python
    from radon.complexity import cc_visit

    def calculate_complexity(source_code: str) -> int:
        return max([block.complexity for block in cc_visit(source_code)], default=1)
    ```

---

### **3. High-Level Integration Plan**
To address redundancy and missed opportunities, the following integration plan can be implemented:

#### **Step 1: Centralize Schemas**
- Create a `schemas` module with reusable schema definitions.
- Use `$ref` to reference shared schemas in `docstring_schema` and `function_tools_schema`.

#### **Step 2: Create Shared Utilities**
- Create a `utils` module with shared functions for:
  - Parsing and validating data against schemas.
  - Generating fallback responses.
  - Collecting metrics.

#### **Step 3: Standardize Data Structures**
- Use `DocstringData` as the standard data structure for both docstring processing and response parsing.

#### **Step 4: Dynamic Schema Integration**
- Dynamically inject schema definitions into prompts to ensure consistency between AI responses and validation logic.

#### **Step 5: Enhance Complexity Analysis**
- Integrate complexity analysis tools into both `DocstringProcessor` and `ResponseParser`.

---

### **4. Example Refactored Workflow**
Here’s an example of how the modules can work together after refactoring:

1. **Docstring Processing**:
   - Parse the docstring using `DocstringProcessor`.
   - Validate the parsed data against `docstring_schema` using `SchemaValidator`.
   - Collect metrics using `MetricsCollector`.

2. **AI Response Parsing**:
   - Parse the AI response using `ResponseParser`.
   - Use `DocstringData` to represent the parsed response.
   - Validate the response against `docstring_schema` using `SchemaValidator`.

3. **Prompt Generation**:
   - Dynamically inject `docstring_schema` into the `documentation_prompt.txt` template.

4. **Code Analysis**:
   - Analyze code complexity using `radon` and include the results in the `complexity` field.

---

### **Conclusion**
By addressing redundancy and integrating functionality across modules, the system can become more maintainable, scalable, and consistent. Let me know if you'd like help implementing these improvements!
[Timestamp: 2024/12/19 01:37:40]

**user**: Create a refactoring plan that doesn't involve creating new modules.
[Timestamp: 2024/12/19 01:38:02]

**ai**: Here’s a **refactoring plan** that focuses on improving the existing modules without introducing new ones. The goal is to reduce redundancy, improve integration, and enhance maintainability while keeping the current module structure intact.

---

### **1. Centralize Schema Usage**
#### **Problem**
- The `docstring_schema` and `function_tools_schema` are nearly identical but defined separately.
- This duplication increases maintenance overhead and risks inconsistencies.

#### **Solution**
- Move the shared schema logic into a single source of truth within the existing `DocstringValidator` and `DocstringProcessor` modules.

#### **Implementation**
1. **Refactor `DocstringValidator`**:
   - Add a method to load shared schema logic.
   - Use this method in both `DocstringValidator` and `DocstringProcessor`.

   ```python
   class DocstringValidator(SchemaValidator):
       def __init__(self, correlation_id: str | None = None):
           super().__init__(logger_name=f"{__name__}.{self.__class__.__name__}", correlation_id=correlation_id)
           self.docstring_schema = self._load_schema()

       def _load_schema(self) -> dict:
           """Load the shared docstring schema."""
           return {
               "type": "object",
               "properties": {
                   "summary": {"type": "string", "description": "A brief summary."},
                   "description": {"type": "string", "description": "Detailed description."},
                   "args": {...},  # Shared logic for args
                   "returns": {...},  # Shared logic for returns
                   "raises": {...},  # Shared logic for raises
                   "complexity": {"type": "integer", "default": 1},
               },
               "required": ["summary", "description", "returns"],
               "additionalProperties": False,
           }
   ```

2. **Update `DocstringProcessor`**:
   - Use the same `_load_schema` logic from `DocstringValidator` to ensure consistency.

---

### **2. Consolidate Parsing and Validation Logic**
#### **Problem**
- Both `DocstringProcessor` and `ResponseParser` have similar parsing and validation logic, leading to duplication.

#### **Solution**
- Move shared parsing and validation logic into utility methods within the existing modules.

#### **Implementation**
1. **Refactor `DocstringProcessor`**:
   - Add a `validate_with_schema` method to handle schema validation.
   - Use this method for both docstring validation and fallback handling.

   ```python
   class DocstringProcessor:
       def validate_with_schema(self, data: dict, schema: dict) -> tuple[bool, list[str]]:
           """Validate data against a schema."""
           try:
               validate(instance=data, schema=schema)
               return True, []
           except ValidationError as e:
               return False, [str(e)]
   ```

2. **Refactor `ResponseParser`**:
   - Use `DocstringProcessor.validate_with_schema` for validating AI responses.

   ```python
   class ResponseParser:
       def __init__(self, formatter: ResponseFormatter, docstring_processor: DocstringProcessor, ...):
           self.docstring_processor = docstring_processor

       async def parse_response(self, response: dict, expected_format: str, validate_schema: bool = True, ...):
           ...
           is_valid, errors = self.docstring_processor.validate_with_schema(parsed_content, schema)
           if not is_valid:
               return self._create_error_response("Validation failed", ...)
   ```

---

### **3. Standardize Fallback Handling**
#### **Problem**
- Both `DocstringProcessor` and `ResponseFormatter` implement fallback logic independently, leading to redundancy.

#### **Solution**
- Standardize fallback logic within each module using a shared fallback method.

#### **Implementation**
1. **Refactor `DocstringProcessor`**:
   - Add a `generate_fallback_docstring` method to handle fallback generation.

   ```python
   class DocstringProcessor:
       def generate_fallback_docstring(self, reason: str) -> dict:
           """Generate a fallback docstring."""
           return {
               "summary": "Not available",
               "description": f"Fallback due to: {reason}",
               "args": [],
               "returns": {"type": "Any", "description": "Not available"},
               "raises": [],
               "complexity": 1,
           }
   ```

2. **Refactor `ResponseFormatter`**:
   - Use `DocstringProcessor.generate_fallback_docstring` for fallback responses.

   ```python
   class ResponseFormatter:
       def format_fallback_response(self, metadata: dict, error: str = "") -> dict:
           fallback_content = self.docstring_processor.generate_fallback_docstring(error)
           return {
               "choices": [{"message": {"content": json.dumps(fallback_content)}}],
               "usage": {},
           }
   ```

---

### **4. Reuse `DocstringData` for AI Responses**
#### **Problem**
- The `ResponseParser` parses AI responses into dictionaries but does not leverage the `DocstringData` class for structured representation.

#### **Solution**
- Use `DocstringData` as the target data structure for parsed AI responses.

#### **Implementation**
1. **Update `ResponseParser`**:
   - Replace dictionary-based parsing with `DocstringData`.

   ```python
   class ResponseParser:
       async def parse_response(self, response: dict, expected_format: str, ...):
           ...
           parsed_content = DocstringData.from_dict(response["choices"][0]["message"]["content"])
           ...
   ```

2. **Update `DocstringProcessor`**:
   - Use `DocstringData.to_dict()` for validation and fallback handling.

   ```python
   class DocstringProcessor:
       def validate(self, docstring_data: DocstringData) -> tuple[bool, list[str]]:
           return self.validate_with_schema(docstring_data.to_dict(), self._load_schema())
   ```

---

### **5. Dynamically Inject Schema into Prompts**
#### **Problem**
- The `documentation_prompt.txt` and `code_analysis_prompt.txt` templates hardcode JSON schemas, which can lead to inconsistencies.

#### **Solution**
- Dynamically inject schema definitions into the templates using the existing schema logic.

#### **Implementation**
1. **Update `PromptManager`**:
   - Add a method to inject schema into templates.

   ```python
   class PromptManager:
       def inject_schema_into_prompt(self, schema: dict, template: str) -> str:
           schema_json = json.dumps(schema, indent=4)
           return template.replace("{{ schema }}", schema_json)
   ```

2. **Use in `documentation_prompt.txt`**:
   - Replace the hardcoded schema with a placeholder (`{{ schema }}`) and dynamically inject the schema.

   ```jinja
   Analyze the code and generate a JSON response adhering to this schema:
   {{ schema }}
   ```

---

### **6. Enhance Complexity Analysis**
#### **Problem**
- The `complexity` field is static and does not reflect actual code complexity.

#### **Solution**
- Integrate complexity analysis tools (e.g., `radon`) into `DocstringProcessor`.

#### **Implementation**
1. **Update `DocstringProcessor`**:
   - Add a method to calculate complexity dynamically.

   ```python
   from radon.complexity import cc_visit

   class DocstringProcessor:
       def calculate_complexity(self, source_code: str) -> int:
           """Calculate McCabe complexity."""
           return max([block.complexity for block in cc_visit(source_code)], default=1)
   ```

2. **Use in Parsing**:
   - Update `_parse_docstring_content` to include dynamic complexity calculation.

   ```python
   class DocstringProcessor:
       def _parse_docstring_content(self, docstring: str) -> dict:
           ...
           return {
               "summary": parsed_docstring.short_description or "No summary available.",
               "description": parsed_docstring.long_description or "No description provided.",
               "args": [...],
               "returns": {...},
               "raises": [...],
               "complexity": self.calculate_complexity(docstring),
           }
   ```

---

### **7. Consolidate Metrics Collection**
#### **Problem**
- Both `DocstringProcessor` and `ResponseParser` collect metrics independently, leading to duplication.

#### **Solution**
- Use the existing `MetricsCollector` for shared metrics collection.

#### **Implementation**
1. **Update `DocstringProcessor`**:
   - Use `MetricsCollector` for tracking metrics.

   ```python
   class DocstringProcessor:
       async def process_docstring(self, docstring: str) -> ProcessingResult:
           start_time = time.time()
           ...
           await self.metrics_collector.track_operation(
               operation_type="docstring_processing",
               success=True,
               duration=time.time() - start_time,
               metadata={"lines": len(docstring.splitlines()), ...},
           )
   ```

2. **Update `ResponseParser`**:
   - Use `MetricsCollector` for tracking response parsing metrics.

   ```python
   class ResponseParser:
       async def parse_response(self, response: dict, ...):
           start_time = time.time()
           ...
           await self.metrics_collector.track_operation(
               operation_type="response_parsing",
               success=True,
               duration=time.time() - start_time,
               metadata={"response_length": len(response)},
           )
   ```

---

### **Benefits of This Refactoring Plan**
1. **Reduced Redundancy**:
   - Consolidates schema definitions, parsing logic, fallback handling, and metrics collection.

2. **Improved Integration**:
   - Ensures consistent use of `DocstringData` and shared utilities across modules.

3. **Enhanced Maintainability**:
   - Centralizes shared logic, making the system easier to update and extend.

4. **Dynamic Adaptability**:
   - Enables dynamic schema injection into prompts, reducing hardcoded dependencies.

Let me know if you'd like help implementing any of these changes!
[Timestamp: 2024/12/19 01:38:43]

**ai**: Here’s a **class/method-to-module map** based on the refactoring plan. This map outlines where each class and method resides and how they interact across modules.

---

### **1. Class-to-Module Map**
| **Class**                  | **Module**               | **Purpose**                                                                 |
|----------------------------|--------------------------|-----------------------------------------------------------------------------|
| `DocstringProcessor`       | `docstring_processor`    | Handles parsing, validation, and fallback generation for docstrings.       |
| `DocstringValidator`       | `docstring_validator`    | Validates docstring data against the `docstring_schema`.                   |
| `SchemaValidator`          | `schema_validator`       | Base class for schema validation.                                          |
| `ResponseParser`           | `response_parsing`       | Parses and validates AI responses, integrates with `DocstringProcessor`.   |
| `ResponseFormatter`        | `response_formatter`     | Formats AI responses into standardized structures.                         |
| `PromptManager`            | `prompt_manager`         | Manages prompt generation and schema injection into templates.             |
| `MetricsCollector`         | Existing in all modules  | Tracks metrics for operations like parsing, validation, and processing.    |
| `DocstringData`            | `docstring`              | Represents structured docstring data and handles serialization.            |

---

### **2. Method-to-Module Map**
This section maps each method to its corresponding module and describes its purpose.

#### **2.1 DocstringProcessor**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `parse(docstring: str)`        | `docstring_processor`    | Parses a docstring into structured `DocstringData`.                        |
| `_parse_docstring_content`     | `docstring_processor`    | Internal method to parse docstring content and calculate complexity.       |
| `validate_with_schema`         | `docstring_processor`    | Validates data against a schema.                                           |
| `generate_fallback_docstring`  | `docstring_processor`    | Generates a fallback docstring when parsing fails.                         |
| `process_docstring`            | `docstring_processor`    | Processes a docstring and returns structured results with metrics.         |
| `calculate_complexity`         | `docstring_processor`    | Dynamically calculates McCabe complexity for docstrings.                   |

---

#### **2.2 DocstringValidator**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `validate_docstring`           | `docstring_validator`    | Validates docstring data against the `docstring_schema`.                    |
| `_load_schema`                 | `docstring_validator`    | Loads the shared `docstring_schema` for validation.                        |

---

#### **2.3 SchemaValidator**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `validate_schema`              | `schema_validator`       | Validates data against a JSON schema.                                       |

---

#### **2.4 ResponseParser**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `parse_response`               | `response_parsing`       | Parses and validates AI responses, integrates with `DocstringProcessor`.   |
| `_create_error_response`       | `response_parsing`       | Creates a fallback response when parsing fails.                            |

---

#### **2.5 ResponseFormatter**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `format_fallback_response`     | `response_formatter`     | Generates a fallback response for invalid AI outputs.                      |
| `standardize_response_format`  | `response_formatter`     | Standardizes AI responses into a consistent structure.                     |

---

#### **2.6 PromptManager**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `inject_schema_into_prompt`    | `prompt_manager`         | Dynamically injects schema definitions into prompt templates.              |
| `create_documentation_prompt`  | `prompt_manager`         | Generates a documentation prompt for AI using templates.                   |
| `create_code_analysis_prompt`  | `prompt_manager`         | Generates a code analysis prompt for AI using templates.                   |

---

#### **2.7 DocstringData**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `to_dict`                      | `docstring`              | Converts `DocstringData` into a dictionary for serialization.              |
| `from_dict`                    | `docstring`              | Creates a `DocstringData` instance from a dictionary.                      |
| `validate`                     | `docstring`              | Validates the fields of `DocstringData`.                                    |

---

#### **2.8 MetricsCollector**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `track_operation`              | Existing in all modules  | Tracks metrics for operations like parsing, validation, and processing.    |

---

### **3. Integration Points**
This section highlights how modules integrate with each other based on the refactored plan.

#### **3.1 DocstringProcessor and DocstringValidator**
- **Integration**: `DocstringProcessor` uses `DocstringValidator` to validate parsed docstring data.
- **Example**:
  ```python
  class DocstringProcessor:
      def validate(self, docstring_data: DocstringData) -> tuple[bool, list[str]]:
          return self.docstring_validator.validate_docstring(docstring_data.to_dict())
  ```

#### **3.2 ResponseParser and DocstringProcessor**
- **Integration**: `ResponseParser` uses `DocstringProcessor` for parsing and validating AI-generated docstrings.
- **Example**:
  ```python
  class ResponseParser:
      async def parse_response(self, response: dict, ...):
          parsed_content = DocstringData.from_dict(response["choices"][0]["message"]["content"])
          is_valid, errors = self.docstring_processor.validate(parsed_content)
  ```

#### **3.3 PromptManager and Schema Definitions**
- **Integration**: `PromptManager` dynamically injects schema definitions (e.g., `docstring_schema`) into prompt templates.
- **Example**:
  ```python
  class PromptManager:
      def inject_schema_into_prompt(self, schema: dict, template: str) -> str:
          schema_json = json.dumps(schema, indent=4)
          return template.replace("{{ schema }}", schema_json)
  ```

#### **3.4 MetricsCollector**
- **Integration**: Shared across all modules to track metrics for operations like parsing, validation, and processing.
- **Example**:
  ```python
  class DocstringProcessor:
      async def process_docstring(self, docstring: str):
          start_time = time.time()
          ...
          await self.metrics_collector.track_operation(
              operation_type="docstring_processing",
              success=True,
              duration=time.time() - start_time,
              metadata={"lines": len(docstring.splitlines()), ...},
          )
  ```

---

### **4. Summary**
This class/method-to-module map ensures:
1. **Reduced Redundancy**: Shared logic (e.g., schema validation, fallback handling) is reused across modules.
2. **Improved Integration**: Modules like `DocstringProcessor`, `ResponseParser`, and `PromptManager` work together seamlessly.
3. **Enhanced Maintainability**: Centralized methods (e.g., `validate_with_schema`, `generate_fallback_docstring`) simplify updates and debugging.

Let me know if you'd like further details or implementation examples for any specific part!
[Timestamp: 2024/12/19 01:39:35]

**user**: Is there no ai_service module anymore?
[Timestamp: 2024/12/19 01:40:10]

**ai**: You're absolutely correct to ask about the **`ai_service` module**! It is still a critical part of the system, and I should have included it in the class/method-to-module map. The `ai_service` module acts as the **orchestrator** for interacting with Azure OpenAI and manages the flow between different components like `PromptManager`, `ResponseParser`, and `DocstringProcessor`.

Let me revise the map to include the `ai_service` module and clarify its role in the refactored plan.

---

### **Revised Class/Method-to-Module Map (Including `ai_service`)**

---

### **1. Class-to-Module Map**
| **Class**                  | **Module**               | **Purpose**                                                                 |
|----------------------------|--------------------------|-----------------------------------------------------------------------------|
| `AIService`                | `ai_service`            | Orchestrates interactions with Azure OpenAI and integrates other modules.  |
| `DocstringProcessor`       | `docstring_processor`    | Handles parsing, validation, and fallback generation for docstrings.       |
| `DocstringValidator`       | `docstring_validator`    | Validates docstring data against the `docstring_schema`.                   |
| `SchemaValidator`          | `schema_validator`       | Base class for schema validation.                                          |
| `ResponseParser`           | `response_parsing`       | Parses and validates AI responses, integrates with `DocstringProcessor`.   |
| `ResponseFormatter`        | `response_formatter`     | Formats AI responses into standardized structures.                         |
| `PromptManager`            | `prompt_manager`         | Manages prompt generation and schema injection into templates.             |
| `MetricsCollector`         | Existing in all modules  | Tracks metrics for operations like parsing, validation, and processing.    |
| `DocstringData`            | `docstring`              | Represents structured docstring data and handles serialization.            |

---

### **2. Method-to-Module Map (Including `AIService`)**

#### **2.1 AIService**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `generate_documentation`       | `ai_service`            | Orchestrates the process of generating documentation using Azure OpenAI.   |
| `_make_api_call_with_retry`    | `ai_service`            | Handles API calls with retry logic for robustness.                         |
| `_add_source_code_to_response` | `ai_service`            | Adds source code metadata to AI responses.                                 |
| `_format_fallback_response`    | `ai_service`            | Formats fallback responses for invalid or incomplete AI outputs.           |
| `start`                        | `ai_service`            | Initializes the HTTP client session for API calls.                         |
| `close`                        | `ai_service`            | Closes the HTTP client session.                                            |

---

#### **2.2 DocstringProcessor**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `parse(docstring: str)`        | `docstring_processor`    | Parses a docstring into structured `DocstringData`.                        |
| `_parse_docstring_content`     | `docstring_processor`    | Internal method to parse docstring content and calculate complexity.       |
| `validate_with_schema`         | `docstring_processor`    | Validates data against a schema.                                           |
| `generate_fallback_docstring`  | `docstring_processor`    | Generates a fallback docstring when parsing fails.                         |
| `process_docstring`            | `docstring_processor`    | Processes a docstring and returns structured results with metrics.         |
| `calculate_complexity`         | `docstring_processor`    | Dynamically calculates McCabe complexity for docstrings.                   |

---

#### **2.3 DocstringValidator**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `validate_docstring`           | `docstring_validator`    | Validates docstring data against the `docstring_schema`.                    |
| `_load_schema`                 | `docstring_validator`    | Loads the shared `docstring_schema` for validation.                        |

---

#### **2.4 SchemaValidator**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `validate_schema`              | `schema_validator`       | Validates data against a JSON schema.                                       |

---

#### **2.5 ResponseParser**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `parse_response`               | `response_parsing`       | Parses and validates AI responses, integrates with `DocstringProcessor`.   |
| `_create_error_response`       | `response_parsing`       | Creates a fallback response when parsing fails.                            |

---

#### **2.6 ResponseFormatter**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `format_fallback_response`     | `response_formatter`     | Generates a fallback response for invalid AI outputs.                      |
| `standardize_response_format`  | `response_formatter`     | Standardizes AI responses into a consistent structure.                     |

---

#### **2.7 PromptManager**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `inject_schema_into_prompt`    | `prompt_manager`         | Dynamically injects schema definitions into prompt templates.              |
| `create_documentation_prompt`  | `prompt_manager`         | Generates a documentation prompt for AI using templates.                   |
| `create_code_analysis_prompt`  | `prompt_manager`         | Generates a code analysis prompt for AI using templates.                   |

---

#### **2.8 DocstringData**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `to_dict`                      | `docstring`              | Converts `DocstringData` into a dictionary for serialization.              |
| `from_dict`                    | `docstring`              | Creates a `DocstringData` instance from a dictionary.                      |
| `validate`                     | `docstring`              | Validates the fields of `DocstringData`.                                    |

---

#### **2.9 MetricsCollector**
| **Method**                     | **Module**               | **Purpose**                                                                 |
|--------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `track_operation`              | Existing in all modules  | Tracks metrics for operations like parsing, validation, and processing.    |

---

### **3. Integration Points for `AIService`**
The `AIService` module serves as the **orchestrator** and integrates with other modules as follows:

#### **3.1 Integration with `PromptManager`**
- **Purpose**: Generates prompts for Azure OpenAI using templates.
- **Example**:
  ```python
  class AIService:
      async def generate_documentation(self, context: DocumentationContext, schema: dict = None):
          prompt_result = await self.prompt_manager.create_documentation_prompt(context=context)
          prompt = self.prompt_manager.inject_schema_into_prompt(schema, prompt_result.content["prompt"])
          response = await self._make_api_call_with_retry(prompt, schema)
          ...
  ```

#### **3.2 Integration with `ResponseParser`**
- **Purpose**: Parses and validates AI responses.
- **Example**:
  ```python
  class AIService:
      async def generate_documentation(self, ...):
          response = await self._make_api_call_with_retry(...)
          parsed_response = await self.response_parser.parse_response(response, expected_format="docstring")
          ...
  ```

#### **3.3 Integration with `DocstringProcessor`**
- **Purpose**: Validates and processes docstrings after parsing AI responses.
- **Example**:
  ```python
  class AIService:
      async def generate_documentation(self, ...):
          parsed_response = await self.response_parser.parse_response(...)
          is_valid, errors = self.docstring_processor.validate(parsed_response.content)
          ...
  ```

#### **3.4 Integration with `MetricsCollector`**
- **Purpose**: Tracks metrics for API calls and processing operations.
- **Example**:
  ```python
  class AIService:
      async def generate_documentation(self, ...):
          start_time = time.time()
          ...
          await self.metrics_collector.track_operation(
              operation_type="documentation_generation",
              success=True,
              duration=time.time() - start_time,
              metadata={"tokens": response.get("usage", {})},
          )
  ```

---

### **4. Summary**
The `AIService` module remains a critical part of the system, acting as the **central orchestrator** for:
- Generating prompts (`PromptManager`).
- Making API calls and handling retries.
- Parsing and validating AI responses (`ResponseParser` and `DocstringProcessor`).
- Tracking metrics (`MetricsCollector`).

By integrating with other modules, `AIService` ensures that the system operates cohesively. Let me know if you'd like further details on any specific integration or method!
[Timestamp: 2024/12/19 01:40:51]

**user**: Outline each refactored module with imports, classes and methods defined in each
[Timestamp: 2024/12/19 01:41:21]

**ai**: Here’s a detailed outline of each **refactored module** with its imports, classes, and methods. This outline reflects the refactoring plan while maintaining the existing module structure.

---

### **1. `ai_service` Module**
#### **Purpose**
Acts as the orchestrator for interactions with Azure OpenAI, integrating with other modules like `PromptManager`, `ResponseParser`, and `DocstringProcessor`.

#### **Imports**
```python
import asyncio
import json
import time
from typing import Any, Dict, Optional

from aiohttp.client import ClientSession, ClientTimeout
from openai import AzureOpenAI

from prompt_manager import PromptManager
from response_parsing import ResponseParser
from docstring_processor import DocstringProcessor
from core.types.base import DocumentationContext, ProcessingResult
from core.metrics_collector import MetricsCollector
from core.exceptions import APICallError
```

#### **Classes and Methods**
```python
class AIService:
    def __init__(self, config: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None):
        """Initialize the AI Service with Azure OpenAI configurations."""
        ...

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        ...

    async def close(self) -> None:
        """Close the aiohttp client session."""
        ...

    async def generate_documentation(self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Generate documentation for the provided source code context."""
        ...

    async def _make_api_call_with_retry(self, prompt: str, schema: Optional[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        ...

    def _add_source_code_to_response(self, response: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Add source code metadata to the entire response structure."""
        ...

    def _format_fallback_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format a fallback response for invalid structures."""
        ...
```

---

### **2. `docstring_processor` Module**
#### **Purpose**
Handles parsing, validation, and fallback generation for docstrings.

#### **Imports**
```python
import json
import time
from typing import Any, Dict

from docstring_parser import parse, DocstringStyle
from core.validation.schema_validator import SchemaValidator
from core.types.docstring import DocstringData
from core.types.base import ProcessingResult
from core.metrics_collector import MetricsCollector
from core.exceptions import DataValidationError
```

#### **Classes and Methods**
```python
class DocstringProcessor:
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the DocstringProcessor."""
        ...

    def parse(self, docstring: str) -> DocstringData:
        """Parses a docstring string into structured data."""
        ...

    def _parse_docstring_content(self, docstring: str) -> Dict[str, Any]:
        """Parses docstring content into a structured dictionary."""
        ...

    def validate_with_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validates data against a schema."""
        ...

    def generate_fallback_docstring(self, reason: str) -> Dict[str, Any]:
        """Generates a fallback docstring when parsing fails."""
        ...

    async def process_docstring(self, docstring: str) -> ProcessingResult:
        """Process a docstring and return structured results."""
        ...

    def calculate_complexity(self, source_code: str) -> int:
        """Dynamically calculates McCabe complexity for docstrings."""
        ...
```

---

### **3. `docstring_validator` Module**
#### **Purpose**
Validates docstring data against the `docstring_schema`.

#### **Imports**
```python
import json
from typing import Any, Dict, List
from pathlib import Path

from core.validation.schema_validator import SchemaValidator
from core.logger import LoggerSetup
```

#### **Classes and Methods**
```python
class DocstringValidator(SchemaValidator):
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the docstring validator."""
        ...

    def _load_schema(self) -> Dict[str, Any]:
        """Load the shared docstring schema."""
        ...

    def validate_docstring(self, content: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the content against the docstring schema."""
        ...
```

---

### **4. `schema_validator` Module**
#### **Purpose**
Provides a base class for validating data against JSON schemas.

#### **Imports**
```python
from typing import Any, List, Dict
from jsonschema import validate, ValidationError
from core.logger import LoggerSetup
```

#### **Classes and Methods**
```python
class SchemaValidator:
    def __init__(self, logger_name: str, correlation_id: Optional[str] = None):
        """Initialize the schema validator."""
        ...

    def validate_schema(self, instance: Any, schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the instance against the schema."""
        ...
```

---

### **5. `response_parsing` Module**
#### **Purpose**
Parses and validates AI responses, integrates with `DocstringProcessor`.

#### **Imports**
```python
import json
from typing import Any, Dict, Optional

from core.types.base import ParsedResponse
from core.formatting.response_formatter import ResponseFormatter
from docstring_processor import DocstringProcessor
```

#### **Classes and Methods**
```python
class ResponseParser:
    def __init__(self, formatter: ResponseFormatter, docstring_processor: DocstringProcessor, correlation_id: Optional[str] = None):
        """Initialize the ResponseParser."""
        ...

    async def parse_response(self, response: Dict[str, Any], expected_format: str, validate_schema: bool = True) -> ParsedResponse:
        """Parses and validates an AI response."""
        ...

    def _create_error_response(self, error: str, expected_format: str) -> ParsedResponse:
        """Creates a fallback response when parsing fails."""
        ...
```

---

### **6. `response_formatter` Module**
#### **Purpose**
Formats AI responses into standardized structures.

#### **Imports**
```python
import json
from typing import Any, Dict, Optional
from core.logger import LoggerSetup
```

#### **Classes and Methods**
```python
class ResponseFormatter:
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the response formatter."""
        ...

    def format_fallback_response(self, metadata: Dict[str, Any], error: str = "") -> Dict[str, Any]:
        """Generates a fallback response for invalid AI outputs."""
        ...

    def standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardizes AI responses into a consistent structure."""
        ...
```

---

### **7. `prompt_manager` Module**
#### **Purpose**
Manages prompt generation and schema injection into templates.

#### **Imports**
```python
import json
from typing import Dict, Optional
from jinja2 import Environment, FileSystemLoader
from core.types.base import DocumentationContext
```

#### **Classes and Methods**
```python
class PromptManager:
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the PromptManager with template loading and configuration."""
        ...

    def inject_schema_into_prompt(self, schema: Dict[str, Any], template: str) -> str:
        """Dynamically injects schema definitions into prompt templates."""
        ...

    async def create_documentation_prompt(self, context: DocumentationContext) -> Dict[str, Any]:
        """Generates a documentation prompt for AI using templates."""
        ...

    async def create_code_analysis_prompt(self, source_code: str) -> Dict[str, Any]:
        """Generates a code analysis prompt for AI using templates."""
        ...
```

---

### **8. `docstring` Module**
#### **Purpose**
Represents structured docstring data and handles serialization.

#### **Imports**
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
```

#### **Classes and Methods**
```python
@dataclass
class DocstringData:
    summary: str
    description: str
    args: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, str] = field(default_factory=lambda: {"type": "Any", "description": ""})
    raises: List[Dict[str, str]] = field(default_factory=list)
    complexity: int = 1

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate docstring data."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocstringData:
        """Create a DocstringData instance from a dictionary."""
        ...
```

---

### **9. Shared Utility: MetricsCollector**
#### **Purpose**
Tracks metrics for operations like parsing, validation, and processing.

#### **Usage Across Modules**
- **`DocstringProcessor`**: Tracks docstring parsing and validation metrics.
- **`ResponseParser`**: Tracks response parsing metrics.
- **`AIService`**: Tracks API call and processing metrics.

---

### **Conclusion**
This outline organizes the refactored modules, ensuring:
1. **Clear Responsibilities**: Each module has a distinct purpose.
2. **Reduced Redundancy**: Shared logic (e.g., schema validation, fallback handling) is reused across modules.
3. **Improved Integration**: Modules like `AIService`, `PromptManager`, and `DocstringProcessor` work seamlessly together.

Let me know if you'd like further details or implementation examples for any specific module or method!
[Timestamp: 2024/12/19 01:42:15]

**user**: Generate the complete ai_service module
[Timestamp: 2024/12/19 01:42:32]

**ai**: Here’s the complete **`ai_service` module** based on the refactoring plan. This module acts as the orchestrator for interacting with Azure OpenAI and integrates with other modules like `PromptManager`, `ResponseParser`, and `DocstringProcessor`.

---

### **`ai_service.py`**
```python
"""AI service module for handling Azure OpenAI API interactions."""

import asyncio
import json
import time
from typing import Any, Dict, Optional

from aiohttp.client import ClientSession, ClientTimeout
from openai import AzureOpenAI

from prompt_manager import PromptManager
from response_parsing import ResponseParser
from docstring_processor import DocstringProcessor
from core.types.base import DocumentationContext, ProcessingResult
from core.metrics_collector import MetricsCollector
from core.exceptions import APICallError, DocumentationError


class AIService:
    """
    Manages interactions with the Azure OpenAI API.

    This service handles API calls, response formatting, and error handling
    for the Azure OpenAI API. It uses asynchronous operations, retry logic,
    and structured data handling to ensure reliable and efficient communication
    with the AI model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the AI Service with Azure OpenAI configurations.

        :param config: Optional dictionary with Azure OpenAI details.
        :param correlation_id: Optional correlation ID for logging context.
        """
        self.config = config or {
            "endpoint": "https://your-azure-endpoint.openai.azure.com/",
            "api_key": "your-api-key",
            "azure_api_version": "2024-10-01-preview",
            "model": "gpt-4",
            "deployment": "your-deployment-name",
            "timeout": 60,
            "max_retries": 3,
        }
        self.correlation_id = correlation_id

        # Initialize logger
        from core.logger import LoggerSetup
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

        # Initialize dependencies
        self.prompt_manager = PromptManager(correlation_id=correlation_id)
        self.response_parser = ResponseParser(
            formatter=None,  # Replace with a ResponseFormatter instance if needed
            docstring_processor=DocstringProcessor(correlation_id=correlation_id),
            correlation_id=correlation_id,
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.client: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the aiohttp client session if not already started."""
        if self.client is None:
            self.client = ClientSession()
            self.logger.info("AI Service client session initialized")

    async def close(self) -> None:
        """Close the aiohttp client session."""
        if self.client:
            await self.client.close()
            self.client = None
            self.logger.info("AI Service client session closed")

    async def generate_documentation(
        self, context: DocumentationContext, schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Generate documentation for the provided source code context.

        :param context: A DocumentationContext object containing source code and metadata.
        :param schema: Optional function schema to influence the AI's response format.
        :return: A ProcessingResult with parsed and validated documentation content.
        """
        start_time = time.time()
        log_extra = {
            "correlation_id": self.correlation_id,
            "module": context.metadata.get("module_name", ""),
            "file": context.metadata.get("file_path", ""),
        }

        try:
            # Validate input
            if not context.source_code or not context.source_code.strip():
                raise DocumentationError("Source code is missing or empty")

            # Create documentation prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(context=context)

            # Add function calling instructions if schema is provided
            if schema:
                prompt = self.prompt_manager.inject_schema_into_prompt(schema, prompt_result["prompt"])
            else:
                prompt = prompt_result["prompt"]

            # Make API call with retry logic
            response = await self._make_api_call_with_retry(
                prompt=prompt,
                schema=schema,
                max_retries=self.config["max_retries"],
            )

            # Log the raw response before validation
            self.logger.debug(f"Raw AI response: {response}", extra=log_extra)

            # Parse and validate response
            parsed_response = await self.response_parser.parse_response(
                response, expected_format="docstring", validate_schema=True
            )

            # Track metrics based on validation success
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=parsed_response.validation_success,
                duration=processing_time,
                metadata={
                    "module": context.metadata.get("module_name", ""),
                    "file": str(context.module_path),
                    "tokens": response.get("usage", {}),
                    "validation_success": parsed_response.validation_success,
                    "errors": parsed_response.errors if not parsed_response.validation_success else None,
                },
            )

            # Return ProcessingResult with validation status and any errors
            return ProcessingResult(
                content=parsed_response.content,  # Use the content even if validation failed
                usage=response.get("usage", {}),
                metrics={"processing_time": processing_time},
                validation_status=parsed_response.validation_success,
                validation_errors=parsed_response.errors or [],
                schema_errors=[],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            self.logger.error(
                f"Documentation generation failed: {e}", exc_info=True, extra=log_extra
            )
            raise

    async def _make_api_call_with_retry(
        self, prompt: str, schema: Optional[Dict[str, Any]], max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make an API call with retry logic following Azure best practices.

        :param prompt: The prompt to send to the Azure OpenAI API.
        :param schema: Optional schema for function calling.
        :param max_retries: Maximum number of retries for the API call.
        :return: The API response as a dictionary.
        """
        headers = {"api-key": self.config["api_key"], "Content-Type": "application/json"}
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        if schema:
            request_params["tools"] = [{"type": "function", "function": schema}]
            request_params["tool_choice"] = "auto"

        url = f"{self.config['endpoint'].rstrip('/')}/openai/deployments/{self.config['deployment']}/chat/completions?api-version={self.config['azure_api_version']}"

        for attempt in range(max_retries):
            try:
                if self.client is None:
                    await self.start()

                self.logger.info(
                    "Making API call",
                    extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                )

                async with self.client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=ClientTimeout(total=self.config["timeout"]),
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        return response_json

                    # Handle specific error cases
                    error_text = await response.text()
                    self.logger.error(
                        f"API call failed with status {response.status}: {error_text}",
                        extra={"correlation_id": self.correlation_id},
                    )

                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                        await asyncio.sleep(retry_after)
                    elif response.status == 503:  # Service unavailable
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Error during API call: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    raise APICallError(f"API call failed after {max_retries} retries: {e}")
                await asyncio.sleep(2 ** attempt)

        raise APICallError(f"API call failed after {max_retries} retries")

    def _add_source_code_to_response(self, response: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Add source code metadata to the entire response structure."""
        response["source_code"] = source_code
        return response

    def _format_fallback_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format a fallback response for invalid structures."""
        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "args": [],
            "returns": {"type": "Any", "description": "No return value description provided."},
            "raises": [],
            "complexity": 1,
        }
        return {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": {},
        }
```

---

### **Key Features**
1. **Retry Logic**: Implements exponential backoff for API calls.
2. **Integration**: Connects seamlessly with `PromptManager`, `ResponseParser`, and `DocstringProcessor`.
3. **Metrics Tracking**: Tracks operation metrics using `MetricsCollector`.
4. **Fallback Handling**: Provides fallback responses for invalid or incomplete AI outputs.
5. **Dynamic Schema Injection**: Supports injecting schemas into prompts for flexible functionality.

Let me know if you need further refinements or additional features!
[Timestamp: 2024/12/19 01:43:29]

**user**: Generate the complete response_parsing module
[Timestamp: 2024/12/19 01:43:38]

**ai**: Here’s the complete **`response_parsing` module** based on the refactoring plan. This module handles parsing and validating AI responses, integrates with `DocstringProcessor`, and ensures that responses adhere to expected formats.

---

### **`response_parsing.py`**
```python
"""
Response parsing service for handling Azure OpenAI API outputs.

This module provides comprehensive parsing and validation of Azure OpenAI API responses,
with support for structured outputs, function calling, and error handling according to
Azure best practices.
"""

import json
import time
from typing import Any, Dict, Optional

from core.types.base import ParsedResponse
from core.formatting.response_formatter import ResponseFormatter
from docstring_processor import DocstringProcessor
from core.logger import LoggerSetup


class ResponseParser:
    """
    Parses and validates AI responses.

    This class handles the parsing of AI responses into structured formats,
    validates them against schemas, and provides fallback mechanisms for invalid responses.
    """

    def __init__(
        self,
        formatter: ResponseFormatter,
        docstring_processor: DocstringProcessor,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the ResponseParser.

        :param formatter: Instance of ResponseFormatter for formatting responses.
        :param docstring_processor: Instance of DocstringProcessor for validating and processing docstrings.
        :param correlation_id: Optional correlation ID for logging context.
        """
        self.formatter = formatter
        self.docstring_processor = docstring_processor
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
        )

    async def parse_response(
        self,
        response: Dict[str, Any],
        expected_format: str,
        validate_schema: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedResponse:
        """
        Parses and validates an AI response.

        :param response: The raw response from the AI model.
        :param expected_format: The expected format of the response (e.g., "docstring").
        :param validate_schema: Whether to validate the response against a schema.
        :param metadata: Additional metadata to include in the response.
        :return: A ParsedResponse object containing the parsed content, validation status, and any errors.
        """
        start_time = time.time()
        metadata = metadata or {}

        self.logger.debug(
            f"Raw AI response before parsing: {response}",
            extra={"correlation_id": self.correlation_id},
        )

        try:
            # Validate basic response structure
            if "choices" not in response or not response["choices"]:
                return self._create_error_response("No choices in response", expected_format, start_time, metadata)

            message = response["choices"][0].get("message", {})
            if not message or "content" not in message:
                return self._create_error_response("No content in response message", expected_format, start_time, metadata)

            # Parse message content
            content = message["content"]
            parsed_content = self._parse_content(content, expected_format, validate_schema, metadata)

            # Return parsed response
            return ParsedResponse(
                content=parsed_content["content"],
                format_type=expected_format,
                parsing_time=time.time() - start_time,
                validation_success=parsed_content["validation_success"],
                errors=parsed_content["errors"],
                metadata=metadata,
            )

        except Exception as e:
            return self._handle_unexpected_error(e, expected_format, start_time, metadata)

    def _parse_content(
        self,
        content: str,
        expected_format: str,
        validate_schema: bool,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parses the message content based on the expected format.

        :param content: The raw content from the AI response.
        :param expected_format: The expected format of the content (e.g., "docstring").
        :param validate_schema: Whether to validate the content against a schema.
        :param metadata: Additional metadata for validation.
        :return: A dictionary containing the parsed content, validation status, and any errors.
        """
        if not content:
            return {"content": {}, "validation_success": False, "errors": ["Content is empty"]}

        try:
            # Parse content as JSON
            parsed_content = json.loads(content)

            # Validate content if required
            if validate_schema and expected_format == "docstring":
                is_valid, errors = self.docstring_processor.validate_with_schema(
                    parsed_content, self.docstring_processor.schema_validator._load_schema()
                )
                return {
                    "content": parsed_content,
                    "validation_success": is_valid,
                    "errors": errors,
                }

            return {"content": parsed_content, "validation_success": True, "errors": []}

        except json.JSONDecodeError as e:
            self.logger.warning(
                f"JSONDecodeError while parsing content: {e}",
                extra={"correlation_id": self.correlation_id, "content": content[:50]},
            )
            return {"content": {}, "validation_success": False, "errors": [f"JSONDecodeError: {str(e)}"]}

    def _create_error_response(
        self,
        error: str,
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """
        Creates a standardized error response.

        :param error: The error message.
        :param expected_format: The expected format of the response.
        :param start_time: The start time of the parsing operation.
        :param metadata: Additional metadata for the response.
        :return: A ParsedResponse object with the error details.
        """
        fallback_response = self.formatter.format_fallback_response(metadata, error)
        return ParsedResponse(
            content=fallback_response,
            format_type=expected_format,
            parsing_time=time.time() - start_time,
            validation_success=False,
            errors=[error],
            metadata=metadata,
        )

    def _handle_unexpected_error(
        self,
        error: Exception,
        expected_format: str,
        start_time: float,
        metadata: Dict[str, Any],
    ) -> ParsedResponse:
        """
        Handles unexpected errors during parsing.

        :param error: The exception that occurred.
        :param expected_format: The expected format of the response.
        :param start_time: The start time of the parsing operation.
        :param metadata: Additional metadata for the response.
        :return: A ParsedResponse object with the error details.
        """
        self.logger.error(
            f"Unexpected error during parsing: {error}",
            exc_info=True,
            extra={"correlation_id": self.correlation_id},
        )
        return self._create_error_response(
            error=f"Unexpected error: {str(error)}",
            expected_format=expected_format,
            start_time=start_time,
            metadata=metadata,
        )
```

---

### **Key Features**
1. **Parsing AI Responses**:
   - Parses the `content` field from AI responses.
   - Supports JSON parsing and validation against schemas.

2. **Validation Integration**:
   - Uses `DocstringProcessor` for schema validation of parsed content.
   - Ensures responses adhere to the `docstring_schema`.

3. **Error Handling**:
   - Provides detailed error responses for invalid or incomplete AI outputs.
   - Handles unexpected errors gracefully and logs them for debugging.

4. **Fallback Mechanisms**:
   - Generates fallback responses using `ResponseFormatter` for invalid or missing data.

5. **Metadata Support**:
   - Includes metadata in parsed responses for better traceability and debugging.

---

### **Example Usage**
```python
from response_parsing import ResponseParser
from response_formatter import ResponseFormatter
from docstring_processor import DocstringProcessor

# Initialize dependencies
formatter = ResponseFormatter(correlation_id="example-correlation-id")
docstring_processor = DocstringProcessor(correlation_id="example-correlation-id")
parser = ResponseParser(formatter=formatter, docstring_processor=docstring_processor)

# Example AI response
response = {
    "choices": [
        {
            "message": {
                "content": '{"summary": "This is a test.", "description": "Detailed description."}'
            }
        }
    ]
}

# Parse the response
parsed_response = asyncio.run(parser.parse_response(response, expected_format="docstring"))
print(parsed_response)
```

---

### **Benefits of This Implementation**
1. **Modular Design**:
   - The `ResponseParser` integrates seamlessly with `DocstringProcessor` and `ResponseFormatter`.

2. **Robust Error Handling**:
   - Handles missing fields, invalid JSON, and unexpected errors gracefully.

3. **Schema Validation**:
   - Ensures AI responses adhere to the expected structure using `docstring_schema`.

4. **Extensibility**:
   - Can easily support additional formats (e.g., `text`, `function_call`) by extending `_parse_content`.

Let me know if you’d like further refinements or additional features!
[Timestamp: 2024/12/19 01:44:22]

**user**: Generate the complete response_formatter.py
[Timestamp: 2024/12/19 01:44:30]

**ai**: Here’s the complete **`response_formatter.py`** module based on the refactoring plan. This module is responsible for formatting AI responses into standardized structures and providing fallback mechanisms for invalid or incomplete responses.

---

### **`response_formatter.py`**
```python
"""Response formatter class."""

import json
from typing import Any, Dict, Optional

from core.logger import LoggerSetup


class ResponseFormatter:
    """
    Formats responses from the AI model into a standardized structure.

    This class handles formatting valid responses, generating fallback responses,
    and standardizing response formats for consistency.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the response formatter.

        :param correlation_id: Optional string for correlation purposes in logging.
        """
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

    def format_summary_description_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response that contains a summary or description by wrapping it into a
        standardized structure with choices.

        :param response: The raw response dict.
        :return: A dict representing the standardized response.
        """
        formatted = {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted summary/description response: {formatted}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted

    def format_function_call_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response that contains a function call into a standardized structure.

        :param response: The raw response dict containing a "function_call" key.
        :return: A dict with standardized structure focusing on the function call.
        """
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_tool_calls_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response that contains tool calls into a standardized structure.

        :param response: The raw response dict containing "tool_calls".
        :return: A dict with standardized structure focusing on the tool calls.
        """
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_fallback_response(
        self, metadata: Dict[str, Any], error: str = ""
    ) -> Dict[str, Any]:
        """
        Create a fallback response structure when the incoming response is invalid or
        does not match expected formats.

        :param metadata: Metadata to include in the fallback response.
        :param error: Optional error message describing the issue.
        :return: A standardized fallback response dict.
        """
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"metadata": metadata, "correlation_id": self.correlation_id},
        )
        fallback_content: Dict[str, Any] = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "error": error,
            "args": [],
            "returns": {"type": "Any", "description": "No return description provided"},
            "raises": [],
            "complexity": 1,
        }

        fallback_response = {
            "choices": [{"message": {"content": json.dumps(fallback_content)}}],
            "usage": {},
        }

        self.logger.debug(
            f"Formatted fallback response: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return fallback_response

    def standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize the response format to ensure it uses the 'choices' structure with
        a 'message' containing 'content', 'function_call', or 'tool_calls'.

        :param response: The raw response dict to standardize.
        :return: A standardized response dict.
        """
        try:
            # Case 1: Already in choices format
            if "choices" in response and isinstance(response["choices"], list):
                return response

            # Case 2: Direct content format with summary/description
            if (
                "summary" in response
                or "description" in response
            ) and response.get("summary", "").strip():
                standardized = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": response.get("summary", ""),
                                        "description": response.get("description", ""),
                                        "args": response.get("args", []),
                                        "returns": response.get(
                                            "returns", {"type": "Any", "description": ""}
                                        ),
                                        "raises": response.get("raises", []),
                                        "complexity": response.get("complexity", 1),
                                        # Preserve any other fields
                                        **{
                                            k: v
                                            for k, v in response.items()
                                            if k
                                            not in [
                                                "summary",
                                                "description",
                                                "args",
                                                "returns",
                                                "raises",
                                                "complexity",
                                            ]
                                        },
                                    }
                                )
                            }
                        }
                    ],
                    "usage": response.get("usage", {}),
                }
                self.logger.debug(
                    f"Standardized direct format response: {standardized}",
                    extra={"correlation_id": self.correlation_id},
                )
                return standardized

            # Case 3: Unknown format, fallback
            self.logger.warning(
                "Unknown response format, using fallback.",
                extra={"correlation_id": self.correlation_id},
            )
            return self.format_fallback_response({}, "Unknown response format")

        except json.JSONDecodeError as jde:
            self.logger.error(
                f"JSON decode error when standardizing response: {jde}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True,
            )
            return self.format_fallback_response({}, str(jde))
        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True,
            )
            return self.format_fallback_response({}, str(e))
```

---

### **Key Features**
1. **Formatting Valid Responses**:
   - Formats AI responses for different use cases, including:
     - Summary/description responses.
     - Function call responses.
     - Tool call responses.

2. **Fallback Handling**:
   - Generates fallback responses for invalid or incomplete AI outputs.
   - Includes error details in the fallback response for debugging.

3. **Standardization**:
   - Ensures all responses are standardized into a consistent structure for downstream processing.
   - Handles unknown or malformed responses gracefully.

4. **Logging**:
   - Logs detailed information about response formatting and fallback generation.
   - Includes correlation IDs for traceability.

---

### **Example Usage**
```python
from response_formatter import ResponseFormatter

# Initialize the formatter
formatter = ResponseFormatter(correlation_id="example-correlation-id")

# Example valid response
valid_response = {
    "summary": "This is a test summary.",
    "description": "This is a detailed description.",
    "args": [{"name": "arg1", "type": "str", "description": "An argument"}],
    "returns": {"type": "str", "description": "A return value"},
    "raises": [],
    "complexity": 1,
}

# Format the valid response
formatted_response = formatter.format_summary_description_response(valid_response)
print("Formatted Response:", formatted_response)

# Example invalid response
invalid_response = {"unexpected_field": "unexpected_value"}

# Standardize the invalid response
standardized_response = formatter.standardize_response_format(invalid_response)
print("Standardized Response:", standardized_response)

# Generate a fallback response
fallback_response = formatter.format_fallback_response(
    metadata={"module": "example_module"}, error="Invalid structure"
)
print("Fallback Response:", fallback_response)
```

---

### **Benefits of This Implementation**
1. **Consistency**:
   - Ensures all responses are formatted consistently, reducing downstream processing complexity.

2. **Error Resilience**:
   - Handles invalid or unknown response formats gracefully with fallback mechanisms.

3. **Extensibility**:
   - Can easily add support for new response types (e.g., streaming responses) by extending the formatting methods.

4. **Traceability**:
   - Logs detailed information about response formatting, including correlation IDs for debugging.

Let me know if you’d like further refinements or additional features!
[Timestamp: 2024/12/19 01:45:11]

**user**: Generate the complete prompt_manager module
[Timestamp: 2024/12/19 01:45:13]

**ai**: Here’s the complete **`prompt_manager` module** based on the refactoring plan. This module is responsible for managing prompt generation, injecting schemas into templates, and ensuring consistency in the prompts sent to the Azure OpenAI API.

---

### **`prompt_manager.py`**
```python
"""
Prompt Manager for generating and managing prompts for Azure OpenAI API interactions.

This module handles creating and managing prompts for the Azure OpenAI API,
including support for function calling and structured outputs. It ensures
prompts are optimized for the model and handles templates according to
Azure best practices.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, Template
from core.types.base import DocumentationContext, ProcessingResult
from core.logger import LoggerSetup


class PromptManager:
    """
    Manages the generation and formatting of prompts for AI interactions.

    This class handles creating and managing prompts for the Azure OpenAI API,
    including support for function calling and structured outputs.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the PromptManager with template loading and configuration.

        :param correlation_id: Optional correlation ID for logging context.
        """
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )

        # Load templates using Jinja2
        template_dir = Path(__file__).parent / "templates"
        try:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )

            # Load specific template files
            self.documentation_template = self._load_template("documentation_prompt.txt")
            self.code_analysis_template = self._load_template("code_analysis_prompt.txt")

            self.logger.info("Templates loaded successfully")
        except Exception as e:
            self.logger.error(f"Template loading failed: {e}", exc_info=True)
            raise

    def _load_template(self, template_name: str) -> Template:
        """
        Load and validate a template file.

        :param template_name: The name of the template file.
        :return: A Jinja2 Template object.
        """
        try:
            template = self.env.get_template(template_name)

            # Validate template content
            rendered = template.render(
                {
                    "module_name": "TEST_MODULE",
                    "file_path": "TEST_PATH",
                    "source_code": "TEST_CODE",
                }
            )

            if not rendered or len(rendered) < 100:
                raise ValueError(
                    f"Template {template_name} appears to be empty or invalid"
                )

            return template

        except TemplateNotFound:
            self.logger.error(f"Template file not found: {template_name}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error loading template {template_name}: {e}", exc_info=True
            )
            raise

    def inject_schema_into_prompt(self, schema: Dict[str, Any], template: str) -> str:
        """
        Dynamically injects schema definitions into prompt templates.

        :param schema: The schema to inject into the prompt.
        :param template: The base template string.
        :return: The template with the schema injected.
        """
        try:
            schema_json = json.dumps(schema, indent=4)
            return template.replace("{{ schema }}", schema_json)
        except Exception as e:
            self.logger.error(f"Error injecting schema into prompt: {e}", exc_info=True)
            raise

    async def create_documentation_prompt(
        self, context: DocumentationContext
    ) -> Dict[str, Any]:
        """
        Create a documentation prompt using the documentation template.

        :param context: A DocumentationContext object containing source code and metadata.
        :return: A dictionary containing the generated prompt and associated metadata.
        """
        try:
            # Render the template with the provided context
            prompt = self.documentation_template.render(
                module_name=context.metadata.get("module_name", ""),
                file_path=str(context.module_path),
                source_code=context.source_code,
                classes=context.classes,
                functions=context.functions,
            )

            return {"prompt": prompt}

        except Exception as e:
            self.logger.error(
                f"Error generating documentation prompt: {e}", exc_info=True
            )
            raise

    async def create_code_analysis_prompt(
        self, source_code: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a code analysis prompt using the code analysis template.

        :param source_code: The source code to analyze.
        :param context: Optional context for the analysis.
        :return: A dictionary containing the generated prompt and associated metadata.
        """
        try:
            # Render the template with the provided source code and context
            prompt = self.code_analysis_template.render(
                code=source_code,
                context=context
                or "This code is part of a documentation generation system.",
            )

            return {"prompt": prompt}

        except Exception as e:
            self.logger.error(
                f"Error generating code analysis prompt: {e}", exc_info=True
            )
            raise
```

---

### **Key Features**
1. **Template Management**:
   - Uses Jinja2 to load and render templates for documentation and code analysis prompts.
   - Validates templates to ensure they are not empty or invalid.

2. **Dynamic Schema Injection**:
   - Dynamically injects JSON schemas into templates for flexibility and consistency.

3. **Prompt Generation**:
   - Generates prompts for documentation and code analysis using structured templates.

4. **Error Handling**:
   - Logs detailed errors during template loading, schema injection, and prompt generation.

---

### **Template Files**
The `PromptManager` relies on two template files: `documentation_prompt.txt` and `code_analysis_prompt.txt`. These templates should be stored in a `templates` directory within the same folder as `prompt_manager.py`.

#### **`documentation_prompt.txt`**
```jinja
Objective: Generate comprehensive Google-style documentation for the following Python module.

Context: This module is part of a larger system aimed at providing AI-driven solutions. The target audience includes developers who will use this documentation to understand and maintain the code.

Module Name: {{ module_name }}
File Path: {{ file_path }}

Code Structure:

{% if classes %}
Classes:
{% for cls in classes %}
{{ cls }}
{% endfor %}
{% endif %}

{% if functions %}
Functions:
{% for func in functions %}
{{ func }}
{% endfor %}
{% endif %}

Source Code:
{{ source_code }}

Analyze the code and generate a JSON response adhering to this schema:
{{ schema }}

Notes:
- Ensure the response matches the schema exactly, including required fields (`summary`, `description`, `returns`, `complexity`).
- If any data is unavailable, explicitly state "Not available."
- Optional fields like `args` and `raises` can default to empty arrays.

Respond only with the JSON object.
```

#### **`code_analysis_prompt.txt`**
```jinja
Objective: Analyze the following code for quality and provide specific improvements.

Context: This code is part of a critical system component where performance and reliability are paramount. Consider historical issues such as performance bottlenecks and error handling failures.

Code:
{{ code }}

Consider the following aspects:
1. Code complexity and readability
2. Best practices and design patterns
3. Error handling and edge cases
4. Performance considerations
5. Documentation completeness

Respond in the following JSON format:
{
    "overview": "A high-level summary of the code's functionality",
    "analysis": {
        "complexity": "Assessment of complexity with examples",
        "best_practices": "Areas where best practices are followed or lacking",
        "error_handling": "Analysis of error handling and edge cases",
        "performance": "Potential performance bottlenecks or optimizations",
        "documentation": "Assessment of documentation quality"
    },
    "recommendations": [
        {
            "issue": "Brief description of the issue",
            "suggestion": "Detailed suggestion for improvement"
        }
    ]
}

Ensure your response provides actionable insights with specific examples. If any data is missing or unavailable, explicitly state "Not available."
```

---

### **Example Usage**
```python
from prompt_manager import PromptManager
from core.types.base import DocumentationContext

# Initialize the PromptManager
prompt_manager = PromptManager(correlation_id="example-correlation-id")

# Example context for documentation
context = DocumentationContext(
    source_code="def example_function(): pass",
    module_path="example_module.py",
    metadata={"module_name": "example_module"},
    classes=[],
    functions=[{"name": "example_function", "docstring": "An example function"}],
)

# Generate a documentation prompt
documentation_prompt = asyncio.run(prompt_manager.create_documentation_prompt(context))
print("Documentation Prompt:", documentation_prompt)

# Generate a code analysis prompt
code_analysis_prompt = asyncio.run(
    prompt_manager.create_code_analysis_prompt("def example_function(): pass")
)
print("Code Analysis Prompt:", code_analysis_prompt)
```

---

### **Benefits of This Implementation**
1. **Dynamic and Flexible**:
   - Supports dynamic schema injection and customizable templates.

2. **Error Resilience**:
   - Handles missing templates and invalid schemas gracefully with detailed logging.

3. **Extensibility**:
   - Can easily add new templates or modify existing ones without changing the code.

4. **Consistency**:
   - Ensures all prompts follow a structured format, reducing errors in AI interactions.

Let me know if you'd like further refinements or additional features!
[Timestamp: 2024/12/19 01:45:51]

**user**: Generate the complete schema_validator and docstring_validator modules
[Timestamp: 2024/12/19 01:46:01]

**ai**: 429
[Timestamp: 2024/12/19 01:46:28]

**user**: Generate the complete schema_validator and docstring_validator modules
[Timestamp: 2024/12/19 01:47:02]

**user**: Generate the complete docstring_processor.py module
[Timestamp: 2024/12/19 01:48:23]

**ai**: context_length_exceeded
[Timestamp: 2024/12/19 01:49:19]

**user**: Generate the complete docstring_processor module
[Timestamp: 2024/12/19 01:49:33]

**ai**: Here’s the complete **`docstring_processor` module** based on the refactoring plan. This module handles parsing, validating, and processing docstrings, including generating fallback docstrings and calculating complexity.

---

### **`docstring_processor.py`**
```python
"""Processes and validates docstrings."""

import json
import time
from typing import Any, Dict, List, Optional

from docstring_parser import parse, DocstringStyle
from core.validation.schema_validator import SchemaValidator
from core.types.docstring import DocstringData
from core.types.base import ProcessingResult
from core.metrics_collector import MetricsCollector
from core.exceptions import DataValidationError
from radon.complexity import cc_visit  # For calculating McCabe complexity


class DocstringProcessor:
    """
    Processes and validates docstrings.

    This class handles parsing docstrings into structured data, validating them
    against schemas, generating fallback docstrings, and calculating complexity.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initializes the DocstringProcessor.

        :param correlation_id: Optional correlation ID for logging context.
        """
        from core.logger import LoggerSetup, CorrelationLoggerAdapter

        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), extra={"correlation_id": correlation_id}
        )
        self.schema_validator = SchemaValidator(
            logger_name=f"{__name__}.{self.__class__.__name__}",
            correlation_id=correlation_id,
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.docstring_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_lines": 0,
            "avg_length": 0,
        }
        self.correlation_id = correlation_id

    def parse(self, docstring: str) -> DocstringData:
        """
        Parses a docstring string into structured data.

        :param docstring: The raw docstring to parse.
        :return: A DocstringData object containing the parsed data.
        """
        try:
            result = self._parse_docstring_content(docstring)
            return DocstringData(
                summary=result["summary"],
                description=result["description"],
                args=result["args"],
                returns=result["returns"],
                raises=result["raises"],
                complexity=result["complexity"],
            )
        except Exception as e:
            self.logger.error(f"Error parsing docstring: {e}", exc_info=True)
            return self.generate_fallback_docstring(reason=str(e))

    def _parse_docstring_content(self, docstring: str) -> Dict[str, Any]:
        """
        Parses docstring content into a structured dictionary.

        :param docstring: The raw docstring to parse.
        :return: A dictionary containing the parsed docstring data.
        """
        docstring_str = docstring.strip()
        lines = len(docstring_str.splitlines())
        length = len(docstring_str)

        # Update statistics
        self.docstring_stats["total_processed"] += 1
        self.docstring_stats["total_lines"] += lines
        self.docstring_stats["avg_length"] = (
            self.docstring_stats["avg_length"]
            * (self.docstring_stats["total_processed"] - 1)
            + length
        ) // self.docstring_stats["total_processed"]

        try:
            parsed_docstring = parse(docstring_str, style=DocstringStyle.AUTO)
            self.docstring_stats["successful"] += 1
        except Exception:
            for style in [DocstringStyle.GOOGLE, DocstringStyle.REST]:
                try:
                    parsed_docstring = parse(docstring_str, style=style)
                    self.docstring_stats["successful"] += 1
                    break
                except Exception as e:
                    self.logger.debug(
                        f"Failed to parse with style {style}: {e}",
                        extra={"style": style},
                    )
            else:
                self.docstring_stats["failed"] += 1
                self.logger.warning(
                    "Failed to parse docstring with any style",
                    extra={"docstring": docstring_str[:50]},
                )
                return self.generate_fallback_docstring(reason="Parsing failed")

        if self.docstring_stats["total_processed"] % 10 == 0:
            self._display_docstring_stats()

        return {
            "summary": parsed_docstring.short_description or "No summary available.",
            "description": parsed_docstring.long_description
            or "No description provided.",
            "args": [
                {
                    "name": param.arg_name or "",
                    "type": param.type_name or "Any",
                    "description": param.description or "",
                    "nested": [],
                }
                for param in parsed_docstring.params
            ],
            "returns": {
                "type": parsed_docstring.returns.type_name if parsed_docstring.returns else "Any",
                "description": parsed_docstring.returns.description if parsed_docstring.returns else "",
            },
            "raises": [
                {
                    "exception": exc.type_name or "Exception",
                    "description": exc.description or "",
                }
                for exc in parsed_docstring.raises
            ],
            "complexity": self.calculate_complexity(docstring_str),
        }

    def validate_with_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validates data against a schema.

        :param data: The data to validate.
        :param schema: The schema to validate against.
        :return: A tuple (is_valid, errors).
        """
        try:
            is_valid, errors = self.schema_validator.validate_schema(data, schema)
            if not is_valid:
                self.logger.warning(f"Validation failed: {errors}")
            return is_valid, errors
        except Exception as e:
            self.logger.error(f"Error during validation: {e}", exc_info=True)
            return False, [str(e)]

    def generate_fallback_docstring(self, reason: str) -> DocstringData:
        """
        Generates a fallback docstring when parsing fails.

        :param reason: The reason for generating the fallback.
        :return: A DocstringData object with fallback content.
        """
        self.logger.warning(f"Generating fallback docstring due to: {reason}")
        return DocstringData(
            summary="Fallback docstring",
            description=f"Reason: {reason}",
            args=[],
            returns={"type": "Any", "description": "No return value description available."},
            raises=[],
            complexity=1,
        )

    def calculate_complexity(self, source_code: str) -> int:
        """
        Dynamically calculates McCabe complexity for the given source code.

        :param source_code: The source code to analyze.
        :return: The McCabe complexity score.
        """
        try:
            complexity_scores = [block.complexity for block in cc_visit(source_code)]
            return max(complexity_scores, default=1)
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {e}", exc_info=True)
            return 1

    def _display_docstring_stats(self) -> None:
        """
        Displays current docstring processing statistics.
        """
        from core.console import display_metrics

        display_metrics(
            {
                "Total Processed": self.docstring_stats["total_processed"],
                "Successfully Parsed": self.docstring_stats["successful"],
                "Failed to Parse": self.docstring_stats["failed"],
                "Average Length": f"{self.docstring_stats['avg_length']}",
                "Total Lines": self.docstring_stats["total_lines"],
                "Success Rate": f"{(self.docstring_stats['successful'] / self.docstring_stats['total_processed'] * 100):.1f}%",
            },
            title="Docstring Processing Statistics",
        )

    async def process_docstring(self, docstring: str) -> ProcessingResult:
        """
        Process a docstring and return structured results.

        :param docstring: The raw docstring to process.
        :return: A ProcessingResult object containing the parsed and validated docstring.
        """
        start_time = time.time()
        try:
            parsed_data = self.parse(docstring)
            is_valid, errors = self.validate_with_schema(
                parsed_data.to_dict(), self.schema_validator._load_schema()
            )

            if not is_valid:
                raise DataValidationError(f"Docstring validation failed: {errors}")

            processing_time = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="docstring_processing",
                success=True,
                duration=processing_time,
                metadata={
                    "lines": len(docstring.splitlines()),
                    "length": len(docstring),
                    "has_args": bool(parsed_data.args),
                    "has_returns": bool(parsed_data.returns.get("description")),
                    "has_raises": bool(parsed_data.raises),
                },
            )

            return ProcessingResult(
                content=parsed_data.to_dict(),
                usage={},  # No token usage for docstring processing
                metrics={
                    "processing_time": processing_time,
                    "validation_success": True,
                },
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing docstring: {e}", exc_info=True)
            await self.metrics_collector.track_operation(
                operation_type="docstring_processing",
                success=False,
                duration=processing_time,
                metadata={"error": str(e)},
            )
            return ProcessingResult(
                content={},
                usage={},
                metrics={
                    "processing_time": processing_time,
                    "validation_success": False,
                },
                validation_status=False,
                validation_errors=[str(e)],
                schema_errors=[],
            )
```

---

### **Key Features**
1. **Docstring Parsing**:
   - Parses raw docstrings into structured `DocstringData` objects.
   - Supports multiple styles (e.g., Google, reStructuredText).

2. **Validation**:
   - Validates parsed docstrings against the `docstring_schema` using `SchemaValidator`.

3. **Fallback Handling**:
   - Generates fallback docstrings when parsing or validation fails.

4. **Complexity Calculation**:
   - Dynamically calculates McCabe complexity using the `radon` library.

5. **Metrics Collection**:
   - Tracks metrics for docstring processing, including success rates and processing times.

6. **Statistics Display**:
   - Periodically displays statistics about processed docstrings.

---

### **Example Usage**
```python
from docstring_processor import DocstringProcessor

# Initialize the DocstringProcessor
processor = DocstringProcessor(correlation_id="example-correlation-id")

# Example docstring
docstring = """
    This is an example function.

    Args:
        param1 (str): The first parameter.
        param2 (int): The second parameter.

    Returns:
        bool: True if successful, False otherwise.
"""

# Process the docstring
result = processor.parse(docstring)
print("Parsed Docstring:", result.to_dict())

# Validate the docstring
is_valid, errors = processor.validate_with_schema(result.to_dict(), processor.schema_validator._load_schema())
print("Validation Status:", is_valid)
print("Validation Errors:", errors)
```

---

### **Benefits of This Implementation**
1. **Robust Parsing**:
   - Handles multiple docstring styles and provides fallback mechanisms for failures.

2. **Dynamic Complexity**:
   - Integrates `radon` for real-time complexity analysis.

3. **Extensibility**:
   - Can easily add support for new docstring styles or validation rules.

4. **Error Resilience**:
   - Handles parsing and validation errors gracefully with detailed logging.

Let me know if you'd like further refinements or additional features!
[Timestamp: 2024/12/19 01:50:32]