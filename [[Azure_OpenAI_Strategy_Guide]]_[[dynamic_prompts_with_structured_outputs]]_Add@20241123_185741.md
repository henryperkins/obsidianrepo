---
epoch: 1732409861628
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# [Azure_OpenAI_Strategy_Guide](Azure_OpenAI_Strategy_Guide.md)_[dynamic_prompts_with_structured_outputs](dynamic_prompts_with_structured_outputs.md)_Add@20241123_185741

**user**: [Azure OpenAI Strategy Guide](./Azure%20OpenAI%20Strategy%20Guide.md) [dynamic prompts with structured outputs](./dynamic%20prompts%20with%20structured%20outputs.md) Add the dynamic_prompts with structured outputs note to the azure strategy guide  
[Timestamp: 2024/11/23 18:57:41]

**ai**: To integrate the "dynamic prompts with structured outputs" note into the Azure OpenAI Strategy Guide, we will add a new section that covers the use of dynamic prompts in conjunction with structured outputs. This section will provide an overview, example implementation, and references to relevant documentation. Here is how the updated strategy guide would look:

---

## 17. Dynamic Prompts with Structured Outputs

**Overview:**  
Leverage dynamic prompts to tailor interactions with the Azure OpenAI API, ensuring that outputs are structured according to predefined schemas. This approach enhances the flexibility and reliability of AI-generated content by dynamically adjusting prompts based on input parameters and expected output formats.

**Example:**

```python
import os
import asyncio
from openai import AsyncAzureOpenAI

# Load environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
deployment_id = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')

# Initialize the Async client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

def create_dynamic_prompt(function_signature: str, additional_context: str = "") -> str:
    """Create a dynamic prompt based on the function signature and additional context."""
    prompt = f"Generate a Python docstring for the following function signature:\n\n{function_signature}\n\n"
    if additional_context:
        prompt += f"Additional context: {additional_context}\n\n"
    prompt += "The docstring should include:\n- `summary`: A concise summary.\n- `args`: A list of arguments with `name`, `type`, and `description`.\n- `returns`: A description of the return value(s).\n\n"
    return prompt

async def generate_structured_docstring(function_signature: str, additional_context: str = ""):
    prompt = create_dynamic_prompt(function_signature, additional_context)

    # Define the expected structure of the output
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "args": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "returns": {"type": "string"}
        },
        "required": ["summary", "args", "returns"]
    }

    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
            function_call={"name": "generate_docstring", "arguments": schema}
        )

        # Process the structured response
        structured_output = response.choices[0].message.content.strip()
        return structured_output

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    function_signature = "def example_function(param1: int, param2: str) -> bool:"
    additional_context = "This function checks if param1 is greater than param2."
    docstring = await generate_structured_docstring(function_signature, additional_context)
    print(docstring)

asyncio.run(main())
```

**Key Points:**
- **Dynamic Prompt Creation**: Tailor prompts to the specific context and requirements of the task.
- **Structured Output Schema**: Define the expected JSON format to ensure consistent and reliable outputs.
- **Function Calling**: Use the `function_call` parameter to instruct the model to adhere to the schema.

**References:**
- [Azure OpenAI Structured Outputs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs)
- [Dynamic Prompts Example on GitHub](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb)

---

This new section provides a comprehensive guide on how to implement dynamic prompts with structured outputs, enhancing the capabilities of Azure OpenAI applications. If you have any further questions or need additional customization, feel free to ask!  
[Timestamp: 2024/11/23 18:57:55]

**user**: Add the rest of the content to it  
[Timestamp: 2024/11/23 18:58:05]

**ai**: Certainly! Let's expand the section on "Dynamic Prompts with Structured Outputs" to include more detailed content, such as additional context, implementation strategies, and potential use cases. Here's how the complete section would look:

---

## 17. Dynamic Prompts with Structured Outputs

**Overview:**  
Dynamic prompts allow you to tailor interactions with the Azure OpenAI API by adjusting the input based on specific conditions or parameters. When combined with structured outputs, this approach ensures that the responses are not only contextually relevant but also adhere to a predefined format, such as JSON. This is particularly useful for applications that require consistent data structures, such as generating documentation, data extraction, or API responses.

**Implementation Strategies:**

1. **Dynamic Prompt Creation:**
   - Construct prompts based on input parameters, function signatures, or user preferences.
   - Include additional context or examples to guide the model towards the desired output.

2. **Define Structured Output Schema:**
   - Specify the expected JSON structure for the output using a schema.
   - Ensure that the schema includes all necessary fields and their types.

3. **API Calls with Function Calling:**
   - Use the `function_call` parameter in the API request to enforce the structured output.
   - Handle the API response by parsing and validating it against the schema.

4. **Error Handling and Validation:**
   - Implement error handling to manage exceptions during API calls.
   - Validate the structured output to ensure it meets the expected schema.

**Example Implementation:**

```python
import os
import asyncio
from openai import AsyncAzureOpenAI

# Load environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
deployment_id = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')

# Initialize the Async client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

def create_dynamic_prompt(function_signature: str, additional_context: str = "") -> str:
    """Create a dynamic prompt based on the function signature and additional context."""
    prompt = f"Generate a Python docstring for the following function signature:\n\n{function_signature}\n\n"
    if additional_context:
        prompt += f"Additional context: {additional_context}\n\n"
    prompt += "The docstring should include:\n- `summary`: A concise summary.\n- `args`: A list of arguments with `name`, `type`, and `description`.\n- `returns`: A description of the return value(s).\n\n"
    return prompt

async def generate_structured_docstring(function_signature: str, additional_context: str = ""):
    prompt = create_dynamic_prompt(function_signature, additional_context)

    # Define the expected structure of the output
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "args": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "returns": {"type": "string"}
        },
        "required": ["summary", "args", "returns"]
    }

    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
            function_call={"name": "generate_docstring", "arguments": schema}
        )

        # Process the structured response
        structured_output = response.choices[0].message.content.strip()
        return structured_output

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    function_signature = "def example_function(param1: int, param2: str) -> bool:"
    additional_context = "This function checks if param1 is greater than param2."
    docstring = await generate_structured_docstring(function_signature, additional_context)
    print(docstring)

asyncio.run(main())
```

**Use Cases:**

- **Documentation Generation:** Automatically generate structured documentation for codebases by dynamically creating prompts based on function signatures and comments.
- **Data Extraction:** Extract structured data from unstructured text by defining schemas that the output must adhere to.
- **API Response Formatting:** Ensure API responses are consistently formatted by using structured outputs in conjunction with dynamic prompts.

**Key Points:**

- **Dynamic Prompt Creation:** Tailor prompts to the specific context and requirements of the task.
- **Structured Output Schema:** Define the expected JSON format to ensure consistent and reliable outputs.
- **Function Calling:** Use the `function_call` parameter to instruct the model to adhere to the schema.
- **Error Handling:** Implement robust error handling to manage exceptions and validate outputs.

**References:**

- [Azure OpenAI Structured Outputs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs)
- [Dynamic Prompts Example on GitHub](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb)

---

This section provides a comprehensive guide on implementing dynamic prompts with structured outputs, enhancing the capabilities of Azure OpenAI applications by ensuring that outputs are both contextually relevant and structurally consistent. If you have further questions or need additional customization, feel free to ask!  
[Timestamp: 2024/11/23 18:58:20]

**user**: Doesn't the guide explain how to use it with function calling too>?  
[Timestamp: 2024/11/23 18:59:02]

**ai**: Certainly! Let's ensure that the section on "Dynamic Prompts with Structured Outputs" explicitly covers the integration with function calling. This involves using the function calling feature to enforce structured outputs and dynamically generating prompts. Here’s how the updated section would look:

---

## 17. Dynamic Prompts with Structured Outputs and Function Calling

**Overview:**  
Combine dynamic prompts with function calling to generate structured outputs using the Azure OpenAI API. This approach allows you to dynamically adjust prompts based on input conditions and ensure that the API returns data in a specified format, such as JSON. This is particularly useful for applications requiring consistent data structures, like generating documentation, extracting information, or formatting API responses.

**Implementation Strategies:**

1. **Dynamic Prompt Creation:**
   - Construct prompts based on input parameters, function signatures, or user preferences.
   - Include additional context or examples to guide the model towards the desired output.

2. **Define Structured Output Schema:**
   - Specify the expected JSON structure for the output using a schema.
   - Ensure that the schema includes all necessary fields and their types.

3. **API Calls with Function Calling:**
   - Use the `function_call` parameter in the API request to enforce the structured output.
   - Define functions that the model can call to generate outputs according to the schema.

4. **Process and Validate Responses:**
   - Parse and validate the API response against the schema.
   - Implement error handling to manage exceptions during API calls.

**Example Implementation:**

```python
import os
import asyncio
from openai import AsyncAzureOpenAI

# Load environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
deployment_id = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')

# Initialize the Async client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

def create_dynamic_prompt(function_signature: str, additional_context: str = "") -> str:
    """Create a dynamic prompt based on the function signature and additional context."""
    prompt = f"Generate a Python docstring for the following function signature:\n\n{function_signature}\n\n"
    if additional_context:
        prompt += f"Additional context: {additional_context}\n\n"
    prompt += "The docstring should include:\n- `summary`: A concise summary.\n- `args`: A list of arguments with `name`, `type`, and `description`.\n- `returns`: A description of the return value(s).\n\n"
    return prompt

async def generate_structured_docstring(function_signature: str, additional_context: str = ""):
    prompt = create_dynamic_prompt(function_signature, additional_context)

    # Define the expected structure of the output
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "args": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "returns": {"type": "string"}
        },
        "required": ["summary", "args", "returns"]
    }

    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
            function_call={"name": "generate_docstring", "arguments": schema}
        )

        # Process the structured response
        structured_output = response.choices[0].message.content.strip()
        return structured_output

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    function_signature = "def example_function(param1: int, param2: str) -> bool:"
    additional_context = "This function checks if param1 is greater than param2."
    docstring = await generate_structured_docstring(function_signature, additional_context)
    print(docstring)

asyncio.run(main())
```

**Use Cases:**

- **Documentation Generation:** Automatically generate structured documentation for codebases by dynamically creating prompts based on function signatures and comments.
- **Data Extraction:** Extract structured data from unstructured text by defining schemas that the output must adhere to.
- **API Response Formatting:** Ensure API responses are consistently formatted by using structured outputs in conjunction with dynamic prompts.

**Key Points:**

- **Dynamic Prompt Creation:** Tailor prompts to the specific context and requirements of the task.
- **Structured Output Schema:** Define the expected JSON format to ensure consistent and reliable outputs.
- **Function Calling:** Use the `function_call` parameter to instruct the model to adhere to the schema.
- **Error Handling:** Implement robust error handling to manage exceptions and validate outputs.

**References:**

- [Azure OpenAI Structured Outputs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs)
- [Dynamic Prompts Example on GitHub](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb)

---

This section now explicitly covers how to use dynamic prompts in conjunction with function calling to achieve structured outputs, providing a comprehensive guide for leveraging these features in Azure OpenAI applications. If you have further questions or need additional customization, feel free to ask!  
[Timestamp: 2024/11/23 18:59:18]
