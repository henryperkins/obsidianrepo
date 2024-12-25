https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#authentication
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs
https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb

To address the issue of invalid JSON format in the responses from the Azure OpenAI API, let's first ensure that the authentication and API call setup are correctly configured according to the Azure documentation. The link you provided outlines the authentication process for Azure OpenAI, which is crucial for making successful API calls.

### Authentication Steps for Azure OpenAI

1. **Set Up Environment Variables**: Ensure your `.env` file is correctly configured with the necessary credentials and endpoint information. This includes the API key, endpoint, deployment ID, and model name.

   ```plaintext
   AZURE_OPENAI_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_ID=your_deployment_id
   AZURE_OPENAI_MODEL_NAME=gpt-4
   ```

2. **Initialize the Client**: Use the `AsyncAzureOpenAI` client for asynchronous operations. Ensure the client is initialized with the correct endpoint and API key.

3. **Make API Requests**: Ensure that the requests are correctly formatted and that the prompt explicitly instructs the model to return a JSON object.

Here’s an example of how you might set up the client and make an API call:

```python
import os
from openai import AsyncAzureOpenAI
import asyncio

# Load environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
deployment_id = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')
model_name = os.getenv('AZURE_OPENAI_MODEL_NAME', 'gpt-4')

# Initialize the Async client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

async def generate_docstring(prompt):
    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in Google Style."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    prompt = "Generate a Python docstring in JSON format for the following function signature:\n\n def example_function(param1: int, param2: str) -> bool:\n\n The docstring should include:\n - `summary`: A concise summary.\n - `args`: A list of arguments with `name`, `type`, and `description`.\n - `returns`: A description of the return value(s).\n\n Ensure the output is a valid JSON object with these keys."
    docstring = await generate_docstring(prompt)
    print(docstring)

asyncio.run(main())
```

### Debugging Tips

- **Log Raw Responses**: Add logging to capture the raw response from the API. This can help identify if the response format is incorrect or if there are missing keys.

- **Validate JSON**: Use a JSON validator to ensure the response is valid JSON. This can help pinpoint syntax errors or structural issues in the JSON.

- **Adjust Prompts**: If the model is not returning JSON, adjust the prompt to be more explicit about the expected format. Include examples if necessary.

- **Test with Different Models**: Some models might be better suited for structured output. If possible, test with different models to see if the issue persists.

By following these steps and ensuring proper authentication and request formatting, you should be able to resolve issues related to invalid JSON responses from the Azure OpenAI API. If problems persist, consider reaching out to Azure support for further assistance.

---
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs

To handle structured outputs from the Azure OpenAI API, especially when dealing with function calling, you can leverage the API's ability to return structured data. This is particularly useful when you want the API to return data in a specific format, such as JSON, which is often required for tasks like generating structured docstrings.

Here's how you can set up your script to make use of structured outputs:

### Steps to Implement Structured Outputs

1. **Define the Function Schema**: Specify the expected structure of the output. This can be done by defining a schema that the API should follow when generating the output.

2. **Update the API Call**: Use the schema in your API request to instruct the model to return data in the specified format.

3. **Parse the Response**: Ensure that the response is parsed according to the defined schema, and handle any errors or discrepancies.

### Example Implementation

Below is an example of how you might implement structured outputs in your script using the Azure OpenAI API:

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

async def generate_structured_docstring(function_signature):
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

    # Create the prompt
    prompt = f"Generate a Python docstring in JSON format for the following function signature:\n\n{function_signature}\n\nThe docstring should include:\n- `summary`: A concise summary.\n- `args`: A list of arguments with `name`, `type`, and `description`.\n- `returns`: A description of the return value(s).\n\nEnsure the output matches the following schema: {schema}"

    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        # Parse the structured response
        structured_output = response.choices[0].message.content.strip()
        return structured_output

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    function_signature = "def example_function(param1: int, param2: str) -> bool:"
    docstring = await generate_structured_docstring(function_signature)
    print(docstring)

asyncio.run(main())
```

### Key Points

- **Schema Definition**: The schema defines the expected structure of the JSON output. It includes properties like `summary`, `args`, and `returns`, each with their respective types and requirements.

- **Prompt Construction**: The prompt explicitly instructs the model to generate output that matches the defined schema. This helps ensure that the response is structured as expected.

- **Error Handling**: The script includes basic error handling to catch and report issues during the API call.

By using structured outputs, you can improve the reliability of the data returned by the Azure OpenAI API, making it easier to integrate into your applications. If you encounter any issues or need further customization, feel free to ask for additional assistance.

---
https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb

The GitHub repository you referenced contains a Jupyter Notebook that demonstrates how to use dynamic prompts with the Azure OpenAI API to generate completions. This approach can be particularly useful when you want to customize the prompt based on specific inputs or conditions dynamically.

Here's a general overview of how you might implement dynamic prompting in your script, inspired by the concepts from the notebook:

### Steps to Implement Dynamic Prompting

1. **Define Dynamic Prompts**: Create a function that constructs prompts based on input parameters or conditions. This allows you to tailor the prompt to the specific context or task.

2. **Use the Azure OpenAI API**: Make API calls using these dynamically generated prompts to get the desired completions.

3. **Process the Response**: Handle the API response, ensuring that it meets your requirements and is properly formatted.

### Example Implementation

Below is an example of how you might implement dynamic prompting in a script using the Azure OpenAI API:

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

async def generate_docstring(function_signature: str, additional_context: str = ""):
    prompt = create_dynamic_prompt(function_signature, additional_context)

    try:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        # Process the response
        docstring = response.choices[0].message.content.strip()
        return docstring

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Example usage
async def main():
    function_signature = "def example_function(param1: int, param2: str) -> bool:"
    additional_context = "This function checks if param1 is greater than param2."
    docstring = await generate_docstring(function_signature, additional_context)
    print(docstring)

asyncio.run(main())
```

### Key Points

- **Dynamic Prompt Creation**: The `create_dynamic_prompt` function constructs a prompt based on the function signature and any additional context you provide. This allows for more tailored and relevant prompts.

- **API Call**: The script uses the `AsyncAzureOpenAI` client to make asynchronous API calls, ensuring efficient handling of requests.

- **Error Handling**: Basic error handling is included to catch and report issues during the API call.

By using dynamic prompts, you can customize the input to the Azure OpenAI API based on the specific needs of your application, potentially improving the relevance and accuracy of the generated outputs. If you have any further questions or need additional customization, feel free to ask!

---

[openai/Basic_Samples/Functions/working_with_functions.ipynb at main · Azure-Samples/openai](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Functions/working_with_functions.ipynb)
https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python#function-calling-with-structured-outputs

To combine dynamic prompts with function calling and structured outputs using the Azure OpenAI API, you'll need to integrate these concepts into a cohesive workflow. This involves dynamically generating prompts based on input parameters, specifying a structured output format, and using function calling to ensure the API returns data in the desired structure.

Here's how you can implement this:

### Steps to Implement Combined Approach

1. **Dynamic Prompt Creation**: Create prompts based on function signatures and additional context.

2. **Define Structured Output Schema**: Specify the expected JSON structure for the output.

3. **Make API Calls with Function Calling**: Use the Azure OpenAI API to generate structured outputs based on the dynamic prompts and schema.

4. **Process and Validate Responses**: Ensure the API response matches the expected structure and handle any discrepancies.

### Example Implementation

Here’s an example script that combines these elements:

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

### Key Points

- **Dynamic Prompt Creation**: The `create_dynamic_prompt` function generates prompts tailored to the function signature and context, ensuring relevant input to the API.

- **Structured Output Schema**: The schema defines the expected JSON format for the docstring, specifying required fields and their types.

- **Function Calling**: The API request includes a `function_call` parameter with the schema, instructing the model to return data in the specified structure.

- **Asynchronous API Calls**: The script uses `AsyncAzureOpenAI` to handle API requests asynchronously, improving efficiency.

This approach ensures that the Azure OpenAI API returns structured outputs that are dynamically tailored to the input function signatures, providing reliable and relevant docstring generation. If you have further questions or need additional customization, feel free to ask!

---

**More info on Function Usage and Dynamic Prompts**

### **Function (Tool) Usage**:

**1. Defining Functions as Tools:**

Functions in Azure OpenAI are specified as tools which the model can choose to use when responding to a request. This is particularly useful for:

- **Structured Outputs**: When you want the output in a specific format, like JSON, for easy parsing.
- **External API Calls**: If your application needs to interact with external services, you can define functions that the AI can call to fetch or update data.

**Example Function Definition:**

```python
function_tool = {
    "type": "function",
    "function": {
        "name": "generate_docstring",
        "description": "Generate a Python docstring with structured information",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "description": "Name of the function"},
                "summary": {"type": "string", "description": "A brief summary of what the function does"},
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
                    },
                    "description": "List of arguments with their name, type, and description"
                },
                "returns": {"type": "string", "description": "Description of what the function returns"}
            },
            "required": ["function_name", "summary", "args", "returns"]
        }
    }
}
```

**2. Using Functions in API Calls:**

When you make an API call, you can include these tools:

```python
response = await self.client.chat.completions.create(
    model=self.args.api_deployment_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant specialized in Python docstring generation."},
        {"role": "user", "content": prompt}
    ],
    tools=[function_tool],
    tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
)
```

- **tool_choice**: Can be set to `"auto"`, `"none"`, or specify a function. Here, we're explicitly telling the model to use the "generate_docstring" function.

**3. Processing Function Call Responses:**

After the API call, you'll need to handle the function call:

```python
if response.choices[0].finish_reason == "function_call":
    # Parse the JSON string returned by the model
    function_call = response.choices[0].message.function_call
    if function_call.name == "generate_docstring":
        docstring_json = json.loads(function_call.arguments)
        # Format the docstring from the JSON
        func_info.docstring = self.format_docstring_from_json(docstring_json)
    else:
        # Handle unexpected function call
        pass
```

### **Dynamic Prompts:**

Dynamic prompts allow for more flexible and context-aware responses from the model. Here are some strategies:

**1. Prompt Construction:**

- **Contextual Information**: Include relevant context in the prompt. For docstring generation, this could be:
  - The function's name, parameters, and return type.
  - Any existing docstring or partial information.

- **User Interaction**: If your application has user interaction, you might dynamically add user-specific instructions or preferences into the prompt.

```python
def construct_prompt(self, func_info: FunctionInfo) -> str:
    prompt = f"""
    Generate a docstring for the following function:

    {generate_function_signature(func_info)}

    **Context**: {func_info.docstring or 'No existing docstring.'}

    Please follow the Google Python Style Guide for docstrings.
    """
    return prompt
```

**2. Variable Content:**

- **Handle Variable Content**: If you're dealing with functions that dynamically change based on runtime conditions or user input, your prompt might need to reflect this:

```python
def dynamic_prompt(self, func_info: FunctionInfo, additional_context: str = ""):
    base_prompt = self.construct_prompt(func_info)
    if additional_context:
        base_prompt = base_prompt + f"\nAdditional Context: {additional_context}"
    return base_prompt
```

**3. Prompt Optimization:**

- **Token Management**: Ensure your dynamic prompt doesn't exceed the model's maximum token limit by truncating or summarizing longer contexts if necessary.

- **Experimentation**: Test different prompt structures to see what yields the best results. Sometimes, adding examples or demonstrations within the prompt can guide the model better.

By combining function calls with dynamic prompts, you're essentially creating a more interactive and precise interaction with the model, where the AI can both understand the context and structure its output accordingly. This approach is particularly powerful for tasks like docstring generation where precision and structure are crucial.