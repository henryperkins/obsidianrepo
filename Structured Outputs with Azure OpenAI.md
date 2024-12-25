To use **structured outputs** with the Azure OpenAI Service's chat completions, you can leverage the `response_format` parameter to ensure the model generates responses in a specific structured format, such as JSON objects or JSON schemas. Here's how you can use structured outputs effectively:

---

### **1. Set the `response_format` Parameter**
The `response_format` parameter allows you to specify the desired output structure. You can choose between:
- **`json_object`**: Ensures the model generates valid JSON output.
- **`json_schema`**: Ensures the model's output adheres to a predefined JSON schema.

---

### **2. Define the `response_format` in the Request**
When making a request to the `/deployments/{deployment-id}/chat/completions` endpoint, include the `response_format` parameter in the request body. Here's how you can structure it:

#### **For `json_object`:**
```json
"response_format": {
  "type": "json_object"
}
```

#### **For `json_schema`:**
```json
"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "example_schema",
    "description": "A schema for structured output",
    "schema": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        },
        "age": {
          "type": "integer"
        },
        "email": {
          "type": "string",
          "format": "email"
        }
      },
      "required": ["name", "age"]
    },
    "strict": true
  }
}
```

- **`type`**: Always set to `json_schema`.
- **`name`**: A unique name for the schema.
- **`description`**: A brief description of the schema's purpose.
- **`schema`**: The JSON schema definition, following the [JSON Schema Standard](https://json-schema.org/understanding-json-schema/).
- **`strict`**: If set to `true`, the model strictly adheres to the schema.

---

### **3. Instruct the Model to Use the Format**
In addition to setting the `response_format`, you should guide the model to produce structured outputs through the `messages` parameter. For example:

#### Example `messages` with a System Instruction:
```json
"messages": [
  {
    "role": "system",
    "content": "You are an assistant that provides responses in JSON format. Ensure all responses are valid JSON."
  },
  {
    "role": "user",
    "content": "Please provide a JSON object with my name, age, and email."
  }
]
```

---

### **4. Validate the Output**
- If you use `json_object`, the model ensures the output is valid JSON.
- If you use `json_schema`, the model guarantees the output conforms to the schema. This is particularly useful for applications requiring strict adherence to a predefined structure.

---

### **5. Example Request**
Here’s a complete example of a request using `json_schema`:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that generates structured JSON responses."
    },
    {
      "role": "user",
      "content": "Provide a JSON object with my name, age, and email."
    }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "user_info",
      "description": "A schema for user information",
      "schema": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "age": {
            "type": "integer"
          },
          "email": {
            "type": "string",
            "format": "email"
          }
        },
        "required": ["name", "age", "email"]
      },
      "strict": true
    }
  }
}
```

---

### **6. Example Response**
If the request is successful, the model will return a response like this:

```json
{
  "name": "John Doe",
  "age": 30,
  "email": "john.doe@example.com"
}
```

---

### **7. Benefits of Structured Outputs**
- **Validation**: Ensures responses are always in the expected format.
- **Reliability**: Reduces the chances of malformed or incomplete responses.
- **Integration**: Simplifies integration with downstream systems that consume structured data.

---

### **8. Use Cases**
- **APIs**: Returning structured data for API responses.
- **Data Pipelines**: Ensuring compatibility with data processing systems.
- **Applications**: Generating structured inputs for forms or workflows.

---


Tool or function calling in Azure OpenAI chat completions allows the model to interact with external tools or functions to enhance its capabilities, such as retrieving data, performing calculations, or executing specific tasks. When combined with **structured outputs**, this feature becomes even more powerful, as it ensures that the model generates well-structured inputs and outputs for tools or functions.

Here’s a detailed explanation of how tool or function calling works with structured outputs:

---

### **1. What is Tool or Function Calling?**
Tool calling enables the model to generate inputs for external tools or functions during a conversation. The model can:
- Decide when to call a tool.
- Generate structured inputs for the tool.
- Incorporate the tool's outputs into its response.

This is particularly useful for tasks like database queries, API calls, or performing calculations.

---

### **2. How to Enable Tool or Function Calling**
To use tools or functions, you need to define them in the `tools` parameter of the request. Each tool must include:
- **Name**: The name of the tool or function.
- **Description**: A description of what the tool does.
- **Parameters**: A JSON Schema defining the expected input structure.

For structured outputs, the model will generate inputs that conform to the defined schema.

---

### **3. Define a Tool with Structured Inputs**
When defining a tool, you use a JSON Schema to specify the expected input format. For example:

#### Example Tool Definition:
```json
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Retrieve the current weather for a given city.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city to get the weather for."
          },
          "unit": {
            "type": "string",
            "description": "The unit for temperature (e.g., Celsius or Fahrenheit).",
            "enum": ["Celsius", "Fahrenheit"]
          }
        },
        "required": ["city"]
      }
    }
  }
]
```

- **`type`**: Specifies the tool type (currently, only `function` is supported).
- **`name`**: The name of the function/tool.
- **`parameters`**: A JSON Schema defining the expected structure of the input.

---

### **4. Guide the Model to Use the Tool**
In the `messages` parameter, include instructions for the model to use the tool when appropriate. For example:

#### Example `messages` with Tool Instructions:
```json
"messages": [
  {
    "role": "system",
    "content": "You are a helpful assistant. Use the `get_weather` function to retrieve weather information when asked about the weather."
  },
  {
    "role": "user",
    "content": "What is the weather in Paris?"
  }
]
```

---

### **5. How the Model Calls the Tool**
When the model determines that it needs to call a tool, it generates a structured input for the tool based on the defined JSON Schema. For example:

#### Example Tool Call Generated by the Model:
```json
{
  "tool_calls": [
    {
      "id": "tool_call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": {
          "city": "Paris",
          "unit": "Celsius"
        }
      }
    }
  ]
}
```

- **`id`**: A unique identifier for the tool call.
- **`type`**: Indicates this is a function call.
- **`name`**: The name of the tool being called.
- **`arguments`**: The structured input generated by the model, conforming to the JSON Schema.

---

### **6. Submit Tool Outputs**
Once the tool processes the request and generates an output, you can submit the output back to the model using the `/threads/{thread_id}/runs/{run_id}/submit_tool_outputs` endpoint.

#### Example Tool Output Submission:
```json
{
  "tool_outputs": [
    {
      "tool_call_id": "tool_call_123",
      "output": {
        "temperature": "22°C",
        "condition": "Sunny"
      }
    }
  ]
}
```

- **`tool_call_id`**: The ID of the tool call.
- **`output`**: The structured output from the tool.

---

### **7. Model Incorporates Tool Outputs**
After receiving the tool's output, the model integrates it into the conversation. For example:

#### Example Response from the Model:
```json
{
  "role": "assistant",
  "content": "The current weather in Paris is 22°C and sunny."
}
```

---

### **8. Using Structured Outputs with Tools**
To ensure structured outputs are used throughout the process:
- **Define JSON Schema for Tools**: Use the `parameters` field to enforce structured inputs for tools.
- **Set `response_format`**: Use `json_object` or `json_schema` to ensure the model's responses are also structured.

#### Example Request Combining Tools and Structured Outputs:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Use the `get_weather` function to retrieve weather information when asked."
    },
    {
      "role": "user",
      "content": "What is the weather in New York?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Retrieve the current weather for a given city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "The name of the city to get the weather for."
            },
            "unit": {
              "type": "string",
              "description": "The unit for temperature (e.g., Celsius or Fahrenheit).",
              "enum": ["Celsius", "Fahrenheit"]
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "response_format": {
    "type": "json_object"
  }
}
```

---

### **9. Benefits of Combining Tool Calling and Structured Outputs**
- **Accuracy**: Ensures inputs and outputs are well-structured and valid.
- **Reliability**: Reduces errors when interacting with external tools.
- **Interoperability**: Simplifies integration with APIs or other systems requiring structured data.
- **Flexibility**: Allows the model to dynamically decide when to call tools and how to process their outputs.

---

### **10. Example Use Cases**
- **Weather Retrieval**: Use a weather API to provide real-time weather updates.
- **Database Queries**: Fetch structured data from a database.
- **Calculations**: Perform complex calculations and return structured results.
- **Task Automation**: Automate workflows by calling external APIs with structured inputs.

---

- [docstring_schema](docstring_schema.md)
- [function_tools_schema](function_tools_schema.md)

To use **tool or function calling** with structured outputs like the `generate_docstring` function and the `google_style_docstring` schema, follow these steps:

---

### **1. Define the Tool with a Schema**
The `generate_docstring` function is defined in the `function_tools_schema.json`. It specifies the structure of the input parameters the model must generate when calling the function. This structure is enforced by the JSON Schema.

#### Example Tool Definition:
```json
{
    "type": "function",
    "function": {
        "name": "generate_docstring",
        "description": "Generate a Python docstring with structured information.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise summary of what the code does."
                },
                "description": {
                    "type": "string",
                    "description": "A detailed description of the functionality."
                },
                "args": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the argument."},
                            "type": {"type": "string", "description": "The type of the argument."},
                            "description": {"type": "string", "description": "A description of the argument."}
                        },
                        "required": ["name", "type", "description"]
                    },
                    "description": "A list of arguments, each with a name, type, and description."
                },
                "returns": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "The return type."},
                        "description": {"type": "string", "description": "A description of the return value."}
                    },
                    "required": ["type", "description"],
                    "description": "An object describing the return value, including its type and a description."
                },
                "raises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exception": {"type": "string", "description": "The type of exception raised."},
                            "description": {"type": "string", "description": "A description of when the exception is raised."}
                        },
                        "required": ["exception", "description"]
                    },
                    "description": "A list of exceptions that may be raised, each with a type and description."
                },
                "complexity": {
                    "type": "integer",
                    "description": "McCabe complexity score"
                }
            },
            "required": ["summary", "description", "args", "returns", "raises", "complexity"]
        }
    }
}
```

---

### **2. Guide the Model to Use the Tool**
In the `messages` parameter, include instructions for the model to use the `generate_docstring` function when required. For example:

#### Example `messages` with Tool Instructions:
```json
"messages": [
    {
        "role": "system",
        "content": "You are a helpful assistant that generates Python docstrings using the `generate_docstring` function."
    },
    {
        "role": "user",
        "content": "Please generate a docstring for a function that calculates the area of a rectangle."
    }
]
```

---

### **3. Model Generates Structured Input for the Tool**
Based on the user query and the tool schema, the model generates a structured input for the tool. For example:

#### Example Tool Call Generated by the Model:
```json
{
    "tool_calls": [
        {
            "id": "tool_call_123",
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "arguments": {
                    "summary": "Calculate the area of a rectangle.",
                    "description": "This function calculates the area of a rectangle given its width and height.",
                    "args": [
                        {
                            "name": "width",
                            "type": "float",
                            "description": "The width of the rectangle."
                        },
                        {
                            "name": "height",
                            "type": "float",
                            "description": "The height of the rectangle."
                        }
                    ],
                    "returns": {
                        "type": "float",
                        "description": "The area of the rectangle."
                    },
                    "raises": [
                        {
                            "exception": "ValueError",
                            "description": "Raised if width or height is negative."
                        }
                    ],
                    "complexity": 2
                }
            }
        }
    ]
}
```

---

### **4. Submit Tool Outputs**
Once the tool processes the request and generates the docstring, you can submit the output back to the model using the `/threads/{thread_id}/runs/{run_id}/submit_tool_outputs` endpoint.

#### Example Tool Output Submission:
```json
{
    "tool_outputs": [
        {
            "tool_call_id": "tool_call_123",
            "output": {
                "summary": "Calculate the area of a rectangle.",
                "description": "This function calculates the area of a rectangle given its width and height.",
                "args": [
                    {
                        "name": "width",
                        "type": "float",
                        "description": "The width of the rectangle."
                    },
                    {
                        "name": "height",
                        "type": "float",
                        "description": "The height of the rectangle."
                    }
                ],
                "returns": {
                    "type": "float",
                    "description": "The area of the rectangle."
                },
                "raises": [
                    {
                        "exception": "ValueError",
                        "description": "Raised if width or height is negative."
                    }
                ],
                "complexity": 2
            }
        }
    ]
}
```

---

### **5. Model Incorporates Tool Outputs**
After receiving the tool's output, the model integrates it into the conversation. For example:

#### Example Response from the Model:
```json
{
    "role": "assistant",
    "content": "Here is the generated docstring:\n\n```python\n\"\"\"\nCalculate the area of a rectangle.\n\nThis function calculates the area of a rectangle given its width and height.\n\nArgs:\n    width (float): The width of the rectangle.\n    height (float): The height of the rectangle.\n\nReturns:\n    float: The area of the rectangle.\n\nRaises:\n    ValueError: Raised if width or height is negative.\n\nComplexity:\n    McCabe complexity score: 2\n\"\"\"\n```"
}
```

---

### **6. Use Structured Outputs for Validation**
The `google_style_docstring` schema from `docstring_schema` ensures the generated docstring adheres to a specific format. You can validate the output against this schema to ensure compliance.

#### Example Validation Schema:
```json
{
    "name": "google_style_docstring",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "description": {"type": "string"},
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
            "returns": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["type", "description"]
            },
            "raises": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "exception": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["exception", "description"]
                }
            },
            "complexity": {"type": "integer"}
        },
        "required": ["summary", "description", "returns"]
    }
}
```

---

### **7. Benefits of Using Tools with Structured Outputs**
- **Accuracy**: Ensures inputs and outputs conform to a predefined structure.
- **Reliability**: Reduces errors when generating or processing docstrings.
- **Interoperability**: Simplifies integration with tools that require structured data.
- **Validation**: Guarantees compliance with the `google_style_docstring` schema.

---

### **8. Example Use Cases**
- **Code Documentation**: Automatically generate docstrings for functions, classes, or modules.
- **API Documentation**: Create structured documentation for APIs.
- **Error Handling**: Include structured information about exceptions raised by the code.

Let me know if you'd like further clarification or examples!