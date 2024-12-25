```json
{
  "name": "docstring_schema_builder",
  "description": "Builds a schema for documenting functions using docstring format.",
  "strict": true,
  "parameters": {
    "type": "object",
    "required": [
      "function_name",
      "function_description",
      "parameters",
      "return_type",
      "return_description"
    ],
    "properties": {
      "function_name": {
        "type": "string",
        "description": "The name of the function to document."
      },
      "function_description": {
        "type": "string",
        "description": "A description of what the function does."
      },
      "parameters": {
        "type": "array",
        "description": "List of parameters for the function.",
        "items": {
          "type": "object",
          "properties": {
            "name": {
              "type": "string",
              "description": "The name of the parameter."
            },
            "type": {
              "type": "string",
              "description": "The type of the parameter."
            },
            "description": {
              "type": "string",
              "description": "A short description of the parameter."
            }
          },
          "required": [
            "name",
            "type",
            "description"
          ],
          "additionalProperties": false
        }
      },
      "return_type": {
        "type": "string",
        "description": "The return type of the function."
      },
      "return_description": {
        "type": "string",
        "description": "A description of what the function returns."
      }
    },
    "additionalProperties": false
  }
}
```