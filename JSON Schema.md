```json
{
  "name": "google_style_docstrings",
  "schema": {
    "type": "object",
    "properties": {
      "description": {
        "type": "string",
        "description": "A brief description of the function or method."
      },
      "parameters": {
        "type": "array",
        "description": "A list of parameters the function takes.",
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
              "description": "A brief description of the parameter."
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
      "returns": {
        "type": "object",
        "description": "Information about the return value of the function.",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of the return value."
          },
          "description": {
            "type": "string",
            "description": "A brief description of the return value."
          }
        },
        "required": [
          "type",
          "description"
        ],
        "additionalProperties": false
      },
      "raises": {
        "type": "array",
        "description": "A list of exceptions that the function can raise.",
        "items": {
          "type": "object",
          "properties": {
            "exception": {
              "type": "string",
              "description": "The type of the exception."
            },
            "description": {
              "type": "string",
              "description": "A brief description of the exception."
            }
          },
          "required": [
            "exception",
            "description"
          ],
          "additionalProperties": false
        }
      }
    },
    "required": [
      "description",
      "parameters",
      "returns",
      "raises"
    ],
    "additionalProperties": false
  },
  "strict": true
}
```