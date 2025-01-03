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