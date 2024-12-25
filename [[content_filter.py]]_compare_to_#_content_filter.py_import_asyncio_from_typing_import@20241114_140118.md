---
epoch: 1731614478291
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
  - _content_filter
---

# [content_filter.py](content_filter.py.md)*compare_to*#_content_filter.py_import_asyncio_from_typing_import@20241114_140118

**user**: [content_filter.py](content_filter.py.md) compare to # content_filter.py  
import asyncio  
from typing import Dict, List, Optional, Union  
from logger import log_info, log_error, log_debug  
from openai import AzureOpenAI

class ContentFilter:  
    """Content filtering implementation using Azure OpenAI content safety."""

    def __init__(self, client: AzureOpenAI):
        """
        Initialize the ContentFilter with a client for content moderation.

        Args:
            client: The Azure OpenAI client used to perform content safety checks.
        """
        self.client = client
        self.blocked_terms = set()
        self.content_categories = {
            "hate": 0.7,
            "sexual": 0.8,
            "violence": 0.8,
            "self-harm": 0.9
        }

    def add_blocked_terms(self, terms: List[str]) -> None:
        """
        Add terms to the blocked terms list.

        Args:
            terms: List of terms to block.
        """
        self.blocked_terms.update(terms)
        log_info(f"Blocked terms updated: {self.blocked_terms}")

    async def check_content(self, text: str) -> Dict[str, Union[bool, Optional[str]]]:
        """
        Check content for safety against blocked terms and content categories.

        Args:
            text: Content to check.

        Returns:
            Dict containing:
                - 'safe': Whether content is safe
                - 'reason': Reason for blocking if unsafe
        """
        # Check against blocked terms
        for term in self.blocked_terms:
            if term.lower() in text.lower():
                log_info(f"Content blocked due to term: {term}")
                return {"safe": False, "reason": f"Blocked term: {term}"}
        
        # Use Azure Content Safety API
        try:
            response = await self.client.moderations.create(input=text)
            results = response.results[0]
            
            for category, threshold in self.content_categories.items():
                if getattr(results.categories, category) > threshold:
                    log_info(f"Content filtered due to category: {category}")
                    return {
                        "safe": False,
                        "reason": f"Content filtered: {category}"
                    }
            
            log_info("Content is safe.")
            return {"safe": True, "reason": None}
        except Exception as e:
            log_error(f"Content filtering error: {e}")
            return {"safe": False, "reason": "Error in content check"}

    async def safe_completion(self, prompt: str) -> Optional[str]:
        """
        Generate a completion with content safety checks.
        
        Args:
            prompt: The input prompt to generate completion for
            
        Returns:
            The filtered response text if safe, None otherwise
            
        Raises:
            ValueError: If content is flagged as unsafe
        """
        # Check input content
        input_check = await self.check_content(prompt)
        if not input_check["safe"]:
            raise ValueError(f"Input content filtered: {input_check['reason']}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )

            output_text = response.choices[0].message.content
            
            # Check output content
            output_check = await self.check_content(output_text)
            if not output_check["safe"]:
                raise ValueError(f"Output content filtered: {output_check['reason']}")
                
            return output_text

        except Exception as e:
            log_error(f"Error during safe completion: {e}")
            return None

async def main():  
    """Example usage of ContentFilter."""  
    try:

# Initialize Azure OpenAI Client

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )

        # Initialize content filter
        content_filter = ContentFilter(client)
        
        # Add blocked terms
        content_filter.add_blocked_terms(["offensive term 1", "offensive term 2"])

        # Generate safe completion
        response = await content_filter.safe_completion("Your prompt here")
        if response:
            print(response)
        
    except ValueError as e:
        log_error(f"Content safety error: {e}")
    except Exception as e:
        log_error(f"Unexpected error: {e}")

if **name** == "**main**":  
    asyncio.run(main())  
[Timestamp: 2024/11/14 14:01:18]

**ai**: The provided code snippets represent two versions of a `ContentFilter` class designed to ensure content safety using Azure OpenAI's content moderation capabilities. Let's compare and highlight the differences between the two implementations:

## Key Differences

1. **Functionality and Structure:**
   - The first snippet includes a `safe_completion` method as part of the `ContentFilter` class, while the second snippet defines `safe_completion` as a standalone function.
   - The first snippet has a `main` function that demonstrates how to use the `ContentFilter` class, whereas the second snippet includes a usage example but comments out the execution of the `main` function.

2. **Error Handling:**
   - Both snippets handle exceptions during content checks and completion generation. However, the first snippet logs errors more comprehensively with `log_error` and provides detailed error messages.

3. **Type Annotations:**
   - The first snippet uses more detailed type annotations, including `Optional`, `Union`, and explicit return types for methods, which enhances code readability and maintainability.

4. **Azure OpenAI Client Usage:**
   - In the first snippet, the Azure OpenAI client is initialized within the `main` function using environment variables, whereas the second snippet assumes the client is already available and passed to the `ContentFilter`.

5. **Logging:**
   - Both snippets use logging to track blocked terms and content filtering decisions. The first snippet includes additional logging for successful content checks.

6. **Code Modularity:**
   - The first snippet encapsulates the content filtering logic within the `ContentFilter` class, making it more modular and easier to extend or modify.

## Suggested Improvements

- **Consistent Error Handling:** Ensure consistent error handling across both snippets, potentially by using custom exceptions for content filtering errors.
- **Environment Configuration:** Consider abstracting environment configuration (e.g., API keys) to a separate configuration module or function to enhance security and maintainability.
- **Testing:** Implement unit tests for both the `check_content` and `safe_completion` functions to ensure they handle various content scenarios correctly.

## Example Test Cases

Here are some suggested test cases to validate the functionality of the `ContentFilter` class:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_check_content_with_blocked_term():
    mock_client = AsyncMock()
    content_filter = ContentFilter(mock_client)
    content_filter.add_blocked_terms(["blocked"])
    
    result = await content_filter.check_content("This is a blocked term.")
    assert result["safe"] is False
    assert "Blocked term" in result["reason"]

@pytest.mark.asyncio
async def test_check_content_with_safe_content():
    mock_client = AsyncMock()
    mock_client.moderations.create.return_value = AsyncMock(results=[{"categories": {"hate": 0.1, "sexual": 0.1, "violence": 0.1, "self-harm": 0.1}}])
    content_filter = ContentFilter(mock_client)
    
    result = await content_filter.check_content("This is safe content.")
    assert result["safe"] is True
    assert result["reason"] is None

@pytest.mark.asyncio
async def test_safe_completion_with_safe_output():
    mock_client = AsyncMock()
    mock_client.moderations.create.return_value = AsyncMock(results=[{"categories": {"hate": 0.1, "sexual": 0.1, "violence": 0.1, "self-harm": 0.1}}])
    mock_client.chat.completions.create.return_value = AsyncMock(choices=[{"message": {"content": "Safe response"}}])
    content_filter = ContentFilter(mock_client)
    
    response = await content_filter.safe_completion("Safe prompt")
    assert response == "Safe response"
```

These test cases use `pytest` and `unittest.mock` to simulate the behavior of the Azure OpenAI client and validate the content filtering logic. Adjust these tests as needed based on your specific implementation and environment.  
[Timestamp: 2024/11/14 14:01:29]

**user**: Traceback (most recent call last):  
  File "/home/azureuser/simple2/main.py", line 7, in <module>  
    from api_client import AzureOpenAIClient  
  File "/home/azureuser/simple2/api_client.py", line 12, in <module>  
    from content_filter import ContentFilter # Import the ContentFilter  
    ====================================^^^^  
  File "/home/azureuser/simple2/content_filter.py", line 3, in <module>  
    import Union  
ModuleNotFoundError: No module named 'Union'  
[Timestamp: 2024/11/14 14:04:38]

**ai**: The error message indicates that there is an incorrect import statement in your `content_filter.py` file. The `Union` type is part of Python's `typing` module, and it should be imported from there. It seems there was an attempt to import `Union` directly, which is not correct.

Here's how you can fix the import statement:

1. Open your `content_filter.py` file.
2. Locate the import section at the top of the file.
3. Ensure that `Union` is imported from the `typing` module.

Here's the corrected import statement:

```python
from typing import Dict, List, Optional, Union
```

Make sure that your `content_filter.py` file includes this line in the import section. This will resolve the `ModuleNotFoundError` related to `Union`.

If you continue to encounter issues, double-check the rest of your import statements to ensure they are correct and that all necessary modules are installed in your environment. If you need further assistance, feel free to provide more details about your setup or any additional error messages.  
[Timestamp: 2024/11/14 14:04:41]

**user**: Traceback (most recent call last):  
  File "/home/azureuser/simple2/main.py", line 7, in <module>  
    from api_client import AzureOpenAIClient  
  File "/home/azureuser/simple2/api_client.py", line 12, in <module>  
    from content_filter import ContentFilter # Import the ContentFilter  
    ====================================^^^^  
  File "/home/azureuser/simple2/content_filter.py", line 6, in <module>  
    class ContentFilter:  
  File "/home/azureuser/simple2/content_filter.py", line 69, in ContentFilter  
    async def safe_completion(self, prompt: str, content_filter: ContentFilter) -> Optional[str]:  
                                                                 ============^  
NameError: name 'ContentFilter' is not defined  
[Timestamp: 2024/11/14 14:05:05]

**ai**: The `NameError` you're encountering is due to a circular reference in the type hint for the `safe_completion` method within the `ContentFilter` class. In Python, when you refer to a class within its own definition, you need to use a forward reference or avoid using the class name directly in the type hint.

Here's how you can resolve this issue:

## Solution

1. **Use a Forward Reference:**  
   You can use a string to refer to the class within its own definition. This is known as a forward reference.

   Update the type hint in the `safe_completion` method like this:

   ```python
   async def safe_completion(self, prompt: str, content_filter: 'ContentFilter') -> Optional[str]:
   ```

2. **Remove the Unnecessary Parameter:**  
   If `content_filter` is not needed as a parameter because it's a method of the `ContentFilter` class, you can remove it entirely:

   ```python
   async def safe_completion(self, prompt: str) -> Optional[str]:
   ```

## Updated Method

Here's how the method should look after applying the changes:

```python
async def safe_completion(self, prompt: str) -> Optional[str]:
    """
    Generate a completion with content safety checks.
    
    Args:
        prompt: The input prompt to generate completion for
        
    Returns:
        The filtered response text if safe, None otherwise
        
    Raises:
        ValueError: If content is flagged as unsafe
    """
    # Check input content
    input_check = await self.check_content(prompt)
    if not input_check["safe"]:
        raise ValueError(f"Input content filtered: {input_check['reason']}")

    try:
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        output_text = response.choices[0].message.content
        
        # Check output content
        output_check = await self.check_content(output_text)
        if not output_check["safe"]:
            raise ValueError(f"Output content filtered: {output_check['reason']}")
            
        return output_text

    except Exception as e:
        log_error(f"Error during safe completion: {e}")
        return None
```

## Additional Tips

- **Circular Imports:** Ensure that there are no circular imports in your project, as they can also lead to similar issues. If `api_client.py` imports `ContentFilter`, make sure `ContentFilter` does not import anything from `api_client.py` that could cause a circular dependency.
- **Testing:** After making these changes, test your code to ensure that the error is resolved and the functionality works as expected.  
[Timestamp: 2024/11/14 14:05:12]

**user**: Traceback (most recent call last):  
  File "/home/azureuser/simple2/main.py", line 11, in <module>  
    from interaction import InteractionHandler  
  File "/home/azureuser/simple2/interaction.py", line 10, in <module>  
    from documentation_analyzer import DocumentationAnalyzer  
  File "/home/azureuser/simple2/documentation_analyzer.py", line 7, in <module>  
    class DocumentationAnalyzer:  
  File "/home/azureuser/simple2/documentation_analyzer.py", line 209, in DocumentationAnalyzer  
    def verify_raises_section(self, function_node: ast.FunctionDef, raises_section: str) -> List[str]:  
                                                                                            ^^^^  
NameError: name 'List' is not defined. Did you mean: 'list'?  
[Timestamp: 2024/11/14 14:05:33]

**ai**: The `NameError` you're encountering indicates that the `List` type hint is being used without being imported from the `typing` module. In Python, `List` is a generic type that requires an import from the `typing` module to be used in type annotations.

Here's how you can resolve this issue:

## Solution

1. **Import `List` from `typing`:**  
   Make sure that you have the correct import statement at the top of your `documentation_analyzer.py` file.

   Add the following import if it's missing:

   ```python
   from typing import List
   ```

2. **Alternative: Use Built-in `list`:**  
   If you prefer not to use the `typing` module, you can use the built-in `list` type directly, although using `List` from `typing` is more explicit and recommended for type hinting.

   Update the type hint to use the built-in `list`:

   ```python
   def verify_raises_section(self, function_node: ast.FunctionDef, raises_section: str) -> list[str]:
   ```

## Example Update

Here's how the relevant part of your `documentation_analyzer.py` file should look:

```python
from typing import List
import ast

class DocumentationAnalyzer:
    # Other methods and attributes

    def verify_raises_section(self, function_node: ast.FunctionDef, raises_section: str) -> List[str]:
        """
        Verify the 'Raises' section of a function's docstring.

        Args:
            function_node: The AST node representing the function.
            raises_section: The 'Raises' section of the docstring.

        Returns:
            A list of issues found in the 'Raises' section.
        """
        # Implementation of the method
        issues = []
        # Example logic to populate issues
        return issues
```

## Additional Tips

- **Consistent Imports:** Ensure that all type hints used in your code are consistently imported from the `typing` module if they are not built-in types.
- **Code Review:** Review other parts of your codebase for similar issues, especially if you are using type hints extensively.
- **Testing:** After making these changes, run your code to verify that the error is resolved and the functionality behaves as expected.  
[Timestamp: 2024/11/14 14:05:46]
