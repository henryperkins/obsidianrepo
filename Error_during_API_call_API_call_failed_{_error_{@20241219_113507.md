---
epoch: 1734629707966
modelKey: azureml://registries/azure-openai/models/gpt-4o/versions/2024-11-20|azure openai
tags:
  - copilot-conversation
---

**user**: Error during API call: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Traceback (most recent call last):
  File "/home/azureuser/simple/core/ai_service.py", line 338, in _make_api_call_with_retry
    raise APICallError(f"API call failed: {error_text}")
core.exceptions.APICallError: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}

[Timestamp: 2024/12/19 11:35:07]

**ai**: It looks like the API call failed because the total number of tokens requested exceeded the model's maximum context length of 128,000 tokens. The error message indicates that you requested 128,007 tokens, with 2,133 tokens in the messages and 125,874 tokens in the completion.

To resolve this issue, you need to reduce the length of either the messages or the completion. Here are a few suggestions:

1. **Shorten the Messages**: Review the content of your messages and see if you can condense or remove any unnecessary information.
2. **Reduce the Completion Length**: If possible, limit the expected length of the completion by being more specific in your request or by asking for less detailed responses.

Please try adjusting your input and retry the API call. If you need further assistance or have any other questions, feel free to ask!
[Timestamp: 2024/12/19 11:35:11]

**user**: [config](./config.md) [token_management](./token_management.md)
[Timestamp: 2024/12/19 11:35:24]

**ai**: The error you encountered is due to exceeding the model's maximum token limit of 128,000 tokens. Based on the provided configuration and token management logic, here are steps to address the issue:

### 1. **Review and Adjust Token Limits**
   - In the `AIConfig` class, the `max_tokens` parameter is set to `128000`. Ensure this value aligns with the model's actual capability.
   - If you are setting a `max_completion_tokens` value in your environment or code, ensure it doesn't exceed the difference between `max_tokens` and the prompt token count.

### 2. **Optimize the Prompt**
   - Use the `_estimate_tokens` method in `TokenManager` to calculate the token count of your prompt before sending the request.
   - If the prompt exceeds the allowed token count, implement a truncation strategy (e.g., summarizing or removing less relevant parts of the prompt).

### 3. **Adjust Completion Token Limit**
   - The `_calculate_max_completion` method dynamically adjusts the completion token limit based on the remaining tokens. Ensure this logic is applied correctly.
   - If `max_tokens` is explicitly set in the request, ensure it is within the available token budget.

### 4. **Implement Truncation Strategy**
   - The `truncation_strategy` parameter in `AIConfig` can be used to define how to handle prompts or completions that exceed the token limit. Implement logic to truncate or summarize content when necessary.

### 5. **Enable Logging for Debugging**
   - Use the logger in `TokenManager` to debug token usage. For example:
     ```python
     self.logger.debug(f"Prompt Tokens: {prompt_tokens}, Available Tokens: {available_tokens}")
     ```
   - This will help identify whether the issue is with the prompt or the completion length.

### 6. **Rate Limiting**
   - Ensure the `_check_rate_limits` function is working correctly to avoid overwhelming the API with requests.

### Example Adjustment
Here’s an example of how you might adjust the `validate_and_prepare_request` method to handle token limits more gracefully:

```python
async def validate_and_prepare_request(
    self,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    truncation_strategy: Optional[str] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    response_format: Optional[str] = None,
    stream_options: Optional[dict] = None
) -> dict[str, Any]:
    try:
        # Check rate limits
        await self._check_rate_limits()

        # Estimate tokens
        prompt_tokens = self._estimate_tokens(prompt)
        if prompt_tokens > self.model_config.max_tokens:
            raise ValueError(f"Prompt exceeds token limit: {prompt_tokens} > {self.model_config.max_tokens}")
        
        available_tokens = self.model_config.max_tokens - prompt_tokens
        max_completion = self._calculate_max_completion(available_tokens, max_tokens)

        # Prepare request parameters
        request_params = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_completion,
            "temperature": temperature or self.config.temperature,
        }

        # Track token usage
        self.track_request(prompt_tokens, max_completion)

        return request_params

    except Exception as e:
        self.logger.error(f"Error preparing request: {e}", exc_info=True)
        raise ProcessingError(f"Failed to prepare request: {str(e)}")
```

### 7. **Testing and Debugging**
   - Test with various prompt sizes to ensure the logic handles edge cases.
   - Use the `get_usage_stats` method in `TokenManager` to monitor token usage trends.

By implementing these adjustments, you should be able to avoid exceeding the token limit and handle requests more efficiently. Let me know if you need further assistance!
[Timestamp: 2024/12/19 11:37:04]

**user**: [ai_service](./ai_service.md) [prompt_manager](./prompt_manager.md)
[Timestamp: 2024/12/19 11:37:33]

**user**: is it really using 128k tokens in one completion ?
[Timestamp: 2024/12/19 11:37:48]

**ai**: From the provided context, the `AIService` and `PromptManager` modules are responsible for managing interactions with the Azure OpenAI API and generating prompts for those interactions. The error message you encountered earlier regarding exceeding the token limit can be addressed by refining the logic in these modules.

### Key Areas to Address in `AIService` and `PromptManager`

#### 1. **Token Management in Prompt Creation**
   - The `PromptManager` is responsible for generating prompts. Ensure that the generated prompt length is within the token limit by integrating token estimation during prompt creation.
   - Use the `TokenManager`'s `_estimate_tokens` method to calculate the token count of the generated prompt.

   **Solution in `PromptManager`:**
   Modify the `create_documentation_prompt` method to check token limits after rendering the template.

   ```python
   async def create_documentation_prompt(
       self,
       context: DocumentationContext,
   ) -> ProcessingResult:
       """
       Create a documentation prompt using the documentation template.

       Args:
           context: Structured context containing all necessary documentation information

       Returns:
           ProcessingResult containing the generated prompt and associated metrics
       """
       print_info("Generating documentation prompt using template.")
       start_time = time.time()

       try:
           # Generate prompt using template
           prompt = self.documentation_template.render(
               module_name=context.metadata.get("module_name", ""),
               file_path=str(context.module_path),
               source_code=context.source_code,
               classes=context.classes,
               functions=context.functions,
           )

           # Estimate token usage
           prompt_tokens = self.token_manager._estimate_tokens(prompt)
           if prompt_tokens > self.token_manager.model_config.max_tokens:
               raise ValueError(
                   f"Generated prompt exceeds token limit: {prompt_tokens} > "
                   f"{self.token_manager.model_config.max_tokens}"
               )

           # Track token usage
           token_usage = await self._calculate_token_usage(prompt)

           # Track metrics
           metrics = await self._create_metrics(prompt, start_time)

           return ProcessingResult(
               content={"prompt": prompt},
               usage=token_usage.__dict__,
               metrics=metrics.__dict__,
               validation_status=True,
               validation_errors=[],
               schema_errors=[],
           )

       except Exception as e:
           self.logger.error(
               f"Error generating documentation prompt: {e}", exc_info=True
           )
           return ProcessingResult(
               content={},
               usage={},
               metrics={},
               validation_status=False,
               validation_errors=[str(e)],
               schema_errors=[],
           )
   ```

#### 2. **Truncation Strategy**
   - If the prompt exceeds the token limit, implement a truncation strategy to reduce its size.
   - For example, truncate the source code or exclude less relevant classes or functions.

   **Example Truncation Logic:**
   ```python
   def truncate_source_code(self, source_code: str, max_tokens: int) -> str:
       """
       Truncate the source code to fit within the token limit.

       Args:
           source_code: The full source code.
           max_tokens: The maximum allowed tokens.

       Returns:
           Truncated source code.
       """
       tokens = self.token_manager._estimate_tokens(source_code)
       if tokens <= max_tokens:
           return source_code

       # Truncate the source code by lines
       lines = source_code.splitlines()
       truncated_code = []
       for line in lines:
           truncated_code.append(line)
           if self.token_manager._estimate_tokens("\n".join(truncated_code)) > max_tokens:
               break

       return "\n".join(truncated_code[:-1])  # Exclude the last line that exceeded the limit
   ```

#### 3. **Completion Token Limit in `AIService`**
   - In the `_make_api_call_with_retry` method, ensure the `max_tokens` parameter for the completion is calculated dynamically based on the prompt token count.

   **Solution in `AIService`:**
   ```python
   request_params = await self.token_manager.validate_and_prepare_request(
       prompt,
       max_tokens=self.config.max_tokens - prompt_tokens,  # Adjust max_tokens dynamically
       temperature=self.config.temperature,
       truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
       tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
       parallel_tool_calls=self.config.parallel_tool_calls,
       response_format=str(self.config.response_format) if self.config.response_format else None,
       stream_options=self.config.stream_options,
   )
   ```

#### 4. **Error Handling for Token Limit Exceedance**
   - If the token limit is exceeded during the API call, log the error and provide a fallback response.

   **Solution in `_make_api_call_with_retry`:**
   ```python
   except ValueError as e:
       if "token limit" in str(e).lower():
           self.logger.error(
               f"Token limit exceeded: {e}", extra=log_extra
           )
           raise APICallError(f"Token limit exceeded: {e}")
   ```

#### 5. **Testing with Large Inputs**
   - Test the system with large source code files to ensure the truncation logic and token management work as expected.
   - Use the `get_usage_stats` method in `TokenManager` to monitor token usage during testing.

### Summary of Changes
- **`PromptManager`**:
  - Added token estimation and truncation logic to `create_documentation_prompt`.
  - Implemented a `truncate_source_code` method for handling large inputs.

- **`AIService`**:
  - Adjusted `max_tokens` dynamically in `_make_api_call_with_retry`.
  - Improved error handling for token limit exceedance.

With these changes, the system should handle large inputs and avoid exceeding the token limit while maintaining robust error handling and logging. Let me know if you need further assistance!
[Timestamp: 2024/12/19 11:37:49]

**ai**: No, it is highly unlikely that the API is actually using **128,000 tokens** in a single completion unless the input prompt or the requested completion is extremely large. Let’s break this down and analyze the situation:

---

### **1. What the Error Means**
The error indicates that the **total token count** (prompt + completion) exceeded the model's maximum context length of **128,000 tokens**. Specifically:
- **Prompt tokens**: 2,133
- **Completion tokens (requested)**: 125,874
- **Total tokens**: 128,007 (exceeding the limit by 7 tokens)

This suggests that the **completion tokens requested** are far too high compared to what is reasonable for most use cases.

---

### **2. Why 125,874 Completion Tokens Is Unrealistic**
- **Practical Limitations**: A completion of 125,874 tokens would correspond to generating a response with approximately **100,000+ words** (assuming ~1.2 tokens per word). This is far beyond the scope of most API use cases.
- **Configuration Issue**: This likely indicates a **misconfiguration** in your code, where the `max_tokens` parameter for the completion is being set to an unreasonably high value.

---

### **3. Where the Issue Might Be**
The problem is likely in how the `max_tokens` parameter is being calculated or passed to the API. Specifically:
1. **In `validate_and_prepare_request`**:
   - The `max_tokens` for the completion might not be properly constrained by the available token budget (i.e., `model_config.max_tokens - prompt_tokens`).
   - If `max_tokens` is not explicitly provided, the fallback logic might be defaulting to an excessively high value.

2. **In the Configuration**:
   - The `max_completion_tokens` in your `AIConfig` might be set to an unreasonable value, such as 125,874, instead of something more practical like 4,096 or 8,000.

3. **In the Request Logic**:
   - Somewhere in your code, the API request might be overriding the calculated `max_completion` with an excessively high value.

---

### **4. How to Fix It**
Here’s how you can debug and resolve the issue:

#### **Step 1: Check the `max_tokens` Parameter**
Inspect the logic in your `validate_and_prepare_request` method. Ensure that the `max_tokens` for the completion is being calculated correctly:
```python
available_tokens = self.model_config.max_tokens - prompt_tokens
max_completion = min(max_tokens or available_tokens, available_tokens)
```
- If `max_tokens` is explicitly provided, it should be capped by the `available_tokens`.
- If `max_tokens` is not provided, it should default to the remaining token budget (`available_tokens`).

#### **Step 2: Add Logging**
Add debug logs to confirm the values being used for `prompt_tokens`, `available_tokens`, and `max_completion`:
```python
self.logger.debug(f"Prompt Tokens: {prompt_tokens}, Available Tokens: {available_tokens}, Max Completion: {max_completion}")
```

#### **Step 3: Check Default Configuration**
Inspect the `AIConfig` class and ensure that `max_completion_tokens` is not set to an unreasonably high value:
```python
max_completion_tokens=get_env_var("AZURE_MAX_COMPLETION_TOKENS", None, int, False)
```
- If this value is set in your environment, make sure it is reasonable (e.g., 4,096 or 8,000).

#### **Step 4: Validate the Request**
Before sending the API request, validate the total token count:
```python
if prompt_tokens + max_completion > self.model_config.max_tokens:
    raise ValueError("Total token count exceeds the model's maximum context length.")
```

---

### **5. Recommended Token Limits**
For most use cases:
- **Prompt tokens**: Typically range from a few hundred to a few thousand.
- **Completion tokens**: Should rarely exceed 4,096 (or 8,000 for larger models like GPT-4).
- **Total tokens**: Should always stay within the model's maximum context length (e.g., 128,000 for GPT-4).

---

### **6. Example Fix**
Here’s an updated version of your `validate_and_prepare_request` method with proper token management:

```python
async def validate_and_prepare_request(
    self,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    truncation_strategy: Optional[str] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    response_format: Optional[str] = None,
    stream_options: Optional[dict] = None
) -> dict[str, Any]:
    try:
        # Check rate limits
        await self._check_rate_limits()

        # Estimate tokens
        prompt_tokens = self._estimate_tokens(prompt)
        available_tokens = self.model_config.max_tokens - prompt_tokens

        if prompt_tokens > self.model_config.max_tokens:
            raise ValueError(f"Prompt exceeds token limit: {prompt_tokens} > {self.model_config.max_tokens}")

        # Calculate max completion tokens
        max_completion = min(max_tokens or available_tokens, available_tokens)

        if prompt_tokens + max_completion > self.model_config.max_tokens:
            raise ValueError(
                f"Total token count exceeds limit: {prompt_tokens + max_completion} > {self.model_config.max_tokens}"
            )

        # Prepare request parameters
        request_params = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_completion,
            "temperature": temperature or self.config.temperature,
        }

        # Track token usage
        self.track_request(prompt_tokens, max_completion)

        return request_params

    except Exception as e:
        self.logger.error(f"Error preparing request: {e}", exc_info=True)
        raise ProcessingError(f"Failed to prepare request: {str(e)}")
```

---

### **7. Conclusion**
The error is almost certainly due to a misconfiguration or oversight in how the `max_tokens` parameter is being calculated or passed. By implementing proper token management and adding logging, you can ensure that the total token count stays within the model's limits.

Let me know if you need further clarification or assistance!
[Timestamp: 2024/12/19 11:38:00]

**user**: Generate the complete, updated module
[Timestamp: 2024/12/19 11:38:31]

**ai**: Here is the updated and complete `TokenManager` module with fixes for token management, proper validation, and logging to ensure that the total token count stays within the model's limits:

---

### **Updated `TokenManager` Module**
```python
"""Token management module for interacting with the OpenAI API."""

import tiktoken
import sentry_sdk
from typing import Any, Tuple, Union, Optional
import time
import asyncio

from core.config import AIConfig
from core.logger import LoggerSetup, CorrelationLoggerAdapter
from core.types import TokenUsage
from core.exceptions import ProcessingError
from core.console import print_info, print_success, print_error
from core.metrics_collector import MetricsCollector


class TokenManager:
    """Manages token usage and cost estimation for Azure OpenAI API interactions."""

    def __init__(
        self,
        model: str,
        config: Optional[AIConfig] = None,
        correlation_id: Optional[str] = None,
        metrics_collector: Optional['MetricsCollector'] = None
    ) -> None:
        """Initialize TokenManager with Azure OpenAI configurations."""
        self.logger = CorrelationLoggerAdapter(
            logger=LoggerSetup.get_logger(__name__),
            extra={"correlation_id": correlation_id}
        )
        self.config = config if config else AIConfig.from_env()
        self.model = model
        self.deployment_id = self.config.deployment
        self.metrics_collector = metrics_collector or MetricsCollector(
            correlation_id=correlation_id
        )
        self.correlation_id = correlation_id

        # Initialize token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Initialize encoding
        try:
            base_model = self._get_base_model_name(self.model)
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError as e:
            self.logger.warning(
                f"Model {self.model} not found. Using cl100k_base encoding.",
                exc_info=True,
                extra={"model": self.model}
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
            sentry_sdk.capture_exception(e)

        self.model_config = self.config.model_limits.get(
            self.model,
            self.config.model_limits["gpt-4o-2024-11-20"]
        )

        self._initialize_rate_limiting()

    def _get_base_model_name(self, model_name: str) -> str:
        """
        Get the base model name from a deployment model name.
        """
        model_mappings = {
            "gpt-4o": "gpt-4o-2024-11-20",  # New primary model
            "gpt-35-turbo": "gpt-3.5-turbo",  # Keep fallback options
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

        for key, value in model_mappings.items():
            if key in model_name.lower():
                return value

        self.logger.warning(
            f"Unknown model {model_name}, defaulting to gpt-4o for token encoding",
             extra={"model": model_name}
        )
        return "gpt-4o-2024-11-20"  # Default to our primary model

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}", exc_info=True, extra={"text_snippet": text[:50]})
            return len(text) // 4

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage statistics."""
        total_tokens = prompt_tokens + completion_tokens
        cost_per_token = self.model_config.cost_per_token
        estimated_cost = total_tokens * cost_per_token

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost
        )

    def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting for Azure OpenAI API."""
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.rate_limit_per_minute = getattr(
            self.model_config, 'rate_limit', 10)  # Default to 10 if not specified
        self.request_times: list[float] = []

    async def validate_and_prepare_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        truncation_strategy: Optional[str] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[str] = None,
        stream_options: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Validate and prepare a request with token management.

        Args:
            prompt: The input prompt for the API.
            max_tokens: The maximum tokens for the completion.
            temperature: Sampling temperature for the model.
            truncation_strategy: Strategy for truncating input if needed.
            tool_choice: Tool selection for function calling.
            parallel_tool_calls: Whether to allow parallel tool calls.
            response_format: Expected response format.
            stream_options: Options for streaming responses.

        Returns:
            A dictionary of request parameters for the API call.

        Raises:
            ValueError: If the total token count exceeds the model's limit.
        """
        try:
            # Check rate limits
            await self._check_rate_limits()

            # Estimate tokens
            prompt_tokens = self._estimate_tokens(prompt)
            available_tokens = self.model_config.max_tokens - prompt_tokens

            if prompt_tokens > self.model_config.max_tokens:
                raise ValueError(
                    f"Prompt exceeds Azure OpenAI token limit: {prompt_tokens} > "
                    f"{self.model_config.max_tokens}"
                )

            # Calculate max completion tokens
            max_completion = min(max_tokens or available_tokens, available_tokens)

            if prompt_tokens + max_completion > self.model_config.max_tokens:
                raise ValueError(
                    f"Total token count exceeds limit: {prompt_tokens + max_completion} > "
                    f"{self.model_config.max_tokens}"
                )

            # Prepare request parameters
            request_params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_completion,
                "temperature": temperature or self.config.temperature,
            }

            # Track token usage
            self.track_request(prompt_tokens, max_completion)

            # Log the input sent to the AI
            self.logger.debug(
                "Prepared Request Parameters",
                extra={"request_params": request_params}
            )
            return request_params

        except Exception as e:
            self.logger.error(f"Error preparing request: {e}", exc_info=True, extra={"prompt_snippet": prompt[:50]})
            print_error(f"Failed to prepare request: {str(e)}")
            raise ProcessingError(f"Failed to prepare request: {str(e)}")

    async def _check_rate_limits(self) -> None:
        """Check and enforce Azure OpenAI rate limits."""
        current_time = time.time()
        
        # Clean old request times
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.request_times) >= self.rate_limit_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                self.logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.2f} seconds.",
                    extra={"wait_time": wait_time}
                )
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

    def get_usage_stats(self) -> dict[str, Union[int, float]]:
        """
        Get current token usage statistics.

        Returns:
            A dictionary containing token usage statistics.
        """
        usage = self._calculate_usage(
            self.total_prompt_tokens, self.total_completion_tokens
        )
        stats = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost": usage.estimated_cost,
        }
        print_info("Token usage stats retrieved.", stats)
        return stats

    def track_request(self, prompt_tokens: int, max_completion: int) -> None:
        """
        Track token usage for a request.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            max_completion: Number of tokens allocated for the completion.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += max_completion
        self.logger.info(
            f"Tracked request - Prompt Tokens: {prompt_tokens}, Max Completion Tokens: {max_completion}",
            extra={"correlation_id": self.correlation_id, "prompt_tokens": prompt_tokens, "max_completion": max_completion}
        )
        print_success(f"Tokens tracked: {prompt_tokens + max_completion} total tokens.")

    async def process_completion(self, completion: Any) -> Tuple[str, dict[str, Any]]:
        """
        Process completion response and track token usage.

        Args:
            completion: The raw completion response from the API.

        Returns:
            A tuple containing the processed content and usage statistics.

        Raises:
            ProcessingError: If there is an error processing the completion response.
        """
        try:
            message = completion["choices"][0]["message"]

            if "function_call" in message:
                content = message["function_call"]["arguments"]
            else:
                content = message.get("content", "")

            usage = completion.get("usage", {})
            
            if usage:
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                
                if self.metrics_collector:
                    await self.metrics_collector.track_operation(
                        "token_usage",
                        success=True,
                        duration=0,
                        usage=self._calculate_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)).__dict__,
                        metadata={
                            "model": self.model,
                            "deployment_id": self.deployment_id,
                            "correlation_id": self.correlation_id
                        },
                    )
            
                self.logger.info(
                    f"Processed completion - Content Length: {len(content)}, Usage: {usage}",
                    extra={"correlation_id": self.correlation_id, "content_length": len(content), "usage": usage}
                )
                print_success("Completion processed successfully.")

            return content, usage if isinstance(usage, dict) else {}

        except Exception as e:
            self.logger.error(f"Error processing completion: {e}", exc_info=True)
            print_error(f"Failed to process completion: {str(e)}")
            raise ProcessingError(f"Failed to process completion: {str(e)}")
```

---

### **Key Updates**
1. **Token Validation**:
   - Ensures the total token count (prompt + completion) does not exceed the model's maximum context length.
   - Raises a `ValueError` if the token count exceeds the limit.

2. **Dynamic Completion Token Calculation**:
   - Dynamically calculates the `max_completion` tokens based on the remaining token budget (`available_tokens`).

3. **Logging**:
   - Added detailed debug logs to track token usage, request parameters, and errors.

4. **Error Handling**:
   - Improved error handling for token-related issues and added meaningful error messages.

5. **Rate Limiting**:
   - Ensures compliance with API rate limits by tracking requests per minute.

---

This module is now robust, ensuring that token limits are respected and providing detailed logging and error handling to debug any issues. Let me know if you need further assistance!
[Timestamp: 2024/12/19 11:38:58]

**user**:   Tokens tracked: 128000 total tokens.
AI Service client session initialized
Making API call

Token usage stats retrieved.
  total_prompt_tokens: 0
  total_completion_tokens: 0
  total_tokens: 0
  estimated_cost: 0.0

Operation: system_cpu
  Success: True
  Duration: 60.00s

Operation: system_memory
  Success: True
  Duration: 60.00s

Operation: system_disk
  Success: True
  Duration: 60.00s

Operation: system_tokens
  Success: True
  Duration: 60.00s
Stored and cleaned up metrics
API call failed
Error during API call: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Traceback (most recent call last):
  File "/home/azureuser/simple/core/ai_service.py", line 338, in _make_api_call_with_retry
    raise APICallError(f"API call failed: {error_text}")
core.exceptions.APICallError: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Making API call
API call failed
Error during API call: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Traceback (most recent call last):
  File "/home/azureuser/simple/core/ai_service.py", line 338, in _make_api_call_with_retry
    raise APICallError(f"API call failed: {error_text}")
core.exceptions.APICallError: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Making API call
API call failed
Error during API call: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
Traceback (most recent call last):
  File "/home/azureuser/simple/core/ai_service.py", line 338, in _make_api_call_with_retry
    raise APICallError(f"API call failed: {error_text}")
core.exceptions.APICallError: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}

Operation: documentation_generation
  Success: False
  Duration: 4.12s
  error: API call failed after 3 retries: API call failed: {
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, you requested 128007 tokens (2133 in the messages, 125874 in the completion). Please reduce the length of the messages or completion.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
[Timestamp: 2024/12/19 11:41:25]

**ai**: The error message confirms that the **completion token count** is being set to an unreasonably high value of **125,874 tokens**, which is causing the total token count (prompt + completion) to exceed the **128,000 token limit** of the model. This behavior indicates a **misconfiguration** or **logic error** in the code that calculates or sets the `max_tokens` parameter for the completion.

Here’s how you can address and resolve the issue:

---

### **1. Root Cause of the Issue**
- The **prompt tokens** are correctly calculated as **2,133 tokens**, which is reasonable.
- The **completion tokens** are being set to **125,874 tokens**, which is unrealistic and far beyond what is typically needed for any response.
- This suggests that either:
  - The `max_tokens` parameter for the completion is being set to an excessively high value.
  - The logic for calculating the `max_tokens` is not properly capping the value based on the model's token limit.

---

### **2. Key Areas to Investigate**
#### **a. `validate_and_prepare_request` in `TokenManager`**
This method is responsible for validating the prompt and calculating the `max_tokens` for the completion. Ensure that:
- The `max_tokens` for the completion is dynamically capped based on the remaining token budget.
- The total token count (prompt + completion) does not exceed the model's maximum context length.

Here’s the updated logic:

```python
async def validate_and_prepare_request(
    self,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    truncation_strategy: Optional[str] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    response_format: Optional[str] = None,
    stream_options: Optional[dict] = None
) -> dict[str, Any]:
    """
    Validate and prepare a request with token management.

    Args:
        prompt: The input prompt for the API.
        max_tokens: The maximum tokens for the completion.
        temperature: Sampling temperature for the model.
        truncation_strategy: Strategy for truncating input if needed.
        tool_choice: Tool selection for function calling.
        parallel_tool_calls: Whether to allow parallel tool calls.
        response_format: Expected response format.
        stream_options: Options for streaming responses.

    Returns:
        A dictionary of request parameters for the API call.

    Raises:
        ValueError: If the total token count exceeds the model's limit.
    """
    try:
        # Check rate limits
        await self._check_rate_limits()

        # Estimate tokens for the prompt
        prompt_tokens = self._estimate_tokens(prompt)
        available_tokens = self.model_config.max_tokens - prompt_tokens

        if prompt_tokens > self.model_config.max_tokens:
            raise ValueError(
                f"Prompt exceeds Azure OpenAI token limit: {prompt_tokens} > "
                f"{self.model_config.max_tokens}"
            )

        # Calculate max completion tokens
        max_completion = min(max_tokens or available_tokens, available_tokens)

        # Validate total token count
        if prompt_tokens + max_completion > self.model_config.max_tokens:
            raise ValueError(
                f"Total token count exceeds limit: {prompt_tokens + max_completion} > "
                f"{self.model_config.max_tokens}"
            )

        # Prepare request parameters
        request_params = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_completion,
            "temperature": temperature or self.config.temperature,
        }

        # Track token usage
        self.track_request(prompt_tokens, max_completion)

        # Log the input sent to the AI
        self.logger.debug(
            "Prepared Request Parameters",
            extra={"request_params": request_params}
        )
        return request_params

    except Exception as e:
        self.logger.error(f"Error preparing request: {e}", exc_info=True, extra={"prompt_snippet": prompt[:50]})
        raise ProcessingError(f"Failed to prepare request: {str(e)}")
```

---

### **3. Add Logging to Debug Token Values**
To understand what values are being calculated and passed, add detailed logging for:
- The **prompt tokens**.
- The **available tokens**.
- The **max completion tokens**.

Example logging:
```python
self.logger.debug(
    f"Token Management Debug: Prompt Tokens = {prompt_tokens}, "
    f"Available Tokens = {available_tokens}, "
    f"Max Completion Tokens = {max_completion}"
)
```

---

### **4. Verify Configuration in `AIConfig`**
Check the `AIConfig` class to ensure that the `max_completion_tokens` is not set to an unreasonably high value. For example:
```python
max_completion_tokens=get_env_var("AZURE_MAX_COMPLETION_TOKENS", None, int, False)
```

- If this environment variable is set, ensure it is reasonable (e.g., 4,096 or 8,000).
- If not set, the fallback logic in `validate_and_prepare_request` should cap the completion tokens dynamically based on the remaining token budget.

---

### **5. Validate Total Token Count Before API Call**
Before making the API call in `_make_api_call_with_retry`, validate the total token count to ensure it does not exceed the model's limit:
```python
if prompt_tokens + max_completion > self.model_config.max_tokens:
    raise ValueError("Total token count exceeds the model's maximum context length.")
```

---

### **6. Example Fix in `_make_api_call_with_retry`**
Here’s how you can integrate the updated `validate_and_prepare_request` logic into the `_make_api_call_with_retry` method in `AIService`:

```python
async def _make_api_call_with_retry(
    self,
    prompt: str,
    function_schema: Optional[dict[str, Any]],
    max_retries: int = 3,
    log_extra: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Make an API call with retry logic following Azure best practices."""
    headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
    if self.correlation_id:
        headers["x-correlation-id"] = self.correlation_id

    # Validate and prepare request parameters
    request_params = await self.token_manager.validate_and_prepare_request(
        prompt,
        max_tokens=self.config.max_tokens,  # Ensure this is capped dynamically
        temperature=self.config.temperature,
        truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
        tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
        parallel_tool_calls=self.config.parallel_tool_calls,
        response_format=str(self.config.response_format) if self.config.response_format else None,
        stream_options=self.config.stream_options,
    )

    # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
    url = (
        f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
        f"/chat/completions?api-version={self.config.azure_api_version}"
    )

    for attempt in range(max_retries):
        try:
            if self._client is None:
                await self.start()

            if self._client:
                self.logger.info(
                    "Making API call",
                    extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                )

                async with self._client.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        self.logger.info(
                            "API call succeeded",
                            extra={"status_code": response.status, "correlation_id": self.correlation_id},
                        )
                        return response_json

                    error_text = await response.text()
                    self.logger.error(
                        "API call failed",
                        extra={
                            "status_code": response.status,
                            "error_text": error_text[:200],
                            "correlation_id": self.correlation_id,
                        },
                    )

                    # Handle specific error cases
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 2**attempt))
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 503:  # Service unavailable
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise APICallError(f"API call failed: {error_text}")

        except asyncio.TimeoutError:
            self.logger.warning(f"Request timeout (attempt {attempt + 1})", extra=log_extra)
            await asyncio.sleep(2**attempt)
        except Exception as e:
            self.logger.error(f"Error during API call: {e}", exc_info=True, extra=log_extra)
            if attempt == max_retries - 1:
                raise APICallError(f"API call failed after {max_retries} retries: {e}")
            await asyncio.sleep(2**attempt)
```

---

### **7. Testing**
- Test the updated logic with various prompt sizes to ensure the total token count stays within the limit.
- Use the logging output to verify the calculated token values for the prompt, completion, and total.

---

By implementing these fixes, the system should correctly manage token limits and avoid exceeding the model's maximum context length. Let me know if you need further assistance!
[Timestamp: 2024/12/19 11:41:54]

**user**: Update the config
[Timestamp: 2024/12/19 11:45:03]

**ai**: TypeError: Load failed
[Timestamp: 2024/12/19 11:46:32]

**ai**: Here’s the updated `AIConfig` and `Config` classes with improvements to ensure proper token management, dynamic calculation of `max_completion_tokens`, and better handling of environment variables.

---

### **Updated `config.py`**

```python
"""Configuration module for AI documentation service."""

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv
import uuid
from pathlib import Path

# Load environment variables
load_dotenv()

# Define base paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS_DIR = ROOT_DIR / "repos"
DOCS_OUTPUT_DIR = ROOT_DIR / "docs_output"


def get_env_var(
    key: str, default: Any = None, var_type: type = str, required: bool = False
) -> Any:
    """Get environment variable with type conversion and validation.

    Args:
        key: Environment variable key
        default: Default value if not found
        var_type: Type to convert the value to
        required: Whether the variable is required

    Returns:
        The environment variable value converted to the specified type

    Raises:
        ValueError: If a required variable is missing or type conversion fails
    """
    value = os.getenv(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable {key} is not set")
        return default

    try:
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        return
[Timestamp: 2024/12/19 11:46:32]