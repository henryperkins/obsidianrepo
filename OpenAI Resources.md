# Merged OpenAI Reference and Resources

## Overview

This document provides a comprehensive guide to using the OpenAI API, including basic usage, advanced features, and practical examples. It covers topics such as API authentication, rate limiting, structured outputs, and function calling. Additionally, it provides insights into handling large datasets, streaming responses, and building tool-using agents.

---

## 1. OpenAI Python Library - Basic and Advanced Usage

### Basic Usage
**Link:** [OpenAI Python Library - Usage](https://github.com/openai/openai-python?tab=readme-ov-file#usage)

- **Installation and Authentication:** Instructions for setting up the OpenAI Python library and authenticating with your API key.
- **Basic Completion Example:**
  ```python
  import openai

  openai.api_key = 'your-api-key-here'

  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Once upon a time",
      max_tokens=50
  )

  print(response.choices[0].text.strip())
  ```

### Advanced Usage
**Link:** [OpenAI Python Library - Advanced](https://github.com/openai/openai-python?tab=readme-ov-file#advanced)

- **Fine-Tuning and Multiple Requests:** Advanced features such as fine-tuning models and managing multiple requests.
- **Example:**
  ```python
  import openai

  openai.api_key = 'your-api-key-here'

  response = openai.Completion.create(
      model="curie:ft-your-organization-id-2023-01-01-12-00-00",
      prompt="Explain the theory of relativity",
      max_tokens=100
  )

  print(response.choices[0].text.strip())
  ```

---

## 2. Structured Outputs and Function Calling

### Structured Outputs Parsing Helpers
**Link:** [Structured Outputs Parsing Helpers](https://github.com/openai/openai-python/blob/main/helpers.md#structured-outputs-parsing-helpers)

- **Parsing Complex Responses:** Helper functions for parsing structured outputs from the API.
- **Example:**
  ```python
  import json

  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Generate a JSON object with name and age",
      max_tokens=50
  )

  structured_output = json.loads(response.choices[0].text.strip())
  print(structured_output)
  ```

### Function Calling with Structured Outputs
**Link:** [Function Calling with Structured Outputs](https://platform.openai.com/docs/guides/function-calling)

- **Defining Function Schemas:**
  ```python
  functions = [
      {
          "name": "process_user_data",
          "description": "Process user information into a structured format",
          "parameters": {
              "type": "object",
              "properties": {
                  "name": {"type": "string"},
                  "age": {"type": "integer"},
                  "interests": {
                      "type": "array",
                      "items": {"type": "string"}
                  }
              },
              "required": ["name", "age"]
          }
      }
  ]
  ```

---

## 3. Handling Rate Limits and Token Counting

### Handling Rate Limits
**Link:** [Handling Rate Limits](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb)

- **Strategies for Managing API Rate Limits:** Implementing exponential backoff and other techniques.
- **Example:**
  ```python
  import time

  def make_request_with_backoff(prompt):
      for attempt in range(5):
          try:
              response = openai.Completion.create(
                  engine="text-davinci-003",
                  prompt=prompt,
                  max_tokens=50
              )
              return response.choices[0].text.strip()
          except openai.error.RateLimitError:
              print(f"Rate limit exceeded. Retrying in {2 ** attempt} seconds...")
              time.sleep(2 ** attempt)
      return None
  ```

### Counting Tokens with Tiktoken
**Link:** [Counting Tokens with Tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)

- **Token Counting for API Requests:**
  ```python
  import tiktoken

  prompt = "Hello, how are you?"
  encoding = tiktoken.encoding_for_model("text-davinci-003")
  num_tokens = len(encoding.encode(prompt))

  print(f"Number of tokens in prompt: {num_tokens}")
  ```

---

## 4. Streaming Responses and Batch Processing

### Streaming Responses
**Link:** [Chat Completions Stream Methods](https://github.com/openai/openai-python/blob/main/helpers.md#chat-completions-stream-methods)

- **Real-Time Interaction and Large Response Handling:**
  ```python
  def stream_completion():
      stream = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": "Write a story"}],
          stream=True
      )
      
      for chunk in stream:
          if chunk.choices[0].delta.content is not None:
              print(chunk.choices[0].delta.content, end="")
  ```

### Batch Processing
**Link:** [Batch Processing](https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing.ipynb)

- **Handling Large Volumes of Data Efficiently:**
  ```python
  prompts = ["Tell me a joke", "What is the capital of France?", "Explain quantum physics"]

  responses = [openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50) for prompt in prompts]

  for response in responses:
      print(response.choices[0].text.strip())
  ```

---

## 5. Building Tool-Using Agents and Advanced Techniques

### Building a Tool-Using Agent with Langchain
**Link:** [Building a Tool-Using Agent with Langchain](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb)

- **Creating Agents with External Tools:**
  ```python
  from langchain import LangChain

  langchain = LangChain(api_key='your-api-key')

  def tool_using_agent(prompt):
      tools = {
          "search": langchain.tools.search,
          "calculator": langchain.tools.calculator
      }
      
      agent = langchain.create_agent(tools)
      response = agent.run(prompt)
      return response
  ```

### Advanced Error Handling and Retry Logic
**Link:** [API Request Parallel Processor](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)

- **Robust Error Handling and Retry Strategies:**
  ```python
  from tenacity import retry, wait_random_exponential, stop_after_attempt

  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
  def completion_with_backoff(**kwargs):
      return client.chat.completions.create(**kwargs)
  ```

These references and examples provide a detailed guide to effectively using the OpenAI API, optimizing performance, and handling complex tasks. Adjust the API keys and parameters as needed for your specific use case.

---
Here are all the links from the merged notes:

1. [OpenAI Python Library - Usage](https://github.com/openai/openai-python?tab=readme-ov-file#usage)
2. [OpenAI Python Library - Advanced](https://github.com/openai/openai-python?tab=readme-ov-file#advanced)
3. [Structured Outputs Parsing Helpers](https://github.com/openai/openai-python/blob/main/helpers.md#structured-outputs-parsing-helpers)
4. [Function Calling with Structured Outputs](https://platform.openai.com/docs/guides/function-calling)
5. [Handling Rate Limits](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb)
6. [Counting Tokens with Tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
7. [Chat Completions Stream Methods](https://github.com/openai/openai-python/blob/main/helpers.md#chat-completions-stream-methods)
8. [Batch Processing](https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing.ipynb)
9. [Building a Tool-Using Agent with Langchain](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb)
10. [API Request Parallel Processor](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)

