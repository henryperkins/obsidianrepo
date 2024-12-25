```python
# pip install azure-ai-inference
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
if not api_key:
  raise Exception("A key should be provided to invoke the endpoint")

client = ChatCompletionsClient(
    endpoint='https://AI21-Jamba-1-5-Large-docs.eastus2.models.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)

model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

payload = {
  "messages": [
    {
      "role": "system",
      "content": "You are a support Engineer"
    },
    {
      "role": "user",
      "content": "I need help with your product. Can you please assist?"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.8,
  "top_p": 0.1
}
response = client.complete(payload)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Usage:")
print("	Prompt tokens:", response.usage.prompt_tokens)
print("	Total tokens:", response.usage.total_tokens)
print("	Completion tokens:", response.usage.completion_tokens)
```

**Endpoint:** https://AI21-Jamba-1-5-Large-docs.eastus2.models.ai.azure.com/v1/chat/completions
**API Key:** IPEOykq7LV2Qi91Gx11BqX8UxE47lNvi
**Name:**  AI21-Jamba-1-5-Large-**docs
**Resource: [azure-sdk-for-python/sdk/ai/azure-ai-inference/samples at main · Azure/azure-sdk-for-python](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-inference/samples)

```python
# pip install azure-ai-inference
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
if not api_key:
  raise Exception("A key should be provided to invoke the endpoint")

client = ChatCompletionsClient(
    endpoint='https://AI21-Jamba-1-5-Large-docs.eastus2.models.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)

model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

payload = {
  "messages": [
    {
      "role": "system",
      "content": "You are a support Engineer"
    },
    {
      "role": "user",
      "content": "I need help with your product. Can you please assist?"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.8,
  "top_p": 0.1
}
response = client.complete(payload)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Usage:")
print("	Prompt tokens:", response.usage.prompt_tokens)
print("	Total tokens:", response.usage.total_tokens)
print("	Completion tokens:", response.usage.completion_tokens)
```

