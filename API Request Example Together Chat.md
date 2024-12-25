```
curl -X POST "https://api.together.xyz/v1/chat/completions" \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "messages": [
        {
                "role": "user",
                "content": "Test"
        },
        {
                "role": "assistant",
                "content": "Hello! How can I assist you today? If you have any questions or need information, feel free to let me know."
        }
],
    "max_tokens": 8192,
    "temperature": 0.3,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1,
    "stop": ["<|im_end|>"],
    "stream": true
  }'
```