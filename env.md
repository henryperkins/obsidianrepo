```plaintext
# Required Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=00000000000000000000000000000000
AZURE_OPENAI_DEPLOYMENT=gpt-4-deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Model Settings
MODEL_TYPE=azure
MODEL_NAME=gpt-4
MAX_TOKENS=1000
TEMPERATURE=0.7
REQUEST_TIMEOUT=30

# Rate Limits
MAX_TOKENS_PER_MINUTE=150000
TOKEN_BUFFER=100
BATCH_SIZE=5

# Cache Settings
CACHE_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Logging
LOG_LEVEL=DEBUG
LOG_DIRECTORY=logs

# Output
OUTPUT_DIR=docs

```