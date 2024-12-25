# RAG Implementation Guide: Pinecone + Claude
*A practical guide with examples and references*

## 1. Basic RAG Implementation
```python
import pinecone
from anthropic import Anthropic

# Basic RAG Pipeline
def basic_rag_pipeline(query: str) -> str:
    # 1. Query vector database
    results = index.query(
        vector=embed_query(query),
        top_k=3,
        include_metadata=True
    )
    
    # 2. Format context from results
    context = "\n".join([r.metadata['text'] for r in results.matches])
    
    # 3. Generate response with context
    response = anthropic.messages.create(
        model="claude-3-opus-20240229",
        messages=[{
            "role": "user",
            "content": f"Context: {context}\nQuestion: {query}"
        }]
    )
    return response.content

```
📚 References:
- [RAG Using Pinecone Notebook](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/Pinecone/rag_using_pinecone.ipynb)
- [Contextual Retrieval Guide](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/retrieval_augmented_generation/guide.ipynb)

## 2. Advanced Features

### Streaming Responses
```python
def stream_rag_response(query: str):
    context = get_context_from_pinecone(query)
    stream = anthropic.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": f"{context}\n{query}"}],
        stream=True
    )
    for chunk in stream:
        yield chunk.content
```
🔗 [Anthropic Streaming Documentation](https://docs.anthropic.com/claude/reference/streaming)

### Prompt Caching
```python
from anthropic import Cache

cache = Cache()
@cache.cached(ttl=3600)
def cached_rag_query(query: str):
    # Your RAG implementation
    pass
```
🔗 [Prompt Caching Guide](https://docs.anthropic.com/claude/docs/prompt-caching)

## 3. Index Management

### Creating and Managing Indexes
```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY")

# Create index
pinecone.create_index(
    name="research-docs",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine"
)

# Upsert data
index = pinecone.Index("research-docs")
index.upsert(vectors=[
    (id1, vector1, {"text": text1}),
    (id2, vector2, {"text": text2})
])
```
🔗 [Pinecone Index Management](https://docs.pinecone.io/docs/manage-indexes)

## 4. Error Handling & Best Practices

### Robust Error Handling
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_rag_query(query: str):
    try:
        results = index.query(...)
        response = anthropic.messages.create(...)
        return response
    except pinecone.exceptions.PineconeException as e:
        logger.error(f"Pinecone error: {e}")
        raise
    except anthropic.APIError as e:
        logger.error(f"Claude error: {e}")
        raise
```
🔗 [Error Handling Documentation](https://github.com/anthropics/anthropic-sdk-python/blob/main/docs/error_handling.md)

## 5. Performance Optimization

### Batch Processing
```python
def batch_process_documents(documents: List[Dict]):
    # Batch embed documents
    embeddings = []
    for batch in chunks(documents, size=100):
        batch_embeddings = embed_batch(batch)
        embeddings.extend(batch_embeddings)
    
    # Batch upsert to Pinecone
    index.upsert(vectors=zip(ids, embeddings, metadata))
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_rag_queries(queries: List[str]):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(rag_query, queries))
    return results
```

## Additional Resources

### Official Documentation
- [Pinecone Python Client](https://github.com/pinecone-io/pinecone-python-client)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Claude API Reference](https://docs.anthropic.com/claude/reference)

### Example Notebooks
- [Summarization Guide](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/summarization/guide.ipynb)
- [Contextual Embeddings](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb)

### Best Practices
- Use appropriate batch sizes (50-100 documents)
- Implement retry logic for API calls
- Cache frequently accessed data
- Monitor performance metrics
- Regular index maintenance

---
## 6. Context Window Management

### Smart Chunking Strategy
```python
from typing import List, Dict
import tiktoken

def smart_chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """Intelligently chunk text while preserving context"""
    
    encoder = tiktoken.encoding_for_model("claude-3-opus-20240229")
    
    def token_count(text: str) -> int:
        return len(encoder.encode(text))
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para_length = token_count(para)
        
        if current_length + para_length > max_tokens:
            # Save current chunk and start new one
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
        
    return chunks
```
🔗 [Claude Context Window Guide](https://docs.anthropic.com/claude/docs/context-window)

## 7. Advanced Retrieval Strategies

### Hybrid Search Implementation
```python
def hybrid_search(query: str, 
                 alpha: float = 0.5) -> List[Dict]:
    """Combine vector and keyword search results"""
    
    # Vector search
    vector_results = index.query(
        vector=embed_query(query),
        top_k=5,
        include_metadata=True
    )
    
    # Keyword search (using metadata filter)
    keyword_results = index.query(
        vector=[0] * 1536,  # placeholder vector
        top_k=5,
        include_metadata=True,
        filter={
            "text": {"$contains": query}
        }
    )
    
    # Combine and re-rank results
    combined_results = merge_and_rerank(
        vector_results=vector_results,
        keyword_results=keyword_results,
        alpha=alpha
    )
    
    return combined_results
```
🔗 [Pinecone Hybrid Search](https://docs.pinecone.io/docs/hybrid-search)

### Semantic Router
```python
from semantic_router import Route, Router
from semantic_router.encoders import CohereEncoder

def create_rag_router():
    """Route queries to appropriate RAG strategies"""
    
    router = Router(
        encoder=CohereEncoder(),
        routes=[
            Route(
                name="technical",
                utterances=[
                    "How do I implement...",
                    "What's the syntax for...",
                    "Debug this error..."
                ]
            ),
            Route(
                name="conceptual",
                utterances=[
                    "Explain the concept...",
                    "What is the difference...",
                    "Why should I use..."
                ]
            )
        ]
    )
    return router

def smart_rag_query(query: str):
    router = create_rag_router()
    route = router(query)
    
    if route.name == "technical":
        return technical_rag_pipeline(query)
    else:
        return conceptual_rag_pipeline(query)
```

## 8. Advanced Context Processing

### Dynamic Context Selection
```python
def select_relevant_context(query: str, 
                          results: List[Dict], 
                          max_tokens: int = 4000) -> str:
    """Dynamically select and order context based on relevance"""
    
    # Score results based on multiple factors
    scored_results = []
    for r in results:
        score = calculate_relevance_score(
            query=query,
            result=r,
            factors={
                'vector_score': 0.4,
                'recency': 0.2,
                'token_overlap': 0.2,
                'metadata_match': 0.2
            }
        )
        scored_results.append((score, r))
    
    # Sort and select within token budget
    scored_results.sort(reverse=True)
    selected_context = []
    current_tokens = 0
    
    for score, result in scored_results:
        tokens = count_tokens(result['text'])
        if current_tokens + tokens <= max_tokens:
            selected_context.append(result['text'])
            current_tokens += tokens
            
    return "\n\n".join(selected_context)
```

## 9. Monitoring and Analytics

### RAG Pipeline Metrics
```python
from dataclasses import dataclass
from datetime import datetime
import prometheus_client as prom

@dataclass
class RAGMetrics:
    query_latency = prom.Histogram(
        'rag_query_latency_seconds',
        'Time spent processing RAG query',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    context_size = prom.Gauge(
        'rag_context_size_tokens',
        'Size of retrieved context in tokens'
    )
    retrieval_count = prom.Counter(
        'rag_retrieval_total',
        'Total number of retrieval operations'
    )

def monitored_rag_query(query: str):
    metrics = RAGMetrics()
    start_time = datetime.now()
    
    try:
        results = retrieve_context(query)
        metrics.retrieval_count.inc()
        
        context = select_relevant_context(query, results)
        metrics.context_size.set(count_tokens(context))
        
        response = generate_response(query, context)
        
        query_time = (datetime.now() - start_time).total_seconds()
        metrics.query_latency.observe(query_time)
        
        return response
        
    except Exception as e:
        # Log error metrics
        raise
```
🔗 [Prometheus Python Client](https://github.com/prometheus/client_python)

## 10. Testing and Evaluation

### RAG Evaluation Framework
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy
)

def evaluate_rag_system(
    test_questions: List[str],
    ground_truth: List[str]
):
    """Evaluate RAG system performance"""
    
    results = []
    for question, truth in zip(test_questions, ground_truth):
        # Get RAG response
        context, answer = rag_query(question)
        
        # Evaluate using multiple metrics
        scores = evaluate(
            question=question,
            ground_truth=truth,
            context=context,
            answer=answer,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_relevancy
            ]
        )
        
        results.append(scores)
    
    return aggregate_scores(results)
```
🔗 [Ragas Evaluation Framework](https://github.com/explodinggradients/ragas)

## Additional Resources

### Advanced Topics
- [Claude Contextual Embeddings](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb)
- [Pinecone Index Optimization](https://docs.pinecone.io/docs/optimize-indexes)
- [RAG Best Practices](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation-rag)

### Monitoring Tools
- [Weights & Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)
- [Prometheus](https://prometheus.io/)

### Testing Frameworks
- [Ragas](https://github.com/explodinggradients/ragas)
- [LangChain Testing](https://python.langchain.com/docs/guides/testing)

This concludes our advanced guide to RAG implementation