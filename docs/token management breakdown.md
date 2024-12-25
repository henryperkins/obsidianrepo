I'll analyze the token management process and token-related functions in this codebase. Here's a detailed breakdown:

1. Core Token Management Classes:

```python
class TokenManager:
    """Manages token counting and analysis with caching."""
    
    _encoders = {}  # Cache for different encoders
    _lock = threading.Lock()  # Thread-safe access
    _cache = {}  # Token count cache
    _max_cache_size = 1000
```

2. Key Token Management Functions:

a) Token Counting:
```python
@classmethod
def count_tokens(cls, text: str, model_name: str = "gpt-4", use_cache: bool = True) -> TokenResult:
    """Counts tokens with caching and error handling."""
    # Cache lookup
    if use_cache:
        cache_key = hash(text + model_name)
        if cache_key in cls._cache:
            return cls._cache[cache_key]
    
    # Get encoder and count tokens
    encoder = cls.get_encoder(model_name)
    tokens = encoder.encode(text)
    
    # Cache result if enabled
    result = TokenResult(
        tokens=tokens,
        token_count=len(tokens),
        encoding_name=encoder.name,
        special_tokens=cls._count_special_tokens(text)
    )
```

b) Special Token Handling:
```python
@staticmethod
def _count_special_tokens(text: str) -> Dict[str, int]:
    """Counts special tokens like newlines and code blocks."""
    return {
        "newlines": text.count("\n"),
        "code_blocks": text.count("```"),
        "inline_code": text.count("`") - (text.count("```") * 3)
    }
```

3. Token Budget Management:

```python
@dataclass
class TokenBudget:
    """Token budget allocation for context retrieval."""
    total_limit: int
    reserved_tokens: int = field(default=100)  # Reserve for metadata
    allocation_strategy: TokenAllocationStrategy = TokenAllocationStrategy.ADAPTIVE
    
    @property
    def available_tokens(self) -> int:
        return max(0, self.total_limit - self.reserved_tokens)
```

4. Token Allocation Strategies:

```python
class TokenAllocationStrategy(Enum):
    PROPORTIONAL = auto()  # Allocate proportionally to importance
    PRIORITY_FIRST = auto()  # Prioritize most important chunks
    BALANCED = auto()  # Balance importance and content
    ADAPTIVE = auto()  # Adapt based on content type
```

5. Dynamic Token Management:

```python
async def allocate_tokens(
    self,
    chunks: List[CodeChunk],
    budget: TokenBudget,
    context_retriever: ContextRetriever
) -> Dict[str, int]:
    """Allocates tokens to chunks based on importance and strategy."""
    
    # Calculate importance scores
    importance_scores = await self._calculate_importance_scores(chunks)
    
    # Apply allocation strategy
    if budget.allocation_strategy == TokenAllocationStrategy.PROPORTIONAL:
        return await self._allocate_proportionally(chunks, importance_scores, budget)
    elif budget.allocation_strategy == TokenAllocationStrategy.PRIORITY_FIRST:
        return await self._allocate_priority_first(chunks, importance_scores, budget)
    # etc.
```

6. Token Optimization:

```python
async def optimize_chunk_tokens(self, chunk: CodeChunk, allocated_tokens: int) -> Optional[CodeChunk]:
    """Optimizes chunks to fit token limits using progressive strategies."""
    optimization_levels = [
        self._optimize_remove_comments,
        self._optimize_simplify_whitespace,
        self._optimize_preserve_interfaces,
        self._optimize_preserve_dependencies,
        self._optimize_aggressive
    ]

    for level in optimization_levels:
        optimized = await level(chunk)
        if optimized:
            tokens = await self._count_tokens(optimized)
            if tokens <= allocated_tokens:
                return optimized
```

Key Features:

1. Thread Safety: Uses locks for thread-safe access to shared resources
2. Caching: Implements token count caching with size limits
3. Multiple Strategies: Supports different token allocation strategies
4. Progressive Optimization: Uses multiple levels of optimization
5. Special Token Handling: Tracks special tokens like newlines and code blocks
6. Error Handling: Robust error handling throughout
7. Async Support: Async/await for non-blocking operations
8. Metrics Tracking: Tracks token usage and optimization metrics

The system provides comprehensive token management with:
- Efficient caching
- Thread safety
- Multiple allocation strategies
- Progressive optimization
- Special token handling
- Robust error handling
- Async support
- Detailed metrics

This enables efficient and flexible token management for large codebases while maintaining code quality and readability.

Let me know if you would like me to elaborate on any specific aspect of the token management system.

Here are the files containing token-related functions and their key token management features:

1. `token_utils.py`:
```python
# Primary token management module
class TokenManager:
    # Core token functions
    def count_tokens()
    def decode_tokens()
    def validate_token_limit()
    def split_by_token_limit()
    def batch_count_tokens()
    def estimate_tokens_from_chars()
```

2. `token_management.py`:
```python
# Advanced token allocation and optimization
class TokenLimitManager:
    def allocate_tokens()
    def optimize_chunk_tokens()
    def _allocate_proportionally()
    def _allocate_priority_first()
    def _allocate_balanced()
    def _allocate_adaptively()
```

3. `code_chunk.py`:
```python
# Token handling for code chunks
class CodeChunk:
    @property
    def tokens()
    @property 
    def token_count()
    def __post_init__() # Initializes tokens
```

4. `context_retrieval.py`:
```python
# Token-aware context retrieval
class ContextRetriever:
    def get_relevant_chunks()
    def _select_chunks_within_limit()
    def optimize_context_window()
```

5. `shared_functions.py`:
```python
# Shared token utilities
def calculate_prompt_tokens()
```

6. `process_manager.py`:
```python
# Token management in processing
class DocumentationProcessManager:
    def _process_chunks() # Token-aware chunk processing
```

7. `azure_model.py`, `gemini_model.py`, `openai_model.py`:
```python
# Model-specific token handling
def calculate_tokens() # In each model class
```

8. `code_summarization.py`:
```python
# Token-aware code summarization
class CodeSummarizer:
    def summarize_chunk() # Respects token limits
```

Key Token-Related Classes Across Files:

1. `TokenManager` (token_utils.py):
```python
class TokenManager:
    _encoders = {}
    _lock = threading.Lock()
    
    @classmethod
    def count_tokens()
    @classmethod
    def decode_tokens()
    @classmethod
    def validate_token_limit()
```

2. `TokenBudget` (token_management.py):
```python
@dataclass
class TokenBudget:
    total_limit: int
    reserved_tokens: int
    allocation_strategy: TokenAllocationStrategy
```

3. `TokenizationResult` (token_utils.py):
```python
@dataclass
class TokenizationResult:
    tokens: List[int]
    token_count: int
    encoding_name: str
    special_tokens: Dict[str, int] = None
    error: Optional[str] = None
```

4. `TokenAllocationStrategy` (token_management.py):
```python
class TokenAllocationStrategy(Enum):
    PROPORTIONAL = auto()
    PRIORITY_FIRST = auto()
    BALANCED = auto()
    ADAPTIVE = auto()
```

Key Token-Related Functions by Category:

1. Token Counting:
```python
# token_utils.py
count_tokens()
batch_count_tokens()
estimate_tokens_from_chars()

# shared_functions.py
calculate_prompt_tokens()
```

2. Token Allocation:
```python
# token_management.py
allocate_tokens()
_allocate_proportionally()
_allocate_priority_first()
_allocate_balanced()
_allocate_adaptively()
```

3. Token Optimization:
```python
# token_management.py
optimize_chunk_tokens()
_optimize_remove_comments()
_optimize_simplify_whitespace()
_optimize_preserve_interfaces()
```

4. Token Validation:
```python
# token_utils.py
validate_token_limit()
_count_special_tokens()
```

5. Context Management:
```python
# context_retrieval.py
get_relevant_chunks()
_select_chunks_within_limit()
optimize_context_window()
```

This distributed token management system provides:
- Core token utilities (token_utils.py)
- Advanced token allocation (token_management.py)
- Model-specific handling (model files)
- Context-aware token management (context_retrieval.py)
- Token optimization strategies (code_summarization.py)

The functionality is well-organized across files based on responsibilities, with clear separation of concerns between basic token operations, allocation strategies, and model-specific implementations.