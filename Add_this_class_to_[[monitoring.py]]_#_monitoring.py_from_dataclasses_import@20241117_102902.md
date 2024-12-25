---
epoch: 1731860942043
modelKey: claude-3-5-sonnet-20241022|anthropic
tags:
  - copilot-conversation
---

# Add_this_class_to_[monitoring.py](monitoring.py.md)_#_monitoring.py_from_dataclasses_import@20241117_102902

**user**: Add this class to [monitoring.py](simple2/monitoring.py.md) # monitoring.py  
from dataclasses import dataclass  
from typing import Dict, Any

@dataclass  
class ModelMetrics:  
    """Metrics structure for model operations."""  
    model_type: str  
    operation: str  
    tokens_used: int  
    response_time: float  
    status: str  
    cost: float  
    error: Optional[str] = None

class SystemMonitor:  
    """Unified monitoring for all AI models."""
    
    def __init__(self):
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }
        
    def log_model_metrics(
        self,
        model_type: str,
        metrics: ModelMetrics
    ):
        """Log metrics for specific model."""
        if model_type not in self.metrics_by_model:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.metrics_by_model[model_type][metrics.operation] = metrics
[Timestamp: 2024/11/17 10:29:02]

**ai**: I'll help you integrate this class into the existing monitoring.py file. I'll modify it to work with the existing SystemMonitor class while maintaining the current functionality. Here's how we can add it:

```python
@dataclass
class ModelMetrics:
    """Metrics structure for model operations."""
    model_type: str
    operation: str
    tokens_used: int
    response_time: float
    status: str
    cost: float
    error: Optional[str] = None

class SystemMonitor:
    """
    System monitoring and metrics tracking.

    This class provides methods to log various system events, including API requests,
    cache operations, docstring changes, and model-specific metrics across different
    AI providers.
    """

    def __init__(self):
        """Initialize the monitoring system with default metrics and start time."""
        self.requests = []
        self.metrics = {}
        self.api_metrics = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.docstring_changes = {
            'added': [], 'updated': [], 
            'retained': [], 'failed': []
        }
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0.0,
            'processed': 0,
            'failed': 0
        }
        # Add model-specific metrics tracking
        self.metrics_by_model = {
            "azure": {},
            "openai": {},
            "gemini": {},
            "claude": {}
        }
        self.start_time = time.time()

    def log_model_metrics(
        self,
        model_type: str,
        metrics: ModelMetrics
    ) -> None:
        """
        Log metrics for specific model.

        Args:
            model_type (str): Type of the AI model (azure/openai/gemini/claude)
            metrics (ModelMetrics): Metrics data for the model operation

        Raises:
            ValueError: If the model type is not supported
        """
        log_debug(f"Logging metrics for model type: {model_type}")
        if model_type not in self.metrics_by_model:
            error_msg = f"Unsupported model type: {model_type}"
            log_error(error_msg)
            raise ValueError(error_msg)
            
        self.metrics_by_model[model_type][metrics.operation] = metrics
        log_info(f"Model metrics logged: {model_type} - Operation: {metrics.operation}")

    # ... (rest of the existing methods remain unchanged)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics summary.

        Returns:
            Dict[str, Any]: Complete metrics summary including runtime, API metrics,
                           cache metrics, docstring changes, and model-specific metrics.
        """
        log_debug("Generating metrics summary")
        summary = super().get_metrics_summary()
        
        # Add model-specific metrics to the summary
        model_metrics_summary = {}
        for model_type, operations in self.metrics_by_model.items():
            model_metrics_summary[model_type] = {
                'total_operations': len(operations),
                'total_tokens': sum(m.tokens_used for m in operations.values()),
                'total_cost': sum(m.cost for m in operations.values()),
                'average_response_time': sum(m.response_time for m in operations.values()) / max(len(operations), 1),
                'error_count': len([m for m in operations.values() if m.error])
            }
        
        summary['model_metrics'] = model_metrics_summary
        log_info(f"Metrics summary generated with model metrics: {summary}")
        return summary
```

This integration:

1. Adds the new `ModelMetrics` dataclass for tracking model-specific metrics
2. Extends the `SystemMonitor` class initialization to include model-specific metrics tracking
3. Adds the `log_model_metrics` method to log metrics for specific AI models
4. Updates the `get_metrics_summary` method to include model-specific metrics in the summary

The changes maintain compatibility with existing functionality while adding the new model-specific monitoring capabilities. The metrics are organized by model type (azure, openai, gemini, claude) and include tracking of operations, tokens, costs, response times, and errors for each model.

To use the new functionality:

```python
# Example usage
monitor = SystemMonitor()

# Log metrics for a specific model
metrics = ModelMetrics(
    model_type="openai",
    operation="text-completion",
    tokens_used=150,
    response_time=0.8,
    status="success",
    cost=0.002
)

monitor.log_model_metrics("openai", metrics)

# Get complete metrics summary including model-specific metrics
summary = monitor.get_metrics_summary()
```

## Sources

- [monitoring.py](obsidian://open?vault=My%20Remote%20Vault&file=monitoring.py)
- [System Monitoring and Metrics Tracking Module for Python](obsidian://open?vault=My%20Remote%20Vault&file=System%20Monitoring%20and%20Metrics%20Tracking%20Module%20for%20Python)  
[Timestamp: 2024/11/17 10:29:32]
