```python
# monitoring.py

import sentry_sdk
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict
from core.logger import LoggerSetup

class MonitoringError(Exception):
    """Base exception for monitoring operations."""
    pass

class MetricsError(MonitoringError):
    """Exception for metrics recording failures."""
    pass

class RAGMonitor:
    """Enhanced monitoring system with comprehensive error tracking."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("monitoring")
        self._initialize_sentry()
        self._error_history: List[Dict[str, Any]] = []
        self._operation_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._last_flush = datetime.now()
    
    def _initialize_sentry(self) -> None:
        """Initialize Sentry with enhanced configuration."""
        try:
            sentry_dsn = os.getenv("SENTRY_DSN")
            if not sentry_dsn:
                raise MonitoringError("SENTRY_DSN environment variable not set")
            
            sentry_sdk.init(
                dsn=sentry_dsn,
                traces_sample_rate=1.0,
                # Enhanced error capture configuration
                send_default_pii=False,
                max_breadcrumbs=50,
                debug=False,
                # Performance monitoring
                enable_tracing=True,
                profiles_sample_rate=1.0
            )
            
            self.logger.info("Sentry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sentry: {str(e)}")
            raise MonitoringError(f"Sentry initialization failed: {str(e)}")
    
    async def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> None:
        """
        Record detailed error information with context.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional error context
            stack_trace: Optional stack trace
        """
        try:
            timestamp = datetime.now()
            
            # Prepare error record
            error_record = {
                "type": error_type,
                "message": error_message,
                "timestamp": timestamp.isoformat(),
                "context": context or {},
                "stack_trace": stack_trace or traceback.format_exc()
            }
            
            # Record in local history
            self._error_history.append(error_record)
            self._error_counts[error_type] += 1
            
            # Send to Sentry with enhanced context
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("error_type", error_type)
                if context:
                    scope.set_context("error_context", context)
                sentry_sdk.capture_exception(
                    error_message,
                    scope=scope
                )
            
            self.logger.error(
                f"Error recorded: {error_type}",
                extra={"error_record": error_record}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record error: {str(e)}")
            raise MetricsError(f"Error recording failed: {str(e)}")
    
    async def record_operation(
        self,
        operation_type: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "success"
    ) -> None:
        """
        Record operation metrics with enhanced tracking.
        
        Args:
            operation_type: Type of operation
            duration: Operation duration in seconds
            metadata: Optional operation metadata
            status: Operation status
        """
        try:
            timestamp = datetime.now()
            
            # Update operation metrics
            metrics = self._operation_metrics[operation_type]
            metrics["total_count"] += 1
            metrics["total_duration"] += duration
            
            if status == "success":
                metrics["success_count"] += 1
            else:
                metrics["error_count"] += 1
            
            # Record detailed operation
            operation_record = {
                "type": operation_type,
                "duration": duration,
                "timestamp": timestamp.isoformat(),
                "status": status,
                "metadata": metadata or {}
            }
            
            # Send metrics to Sentry
            with sentry_sdk.start_transaction(
                op=operation_type,
                name=f"{operation_type}_operation"
            ) as transaction:
                transaction.set_measurement("duration", duration)
                if metadata:
                    transaction.set_context("metadata", metadata)
            
            self.logger.info(
                f"Operation recorded: {operation_type}",
                extra={"operation_record": operation_record}
            )
            
            # Check if metrics should be flushed
            await self._check_flush_metrics()
            
        except Exception as e:
            self.logger.error(f"Failed to record operation: {str(e)}")
            raise MetricsError(f"Operation recording failed: {str(e)}")
    
    async def get_error_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get summary of error metrics.
        
        Args:
            time_window: Optional time window for filtering
            
        Returns:
            Dict[str, Any]: Error metrics summary
        """
        try:
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_errors = [
                    error for error in self._error_history
                    if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
                ]
            else:
                filtered_errors = self._error_history
            
            # Calculate error metrics
            error_types = defaultdict(int)
            error_timeline = defaultdict(int)
            
            for error in filtered_errors:
                error_types[error["type"]] += 1
                timestamp = datetime.fromisoformat(error["timestamp"])
                date_key = timestamp.strftime("%Y-%m-%d")
                error_timeline[date_key] += 1
            
            return {
                "total_errors": len(filtered_errors),
                "error_types": dict(error_types),
                "error_timeline": dict(error_timeline),
                "most_common_error": max(
                    error_types.items(),
                    key=lambda x: x[1],
                    default=("none", 0)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get error summary: {str(e)}")
            return {}
    
    async def get_operation_metrics(
        self,
        operation_type: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get detailed operation metrics.
        
        Args:
            operation_type: Optional operation type filter
            time_window: Optional time window for filtering
            
        Returns:
            Dict[str, Any]: Operation metrics
        """
        try:
            if operation_type:
                metrics = self._operation_metrics.get(operation_type, {})
                if not metrics:
                    return {}
                
                return {
                    "total_operations": metrics["total_count"],
                    "success_rate": (
                        metrics["success_count"] / metrics["total_count"]
                        if metrics["total_count"] > 0 else 0
                    ),
                    "average_duration": (
                        metrics["total_duration"] / metrics["total_count"]
                        if metrics["total_count"] > 0 else 0
                    )
                }
            else:
                return {
                    op_type: {
                        "total_operations": metrics["total_count"],
                        "success_rate": (
                            metrics["success_count"] / metrics["total_count"]
                            if metrics["total_count"] > 0 else 0
                        ),
                        "average_duration": (
                            metrics["total_duration"] / metrics["total_count"]
                            if metrics["total_count"] > 0 else 0
                        )
                    }
                    for op_type, metrics in self._operation_metrics.items()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get operation metrics: {str(e)}")
            return {}
    
    async def _check_flush_metrics(self) -> None:
        """Check and flush metrics if needed."""
        try:
            now = datetime.now()
            if (now - self._last_flush) >= timedelta(minutes=5):
                await self.flush_metrics()
                self._last_flush = now
                
        except Exception as e:
            self.logger.error(f"Failed to check metrics flush: {str(e)}")
    
    async def flush_metrics(self) -> None:
        """Flush metrics to persistent storage."""
        try:
            # Prepare metrics data
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "error_metrics": {
                    "total_errors": len(self._error_history),
                    "error_counts": dict(self._error_counts)
                },
                "operation_metrics": dict(self._operation_metrics)
            }
            
            # Store metrics
            metrics_file = os.path.join(
                os.getenv("METRICS_DIR", "metrics"),
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Metrics flushed to {metrics_file}")
            
            # Clear in-memory metrics
            self._error_history.clear()
            self._error_counts.clear()
            self._operation_metrics.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {str(e)}")
            raise MetricsError(f"Metrics flush failed: {str(e)}")
```