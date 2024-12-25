```python
# metrics.py

from typing import Dict, Any, Optional, List, DefaultDict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
from core.logger import LoggerSetup

@dataclass
class OperationMetrics:
    """Metrics for an operation."""
    total_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ServiceMetrics:
    """Metrics for a service."""
    operations: DefaultDict[str, OperationMetrics] = field(
        default_factory=lambda: defaultdict(OperationMetrics)
    )
    total_errors: int = 0
    total_operations: int = 0

class MetricsCollector:
    """Enhanced metrics collection and analysis."""
    
    def __init__(self, retention_days: int = 7):
        self.logger = LoggerSetup.get_logger("metrics_collector")
        self.retention_days = retention_days
        self._service_metrics: DefaultDict[str, ServiceMetrics] = defaultdict(ServiceMetrics)
        self._operation_history: List[Dict[str, Any]] = []
        
    async def record_operation(
        self,
        service: str,
        operation: str,
        duration: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record operation metrics with detailed tracking.
        
        Args:
            service: Service name
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
            metadata: Optional operation metadata
            error: Optional error message
        """
        try:
            timestamp = datetime.now()
            
            # Update service metrics
            service_metric = self._service_metrics[service]
            operation_metric = service_metric.operations[operation]
            
            # Update operation counts
            operation_metric.total_count += 1
            if success:
                operation_metric.success_count += 1
            else:
                operation_metric.error_count += 1
                service_metric.total_errors += 1
            
            # Update duration metrics
            operation_metric.total_duration += duration
            operation_metric.min_duration = min(operation_metric.min_duration, duration)
            operation_metric.max_duration = max(operation_metric.max_duration, duration)
            operation_metric.last_timestamp = timestamp
            
            # Record detailed operation history
            self._operation_history.append({
                "service": service,
                "operation": operation,
                "duration": duration,
                "success": success,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {},
                "error": error
            })
            
            # Cleanup old history
            await self._cleanup_old_metrics()
            
            self.logger.debug(
                f"Recorded operation metrics for {service}.{operation}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record operation metrics: {str(e)}")
    
    async def get_service_summary(
        self,
        service: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get summary metrics for a service.
        
        Args:
            service: Service name
            time_window: Optional time window for metrics
            
        Returns:
            Dict[str, Any]: Service metrics summary
        """
        try:
            service_metric = self._service_metrics[service]
            
            # Filter operations by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                operations = {
                    op: metrics
                    for op, metrics in service_metric.operations.items()
                    if metrics.last_timestamp >= cutoff_time
                }
            else:
                operations = service_metric.operations
            
            # Calculate summary metrics
            total_operations = sum(m.total_count for m in operations.values())
            total_errors = sum(m.error_count for m in operations.values())
            total_duration = sum(m.total_duration for m in operations.values())
            
            return {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_operations if total_operations > 0 else 0),
                "average_duration": total_duration / total_operations if total_operations > 0 else 0,
                "operations": {
                    op: {
                        "total_count": metrics.total_count,
                        "success_count": metrics.success_count,
                        "error_count": metrics.error_count,
                        "average_duration": (
                            metrics.total_duration / metrics.total_count 
                            if metrics.total_count > 0 else 0
                        ),
                        "min_duration": metrics.min_duration,
                        "max_duration": metrics.max_duration
                    }
                    for op, metrics in operations.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get service summary: {str(e)}")
            return {}
    
    async def get_error_analysis(
        self,
        service: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get detailed error analysis.
        
        Args:
            service: Optional service filter
            time_window: Optional time window
            
        Returns:
            Dict[str, Any]: Error analysis report
        """
        try:
            # Filter operations by criteria
            operations = self._filter_operations(service, time_window)
            
            # Group errors by type
            error_types: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
            for op in operations:
                if not op["success"] and op["error"]:
                    error_types[op["error"]].append(op)
            
            return {
                "total_errors": len([op for op in operations if not op["success"]]),
                "error_types": {
                    error_type: {
                        "count": len(occurrences),
                        "latest_occurrence": max(
                            occ["timestamp"] for occ in occurrences
                        ),
                        "affected_operations": list(set(
                            f"{occ['service']}.{occ['operation']}"
                            for occ in occurrences
                        ))
                    }
                    for error_type, occurrences in error_types.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get error analysis: {str(e)}")
            return {}
    
    async def get_performance_analysis(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get detailed performance analysis.
        
        Args:
            service: Optional service filter
            operation: Optional operation filter
            time_window: Optional time window
            
        Returns:
            Dict[str, Any]: Performance analysis report
        """
        try:
            # Filter operations by criteria
            operations = self._filter_operations(service, time_window)
            if operation:
                operations = [op for op in operations if op["operation"] == operation]
            
            if not operations:
                return {}
            
            # Calculate performance metrics
            durations = [op["duration"] for op in operations]
            durations.sort()
            
            return {
                "total_operations": len(operations),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "median_duration": durations[len(durations) // 2],
                "percentiles": {
                    "p50": durations[int(len(durations) * 0.5)],
                    "p90": durations[int(len(durations) * 0.9)],
                    "p95": durations[int(len(durations) * 0.95)],
                    "p99": durations[int(len(durations) * 0.99)]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance analysis: {str(e)}")
            return {}
    
    def _filter_operations(
        self,
        service: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Filter operations by criteria."""
        operations = self._operation_history
        
        if service:
            operations = [op for op in operations if op["service"] == service]
            
        if time_window:
            cutoff_time = datetime.now() - time_window
            operations = [
                op for op in operations
                if datetime.fromisoformat(op["timestamp"]) >= cutoff_time
            ]
            
        return operations
    
    async def _cleanup_old_metrics(self) -> None:
        """Cleanup metrics older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            self._operation_history = [
                op for op in self._operation_history
                if datetime.fromisoformat(op["timestamp"]) >= cutoff_time
            ]
            
            self.logger.debug("Cleaned up old metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup metrics: {str(e)}")
            
    async def export_metrics(self, filepath: str) -> None:
        """Export metrics to file."""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "retention_days": self.retention_days,
                "service_metrics": {
                    service: {
                        "total_operations": metrics.total_operations,
                        "total_errors": metrics.total_errors,
                        "operations": {
                            op: {
                                "total_count": op_metrics.total_count,
                                "success_count": op_metrics.success_count,
                                "error_count": op_metrics.error_count,
                                "average_duration": (
                                    op_metrics.total_duration / op_metrics.total_count
                                    if op_metrics.total_count > 0 else 0
                                )
                            }
                            for op, op_metrics in metrics.operations.items()
                        }
                    }
                    for service, metrics in self._service_metrics.items()
                },
                "operation_history": self._operation_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.logger.info(f"Exported metrics to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            raise
```