```python
# batch_processor.py

from typing import List, Dict, Any, Optional, TypeVar, Generic
from datetime import datetime
import asyncio
from core.logger import LoggerSetup
import time
from dataclasses import dataclass

T = TypeVar('T')  # Type variable for batch items
R = TypeVar('R')  # Type variable for results

@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    total_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    items_per_second: float
    errors: List[Dict[str, Any]]

class BatchProcessingError(Exception):
    """Exception for batch processing failures."""
    def __init__(self, message: str, batch_index: int, failed_items: List[Any]):
        self.batch_index = batch_index
        self.failed_items = failed_items
        super().__init__(message)

class BatchProcessor(Generic[T, R]):
    """Enhanced batch processor with performance optimization."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 5,
        retry_attempts: int = 3
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.retry_attempts = retry_attempts
        self.logger = LoggerSetup.get_logger("batch_processor")
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self.metrics = MetricsCollector()
    
    async def process_batch(
        self,
        items: List[T],
        processor: callable,
        error_handler: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process items in optimized batches with monitoring.
        
        Args:
            items: List of items to process
            processor: Async function to process each batch
            error_handler: Optional error handling function
            
        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        start_time = time.perf_counter()
        total_items = len(items)
        processed_items = 0
        errors = []
        results = []

        try:
            # Split items into batches
            batches = [
                items[i:i + self.batch_size]
                for i in range(0, len(items), self.batch_size)
            ]
            
            # Process batches with concurrency control
            batch_tasks = []
            for batch_index, batch in enumerate(batches):
                task = asyncio.create_task(
                    self._process_single_batch(
                        batch,
                        batch_index,
                        processor,
                        error_handler
                    )
                )
                batch_tasks.append(task)
                
                # Wait if we've reached max concurrent batches
                if len(batch_tasks) >= self.max_concurrent_batches:
                    completed_tasks = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    results.extend(self._handle_batch_results(completed_tasks, errors))
                    batch_tasks = []
                    
            # Process any remaining batches
            if batch_tasks:
                completed_tasks = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(self._handle_batch_results(completed_tasks, errors))
            
            # Calculate metrics
            processing_time = time.perf_counter() - start_time
            successful_items = len([r for r in results if r is not None])
            
            metrics = BatchMetrics(
                total_items=total_items,
                successful_items=successful_items,
                failed_items=total_items - successful_items,
                processing_time=processing_time,
                items_per_second=total_items / processing_time,
                errors=errors
            )
            
            # Record processing metrics
            await self.metrics.record_batch_operation(
                total_items=total_items,
                successful_items=successful_items,
                processing_time=processing_time,
                errors=errors
            )
            
            return {
                "status": "completed",
                "results": results,
                "metrics": metrics.__dict__
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    async def _process_single_batch(
        self,
        batch: List[T],
        batch_index: int,
        processor: callable,
        error_handler: Optional[callable]
    ) -> List[Optional[R]]:
        """
        Process a single batch with retries and error handling.
        
        Args:
            batch: List of items in the batch
            batch_index: Index of the batch
            processor: Processing function
            error_handler: Optional error handler
            
        Returns:
            List[Optional[R]]: Batch processing results
        """
        async with self._processing_semaphore:
            batch_start = time.perf_counter()
            retries = 0
            last_error = None
            
            while retries < self.retry_attempts:
                try:
                    self.logger.debug(
                        f"Processing batch {batch_index} "
                        f"(attempt {retries + 1}/{self.retry_attempts})"
                    )
                    
                    # Process batch
                    results = await processor(batch)
                    
                    # Record successful batch
                    batch_time = time.perf_counter() - batch_start
                    await self.metrics.record_batch_metrics(
                        batch_size=len(batch),
                        processing_time=batch_time,
                        success=True
                    )
                    
                    return results
                    
                except Exception as e:
                    last_error = e
                    retries += 1
                    
                    if error_handler:
                        try:
                            await error_handler(batch, e)
                        except Exception as handler_error:
                            self.logger.error(
                                f"Error handler failed: {str(handler_error)}"
                            )
                    
                    if retries < self.retry_attempts:
                        wait_time = (2 ** retries) + random.uniform(0, 1)
                        self.logger.warning(
                            f"Batch {batch_index} failed, "
                            f"retrying in {wait_time:.2f}s"
                        )
                        await asyncio.sleep(wait_time)
                    
            # Record failed batch
            batch_time = time.perf_counter() - batch_start
            await self.metrics.record_batch_metrics(
                batch_size=len(batch),
                processing_time=batch_time,
                success=False,
                error=str(last_error)
            )
            
            raise BatchProcessingError(
                f"Batch {batch_index} failed after {retries} attempts: {str(last_error)}",
                batch_index,
                batch
            )
    
    def _handle_batch_results(
        self,
        completed_tasks: List[Any],
        errors: List[Dict[str, Any]]
    ) -> List[Optional[R]]:
        """
        Handle results from completed batch tasks.
        
        Args:
            completed_tasks: List of completed task results
            errors: List to store encountered errors
            
        Returns:
            List[Optional[R]]: Processed results
        """
        results = []
        
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                if isinstance(task_result, BatchProcessingError):
                    errors.append({
                        "batch_index": task_result.batch_index,
                        "error": str(task_result),
                        "failed_items": task_result.failed_items
                    })
                else:
                    errors.append({
                        "error": str(task_result),
                        "type": type(task_result).__name__
                    })
                results.extend([None] * len(task_result.failed_items))
            else:
                results.extend(task_result)
                
        return results

    async def optimize_batch_size(
        self,
        sample_items: List[T],
        processor: callable,
        target_throughput: float
    ) -> int:
        """
        Optimize batch size based on performance testing.
        
        Args:
            sample_items: Sample items for testing
            processor: Processing function
            target_throughput: Target items per second
            
        Returns:
            int: Optimized batch size
        """
        try:
            test_sizes = [5, 10, 20, 50, 100]
            best_size = self.batch_size
            best_throughput = 0
            
            for size in test_sizes:
                if len(sample_items) < size:
                    break
                    
                # Test performance with current size
                test_batch = sample_items[:size]
                start_time = time.perf_counter()
                
                try:
                    await processor(test_batch)
                    duration = time.perf_counter() - start_time
                    throughput = size / duration
                    
                    self.logger.debug(
                        f"Batch size {size}: {throughput:.2f} items/s"
                    )
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_size = size
                        
                    if throughput >= target_throughput:
                        break
                        
                except Exception as e:
                    self.logger.warning(
                        f"Size {size} failed performance test: {str(e)}"
                    )
                    break
            
            return best_size
            
        except Exception as e:
            self.logger.error(f"Batch size optimization failed: {str(e)}")
            return self.batch_size
```