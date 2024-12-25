Centralizing metrics collection can be achieved by leveraging the `MetricsCollector` class in `metrics_collector.py` and integrating it with the `SystemMonitor` in `monitoring.py`. Here's a suggested approach:

1. **Adapt `SystemMonitor` to use `MetricsCollector`:** Modify the `SystemMonitor` to store an instance of `MetricsCollector`. Instead of storing metrics internally in `SystemMonitor._metrics`, use the `MetricsCollector.track_operation()` method to record all gathered metrics. This will send the system resource usage (CPU, memory, disk) and optionally the token usage data to the same central location.

2. **Standardize the Metric Format:** Ensure that both `SystemMonitor` and any other modules using `MetricsCollector` (like the documentation generation process in `main.py`) use a consistent format for reporting metrics. This will make analysis and reporting easier. A common format could include fields like:
    * `operation_type`: (e.g., "system_resource_usage", "token_usage", "documentation_generation")
    * `timestamp`: Date and time of the metric
    * `success`: Boolean indicating operation success/failure
    * `duration`:  Operation duration (if applicable)
    * `metrics`: A dictionary containing specific metric values (e.g., `cpu_percent`, `memory_available`, `prompt_tokens`)

3. **Remove Redundant Metric Handling in `SystemMonitor`:**  Once `MetricsCollector` is used, remove the internal metric storage (`_metrics`), `_store_metrics`, and `_calculate_averages` methods from `SystemMonitor`. The `MetricsCollector` will handle these responsibilities. The `SystemMonitor` should focus solely on collecting the raw data and passing it to the `MetricsCollector`.

4. **Update `get_metrics` in `SystemMonitor`:**  Modify the `get_metrics` method to retrieve the collected metrics from the `MetricsCollector` instance. You can then perform any necessary calculations (like averages) or status determination based on the data from `MetricsCollector`.

5. **Consider Asynchronous Operations:** Since `MetricsCollector.track_operation` is asynchronous, ensure that the `SystemMonitor` handles the calls correctly, potentially using `asyncio.create_task` to avoid blocking the monitoring loop.

By implementing these changes, you'll have a single point of entry for all metrics, simplifying analysis and reporting. Let me know if you'd like a more concrete code example for these modifications.
