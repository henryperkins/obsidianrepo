---
epoch: 1734652776946
modelKey: gemini-2.0-flash-thinking-exp-1219|google
tags:
  - copilot-conversation
---

**user**: [metrics_collector](./metrics_collector.md)
[Timestamp: 2024/12/19 17:59:36]

**ai**: The user is asking about the `@metrics_collector` tool. I need to look at the provided context for information about this tool. I see the code for the `MetricsCollector` class and its documentation in the provided context. I can now describe what this tool does and how it functions based on the code and documentation.:

*   **Collects Code Metrics:** It gathers various metrics about Python code, such as cyclomatic complexity, cognitive complexity, maintainability index, Halstead metrics, and lines of code. It also tracks the number of functions and classes scanned.
*   **Tracks Scan Progress:** It can update the progress of code scans, specifically for functions and classes.
*   **Records Operations:** It logs different operations performed, including their success status, duration, and any associated metadata or usage information. This includes tracking token usage for language models.
*   **Stores Metrics History:** It maintains a history of collected metrics for each module analyzed, saving this data to a `metrics_history.json` file.
*   **Provides Aggregated Data:** It can provide aggregated statistics, such as total token usage and costs.
*   **Supports Validation Metrics:** It can track the success or failure of schema validation processes.
*   **Uses a Singleton Pattern:** The `MetricsCollector` class is implemented as a singleton, ensuring only one instance exists throughout the application.
*   **Includes Logging:** It uses a logger to record events and potential errors.
*   **Provides Console Output:** It prints formatted information about collected metrics and tracked operations to the console.

In essence, `@metrics_collector` acts as a central hub for gathering and managing performance and complexity metrics during code analysis processes.

[Timestamp: 2024/12/19 17:59:40]

**user**: # In metrics_collector.py

class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    _instance = None
    _initialized = False

    # ... (rest of the class)

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module."""
        try:
            if not module_name or module_name == "default_module":
                # Try to determine actual module name
                if hasattr(metrics, "module_name") and metrics.module_name:
                    module_name = metrics.module_name

            if not module_name or not metrics:
                self.logger.warning(
                    f"Invalid metrics data received with correlation ID: {self.correlation_id}"
                )
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name]
                )
                if current_metrics == last_metrics:
                    return

            self.current_module_metrics[module_name] = metrics

            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics,
                "correlation_id": self.correlation_id,
            }

            # Add formatted console output - Consolidated output
            print_info(f"Metrics for module: {module_name}")
            display_metrics(current_metrics)

            if module_name in self.metrics_history:
                if self.metrics_history[module_name]:
                    last_entry = self.metrics_history[module_name][-1]
                    if last_entry.get("metrics", {}) != current_metrics:
                        self.metrics_history[module_name].append(entry)
                        self._save_history()
                else:
                    self.metrics_history[module_name] = [entry]
                    self._save_history()
            else:
                self.metrics_history[module_name] = [entry]
                self._save_history()

        except Exception as e:
            print_error(
                f"Error collecting metrics: {e}",
                correlation_id=self.correlation_id
            )

    # ... (rest of the class)
[Timestamp: 2024/12/19 17:59:43]

**ai**: The user is pointing out a line in the `collect_metrics` method of the `MetricsCollector` class in `metrics_collector.py`. They've provided the code snippet and highlighted a specific part: `last_entry.get("metrics", {selectedText})`.

The `selectedText` seems to be a placeholder for the entire active note, which contains logs and analysis of a previous issue. The user is likely asking about the meaning or potential issue with using `{selectedText}` as the default value in the `get` method.

My plan is to:
1. **Identify the purpose of this line:** Understand why the code is checking the last entry's metrics.
2. **Explain the `get` method:** Explain how the `get` method works with its default value.
3. **Point out the error:** Explain that `{selectedText}` is not a valid Python expression for a default value in this context.
4. **Suggest a correction:** Recommend using an empty dictionary `{}` as the default value.selectedText}) != current_metrics:
```

Here's what's happening:

1. **`last_entry.get("metrics", {selectedText})`**: This part is trying to retrieve the value associated with the key `"metrics"` from the `last_entry` dictionary.
    *   If the key `"metrics"` exists in `last_entry`, its value is returned.
    *   If the key `"metrics"` **does not** exist, the second argument to `get()`, which is `{selectedText}`, is supposed to be returned as the default value.

2. **The Issue with `{selectedText}`**: The problem here is that `{selectedText}` is not a valid Python expression in this context. `selectedText` is a string containing the content of your active note, and you can't directly use a string like that to define a dictionary within the `get()` method. Python would interpret this as trying to create a set containing the string `selectedText`, which is not the intended behavior.

3. **Intended Behavior**: The goal of this line is to check if the metrics collected in the current run are different from the metrics collected in the previous run for the same module. If they are different, the new metrics are added to the history. The default value in the `get()` method is meant to handle the case where the `"metrics"` key doesn't exist in `last_entry` (which shouldn't happen if there's a previous entry).

**Correcting the Code:**

The correct way to provide a default value of an empty dictionary is to use `{}`:

```python
if last_entry.get("metrics", {}) != current_metrics:
```

**Explanation of the Correction:**

*   By using `{}`, you are explicitly telling Python that if the `"metrics"` key is not found in `last_entry`, the default value should be an empty dictionary. This allows for a proper comparison with `current_metrics`.

**Why this might not have caused an immediate error:**

It's possible that in most cases, the `"metrics"` key does exist in `last_entry`, so the default value is never actually used. However, having an invalid default value like `{selectedText}` is still a bug and could lead to unexpected behavior if the `"metrics"` key were ever missing for some reason.

Thank you for pointing this out! It's important to have correct default values to ensure the code functions as intended in all scenarios.

[Timestamp: 2024/12/19 17:59:49]

**user**: Okay, let's clean up and organize the output to make it more readable without going overboard with `rich`. We'll focus on the `docs.py` file where most of the console output seems to originate, and make strategic adjustments in `console.py`.

Here's a plan, focusing on the identified areas for improvement:

1. **Reduce redundant section breaks:** We'll analyze the output in `docs.py` and remove unnecessary calls to `print_section_break()`.
2. **Consolidate related information:** We'll group related `print_*` calls together for better context.
3. **Refine messages:** We'll make sure the messages are clear and concise.
4. **Adjust `console.py` functions (if needed):** We might tweak the existing functions to better suit the output needs.

Let's start by modifying the `generate_module_documentation` and `generate_documentation` methods in `docs.py`, as these seem to produce a lot of the output.

```python
# In docs.py

    async def generate_module_documentation(
        self, file_path: Path, output_dir: Path, source_code: str | None = None
    ) -> None:
        """
        Generates documentation for a single module file.
        ...
        """
        start_time = datetime.now()
        log_extra = {"correlation_id": self.correlation_id}
        try:
            # Validate file type
            if not file_path.suffix == ".py":
                self.logger.warning(f"Skipping non-Python file: {file_path}", extra=log_extra)
                return  # Early exit

            # Read source code if not provided
            if source_code is None:
                self.logger.info(
                    f"Reading source code from {file_path}", extra=log_extra
                )
                source_code = await read_file_safe_async(file_path)

            if source_code:
                self.logger.info(
                    f"Source code read. Length: {len(source_code)}",
                    extra=log_extra,
                )
            else:
                self.logger.warning(f"Source code is missing or empty for {file_path}", extra=log_extra)
                return  # Early exit for empty files

            # Prepare context for documentation generation
            module_name = file_path.stem
            self.logger.debug(f"Setting metadata: module_name={module_name}, file_path={file_path}")
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": module_name,
                    "creation_time": datetime.now().isoformat(),
                },
            )

            print_info(f"Generating documentation for {file_path}") # More direct info
            output_dir = ensure_directory(output_dir)
            output_path = output_dir / file_path.with_suffix(".md").name

            # Generate documentation
            updated_code, markdown_doc = await self.generate_documentation(context)

            # Write outputs
            output_path.write_text(markdown_doc, encoding="utf-8")
            if updated_code:
                file_path.write_text(updated_code, encoding="utf-8")

            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=True,
                duration=processing_time,
                metadata={"module_path": str(file_path)},
            )
            print_success(f"Successfully processed file: {file_path}") # Use print_success
            self.logger.info(f"Documentation written to {output_path}", extra=log_extra)

        except DocumentationError as doc_error:
            print_error(f"Module documentation generation failed for {file_path}: {doc_error}", correlation_id=self.correlation_id) # Use print_error
            self.logger.error(f"Module documentation generation failed for {file_path}: {doc_error}", exc_info=True, extra=log_extra)
            raise DocumentationError(error_msg) from doc_error
        except Exception as gen_error:
            print_error(f"Unexpected error generating documentation for {file_path}: {gen_error}", correlation_id=self.correlation_id) # Use print_error
            self.logger.error(f"Unexpected error generating documentation for {file_path}: {gen_error}", exc_info=True, extra=log_extra)
            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.track_operation(
                operation_type="module_documentation_generation",
                success=False,
                duration=processing_time,
                metadata={"module_path": str(file_path), "error": str(gen_error)},
            )
            raise DocumentationError(error_msg) from gen_error

    async def generate_documentation(
        self,
        context: DocumentationContext
    ) -> tuple[str, str]:
        """
        Generates documentation for the given context.

        Args:
            context: Structured documentation context containing all necessary information

        Returns:
            Tuple of (original_source, markdown_documentation)

        Raises:
            DocumentationError: If documentation generation fails
        """
        start_time = datetime.now()
        module_name = ""
        log_extra = {"correlation_id": self.correlation_id}

        try:
            # Step 1: Validate source code
            if not context or not context.source_code:
                raise DocumentationError(
                    f"Source code is missing or context is invalid for {context.module_path}"
                )

            original_source = context.source_code.strip()
            if not original_source:
                raise DocumentationError(
                    f"Source code is empty after stripping whitespace for {context.module_path}"
                )

            module_name = context.metadata.get("module_name", "") if context.metadata else ""

            print_info(f"Processing module: {module_name or context.module_path.name}") # More informative

            # Step 2: Extract code elements
            extraction_result: ExtractionResult = await self.code_extractor.extract_code(
                original_source
            )

            if not extraction_result.source_code:
                raise DocumentationError("Extraction failed - no valid code elements found.")

            # Convert extracted elements to proper types
            classes = self._convert_to_extracted_classes(extraction_result.classes)
            functions = self._convert_to_extracted_functions(extraction_result.functions)

            print_status("Code Extraction Results", { # Group related info
                "Classes": len(classes),
                "Functions": len(functions),
                "Variables": len(extraction_result.variables or []),
                "Constants": len(extraction_result.constants or []),
                "Lines of Code": extraction_result.loc,
                "Cyclomatic Complexity": extraction_result.complexity.cyclomatic_complexity,
                "Maintainability Index": f"{extraction_result.complexity.maintainability_index:.2f}",
                "Halstead Volume": f"{extraction_result.complexity.halstead.volume:.2f}",
                "Dependencies": len(extraction_result.imports),
            })
            await self._track_extraction_metrics(start_time, module_name, extraction_result) # Assuming this exists or you'll create it

            # Step 3: Create AI prompt
            prompt_result = await self.prompt_manager.create_documentation_prompt(
                DocumentationContext(
                    source_code=original_source,
                    module_path=context.module_path,
                    include_source=True,
                    metadata=context.metadata,
                    classes=classes,
                    functions=functions
                )
            )
            print_info("AI Prompt generated.") # Simple confirmation

            # Step 4: Generate documentation with AI service
            processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                context,
                schema=None  # Add schema if needed
            )

            # Step 5: Parse AI response
            # The response is already parsed by the AI service
            parsed_response = processing_result

            # Step 6: Create documentation data
            documentation_data = DocumentationData(
                module_name=module_name,
                module_path=context.module_path or Path(),
                module_summary=str(parsed_response.content.get("summary", "")),
                source_code=original_source,
                docstring_data=self._create_docstring_data(parsed_response.content),
                ai_content=processing_result.content,
                code_metadata={
                    "classes": [
                        cls.to_dict() for cls in classes
                    ],  # Convert to dictionaries
                    "functions": [
                        func.to_dict() for func in functions
                    ],  # Convert to dictionaries
                    "variables": extraction_result.variables or [],
                    "constants": extraction_result.constants or [],
                    "module_docstring": extraction_result.module_docstring,
                    "source_code": original_source,
                },
            )

            # Step 7: Generate markdown
            markdown_doc = self.markdown_generator.generate(documentation_data)

            # Step 8: Track metrics
            await self._track_generation_metrics(
                start_time=start_time,
                module_name=module_name,
                processing_result=processing_result
            )

            return original_source, markdown_doc

        except Exception as error:
            await self._handle_generation_error(
                error=error,
                start_time=start_time,
                module_name=module_name
            )
            raise
```

**Changes Made:**

*   **`generate_module_documentation`:**
    *   Combined logging of reading source code into a single `print_info`.
    *   Used `print_info` for a more direct message about generating documentation.
    *   Switched to `print_success` for successful processing.
    *   Used `print_error` for both `DocumentationError` and generic `Exception`.
    *   Removed redundant logging that was already covered by `print_*` functions.
*   **`generate_documentation`:**
    *   Added a more informative `print_info` at the start of module processing.
    *   Grouped code extraction results into a single `print_status` call for better readability.
    *   Added a simple confirmation for AI prompt generation.
    *   Removed redundant logging.

Now, let's adjust how metrics are displayed in `metrics_collector.py` to avoid the extra section breaks around them.

```python
# In metrics_collector.py

class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    _instance = None
    _initialized = False

    # ... (rest of the class)

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module."""
        try:
            if not module_name or module_name == "default_module":
                # Try to determine actual module name
                if hasattr(metrics, "module_name") and metrics.module_name:
                    module_name = metrics.module_name

            if not module_name or not metrics:
                self.logger.warning(
                    f"Invalid metrics data received with correlation ID: {self.correlation_id}"
                )
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name]
                )
                if current_metrics == last_metrics:
                    return

            self.current_module_metrics[module_name] = metrics

            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics,
                "correlation_id": self.correlation_id,
            }

            # Add formatted console output - Consolidated output
            print_info(f"Metrics for module: {module_name}")
            display_metrics(current_metrics)

            if module_name in self.metrics_history:
                if self.metrics_history[module_name]:
                    last_entry = self.metrics_history[module_name][-1]
                    if last_entry.get("metrics", {}) != current_metrics:
                        self.metrics_history[module_name].append(entry)
                        self._save_history()
                else:
                    self.metrics_history[module_name] = [entry]
                    self._save_history()
            else:
                self.metrics_history[module_name] = [entry]
                self._save_history()

        except Exception as e:
            print_error(
                f"Error collecting metrics: {e}",
                correlation_id=self.correlation_id
            )

    # ... (rest of the class)
```

**Changes in `metrics_collector.py`:**

*   In `collect_metrics`, instead of printing a section break and then the metrics, we now print an `info` message indicating the module, followed directly by the `display_metrics` output. This removes the extra visual break.

Finally, let's adjust the `track_operation` method to avoid unnecessary breaks as well.

```python
# In metrics_collector.py

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Track an operation with its metrics."""
        try:
            operation: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": operation_type,
                "success": success,
                "duration": duration,
                "correlation_id": self.correlation_id,
            }
            if metadata:
                operation["metadata"] = metadata
            if usage:
                operation["usage"] = usage

            self.operations.append(operation)

            # Add formatted output - Consolidated output
            status_details = {
                "Success": success,
                "Duration": f"{duration:.2f}s",
                **(metadata or {})
            }
            print_status(f"Operation: {operation_type}", status_details)

        except Exception as e:
            print_error(
                f"Error tracking operation: {e}",
                correlation_id=self.correlation_id
            )
```

**Changes in `metrics_collector.py` for `track_operation`:**

*   The `print_status` call now includes all relevant details, removing the need for a preceding section break.

With these changes, the output should be more concise and logically grouped. The redundant section breaks should be reduced, and related information will be presented together.

**Further Potential Improvements (If Needed):**

*   **Custom Formatting in `display_metrics`:** If the default table format isn't ideal, you can customize the output within the `display_metrics` function in `console.py`.
*   **Conditional Output:** You could add flags or verbosity levels to control the amount of output. For example, only show detailed metrics if a `--verbose` flag is used.
*   **Specific `print_*` for Key Steps:** If certain steps are particularly important, you could create new `print_step_start()` and `print_step_end()` functions in `console.py` to visually highlight them.

Remember to test these changes to ensure the output is clearer and more organized to your liking. You might need to tweak things further based on the specific flow and output of your script.

[Timestamp: 2024/12/19 18:00:08]

**ai**: The user wants to refactor the console output in `docs.py` and `metrics_collector.py` to be more readable by reducing redundant section breaks and consolidating related information. They've provided modified code for `generate_module_documentation`, `generate_documentation`, `collect_metrics`, and `track_operation`.

My plan is to:
1. **Review the changes in `docs.py`:**  Check how the `print_*` functions are used and if the consolidation makes sense.
2. **Review the changes in `metrics_collector.py`:** Check how the section breaks are removed and if the output remains informative.
3. **Address the `selectedText` issue:**  Point out the error in the `collect_metrics` method and reiterate the correct way to fix it.
4. **Provide feedback on the changes:**  Summarize the improvements and suggest further potential enhancements if needed.Okay, let's review the changes you've made to improve the console output.

**Changes in `docs.py`:**

The modifications in `generate_module_documentation` and `generate_documentation` look good. Here's a breakdown of the improvements:

*   **Reduced Redundancy:** You've successfully removed redundant logging and section breaks, making the output cleaner.
*   **Consolidated Information:** Grouping related information, like the code extraction results in `generate_documentation`, into a single `print_status` call significantly improves readability and context.
*   **Clearer Messages:** Using more direct and informative messages in `print_info` and utilizing `print_success` and `print_error` appropriately enhances the clarity of the output.

**Changes in `metrics_collector.py`:**

*   **`collect_metrics`:**  You've correctly removed the extra section breaks around the metrics output. Printing the module name followed directly by the metrics table is a good way to present this information concisely.
*   **`track_operation`:**  Consolidating the operation details into a single `print_status` call is a positive change. It avoids unnecessary visual breaks and presents the key information together.

**Addressing the `selectedText` Issue (Again):**

I noticed that in the modified `collect_metrics` code you provided, the line still contains the error:

```python
if last_entry.get("metrics", {selectedText}) != current_metrics:
```

As I mentioned before, `{selectedText}` is not a valid way to specify a default value of an empty dictionary. Please change this to:

```python
if last_entry.get("metrics", {}) != current_metrics:
```

This correction is crucial for the code to function correctly when the `"metrics"` key is not present in `last_entry`.

**Overall Feedback:**

The changes you've implemented are a significant step towards making the console output more readable and organized. By reducing redundant breaks and grouping related information, you've created a more streamlined and easier-to-understand output flow.

**Regarding the `selectedText` in `track_operation`:**

Similarly, in the `track_operation` method, you have:

```python
status_details = {
    "Success": success,
    "Duration": f"{duration:.2f}s",
    **(metadata or {selectedText})
}
```

The `or {selectedText}` part is also incorrect. If `metadata` is `None`, you likely intend to provide an empty dictionary as the default. The correct way to do this is:

```python
status_details = {
    "Success": success,
    "Duration": f"{duration:.2f}s",
    **(metadata or {})
}
```

**Further Potential Improvements (as you mentioned):**

The suggestions you listed for further improvements are excellent and can be implemented as needed:

*   **Custom Formatting in `display_metrics`:**  If the default table format isn't perfect, customizing it in `console.py` would be the next logical step.
*   **Conditional Output/Verbosity Levels:** Implementing verbosity levels (e.g., using a `--verbose` flag) is a common and effective way to control the amount of detail in the output. This is particularly useful for tools that can produce a lot of information.
*   **Specific `print_*` for Key Steps:** Creating functions like `print_step_start()` and `print_step_end()` in `console.py` can be a good way to visually highlight important stages in the process.

**Next Steps:**

1. **Correct the `selectedText` errors:** Ensure you replace `{selectedText}` with `{}` in both `collect_metrics` and `track_operation`.
2. **Test the changes:** Run your code with these modifications to confirm that the output is indeed cleaner and more organized. Pay attention to different scenarios and edge cases.
3. **Consider further enhancements:** If the output is still not ideal in certain situations, explore the additional improvements you suggested, such as customizing `display_metrics` or adding verbosity levels.

You're on the right track to creating a more user-friendly console experience!

[Timestamp: 2024/12/19 18:00:42]

**user**: Generate teh updated metrics_collector module
[Timestamp: 2024/12/19 18:00:49]

**ai**: The user wants the updated `metrics_collector` module code, incorporating the previously discussed changes for cleaner console output and the correction of the `selectedText` error.

**Plan:**
1. Retrieve the original `metrics_collector` code.
2. Apply the changes discussed for `collect_metrics` to remove extra section breaks and correct the `selectedText` error.
3. Apply the changes discussed for `track_operation` to remove extra section breaks and correct the `selectedText` error.
4. Present the complete updated code.import json
import os
import uuid

from core.logger import LoggerSetup
from core.types.base import MetricData
from core.console import (
    print_error,
    print_info,
    print_status,
    print_section_break,
    display_metrics
)

class MetricsCollector:
    """Collects and stores metrics data for code analysis."""

    _instance = None
    _initialized = False

    def __new__(cls, correlation_id: str | None = None) -> "MetricsCollector":
        """Ensure only one instance exists (singleton pattern)."""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            correlation_id: Optional correlation ID for tracking metrics
        """
        if not self._initialized:
            self.logger = LoggerSetup.get_logger(__name__, correlation_id)
            self.correlation_id = correlation_id or str(uuid.uuid4())
            self.metrics_history: dict[str, list[dict[str, Any]]] = {}
            self.operations: list[dict[str, Any]] = []
            self.current_module_metrics: dict[str, Any] = {}
            self.accumulated_functions = 0
            self.accumulated_classes = 0
            self.current_module: str | None = None
            self.has_metrics = False
            self._load_history()
            self.__class__._initialized = True

    def collect_metrics(self, module_name: str, metrics: MetricData) -> None:
        """Collect metrics for a module."""
        try:
            if not module_name or module_name == "default_module":
                # Try to determine actual module name
                if hasattr(metrics, "module_name") and metrics.module_name:
                    module_name = metrics.module_name

            if not module_name or not metrics:
                self.logger.warning(
                    f"Invalid metrics data received with correlation ID: {self.correlation_id}"
                )
                return

            if module_name not in self.metrics_history:
                self.metrics_history[module_name] = []

            current_metrics = self._metrics_to_dict(metrics)
            if module_name in self.current_module_metrics:
                last_metrics = self._metrics_to_dict(
                    self.current_module_metrics[module_name]
                )
                if current_metrics == last_metrics:
                    return

            self.current_module_metrics[module_name] = metrics

            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics,
                "correlation_id": self.correlation_id,
            }

            # Add formatted console output - Consolidated output
            print_info(f"Metrics for module: {module_name}")
            display_metrics(current_metrics)

            if module_name in self.metrics_history:
                if self.metrics_history[module_name]:
                    last_entry = self.metrics_history[module_name][-1]
                    if last_entry.get("metrics", {}) != current_metrics:
                        self.metrics_history[module_name].append(entry)
                        self._save_history()
                else:
                    self.metrics_history[module_name] = [entry]
                    self._save_history()
            else:
                self.metrics_history[module_name] = [entry]
                self._save_history()

        except Exception as e:
            print_error(
                f"Error collecting metrics: {e}",
                correlation_id=self.correlation_id
            )

    def update_scan_progress(self, module_name: str, item_type: str, item_name: str) -> None:
        """Update the scan progress for functions or classes."""
        try:
            if module_name == "default_module":
                return

            if item_type == 'function':
                self.accumulated_functions = total
                if self.current_module_metrics and self.current_module:
                    metrics = self.current_module_metrics[self.current_module]
                    metrics.total_functions = total
                    metrics.scanned_functions = scanned
            elif item_type == 'class':
                self.accumulated_classes = total
                if self.current_module_metrics and self.current_module:
                    metrics = self.current_module_metrics[self.current_module]
                    metrics.total_classes = total
                    metrics.scanned_classes = scanned

            self.has_metrics = True

        except Exception as e:
            self.logger.error(
                f"Error updating scan progress: {e} with correlation ID: {self.correlation_id}"
            )

    def _metrics_to_dict(self, metrics: MetricData) -> dict[str, Any]:
        """Convert MetricData to dictionary format."""
        try:
            return {
                "cyclomatic_complexity": getattr(metrics, "cyclomatic_complexity", 0),
                "cognitive_complexity": getattr(metrics, "cognitive_complexity", 0),
                "maintainability_index": getattr(metrics, "maintainability_index", 0.0),
                "halstead_metrics": getattr(metrics, "halstead_metrics", {}),
                "lines_of_code": getattr(metrics, "lines_of_code", 0),
                "total_functions": getattr(metrics, "total_functions", 0),
                "scanned_functions": getattr(metrics, "scanned_functions", 0),
                "function_scan_ratio": metrics.function_scan_ratio,
                "total_classes": getattr(metrics, "total_classes", 0),
                "scanned_classes": getattr(metrics, "scanned_classes", 0),
                "class_scan_ratio": metrics.class_scan_ratio,
                "complexity_graph": getattr(metrics, "complexity_graph", None),
            }
        except Exception as e:
            self.logger.error(
                f"Error converting metrics to dict: {e} with correlation ID: {self.correlation_id}"
            )
            return {}

    async def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Track an operation with its metrics."""
        try:
            operation: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": operation_type,
                "success": success,
                "duration": duration,
                "correlation_id": self.correlation_id,
            }
            if metadata:
                operation["metadata"] = metadata
            if usage:
                operation["usage"] = usage

            self.operations.append(operation)

            # Add formatted output - Consolidated output
            status_details = {
                "Success": success,
                "Duration": f"{duration:.2f}s",
                **(metadata or {})
            }
            print_status(f"Operation: {operation_type}", status_details)

        except Exception as e:
            print_error(
                f"Error tracking operation: {e}",
                correlation_id=self.correlation_id
            )

    async def close(self) -> None:
        """Clean up resources and save final state."""
        try:
            self._save_history()
        except Exception as e:
            self.logger.error(
                f"Error closing MetricsCollector: {e} with correlation ID: {self.correlation_id}"
            )

    def _load_history(self) -> None:
        """Load metrics history from storage."""
        try:
            if os.path.exists("metrics_history.json"):
                with open("metrics_history.json", "r") as f:
                    self.metrics_history = json.load(f)
        except Exception as e:
            self.logger.error(
                f"Error loading metrics history: {str(e)} with correlation ID: {self.correlation_id}"
            )
            self.metrics_history = {}

    def _save_history(self) -> None:
        """Save metrics history to storage."""
        try:
            with open("metrics_history.json", "w") as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(
                f"Error saving metrics history: {str(e)} with correlation ID: {self.correlation_id}",
                exc_info=True
            )

    def clear_history(self) -> None:
        """Clear all metrics history."""
        self.metrics_history = {}
        if os.path.exists("metrics_history.json"):
            os.remove("metrics_history.json")

    def get_metrics(self) -> dict[str, Any]:
        """Get the current metrics data."""
        return {
            "current_metrics": self.current_module_metrics,
            "history": self.metrics_history,
            "operations": self.operations,
        }

    def get_metrics_history(self, module_name: str) -> list[dict[str, Any]]:
        """Get metrics history for a specific module."""
        return self.metrics_history.get(module_name, [])

    def collect_token_usage(self, prompt_tokens: int, completion_tokens: int, cost: float, model: str) -> None:
        """Collect metrics specifically for token usage."""
        try:
            self.operations.append({
                "timestamp": datetime.now().isoformat(),
                "operation_type": "token_usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost": cost,
                "model": model,
                "correlation_id": self.correlation_id,
            })
            self.logger.info(
                f"Token usage collected: {prompt_tokens + completion_tokens} tokens, ${cost:.4f}.",
                extra={"model": model, "correlation_id": self.correlation_id}
            )
        except Exception as e:
            self.logger.error(f"Error collecting token usage: {e}", exc_info=True)

    def collect_validation_metrics(self, success: bool) -> None:
        """Collect metrics for schema validation results."""
        try:
            if success:
                self._increment_metric("validation_success")
            else:
                self._increment_metric("validation_failure")
        except Exception as e:
            self.logger.error(f"Error collecting validation metrics: {e}", exc_info=True)

    def _increment_metric(self, metric_name: str) -> None:
        """Increment a specific metric."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = 0
        self.metrics_history[metric_name] += 1

    def get_aggregated_token_usage(self) -> dict[str, Union[int, float]]:
        """Aggregate token usage statistics across operations."""
        total_prompt_tokens = sum(op.get("prompt_tokens", 0) for op in self.operations if op["operation_type"] == "token_usage")
        total_completion_tokens = sum(op.get("completion_tokens", 0) for op in self.operations if op["operation_type"] == "token_usage")
        total_cost = sum(op.get("total_cost", 0) for op in self.operations if op["operation_type"] == "token_usage")

        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_cost": total_cost,
        }
```
[Timestamp: 2024/12/19 18:01:11]