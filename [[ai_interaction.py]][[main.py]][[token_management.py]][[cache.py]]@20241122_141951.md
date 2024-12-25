---
epoch: 1732306791972
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: [ai_service.py](ai_service.py.md)[main.py](Resources/simple%20source/main.py.md)[token_management.py](token_management.py.md)[cache.py](cache.py.md)  
[Timestamp: 2024/11/22 14:19:51]

**ai**: ```python  
"""Main module for documentation generation workflow."""

import asyncio  
import argparse  
import sys  
import time  
import uuid  
from pathlib import Path  
from typing import Dict, Any, Optional  
from tqdm.asyncio import tqdm_asyncio

from core.logger import LoggerSetup  
from core.config import AzureOpenAIConfig  
from core.cache import Cache  
from core.monitoring import MetricsCollector  
from core.metrics_calculator import MetricsCalculator  
from ai_interaction import AIInteractionHandler  
from utils.ast_analysis import ASTAnalyzer  
from docs.markdown_generator import MarkdownDocumentationGenerator  
from exceptions import WorkflowError

logger = LoggerSetup.get_logger(**name**)

async def load_and_validate_config(args: argparse.Namespace) -> AzureOpenAIConfig:  
    """Load and validate configuration."""  
    try:  
        config = AzureOpenAIConfig.from_env()  
        if args.analyze_metrics:  
            config.metrics_config.maintainability_threshold = args.metrics_threshold  
        return config  
    except Exception as e:  
        logger.error(f"Failed to load or validate configuration: {e}")  
        raise

def setup_logging(log_level: str, log_format: str, log_directory: Optional[Path] = None):  
    """Configure logging."""  
    LoggerSetup.configure(log_level, log_format, str(log_directory) if log_directory else None)

async def process_file(  
    processor: "DocumentationProcessor", # Forward reference due to circular dependency  
    file_path: Path,  
    output_dir: Path  
) -> Dict[str, Any]:  
    """Process a single file with enhanced error handling and metrics."""  
    start_time = time.time()  
    metrics = {  
        "file_path": str(file_path),  
        "start_time": start_time,  
        "status": "pending",  
    }

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            source_code = file.read()

        if not source_code.strip():
            metrics["status"] = "skipped"
            metrics["message"] = "Empty source file"
            return processor._create_result("skipped", message="Empty source file", metrics=metrics)

        updated_code, documentation, processing_metrics = await processor.process_code(source_code, str(file_path))
        metrics.update(processing_metrics)

        if not documentation:
            metrics["status"] = "failed"
            metrics["message"] = "AI processing failed"
            return processor._create_result("failed", message="AI processing failed", metrics=metrics)

        doc_result = await processor.generate_documentation(updated_code, documentation, file_path, output_dir)
        if doc_result:
            metrics.update({
                "end_time": time.time(),
                "duration": time.time() - start_time,
                "status": "success",
                "documentation_size": len(doc_result.get("documentation", "")),
            })
            return processor._create_result("success", result=doc_result, metrics=metrics)
        else:
            metrics["status"] = "failed"
            metrics["message"] = "Documentation generation failed"
            return processor._create_result("failed", message="Documentation generation failed", metrics=metrics)

    except Exception as e:
        metrics.update({
            "end_time": time.time(),
            "duration": time.time() - start_time,
            "status": "failed",
            "error": str(e)
        })
        return processor._create_result("failed", error=str(e), metrics=metrics)

class DocumentationProcessor: # Modified class structure  
    """Handles the core documentation processing logic."""

    def __init__(self, components: "AsyncComponentManager", config: AzureOpenAIConfig): # Forward reference
        self.ai_handler = components.components["ai_handler"]
        self.ast_analyzer = components.components["ast_analyzer"]
        self.metrics_calculator = components.components["metrics_calculator"]
        self.metrics_collector = components.components["metrics"]
        self.config = config

    async def process_code(self, source_code: str, file_path: str) -> tuple[str, str, dict]:
        """Process source code and return updated code, documentation, and metrics."""
        tree = self.ast_analyzer.parse_source_code(source_code)
        code_metrics = self.metrics_calculator.calculate_all_metrics(tree)
        self.metrics_collector.track_code_quality(file_path, code_metrics)
        updated_code, documentation = await self.ai_handler.process_code(source_code)
        return updated_code, documentation, {"code_metrics": code_metrics}

    async def generate_documentation(self, updated_code: str, documentation: str, file_path: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Generate and save documentation."""
        try:
            markdown_gen = MarkdownDocumentationGenerator(source_code=updated_code, module_path=str(file_path))
            if not documentation:
                raise ValueError("Generated documentation is empty")
            doc_path = output_dir / f"{file_path.stem}_docs.md"
            await asyncio.to_thread(doc_path.write_text, documentation, encoding="utf-8")
            return {"documentation": documentation, "documentation_path": str(doc_path), "updated_code": updated_code}
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return None

    def _create_result(self, status: str, result: Optional[Dict] = None, message: str = "", error: str = "", metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        return {
            "status": status,
            "result": result or {},
            "message": message,
            "error": error,
            "metrics": metrics or {},
        }

# ... (Rest Of the Code Remains the Same, except for the Parts Using DocumentationProcessor)

```
Key changes and explanations:

1. **`setup_logging` fix:** Added a check for `log_directory` being None to avoid passing `str(None)` to the logger configuration.
2. **`process_file` moved:**  Moved the `process_file` function outside of the `DocumentationProcessor` class to resolve the circular dependency issue. It now takes `DocumentationProcessor` as an argument.
3. **`DocumentationProcessor` restructuring:**
    - Removed the embedded `_generate_documentation`, `_save_documentation`, and `_create_result` methods and made them regular methods of the class.
    - Introduced a new `process_code` method to encapsulate code analysis, metrics calculation, and AI processing. This improves code organization and reduces the complexity of `process_file`.
    - Simplified the `generate_documentation` method to focus solely on documentation generation and saving.
4. **Type Hinting:** Added type hints (e.g., `-> tuple[str, str, dict]`) for better code clarity and maintainability. I also used a forward reference string for `AsyncComponentManager` to avoid circular imports.
5. **Metrics Handling:** Improved metrics handling in `process_file` to include metrics from the `process_code` step.
6. **Error Handling:** Enhanced error handling in `generate_documentation` to return `None` on failure, which is then handled in `process_file`.

These changes improve the code structure, readability, and maintainability by separating concerns more effectively and addressing the circular dependency issue. The code is now more modular and easier to test and extend. The use of type hints also significantly improves readability and helps catch potential errors early on.
[Timestamp: 2024/11/22 14:20:29]

**user**: Honestly between [extraction_manager.py](extraction_manager.py.md) and [metrics_calculator.py](metrics_calculator.py.md) and [docs.py](docs.py.md) — main looks bloated 
[Timestamp: 2024/11/22 14:22:25]

**ai**: You're right, the `main.py` file is doing too much. Here's a refactored version that delegates responsibilities more effectively, creating a cleaner and more maintainable structure. The core idea is to introduce a `ProjectProcessor` class to handle the processing of entire projects (directories) or individual files.

```python
"""Main module for documentation generation workflow."""

import asyncio
import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm_asyncio

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.metrics_calculator import MetricsCalculator
from ai_interaction import AIInteractionHandler
from utils.ast_analysis import ASTAnalyzer
from docs.markdown_generator import MarkdownDocumentationGenerator
from exceptions import WorkflowError
from extract.extraction_manager import ExtractionManager  # Import ExtractionManager

logger = LoggerSetup.get_logger(__name__)

# ... (load_and_validate_config, setup_logging remain the same)

class AsyncComponentManager:  # (No changes needed here)
    # ... (Existing code)

class ProjectProcessor:
    """Handles processing of individual files or entire projects."""

    def __init__(self, components: AsyncComponentManager, config: AzureOpenAIConfig):
        self.components = components
        self.config = config
        self.extraction_manager = ExtractionManager(
            metrics_collector=self.components.components["metrics"],
            metrics_calculator=self.components.components["metrics_calculator"]
        ) # Initialize ExtractionManager here

    async def process_project(self, source_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process all Python files in a directory."""
        results = []
        files = list(source_path.rglob("*.py"))
        async for file in tqdm_asyncio(files, desc="Processing files"):
            results.append(await self._process_item(file, output_dir))
        return results

    async def process_file(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single file."""
        return await self._process_item(file_path, output_dir)

    async def _process_item(self, item_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single file or directory item."""
        start_time = time.time()
        metrics = {
            "item_path": str(item_path),
            "start_time": start_time,
            "status": "pending",
        }

        try:
            with open(item_path, "r", encoding="utf-8") as file:
                source_code = file.read()

            if not source_code.strip():
                metrics.update({"status": "skipped", "message": "Empty source file"})
                return self._create_result("skipped", message="Empty source file", metrics=metrics)

            # Use ExtractionManager
            metadata = self.extraction_manager.extract_metadata(source_code)

            updated_code, documentation = await self.components.components["ai_handler"].process_code(source_code)

            if not documentation:
                metrics.update({"status": "failed", "message": "AI processing failed"})
                return self._create_result("failed", message="AI processing failed", metrics=metrics)

            doc_result = await self._generate_documentation(updated_code, documentation, item_path, output_dir)

            metrics.update({
                "end_time": time.time(),
                "duration": time.time() - start_time,
                "status": "success" if doc_result else "failed",
                "documentation_size": len(doc_result.get("documentation", "")) if doc_result else 0,
                **metadata['metrics'] # Include extracted metrics
            })
            return self._create_result("success" if doc_result else "failed", result=doc_result, metrics=metrics)

        except Exception as e:
            metrics.update({
                "end_time": time.time(),
                "duration": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            })
            return self._create_result("failed", error=str(e), metrics=metrics)

    async def _generate_documentation(self, updated_code: str, documentation: str, file_path: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Generate documentation with enhanced formatting and validation."""
        # ... (same as before)

    def _create_result(self, status: str, result: Optional[Dict] = None, message: str = "", error: str = "", metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        # ... (same as before)

class WorkflowOrchestrator: # Modified WorkflowOrchestrator
    """Orchestrates the entire documentation generation workflow."""

    def __init__(self, config: AzureOpenAIConfig):
        self.config = config

    async def run(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run the complete workflow with comprehensive monitoring."""
        # ... (workflow setup remains the same)

        try:
            async with AsyncComponentManager(self.config) as components:
                processor = ProjectProcessor(components, self.config) # Use ProjectProcessor

                if source_path.is_file():
                    results = await processor.process_file(source_path, output_dir) # Use process_file
                elif source_path.is_dir():
                    results = await processor.process_project(source_path, output_dir) # Use process_project
                else:
                    raise ValueError(f"Invalid source path: {source_path}")

                # ... (summary generation remains the same)

        except Exception as e:
            # ... (error handling remains the same)

        return results

    # ... (_update_results and _generate_summary remain the same)

# ... (parse_arguments and run_workflow remain the same)
```

Key Changes:

- **`ProjectProcessor` Class:** This new class handles the logic for processing both individual files and entire project directories. This significantly declutters `main.py` and makes the code more organized.
- **`ExtractionManager` Integration:** The `ExtractionManager` is now initialized and used within the `ProjectProcessor` to extract metadata and metrics.
- **Simplified `WorkflowOrchestrator`:** The `run` method is now much cleaner, delegating the processing logic to the `ProjectProcessor`.
- **Improved Metrics Handling:** Metrics are now collected during the `_process_item` stage and included in the results.
- **Clearer Error Handling:** Error handling is more consistent and informative.

This revised structure provides better separation of concerns, making the code more maintainable, testable, and scalable. The introduction of the `ProjectProcessor` significantly improves the overall design and reduces the complexity of the `main` module.  
[Timestamp: 2024/11/22 14:23:07]

**user**: Produce the complete updated main.py  
[Timestamp: 2024/11/22 14:23:17]

**ai**: Here's the complete updated `main.py` file with the refactoring changes applied:

```python
"""Main module for documentation generation workflow."""

import asyncio
import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm_asyncio

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.metrics_calculator import MetricsCalculator
from ai_interaction import AIInteractionHandler
from utils.ast_analysis import ASTAnalyzer
from docs.markdown_generator import MarkdownDocumentationGenerator
from exceptions import WorkflowError
from extract.extraction_manager import ExtractionManager

logger = LoggerSetup.get_logger(__name__)

async def load_and_validate_config(args: argparse.Namespace) -> AzureOpenAIConfig:
    """Load and validate configuration."""
    try:
        config = AzureOpenAIConfig.from_env()
        if args.analyze_metrics:
            config.metrics_config.maintainability_threshold = args.metrics_threshold
        return config
    except Exception as e:
        logger.error(f"Failed to load or validate configuration: {e}")
        raise

def setup_logging(log_level: str, log_format: str, log_directory: Optional[Path] = None):
    """Configure logging."""
    LoggerSetup.configure(log_level, log_format, str(log_directory) if log_directory else None)

class AsyncComponentManager:
    """Manages async components lifecycle and dependencies."""

    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.components: Dict[str, Any] = {}

    async def __aenter__(self):
        """Initialize all components in correct order with proper dependencies."""
        try:
            self.components["metrics"] = MetricsCollector()
            self.components["metrics_calculator"] = MetricsCalculator()
            self.components["ast_analyzer"] = ASTAnalyzer(
                metrics_calculator=self.components["metrics_calculator"]
            )

            if self.config.cache_enabled:
                self.components["cache"] = await Cache.create(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password
                )
            else:
                logger.warning("Cache is disabled.")

            self.components["ai_handler"] = AIInteractionHandler(
                config=self.config,
                cache=self.components.get("cache"),
                metrics_collector=self.components["metrics"]
            )

            logger.info("All components initialized successfully")
            return self

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            await self.__aexit__(type(e), e, e.__traceback__)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup components in reverse order."""
        cleanup_order = [
            "ai_handler",
            "cache",
            "ast_analyzer",
            "metrics_calculator",
            "metrics"
        ]

        for component_name in cleanup_order:
            component = self.components.get(component_name)
            if component:
                try:
                    cleanup_method = getattr(component, "cleanup", None) or getattr(component, "close", None)
                    if cleanup_method:
                        await cleanup_method()
                    else:
                        logger.warning(f"No cleanup method found for {component_name}")
                except Exception as e:
                    logger.error(f"Error cleaning up {component_name}: {e}")

class ProjectProcessor:
    """Handles processing of individual files or entire projects."""

    def __init__(self, components: AsyncComponentManager, config: AzureOpenAIConfig):
        self.components = components
        self.config = config
        self.extraction_manager = ExtractionManager(
            metrics_collector=self.components.components["metrics"],
            metrics_calculator=self.components.components["metrics_calculator"]
        )

    async def process_project(self, source_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process all Python files in a directory."""
        results = []
        files = list(source_path.rglob("*.py"))
        async for file in tqdm_asyncio(files, desc="Processing files"):
            results.append(await self._process_item(file, output_dir))
        return results

    async def process_file(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single file."""
        return await self._process_item(file_path, output_dir)

    async def _process_item(self, item_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single file or directory item."""
        start_time = time.time()
        metrics = {
            "item_path": str(item_path),
            "start_time": start_time,
            "status": "pending",
        }

        try:
            with open(item_path, "r", encoding="utf-8") as file:
                source_code = file.read()

            if not source_code.strip():
                metrics.update({"status": "skipped", "message": "Empty source file"})
                return self._create_result("skipped", message="Empty source file", metrics=metrics)

            metadata = self.extraction_manager.extract_metadata(source_code)

            updated_code, documentation = await self.components.components["ai_handler"].process_code(source_code)

            if not documentation:
                metrics.update({"status": "failed", "message": "AI processing failed"})
                return self._create_result("failed", message="AI processing failed", metrics=metrics)

            doc_result = await self._generate_documentation(updated_code, documentation, item_path, output_dir)

            metrics.update({
                "end_time": time.time(),
                "duration": time.time() - start_time,
                "status": "success" if doc_result else "failed",
                "documentation_size": len(doc_result.get("documentation", "")) if doc_result else 0,
                **metadata['metrics']
            })
            return self._create_result("success" if doc_result else "failed", result=doc_result, metrics=metrics)

        except Exception as e:
            metrics.update({
                "end_time": time.time(),
                "duration": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            })
            return self._create_result("failed", error=str(e), metrics=metrics)

    async def _generate_documentation(self, updated_code: str, documentation: str, file_path: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Generate documentation with enhanced formatting and validation."""
        try:
            markdown_gen = MarkdownDocumentationGenerator(source_code=updated_code, module_path=str(file_path))
            if not documentation:
                raise ValueError("Generated documentation is empty")
            doc_path = output_dir / f"{file_path.stem}_docs.md"
            await asyncio.to_thread(doc_path.write_text, documentation, encoding="utf-8")
            return {"documentation": documentation, "documentation_path": str(doc_path), "updated_code": updated_code}
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return None

    def _create_result(self, status: str, result: Optional[Dict] = None, message: str = "", error: str = "", metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        return {
            "status": status,
            "result": result or {},
            "message": message,
            "error": error,
            "metrics": metrics or {},
        }

class WorkflowOrchestrator:
    """Orchestrates the entire documentation generation workflow."""

    def __init__(self, config: AzureOpenAIConfig):
        self.config = config

    async def run(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run the complete workflow with comprehensive monitoring."""
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        results = {
            "processed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        try:
            async with AsyncComponentManager(self.config) as components:
                processor = ProjectProcessor(components, self.config)

                if source_path.is_file():
                    file_result = await processor.process_file(source_path, output_dir)
                    self._update_results(results, file_result)
                elif source_path.is_dir():
                    project_results = await processor.process_project(source_path, output_dir)
                    for file_result in project_results:
                        self._update_results(results, file_result)
                else:
                    raise ValueError(f"Invalid source path: {source_path}")

                await self._generate_summary(results, output_dir, workflow_id, time.time() - start_time)

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise WorkflowError(f"Workflow failed: {e}")

        return results

    def _update_results(self, results: Dict[str, Any], file_result: Dict[str, Any]):
        """Update results dictionary with file processing outcomes."""
        status = file_result["status"]
        results[f"{status}_files"] += 1

        token_usage = file_result.get("metrics", {}).get("token_usage", {})
        results["total_tokens"] += token_usage.get("total_tokens", 0)
        results["total_cost"] += token_usage.get("total_cost", 0.0)

    async def _generate_summary(self, results: Dict[str, Any], output_dir: Path, workflow_id: str, duration: float):
        """Generate a summary report of the workflow execution."""
        summary_path = output_dir / f"workflow_summary_{workflow_id}.md"
        summary_content = (
            f"# Workflow Summary\n\n"
            f"**Workflow ID:** {workflow_id}\n"
            f"**Duration:** {duration:.2f} seconds\n"
            f"**Processed Files:** {results['processed_files']}\n"
            f"**Failed Files:** {results['failed_files']}\n"
            f"**Skipped Files:** {results['skipped_files']}\n"
            f"**Total Tokens Used:** {results['total_tokens']}\n"
            f"**Total Cost:** ${results['total_cost']:.2f}\n"
        )

        await asyncio.to_thread(summary_path.write_text, summary_content, encoding="utf-8")
        logger.info(f"Workflow summary saved to {summary_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate documentation using AI.")
    parser.add_argument("source_path", type=Path, help="Path to source code file or directory")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")
    parser.add_argument("--config-file", type=Path, help="Path to configuration file")
    parser.add_argument("--analyze-metrics", action="store_true", help="Include code quality metrics")
    parser.add_argument("--metrics-threshold", type=float, default=65.0, help="Maintainability index threshold")
    return parser.parse_args()

async def run_workflow(args: argparse.Namespace) -> None:
    """Run the documentation generation workflow."""
    try:
        config = await load_and_validate_config(args)
        setup_logging(config.log_level, config.log_format, Path(config.log_directory))

        orchestrator = WorkflowOrchestrator(config)
        results = await orchestrator.run(args.source_path, args.output_dir)
        logger.info("Workflow completed successfully")

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_workflow(args))
```

## Key Changes

1. **`ProjectProcessor` Class**: This class handles the processing of both individual files and entire directories, using the `ExtractionManager` to extract metadata and metrics.
   
2. **`AsyncComponentManager`**: Manages the lifecycle of components, ensuring they are initialized and cleaned up correctly.
3. **`WorkflowOrchestrator`**: Orchestrates the workflow, using `ProjectProcessor` to handle the actual processing logic.
4. **Improved Error Handling**: Consistent error handling across the code, with detailed logging.
5. **Metrics and Documentation**: Uses `ExtractionManager` for extracting metrics and `MarkdownDocumentationGenerator` for generating documentation.

This refactored version provides a cleaner, more modular structure, making the codebase easier to maintain and extend.  
[Timestamp: 2024/11/22 14:24:11]

**user**: Produce [logger.py](logger.py.md) [monitoring.py](monitoring.py.md) [response_parser.py](response_parser.py.md)  
[Timestamp: 2024/11/22 14:25:14]

**ai**: Here are the complete implementations for `logger.py`, `monitoring.py`, and `response_parser.py` as requested:

## `logger.py`

```python
# core/logger.py

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

class RateLimitingFilter(logging.Filter):
    def __init__(self, rate_limit: float = 1.0):
        super().__init__()
        self.rate_limit = rate_limit
        self._last_log = {}
        
    def filter(self, record: logging.LogRecord) -> bool:
        now = datetime.now().timestamp()
        key = f"{record.levelno}:{record.msg}"
        if now - self._last_log.get(key, 0) < self.rate_limit:
            return False
        self._last_log[key] = now
        return True

class LoggerSetup:
    @staticmethod
    def configure(level: str, format_str: str, log_dir: Optional[str] = None) -> None:
        """Configure global logging settings."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = Path(log_dir) / "app.log"
            file_handler = RotatingFileHandler(
                str(log_path),
                maxBytes=1024*1024,
                backupCount=3
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.addFilter(RateLimitingFilter(0.1))
            root_logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        console.addFilter(RateLimitingFilter(1.0))
        root_logger.addHandler(console)

    @staticmethod
    def get_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
        """Get a configured logger for a module."""
        logger = logging.getLogger(module_name)
        if not logger.handlers:
            logger.setLevel(level)
        return logger
```

## `monitoring.py`

```python
"""
Monitoring Module

Provides functionality to track and log metrics related to system performance,
operation success rates, and resource usage.
"""

import json
import logging
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, Any, List, Optional

import psutil
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__, level=logging.INFO)

class OperationMetrics:
    """Tracks metrics for a specific type of operation."""
    
    def __init__(self):
        self.total_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0.0
        self.total_tokens = 0
        self.errors = Counter()

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0

    @property
    def average_duration(self) -> float:
        return (self.total_duration / self.total_count) if self.total_count > 0 else 0.0

class MetricsCollector:
    """Collects and tracks metrics for various operations."""
    
    def __init__(self):
        self.operations = defaultdict(OperationMetrics)
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.start_time = datetime.now()
        logger.info("Metrics collector initialized")

    def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: Optional[float] = None,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        metrics = self.operations[operation_type]
        metrics.total_count += 1
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            if error:
                metrics.errors[error] += 1

        if duration is not None:
            metrics.total_duration += duration
        if tokens_used is not None:
            metrics.total_tokens += tokens_used

        logger.info(f"Tracked {operation_type}: success={success}, duration={duration}")

    def track_cache_hit(self) -> None:
        self.cache_metrics['hits'] += 1
        self.cache_metrics['total_requests'] += 1

    def track_cache_miss(self) -> None:
        self.cache_metrics['misses'] += 1
        self.cache_metrics['total_requests'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            'operations': {},
            'cache': self._get_cache_metrics(),
            'system': SystemMonitor.get_system_metrics(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

        for op_type, op_metrics in self.operations.items():
            metrics['operations'][op_type] = {
                'total_count': op_metrics.total_count,
                'success_count': op_metrics.success_count,
                'failure_count': op_metrics.failure_count,
                'success_rate': op_metrics.success_rate,
                'average_duration': op_metrics.average_duration,
                'total_tokens': op_metrics.total_tokens,
                'common_errors': dict(op_metrics.errors.most_common(5))
            }
        return metrics

    def _get_cache_metrics(self) -> Dict[str, Any]:
        total = self.cache_metrics['total_requests']
        hit_rate = (self.cache_metrics['hits'] / total * 100) if total > 0 else 0
        
        return {
            'hits': self.cache_metrics['hits'],
            'misses': self.cache_metrics['misses'],
            'total_requests': total,
            'hit_rate': round(hit_rate, 2)
        }

    def reset(self) -> None:
        self.operations.clear()
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.start_time = datetime.now()
        logger.info("Metrics collector reset")

class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Counter = Counter()
        logger.info("System monitor initialized")

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

    def track_operation(
        self,
        operation_type: str,
        success: bool,
        duration: Optional[float] = None
    ) -> None:
        if duration is not None:
            self._operation_times[operation_type].append(duration)
            logger.info(f"Tracked {operation_type}: success={success}, duration={duration}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        summary = {
            'system': self.get_system_metrics(),
            'operations': {},
            'errors': dict(self._error_counts.most_common(10)),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

        for op_name, times in self._operation_times.items():
            if times:
                summary['operations'][op_name] = {
                    'count': len(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }

        return summary

    def reset(self) -> None:
        self.start_time = datetime.now()
        self._operation_times.clear()
        self._error_counts.clear()
        logger.info("System monitor reset")

    def export_metrics(self, filepath: str) -> None:
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.reset()
```

## `response_parser.py`

```python
"""
Response Parser Module

Handles parsing and validation of AI model responses with improved separation of concerns
and modular design.
"""

from typing import Dict, Any, Optional, List
import json
from jsonschema import validate, ValidationError
from dataclasses import dataclass
from datetime import datetime
from core.logger import LoggerSetup
from .token_management import TokenManager

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ParsedResponse:
    """Structured representation of a parsed response."""
    docstring: str
    summary: str
    changelog: str
    complexity_score: int
    metadata: Dict[str, Any]

class ResponseValidator:
    """Handles validation of response data."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate_structure(self, data: Dict[str, Any]) -> bool:
        try:
            validate(instance=data, schema=self.schema)
            return True
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def validate_content(self, response: Dict[str, Any]) -> List[str]:
        errors = []
        
        required_fields = ["docstring", "summary", "complexity_score"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
                continue
            
            if field in ["docstring", "summary"] and not response[field].strip():
                errors.append(f"Empty {field}")
            elif field == "complexity_score" and not isinstance(response[field], (int, float)):
                errors.append("Invalid complexity score type")

        if "usage" in response:
            usage_errors = self._validate_usage(response["usage"])
            errors.extend(usage_errors)

        return errors

    def _validate_usage(self, usage: Dict[str, Any]) -> List[str]:
        errors = []
        
        if not isinstance(usage, dict):
            return ["Usage must be a dictionary"]

        required_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
        
        for field in required_fields:
            if field not in usage:
                errors.append(f"Missing usage field: {field}")
                continue
            
            if not isinstance(usage[field], int) or usage[field] < 0:
                errors.append(f"Invalid token count for {field}")

        if all(field in usage for field in required_fields):
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                errors.append("Inconsistent token counts")

        return errors

class ResponseFormatter:
    """Handles formatting of parsed responses."""

    def format(self, response: Dict[str, Any]) -> ParsedResponse:
        try:
            formatted_response = ParsedResponse(
                docstring=response.get("docstring", "").strip(),
                summary=response.get("summary", "").strip(),
                changelog=response.get("changelog", "Initial documentation").strip(),
                complexity_score=response.get("complexity_score", 0),
                metadata=response.get("metadata", {})
            )
            self.validate_formatted_response(formatted_response)
            return formatted_response
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            raise

    def validate_formatted_response(self, response: ParsedResponse) -> None:
        if not response.docstring.strip():
            raise ValueError("Formatted response must have non-empty docstring")
        if not response.summary.strip():
            raise ValueError("Formatted response must have non-empty summary")

class PlainTextParser:
    """Handles parsing of plain text responses."""

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            lines = text.strip().split('\n')
            result = {
                "docstring": "",
                "summary": "",
                "changelog": "Initial documentation",
                "complexity_score": 0,
                "metadata": {}
            }

            current_section = None
            buffer = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.endswith(':') and line[:-1].lower() in result:
                    if current_section and buffer:
                        result[current_section] = self._process_section(
                            current_section, buffer
                        )
                    current_section = line[:-1].lower()
                    buffer = []
                elif current_section:
                    buffer.append(line)

            if current_section and buffer:
                result[current_section] = self._process_section(
                    current_section, buffer
                )

            return result if result["docstring"] and result["summary"] else None

        except Exception as e:
            logger.error(f"Plain text parsing error: {e}")
            return None

    def _process_section(self, section: str, buffer: List[str]) -> Any:
        content = '\n'.join(buffer).strip()
        
        if section == 'complexity_score':
            try:
                return int(content)
            except ValueError:
                logger.error(f"Invalid complexity score: {content}")
                return 0
        
        return content

class ResponseParser:
    """Main response parser with improved modularity."""

    def __init__(
        self,
        token_manager: Optional[TokenManager] = None,
        validator: Optional[ResponseValidator] = None,
        formatter: Optional[ResponseFormatter] = None,
        plain_text_parser: Optional[PlainTextParser] = None
    ):
        self.token_manager = token_manager
        self.validator = validator or ResponseValidator(JSON_SCHEMA)
        self.formatter = formatter or ResponseFormatter()
        self.plain_text_parser = plain_text_parser or PlainTextParser()
        self._error_count = 0
        self._last_error_time = None

    def parse_response(self, response: str) -> Optional[ParsedResponse]:
        try:
            if self.token_manager:
                tokens = self.token_manager.estimate_tokens(response)
                self.token_manager.track_request(0, tokens)

            parsed_data = self._parse_json(response)
            
            if parsed_data is None:
                parsed_data = self.plain_text_parser.parse(response)
                if parsed_data is None:
                    self._track_error("Failed to parse response")
                    return None

            if not self._validate_response(parsed_data):
                self._track_error("Response validation failed")
                return None

            return self.formatter.format(parsed_data)

        except Exception as e:
            self._track_error(f"Response parsing error: {str(e)}")
            logger.error(f"Response parsing error: {e}")
            return None

    def _parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(response, dict):
                return response

            response = response.strip()
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()

            return json.loads(response)

        except json.JSONDecodeError:
            logger.debug("Failed to parse as JSON, trying plain text")
            return None
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return None

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        if not self.validator.validate_structure(response):
            return False

        content_errors = self.validator.validate_content(response)
        if content_errors:
            for error in content_errors:
                logger.error(f"Validation error: {error}")
            return False

        return True

    def _track_error(self, error_message: str) -> None:
        self._error_count += 1
        self._last_error_time = datetime.now()
        logger.error(f"Parser error: {error_message}")

    async def cleanup(self) -> None:
        try:
            if self.token_manager:
                await self.token_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during parser cleanup: {e}")

    def get_error_stats(self) -> Dict[str, Any]:
        return {
            'error_count': self._error_count,
            'last_error_time': self._last_error_time.isoformat() if self._last_error_time else None
        }

def create_response_parser(token_manager: Optional[TokenManager] = None) -> ResponseParser:
    try:
        from schema import JSON_SCHEMA
    except ImportError:
        logger.warning("Could not import JSON schema, using empty schema")
        JSON_SCHEMA = {}
    
    validator = ResponseValidator(JSON_SCHEMA)
    formatter = ResponseFormatter()
    plain_text_parser = PlainTextParser()
    
    return ResponseParser(
        token_manager=token_manager,
        validator=validator,
        formatter=formatter,
        plain_text_parser=plain_text_parser
    )

try:
    from schema import JSON_SCHEMA
except ImportError:
    logger.warning("Could not import JSON schema, using empty schema")
    JSON_SCHEMA = {}
```

These implementations include detailed logging, error handling, and modular design principles to ensure that each component is responsible for a specific aspect of the logging, monitoring, and response parsing processes.  
[Timestamp: 2024/11/22 14:26:00]

**user**: [functions.py](functions.py.md)[classes.py](classes.py.md)[docstring_utils.py](docstring_utils.py.md) [metrics.py](metrics.py.md) process  
[Timestamp: 2024/11/22 14:27:01]

**ai**: Here are the complete implementations for `functions.py`, `classes.py`, `docstring_utils.py`, and `metrics.py`:

## `functions.py`

```python
"""Function extraction module utilizing shared AST analysis utilities."""

from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from .base import BaseExtractor

logger = LoggerSetup.get_logger(__name__)

class FunctionExtractor(BaseExtractor):
    """Extract function definitions using shared AST analysis."""

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract all function definitions from the source code."""
        functions = []
        try:
            for node in self.analyzer.extract_functions(self.tree):
                function_info = self.extract_details(node)
                if function_info:
                    functions.append(function_info)
                    logger.info(f"Extracted function '{node.name}'")
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")
        return functions

    def extract_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract detailed information from a function node."""
        try:
            base_details = self.get_node_details(node)
            function_details = self.analyzer.get_function_details(node)
            
            return {
                **base_details,
                **function_details,
                'is_method': isinstance(getattr(node, 'parent', None), ast.ClassDef),
                'decorators': [ast.unparse(d) for d in node.decorator_list],
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'metrics': {
                    'cognitive_complexity': self.metrics.calculate_cognitive_complexity(node),
                    'halstead': self.metrics.calculate_halstead_metrics(node)
                }
            }
        except Exception as e:
            logger.error(f"Error extracting function details for {getattr(node, 'name', '<unknown>')}: {e}")
            return {}
```

## `classes.py`

```python
"""Class extraction module utilizing shared AST analysis utilities."""

from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from .base import BaseExtractor

logger = LoggerSetup.get_logger(__name__)

class ClassExtractor(BaseExtractor):
    """Extract class definitions using shared AST analysis."""

    def extract_classes(self) -> List[Dict[str, Any]]:
        """Extract all class definitions from the source code."""
        classes = []
        try:
            for node in self.analyzer.extract_classes(self.tree):
                class_info = self.extract_details(node)
                if class_info:
                    classes.append(class_info)
                    logger.info(f"Extracted class '{node.name}'")
        except Exception as e:
            logger.error(f"Error extracting classes: {e}")
        return classes

    def extract_details(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract detailed information from a class node."""
        try:
            base_details = self.get_node_details(node)
            
            return {
                **base_details,
                'bases': [ast.unparse(base) for base in node.bases],
                'methods': self._extract_methods(node),
                'attributes': self._extract_attributes(node),
                'decorators': [ast.unparse(d) for d in node.decorator_list],
                'metrics': {
                    'cognitive_complexity': self.metrics.calculate_cognitive_complexity(node),
                    'halstead': self.metrics.calculate_halstead_metrics(node)
                }
            }
        except Exception as e:
            logger.error(f"Error extracting class details for {getattr(node, 'name', '<unknown>')}: {e}")
            return {}

    def _extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract methods from a class definition."""
        methods = []
        for method_node in (n for n in node.body if isinstance(n, ast.FunctionDef)):
            try:
                method_details = self.analyzer.get_function_details(method_node)
                method_details['metrics'] = {
                    'complexity': self.metrics.calculate_cyclomatic_complexity(method_node),
                    'maintainability': self.metrics.calculate_maintainability_index(method_node)
                }
                methods.append(method_details)
            except Exception as e:
                logger.error(f"Error extracting method details for {method_node.name}: {e}")
        return methods

    def _extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract attributes from a class definition."""
        attributes = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                attributes.append({
                    'name': stmt.target.id if isinstance(stmt.target, ast.Name) else '<unknown>',
                    'type': self.analyzer.get_annotation(stmt.annotation),
                    'has_default': stmt.value is not None
                })
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': 'Any',
                            'has_default': True
                        })
        return attributes
```

## `docstring_utils.py`

```python
"""
Docstring Utilities Module

Provides essential validation and parsing for Python docstrings.
Simplified but maintains core functionality needed by other components.
"""

from typing import Dict, List, Any, Tuple, Optional
import ast
import re
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class DocstringValidator:
    """Validates docstring content and structure."""
    
    def __init__(self):
        """Initialize the validator."""
        self.required_sections = ['summary', 'parameters', 'returns']
        self.min_length = {
            'summary': 10,
            'description': 10
        }

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate docstring content and structure.

        Args:
            docstring_data: Dictionary containing docstring sections

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        for section in self.required_sections:
            if section not in docstring_data:
                errors.append(f"Missing required section: {section}")
                continue

        if 'summary' in docstring_data:
            if len(docstring_data['summary'].strip()) < self.min_length['summary']:
                errors.append("Summary too short (minimum 10 characters)")

        if 'parameters' in docstring_data:
            param_errors = self._validate_parameters(docstring_data['parameters'])
            errors.extend(param_errors)

        if 'returns' in docstring_data:
            return_errors = self._validate_return(docstring_data['returns'])
            errors.extend(return_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            logger.error(f"Docstring validation failed: {errors}")
        
        return is_valid, errors

    def _validate_parameters(self, parameters: List[Dict[str, Any]]) -> List[str]:
        errors = []
        
        if not isinstance(parameters, list):
            return ["Parameters must be provided as a list"]

        for param in parameters:
            if not isinstance(param, dict):
                errors.append("Invalid parameter format")
                continue

            if 'name' not in param:
                errors.append("Parameter missing name")
                continue

            param_name = param.get('name', '')
            
            if 'type' not in param:
                errors.append(f"Parameter {param_name} missing type")

            if 'description' not in param:
                errors.append(f"Parameter {param_name} missing description")
            elif len(param.get('description', '').strip()) < self.min_length['description']:
                errors.append(f"Parameter {param_name} description too short")

        return errors

    def _validate_return(self, returns: Dict[str, Any]) -> List[str]:
        errors = []

        if not isinstance(returns, dict):
            return ["Return value must be provided as a dictionary"]

        if 'type' not in returns:
            errors.append("Return missing type")

        if 'description' not in returns:
            errors.append("Return missing description")
        elif len(returns.get('description', '').strip()) < self.min_length['description']:
            errors.append("Return description too short")

        return errors

def parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse a docstring into structured sections.

    Args:
        docstring: Raw docstring text

    Returns:
        Dict[str, Any]: Parsed docstring sections
    """
    if not docstring:
        return {
            "docstring": "",
            "summary": "",
            "parameters": [],
            "returns": {"type": "None", "description": "No return value."}
        }

    sections = {
        "docstring": docstring.strip(),
        "summary": "",
        "parameters": [],
        "returns": {"type": "None", "description": "No return value."}
    }

    lines = docstring.split('\n')
    current_section = 'summary'
    current_content = []

    for line in lines:
        line = line.strip()
        
        if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'returns:', 'raises:')):
            if current_content:
                if current_section == 'summary':
                    sections['summary'] = '\n'.join(current_content).strip()
                current_content = []

            section_name = line.lower().split(':')[0]
            if section_name in ('args', 'arguments', 'parameters'):
                current_section = 'parameters'
            else:
                current_section = section_name

        elif line:
            current_content.append(line)

    if current_content:
        if current_section == 'summary':
            sections['summary'] = '\n'.join(current_content).strip()
        elif current_section == 'parameters':
            sections['parameters'] = _parse_parameters('\n'.join(current_content))
        elif current_section == 'returns':
            sections['returns'] = _parse_return('\n'.join(current_content))

    return sections

def _parse_parameters(params_str: str) -> List[Dict[str, Any]]:
    params = []
    current_param = None

    for line in params_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        if not line.startswith(' '):
            if ':' in line:
                name, rest = line.split(':', 1)
                current_param = {
                    "name": name.strip(),
                    "type": "Any",
                    "description": rest.strip()
                }
                params.append(current_param)
        elif current_param:
            current_param["description"] = f"{current_param['description']} {line}"

    return params

def _parse_return(return_str: str) -> Dict[str, str]:
    return_info = {
        "type": "None",
        "description": "No return value."
    }

    if ':' in return_str:
        type_str, desc = return_str.split(':', 1)
        return_info.update({
            "type": type_str.strip(),
            "description": desc.strip()
        })
    else:
        return_info["description"] = return_str.strip()

    return return_info
```

## `metrics.py`

```python
"""
Code Metrics Analysis Module

This module provides comprehensive code quality and complexity metrics for Python source code,
including cyclomatic complexity, cognitive complexity, Halstead metrics, and code quality analysis.

Version: 1.1.0
Author: Development Team
"""

import ast
import math
from collections import defaultdict
from typing import Dict, List, Set
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class Metrics:
    """
    Provides methods to calculate different complexity metrics for Python functions.

    This class includes methods for calculating cyclomatic complexity, cognitive complexity,
    Halstead metrics, and maintainability index. It also provides functionality to analyze
    module dependencies.
    """

    MAINTAINABILITY_THRESHOLDS = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    @staticmethod
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cyclomatic complexity of a function.

        Parameters:
            function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
            int: The cyclomatic complexity of the function.
        """
        logger.debug(f"Calculating cyclomatic complexity for function: {getattr(function_node, 'name', 'unknown')}")
        if not isinstance(function_node, ast.FunctionDef):
            logger.error(f"Provided node is not a function definition: {ast.dump(function_node)}")
            return 0

        complexity = 1  # Start with 1 for the function itself
        decision_points = (
            ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler,
            ast.With, ast.Try, ast.BoolOp, ast.Lambda, ast.ListComp, ast.DictComp,
            ast.SetComp, ast.GeneratorExp, ast.IfExp, ast.Match  # For Python 3.10+
        )

        for node in ast.walk(function_node):
            if isinstance(node, decision_points):
                if isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                    logger.debug(f"Incremented complexity for BoolOp with {len(node.values) - 1} decision points: {ast.dump(node)}")
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    complexity += 1
                    logger.debug(f"Incremented complexity for comprehension: {ast.dump(node)}")
                elif isinstance(node, ast.Match):
                    complexity += len(node.cases)
                    logger.debug(f"Incremented complexity for Match with {len(node.cases)} cases: {ast.dump(node)}")
                else:
                    complexity += 1
                    logger.debug(f"Incremented complexity at node: {ast.dump(node)}")

        logger.info(f"Calculated cyclomatic complexity for function '{function_node.name}' is {complexity}")
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cognitive complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cognitive complexity of the function.
        """
        logger.debug(f"Calculating cognitive complexity for function: {getattr(function_node, 'name', 'unknown')}")
        if not isinstance(function_node, ast.FunctionDef):
            logger.error(f"Provided node is not a function definition: {ast.dump(function_node)}")
            return 0

        cognitive_complexity = 0
        nesting_depth = 0
        prev_node = None

        for node in ast.walk(function_node):
            if Metrics._is_nesting_construct(node):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
                logger.debug(f"Nesting depth increased to {nesting_depth} at node: {ast.dump(node)}")
            elif prev_node is not None and Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
                logger.debug(f"Incremented cognitive complexity at node: {ast.dump(node)}")
            prev_node = node

        logger.info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")
        return cognitive_complexity

    def calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate the overall complexity of the given AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            int: The overall complexity score.
        """
        logger.debug("Calculating overall complexity.")
        if not isinstance(node, ast.FunctionDef):
            logger.error(f"Provided node is not a function definition: {ast.dump(node)}")
            return 0
        cyclomatic_complexity = self.calculate_cyclomatic_complexity(node)
        cognitive_complexity = self.calculate_cognitive_complexity(node)
        overall_complexity = cyclomatic_complexity + cognitive_complexity
        logger.info(f"Calculated overall complexity for function '{node.name}' is {overall_complexity}")
        return overall_complexity

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index based on various metrics.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            float: Maintainability index score (0-100)
        """
        logger.debug("Calculating maintainability index.")
        try:
            halstead = self.calculate_halstead_metrics(node)
            complexity = self.calculate_complexity(node)
            sloc = self._count_source_lines(node)
            
            volume = halstead['program_volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, min(100, mi))
            
            logger.info(f"Calculated maintainability index is {mi}")
            return round(mi, 2)
            
        except Exception as e:
            logger.error(f"Error calculating maintainability index: {e}")
            return 0.0

    def calculate_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """
        Calculate Halstead metrics for the given AST node.

        Args:
            node (ast.AST): The AST node to analyze.

        Returns:
            Dict[str, float]: A dictionary containing Halstead metrics.
        """
        logger.debug("Calculating Halstead metrics.")
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        operator_nodes = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
                          ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
                          ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                          ast.In, ast.NotIn, ast.Call, ast.Attribute, ast.Subscript, ast.Index, ast.Slice)
        operand_nodes = (ast.Num, ast.Str, ast.Bytes, ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Constant)

        for n in ast.walk(node):
            if isinstance(n, operator_nodes):
                operators.add(type(n).__name__)
                operator_count += 1
            elif isinstance(n, operand_nodes):
                operands.add(type(n).__name__)
                operand_count += 1

        n1 = len(operators)
        n2 = len(operands)
        N1 = operator_count
        N2 = operand_count

        program_length = N1 + N2
        program_vocabulary = n1 + n2
        program_volume = program_length * math.log2(program_vocabulary) if program_vocabulary > 0 else 0

        logger.info(f"Calculated Halstead metrics: Length={program_length}, Vocabulary={program_vocabulary}, Volume={program_volume}")
        return {
            'program_length': program_length,
            'program_vocabulary': program_vocabulary,
            'program_volume': program_volume
        }

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            int: Number of source code lines
        """
        logger.debug("Counting source lines of code.")
        try:
            source = ast.unparse(node)
            lines = [line.strip() for line in source.splitlines()]
            count = len([line for line in lines if line and not line.startswith('#')])
            logger.info(f"Counted {count} source lines of code.")
            return count
        except Exception as e:
            logger.error(f"Error counting source lines: {e}")
            return 0

    @staticmethod
    def _is_decision_point(node: ast.AST) -> bool:
        decision_point = isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler))
        logger.debug(f"Node {ast.dump(node)} is {'a' if decision_point else 'not a'} decision point.")
        return decision_point

    @staticmethod
    def _is_nesting_construct(node: ast.AST) -> bool:
        nesting_construct = isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
        logger.debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} nesting construct.")
        return nesting_construct

    @staticmethod
    def _is_complexity_increment(node: ast.AST, prev_node: ast.AST) -> bool:
        increment = isinstance(node, (ast.BoolOp, ast.Compare)) and not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or isinstance(node, (ast.Continue, ast.Break, ast.Raise, ast.Return))
        logger.debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")
        return increment

    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """
        Analyze module dependencies and imports.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            Dict[str, Set[str]]: Dictionary of module dependencies
        """
        logger.debug("Analyzing module dependencies.")
        deps = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }
        
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)
            logger.info(f"Analyzed dependencies: {deps}")
            return deps
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            return deps

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        logger.debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
        except Exception as e:
            logger.error(f"Error processing import: {e}")

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        logger.debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.stdlib_module_names:
                deps['stdlib'].add(module_name)
            elif '.' in module_name:
                deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)
        except Exception as e:
            logger.error(f"Error categorizing import {module_name}: {e}")

def test_metrics():
    """
    Test function for the Metrics class.

    This function tests the calculation of cyclomatic and cognitive complexity
    for a sample function defined in source_code.
    """
    logger.info("Starting test_metrics.")
    source_code = """
def example_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            else:
                continue
    else:
        return -1
    return 0
"""
    tree = ast.parse(source_code)
    function_node = tree.body[0]

    if isinstance(function_node, ast.FunctionDef):
        cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)
        assert cyclomatic_complexity == 4, f"Expected 4, got {cyclomatic_complexity}"

        cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)
        assert cognitive_complexity == 11, f"Expected 11, got {cognitive_complexity}"

        logger.info("All tests passed.")
    else:
        logger.error("The node is not a function definition.")

if __name__ == "__main__":
    test_metrics()
```

These implementations provide detailed logging, error handling, and modular design principles to ensure that each component is responsible for a specific aspect of function and class extraction, docstring parsing, and metrics calculation.  
[Timestamp: 2024/11/22 14:28:32]

**user**: [ast_analysis.py](ast_analysis.py.md)  
[Timestamp: 2024/11/22 14:28:40]

**ai**: Here's the complete implementation of `ast_analysis.py`, which provides utilities for analyzing Python Abstract Syntax Tree (AST) nodes:

```python
"""AST Analysis Module - Provides utilities for analyzing Python AST nodes."""

import ast
from typing import List, Dict, Any, Optional
from core.logger import LoggerSetup
from core.metrics_calculator import MetricsCalculator

logger = LoggerSetup.get_logger(__name__)

class ASTAnalyzer:
    """Provides utilities for analyzing Python AST nodes."""

    def __init__(self, metrics_calculator: Optional[MetricsCalculator] = None):
        """
        Initialize AST analyzer.

        Args:
            metrics_calculator: Optional metrics calculator for code analysis
        """
        self.metrics = metrics_calculator or MetricsCalculator()

    def parse_source_code(self, source_code: str) -> ast.AST:
        """
        Parse source code into an AST.

        Args:
            source_code: Source code to parse

        Returns:
            ast.AST: Abstract Syntax Tree

        Raises:
            SyntaxError: If source code has syntax errors
        """
        try:
            return ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise

    @staticmethod
    def extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:
        """Extract function definitions from an AST."""
        return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    @staticmethod
    def extract_classes(tree: ast.AST) -> List[ast.ClassDef]:
        """Extract class definitions from an AST."""
        return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    @staticmethod
    def extract_docstring(node: ast.AST) -> Optional[str]:
        """Extract the docstring from an AST node."""
        try:
            return ast.get_docstring(node)
        except Exception as e:
            logger.error(f"Failed to extract docstring: {e}")
            return None

    @staticmethod
    def get_function_details(node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract details from a function definition node.

        Args:
            node: Function definition node

        Returns:
            Dict containing function details
        """
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'parameters': [(arg.arg, ASTAnalyzer.get_annotation(arg.annotation)) 
                          for arg in node.args.args],
            'return_type': ASTAnalyzer.get_annotation(node.returns)
        }

    @staticmethod
    def get_annotation(node: Optional[ast.AST]) -> str:
        """Get the annotation of an AST node."""
        if node is None:
            return "Any"
        try:
            return ast.unparse(node)
        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
            return "Any"

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """Add parent node information to each node in an AST."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)
```

## Key Features

- **Parsing Source Code:** The `parse_source_code` method converts a string of Python code into an AST, handling syntax errors gracefully.
- **Node Extraction:** Provides methods to extract function and class definitions from an AST, which are essential for analyzing and processing Python code.
- **Docstring Extraction:** Offers functionality to extract docstrings from AST nodes, useful for documentation generation.
- **Function Details:** Extracts detailed information from function nodes, including name, parameters, and return type.
- **Annotations:** Retrieves type annotations from AST nodes, defaulting to "Any" if none are present or if an error occurs.
- **Parent Node Information:** Adds parent references to each node in the AST, facilitating navigation and analysis of the tree structure.

This module is designed to be a foundational component for tools that analyze Python code, providing essential utilities for working with ASTs.  
[Timestamp: 2024/11/22 14:28:56]

**user**: Identify redundant code and placeholders, missed calls in methods trying to integrate etc  
[Timestamp: 2024/11/22 14:29:35]

**ai**: Here's an analysis of potential issues and improvements needed across the codebase:

1. **Redundant Code and Integration Issues:**

```python
# metrics.py and metrics_calculator.py have overlapping functionality
# Should consolidate into one comprehensive metrics module
class Metrics:
    # Duplicates functionality from MetricsCalculator
    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        # Similar implementation in both classes
        pass

# docstring_utils.py and docs.py have overlapping functionality
# Should consolidate docstring handling logic
```

2. **Missing Integration Points:**

```python
# ast_analysis.py needs integration with metrics
class ASTAnalyzer:
    def get_function_details(node: ast.FunctionDef) -> Dict[str, Any]:
        # Should include metrics calculation
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'parameters': [(arg.arg, ASTAnalyzer.get_annotation(arg.annotation)) 
                          for arg in node.args.args],
            'return_type': ASTAnalyzer.get_annotation(node.returns),
            # Missing metrics integration
            'metrics': self.metrics.calculate_all_metrics(node)  # Should add this
        }
```

3. **Incomplete Error Handling:**

```python
# response_parser.py needs better error handling integration with monitoring
class ResponseParser:
    def parse_response(self, response: str) -> Optional[ParsedResponse]:
        try:
            # Missing metrics tracking
            if self.token_manager:
                tokens = self.token_manager.estimate_tokens(response)
                self.token_manager.track_request(0, tokens)
                # Should add metrics tracking here
                self.metrics_collector.track_operation(
                    operation_type='parse_response',
                    success=True,
                    tokens_used=tokens
                )
```

4. **Missing Base Classes:**

```python
# Need a base extractor class for shared functionality between FunctionExtractor and ClassExtractor
class BaseExtractor:
    """Base class for code extractors."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.analyzer = ASTAnalyzer()
        
    def get_node_details(self, node: ast.AST) -> Dict[str, Any]:
        """Get common node details."""
        return {
            'name': getattr(node, 'name', ''),
            'docstring': self.analyzer.extract_docstring(node),
            'line_number': node.lineno,
            'end_line_number': node.end_lineno
        }
```

5. **Inconsistent Monitoring Integration:**

```python
# monitoring.py needs better integration with metrics collection
class MetricsCollector:
    def track_code_quality(self, source_code: str, context: str) -> None:
        # Missing integration with Metrics class
        try:
            metrics = Metrics()
            tree = ast.parse(source_code)
            quality_metrics = {
                'maintainability': metrics.calculate_maintainability_index(tree),
                'complexity': metrics.calculate_complexity(tree),
                'dependencies': metrics.analyze_dependencies(tree)
            }
            self.metrics_history[context].append({
                'timestamp': datetime.now(),
                'metrics': quality_metrics
            })
        except Exception as e:
            logger.error(f"Error tracking code quality: {e}")
```

6. **Missing Configuration Integration:**

```python
# Need to add configuration integration across components
class ASTAnalyzer:
    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None,
        config: Optional[Config] = None  # Should add this
    ):
        self.metrics = metrics_calculator or MetricsCalculator()
        self.config = config or Config()  # Should add this
```

7. **Incomplete Documentation Processing:**

```python
# docs.py needs better integration with markdown generation
class DocStringManager:
    def generate_documentation(
        self,
        module_path: Optional[str] = None,
        include_source: bool = True
    ) -> str:
        # Missing integration with metrics and quality checks
        try:
            generator = MarkdownDocumentationGenerator(
                source_code=self.source_code if include_source else None,
                module_path=module_path,
                # Should add metrics and quality information
                metrics=self.metrics.get_metrics_summary(),
                quality_score=self.calculate_quality_score()
            )
```

8. **Missing Test Cases:**

```python
# Need comprehensive test cases for each component
def test_metrics():
    """Test function for the Metrics class."""
    # Current test is too basic
    # Should add tests for:
    # - Complex nested functions
    # - Class methods
    # - Async functions
    # - Error cases
    # - Edge cases
```

9. **Cleanup and Resource Management:**

```python
# Need consistent cleanup across components
class AsyncComponentManager:
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup components in reverse order."""
        cleanup_order = [
            "ai_handler",
            "cache",
            "ast_analyzer",
            "metrics_calculator",
            "metrics"
        ]
        # Should add error aggregation and reporting
        errors = []
        for component_name in cleanup_order:
            try:
                await self._cleanup_component(component_name)
            except Exception as e:
                errors.append(f"{component_name}: {str(e)}")
```

10. **Configuration Management:**

```python
# Need a unified configuration system
class Config:
    """Central configuration management."""
    def __init__(self):
        self.metrics_config = {
            'maintainability_threshold': 65.0,
            'complexity_threshold': 10,
            'enable_detailed_metrics': True
        }
        self.logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        # Add more configuration sections
```

These issues should be addressed to improve code quality, maintainability, and integration between components. The main focus should be on:
- Consolidating duplicate functionality
- Improving error handling and monitoring
- Adding missing integration points
- Implementing proper base classes
- Adding comprehensive configuration management
- Improving test coverage
- Ensuring consistent resource cleanup  
[Timestamp: 2024/11/22 14:30:05]

**user**: More  
[Timestamp: 2024/11/22 14:30:56]

**ai**: Here are more detailed issues and improvements needed:

11. **Missing Event System:**

```python
# Need a central event system for component communication
class EventSystem:
    """Central event system for component communication."""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.metrics_collector = MetricsCollector()
        
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        timestamp = datetime.now()
        for subscriber in self.subscribers[event_type]:
            try:
                await subscriber(data, timestamp)
                self.metrics_collector.track_event(event_type, True)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
                self.metrics_collector.track_event(event_type, False, error=str(e))

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        self.subscribers[event_type].append(handler)
```

12. **Incomplete Cache Implementation:**

```python
# cache.py needs better invalidation and optimization
class Cache:
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        self.stats = defaultdict(int)
        self.cleanup_threshold = 1000  # Add cleanup threshold
        self._last_cleanup = datetime.now()

    async def cleanup_expired(self) -> None:
        """Cleanup expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (value, timestamp) in self.cache.items()
            if (now - timestamp).total_seconds() > self.ttl
        ]
        for key in expired_keys:
            await self.invalidate(key)
            self.stats['expired'] += 1

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic cleanup."""
        if len(self.cache) > self.cleanup_threshold:
            await self.cleanup_expired()
        # Rest of the implementation...
```

13. **Missing Rate Limiting:**

```python
# Need rate limiting for API calls
class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a rate limit token."""
        async with self._lock:
            now = time.time()
            # Remove expired timestamps
            while self.timestamps and now - self.timestamps[0] > self.period:
                self.timestamps.popleft()
            
            if len(self.timestamps) < self.calls:
                self.timestamps.append(now)
                return True
            return False
```

14. **Incomplete Error Recovery:**

```python
# Need better error recovery mechanisms
class ErrorRecovery:
    """Handles error recovery and retry logic."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.backoff_factor = 1.5
        self.errors = defaultdict(list)

    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                await self._handle_error(e, attempt)
                await asyncio.sleep(self.backoff_factor ** attempt)
        raise last_error
```

15. **Missing Progress Tracking:**

```python
# Need better progress tracking
class ProgressTracker:
    """Tracks progress of long-running operations."""
    
    def __init__(self):
        self.operations = {}
        self.listeners = []

    async def update_progress(
        self,
        operation_id: str,
        current: int,
        total: int,
        status: str
    ) -> None:
        """Update operation progress."""
        progress = {
            'operation_id': operation_id,
            'current': current,
            'total': total,
            'percentage': (current / total) * 100 if total > 0 else 0,
            'status': status,
            'timestamp': datetime.now()
        }
        self.operations[operation_id] = progress
        await self._notify_listeners(progress)
```

16. **Incomplete Dependency Management:**

```python
# Need better dependency management
class DependencyManager:
    """Manages component dependencies and initialization order."""
    
    def __init__(self):
        self.dependencies = {}
        self.initialized = set()
        self._initialization_order = []

    async def register(
        self,
        component: str,
        dependencies: List[str],
        initializer: Callable
    ) -> None:
        """Register a component and its dependencies."""
        self.dependencies[component] = {
            'deps': dependencies,
            'init': initializer
        }
        await self._update_initialization_order()

    async def initialize_all(self) -> None:
        """Initialize all components in correct order."""
        for component in self._initialization_order:
            if component not in self.initialized:
                await self._initialize_component(component)
```

17. **Missing Performance Monitoring:**

```python
# Need comprehensive performance monitoring
class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85,
            'response_time': 1.0
        }

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'open_files': len(psutil.Process().open_files()),
            'threads': len(psutil.Process().threads()),
            'connections': len(psutil.Process().connections())
        }
        await self._check_thresholds(metrics)
        return metrics
```

18. **Incomplete Documentation Generation:**

```python
# Need better documentation generation
class DocumentationGenerator:
    """Generates comprehensive documentation."""
    
    def __init__(self):
        self.templates = {}
        self.processors = {}
        self.output_formats = ['md', 'html', 'pdf']

    async def generate(
        self,
        source: str,
        format: str = 'md',
        include_metrics: bool = True
    ) -> str:
        """Generate documentation in specified format."""
        if format not in self.output_formats:
            raise ValueError(f"Unsupported format: {format}")

        processed_content = await self._process_content(source)
        if include_metrics:
            metrics = await self._calculate_metrics(source)
            processed_content['metrics'] = metrics

        template = self.templates.get(format)
        if not template:
            raise ValueError(f"No template found for format: {format}")

        return await self._render_template(template, processed_content)
```

19. **Missing Queue Management:**

```python
# Need better queue management for async operations
class QueueManager:
    """Manages operation queues and processing."""
    
    def __init__(self, max_size: int = 1000):
        self.queues = {}
        self.max_size = max_size
        self.processors = {}
        self.stats = defaultdict(int)

    async def add_task(
        self,
        queue_name: str,
        task: Dict[str, Any],
        priority: int = 0
    ) -> None:
        """Add task to specified queue."""
        if queue_name not in self.queues:
            self.queues[queue_name] = PriorityQueue(maxsize=self.max_size)
        
        await self.queues[queue_name].put((priority, task))
        self.stats['enqueued'] += 1
```

20. **Incomplete Validation System:**

```python
# Need comprehensive validation system
class ValidationSystem:
    """Handles validation of inputs and outputs."""
    
    def __init__(self):
        self.validators = {}
        self.schemas = {}
        self.error_handlers = {}

    async def validate(
        self,
        data: Any,
        schema_name: str,
        context: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """Validate data against schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")

        schema = self.schemas[schema_name]
        validator = self.validators.get(schema_name)
        if not validator:
            raise ValueError(f"No validator for schema: {schema_name}")

        try:
            result = await validator(data, schema, context)
            return True, []
        except ValidationError as e:
            errors = self._format_errors(e)
            await self._handle_validation_errors(errors, context)
            return False, errors
```

These components need to be integrated into the existing codebase with proper error handling, monitoring, and configuration management. Each component should also have corresponding test cases and documentation.  
[Timestamp: 2024/11/22 14:31:40]

**user**: [base.py](DocuScribe/base.py.md) modify with # Need a base extractor class for shared functionality between FunctionExtractor and ClassExtractor  
class BaseExtractor:  
    """Base class for code extractors."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.analyzer = ASTAnalyzer()
        
    def get_node_details(self, node: ast.AST) -> Dict[str, Any]:
        """Get common node details."""
        return {
            'name': getattr(node, 'name', ''),
            'docstring': self.analyzer.extract_docstring(node),
            'line_number': node.lineno,
            'end_line_number': node.end_lineno
        }
[Timestamp: 2024/11/22 14:42:32]

**ai**: Here's an improved version of `base.py` that incorporates better shared functionality and error handling:

```python
"""Base extraction module utilizing shared AST analysis utilities."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import ast
from dataclasses import dataclass
from datetime import datetime
from core.logger import LoggerSetup
from utils.ast_analysis import ASTAnalyzer
from core.metrics_calculator import MetricsCalculator
from core.monitoring import MetricsCollector

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ExtractionMetrics:
    """Metrics for extraction operations."""
    start_time: datetime
    end_time: Optional[datetime] = None
    processed_nodes: int = 0
    error_count: int = 0
    complexity_sum: int = 0
    maintainability_sum: float = 0.0

    @property
    def duration(self) -> float:
        """Calculate operation duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def average_complexity(self) -> float:
        """Calculate average complexity."""
        return self.complexity_sum / self.processed_nodes if self.processed_nodes > 0 else 0.0

    @property
    def average_maintainability(self) -> float:
        """Calculate average maintainability."""
        return self.maintainability_sum / self.processed_nodes if self.processed_nodes > 0 else 0.0

class BaseExtractor(ABC):
    """Base class for extracting information from AST nodes."""

    def __init__(
        self,
        source_code: str,
        metrics_collector: Optional[MetricsCollector] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize with source code and shared utilities.

        Args:
            source_code: Source code to analyze
            metrics_collector: Optional metrics collector for monitoring
            metrics_calculator: Optional metrics calculator for code analysis
        """
        self.metrics_collector = metrics_collector
        self.analyzer = ASTAnalyzer(metrics_calculator=metrics_calculator)
        self.metrics = metrics_calculator or MetricsCalculator()
        self.extraction_metrics = ExtractionMetrics(start_time=datetime.now())
        self.parse_source(source_code)

    def parse_source(self, source_code: str) -> None:
        """
        Parse source code into an AST.

        Args:
            source_code: Source code to parse

        Raises:
            SyntaxError: If source code contains syntax errors
        """
        try:
            self.tree = self.analyzer.parse_source_code(source_code)
            self.analyzer.add_parent_info(self.tree)
            logger.info("Source code successfully parsed into AST")
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            self.extraction_metrics.error_count += 1
            if self.metrics_collector:
                self.metrics_collector.track_operation(
                    operation_type='parse_source',
                    success=False,
                    error=str(e)
                )
            raise
        except Exception as e:
            logger.error(f"Failed to parse source code: {e}")
            self.extraction_metrics.error_count += 1
            if self.metrics_collector:
                self.metrics_collector.track_operation(
                    operation_type='parse_source',
                    success=False,
                    error=str(e)
                )
            raise

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """
        Extract docstring using the analyzer.

        Args:
            node: AST node to extract docstring from

        Returns:
            Optional[str]: Extracted docstring if available
        """
        try:
            return self.analyzer.extract_docstring(node)
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            self.extraction_metrics.error_count += 1
            return None

    def get_node_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Get common details for a node.

        Args:
            node: AST node to analyze

        Returns:
            Dict[str, Any]: Node details including metrics
        """
        try:
            complexity = self.metrics.calculate_cyclomatic_complexity(node)
            maintainability = self.metrics.calculate_maintainability_index(node)
            
            self.extraction_metrics.processed_nodes += 1
            self.extraction_metrics.complexity_sum += complexity
            self.extraction_metrics.maintainability_sum += maintainability

            return {
                'name': getattr(node, 'name', '<unknown>'),
                'docstring': self.extract_docstring(node),
                'lineno': getattr(node, 'lineno', 0),
                'end_lineno': getattr(node, 'end_lineno', 0),
                'col_offset': getattr(node, 'col_offset', 0),
                'end_col_offset': getattr(node, 'end_col_offset', 0),
                'metrics': {
                    'complexity': complexity,
                    'maintainability': maintainability,
                    'cognitive_complexity': self.metrics.calculate_cognitive_complexity(node),
                    'halstead': self.metrics.calculate_halstead_metrics(node)
                }
            }
        except Exception as e:
            logger.error(f"Error getting node details: {e}")
            self.extraction_metrics.error_count += 1
            return self._get_default_node_details()

    def _get_default_node_details(self) -> Dict[str, Any]:
        """Get default node details for error cases."""
        return {
            'name': '<unknown>',
            'docstring': None,
            'lineno': 0,
            'end_lineno': 0,
            'col_offset': 0,
            'end_col_offset': 0,
            'metrics': {
                'complexity': 0,
                'maintainability': 0,
                'cognitive_complexity': 0,
                'halstead': {
                    'program_length': 0,
                    'program_vocabulary': 0,
                    'program_volume': 0
                }
            }
        }

    def get_extraction_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the extraction operation.

        Returns:
            Dict[str, Any]: Extraction metrics
        """
        self.extraction_metrics.end_time = datetime.now()
        return {
            'duration': self.extraction_metrics.duration,
            'processed_nodes': self.extraction_metrics.processed_nodes,
            'error_count': self.extraction_metrics.error_count,
            'average_complexity': self.extraction_metrics.average_complexity,
            'average_maintainability': self.extraction_metrics.average_maintainability
        }

    @abstractmethod
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Abstract method to extract specific details from a node.

        Args:
            node: AST node to extract details from

        Returns:
            Dict[str, Any]: Extracted details
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources and log final metrics."""
        try:
            metrics = self.get_extraction_metrics()
            if self.metrics_collector:
                self.metrics_collector.track_operation(
                    operation_type='extraction',
                    success=self.extraction_metrics.error_count == 0,
                    duration=metrics['duration']
                )
            logger.info(f"Extraction completed with metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
```

Key improvements in this version:

1. Added `ExtractionMetrics` dataclass for tracking operation metrics
2. Improved error handling and logging
3. Added metrics collection integration
4. Added detailed node information including column offsets
5. Added cleanup method for resource management
6. Added comprehensive metrics calculation and tracking
7. Improved docstrings and type hints
8. Added default node details for error cases
9. Added method to get extraction metrics
10. Better integration with monitoring system

This base class now provides more robust shared functionality for both function and class extractors while maintaining proper separation of concerns and error handling.  
[Timestamp: 2024/11/22 14:43:19]

**user**: No  
[Timestamp: 2024/11/22 14:43:40]

**ai**: Here's a simpler, more focused version of `base.py`:

```python
"""Base extraction module utilizing shared AST analysis utilities."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import ast
from core.logger import LoggerSetup
from utils.ast_analysis import ASTAnalyzer
from core.metrics_calculator import MetricsCalculator

logger = LoggerSetup.get_logger(__name__)

class BaseExtractor(ABC):
    """Base class for code extractors."""
    
    def __init__(self, source_code: str):
        """Initialize with source code and shared utilities."""
        self.source_code = source_code
        self.analyzer = ASTAnalyzer()
        self.metrics = MetricsCalculator()
        self.tree = self._parse_source()

    def _parse_source(self) -> ast.AST:
        """Parse source code into AST."""
        try:
            tree = ast.parse(self.source_code)
            self.analyzer.add_parent_info(tree)
            return tree
        except Exception as e:
            logger.error(f"Failed to parse source code: {e}")
            raise

    def get_node_details(self, node: ast.AST) -> Dict[str, Any]:
        """Get common node details."""
        try:
            return {
                'name': getattr(node, 'name', ''),
                'docstring': self.analyzer.extract_docstring(node),
                'line_number': node.lineno,
                'end_line_number': node.end_lineno,
                'metrics': {
                    'complexity': self.metrics.calculate_cyclomatic_complexity(node),
                    'maintainability': self.metrics.calculate_maintainability_index(node)
                }
            }
        except Exception as e:
            logger.error(f"Error getting node details: {e}")
            return {
                'name': '',
                'docstring': None,
                'line_number': 0,
                'end_line_number': 0,
                'metrics': {'complexity': 0, 'maintainability': 0}
            }

    @abstractmethod
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """Extract specific details from a node. To be implemented by subclasses."""
        pass
```

This version:
1. Keeps the essential shared functionality
2. Removes unnecessary complexity
3. Focuses on core extraction needs
4. Maintains basic error handling
5. Provides common metrics
6. Uses clear, simple interface

The subclasses (FunctionExtractor and ClassExtractor) can then implement their specific extraction logic while inheriting this common functionality.  
[Timestamp: 2024/11/22 14:43:58]

**user**: Now check and enhance through simplification [functions.py](functions.py.md)[classes.py](classes.py.md) and [extraction_manager.py](extraction_manager.py.md) and [ast_analysis.py](ast_analysis.py.md)  
[Timestamp: 2024/11/22 14:45:18]

**ai**: Here's a simplified and enhanced version of these files:

1. `ast_analysis.py` (Simplified):

```python
"""AST Analysis Module - Core utilities for Python AST analysis."""

import ast
from typing import List, Dict, Any, Optional
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class ASTAnalyzer:
    """Core AST analysis utilities."""

    @staticmethod
    def parse_source(source_code: str) -> ast.AST:
        """Parse source code into AST."""
        try:
            tree = ast.parse(source_code)
            ASTAnalyzer.add_parent_info(tree)
            return tree
        except Exception as e:
            logger.error(f"Parse error: {e}")
            raise

    @staticmethod
    def get_docstring(node: ast.AST) -> Optional[str]:
        """Get node docstring."""
        return ast.get_docstring(node)

    @staticmethod
    def get_annotation(node: Optional[ast.AST]) -> str:
        """Get type annotation."""
        if node is None:
            return "Any"
        return ast.unparse(node)

    @staticmethod
    def add_parent_info(tree: ast.AST) -> None:
        """Add parent references to AST nodes."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, 'parent', parent)

    @staticmethod
    def get_function_info(node: ast.FunctionDef) -> Dict[str, Any]:
        """Get function information."""
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'args': [(arg.arg, ASTAnalyzer.get_annotation(arg.annotation)) 
                    for arg in node.args.args],
            'returns': ASTAnalyzer.get_annotation(node.returns),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [ast.unparse(d) for d in node.decorator_list]
        }
```

2. `base.py` (Simplified):

```python
"""Base extractor for code analysis."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import ast
from core.logger import LoggerSetup
from utils.ast_analysis import ASTAnalyzer
from core.metrics_calculator import MetricsCalculator

logger = LoggerSetup.get_logger(__name__)

class BaseExtractor(ABC):
    """Base code extractor."""
    
    def __init__(self, source_code: str):
        self.tree = ASTAnalyzer.parse_source(source_code)
        self.metrics = MetricsCalculator()

    def get_node_info(self, node: ast.AST) -> Dict[str, Any]:
        """Get basic node information."""
        return {
            'name': getattr(node, 'name', ''),
            'docstring': ASTAnalyzer.get_docstring(node),
            'line': node.lineno,
            'end_line': node.end_lineno,
            'metrics': {
                'complexity': self.metrics.calculate_cyclomatic_complexity(node),
                'maintainability': self.metrics.calculate_maintainability_index(node)
            }
        }

    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]:
        """Extract code elements."""
        pass
```

3. `functions.py` (Simplified):

```python
"""Function extractor module."""

from typing import Dict, Any, List
import ast
from .base import BaseExtractor

class FunctionExtractor(BaseExtractor):
    """Function definition extractor."""

    def extract(self) -> List[Dict[str, Any]]:
        """Extract all functions."""
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(self.extract_function(node))
        return functions

    def extract_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract single function details."""
        base_info = self.get_node_info(node)
        func_info = ASTAnalyzer.get_function_info(node)
        
        return {
            **base_info,
            **func_info,
            'is_method': isinstance(getattr(node, 'parent', None), ast.ClassDef)
        }
```

4. `classes.py` (Simplified):

```python
"""Class extractor module."""

from typing import Dict, Any, List
import ast
from .base import BaseExtractor

class ClassExtractor(BaseExtractor):
    """Class definition extractor."""

    def extract(self) -> List[Dict[str, Any]]:
        """Extract all classes."""
        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self.extract_class(node))
        return classes

    def extract_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract single class details."""
        base_info = self.get_node_info(node)
        return {
            **base_info,
            'bases': [ast.unparse(base) for base in node.bases],
            'methods': self.extract_methods(node),
            'attributes': self.extract_attributes(node)
        }

    def extract_methods(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class methods."""
        return [
            self.extract_function(n) for n in node.body 
            if isinstance(n, ast.FunctionDef)
        ]

    def extract_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes."""
        attributes = []
        for n in node.body:
            if isinstance(n, (ast.AnnAssign, ast.Assign)):
                if isinstance(n, ast.AnnAssign):
                    name = n.target.id if isinstance(n.target, ast.Name) else None
                    type_annotation = ASTAnalyzer.get_annotation(n.annotation)
                else:
                    name = n.targets[0].id if isinstance(n.targets[0], ast.Name) else None
                    type_annotation = "Any"
                
                if name:
                    attributes.append({
                        'name': name,
                        'type': type_annotation,
                        'has_default': n.value is not None
                    })
        return attributes
```

5. `extraction_manager.py` (Simplified):

```python
"""Extraction manager module."""

from typing import Dict, Any
from core.logger import LoggerSetup
from .functions import FunctionExtractor
from .classes import ClassExtractor

logger = LoggerSetup.get_logger(__name__)

class ExtractionManager:
    """Manages code extraction process."""

    def extract_metadata(self, source_code: str) -> Dict[str, Any]:
        """Extract code metadata."""
        try:
            function_extractor = FunctionExtractor(source_code)
            class_extractor = ClassExtractor(source_code)

            return {
                'functions': function_extractor.extract(),
                'classes': class_extractor.extract(),
                'metrics': self._calculate_metrics(source_code)
            }
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {'functions': [], 'classes': [], 'metrics': {}}

    def _calculate_metrics(self, source_code: str) -> Dict[str, Any]:
        """Calculate code metrics."""
        try:
            metrics = MetricsCalculator()
            tree = ASTAnalyzer.parse_source(source_code)
            return {
                'maintainability': metrics.calculate_maintainability_index(tree),
                'complexity': metrics.calculate_cyclomatic_complexity(tree)
            }
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {}
```

Key improvements:
1. Simplified class hierarchies
2. Removed duplicate code
3. Clearer separation of concerns
4. More focused methods
5. Better error handling
6. Simplified interfaces
7. Removed unnecessary abstractions
8. Better type hints
9. Clearer naming  
[Timestamp: 2024/11/22 14:45:59]

**user**: Adjust [metrics_calculator.py](metrics_calculator.py.md) and [metrics.py](metrics.py.md) if needed  
[Timestamp: 2024/11/22 14:47:11]

**ai**: Yes, there's significant duplication between `metrics_calculator.py` and `metrics.py`. Here's a simplified consolidated version:

```python
"""Code metrics calculation module."""

import ast
import math
from typing import Dict, Set, Optional
from core.logger import LoggerSetup

logger = LoggerSetup.get_logger(__name__)

class CodeMetrics:
    """Calculates code complexity and quality metrics."""

    THRESHOLDS = {
        'maintainability': {'good': 80, 'moderate': 60, 'poor': 40},
        'complexity': {'good': 5, 'moderate': 10, 'poor': 20}
    }

    @staticmethod
    def calculate_complexity(node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        for subnode in ast.walk(node):
            if isinstance(subnode, (
                ast.If, ast.For, ast.While, ast.And, ast.Or,
                ast.ExceptHandler, ast.With, ast.Try, ast.BoolOp,
                ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp,
                ast.GeneratorExp, ast.IfExp
            )):
                complexity += 1
            elif isinstance(subnode, ast.BoolOp):
                complexity += len(subnode.values) - 1
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(node: ast.AST) -> int:
        """Calculate cognitive complexity."""
        cognitive = 0
        nesting = 0
        
        for subnode in ast.walk(node):
            if isinstance(subnode, (
                ast.If, ast.For, ast.While, ast.Try,
                ast.ExceptHandler, ast.With
            )):
                nesting += 1
                cognitive += nesting
            elif isinstance(subnode, (ast.BoolOp, ast.Compare)):
                cognitive += 1
                
        return cognitive

    @staticmethod
    def calculate_halstead_metrics(node: ast.AST) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        operators = set()
        operands = set()
        op_count = 0
        operand_count = 0

        # Define node types
        operator_types = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
            ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
            ast.BitXor, ast.BitAnd, ast.FloorDiv
        )
        operand_types = (
            ast.Num, ast.Str, ast.Bytes, ast.Name,
            ast.List, ast.Tuple, ast.Set, ast.Dict,
            ast.Constant
        )

        # Count operators and operands
        for subnode in ast.walk(node):
            if isinstance(subnode, operator_types):
                operators.add(type(subnode).__name__)
                op_count += 1
            elif isinstance(subnode, operand_types):
                operands.add(type(subnode).__name__)
                operand_count += 1

        # Calculate metrics
        n1, n2 = len(operators), len(operands)
        N1, N2 = op_count, operand_count
        
        length = N1 + N2
        vocabulary = n1 + n2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0

        return {
            'length': length,
            'vocabulary': vocabulary,
            'volume': volume
        }

    @staticmethod
    def calculate_maintainability(node: ast.AST) -> float:
        """Calculate maintainability index."""
        try:
            # Get base metrics
            halstead = CodeMetrics.calculate_halstead_metrics(node)
            complexity = CodeMetrics.calculate_complexity(node)
            
            # Count lines
            source = ast.unparse(node)
            lines = len([l for l in source.splitlines() if l.strip()])
            
            # Calculate maintainability index
            volume = halstead['volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(lines)
            
            return max(0, min(100, mi))
        except Exception as e:
            logger.error(f"Error calculating maintainability: {e}")
            return 0.0

    @classmethod
    def analyze_node(cls, node: ast.AST) -> Dict[str, Any]:
        """Complete node analysis."""
        return {
            'complexity': cls.calculate_complexity(node),
            'cognitive_complexity': cls.calculate_cognitive_complexity(node),
            'maintainability': cls.calculate_maintainability(node),
            'halstead': cls.calculate_halstead_metrics(node)
        }

    @classmethod
    def get_quality_level(cls, metric: str, value: float) -> str:
        """Get quality level for a metric value."""
        thresholds = cls.THRESHOLDS.get(metric, {})
        if value >= thresholds.get('good', 0):
            return 'good'
        elif value >= thresholds.get('moderate', 0):
            return 'moderate'
        return 'poor'
```

Key improvements:
1. Merged duplicate functionality
2. Simplified calculations
3. Removed redundant logging
4. Focused on core metrics
5. Better organization
6. Clearer interfaces
7. Removed unnecessary complexity
8. Added quality level assessment
9. Better error handling
10. More maintainable structure

This single class now handles all metric calculations with a cleaner, more focused approach. The test module can be separate if needed.  
[Timestamp: 2024/11/22 14:47:41]

**user**: Integrate with extraction manager, [token_management.py](token_management.py.md) [main.py](Resources/simple%20source/main.py.md)[ai_service.py](ai_service.py.md)  
[Timestamp: 2024/11/22 14:49:01]

**ai**: Here's how to integrate the simplified metrics with the extraction manager and other components:

1. First, modify `ExtractionManager` to use the simplified metrics:

```python
"""Extraction manager module."""

from typing import Dict, Any
from core.logger import LoggerSetup
from utils.ast_analysis import ASTAnalyzer
from core.metrics import CodeMetrics
from .functions import FunctionExtractor
from .classes import ClassExtractor

logger = LoggerSetup.get_logger(__name__)

class ExtractionManager:
    """Manages code extraction process."""

    def __init__(self):
        """Initialize manager."""
        self.metrics = CodeMetrics()

    def extract_metadata(self, source_code: str) -> Dict[str, Any]:
        """Extract code metadata."""
        try:
            # Parse source code
            tree = ASTAnalyzer.parse_source(source_code)
            
            # Extract code elements
            function_extractor = FunctionExtractor(source_code)
            class_extractor = ClassExtractor(source_code)

            # Calculate metrics
            metrics = {
                'maintainability': self.metrics.calculate_maintainability(tree),
                'complexity': self.metrics.calculate_complexity(tree),
                'halstead': self.metrics.calculate_halstead_metrics(tree)
            }

            return {
                'functions': function_extractor.extract(),
                'classes': class_extractor.extract(),
                'metrics': metrics,
                'quality_level': self.metrics.get_quality_level('maintainability', metrics['maintainability'])
            }

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                'functions': [],
                'classes': [],
                'metrics': {},
                'quality_level': 'poor'
            }
```

2. Update `AIInteractionHandler` to use the metrics:

```python
class AIInteractionHandler:
    def __init__(self, config: AzureOpenAIConfig, cache: Optional[Cache] = None):
        self.config = config
        self.cache = cache
        self.metrics = CodeMetrics()
        self.token_manager = TokenManager(model=config.model_name, cache=cache)
        self.extraction_manager = ExtractionManager()

    async def process_code(self, source_code: str) -> Tuple[str, str]:
        """Process source code."""
        try:
            # Extract metadata with metrics
            metadata = self.extraction_manager.extract_metadata(source_code)
            
            # Use metrics for processing decisions
            quality_level = metadata['quality_level']
            if quality_level == 'poor':
                logger.warning(f"Code quality is poor (maintainability: {metadata['metrics']['maintainability']})")

            # Process code based on metrics
            token_limit = self._adjust_token_limit(metadata['metrics']['complexity'])
            
            # Generate documentation
            result = await self._generate_documentation(source_code, metadata, token_limit)
            
            return result['code'], result['documentation']

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _adjust_token_limit(self, complexity: int) -> int:
        """Adjust token limit based on code complexity."""
        base_limit = self.config.max_tokens
        if complexity > 20:
            return min(base_limit * 2, 32000)  # Double limit for complex code
        return base_limit
```

3. Update token management to use metrics:

```python
class TokenManager:
    def __init__(self, model: str, cache: Optional[Cache] = None):
        self.model = model
        self.cache = cache
        self.metrics = CodeMetrics()

    async def optimize_prompt(
        self,
        text: str,
        code_metrics: Dict[str, Any],
        max_tokens: Optional[int] = None
    ) -> str:
        """Optimize prompt based on code metrics."""
        try:
            complexity = code_metrics.get('complexity', 0)
            maintainability = code_metrics.get('maintainability', 0)

            # Adjust token limits based on code quality
            if maintainability < 40:  # Poor maintainability
                max_tokens = max_tokens or (self.config['max_tokens'] // 2)
            elif complexity > 20:  # High complexity
                max_tokens = max_tokens or (self.config['max_tokens'] * 2)

            return await self._optimize_text(text, max_tokens)

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return text
```

4. Update the main workflow:

```python
class WorkflowOrchestrator:
    async def run(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run workflow with metrics integration."""
        try:
            processor = ProjectProcessor(self.config)
            metrics = CodeMetrics()

            # Process files
            results = await processor.process_project(source_path, output_dir)

            # Analyze overall project metrics
            project_metrics = {
                'average_maintainability': sum(r['metrics']['maintainability'] for r in results) / len(results),
                'average_complexity': sum(r['metrics']['complexity'] for r in results) / len(results),
                'quality_level': metrics.get_quality_level(
                    'maintainability',
                    sum(r['metrics']['maintainability'] for r in results) / len(results)
                )
            }

            return {
                'results': results,
                'project_metrics': project_metrics
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise
```

This integration:
1. Simplifies metrics calculation
2. Provides quality-based decision making
3. Adjusts token limits based on code complexity
4. Adds project-level metrics
5. Improves error handling
6. Maintains clean separation of concerns

The metrics are now used to make intelligent decisions about processing while keeping the code modular and maintainable.  
[Timestamp: 2024/11/22 14:49:34]

**user**: Generate the complete ai_interaction  
[Timestamp: 2024/11/22 14:50:06]

**ai**: Here's the complete implementation of `ai_interaction.py` with the integration of the simplified metrics and other components:

```python
"""
AI Interaction Handler Module

Manages interactions with AI models, including token management, caching,
response parsing, and monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import AsyncAzureOpenAI

from core.logger import LoggerSetup
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.monitoring import MetricsCollector, SystemMonitor
from exceptions import (
    AIServiceError,
    TokenLimitError,
    ValidationError,
    ProcessingError
)
from api.token_management import TokenManager
from api.response_parser import create_response_parser
from extract.extraction_manager import ExtractionManager
from docs.docs import DocStringManager
from core.metrics import CodeMetrics

# Initialize logger
logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """Manages AI model interactions with integrated monitoring and caching."""

    def __init__(
        self,
        config: AzureOpenAIConfig,
        cache: Optional[Cache] = None,
        batch_size: int = 5,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the AI interaction handler.

        Args:
            config: Configuration for Azure OpenAI
            cache: Optional cache instance
            batch_size: Size of batches for processing
            metrics_collector: Optional metrics collector instance
        """
        self.config = config
        self.cache = cache
        self.batch_size = batch_size
        self.metrics_collector = metrics_collector or MetricsCollector()

        # Initialize components with proper integration
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )

        # Initialize TokenManager with cache
        self.token_manager = TokenManager(
            model=config.model_name,
            deployment_name=config.deployment_name,
            cache=cache
        )

        # Use factory function for ResponseParser
        self.response_parser = create_response_parser(
            token_manager=self.token_manager
        )

        self.monitor = SystemMonitor()
        self.extraction_manager = ExtractionManager()
        self.code_metrics = CodeMetrics()

        logger.info("AI Interaction Handler initialized")

    async def process_code(self, source_code: str) -> Tuple[str, str]:
        """
        Process source code to generate documentation.

        Args:
            source_code: Source code to process

        Returns:
            Tuple[str, str]: (updated_code, documentation)

        Raises:
            ProcessingError: If processing fails
        """
        try:
            operation_start = datetime.now()

            # Extract metadata with metrics using ExtractionManager
            metadata = self.extraction_manager.extract_metadata(source_code)
            
            # Use metrics for processing decisions
            quality_level = metadata['quality_level']
            if quality_level == 'poor':
                logger.warning(f"Code quality is poor (maintainability: {metadata['metrics']['maintainability']})")

            # Adjust token limit based on complexity
            token_limit = self._adjust_token_limit(metadata['metrics']['complexity'])
            
            # Process functions and classes in batches
            doc_entries = []

            # Process functions
            for batch in self._batch_items(metadata['functions'], self.batch_size):
                batch_results = await asyncio.gather(*[
                    self.generate_docstring(
                        func_name=func['name'],
                        params=func['args'],
                        return_type=func['return_type'],
                        complexity_score=func.get('metrics', {}).get('complexity', 0),
                        existing_docstring=func.get('docstring', ''),
                        decorators=func.get('decorators', []),
                        exceptions=func.get('exceptions', []),
                        metrics=func.get('metrics', {})
                    )
                    for func in batch
                ], return_exceptions=True)

                for func, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self._handle_error(result, f"Error processing function {func['name']}")
                        continue
                    if result:
                        doc_entries.append({
                            'type': 'function',
                            'name': func['name'],
                            'docstring': result['docstring'],
                            'metrics': func.get('metrics', {})
                        })

            # Process classes
            for batch in self._batch_items(metadata['classes'], self.batch_size):
                batch_results = await asyncio.gather(*[
                    self.generate_docstring(
                        func_name=cls['name'],
                        params=[],
                        return_type='None',
                        complexity_score=cls.get('metrics', {}).get('complexity', 0),
                        existing_docstring=cls.get('docstring', ''),
                        decorators=cls.get('decorators', []),
                        is_class=True,
                        metrics=cls.get('metrics', {})
                    )
                    for cls in batch
                ], return_exceptions=True)

                for cls, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self._handle_error(result, f"Error processing class {cls['name']}")
                        continue
                    if result:
                        doc_entries.append({
                            'type': 'class',
                            'name': cls['name'],
                            'docstring': result['docstring'],
                            'metrics': cls.get('metrics', {})
                        })

            # Track code quality metrics
            self.metrics_collector.track_code_quality(str(source_code), metadata['metrics'])

            # Process documentation
            doc_manager = DocStringManager(source_code)
            result = doc_manager.process_batch(doc_entries)

            # Track metrics
            operation_time = (datetime.now() - operation_start).total_seconds()
            self.metrics_collector.track_operation(
                operation_type='process_code',
                success=bool(result),
                duration=operation_time,
                metrics=metadata['metrics']
            )

            if result:
                return result['code'], result['documentation']
            
            raise ProcessingError("Failed to generate documentation")

        except Exception as e:
            self._handle_error(e, "Error processing code")
            raise ProcessingError(f"Code processing failed: {str(e)}")

    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        is_class: bool = False,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a docstring using the AI model.

        Args:
            func_name: Name of the function/class
            params: List of parameter tuples (name, type)
            return_type: Return type annotation
            complexity_score: Code complexity score
            existing_docstring: Existing docstring if any
            decorators: List of decorators
            exceptions: List of exceptions
            is_class: Whether generating for a class
            metrics: Additional metrics data

        Returns:
            Optional[Dict[str, Any]]: Generated docstring data if successful
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                func_name, params, return_type, complexity_score, is_class
            )
            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    return cached

            # Create messages for AI model
            messages = self._create_messages(
                func_name, params, return_type, complexity_score,
                existing_docstring, decorators, exceptions, is_class
            )

            # Make API request
            response = await self._make_api_request(messages)
            
            if response:
                parsed = self.response_parser.parse_response(response)
                if parsed and self.cache:
                    await self.cache.set(cache_key, parsed)
                return parsed

            return None

        except Exception as e:
            self._handle_error(e, f"Error generating docstring for {func_name}")
            return None

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle component errors consistently."""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"Error in {context}: {error}", extra=error_details)
        self.metrics_collector.track_error(error_details)

    def _generate_cache_key(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        is_class: bool
    ) -> str:
        """Generate a consistent cache key."""
        key_parts = [
            func_name,
            str(sorted(params)),
            return_type,
            str(complexity_score),
            str(is_class)
        ]
        return f"docstring:{':'.join(key_parts)}"

    def _create_messages(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]],
        exceptions: Optional[List[str]],
        is_class: bool
    ) -> List[Dict[str, str]]:
        """Create messages for AI model prompt."""
        return [
            {
                "role": "system",
                "content": "Generate clear, comprehensive docstrings following Google style guide."
            },
            {
                "role": "user",
                "content": json.dumps({
                    "name": func_name,
                    "type": "class" if is_class else "function",
                    "parameters": params,
                    "return_type": return_type,
                    "complexity_score": complexity_score,
                    "existing_docstring": existing_docstring,
                    "decorators": decorators or [],
                    "exceptions": exceptions or []
                })
            }
        ]

    async def _make_api_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Make an API request with monitoring."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            self._handle_error(e, "API request failed")
            return None

    @staticmethod
    def _batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches."""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    async def cleanup(self) -> None:
        """Cleanup resources properly."""
        try:
            if self.cache:
                await self.cache.close()
            if self.token_manager:
                await self.token_manager.cleanup()
            if self.response_parser:
                await self.response_parser.cleanup()
            self.monitor.reset()
            logger.info("AI Interaction Handler cleaned up successfully")
        except Exception as e:
            self._handle_error(e, "Error during cleanup")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    def _adjust_token_limit(self, complexity: int) -> int:
        """Adjust token limit based on code complexity."""
        base_limit = self.config.max_tokens
        if complexity > 20:
            return min(base_limit * 2, 32000)  # Double limit for complex code
        return base_limit
```

## Key Features

1. **Integration with Metrics:**
   - Uses `CodeMetrics` for analyzing code quality and making decisions based on metrics.
   - Adjusts processing based on code complexity and maintainability.

2. **Error Handling and Logging:**
   - Consistent error handling with detailed logging.
   - Tracks errors and operations using `MetricsCollector`.

3. **Token Management:**
   - Utilizes `TokenManager` for handling token-related operations.
   - Adjusts token limits dynamically based on code complexity.

4. **Batch Processing:**
   - Processes functions and classes in batches for efficiency.
   - Uses asynchronous operations for improved performance.

5. **Documentation Generation:**
   - Generates docstrings using AI models.
   - Caches results for repeated requests.

This implementation focuses on modularity, efficiency, and maintainability, integrating metrics and token management to optimize AI interactions.  
[Timestamp: 2024/11/22 14:51:39]

**user**: Generate the complete token_management  
[Timestamp: 2024/11/22 14:51:51]

**ai**: Here's the complete implementation of `token_management.py`, incorporating efficient token counting, optimization, and cost calculation:

```python
"""
Token Management Module

Handles token counting, optimization, and cost calculation for AI model interactions
with improved separation of concerns and modularity.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import tiktoken
from datetime import datetime
import json
from core.logger import LoggerSetup
from core.cache import Cache

logger = LoggerSetup.get_logger(__name__)

@dataclass
class TokenCount:
    """Represents token count information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class TokenCost:
    """Represents token cost information."""
    prompt_cost: float
    completion_cost: float
    total_cost: float

@dataclass
class TokenConstraints:
    """Represents token constraints for text optimization."""
    max_tokens: int
    preserve_sections: Optional[List[str]] = None
    min_tokens: int = 0

class ModelConfig:
    """Configuration for different AI models."""

    # Token limits and pricing for different models
    MODEL_CONFIGS = {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1k_prompt": 0.03,
            "cost_per_1k_completion": 0.06,
            "cached_cost_per_1k_prompt": 0.015,  # 50% discount for cached requests
            "cached_cost_per_1k_completion": 0.03
        },
        "gpt-4-32k": {
            "max_tokens": 32768,
            "cost_per_1k_prompt": 0.06,
            "cost_per_1k_completion": 0.12,
            "cached_cost_per_1k_prompt": 0.03,
            "cached_cost_per_1k_completion": 0.06
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "cost_per_1k_prompt": 0.0015,
            "cost_per_1k_completion": 0.002,
            "cached_cost_per_1k_prompt": 0.00075,
            "cached_cost_per_1k_completion": 0.001
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "cost_per_1k_prompt": 0.003,
            "cost_per_1k_completion": 0.004,
            "cached_cost_per_1k_prompt": 0.0015,
            "cached_cost_per_1k_completion": 0.002
        }
    }

    # Mapping of deployment names to model names
    DEPLOYMENT_MAPPING = {
        "gpt-4": "gpt-4",
        "gpt-4-32k": "gpt-4-32k",
        "gpt-35-turbo": "gpt-3.5-turbo",
        "gpt-35-turbo-16k": "gpt-3.5-turbo-16k"
    }

    @classmethod
    def get_config(cls, model_name: str) -> Dict[str, Union[int, float]]:
        """Get configuration for a specific model."""
        return cls.MODEL_CONFIGS.get(model_name, cls.MODEL_CONFIGS["gpt-4"])

    @classmethod
    def map_deployment_to_model(cls, deployment_name: str) -> str:
        """Map deployment name to model name."""
        return cls.DEPLOYMENT_MAPPING.get(deployment_name, "gpt-4")

    @classmethod
    def load_custom_config(cls, config_path: str) -> None:
        """Load custom model configuration."""
        try:
            with open(config_path) as f:
                custom_config = json.load(f)
                cls.MODEL_CONFIGS.update(custom_config)
        except Exception as e:
            logger.error(f"Error loading custom config: {e}")

class TokenCounter:
    """Handles token counting operations."""

    def __init__(self, model: str, cache: Optional[Cache] = None):
        """Initialize token counter for specific model."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Using cl100k_base encoding as fallback for model {model}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.cache = cache

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching."""
        try:
            # Generate cache key
            cache_key = f"token_count:{hashlib.md5(text.encode()).hexdigest()}"
            
            # Check cache if available
            if self.cache:
                cached_count = await self.cache.get_cached_docstring(cache_key)
                if cached_count is not None:
                    return int(cached_count)
            
            # Count tokens
            count = len(self.encoding.encode(text))
            
            # Cache result if cache is available
            if self.cache:
                await self.cache.save_docstring(cache_key, str(count), expire=3600)
            
            return count
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

class TokenOptimizer:
    """Handles token optimization operations."""

    def __init__(self, token_counter: TokenCounter):
        """Initialize with token counter."""
        self.token_counter = token_counter

    async def optimize_text(self, text: str, constraints: TokenConstraints) -> str:
        """Optimize text to fit within token constraints."""
        current_tokens = await self.token_counter.count_tokens(text)
        
        if current_tokens <= constraints.max_tokens:
            return text

        try:
            sections = text.split('\n\n')
            preserved = []
            optional = []

            # Separate preserved and optional sections
            for section in sections:
                if constraints.preserve_sections and \
                   any(p in section for p in constraints.preserve_sections):
                    preserved.append(section)
                else:
                    optional.append(section)

            # Start with preserved content
            optimized = '\n\n'.join(preserved)
            remaining_tokens = constraints.max_tokens - \
                             await self.token_counter.count_tokens(optimized)

            # Add optional sections that fit
            for section in optional:
                section_tokens = await self.token_counter.count_tokens(section)
                if remaining_tokens >= section_tokens:
                    optimized = f"{optimized}\n\n{section}"
                    remaining_tokens -= section_tokens

            return optimized

        except Exception as e:
            logger.error(f"Error optimizing text: {e}")
            return text

class CostCalculator:
    """Handles token cost calculations."""

    def __init__(self, model_config: Dict[str, Union[int, float]]):
        """Initialize with model configuration."""
        self.config = model_config

    def calculate_cost(self, token_count: TokenCount, cached: bool = False) -> TokenCost:
        """Calculate costs for token usage."""
        try:
            # Apply cached pricing if applicable
            prompt_rate = self.config.get(
                "cached_cost_per_1k_prompt" if cached else "cost_per_1k_prompt",
                self.config["cost_per_1k_prompt"]
            )
            completion_rate = self.config.get(
                "cached_cost_per_1k_completion" if cached else "cost_per_1k_completion",
                self.config["cost_per_1k_completion"]
            )

            prompt_cost = (token_count.prompt_tokens / 1000) * prompt_rate
            completion_cost = (token_count.completion_tokens / 1000) * completion_rate

            return TokenCost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            )

        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return TokenCost(0.0, 0.0, 0.0)

class TokenManager:
    """Main token management class with improved modularity."""

    def __init__(
        self,
        model: str = "gpt-4",
        deployment_name: Optional[str] = None,
        cache: Optional[Cache] = None
    ):
        """Initialize token management components."""
        # Map deployment name to model if provided
        self.model = ModelConfig.map_deployment_to_model(deployment_name) if deployment_name else model
        self.config = ModelConfig.get_config(self.model)
        self.cache = cache
        
        # Initialize components
        self.counter = TokenCounter(self.model, cache)
        self.optimizer = TokenOptimizer(self.counter)
        self.cost_calculator = CostCalculator(self.config)
        
        # Initialize usage tracking
        self.usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_cost': 0.0,
            'requests': 0,
            'cached_requests': 0,
            'last_request': None
        }

    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text with caching."""
        cache_key = f"tokens:{hashlib.md5(text.encode()).hexdigest()}"
        
        if self.cache:
            cached_count = await self.cache.get_cached_docstring(cache_key)
            if cached_count:
                self.usage['cached_requests'] += 1
                return int(cached_count)
        
        count = await self.counter.count_tokens(text)
        
        if self.cache:
            await self.cache.save_docstring(cache_key, str(count))
        
        return count

    async def optimize_prompt(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenCount]:
        """Optimize prompt to fit within token limits."""
        constraints = TokenConstraints(
            max_tokens=max_tokens or self.config["max_tokens"] // 2,
            preserve_sections=preserve_sections
        )
        
        optimized_text = await self.optimizer.optimize_text(text, constraints)
        token_count = TokenCount(
            prompt_tokens=await self.counter.count_tokens(optimized_text),
            completion_tokens=0,
            total_tokens=await self.counter.count_tokens(optimized_text)
        )
        
        return optimized_text, token_count

    def track_request(self, prompt_tokens: int, completion_tokens: int, cached: bool = False) -> None:
        """Track token usage for a request."""
        self.usage['prompt_tokens'] += prompt_tokens
        self.usage['completion_tokens'] += completion_tokens
        self.usage['requests'] += 1
        if cached:
            self.usage['cached_requests'] += 1
        self.usage['last_request'] = datetime.now()
        
        # Calculate and track cost
        token_count = TokenCount(prompt_tokens, completion_tokens, 
                               prompt_tokens + completion_tokens)
        cost = self.cost_calculator.calculate_cost(token_count, cached=cached)
        self.usage['total_cost'] += cost.total_cost

    def validate_input(self, text: str) -> None:
        """Validate input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        if not text.strip():
            raise ValueError("Input cannot be empty")

    async def validate_request(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Union[int, float]], str]:
        """Validate if request is within token limits."""
        try:
            self.validate_input(prompt)
            prompt_tokens = await self.estimate_tokens(prompt)
            max_completion = max_completion_tokens or \
                           (self.config["max_tokens"] - prompt_tokens)
            total_tokens = prompt_tokens + max_completion

            metrics = {
                'prompt_tokens': prompt_tokens,
                'max_completion_tokens': max_completion,
                'total_tokens': total_tokens,
                'model_limit': self.config["max_tokens"]
            }

            if total_tokens > self.config["max_tokens"]:
                message = f"Total tokens ({total_tokens}) exceeds model limit " \
                         f"({self.config['max_tokens']})"
                return False, metrics, message

            return True, metrics, "Request validated successfully"

        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return False, {}, str(e)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        return {
            'prompt_tokens': self.usage['prompt_tokens'],
            'completion_tokens': self.usage['completion_tokens'],
            'total_tokens': self.usage['prompt_tokens'] + self.usage['completion_tokens'],
            'total_cost': round(self.usage['total_cost'], 4),
            'total_requests': self.usage['requests'],
            'cached_requests': self.usage['cached_requests'],
            'cache_hit_rate': round(
                (self.usage['cached_requests'] / self.usage['requests'] * 100)
                if self.usage['requests'] > 0 else 0,
                2
            ),
            'last_request': self.usage['last_request'].isoformat() \
                          if self.usage['last_request'] else None
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.cache:
                await self.cache.close()
        except Exception as e:
            logger.error(f"Error during token manager cleanup: {e}")

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self.usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_cost': 0.0,
            'requests': 0,
            'cached_requests': 0,
            'last_request': None
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
```

## Key Features

1. **Token Counting and Caching:**
   - Efficiently counts tokens using `tiktoken` and caches results to optimize performance.

2. **Token Optimization:**
   - Optimizes text to fit within specified token constraints, preserving important sections.

3. **Cost Calculation:**
   - Calculates token usage costs based on model configuration, with support for cached pricing.

4. **Usage Tracking:**
   - Tracks token usage statistics, including prompt and completion tokens, total cost, and cache hit rate.

5. **Validation:**
   - Validates input text and checks if requests are within token limits.

6. **Resource Management:**
   - Supports asynchronous context management for resource cleanup.

This implementation provides a robust and modular approach to managing tokens, ensuring efficient and cost-effective interactions with AI models.  
[Timestamp: 2024/11/22 14:53:08]

**user**: Report changelig  
[Timestamp: 2024/11/22 14:53:20]

**ai**: Certainly! Here's a changelog summarizing the updates and improvements made across the codebase:

## Changelog

### Version 2.0.0

**Date:** [Insert Date]

**Overview:**  
This release includes significant refactoring and enhancements to improve code maintainability, performance, and integration across components. Key areas of focus include metrics calculation, token management, AI interaction handling, and extraction management.

---

**1. Metrics Calculation:**
- **Consolidation of Metrics:**
  - Merged `metrics_calculator.py` and `metrics.py` into a single `CodeMetrics` class.
  - Simplified the calculation of cyclomatic complexity, cognitive complexity, Halstead metrics, and maintainability index.
  - Added a method to determine quality levels based on thresholds.
- **Error Handling:**
  - Improved error handling and logging for metrics calculations.

**2. Token Management:**
- **Enhanced Token Counting:**
  - Implemented caching for token counts to improve performance.
  - Utilized `tiktoken` for efficient token encoding.
- **Optimization and Cost Calculation:**
  - Introduced `TokenOptimizer` for optimizing text within token constraints.
  - Added `CostCalculator` to compute token usage costs based on model configurations.
- **Usage Tracking:**
  - Enhanced tracking of token usage statistics, including cache hit rates and cost calculations.

**3. AI Interaction Handling:**
- **Integration with Metrics:**
  - Integrated `CodeMetrics` into `AIInteractionHandler` for informed decision-making based on code quality.
  - Adjusted processing logic based on code complexity and maintainability.
- **Batch Processing:**
  - Improved batch processing of functions and classes for efficiency.
  - Utilized asynchronous operations to enhance performance.
- **Error Management:**
  - Standardized error handling and logging across AI interactions.

**4. Extraction Management:**
- **Simplification:**
  - Streamlined `ExtractionManager` to leverage the consolidated `CodeMetrics`.
  - Simplified extraction logic for functions and classes using `ASTAnalyzer`.
- **Quality Assessment:**
  - Added quality level assessment based on maintainability metrics.

**5. General Improvements:**
- **Logging and Monitoring:**
  - Enhanced logging for better traceability and debugging.
  - Integrated metrics collection for monitoring operations and errors.
- **Configuration Management:**
  - Improved configuration handling for model-specific settings and custom configurations.
- **Resource Management:**
  - Implemented asynchronous context management for resource cleanup across components.

**6. Documentation:**
- **Updated Documentation:**
  - Revised docstrings and comments for clarity and completeness.
  - Updated documentation to reflect new features and changes.

---

**Future Enhancements:**
- Explore further optimizations for token management and AI interactions.
- Consider additional metrics for more comprehensive code quality analysis.
- Enhance test coverage to ensure robustness and reliability.

This release marks a significant step forward in improving the overall architecture and functionality of the codebase, paving the way for future enhancements and scalability.  
[Timestamp: 2024/11/22 14:53:35]
