---
epoch: 1732206861655
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# Fix_the_indentation_issues_in_this_script_import_asyncio_import@20241121_103421

**user**: Fix the indentation issues in this script:  
import asyncio  
import argparse  
import sys  
import time  
import uuid  
from datetime import datetime  
from pathlib import Path  
from typing import Dict, Any, Optional, AsyncContextManager

from tqdm.asyncio import tqdm_asyncio

from core.logger import LoggerSetup  
from core.config import AzureOpenAIConfig as Config  
from core.cache import Cache  
from core.monitoring import MetricsCollector, SystemMonitor  
from ai_interaction import AIInteractionHandler  
from api.token_management import TokenManager  
from docs.markdown_generator import MarkdownDocumentationGenerator  
from exceptions import WorkflowError

logger = LoggerSetup.get_logger(**name**)

async def load_and_validate_config(args: argparse.Namespace) -> Config:  
    """Load and validate configuration."""  
    try:  
        config = Config.from_env()  
        return config  
    except Exception as e:  
        logger.error(f"Failed to load or validate configuration: {e}")  
        raise

def setup_logging(log_level: str, log_format: str,  
                  log_directory: Optional[Path] = None):  
    """Configure logging."""  
    LoggerSetup.configure(log_level, log_format, log_directory)

async def initialize_components(config: Config) -> AsyncContextManager:  
    """Initialize all required components with proper cleanup."""  
    return AsyncComponentManager(config)

class AsyncComponentManager:  
    """Manages async components lifecycle and dependencies."""

    def __init__(self, config: Config):
        self.config = config

    async def __aenter__(self):
        """Initialize all components in correct order."""
        self.components = {}
        try:
            # Initialize monitoring first
            self.components["metrics"] = MetricsCollector(
                self.config.monitoring_config
            )
            self.components["system_monitor"] = SystemMonitor(
                self.config.monitoring_config
            )

            # Initialize cache if enabled
            if self.config.cache_config.enabled:
                self.components["cache"] = await Cache(
                    self.config.cache_config
                ).connect()

            # Initialize token manager
            self.components["token_manager"] = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                max_tokens=self.config.max_tokens,
                cache_enabled=self.config.cache_enabled,
            )

            # Initialize AI handler with dependencies
            self.components["ai_handler"] = AIInteractionHandler(
                config=self.config,
                cache=self.components.get("cache"),
                token_manager=self.components["token_manager"],
                metrics_collector=self.components["metrics"],
                system_monitor=self.components["system_monitor"],
            )

            return self

        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup components in reverse order."""
            try:
                component = self.components[component_name]
                if hasattr(component, "close"):
                    await component.close()
            except Exception as e:
                logger.error(f"Error closing {component_name}: {e}")

class DocumentationProcessor:  
    """Handles the core documentation processing logic."""

    def __init__(self, components: AsyncComponentManager, config: Config):
        self.components = components
        self.config = config
        self.logger = LoggerSetup.get_logger(__name__)

    async def process_file(
        self, file_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Process a single file with enhanced error handling and metrics."""
        start_time = time.time()
        metrics = {
            "file_path": str(file_path),
            "start_time": start_time,
            "status": "pending",
        }

        try:
            # Read source code
            source_code = await self._read_source_code(file_path)
            if not source_code.strip():
                return self._create_result(
                    "skipped", message="Empty source file", metrics=metrics
                )

            # Process code with AI handler
            updated_code, documentation = await self.components.components[
                "ai_handler"
            ].process_code(source_code)

            if not documentation:
                return self._create_result(
                    "failed", message="AI processing failed", metrics=metrics
                )

            doc_result = await self._generate_documentation(
                updated_code, documentation, file_path, output_dir
            )

            metrics.update(
                {
                    "end_time": time.time(),
                    "duration": time.time() - start_time,
                    "status": "success",
                    "token_usage": self.components.components[
                        "token_manager"
                    ].get_usage_stats(),
                    "documentation_size": len(doc_result.get("documentation", "")),
                }
            )

            return self._create_result("success", result=doc_result, metrics=metrics)

        except Exception as e:
            metrics.update(
                {
                    "end_time": time.time(),
                    "duration": time.time() - start_time,
                    "status": "failed",
                    "error": str(e),
                }
            )
            return self._create_result("failed", error=str(e), metrics=metrics)

    async def _read_source_code(self, file_path: Path) -> str:
        """Read source code with encoding detection."""
        try:
            return await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            # Implement encoding detection if needed
            return await asyncio.to_thread(file_path.read_text, encoding="latin-1")

    async def _generate_documentation(
        self,
        updated_code: str,
        documentation: str,
        file_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generate documentation with enhanced formatting and validation."""
        try:
            markdown_gen = MarkdownDocumentationGenerator(
                source_code=updated_code, module_path=str(file_path)
            )

            markdown_gen.add_metadata(
                {
                    "generated_at": datetime.now().isoformat(),
                    "token_usage": self.components.components[
                        "token_manager"
                    ].get_usage_stats(),
                    "ai_model": self.config.model_name,
                }
            )

            if not documentation:
                raise Exception("Generated documentation is empty")

            doc_path = output_dir / f"{file_path.stem}_docs.md"
            await self._save_documentation(doc_path, documentation)

            return {
                "documentation": documentation,
                "documentation_path": str(doc_path),
                "updated_code": updated_code,
            }

        except Exception as e:
            raise Exception(f"Documentation generation failed: {e}") from e

    async def _save_documentation(self, doc_path: Path, documentation: str):
        """Save documentation to file."""
        await asyncio.to_thread(doc_path.write_text, documentation, encoding="utf-8")

    def _create_result(
        self, status: str, result: Optional[Dict] = None, message: str = "", error: str = "", metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
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

    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerSetup.get_logger(__name__)

    async def run(
        self, source_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
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
                processor = DocumentationProcessor(components, self.config)
                if source_path.is_file():
                    file_result = await processor.process_file(source_path, output_dir)
                    self._update_results(results, file_result)
                elif source_path.is_dir():
                    files = list(source_path.rglob("*.py"))
                    async for file in tqdm_asyncio(files, desc="Processing files"):
                        file_result = await processor.process_file(file, output_dir)
                        self._update_results(results, file_result)
                else:
                    raise ValueError(f"Invalid source path: {source_path}")

                await self._generate_summary(
                    results, output_dir, workflow_id, time.time() - start_time
                )

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise WorkflowError(f"Workflow failed: {e}") from e

        return results

    def _update_results(self, results: Dict[str, Any], file_result: Dict[str, Any]) -> None:
        """Update results dictionary with file processing outcomes."""
        status = file_result["status"]
        results[f"{status}_files"] += 1

        token_usage = file_result.get("metrics", {}).get("token_usage", {})
        results["total_tokens"] += token_usage.get("total_tokens", 0)
        results["total_cost"] += token_usage.get("total_cost", 0.0)

    async def _generate_summary(
        self, results: Dict[str, Any], output_dir: Path, workflow_id: str, duration: float
    ) -> None:
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

        await asyncio.to_thread(
            summary_path.write_text, summary_content, encoding="utf-8"
        )
        self.logger.info(f"Workflow summary saved to {summary_path}")

def parse_arguments() -> argparse.Namespace:  
    """Parse command-line arguments."""  
    parser = argparse.ArgumentParser(description="Generate documentation using AI.")  
    parser.add_argument("source_path", type=Path, help="Path to the source code file or directory")  
    parser.add_argument("output_dir", type=Path, help="Path to the output directory")  
    parser.add_argument("--config-file", type=Path, help="Path to the configuration file")  
    return parser.parse_args()

async def run_workflow(args: argparse.Namespace) -> None:  
    """Run the documentation generation workflow."""  
    try:  
        config = await load_and_validate_config(args)  
        setup_logging(  
            config.log_level, config.log_format, config.log_directory  
        ) # Assuming these are in your Config

        async with initialize_components(config) as components:
            orchestrator = WorkflowOrchestrator(config)
            await orchestrator.run(args.source_path, args.output_dir)
            logger.info("Workflow completed successfully")

    except Exception as e:
        handle_workflow_error(e)
        sys.exit(1)

def handle_workflow_error(e: Exception):  
    """Handle workflow errors."""  
    logger.error(f"Workflow failed: {str(e)}")

if **name** == "**main**":  
    args = parse_arguments()  
    asyncio.run(run_workflow(args))

[Timestamp: 2024/11/21 10:34:21]

**ai**: Here is the corrected script with proper indentation:

```python
import asyncio
import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, AsyncContextManager

from tqdm.asyncio import tqdm_asyncio

from core.logger import LoggerSetup
from core.config import AzureOpenAIConfig as Config
from core.cache import Cache
from core.monitoring import MetricsCollector, SystemMonitor
from ai_interaction import AIInteractionHandler
from api.token_management import TokenManager
from docs.markdown_generator import MarkdownDocumentationGenerator
from exceptions import WorkflowError

logger = LoggerSetup.get_logger(__name__)


async def load_and_validate_config(args: argparse.Namespace) -> Config:
    """Load and validate configuration."""
    try:
        config = Config.from_env()
        return config
    except Exception as e:
        logger.error(f"Failed to load or validate configuration: {e}")
        raise


def setup_logging(log_level: str, log_format: str, 
                  log_directory: Optional[Path] = None):
    """Configure logging."""
    LoggerSetup.configure(log_level, log_format, log_directory)


async def initialize_components(config: Config) -> AsyncContextManager:
    """Initialize all required components with proper cleanup."""
    return AsyncComponentManager(config)


class AsyncComponentManager:
    """Manages async components lifecycle and dependencies."""

    def __init__(self, config: Config):
        self.config = config

    async def __aenter__(self):
        """Initialize all components in correct order."""
        self.components = {}
        try:
            # Initialize monitoring first
            self.components["metrics"] = MetricsCollector(
                self.config.monitoring_config
            )
            self.components["system_monitor"] = SystemMonitor(
                self.config.monitoring_config
            )

            # Initialize cache if enabled
            if self.config.cache_config.enabled:
                self.components["cache"] = await Cache(
                    self.config.cache_config
                ).connect()

            # Initialize token manager
            self.components["token_manager"] = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                max_tokens=self.config.max_tokens,
                cache_enabled=self.config.cache_enabled,
            )

            # Initialize AI handler with dependencies
            self.components["ai_handler"] = AIInteractionHandler(
                config=self.config,
                cache=self.components.get("cache"),
                token_manager=self.components["token_manager"],
                metrics_collector=self.components["metrics"],
                system_monitor=self.components["system_monitor"],
            )

            return self

        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup components in reverse order."""
        for component_name in reversed(self.components):
            try:
                component = self.components[component_name]
                if hasattr(component, "close"):
                    await component.close()
            except Exception as e:
                logger.error(f"Error closing {component_name}: {e}")


class DocumentationProcessor:
    """Handles the core documentation processing logic."""

    def __init__(self, components: AsyncComponentManager, config: Config):
        self.components = components
        self.config = config
        self.logger = LoggerSetup.get_logger(__name__)

    async def process_file(
        self, file_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Process a single file with enhanced error handling and metrics."""
        start_time = time.time()
        metrics = {
            "file_path": str(file_path),
            "start_time": start_time,
            "status": "pending",
        }

        try:
            # Read source code
            source_code = await self._read_source_code(file_path)
            if not source_code.strip():
                return self._create_result(
                    "skipped", message="Empty source file", metrics=metrics
                )

            # Process code with AI handler
            updated_code, documentation = await self.components.components[
                "ai_handler"
            ].process_code(source_code)

            if not documentation:
                return self._create_result(
                    "failed", message="AI processing failed", metrics=metrics
                )

            doc_result = await self._generate_documentation(
                updated_code, documentation, file_path, output_dir
            )

            metrics.update(
                {
                    "end_time": time.time(),
                    "duration": time.time() - start_time,
                    "status": "success",
                    "token_usage": self.components.components[
                        "token_manager"
                    ].get_usage_stats(),
                    "documentation_size": len(doc_result.get("documentation", "")),
                }
            )

            return self._create_result("success", result=doc_result, metrics=metrics)

        except Exception as e:
            metrics.update(
                {
                    "end_time": time.time(),
                    "duration": time.time() - start_time,
                    "status": "failed",
                    "error": str(e),
                }
            )
            return self._create_result("failed", error=str(e), metrics=metrics)

    async def _read_source_code(self, file_path: Path) -> str:
        """Read source code with encoding detection."""
        try:
            return await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            # Implement encoding detection if needed
            return await asyncio.to_thread(file_path.read_text, encoding="latin-1")

    async def _generate_documentation(
        self,
        updated_code: str,
        documentation: str,
        file_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generate documentation with enhanced formatting and validation."""
        try:
            markdown_gen = MarkdownDocumentationGenerator(
                source_code=updated_code, module_path=str(file_path)
            )

            markdown_gen.add_metadata(
                {
                    "generated_at": datetime.now().isoformat(),
                    "token_usage": self.components.components[
                        "token_manager"
                    ].get_usage_stats(),
                    "ai_model": self.config.model_name,
                }
            )

            if not documentation:
                raise Exception("Generated documentation is empty")

            doc_path = output_dir / f"{file_path.stem}_docs.md"
            await self._save_documentation(doc_path, documentation)

            return {
                "documentation": documentation,
                "documentation_path": str(doc_path),
                "updated_code": updated_code,
            }

        except Exception as e:
            raise Exception(f"Documentation generation failed: {e}") from e

    async def _save_documentation(self, doc_path: Path, documentation: str):
        """Save documentation to file."""
        await asyncio.to_thread(doc_path.write_text, documentation, encoding="utf-8")

    def _create_result(
        self, status: str, result: Optional[Dict] = None, message: str = "", error: str = "", metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
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

    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerSetup.get_logger(__name__)

    async def run(
        self, source_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
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
                processor = DocumentationProcessor(components, self.config)
                if source_path.is_file():
                    file_result = await processor.process_file(source_path, output_dir)
                    self._update_results(results, file_result)
                elif source_path.is_dir():
                    files = list(source_path.rglob("*.py"))
                    async for file in tqdm_asyncio(files, desc="Processing files"):
                        file_result = await processor.process_file(file, output_dir)
                        self._update_results(results, file_result)
                else:
                    raise ValueError(f"Invalid source path: {source_path}")

                await self._generate_summary(
                    results, output_dir, workflow_id, time.time() - start_time
                )

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise WorkflowError(f"Workflow failed: {e}") from e

        return results

    def _update_results(self, results: Dict[str, Any], file_result: Dict[str, Any]) -> None:
        """Update results dictionary with file processing outcomes."""
        status = file_result["status"]
        results[f"{status}_files"] += 1

        token_usage = file_result.get("metrics", {}).get("token_usage", {})
        results["total_tokens"] += token_usage.get("total_tokens", 0)
        results["total_cost"] += token_usage.get("total_cost", 0.0)

    async def _generate_summary(
        self, results: Dict[str, Any], output_dir: Path, workflow_id: str, duration: float
    ) -> None:
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

        await asyncio.to_thread(
            summary_path.write_text, summary_content, encoding="utf-8"
        )
        self.logger.info(f"Workflow summary saved to {summary_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate documentation using AI.")
    parser.add_argument("source_path", type=Path, help="Path to the source code file or directory")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory")
    parser.add_argument("--config-file", type=Path, help="Path to the configuration file")
    return parser.parse_args()


async def run_workflow(args: argparse.Namespace) -> None:
    """Run the documentation generation workflow."""
    try:
        config = await load_and_validate_config(args)
        setup_logging(
            config.log_level, config.log_format, config.log_directory
        )  # Assuming these are in your Config

        async with initialize_components(config) as components:
            orchestrator = WorkflowOrchestrator(config)
            await orchestrator.run(args.source_path, args.output_dir)
            logger.info("Workflow completed successfully")

    except Exception as e:
        handle_workflow_error(e)
        sys.exit(1)


def handle_workflow_error(e: Exception):
    """Handle workflow errors."""
    logger.error(f"Workflow failed: {str(e)}")


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_workflow(args))
```

## Key Changes

- Corrected the indentation in the `AsyncComponentManager` class, particularly in the `__aexit__` method's loop.
- Fixed the indentation in the `DocumentationProcessor` class, specifically in the `process_file` method and its helper methods.
- Ensured consistent indentation throughout the script to adhere to Python's indentation rules.  
[Timestamp: 2024/11/21 10:35:16]
