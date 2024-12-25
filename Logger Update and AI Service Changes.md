---
epoch: 1733599339752
modelKey: gpt-4o-2024-11-20|azure openai
tags:
  - copilot-conversation
---

**user**: [logger.py](logger.py.md) [Logger changes](Logger%20changes.md) [main.py](main.py.md) [docs.py](docs.py.md) [docstring_processor.py](docstring_processor.py.md) [ai_service.py](ai_service.py.md) make the changes necessary outlined in Logger changes to the docs main ai_interaction and docstring_processor modules. Provide the complete, updated files in your response. 
[Timestamp: 2024/12/07 13:22:19]

**ai**: Here are the updated versions of your modules based on the changes outlined in [Logger changes](Logger%20changes.md) for the logger file [logger.py](logger.py.md):

### Updated `main.py`

```python
import argparse
import asyncio
import re
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

from ai_interaction import AIInteractionHandler
from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docs import DocumentationOrchestrator
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.logger import LoggerSetup, log_error, log_debug, log_info
from core.metrics import Metrics, MetricsCollector
from core.monitoring import SystemMonitor
from core.response_parsing import ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext
from core.utils import GitUtils
from exceptions import ConfigurationError, DocumentationError
from repository_handler import RepositoryHandler

load_dotenv()

# Configure logger
LoggerSetup.configure(level="DEBUG")

class DocumentationGenerator:
    """Class responsible for generating documentation for Python files or repositories."""

    def __init__(self, config: Optional[AzureOpenAIConfig] = None) -> None:
        """Initialize the documentation generator.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
        """
        self.config = config or AzureOpenAIConfig.from_env()
        self.logger = LoggerSetup.get_logger(__name__)
        self._initialized = False

        # Store components but don't initialize them yet
        self.api_client: Optional[APIClient] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.doc_orchestrator: Optional[DocumentationOrchestrator] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize components
            self.api_client = APIClient(
                config=self.config,
                response_parser=ResponseParsingService(),
                token_manager=TokenManager()
            )
            self.ai_handler = AIInteractionHandler(
                config=self.config,
                cache=Cache(),
                token_manager=TokenManager(),
                response_parser=ResponseParsingService(),
                metrics=Metrics(),
                docstring_schema=load_schema("docstring_schema")
            )
            self.doc_orchestrator = DocumentationOrchestrator(
                ai_handler=self.ai_handler,
                docstring_processor=DocstringProcessor(metrics=Metrics()),
                code_extractor=CodeExtractor(ExtractionContext()),
                metrics=Metrics(),
                response_parser=ResponseParsingService()
            )
            self.system_monitor = SystemMonitor()
            self.metrics_collector = MetricsCollector()

            # Start monitoring
            await self.system_monitor.start()
            self._initialized = True
            self.logger.info("All components initialized successfully")
        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            log_error(error_msg, exc_info=True, extra={"error": str(e)})
            await self.cleanup()
            raise ConfigurationError(error_msg) from e

    async def process_file(self, file_path: Path, output_path: Path) -> bool:
        """Process a single file.

        Args:
            file_path (Path): Path to the file to process.
            output_path (Path): Path where the output should be saved.

        Returns:
            bool: True if processing was successful, otherwise False.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info("Processing file", extra={"file_path": str(file_path)})

            if not file_path.exists():
                self.logger.error("File not found", extra={"file_path": str(file_path)})
                return False

            try:
                source_code = file_path.read_text(encoding='utf-8')
                source_code = self._fix_indentation(source_code)
                if not source_code.strip():
                    self.logger.error("Empty file", extra={"file_path": str(file_path)})
                    return False
            except UnicodeDecodeError:
                try:
                    source_code = file_path.read_text(encoding='latin-1')
                    source_code = self._fix_indentation(source_code)
                except Exception as e:
                    self.logger.error("Failed to read file", exc_info=True, extra={"file_path": str(file_path), "error": str(e)})
                    return False

            result = await self.ai_handler.process_code(source_code)
            if not result:
                self.logger.error("No documentation generated", extra={"file_path": str(file_path)})
                return False

            updated_code = result.get('code')
            documentation = result.get('documentation')

            if not updated_code or not documentation:
                self.logger.error("Invalid result", extra={"file_path": str(file_path)})
                return False

            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")

            self.logger.info("Successfully processed file", extra={"file_path": str(file_path)})
            return True

        except Exception as e:
            self.logger.error("Error processing file", exc_info=True, extra={"file_path": str(file_path), "error": str(e)})
            return False

    def _fix_indentation(self, source_code: str) -> str:
        """Fix inconsistent indentation in source code.

        Args:
            source_code (str): The source code to fix.

        Returns:
            str: The corrected source code with consistent indentation.
        """
        lines = source_code.splitlines()
        fixed_lines: List[str] = []

        for line in lines:
            # Convert tabs to spaces
            fixed_line = line.expandtabs(4)
            indent_count = len(fixed_line) - len(fixed_line.lstrip())
            if indent_count > 0:
                proper_indent = (indent_count // 4) * 4
                fixed_line = (" " * proper_indent) + fixed_line.lstrip()

            fixed_lines.append(fixed_line)

        return "\n".join(fixed_lines)

    async def process_repository(self, repo_path: str, output_dir: Path = Path("docs")) -> bool:
        """Process a repository for documentation.

        Args:
            repo_path (str): The path to the repository to process.
            output_dir (Path): The output directory for documentation.

        Returns:
            bool: True if the repository was processed successfully, otherwise False.
        """
        try:
            if self._is_valid_url(repo_path):
                self.logger.info("Cloning repository from URL", extra={"repo_url": repo_path})
                async with RepositoryHandler(repo_path=Path.cwd()) as repo_handler:
                    cloned_repo_path = await repo_handler.clone_repository(repo_path)
                    self.logger.info("Repository cloned", extra={"cloned_repo_path": str(cloned_repo_path)})
                    success = await self._process_local_repository(cloned_repo_path, output_dir)
                    return success
            else:
                self.logger.info("Processing local repository", extra={"repo_path": repo_path})
                local_repo_path = Path(repo_path)
                if not local_repo_path.exists():
                    raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
                success = await self._process_local_repository(local_repo_path, output_dir)
                return success
        except Exception as e:
            self.logger.error("Repository processing failed", exc_info=True, extra={"repo_path": repo_path, "error": str(e)})
            return False

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, otherwise False.
        """
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
            r'(?:\S+(?::\S*)?@)?'  # user and password
            r'(?:'
            r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])\.){3}'
            r'(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5])'  # IP address
            r'|'
            r'(?:(?:[a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,})'  # Domain name
            r')'
            r'(?::\d{2,5})?'  # Port
            r'(?:/\S*)?',
            re.IGNORECASE
        )
        return re.match(regex, url) is not None

    async def _process_local_repository(self, repo_path: Path, output_dir: Path) -> bool:
        """Process a local repository for documentation.

        Args:
            repo_path (Path): The path to the local repository.
            output_dir (Path): The output directory for documentation.

        Returns:
            bool: True if the local repository was processed successfully, otherwise False.
        """
        try:
            python_files = GitUtils.get_python_files(repo_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            for file_path in python_files:
                output_file = output_dir / (file_path.stem + ".md")
                success = await self.process_file(file_path, output_file)
                if not success:
                    self.logger.error("Failed to process file", extra={"file_path": str(file_path)})
            return True
        except Exception as e:
            self.logger.error("Error processing local repository", exc_info=True, extra={"repo_path": str(repo_path), "error": str(e)})
            return False

    async def display_metrics(self) -> None:
        """Display collected metrics and system performance metrics."""
        try:
            collected_metrics = self.metrics_collector.get_metrics()
            system_metrics = self.system_monitor.get_metrics()

            print("=== Documentation Generation Metrics ===")
            for metric in collected_metrics['operations']:
                print(f"Operation: {metric['operation_type']}")
                print(f"Success: {metric['success']}")
                print(f"Duration: {metric['duration']} seconds")
                print(f"Usage: {metric['usage']}")
                print(f"Validation Success: {metric['validation_success']}")
                print(f"Timestamp: {metric['timestamp']}")
                print("-" * 40)

            print("=== System Performance Metrics ===")
            print(system_metrics)
            print("-" * 40)

        except Exception as e:
            self.logger.error("Error displaying metrics", exc_info=True, extra={"error": str(e)})
            print(f"Error displaying metrics: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources used by the DocumentationGenerator."""
        try:
            if self.api_client:
                await self.api_client.close()
            if self.ai_handler:
                await self.ai_handler.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.system_monitor:
                await self.system_monitor.stop()

            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error("Unexpected error during cleanup", exc_info=True, extra={"error": str(e)})


async def main(args: argparse.Namespace) -> int:
    """Main function to manage the documentation generation process based on arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    doc_generator = DocumentationGenerator()
    try:
        await doc_generator.initialize()

        if args.repository:
            success = await doc_generator.process_repository(args.repository, Path(args.output))
            if success:
                print("Repository documentation generated successfully.")
            else:
                print("Failed to generate repository documentation.")

        if args.files:
            for file in args.files:
                file_path = Path(file)
                output_path = Path(args.output) / file_path.stem
                success = await doc_generator.process_file(file_path, output_path)
                if success:
                    print(f"Documentation for {file} generated successfully.")
                else:
                    print(f"Failed to generate documentation for {file}.")

        await doc_generator.display_metrics()

    except DocumentationError as de:
        log_error("Documentation generation failed", exc_info=True, extra={"error": str(de)})
        return 1
    except Exception as e:
        log_error("Unexpected error", exc_info=True, extra={"error": str(e)})
        return 1
    finally:
        await doc_generator.cleanup()

    return 0


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from the command line.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files or repositories."
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="URL or local path of the repository to process",
    )
    parser.add_argument("--files", nargs="+", help="Python files to process")
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        log_info("Documentation generation interrupted by user.")
    except Exception as e:
        log_error("Failed to run documentation generator", exc_info=True, extra={"error": str(e)})
    finally:
        LoggerSetup.shutdown()
```

### Updated `docs.py`

```python
from typing import TYPE_CHECKING, Type, Union, Any, Optional, Tuple, Dict, List
from pathlib import Path
from datetime import datetime
import json

if TYPE_CHECKING:
    from ai_interaction import AIInteractionHandler

from core.logger import LoggerSetup, log_info, log_debug, log_error
from core.types import DocumentationContext, ExtractionContext, ExtractionResult
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.metrics import Metrics
from core.schema_loader import load_schema
from exceptions import DocumentationError

# Set up the logger
logger = LoggerSetup.get_logger(__name__)

class DocumentationOrchestrator:
    """Orchestrates the process of generating documentation from source code files.

    Attributes:
        ai_handler (AIInteractionHandler): Handles AI interactions for generating docstrings.
        docstring_schema (Dict[str, Any]): Schema used for validating docstring formats.
        metrics (Metrics): Assists in calculating various code metrics.
        code_extractor (CodeExtractor): Parses and extracts code elements from source code.
        docstring_processor (DocstringProcessor): Processes and validates docstrings.
        response_parser (ResponseParsingService): Parses responses from the AI model.
    """

    def __init__(
        self,
        ai_handler: Optional['AIInteractionHandler'] = None,
        docstring_processor: Optional[DocstringProcessor] = None,
        code_extractor: Optional[CodeExtractor] = None,
        metrics: Optional[Metrics] = None,
        response_parser: Optional[ResponseParsingService] = None
    ) -> None:
        """Initialize the DocumentationOrchestrator.

        Args:
            ai_handler (Optional[AIInteractionHandler]): The AI handler for generating docstrings.
            docstring_processor (Optional[DocstringProcessor]): Manager for processing docstrings.
            code_extractor (Optional[CodeExtractor]): Extractor for code elements from source.
            metrics (Optional[Metrics]): Placeholder for handling different metrics.
            response_parser (Optional[ResponseParsingService]): Parser for AI model responses.

        Raises:
            ValueError: If ai_handler is not provided.
        """
        self.logger = logger
        if ai_handler is None:
            raise ValueError("ai_handler is required for DocumentationOrchestrator")
        self.ai_handler: 'AIInteractionHandler' = ai_handler
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")
        self.metrics: Metrics = metrics or Metrics()
        self.code_extractor: CodeExtractor = code_extractor or CodeExtractor(ExtractionContext())
        self.docstring_processor: DocstringProcessor = docstring_processor or DocstringProcessor(metrics=self.metrics)
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()

    async def generate_documentation(self, context: DocumentationContext) -> Tuple[str, str]:
        """Generate documentation for the given source code.

        Args:
            context (DocumentationContext): Holds information regarding the source code and its environment.

        Returns:
            Tuple[str, str]: The updated source code and generated documentation.

        Raises:
            DocumentationError: If code elements extraction or AI response validation fails.
        """
        try:
            log_info("Starting documentation generation process")

            extraction_result: Optional[ExtractionResult] = await self.code_extractor.extract_code(context.source_code)
            if not extraction_result:
                log_error("Failed to extract code elements")
                raise DocumentationError("Code elements extraction failed")
            log_info("Code extraction completed successfully")

            extraction_dict = extraction_result.to_dict()
            prompt: str = await self.ai_handler.create_dynamic_prompt(extraction_dict)
            log_info("Prompt generated for AI model")

            ai_response: Union[str, Dict[str, Any]] = await self.ai_handler._interact_with_ai(prompt)
            log_info("AI interaction completed successfully")

            parsed_response = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )
            if not parsed_response.validation_success:
                log_error("Validation failed for AI response")
                raise DocumentationError("AI response validation failed")
            log_info("AI response parsed and validated successfully")

            updated_code, updated_documentation = await self.ai_handler._integrate_ai_response(
                parsed_response.content, extraction_result
            )
            log_info("Docstring integration completed successfully")

            log_info("Documentation generation completed successfully")
            return updated_code, updated_documentation

        except DocumentationError as de:
            log_error(f"Documentation error: {de}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during documentation generation: {e}"
            log_error(error_msg)
            raise DocumentationError(error_msg)

    async def generate_module_documentation(self, file_path: Path, output_dir: Path) -> None:
        """Generate documentation for a single module.

        Args:
            file_path (Path): Path to the module file.
            output_dir (Path): Directory where the documentation will be output.

        Raises:
            DocumentationError: If there is an error during documentation generation.
        """
        try:
            log_info(f"Generating documentation for {file_path}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.with_suffix(".md").name

            source_code = file_path.read_text(encoding="utf-8")
            context = DocumentationContext(
                source_code=source_code,
                module_path=file_path,
                include_source=True,
                metadata={
                    "file_path": str(file_path),
                    "module_name": file_path.stem,
                    "creation_time": datetime.now().isoformat(),
                },
            )

            updated_code, documentation = await self.generate_documentation(context)
            output_path.write_text(documentation, encoding="utf-8")
            file_path.write_text(updated_code, encoding="utf-8")
            log_info(f"Documentation written to {output_path}")

        except Exception as e:
            error_msg = f"Module documentation generation failed for {file_path}: {e}"
            log_error(error_msg)
            raise DocumentationError(error_msg)

    async def generate_batch_documentation(self, file_paths: List[Path], output_dir: Path) -> Dict[Path, bool]:
        """Generate documentation for multiple files.

        Args:
            file_paths (List[Path]): List of file paths for which documentation should be generated.
            output_dir (Path): Directory where documentation results will be stored.

        Returns:
            Dict[Path, bool]: A dictionary mapping each file path to a boolean indicating success.
        """
        results = {}
        for file_path in file_paths:
            try:
                await self.generate_module_documentation(file_path, output_dir)
                results[file_path] = True
            except DocumentationError as e:
                log_error(f"Failed to generate docs for {file_path}: {e}")
                results[file_path] = False
        return results

    def _generate_markdown(self, context: Dict[str, Any]) -> str:
        """Generate markdown documentation.

        Args:
            context (Dict[str, Any]): Context used to generate documentation.

        Returns:
            str: Generated markdown documentation.
        """
        return "Generated markdown documentation"

    async def __aenter__(self) -> "DocumentationOrchestrator":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        """Exit the async context manager."""
        pass
```

### Updated `docstring_processor.py`

```python
"""
Module for processing and validating docstrings.

This module provides utilities for parsing, validating, and formatting docstrings in Python code.
It supports parsing docstrings from various formats, validating them against a schema, integrating
them into source code, and generating updated source code from modified ASTs.

Classes:
    DocstringProcessor: Class for processing and validating docstrings.
"""

import ast
import json
from typing import Optional, Dict, Any, List, Union, Tuple

from docstring_parser import parse as parse_docstring, Docstring

from core.logger import LoggerSetup, log_error, log_debug, log_info
from core.metrics import Metrics
from core.types import DocstringData
from exceptions import DocumentationError
from core.utils import FormattingUtils, ValidationUtils
from core.schema_loader import load_schema

try:
    import astor
except ImportError as e:
    raise ImportError(
        "The 'astor' library is required for Python versions < 3.9 to generate code from AST. "
        "Please install it using 'pip install astor'."
    ) from e

# Set up the logger
logger = LoggerSetup.get_logger(__name__)

class DocstringProcessor:
    """
    Processes docstrings by parsing and validating them.

    This class provides methods to parse raw docstrings, validate them,
    integrate them into source code, and generate documentation.
    """

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        """Initialize the DocstringProcessor.

        Args:
            metrics (Optional[Metrics]): The metrics instance for handling code metrics.
        """
        self.logger = logger
        self.metrics = metrics or Metrics()
        self.docstring_schema: Dict[str, Any] = load_schema("docstring_schema")

    def _validate_docstring_dict(self, docstring_dict: Dict[str, Any]) -> None:
        """Validate that required keys exist in the docstring dictionary.

        Args:
            docstring_dict (Dict[str, Any]): The docstring dictionary to validate.

        Raises:
            DocumentationError: If required keys are missing from the docstring dictionary.
        """
        required_keys = {'summary', 'description', 'args', 'returns', 'raises'}
        missing_keys = required_keys - docstring_dict.keys()
        if missing_keys:
            raise DocumentationError(f"Docstring dictionary missing keys: {missing_keys}")

    def parse(self, docstring: Union[Dict[str, Any], str]) -> DocstringData:
        """Parse a docstring from a string or dictionary.

        Args:
            docstring (Union[Dict[str, Any], str]): The docstring to parse.

        Returns:
            DocstringData: A structured representation of the parsed docstring.

        Raises:
            DocumentationError: If the docstring cannot be parsed.
        """
        try:
            if isinstance(docstring, dict):
                self._validate_docstring_dict(docstring)
                if 'complexity' not in docstring:
                    docstring['complexity'] = 1
                return self._create_docstring_data_from_dict(docstring)
            elif isinstance(docstring, str):
                docstring_str = docstring.strip()
                if docstring_str.startswith('{') and docstring_str.endswith('}'):
                    try:
                        doc_dict = json.loads(docstring_str)
                        if 'complexity' not in doc_dict:
                            doc_dict['complexity'] = 1
                        return self.parse(doc_dict)
                    except json.JSONDecodeError as e:
                        log_error(f"JSON parsing error: {e}")
                        raise DocumentationError(f"Invalid JSON format in docstring: {e}") from e

                try:
                    parsed: Docstring = parse_docstring(docstring_str)
                    docstring_dict = {
                        'summary': parsed.short_description or '',
                        'description': parsed.long_description or '',
                        'args': [{'name': p.arg_name, 'type': p.type_name or 'Any', 'description': p.description or ''}
                                 for p in parsed.params],
                        'returns': {
                            'type': parsed.returns.type_name if parsed.returns else 'Any',
                            'description': parsed.returns.description if parsed.returns else ''
                        },
                        'raises': [{'exception': e.type_name, 'description': e.description}
                                   for e in (parsed.raises or [])],
                        'complexity': 1
                    }
                    return self._create_docstring_data_from_dict(docstring_dict)
                except Exception as e:
                    log_error(f"Docstring parsing error: {e}")
                    raise DocumentationError(f"Failed to parse docstring: {e}") from e
            raise DocumentationError("Docstring must be either a dictionary or a string.")
        except DocumentationError:
            raise
        except Exception as e:
            log_error(f"Unexpected error in parse method: {e}")
            raise DocumentationError(f"Unexpected error during parsing: {e}") from e

    def _create_docstring_data_from_dict(self, docstring_dict: Dict[str, Any]) -> DocstringData:
        """Create DocstringData from a dictionary representation.

        Args:
            docstring_dict (Dict[str, Any]): The dictionary containing docstring information.

        Returns:
            DocstringData: The structured docstring data.

        Raises:
            DocumentationError: If creation of DocstringData fails.
        """
        try:
            docstring_dict['args'] = docstring_dict.get('args', [])
            docstring_dict['raises'] = docstring_dict.get('raises', [])
            returns = docstring_dict.get('returns', {})
            if not isinstance(returns, dict):
                returns = {'type': 'Any', 'description': ''}
            if not returns.get('type'):
                returns['type'] = 'Any'
            if not returns.get('description'):
                returns['description'] = ''

            complexity = docstring_dict.get('complexity', 1)

            return DocstringData(
                summary=docstring_dict.get('summary', ''),
                description=docstring_dict.get('description', ''),
                args=docstring_dict['args'],
                returns=returns,
                raises=docstring_dict['raises'],
                complexity=complexity
            )
        except Exception as e:
            log_error(f"Error creating DocstringData from dict: {e}")
            raise DocumentationError(f"Failed to create DocstringData from dict: {e}") from e

    def process_batch(self, doc_entries: List[Dict[str, Any]], source_code: str) -> Dict[str, str]:
        """Process a batch of docstring entries and integrate them into the source code.

        Args:
            doc_entries (List[Dict[str, Any]]): The docstring entries to process.
            source_code (str): The source code to integrate the docstrings into.

        Returns:
            Dict[str, str]: A dictionary containing the updated code and documentation.
        """
        try:
            tree = ast.parse(source_code)
            log_debug(f"Processing {len(doc_entries)} documentation entries")

            processed_entries: List[Dict[str, Any]] = []
            for entry in doc_entries:
                try:
                    log_debug(f"Processing entry: {entry}")

                    if not isinstance(entry, dict):
                        log_error(f"Entry is not a dictionary: {type(entry)}")
                        continue

                    if 'summary' in entry and 'name' not in entry:
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                                docstring = self.format(DocstringData(**entry))
                                processed_entries.append({
                                    'name': node.name,
                                    'docstring': docstring,
                                    'type': type(node).__name__
                                })
                                log_debug(f"Created processed entry for {node.name}")
                                break
                    else:
                        if 'docstring' not in entry and 'summary' in entry:
                            entry['docstring'] = self.format(DocstringData(**entry))
                        processed_entries.append(entry)
                        log_debug(f"Added entry with name: {entry.get('name')}")

                except Exception as e:
                    log_error(f"Error processing entry: {e}")
                    continue

            if not processed_entries:
                log_error("No valid entries were processed")
                return {'code': source_code, 'documentation': ""}

            updated_tree: Optional[ast.AST] = self._insert_docstrings(tree, processed_entries)
            if not updated_tree:
                log_error("Failed to update AST with docstrings")
                return {'code': source_code, 'documentation': ""}

            updated_code: Optional[str] = self._generate_code_from_ast(updated_tree)
            if not updated_code:
                log_error("Failed to generate code from AST")
                return {'code': source_code, 'documentation': ""}

            documentation: str = self._generate_documentation(processed_entries)
            if not documentation:
                log_error("Failed to generate documentation")
                return {'code': updated_code, 'documentation': ""}

            log_info("Successfully processed batch")
            return {'code': updated_code, 'documentation': documentation}

        except Exception as e:
            log_error(f"Unexpected error processing batch: {e}")
            return {'code': source_code, 'documentation': ""}

    def _generate_documentation(self, doc_entries: List[Dict[str, Any]]) -> str:
        """Generate markdown documentation from processed docstring entries.

        Args:
            doc_entries (List[Dict[str, Any]]): The processed docstring entries.

        Returns:
            str: The generated markdown documentation.
        """
        try:
            log_debug(f"Generating documentation for {len(doc_entries)} entries")
            doc_parts: List[str] = ["# API Documentation\n\n"]

            module_entry = next((entry for entry in doc_entries if entry.get('name') == '__module__'), None)
            if module_entry:
                doc_parts.extend([
                    "## Module Overview\n\n",
                    f"{module_entry.get('docstring', '')}\n\n"
                ])

            class_entries = [entry for entry in doc_entries if entry.get('type') == 'ClassDef']
            if class_entries:
                doc_parts.append("## Classes\n\n")
                for entry in class_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            func_entries = [entry for entry in doc_entries if entry.get('type') in ('FunctionDef', 'AsyncFunctionDef')]
            if func_entries:
                doc_parts.append("## Functions\n\n")
                for entry in func_entries:
                    doc_parts.extend([
                        f"### {entry['name']}\n\n",
                        f"{entry.get('docstring', '')}\n\n"
                    ])

            documentation: str = ''.join(doc_parts)
            log_debug(f"Generated documentation length: {len(documentation)}")
            return documentation

        except Exception as e:
            log_error(f"Error generating documentation: {e}")
            return ""

    def _insert_docstrings(self, tree: ast.AST, doc_entries: List[Dict[str, Any]]) -> Optional[ast.AST]:
        """Insert docstrings into the AST at relevant locations for each entry.

        Args:
            tree (ast.AST): The AST representation of the code.
            doc_entries (List[Dict[str, Any]]): The list of processed docstrings and their locations.

        Returns:
            Optional[ast.AST]: The updated AST with docstrings inserted, or None on failure.
        """
        try:
            docstring_map: Dict[str, str] = {
                entry['name']: entry['docstring']
                for entry in doc_entries
                if 'name' in entry and 'docstring' in entry
            }

            log_debug(f"Created docstring map with {len(docstring_map)} entries")

            class DocstringInserter(ast.NodeTransformer):
                def visit_Module(self, node: ast.Module) -> ast.Module:
                    self.generic_visit(node)
                    if '__module__' in docstring_map and not ast.get_docstring(node):
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring_map['__module__']))
                        node.body.insert(0, docstring_node)
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                    return self.visit_FunctionDef(node)

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                    self.generic_visit(node)
                    if node.name in docstring_map:
                        docstring = docstring_map[node.name]
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                            node.body[0] = docstring_node
                        else:
                            node.body.insert(0, docstring_node)
                    return node

            transformer = DocstringInserter()
            new_tree = transformer.visit(tree)
            return new_tree

        except Exception as e:
            log_error(f"Error inserting docstrings: {e}")
            return None

    def _generate_code_from_ast(self, tree: ast.AST) -> Optional[str]:
        """Generate source code from an AST.

        Args:
            tree (ast.AST): The AST representation of the code.

        Returns:
            Optional[str]: The generated source code, or None on failure.
        """
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(tree)
            else:
                import astor
                return astor.to_source(tree)
        except Exception as e:
            log_error(f"Error generating code from AST: {e}")
            return None

    def format(self, data: DocstringData) -> str:
        """Format the docstring data into a human-readable string.

        Args:
            data (DocstringData): The data object containing docstring information.

        Returns:
            str: The formatted docstring.
        """
        return FormattingUtils.format_docstring(data.__dict__)

    def validate(self, data: DocstringData) -> Tuple[bool, List[str]]:
        """Validate the docstring data against the schema.

        Args:
            data (DocstringData): The data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing success flag and list of validation errors.
        """
        return ValidationUtils.validate_docstring(data.__dict__, self.docstring_schema)
```

### Updated `ai_interaction.py`

```python
"""
This module provides classes and functions for handling AI interactions, processing source code,
generating dynamic prompts, and integrating AI-generated documentation back into the source code.

Classes:
    CustomJSONEncoder: Custom JSON encoder that can handle sets and other non-serializable types.
    AIInteractionHandler: Handles AI interactions for generating enriched prompts and managing responses.

Functions:
    serialize_for_logging(obj: Any) -> str: Safely serialize any object for logging purposes.
"""

import ast
import asyncio
import json
import re
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI

from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.markdown_generator import MarkdownGenerator
from core.metrics import Metrics
from core.response_parsing import ParsedResponse, ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext, ExtractionResult
from exceptions import ConfigurationError, ProcessingError

from logger import LoggerSetup, log_error, log_debug, log_info, log_warning

# Set up the logger
logger = LoggerSetup.get_logger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
        """Override the default JSON encoding.

        Args:
            obj (Any): The object to encode.

        Returns:
            Any: The JSON-encoded version of the object or a string representation.
        """
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, types.ModuleType)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return super().default(obj)


def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging purposes.

    Args:
        obj (Any): The object to serialize.

    Returns:
        str: A JSON string representation of the object.
    """
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"


class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Optional[Cache] = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: Dict[str, Any] = docstring_schema or load_schema()

        try:
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint
            )
            log_info("AI client initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize AI client: {e}")
            raise

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response (Union[str, Dict[str, Any]]): The response to truncate (either string or dictionary).
            length (int): Maximum length of the truncated response.

        Returns:
            str: Truncated string representation of the response.
        """
        try:
            if isinstance(response, dict):
                json_str = json.dumps(response, indent=2)
                return (json_str[:length] + "...") if len(json_str) > length else json_str
            elif isinstance(response, str):
                return (response[:length] + "...") if len(response) > length else response
            else:
                str_response = str(response)
                return (str_response[:length] + "...") if len(str_response) > length else str_response
        except Exception as e:
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Process the source code to extract metadata, interact with the AI, and integrate responses.

        Args:
            source_code (str): The source code to process.

        Returns:
            Optional[Dict[str, Any]]: The updated code and documentation if successful, otherwise None.
        """
        try:
            tree: ast.AST = ast.parse(source_code)
            context = ExtractionContext()
            context.source_code = source_code
            context.tree = tree

            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                match = re.search(r"Module:?\s*([^\n\.]+)", module_docstring)
                if match:
                    context.module_name = match.group(1).strip()

            extractor = CodeExtractor(context)
            extraction_result: Optional[ExtractionResult] = await extractor.extract_code(source_code)
            if not extraction_result:
                log_error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            log_debug("Generated prompt for AI")

            ai_response: Union[str, Dict[str, Any]] = await self._interact_with_ai(prompt)
            log_debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                log_error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                log_error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            log_error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            log_error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.

        Args:
            ai_response (Dict[str, Any]): The AI-generated response to integrate.
            extraction_result (ExtractionResult): The extraction result containing the source code.

        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            log_debug("Starting AI response integration")
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = self._create_processed_response(ai_response)

            integration_result = self._process_docstrings(processed_response, extraction_result.source_code)
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            if not isinstance(code, str):
                raise ProcessingError("Expected 'code' to be a string in integration result")

            documentation = self._generate_markdown_documentation(ai_response, extraction_result)
            return code, documentation

        except Exception as e:
            log_error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """
        Generate markdown documentation from AI response and extraction result.

        Args:
            ai_response (Dict[str, Any]): The AI response data.
            extraction_result (ExtractionResult): The extraction result containing source code info.

        Returns:
            str: Generated markdown documentation.
        """
        markdown_gen = MarkdownGenerator()
        markdown_context: Dict[str, Any] = {
            "module_name": extraction_result.module_name,
            "file_path": extraction_result.file_path,
            "description": ai_response.get("description", ""),
            "classes": extraction_result.classes,
            "functions": extraction_result.functions,
            "constants": extraction_result.constants,
            "source_code": extraction_result.source_code,
            "ai_documentation": ai_response,
        }
        return markdown_gen.generate(markdown_context)

    def _ensure_required_fields(self, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the AI response has the required fields, and repair if necessary.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            Dict[str, Any]: The AI response with ensured fields.
        """
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            log_error(f"AI response missing required fields: {missing}")

            for field in missing:
                if field == "args":
                    ai_response["args"] = []
                elif field == "returns":
                    ai_response["returns"] = {"type": "None", "description": ""}
                elif field == "raises":
                    ai_response["raises"] = []
                else:
                    ai_response[field] = ""
        return ai_response

    def _create_processed_response(self, ai_response: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Create a list with the processed response.

        Args:
            ai_response (Dict[str, Any]): The AI response data.

        Returns:
            List[Dict[str, Union[str, Dict[str, Any]]]]: Processed response for integration.
        """
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """
        Process the docstrings using DocstringProcessor.

        Args:
            processed_response (List[Dict[str, Union[str, Dict[str, Any]]]]): The processed response data.
            source_code (str): The source code to modify.

        Returns:
            Dict[str, Any]: The integration result with updated code and docstrings.
        """
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        log_debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> Union[str, Dict[str, Any]]:
        """
        Interact with the AI model to generate responses.

        Args:
            prompt (str): The prompt to send to the AI model.

        Returns:
            Union[str, Dict[str, Any]]: The AI-generated response.
        """
        try:
            request_params: Dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            log_debug("Sending request to AI")

            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[
                    {"role": "system", "content": "You are a Python documentation expert. Generate complete docstrings in Google format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=request_params.get('temperature', 0.7)
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            log_debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                log_debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse AI response as JSON: {e}")

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        log_debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        log_error(f"Failed to parse extracted JSON: {e2}")

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            log_error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            log_error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise Exception(f"Unexpected error processing request: {str(e)}")

    async def create_dynamic_prompt(self, extracted_info: Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> str:
        """
        Create a dynamic prompt for the AI model.

        Args:
            extracted_info (Dict[str, Union[str, List[Dict[str, Any]], Dict[str, Any]]]): The extracted information for the prompt.

        Returns:
            str: The generated prompt.
        """
        try:
            log_debug("Creating dynamic prompt")
            log_debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

            prompt_parts: List[str] = [
                "Generate a complete Python documentation structure as a single JSON object.\n\n",
                "Required JSON structure:\n",
                "{\n",
                '  "summary": "One-line summary of the code",\n',
                '  "description": "Detailed description of functionality",\n',
                '  "args": [{"name": "param1", "type": "str", "description": "param description"}],\n',
                '  "returns": {"type": "ReturnType", "description": "return description"},\n',
                '  "raises": [{"exception": "ValueError", "description": "error description"}]\n',
                "}\n\n",
                "Code Analysis:\n"
            ]

            if extracted_info.get("module_docstring"):
                prompt_parts.append(f"Current Module Documentation:\n{extracted_info['module_docstring']}\n\n")

            if extracted_info.get("classes"):
                prompt_parts.append("Classes:\n")
                for cls in extracted_info["classes"]:
                    prompt_parts.append(f"- {cls['name']}\n")
                    if cls.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {cls['docstring']}\n")
                    if cls.get("methods"):
                        prompt_parts.append("  Methods:\n")
                        for method in cls["methods"]:
                            prompt_parts.append(f"    - {method['name']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("functions"):
                prompt_parts.append("Functions:\n")
                for func in extracted_info["functions"]:
                    prompt_parts.append(f"- {func['name']}\n")
                    if func.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {func['docstring']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("dependencies"):
                prompt_parts.append("Dependencies:\n")
                for dep_type, deps in extracted_info["dependencies"].items():
                    if dep_type == "maintainability_impact":
                        # Handle maintainability_impact separately
                        prompt_parts.append(f"- {dep_type}: {deps}\n")
                        continue

                    if not isinstance(deps, (list, set, tuple)):
                        log_error(f"Non-iterable dependencies for {dep_type}: {deps} ({type(deps)})")
                        continue

                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            log_debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            log_error(f"Error creating prompt: {e}", exc_info=True)
            raise Exception(f"Error creating dynamic prompt: {str(e)}")

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name (str): Name of the function or class.
            is_class (bool): Whether the target is a class.
            params (Optional[List[Dict[str, Any]]]): List of parameters with their types and descriptions.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring to enhance.
            decorators (Optional[List[str]]): List of decorators applied to the function.
            exceptions (Optional[List[Dict[str, str]]]): List of exceptions raised by the function.

        Returns:
            Dict[str, Any]: Generated docstring content.
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info: Dict[str, Any] = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                stop=["END"],
            )
            response_content: Optional[str] = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            log_info(f"Generated docstring for {func_name}")
            return parsed_response.content

        except json.JSONDecodeError as e:
            log_error(f"JSON decoding error while generating docstring for {func_name}: {e}")
            raise
        except ProcessingError as e:
            log_error(f"Processing error while generating docstring for {func_name}: {e}")
            raise
        except Exception as e:
            log_error(f"Unexpected error generating docstring for {func_name}: {e}")
            raise

    async def _verify_deployment(self) -> bool:
        """
        Verify that the configured deployment exists and is accessible.

        Returns:
            bool: True if the deployment exists and is accessible, otherwise False.
        """
        try:
            test_params: Dict[str, Any] = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }
            log_debug(f"Verifying deployment with parameters: {test_params}")
            response = await self.client.chat.completions.create(**test_params)
            log_debug(f"Deployment verification response: {response}")
            return True
        except Exception as e:
            log_error(f"Deployment verification failed: {e}", exc_info=True)
            return False

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Verifies the deployment configuration and raises a ConfigurationError if the deployment is not accessible.

        Returns:
            AIInteractionHandler: Initialized AIInteractionHandler instance.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.
        """
        log_debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            log_debug("Cache resources have been cleaned up")
        log_info("AIInteractionHandler resources have been cleaned up")
```

These updated modules now utilize the module-level logging functions (`log_error`, `log_debug`, `log_info`, and `log_warning`) of the new `LoggerSetup` class. This provides a more streamlined and centralized logging approach across your codebase.
[Timestamp: 2024/12/07 13:25:08]