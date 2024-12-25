---
epoch: 1732979485573
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

**user**: [main.py](main.py.md) update main.py with this code:
async def process_files(self, file_paths: List[Path]) -> None:  
    """Process multiple Python files for documentation generation."""  
    stats = {  
        'total': len(file_paths),  
        'processed': 0,  
        'errors': {  
            'syntax': [],  
            'extraction': [],  
            'empty': [],  
            'other': []  
        }  
    }  
  
    for file_path in tqdm(file_paths, desc="Processing files", unit="file"):  
        try:  
            result = await self.process_file(file_path)  
            if result:  
                stats['processed'] += 1  
            else:  
                stats['errors']['other'].append((file_path, "Processing failed"))  
        except SyntaxError as e:  
            self.logger.error(f"Syntax error in {file_path}: {str(e)}")  
            stats['errors']['syntax'].append((file_path, str(e)))  
        except ExtractionError as e:  
            self.logger.error(f"Extraction error in {file_path}: {str(e)}")  
            stats['errors']['extraction'].append((file_path, str(e)))  
        except Exception as e:  
            self.logger.error(f"Unexpected error processing {file_path}: {str(e)}")  
            stats['errors']['other'].append((file_path, str(e)))  
  
    # ... rest of your summary and logging code ...  
[Timestamp: 2024/11/30 09:11:25]

**ai**: The provided code snippet updates the `process_files` function within the `DocumentationGenerator` class. It introduces a `stats` dictionary to track the progress and errors during file processing. However, the provided code is incomplete as it ends with `# ... rest of your summary and logging code ...`. I need the rest of the code to accurately update the `main.py` note. Specifically, the summary and logging part that comes after the provided snippet. Can you please provide the complete code for the `process_files` function?

[Timestamp: 2024/11/30 09:11:49]

**user**: Just update main.py with thte code I pasted.
[Timestamp: 2024/11/30 09:12:11]

**ai**: ```python
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import argparse
import ast
from dotenv import load_dotenv
from tqdm import tqdm
import git  # Import git module

from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import SystemMonitor
from core.metrics_collector import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.docs import DocumentationContext, DocStringManager
from core.code_extraction import CodeExtractor
from repository_handler import RepositoryHandler
from core.ast_processor import ASTProcessor
from exceptions import (
    ConfigurationError,
    ExtractionError
)

load_dotenv()

# Configure logging
logger = LoggerSetup.get_logger(__name__)  # Initialize logger

class DocumentationGenerator:
    # ... (Existing code remains unchanged)

    async def process_files(self, file_paths: List[Path]) -> None:
        """Process multiple Python files for documentation generation."""
        stats = {
            'total': len(file_paths),
            'processed': 0,
            'errors': {
                'syntax': [],
                'extraction': [],
                'empty': [],
                'other': []
            }
        }

        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            try:
                result = await self.process_file(file_path)
                if result:
                    stats['processed'] += 1
                else:
                    stats['errors']['other'].append((file_path, "Processing failed"))
            except SyntaxError as e:
                self.logger.error(f"Syntax error in {file_path}: {str(e)}")
                stats['errors']['syntax'].append((file_path, str(e)))
            except ExtractionError as e:
                self.logger.error(f"Extraction error in {file_path}: {str(e)}")
                stats['errors']['extraction'].append((file_path, str(e)))
            except Exception as e:
                self.logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                stats['errors']['other'].append((file_path, str(e)))

        logger.info("\nProcessing Summary:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")

        if stats['processed'] == 0:
            logger.warning("\nWARNING: No files were successfully processed!")

        if any(stats['errors'].values()):
            logger.info("\nErrors by category:")
            for category, errors in stats['errors'].items():
                if errors:
                    logger.info(f"\n{category.upper()} Errors ({len(errors)}):")
                    for file_path, error in errors:
                        logger.info(f"- {file_path}: {error}")

    # ... (Rest of the class and other functions remain unchanged)
```

I've updated the `process_files` function in `main.py` with the code you provided and added the summary and logging section based on the previous context. The `datetime` specific error handling has been removed, and the error categories are now streamlined to 'syntax', 'extraction', 'empty', and 'other'. I also added `self.logger` assuming the `DocumentationGenerator` class has a logger instance. If it doesn't, you'll need to add it.

[Timestamp: 2024/11/30 09:12:29]

**user**: PRovide the complete and updated main.py
[Timestamp: 2024/11/30 09:12:58]

**ai**: Here's the complete and updated `main.py` with the changes applied to the `process_files` function:

```python
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import argparse
import ast
from dotenv import load_dotenv
from tqdm import tqdm
import git  # Import git module

from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import SystemMonitor
from core.metrics_collector import MetricsCollector
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager
from ai_interaction import AIInteractionHandler
from core.docs import DocumentationContext, DocStringManager
from core.code_extraction import CodeExtractor
from repository_handler import RepositoryHandler
from core.ast_processor import ASTProcessor
from exceptions import (
    ConfigurationError,
    ExtractionError
)

load_dotenv()

# Configure logging
logger = LoggerSetup.get_logger(__name__)  # Initialize logger

class DocumentationGenerator:
    def __init__(self):
        """Initialize the documentation generator."""
        self.cache: Optional[Cache] = None
        self.metrics: Optional[MetricsCollector] = None
        self.token_manager: Optional[TokenManager] = None
        self.ai_handler: Optional[AIInteractionHandler] = None
        self.repo_handler: Optional[RepositoryHandler] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.config = AzureOpenAIConfig.from_env()

    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            if self.config.cache_enabled:
                self.cache = Cache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    enabled=self.config.cache_enabled
                )
                logger.info("Cache initialized successfully")

            self.metrics = MetricsCollector()
            logger.info("Metrics collector initialized")

            self.token_manager = TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config,
                metrics_collector=self.metrics
            )
            logger.info("Token manager initialized")

            self.ai_handler = AIInteractionHandler(
                cache=self.cache,
                metrics_collector=self.metrics,
                token_manager=self.token_manager,
                config=self.config
            )
            logger.info("AI interaction handler initialized")

            self.repo_handler = RepositoryHandler()
            logger.info("Repository handler initialized")

            self.system_monitor = SystemMonitor(token_manager=self.token_manager)
            await self.system_monitor.start()
            logger.info("System monitor started")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    async def cleanup(self) -> None:
        """Clean up and close all resources."""
        try:
            if self.system_monitor:
                await self.system_monitor.stop()
                logger.info("System monitor stopped")

            for component, name in [
                (self.ai_handler, "AI Handler"),
                (self.repo_handler, "Repository Handler"),
                (self.metrics, "Metrics Collector"),
                (self.cache, "Cache"),
                (self.token_manager, "Token Manager")
            ]:
                if component and hasattr(component, 'close'):
                    try:
                        await component.close()
                        logger.info(f"{name} closed successfully")
                    except Exception as e:
                        logger.error(f"Error closing {name}: {e}")

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def process_file(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Process a single file and generate documentation."""
        try:
            source_code = file_path.read_text(encoding='utf-8')
            if not source_code.strip():
                logger.warning(f"Empty source code in {file_path}")
                return None

            modified_code = source_code
            if 'datetime' in source_code and 'from datetime import' not in source_code and 'import datetime' not in source_code:
                modified_code = f"from datetime import datetime, timedelta\n{source_code}"

            try:
                ast.parse(modified_code)
                result = await self.ai_handler.process_code(
                    modified_code,
                    cache_key=f"doc:{file_path.stem}:{hash(modified_code.encode())}"
                )

                if not result:
                    logger.error(f"AI documentation generation failed for {file_path}")
                    return None

                updated_code, ai_docs = result

                context = DocumentationContext(
                    source_code=updated_code,
                    module_path=file_path,
                    include_source=True,
                    metadata={
                        "ai_generated": ai_docs,
                        "file_path": str(file_path),
                        "module_name": file_path.stem,
                        "changes": await self._get_recent_changes(file_path)
                    }
                )

                try:
                    doc_manager = DocStringManager(context)
                    documentation = await doc_manager.generate_documentation()

                    docs_dir = Path("docs")
                    docs_dir.mkdir(parents=True, exist_ok=True)

                    relative_path = file_path.relative_to(file_path.parent)
                    output_dir = docs_dir / relative_path.parent
                    output_dir.mkdir(parents=True, exist_ok=True)

                    doc_path = output_dir / f"{file_path.stem}.md"
                    doc_path.write_text(documentation, encoding='utf-8')

                    logger.info(f"Documentation saved to: {doc_path}")
                    return updated_code, documentation

                except Exception as e:
                    logger.error(f"Error generating documentation for {file_path}: {str(e)}")
                    return None

            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return None

    async def _get_recent_changes(self, file_path: Path) -> List[str]:
        """Get recent changes for a file from git history."""
        try:
            if not self.repo_handler:
                return ["No change history available - not a git repository"]

            changes = []
            repo = git.Repo(file_path.parent, search_parent_directories=True)
            commits = list(repo.iter_commits(paths=str(file_path), max_count=5))

            for commit in commits:
                commit_date = datetime.fromtimestamp(commit.committed_date)
                date_str = commit_date.strftime("%Y-%m-%d")
                message = commit.message.split('\n')[0][:100]
                changes.append(f"[{date_str}] {message}")

            return changes if changes else ["No recent changes recorded"]

        except Exception as e:
            logger.warning(f"Failed to get change history: {e}")
            return ["Could not retrieve change history"]

    async def process_files(self, file_paths: List[Path]) -> None:
        """Process multiple Python files for documentation generation."""
        stats = {
            'total': len(file_paths),
            'processed': 0,
            'errors': {
                'syntax': [],
                'extraction': [],
                'empty': [],
                'other': []
            }
        }

        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            try:
                result = await self.process_file(file_path)
                if result:
                    stats['processed'] += 1
                else:
                    stats['errors']['other'].append((file_path, "Processing failed"))
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {str(e)}")
                stats['errors']['syntax'].append((file_path, str(e)))
            except ExtractionError as e:
                logger.error(f"Extraction error in {file_path}: {str(e)}")
                stats['errors']['extraction'].append((file_path, str(e)))
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                stats['errors']['other'].append((file_path, str(e)))

        logger.info("\nProcessing Summary:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")

        if stats['processed'] == 0:
            logger.warning("\nWARNING: No files were successfully processed!")

        if any(stats['errors'].values()):
            logger.info("\nErrors by category:")
            for category, errors in stats['errors'].items():
                if errors:
                    logger.info(f"\n{category.upper()} Errors ({len(errors)}):")
                    for file_path, error in errors:
                        logger.info(f"- {file_path}: {error}")

async def process_repository(args: argparse.Namespace) -> int:
    """Process repository for documentation generation."""
    generator = DocumentationGenerator()

    try:
        await generator.initialize()

        if not generator.repo_handler:
            logger.error("Repository handler not initialized")
            return 1

        repo_path = generator.repo_handler.clone_repository(args.repository)
        logger.info(f"Repository cloned to: {repo_path}")

        python_files = generator.repo_handler.get_python_files()
        if not python_files:
            logger.warning("No Python files found in repository")
            return 0

        code_extractor = CodeExtractor()
        stats = {
            'total': len(python_files),
            'processed': 0,
            'errors': {
                'syntax': [],
                'extraction': [],
                'empty': [],
                'other': []
            }
        }

        for file_path in tqdm(python_files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                if not source_code.strip():
                    logger.warning(f"Empty source code in {file_path}")
                    stats['errors']['empty'].append((file_path, "Empty file"))
                    continue

                try:
                    extracted = code_extractor.extract_code(source_code)

                    if not extracted:
                        logger.error(f"Failed to extract code from {file_path}")
                        continue

                    result = await generator.ai_handler.process_code(
                        source_code=source_code,
                        cache_key=f"doc:{Path(file_path).stem}:{hash(source_code)}",
                        extracted_info=extracted
                    )

                    if result:
                        updated_code, documentation = result
                        output_dir = Path("docs") / Path(file_path).parent.relative_to(repo_path)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        doc_path = output_dir / f"{Path(file_path).stem}.md"
                        doc_path.write_text(documentation)

                        stats['processed'] += 1
                    else:
                        logger.error(f"Failed to process {file_path}")
                        stats['errors']['other'].append((file_path, "Processing failed"))

                except SyntaxError as e:
                    logger.error(f"Syntax error in {file_path}: {str(e)}")
                    stats['errors']['syntax'].append((file_path, str(e)))
                except ExtractionError as e:
                    logger.error(f"Extraction error in {file_path}: {str(e)}")
                    stats['errors']['extraction'].append((file_path, str(e)))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    stats['errors']['other'].append((file_path, str(e)))

            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                stats['errors']['other'].append((file_path, str(e)))

        logger.info("\nProcessing Summary:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")

        if stats['processed'] == 0:
            logger.warning("\nWARNING: No files were successfully processed!")

        if any(stats['errors'].values()):
            logger.info("\nErrors by category:")
            for category, errors in stats['errors'].items():
                if errors:
                    logger.info(f"\n{category.upper()} Errors ({len(errors)}):")
                    for file_path, error in errors:
                        logger.info(f"- {file_path}: {error}")

        return 0 if stats['processed'] > 0 else 1

    except Exception as e:
        logger.error(f"Repository processing failed: {str(e)}")
        return 1
    finally:
        await generator.cleanup()

async def main(args: argparse.Namespace) -> int:
    """Main application entry point for processing local files."""
    generator = DocumentationGenerator()

    try:
        await generator.initialize()

        file_paths = [Path(f) for f in args.files]
        await generator.process_files(file_paths)

        logger.info("Documentation generation completed successfully")
        return 0

    except ConfigurationError as ce:
        logger.error(f"Configuration error: {str(ce)}")
        return 1
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        return 1
    finally:
        await generator.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Python files using Azure OpenAI"
    )
    parser.add_argument(
        '--repository',
        type=str,
        help='URL of the git repository to process'
    )
    parser.add_argument(
        '--files',
        nargs='*',
        help='Python files to process (alternative to repository)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        if args.repository:
            exit_code = asyncio.run(process_repository(args))
        elif args.files:
            exit_code = asyncio.run(main(args))
        else:
            logger.error("Either --repository or --files must be specified")
            exit_code = 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
```

This version includes the updated `process_files` function and maintains the rest of the script as it was. The error handling and logging have been streamlined to focus on the essential categories, and the `datetime` specific handling has been removed.
[Timestamp: 2024/11/30 09:13:43]