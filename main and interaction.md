```python
"""
Main Module for Docstring Workflow System

This module serves as the entry point for the docstring generation workflow using Azure OpenAI.
It handles command-line arguments, initializes necessary components, and orchestrates the
processing of Python source files to generate and update docstrings.

Version: 1.1.0
Author: Development Team
"""

import argparse
import asyncio
import os
import shutil
import tempfile
import subprocess
import signal
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv
from interaction import InteractionHandler
from logger import log_info, log_error, log_debug, log_warning
from monitoring import SystemMonitor
from utils import ensure_directory
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from docs import MarkdownGenerator
from api_client import AzureOpenAIClient
from config import AzureOpenAIConfig
from cache import Cache

# Load environment variables from .env file
load_dotenv()

monitor = SystemMonitor()

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    log_info("Received shutdown signal, cleaning up...")
    raise KeyboardInterrupt

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

@contextmanager
def cleanup_context(temp_dir: Optional[str] = None):
    """Context manager for cleanup operations."""
    try:
        yield
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                log_debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                log_info("Cleanup completed successfully")
            except Exception as e:
                log_error(f"Error during cleanup: {e}")

async def initialize_client() -> AzureOpenAIClient:
    """
    Initialize the Azure OpenAI client with proper retry logic.
    
    Returns:
        AzureOpenAIClient: Configured client instance
    """
    config = AzureOpenAIConfig.from_env()
    return AzureOpenAIClient(config)

def load_source_file(file_path: str) -> str:
    """
    Load the source code from a specified file path.

    Args:
        file_path (str): The path to the source file.

    Returns:
        str: The content of the source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    try:
        log_debug(f"Attempting to load source file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
            log_info(f"Successfully loaded source code from '{file_path}'")
            return source_code
    except FileNotFoundError:
        log_error(f"File '{file_path}' not found.")
        raise
    except IOError as e:
        log_error(f"Failed to read file '{file_path}': {e}")
        raise

def save_updated_source(file_path: str, updated_code: str) -> None:
    """
    Save the updated source code to a specified file path.

    Args:
        file_path (str): The path to save the updated source code.
        updated_code (str): The updated source code to save.

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        log_debug(f"Attempting to save updated source code to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_code)
            log_info(f"Successfully saved updated source code to '{file_path}'")
    except IOError as e:
        log_error(f"Failed to save updated source code to '{file_path}': {e}")
        raise

async def process_file(file_path: str, args: argparse.Namespace, client: AzureOpenAIClient, cache: Cache) -> None:
    """
    Process a single Python file to extract and update docstrings.

    Args:
        file_path (str): The path to the Python file to process
        args (argparse.Namespace): The command-line arguments
        client (AzureOpenAIClient): The Azure OpenAI client for generating docstrings
        cache (Cache): The cache instance for caching responses

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If there is an error reading or writing the file
    """
    log_debug(f"Processing file: {file_path}")
    start_time = time.time()
    interaction_handler = InteractionHandler(client=client, cache=cache)
    failed_items = []

    try:
        source_code = load_source_file(file_path)
        
        # Process all functions and classes
        updated_code, documentation_entries = await interaction_handler.process_all_functions(source_code)
        
        # Generate documentation
        markdown_generator = MarkdownGenerator()
        for entry in documentation_entries:
            if 'function_name' in entry:
                markdown_generator.add_header(entry['function_name'])
                markdown_generator.add_code_block(entry['docstring'], language="python")
            elif 'class_name' in entry:
                markdown_generator.add_header(entry['class_name'])
                markdown_generator.add_code_block(entry['docstring'], language="python")
        
        markdown_content = markdown_generator.generate_markdown()
        
        # Save documentation and updated source
        ensure_directory(args.output_dir)
        markdown_file_path = os.path.join(args.output_dir, f"{os.path.basename(file_path)}.md")
        with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
            log_info(f"Markdown documentation saved to '{markdown_file_path}'")
        
        # Update source code with new docstrings
        save_updated_source(file_path, updated_code)
        
        # Report results
        monitor.log_operation_complete(file_path, time.time() - start_time, len(failed_items))
        
        if failed_items:
            log_warning(f"Failed to process {len(failed_items)} items in {file_path}")
            for item_type, item_name in failed_items:
                log_warning(f"Failed {item_type}: {item_name}")

    except Exception as e:
        log_error(f"Error processing file {file_path}: {e}")
        monitor.log_request(file_path, "error", time.time() - start_time, error=str(e))
        raise

async def run_workflow(args: argparse.Namespace) -> None:
    """Run the docstring generation workflow for the specified source path."""
    source_path = args.source_path
    temp_dir = None

    try:
        # Initialize client with timeout
        client = await asyncio.wait_for(initialize_client(), timeout=30)
        cache = Cache(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            password=args.redis_password,
            default_ttl=args.cache_ttl
        )
        
        if source_path.startswith(('http://', 'https://')):
            temp_dir = tempfile.mkdtemp()
            try:
                log_debug(f"Cloning repository from URL: {source_path} to temp directory: {temp_dir}")
                # Use async subprocess
                process = await asyncio.create_subprocess_exec(
                    'git', 'clone', source_path, temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minute timeout
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, ['git', 'clone'])
                    source_path = temp_dir
                except asyncio.TimeoutError:
                    log_error("Repository cloning timed out")
                    if process:
                        process.terminate()
                    return
            except Exception as e:
                log_error(f"Failed to clone repository: {e}")
                return

        with cleanup_context(temp_dir):
            if os.path.isdir(source_path):
                log_debug(f"Processing directory: {source_path}")
                tasks = []
                for root, _, files in os.walk(source_path):
                    for file in files:
                        if file.endswith('.py'):
                            task = asyncio.create_task(
                                process_file(os.path.join(root, file), args, client, cache)
                            )
                            tasks.append(task)
                
                # Process tasks with timeout and proper cancellation handling
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    log_error("Tasks cancelled - attempting graceful shutdown")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for tasks to complete cancellation
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
            else:
                log_debug(f"Processing single file: {source_path}")
                await process_file(source_path, args, client, cache)

    except asyncio.CancelledError:
        log_error("Workflow cancelled - performing cleanup")
        raise
    except Exception as e:
        log_error(f"Workflow error: {str(e)}")
        raise
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DocStrings Workflow System with Azure OpenAI and Redis Caching')
    parser.add_argument('source_path', help='Path to the Python source file, directory, or Git repository to be processed.')
    parser.add_argument('api_key', help='Your Azure OpenAI API key.')
    parser.add_argument('--endpoint', default='https://your-azure-openai-endpoint.openai.azure.com/', help='Your Azure OpenAI endpoint.')
    parser.add_argument('--output-dir', default='output', help='Directory to save the updated source code and markdown documentation.')
    parser.add_argument('--documentation-file', help='Path to save the generated documentation.')
    parser.add_argument('--redis-host', default='localhost', help='Redis server host.')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port.')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database number.')
    parser.add_argument('--redis-password', help='Redis server password if required.')
    parser.add_argument('--cache-ttl', type=int, default=86400, help='Default TTL for cache entries in seconds.')
    args = parser.parse_args()

    try:
        # Run with proper task and event loop management
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_workflow(args))
        except (asyncio.CancelledError, KeyboardInterrupt):
            log_error("Program interrupted")
        except Exception as e:
            log_error(f"Program error: {e}")
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for task cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    finally:
        log_info("Program shutdown complete")
```

```python
"""
Interaction Handler Module

This module manages the orchestration of docstring generation with integrated caching,
monitoring, and error handling. It processes functions and classes in batches and interacts with
the Azure OpenAI API to generate documentation.

Version: 1.3.0
Author: Development Team
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Tuple, Optional, List
import ast
from api_client import AzureOpenAIClient
from cache import Cache
from logger import log_info, log_error, log_debug, log_warning
from monitoring import SystemMonitor
from extract.extraction_manager import ExtractionManager
from documentation_analyzer import DocumentationAnalyzer
from response_parser import ResponseParser

class InteractionHandler:
    """
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling.
    """

    def __init__(
        self,
        client: AzureOpenAIClient,
        cache: Cache,
        batch_size: int = 5
    ):
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            client (AzureOpenAIClient): The Azure OpenAI client instance.
            cache (Cache): The cache instance for caching responses.
            batch_size (int): Number of functions to process concurrently.
        """
        self.client = client
        self.cache = cache
        self.monitor = SystemMonitor()
        self.doc_analyzer = DocumentationAnalyzer()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        self.response_parser = ResponseParser()
        log_info("Interaction Handler initialized with batch processing capability")

    async def process_all_functions(self, source_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions and classes in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions and classes.")
        try:
            # Extract metadata using the centralized manager
            metadata = self.extraction_manager.extract_metadata(source_code)
            functions = metadata['functions']
            classes = metadata.get('classes', [])
            
            log_info(f"Extracted {len(functions)} functions and {len(classes)} classes from source code.")

            # Process functions in batches
            function_results = []
            for i in range(0, len(functions), self.batch_size):
                batch = functions[i: i + self.batch_size]
                batch_tasks = [self.process_function(source_code, func_info) for func_info in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                function_results.extend(batch_results)

            # Process classes
            class_results = []
            for class_info in classes:
                class_result = await self.process_class(source_code, class_info)
                if class_result:
                    class_results.append(class_result)

            # Update source code and generate documentation
            documentation_entries = []

            # Add function documentation
            for function_info, (docstring, metadata) in zip(functions, function_results):
                if docstring:
                    if metadata:
                        documentation_entries.append({
                            "function_name": function_info["name"],
                            "complexity_score": metadata.get("complexity_score", 0),
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "changelog": metadata.get("changelog", ""),
                        })

            # Add class documentation
            for class_info, (docstring, metadata) in zip(classes, class_results):
                if docstring:
                    if metadata:
                        documentation_entries.append({
                            "class_name": class_info["name"],
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "methods": class_info.get("methods", [])
                        })

            # Log final metrics
            total_items = len(functions) + len(classes)
            self.monitor.log_batch_completion(total_items)
            log_info("Batch processing completed successfully.")

            return source_code, documentation_entries

        except Exception as e:
            log_error(f"Error in batch processing: {str(e)}")
            return None, None

    async def process_function(self, source_code: str, function_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function
            function_info (Dict): Metadata about the function to process

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed
        """
        async with self.semaphore:
            func_name = function_info.get('name', 'unknown')
            start_time = time.time()

            try:
                # Check cache first
                cache_key = self._generate_cache_key(function_info['node'])
                cached_response = await self.cache.get_cached_docstring(cache_key)
                
                if cached_response:
                    # Validate cached response
                    parsed_cached = self.response_parser.parse_docstring_response(
                        json.dumps(cached_response)
                    )
                    if parsed_cached and self.doc_analyzer.is_docstring_complete(parsed_cached['docstring']):
                        log_info(f"Using valid cached docstring for {func_name}")
                        self.monitor.log_cache_hit(func_name)
                        return parsed_cached['docstring'], cached_response
                    else:
                        log_warning(f"Invalid cached docstring found for {func_name}, will regenerate")
                        await self.cache.invalidate_by_tags([cache_key])

                # Check existing docstring
                existing_docstring = function_info.get('docstring')
                if existing_docstring and self.doc_analyzer.is_docstring_complete(existing_docstring):
                    log_info(f"Existing complete docstring found for {func_name}")
                    return existing_docstring, None

                # Attempt to generate new docstring
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        response = await self.client.generate_docstring(
                            func_name=function_info['name'],
                            params=function_info['args'],
                            return_type=function_info['returns'],
                            complexity_score=function_info.get('complexity_score', 0),
                            existing_docstring=function_info['docstring'],
                            decorators=function_info['decorators'],
                            exceptions=function_info.get('exceptions', [])
                        )

                        if not response:
                            log_error(f"Failed to generate docstring for {func_name} (attempt {attempt + 1}/{max_attempts})")
                            continue

                        # Parse and validate the response
                        parsed_response = self.response_parser.parse_json_response(
                            json.dumps(response['content'])
                        )

                        if not parsed_response:
                            log_error(f"Failed to parse response for {func_name} (attempt {attempt + 1}/{max_attempts})")
                            continue

                        # Validate the generated docstring
                        if self.doc_analyzer.is_docstring_complete(parsed_response['docstring']):
                            # Cache successful generation
                            await self.cache.save_docstring(
                                cache_key,
                                {
                                    'docstring': parsed_response['docstring'],
                                    'metadata': {
                                        'timestamp': time.time(),
                                        'function_name': func_name,
                                        'summary': parsed_response.get('summary', ''),
                                        'complexity_score': parsed_response.get('complexity_score', 0)
                                    }
                                },
                                tags=[f"func:{func_name}"]
                            )

                            # Log success metrics
                            self.monitor.log_operation_complete(
                                func_name=func_name,
                                duration=time.time() - start_time,
                                tokens=response['usage']['total_tokens']
                            )

                            log_info(f"Successfully generated and cached docstring for {func_name}")
                            return parsed_response['docstring'], parsed_response

                        else:
                            log_warning(f"Generated docstring incomplete for {func_name} (attempt {attempt + 1}/{max_attempts})")
                            self.monitor.log_docstring_issue(func_name, "incomplete_generated")

                    except asyncio.TimeoutError:
                        log_error(f"Timeout generating docstring for {func_name} (attempt {attempt + 1}/{max_attempts})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    except Exception as e:
                        log_error(f"Error generating docstring for {func_name} (attempt {attempt + 1}/{max_attempts}): {e}")
                        await asyncio.sleep(2 ** attempt)

                # If all attempts fail
                log_error(f"Failed to generate valid docstring for {func_name} after {max_attempts} attempts")
                self.monitor.log_docstring_failure(func_name, "max_attempts_exceeded")
                return None, None

            except Exception as e:
                log_error(f"Unexpected error processing function {func_name}: {e}")
                self.monitor.log_docstring_failure(func_name, str(e))
                return None, None
            finally:
                # Log timing metrics regardless of outcome
                self.monitor.log_request(
                    func_name,
                    status="complete",
                    response_time=time.time() - start_time
                )

    async def process_class(self, source_code: str, class_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Process a single class with enhanced error handling and monitoring."""
        class_name = class_info.get('name', 'unknown')
        start_time = time.time()

        try:
            # Generate docstring for class
            response = await self.client.generate_docstring(
                func_name=class_name,
                params=[],
                return_type="None",
                complexity_score=0,
                existing_docstring=class_info.get('docstring', ''),
                decorators=class_info.get('decorators', []),
                exceptions=[]
            )

            if response and response.get('content'):
                return response['content'].get('docstring'), response['content']
            return None, None

        except Exception as e:
            log_error(f"Error processing class {class_name}: {e}")
            return None, None
        finally:
            self.monitor.log_request(
                class_name,
                status="complete",
                response_time=time.time() - start_time
            )

    def _generate_cache_key(self, function_node: ast.FunctionDef) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_node (ast.FunctionDef): The function node to generate a key for.

        Returns:
            str: The generated cache key.
        """
        func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
        return hashlib.md5(func_signature.encode()).hexdigest()
```