```python
# logger.py

import logging
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from threading import local
from functools import wraps

class LogContext:
    """Thread-local storage for log context."""
    _context = local()

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get current context dictionary."""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        return cls._context.data

    @classmethod
    def set_context(cls, key: str, value: Any) -> None:
        """Set context value."""
        context = cls.get_context()
        context[key] = value

    @classmethod
    def clear_context(cls) -> None:
        """Clear current context."""
        cls._context.data = {}

class StructuredLogRecord(logging.LogRecord):
    """Enhanced log record with structured data support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = LogContext.get_context()

class EnhancedLogger(logging.Logger):
    """Enhanced logger with context and structured logging."""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, 
                  func=None, extra=None, sinfo=None):
        """Create a structured log record."""
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, 
                                   func, sinfo)
        if extra:
            for key, value in extra.items():
                setattr(record, key, value)
        return record

class LoggerSetup:
    """Enhanced logging system setup."""
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __init__(self):
        # Register enhanced logger class
        logging.setLoggerClass(EnhancedLogger)
        
        # Ensure log directory exists
        self.log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup log rotation parameters
        self.max_bytes = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        
        # Setup formatters
        self.formatter = logging.Formatter(
            fmt=self.LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        self.json_formatter = JsonFormatter()

    @staticmethod
    def get_logger(module_name: str, enable_json: bool = False) -> logging.Logger:
        """
        Get configured logger for module.
        
        Args:
            module_name: Name of the module
            enable_json: Whether to enable JSON formatting
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(module_name)
        
        if not logger.handlers:  # Avoid adding handlers multiple times
            setup = LoggerSetup()
            setup._configure_logger(logger, module_name, enable_json)
            
        return logger

    def _configure_logger(self, 
                         logger: logging.Logger, 
                         module_name: str,
                         enable_json: bool) -> None:
        """Configure logger with handlers and formatters."""
        logger.setLevel(logging.DEBUG)
        
        # Add file handler
        log_file = os.path.join(self.log_dir, f"{module_name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            self.json_formatter if enable_json else self.formatter
        )
        logger.addHandler(file_handler)
        
        # Add error file handler
        error_file = os.path.join(self.log_dir, f"{module_name}_error.log")
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            self.json_formatter if enable_json else self.formatter
        )
        logger.addHandler(error_handler)
        
        # Add console handler in development
        if os.getenv("ENVIRONMENT") == "development":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(self.formatter)
            logger.addHandler(console_handler)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: StructuredLogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "context": getattr(record, 'context', {})
        }
        
        # Add error information if present
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stack_trace": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                          "filename", "funcName", "levelname", "levelno", "lineno",
                          "module", "msecs", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "stack_info", "thread",
                          "threadName", "context"]:
                log_data[key] = value
        
        return json.dumps(log_data)

@contextmanager
def log_context(**kwargs):
    """Context manager for adding context to logs."""
    try:
        # Save existing context
        previous_context = LogContext.get_context().copy()
        
        # Update context with new values
        LogContext.get_context().update(kwargs)
        yield
        
    finally:
        # Restore previous context
        LogContext._context.data = previous_context

def log_operation(operation_name: str):
    """Decorator for logging operation execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = LoggerSetup.get_logger(func.__module__)
            start_time = datetime.now()
            
            with log_context(operation=operation_name):
                try:
                    logger.info(f"Starting operation: {operation_name}")
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.info(
                        f"Operation completed: {operation_name}",
                        extra={
                            "duration": duration,
                            "status": "success"
                        }
                    )
                    return result
                    
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.error(
                        f"Operation failed: {operation_name}",
                        extra={
                            "duration": duration,
                            "status": "error",
                            "error": str(e),
                            "stack_trace": traceback.format_exc()
                        }
                    )
                    raise
        
        return wrapper
    return decorator

def log_async_errors(logger: logging.Logger):
    """Decorator for logging unhandled errors in async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Unhandled error in async function: {func.__name__}",
                    extra={
                        "error": str(e),
                        "stack_trace": traceback.format_exc()
                    }
                )
                raise
        return wrapper
    return decorator
```