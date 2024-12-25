Migrating existing modules to use the new logging setup involves a few systematic steps to ensure consistency and take advantage of the features provided by the `logger.py`. Here's a guide on how to migrate your existing modules:

### Migration Guide for Existing Modules

#### Step 1: Remove Old Logging Code

- Identify and remove any old logging configuration code or custom loggers tied to individual modules. Focus on finding outdated log setup within existing modules.

#### Step 2: Import the New Logger

Replace existing logging imports or setups with the new utility functions and classes from `logger.py`:

```python
from logger import log_error, log_debug, log_info, log_warning, LoggerSetup, CorrelationLoggerAdapter
```

#### Step 3: Initialize the Logger

For each module, decide if you want basic logging or if correlation IDs will benefit your log tracking (typically useful in complex systems or distributed applications):

```python
# Basic setup without correlation ID
logger = LoggerSetup.get_logger(__name__)

# Optional: Advanced setup with correlation ID
correlation_adapter = CorrelationLoggerAdapter(LoggerSetup.get_logger(__name__))
```

#### Step 4: Replace Log Statements

Identify existing log statements and replace them with the new logging utility functions. Ensure that you update the log level to match your current structure:

```python
# Replace old logging calls with new ones
# From something like this:
# logger.debug("Debugging details here")
# logger.error("Error occurred", exc_info=True)

# To this:
log_debug("Debugging details here")
log_error("Error occurred", exc_info=True)
```

#### Step 5: Ensure Consistent Log Structure

Ensure that each log statement, especially for WARNING and ERROR levels, includes any required `extra` parameters:

```python
# With a correlation adapter for consistent IDs
correlation_adapter.info("Processed the dataset with ID: 1234", extra={"context": "Processing"})
```

#### Step 6: Global Exception Handling

Ensure your main application file or entry point includes global exception handling setup:

```python
import sys
from logger import LoggerSetup

sys.excepthook = LoggerSetup.handle_exception
```

#### Step 7: Review and Test

- **Review**: Go through each changed module to verify that logging statements communicate the intended process flows and errors.
- **Test**: Run unit tests or integration tests to confirm that logs are being captured correctly and no expected log information is lost.

#### Additional Tips

- **Maintain Readability**: Ensure log messages remain clear and informative.
- **Agreement on Levels**: Work with your team to establish when to use each log level, aiding both developers and operations teams in monitoring applications.
- **Iterate**: As you become more accustomed to the new logging setup, identify common log message patterns that can be abstracted or improved across modules.

By following this migration guide, you will ensure your existing modules are brought into alignment with your new logging infrastructure, allowing for a more consistent and maintainable system.