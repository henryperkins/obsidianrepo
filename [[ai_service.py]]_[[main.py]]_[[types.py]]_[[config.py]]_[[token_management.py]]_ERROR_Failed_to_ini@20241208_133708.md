---
epoch: 1733686628884
modelKey: gpt-4o-2024-11-20|azure openai
tags:
  - copilot-conversation
---

**user**: [ai_service.py](ai_service.py.md) [main.py](main.py.md) [types.py](types.py.md) [config.py](config.py.md) [token_management.py](token_management.py.md) ERROR: Failed to initialize AI service
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 46, in __init__
    self._client = AsyncAzureOpenAI(
                   ^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\venv\Lib\site-packages\openai\lib\azure.py", line 417, in __init__
    raise ValueError(
ValueError: Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable
ERROR: Failed to run documentation generator: AI service initialization failed: Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 46, in __init__
    self._client = AsyncAzureOpenAI(
                   ^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\venv\Lib\site-packages\openai\lib\azure.py", line 417, in __init__
    raise ValueError(
ValueError: Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\main.py", line 286, in <module>
    exit_code = asyncio.run(main(args))
                ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0\Lib\asyncio\runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2288.0_x64__qbz5n2kfra8p0\Lib\asyncio\base_events.py", line 686, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\main.py", line 222, in main
    doc_generator = DocumentationGenerator()
                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\main.py", line 40, in __init__
    self.ai_service = AIService(config=self.config.ai)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 58, in __init__
    raise ConfigurationError(f"AI service initialization failed: {e}") from e
exceptions.ConfigurationError: AI service initialization failed: Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable
[Timestamp: 2024/12/08 13:37:08]

**ai**: Error: Model request failed: Chat model is not initialized properly, check your API key in Copilot setting and make sure you have API access.
[Timestamp: 2024/12/08 13:37:08]