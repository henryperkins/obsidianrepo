Fix code duplication and redundant processes

### Current behavior

- 🛠️ The `optimize_prompt` function in api/token_management.py contains redundant processes for trimming and truncating text.
    
- 🔄 The `estimate_tokens` function in api/token_management.py is duplicated as a module-level function and a method within the `TokenManager` class.
    
- 📊 The `collect_metrics` method in core/metrics_collector.py has redundant checks and processes for updating metrics.
    
- 🗂️ The `extract_classes` method in core/extraction/class_extractor.py and `extract_functions` method in core/extraction/function_extractor.py have similar logic for processing AST nodes.
    
- 🧹 The `read_file_safe` and `read_file_safe_async` functions in utils.py have redundant error handling logic.
    

Add item

### Proposed solution

- 🛠️ The `optimize_prompt` function in api/token_management.py now uses a single method for trimming and truncating text, reducing redundancy.
    
- 🔄 The `estimate_tokens` function in api/token_management.py is now only defined as a method within the `TokenManager` class, removing the module-level duplication.
    
- 📊 The `collect_metrics` method in core/metrics_collector.py now has streamlined checks and processes for updating metrics, eliminating redundancy.
    
- 🗂️ The `extract_classes` method in core/extraction/class_extractor.py and `extract_functions` method in core/extraction/function_extractor.py now share common logic for processing AST nodes, reducing code duplication.
    
- 🧹 The `read_file_safe` and `read_file_safe_async` functions in utils.py now have consolidated error handling logic, improving code maintainability.