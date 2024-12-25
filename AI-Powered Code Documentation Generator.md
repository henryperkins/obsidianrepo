# AI-Powered Code Documentation Generator

**Overview:**

DocScribe automates Google-style documentation for multi-language codebases using OpenAI's gpt-4o API.

**Components:**

1. **Repository Traversal:**
   - Identifies files via extensions and exclusion patterns.

2. **Code Analysis:**
   - Extracts functions, classes, HTML tags, etc.
   - Uses parsers like `ast` for Python.

3. **AI Documentation:**
   - Generates documentation with GPT-4.
   - Constructs prompts with code details and project info.

4. **Documentation Insertion:**
   - Inserts docs into code with correct formatting.

5. **Documentation Report:**
   - Creates a Markdown summary for each file.

6. **Configuration:**
   - Customizable via `config.json` and CLI.

7. **Error Handling:**
   - Robust error handling and logging.

**LLM Assistance Tasks:**

1. **Enhance Error Handling:**
   - **Instruction:** Identify common exceptions in API calls and file operations. Provide examples of try-except blocks with logging for each scenario.

2. **Improve Concurrency:**
   - **Instruction:** Explore alternatives to `asyncio.Semaphore`. Explain the use of asynchronous queues with examples, highlighting advantages and potential pitfalls.

3. **Refactor for Modularity:**
   - **Instruction:** Break down large functions into smaller, reusable components. Provide a clear outline of proposed modules and their responsibilities.

4. **Add Language Support:**
   - **Instruction:** Research libraries for parsing Java, C++, and Go. Provide a step-by-step integration plan, including potential challenges and solutions.

5. **Improve Report:**
   - **Instruction:** Suggest Markdown enhancements like tables of contents and collapsible sections. Provide library recommendations and examples for implementation.
