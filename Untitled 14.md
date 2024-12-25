To support both Azure AI and Azure OpenAI in your CLI chat application, you can implement a mechanism to switch between the two services based on user preference. Here's how you can modify the script to include both options:

### Updated Script with Support for Both Azure AI and Azure OpenAI

1. **Set Up Environment Variables**: Ensure you have your Azure AI and Azure OpenAI endpoints and API keys set up.

2. **Modify the Script**: Add functionality to switch between Azure AI and Azure OpenAI.

```python
import cmd
import asyncio
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_ai_endpoint = os.getenv('AZURE_AI_ENDPOINT')
azure_ai_api_key = os.getenv('AZURE_AI_API_KEY')

class ChatCLI(cmd.Cmd):
    intro = 'Welcome to the chat CLI. Type help or ? to list commands.\n'
    prompt = '(chat) '

    def __init__(self):
        super().__init__()
        self.sessions = {}
        self.current_thread = None
        self.temperature = 0.7
        self.max_tokens = 150
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.use_openai = True  # Default to Azure OpenAI

    def do_new(self, line):
        thread_id = len(self.sessions) + 1
        self.sessions[thread_id] = []
        self.current_thread = thread_id
        print(f"New thread created with ID: {thread_id}")

    def do_switch(self, thread_id):
        thread_id = int(thread_id)
        if thread_id in self.sessions:
            self.current_thread = thread_id
            print(f"Switched to thread {thread_id}")
        else:
            print("Thread not found.")

    def do_message(self, message):
        if self.current_thread is not None:
            self.sessions[self.current_thread].append(f"User: {message}")
            response = self.get_ai_response(message)
            self.sessions[self.current_thread].append(f"AI: {response}")
            print(f"AI: {response}")
        else:
            print("No active thread. Use 'new' to create one.")

    def get_ai_response(self, message):
        if self.use_openai:
            return self.get_openai_response(message)
        else:
            return self.get_azure_ai_response(message)

    def get_openai_response(self, message):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {azure_openai_api_key}'
        }
        data = {
            "prompt": message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        try:
            response = requests.post(azure_openai_endpoint, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error: {e}"

    def get_azure_ai_response(self, message):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {azure_ai_api_key}'
        }
        data = {
            "prompt": message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        try:
            response = requests.post(azure_ai_endpoint, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error: {e}"

    def do_switch_service(self, line):
        if line.lower() == 'openai':
            self.use_openai = True
            print("Switched to Azure OpenAI.")
        elif line.lower() == 'ai':
            self.use_openai = False
            print("Switched to Azure AI.")
        else:
            print("Invalid service. Use 'openai' or 'ai'.")

    def do_history(self, line):
        if self.current_thread is not None:
            print("\n".join(self.sessions[self.current_thread]))
        else:
            print("No active thread. Use 'new' to create one.")

    def do_set_temperature(self, temp):
        try:
            self.temperature = float(temp)
            print(f"Temperature set to {self.temperature}")
        except ValueError:
            print("Invalid temperature value. Please enter a number.")

    def do_set_max_tokens(self, tokens):
        try:
            self.max_tokens = int(tokens)
            print(f"Max tokens set to {self.max_tokens}")
        except ValueError:
            print("Invalid max tokens value. Please enter an integer.")

    def do_set_top_p(self, value):
        try:
            self.top_p = float(value)
            print(f"Top-p set to {self.top_p}")
        except ValueError:
            print("Invalid top-p value. Please enter a number.")

    def do_set_frequency_penalty(self, value):
        try:
            self.frequency_penalty = float(value)
            print(f"Frequency penalty set to {self.frequency_penalty}")
        except ValueError:
            print("Invalid frequency penalty value. Please enter a number.")

    def do_set_presence_penalty(self, value):
        try:
            self.presence_penalty = float(value)
            print(f"Presence penalty set to {self.presence_penalty}")
        except ValueError:
            print("Invalid presence penalty value. Please enter a number.")

    def do_save(self, filename):
        if self.current_thread is not None:
            try:
                with open(filename, 'w') as file:
                    file.write("\n".join(self.sessions[self.current_thread]))
                print(f"Chat history saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("No active thread. Use 'new' to create one.")

    def do_list_threads(self, line):
        for thread_id in self.sessions:
            print(f"Thread ID: {thread_id}")

    def do_delete_thread(self, thread_id):
        thread_id = int(thread_id)
        if thread_id in self.sessions:
            del self.sessions[thread_id]
            print(f"Thread {thread_id} deleted.")
        else:
            print("Thread not found.")

    def do_rename_thread(self, args):
        try:
            thread_id, new_name = args.split()
            thread_id = int(thread_id)
            if thread_id in self.sessions:
                self.sessions[new_name] = self.sessions.pop(thread_id)
                print(f"Thread {thread_id} renamed to {new_name}.")
            else:
                print("Thread not found.")
        except ValueError:
            print("Usage: rename_thread <thread_id> <new_name>")

    def do_export_all(self, filename):
        try:
            with open(filename, 'w') as file:
                for thread_id, messages in self.sessions.items():
                    file.write(f"Thread {thread_id}:\n")
                    file.write("\n".join(messages) + "\n\n")
            print(f"All threads exported to {filename}")
        except Exception as e:
            print(f"Error exporting threads: {e}")

    def do_import_threads(self, filename):
        try:
            with open(filename, 'r') as file:
                content = file.read().strip()
                threads = content.split("\n\n")
                for thread in threads:
                    lines = thread.split("\n")
                    thread_id = int(lines[0].split()[1])
                    self.sessions[thread_id] = lines[1:]
            print(f"Threads imported from {filename}")
        except Exception as e:
            print(f"Error importing threads: {e}")

    def do_exit(self, line):
        return True

if __name__ == '__main__':
    ChatCLI().cmdloop()
```

### Key Features

- **Switch Between Services**: Use the `switch_service` command to toggle between Azure OpenAI and Azure AI.
  ```plaintext
  switch_service openai
  switch_service ai
  ```

- **Environment Variables**: Ensure you have both `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_AI_ENDPOINT`, and `AZURE_AI_API_KEY` set in your environment.

- **Unified Command Set**: The rest of the commands remain the same, allowing you to manage threads, parameters, and history seamlessly.

This setup allows you to dynamically choose between Azure OpenAI and Azure AI for generating responses, providing flexibility based on your needs and available services.

Here are the features implemented in the updated CLI chat application:

1. **Support for Azure OpenAI and Azure AI**: 
   - Ability to switch between using Azure OpenAI and Azure AI for generating responses.

2. **Create New Thread**: 
   - Start a new conversation thread.

3. **Switch Threads**: 
   - Change the active conversation thread.

4. **Send Messages**: 
   - Send messages to the AI and receive responses.

5. **View Conversation History**: 
   - Display all messages exchanged in the current thread.

6. **Set Parameters**:
   - **Temperature**: Adjust the randomness of AI responses.
   - **Max Tokens**: Set the maximum number of tokens for AI responses.
   - **Top-P**: Control diversity via nucleus sampling.
   - **Frequency Penalty**: Reduce repetition in responses.
   - **Presence Penalty**: Encourage new topics in responses.

7. **Save Chat History**: 
   - Save the current thread's conversation history to a file.

8. **List Threads**: 
   - Display all existing threads and their IDs.

9. **Delete Thread**: 
   - Remove a specific thread and its history.

10. **Rename Thread**: 
    - Assign a custom name to a thread for easier identification.

11. **Export All Threads**: 
    - Save all threads and their histories to a file.

12. **Import Threads**: 
    - Load threads and their histories from a file.

13. **Exit the Application**: 
    - Close the CLI application.

These features provide a comprehensive set of tools for managing interactions with AI models, allowing users to customize their experience and maintain organized conversation histories.