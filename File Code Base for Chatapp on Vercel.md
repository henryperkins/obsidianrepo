1. **Chat Component (`Chat.tsx`)**
2. **Chat Styles (`Chat.css`)**
3. **FewShotForm Component (`FewShotForm.tsx`)**
4. **FileUploadForm Component (`FileUploadForm.tsx`)**
5. **ConversationList Component (`ConversationList.tsx`)**
6. **SearchForm Component (`SearchForm.tsx`)**

---

## **1. Chat Component (`Chat.tsx`)**

**File:** `frontend/src/components/Chat.tsx`

```tsx
import React, { useState, useEffect, useRef, ChangeEvent } from 'react';
import Pusher from 'pusher-js';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import './Chat.css';
import fetchWithAuth from '../utils/fetchWithAuth';
import { API_BASE_URL } from '../utils/config';

const notyf = new Notyf();

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [userMessage, setUserMessage] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    startNewConversation();
    setupPusher();

    // Auto-focus input field on component mount
    inputRef.current?.focus();

    // Clean up Pusher subscriptions on unmount
    return () => {
      Pusher.instances.forEach((instance) => instance.disconnect());
    };
  }, []);

  useEffect(() => {
    // Scroll to the bottom when messages update
    chatHistoryRef.current?.scrollTo({
      top: chatHistoryRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

  const setupPusher = () => {
    const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY || '', {
      cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER || '',
    });

    const channel = pusher.subscribe('chat-channel');

    channel.bind('new-message', (data: Message & { conversation_id: string }) => {
      if (data.conversation_id !== conversationId) return;

      setMessages((prevMessages) => [...prevMessages, { role: data.role, content: data.content }]);

      if (data.role === 'assistant') {
        setIsTyping(false);
      }
    });
  };

  const startNewConversation = async () => {
    try {
      const response = await fetchWithAuth('/api/start_conversation', { method: 'POST' });
      const data = await response.json();
      setConversationId(data.conversation_id);
      setMessages([]);
      notyf.success('Started a new conversation.');
    } catch (error: any) {
      notyf.error(error.message || 'Failed to start a new conversation.');
    }
  };

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const message = userMessage.trim();
    setMessages((prevMessages) => [...prevMessages, { role: 'user', content: message }]);
    setUserMessage('');
    setIsTyping(true);

    try {
      await fetchWithAuth('/api/send_message', {
        method: 'POST',
        body: JSON.stringify({ conversation_id: conversationId, message }),
      });
    } catch (error: any) {
      notyf.error(error.message || 'Failed to send message.');
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setUserMessage(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  // Placeholder functions for navigation buttons
  const resetConversation = async () => {
    try {
      await fetchWithAuth('/api/reset_conversation', {
        method: 'POST',
        body: JSON.stringify({ conversation_id: conversationId }),
      });
      setMessages([]);
      notyf.success('Conversation reset.');
    } catch (error: any) {
      notyf.error(error.message || 'Failed to reset conversation.');
    }
  };

  const listConversations = () => {
    // Implement conversation listing logic
    toggleSidebar();
  };

  const loadConversation = async (conversation_id: string) => {
    try {
      const response = await fetchWithAuth(`/api/load_conversation/${conversation_id}`, {
        method: 'GET',
      });
      const data = await response.json();
      setConversationId(conversation_id);
      setMessages(data.conversation);
      notyf.success('Conversation loaded.');
    } catch (error: any) {
      notyf.error(error.message || 'Failed to load conversation.');
    }
  };

  return (
    <div className={`chat-page ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      {/* Sidebar for Conversation List */}
      <aside className={`conversation-sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <ConversationList loadConversation={loadConversation} />
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Header */}
        <header className="chat-header">
          <h1>Llama Token Chatbot</h1>
          <nav className="chat-nav">
            <button onClick={toggleSidebar} title="Toggle Conversation History" aria-label="Toggle Conversation History">
              <i className="fas fa-bars"></i>
            </button>
            <button onClick={startNewConversation} title="New Conversation" aria-label="Start New Conversation">
              <i className="fas fa-plus"></i>
            </button>
            <button onClick={resetConversation} title="Reset Conversation" aria-label="Reset Conversation">
              <i className="fas fa-redo"></i>
            </button>
            {/* Add more buttons as needed */}
          </nav>
        </header>

        {/* Conversation Area */}
        <div className="chat-container">
          <div className="chat-history" ref={chatHistoryRef}>
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
            {isTyping && (
              <div className="message assistant">
                <div className="message-content typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>

          {/* Message Input Area */}
          <div className="message-input">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
            >
              <textarea
                ref={inputRef}
                value={userMessage}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder="Type your message and press Enter..."
                className="message-input-field"
                rows={1}
              />
              <button type="submit" className="send-button" aria-label="Send Message">
                <i className="fas fa-paper-plane"></i>
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Chat;
```

---

## **2. Chat Styles (`Chat.css`)**

**File:** `frontend/src/components/Chat.css`

```css
/* Base styles */
.chat-page {
  display: flex;
  height: 100vh;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  background-color: [[f5f5f5]];
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  transition: margin-left 0.3s ease;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 15px 20px;
  background-color: [[ffffff]];
  border-bottom: 1px solid [[e0e0e0]];
}

.chat-header h1 {
  margin: 0;
  font-size: 24px;
  color: #333333;
}

.chat-nav button {
  background: none;
  border: none;
  margin-left: 15px;
  cursor: pointer;
}

.chat-nav i {
  font-size: 20px;
  color: #333333;
}

.chat-nav i:hover {
  color: [[007bff]];
}

/* Sidebar styles */
.conversation-sidebar {
  position: fixed;
  left: -250px;
  top: 0;
  width: 250px;
  height: 100%;
  background-color: [[ffffff]];
  border-right: 1px solid [[e0e0e0]];
  overflow-y: auto;
  transition: left 0.3s ease;
  z-index: 1000;
}

.conversation-sidebar.open {
  left: 0;
}

.chat-main.sidebar-open {
  margin-left: 250px;
}

/* Chat container */
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-history {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.message {
  display: flex;
  margin-bottom: 15px;
}

.message-content {
  max-width: 60%;
  padding: 15px;
  border-radius: 10px;
  font-size: 16px;
  line-height: 1.5;
}

.message.user {
  justify-content: flex-end;
}

.message.user .message-content {
  background-color: [[dcf8c6]]; /* Light green background for user messages */
  color: #000000;
  border-top-right-radius: 0;
}

.message.assistant {
  justify-content: flex-start;
}

.message.assistant .message-content {
  background-color: [[ffffff]];
  color: #000000;
  border-top-left-radius: 0;
  border: 1px solid [[e0e0e0]];
}

/* Typing indicator */
.typing-indicator {
  display: flex;
  align-items: center;
  justify-content: flex-start;
}

.typing-indicator span {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin: 0 2px;
  background-color: [[cccccc]];
  border-radius: 50%;
  animation: typing 1s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0% {
    opacity: 0.2;
    transform: scale(1);
  }
  20% {
    opacity: 1;
    transform: scale(1.3);
  }
  100% {
    opacity: 0.2;
    transform: scale(1);
  }
}

/* Message input area */
.message-input {
  padding: 15px 20px;
  background-color: [[ffffff]];
  border-top: 1px solid [[e0e0e0]];
}

.message-input form {
  display: flex;
  align-items: center;
}

.message-input-field {
  flex: 1;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid [[e0e0e0]];
  border-radius: 20px;
  outline: none;
  resize: none;
  max-height: 150px;
  overflow-y: auto;
}

.message-input-field:focus {
  border-color: [[007bff]];
}

.send-button {
  margin-left: 10px;
  padding: 10px 15px;
  font-size: 16px;
  background-color: [[007bff]]; /* Blue button color */
  color: [[ffffff]];
  border: none;
  border-radius: 50%;
  cursor: pointer;
}

.send-button i {
  font-size: 18px;
}

.send-button:hover {
  background-color: [[0056b3]];
}

/* Scrollbar styling */
.chat-history::-webkit-scrollbar {
  width: 8px;
}

.chat-history::-webkit-scrollbar-track {
  background: [[f1f1f1]];
}

.chat-history::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

.chat-history::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .message-content {
    max-width: 80%;
    font-size: 14px;
  }

  .message-input-field {
    font-size: 14px;
  }

  .send-button {
    font-size: 14px;
    padding: 8px 12px;
  }
}

/* Dark mode styles */
.chat-page.dark-mode {
  background-color: [[2c2c2c]];
  color: [[ffffff]];
}

.chat-page.dark-mode .chat-header {
  background-color: [[1f1f1f]];
  border-bottom-color: #444444;
}

.chat-page.dark-mode .message.assistant .message-content {
  background-color: [[3a3a3a]];
  border-color: #555555;
}

.chat-page.dark-mode .message.user .message-content {
  background-color: [[4a4a4a]];
}

.chat-page.dark-mode .message-input {
  background-color: [[1f1f1f]];
  border-top-color: #444444;
}

.chat-page.dark-mode .message-input-field {
  background-color: [[3a3a3a]];
  color: [[ffffff]];
  border-color: #555555;
}

.chat-page.dark-mode .send-button {
  background-color: [[007bff]];
}

.chat-page.dark-mode .send-button:hover {
  background-color: [[0056b3]];
}
```

---

## **3. FewShotForm Component (`FewShotForm.tsx`)**

**File:** `frontend/src/components/FewShotForm.tsx`

```tsx
import React, { useState } from 'react';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import fetchWithAuth from '../utils/fetchWithAuth';

const notyf = new Notyf();

const FewShotForm: React.FC = () => {
  const [userPrompt, setUserPrompt] = useState('');
  const [assistantResponse, setAssistantResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!userPrompt.trim() || !assistantResponse.trim()) {
      notyf.error('Both user prompt and assistant response are required.');
      return;
    }

    try {
      const response = await fetchWithAuth('/api/add_few_shot_example', {
        method: 'POST',
        body: JSON.stringify({
          user_prompt: userPrompt.trim(),
          assistant_response: assistantResponse.trim(),
        }),
      });

      if (response.ok) {
        notyf.success('Few-shot example added successfully.');
        setUserPrompt('');
        setAssistantResponse('');
      } else {
        const errorData = await response.json();
        notyf.error(errorData.message || 'Failed to add few-shot example.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Failed to add few-shot example.');
    }
  };

  return (
    <section className="few-shot-form">
      <h2>Add Few-Shot Example</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="user-prompt">User Prompt:</label>
          <input
            type="text"
            id="user-prompt"
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            placeholder="Enter User Prompt"
            aria-label="User Prompt"
          />
        </div>
        <div className="form-group">
          <label htmlFor="assistant-response">Assistant Response:</label>
          <input
            type="text"
            id="assistant-response"
            value={assistantResponse}
            onChange={(e) => setAssistantResponse(e.target.value)}
            placeholder="Enter Assistant Response"
            aria-label="Assistant Response"
          />
        </div>
        <button type="submit" className="btn btn-add">
          Add Example
        </button>
      </form>
    </section>
  );
};

export default FewShotForm;
```

**Styles for FewShotForm (Optional):**

```css
/* Few-Shot Form styles */
.few-shot-form {
  padding: 20px;
  background-color: [[ffffff]];
  border: 1px solid [[e0e0e0]];
  margin: 20px 0;
}

.few-shot-form h2 {
  margin-top: 0;
}

.few-shot-form .form-group {
  margin-bottom: 15px;
}

.few-shot-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.few-shot-form input {
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid [[e0e0e0]];
  border-radius: 5px;
  outline: none;
}

.few-shot-form input:focus {
  border-color: [[007bff]];
}

.few-shot-form .btn-add {
  padding: 10px 20px;
  font-size: 16px;
  background-color: [[007bff]];
  color: [[ffffff]];
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.few-shot-form .btn-add:hover {
  background-color: [[0056b3]];
}
```

---

## **4. FileUploadForm Component (`FileUploadForm.tsx`)**

**File:** `frontend/src/components/FileUploadForm.tsx`

```tsx
import React, { useState } from 'react';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import fetchWithAuth from '../utils/fetchWithAuth';

const notyf = new Notyf();

const FileUploadForm: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      notyf.error('Please select a file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetchWithAuth('/api/upload_file', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        notyf.success('File uploaded and analyzed successfully.');
        setFile(null);
        // Handle response data as needed
      } else {
        const errorData = await response.json();
        notyf.error(errorData.message || 'Failed to upload file.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Failed to upload file.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <section className="file-upload-form">
      <h2>Upload File for Analysis</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="file-input">Choose a file to upload:</label>
          <input
            type="file"
            id="file-input"
            accept=".txt,.json,.md"
            onChange={handleFileChange}
            aria-label="File Input"
          />
        </div>
        <button type="submit" className="btn btn-upload">
          Upload File
        </button>
      </form>
    </section>
  );
};

export default FileUploadForm;
```

**Styles for FileUploadForm (Optional):**

```css
/* File Upload Form styles */
.file-upload-form {
  padding: 20px;
  background-color: [[ffffff]];
  border: 1px solid [[e0e0e0]];
  margin: 20px 0;
}

.file-upload-form h2 {
  margin-top: 0;
}

.file-upload-form .form-group {
  margin-bottom: 15px;
}

.file-upload-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.file-upload-form input[type='file'] {
  font-size: 16px;
}

.file-upload-form .btn-upload {
  padding: 10px 20px;
  font-size: 16px;
  background-color: [[007bff]];
  color: [[ffffff]];
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.file-upload-form .btn-upload:hover {
  background-color: [[0056b3]];
}
```

---

## **5. ConversationList Component (`ConversationList.tsx`)**

**File:** `frontend/src/components/ConversationList.tsx`

```tsx
import React, { useState, useEffect } from 'react';
import fetchWithAuth from '../utils/fetchWithAuth';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';

const notyf = new Notyf();

interface Conversation {
  conversation_id: string;
  title: string;
  last_updated: string;
}

interface Props {
  loadConversation: (conversation_id: string) => void;
}

const ConversationList: React.FC<Props> = ({ loadConversation }) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      const response = await fetchWithAuth('/api/list_conversations', { method: 'GET' });
      const data = await response.json();
      setConversations(data.conversations);
    } catch (error: any) {
      notyf.error(error.message || 'Failed to load conversations.');
    }
  };

  const handleConversationClick = (conversation_id: string) => {
    loadConversation(conversation_id);
  };

  return (
    <div className="conversation-list">
      <h2>Conversations</h2>
      <ul>
        {conversations.map((conv) => (
          <li key={conv.conversation_id}>
            <button onClick={() => handleConversationClick(conv.conversation_id)}>
              {conv.title || 'Untitled Conversation'}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ConversationList;
```

**Styles for ConversationList (Optional):**

```css
/* Conversation List styles */
.conversation-list {
  padding: 20px;
}

.conversation-list h2 {
  margin-top: 0;
}

.conversation-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.conversation-list li {
  margin-bottom: 10px;
}

.conversation-list button {
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 10px;
  font-size: 16px;
  cursor: pointer;
  border-radius: 5px;
  transition: background-color 0.2s;
}

.conversation-list button:hover {
  background-color: [[f0f0f0]];
}
```

---

## **6. SearchForm Component (`SearchForm.tsx`)**

**File:** `frontend/src/components/SearchForm.tsx`

```tsx
import React, { useState } from 'react';
import fetchWithAuth from '../utils/fetchWithAuth';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';

const notyf = new Notyf();

const SearchForm: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!searchQuery.trim()) {
      notyf.error('Please enter a search query.');
      return;
    }

    try {
      const response = await fetchWithAuth('/api/search_conversations', {
        method: 'POST',
        body: JSON.stringify({ query: searchQuery.trim() }),
      });

      const data = await response.json();
      setSearchResults(data.results);
    } catch (error: any) {
      notyf.error(error.message || 'Failed to perform search.');
    }
  };

  return (
    <section className="search-form">
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search conversations..."
          aria-label="Search Conversations"
        />
        <button type="submit" className="btn-search" aria-label="Search">
          <i className="fas fa-search"></i>
        </button>
      </form>

      {searchResults.length > 0 && (
        <div className="search-results">
          <h3>Search Results</h3>
          <ul>
            {searchResults.map((result, index) => (
              <li key={index}>
                {/* Display search result items */}
                {result.title}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
};

export default SearchForm;
```

**Styles for SearchForm (Optional):**

```css
/* Search Form styles */
.search-form {
  padding: 15px 20px;
  background-color: [[ffffff]];
  border-bottom: 1px solid [[e0e0e0]];
}

.search-form form {
  display: flex;
  align-items: center;
}

.search-form input {
  flex: 1;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid [[e0e0e0]];
  border-radius: 20px;
  outline: none;
}

.search-form input:focus {
  border-color: [[007bff]];
}

.btn-search {
  margin-left: 10px;
  background: none;
  border: none;
  font-size: 20px;
  color: #333333;
  cursor: pointer;
}

.btn-search:hover {
  color: [[007bff]];
}

.search-results {
  padding: 20px;
}

.search-results h3 {
  margin-top: 0;
}

.search-results ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.search-results li {
  margin-bottom: 10px;
}
```

---

## **Additional Notes**

- **Importing Font Awesome Icons:**

  To use Font Awesome icons (e.g., in the `Chat.tsx` and `SearchForm.tsx` components), ensure you have installed the Font Awesome packages:

  ```bash
  npm install @fortawesome/react-fontawesome @fortawesome/free-solid-svg-icons @fortawesome/fontawesome-svg-core
  ```

  Then, import the icons as needed:

  ```tsx
  import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
  import { faBars, faPlus, faRedo, faPaperPlane, faSearch } from '@fortawesome/free-solid-svg-icons';
  ```

- **Notyf Notifications:**

  Make sure to import Notyf CSS in your main CSS file or the component where it's used.

  ```tsx
  import 'notyf/notyf.min.css';
  ```

- **Fetch Utility Function (`fetchWithAuth.ts`):**

  Ensure you have the `fetchWithAuth` utility function to handle API requests with authentication.

  **File:** `frontend/src/utils/fetchWithAuth.ts`

  ```tsx
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';

  const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
    const token = localStorage.getItem('jwt_token');
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };

    const response = await fetch(`${API_BASE_URL}${url}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return response;
  };

  export default fetchWithAuth;
  ```

- **Configuration File (`config.ts`):**

  **File:** `frontend/src/utils/config.ts`

  ```tsx
  export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';
  ```

---

## **Conclusion**



**Example Directory Structure:**

```
frontend/
├── src/
│   ├── components/
│   │   ├── Chat.tsx
│   │   ├── Chat.css
│   │   ├── FewShotForm.tsx
│   │   ├── FileUploadForm.tsx
│   │   ├── ConversationList.tsx
│   │   ├── SearchForm.tsx
│   ├── pages/
│   │   └── index.tsx
│   ├── utils/
│       ├── fetchWithAuth.ts
│       └── config.ts
```


## Backend Directory Structure**

Here's the expected directory structure for the backend application:

```
apps/
└── backend/
    ├── package.json
    ├── tsconfig.json
    ├── next.config.js
    ├── pages/
    │   └── api/
    │       ├── add_few_shot_example.ts
    │       ├── get_config.ts
    │       ├── list_conversations.ts
    │       ├── load_conversation/
    │       │   └── [conversation_id].ts
    │       ├── reset_conversation.ts
    │       ├── save_history.ts
    │       ├── search_conversations.ts
    │       ├── send_message.ts
    │       ├── start_conversation.ts
    │       └── upload_file.ts
    ├── utils/
    │   ├── auth.ts
    │   ├── azure.ts
    │   ├── helpers.ts
    │   └── mongodb.ts
    ├── middleware/
    │   └── cors.ts
    ├── types/
    │   └── index.d.ts
    └── .eslintrc.json
```

---

## **1. `package.json`**

**File:** `apps/backend/package.json`

```json
{
  "name": "backend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 4000",
    "build": "next build",
    "start": "next start -p 4000",
    "lint": "next lint",
    "test": "jest"
  },
  "dependencies": {
    "axios": "^0.27.2",
    "dotenv": "^16.0.3",
    "formidable": "^2.0.1",
    "gpt-3-encoder": "^1.1.4",
    "jsonwebtoken": "^9.0.0",
    "mongodb": "^4.12.1",
    "next": "13.5.2",
    "nextjs-cors": "^2.0.6",
    "pusher": "^5.0.3"
  },
  "devDependencies": {
    "@types/formidable": "^2.0.5",
    "@types/jest": "^29.2.3",
    "@types/jsonwebtoken": "^9.0.1",
    "@types/node": "^18.11.9",
    "eslint": "^8.26.0",
    "eslint-config-next": "13.5.2",
    "jest": "^29.2.1",
    "ts-jest": "^29.0.3",
    "typescript": "^4.9.4"
  },
  "engines": {
    "node": ">=14.17.0"
  }
}
```

---

## **2. `tsconfig.json`**

**File:** `apps/backend/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES6",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "esModuleInterop": true,
    "jsx": "preserve",
    "incremental": true,
    "baseUrl": ".",
    "paths": {
      "@/utils/*": ["utils/*"],
      "@/middleware/*": ["middleware/*"],
      "@/types/*": ["types/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

---

## **3. `next.config.js`** (Optional)

If you need custom configurations for Next.js.

**File:** `apps/backend/next.config.js`

```js
/** @type {import('next').NextConfig} */
module.exports = {
  // Enable TypeScript strict mode
  typescript: {
    ignoreBuildErrors: false,
  },
  // Other Next.js configurations can be added here
};
```

---

## **4. `pages/api/` Directory with API Route Files**

### **Overview**

The API routes handle various backend functionalities, such as starting a conversation, sending messages, resetting conversations, etc.

Each file corresponds to an endpoint.

---

### **4.1. `start_conversation.ts`**

**File:** `apps/backend/pages/api/start_conversation.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { v4 as uuidv4 } from 'uuid';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    const conversationId = uuidv4();
    await conversations.insertOne({
      conversation_id: conversationId,
      user_id: user.id,
      messages: [],
      created_at: new Date(),
      updated_at: new Date(),
    });

    res.status(200).json({ conversation_id: conversationId });
  } catch (error: any) {
    console.error('Error starting conversation:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.2. `send_message.ts`**

**File:** `apps/backend/pages/api/send_message.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { getAzureResponse } from '@/utils/azure';
import { PusherInstance } from '@/utils/pusher';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const { conversation_id, message } = req.body;

  if (!conversation_id || !message) {
    return res.status(400).json({ message: 'Conversation ID and message are required.' });
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    const conversation = await conversations.findOne({
      conversation_id,
      user_id: user.id,
    });

    if (!conversation) {
      return res.status(404).json({ message: 'Conversation not found.' });
    }

    // Add user's message to the conversation
    await conversations.updateOne(
      { conversation_id },
      {
        $push: {
          messages: { role: 'user', content: message },
        },
        $set: { updated_at: new Date() },
      }
    );

    // Get assistant's response from Azure OpenAI API
    const assistantResponse = await getAzureResponse([...conversation.messages, { role: 'user', content: message }]);

    // Add assistant's response to the conversation
    await conversations.updateOne(
      { conversation_id },
      {
        $push: {
          messages: { role: 'assistant', content: assistantResponse },
        },
        $set: { updated_at: new Date() },
      }
    );

    // Emit the messages via Pusher
    const pusher = PusherInstance;
    pusher.trigger('chat-channel', 'new-message', {
      conversation_id,
      role: 'user',
      content: message,
    });
    pusher.trigger('chat-channel', 'new-message', {
      conversation_id,
      role: 'assistant',
      content: assistantResponse,
    });

    res.status(200).json({ message: 'Message sent successfully.' });
  } catch (error: any) {
    console.error('Error sending message:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.3. `reset_conversation.ts`**

**File:** `apps/backend/pages/api/reset_conversation.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const { conversation_id } = req.body;

  if (!conversation_id) {
    return res.status(400).json({ message: 'Conversation ID is required.' });
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    await conversations.updateOne(
      { conversation_id, user_id: user.id },
      {
        $set: {
          messages: [],
          updated_at: new Date(),
        },
      }
    );

    res.status(200).json({ message: 'Conversation reset successfully.' });
  } catch (error: any) {
    console.error('Error resetting conversation:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.4. `list_conversations.ts`**

**File:** `apps/backend/pages/api/list_conversations.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    const convos = await conversations
      .find({ user_id: user.id })
      .project({ conversation_id: 1, title: 1, updated_at: 1 })
      .sort({ updated_at: -1 })
      .toArray();

    res.status(200).json({ conversations: convos });
  } catch (error: any) {
    console.error('Error listing conversations:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.5. `load_conversation/[conversation_id].ts`**

**File:** `apps/backend/pages/api/load_conversation/[conversation_id].ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  const { conversation_id } = req.query;

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  if (!conversation_id || typeof conversation_id !== 'string') {
    return res.status(400).json({ message: 'Conversation ID is required.' });
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    const conversation = await conversations.findOne({
      conversation_id,
      user_id: user.id,
    });

    if (!conversation) {
      return res.status(404).json({ message: 'Conversation not found.' });
    }

    res.status(200).json({ conversation: conversation.messages });
  } catch (error: any) {
    console.error('Error loading conversation:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.6. `add_few_shot_example.ts`**

**File:** `apps/backend/pages/api/add_few_shot_example.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const { user_prompt, assistant_response } = req.body;

  if (!user_prompt || !assistant_response) {
    return res.status(400).json({ message: 'User prompt and assistant response are required.' });
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const fewShotExamples = db.collection('few_shot_examples');

    await fewShotExamples.insertOne({
      user_id: user.id,
      user_prompt,
      assistant_response,
      created_at: new Date(),
    });

    res.status(200).json({ message: 'Few-shot example added successfully.' });
  } catch (error: any) {
    console.error('Error adding few-shot example:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.7. `upload_file.ts`**

**File:** `apps/backend/pages/api/upload_file.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { IncomingForm } from 'formidable';
import { authenticate } from '@/utils/auth';
import { allowedFile, fileSizeUnderLimit, MAX_FILE_SIZE_MB } from '@/utils/helpers';
import { analyzeFileContent } from '@/utils/azure';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const form = new IncomingForm();

  form.parse(req, async (err, fields, files) => {
    if (err) {
      console.error('Formidable error:', err);
      return res.status(500).json({ message: 'Error parsing the uploaded file.' });
    }

    const file = files.file as formidable.File;

    if (!file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }

    if (!allowedFile(file.originalFilename || '')) {
      return res.status(400).json({ message: 'File type not allowed.' });
    }

    if (!fileSizeUnderLimit(file.size)) {
      return res
        .status(400)
        .json({ message: `File size exceeds the limit of ${MAX_FILE_SIZE_MB} MB.` });
    }

    try {
      // Read file content
      const content = await fs.promises.readFile(file.filepath, 'utf-8');

      // Analyze file content using Azure OpenAI API
      const analysis = await analyzeFileContent(content);

      res.status(200).json({ analysis });
    } catch (error: any) {
      console.error('Error processing file:', error);
      res.status(500).json({ message: 'An error occurred.', error: error.message });
    }
  });
}
```

---

### **4.8. `search_conversations.ts`**

**File:** `apps/backend/pages/api/search_conversations.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const { query } = req.body;

  if (!query) {
    return res.status(400).json({ message: 'Search query is required.' });
  }

  try {
    const client = await clientPromise;
    const db = client.db('your-database-name');
    const conversations = db.collection('conversations');

    const results = await conversations
      .find({
        user_id: user.id,
        'messages.content': { $regex: query, $options: 'i' },
      })
      .project({ conversation_id: 1, title: 1, updated_at: 1 })
      .toArray();

    res.status(200).json({ results });
  } catch (error: any) {
    console.error('Error searching conversations:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

### **4.9. `get_config.ts`**

**File:** `apps/backend/pages/api/get_config.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  try {
    const config = {
      max_tokens: process.env.MAX_TOKENS || '128000',
      reply_tokens: process.env.REPLY_TOKENS || '800',
      chunk_size_tokens: process.env.CHUNK_SIZE_TOKENS || '1000',
    };

    res.status(200).json({ config });
  } catch (error: any) {
    console.error('Error getting config:', error);
    res.status(500).json({ message: 'An error occurred.', error: error.message });
  }
}
```

---

## **5. `utils/` Directory with Utility Functions**

### **5.1. `auth.ts`**

Handles authentication using JWT tokens.

**File:** `apps/backend/utils/auth.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import jwt from 'jsonwebtoken';

interface User {
  id: string;
  email: string;
}

export function authenticate(req: NextApiRequest, res: NextApiResponse): User | null {
  const token = req.headers.authorization?.split(' ')[1];

  if (!token) {
    res.status(401).json({ message: 'Authorization token missing.' });
    return null;
  }

  try {
    const secret = process.env.JWT_SECRET || 'your_jwt_secret';
    const decoded = jwt.verify(token, secret) as User;
    return decoded;
  } catch (error) {
    res.status(401).json({ message: 'Invalid authorization token.' });
    return null;
  }
}
```

---

### **5.2. `mongodb.ts`**

Manages the MongoDB connection.

**File:** `apps/backend/utils/mongodb.ts`

```typescript
import { MongoClient } from 'mongodb';

const uri = process.env.MONGODB_URI || '';
const options = {};

if (!uri) {
  throw new Error('Please add your MongoDB URI to .env.local');
}

let client: MongoClient;
let clientPromise: Promise<MongoClient>;

declare global {
  // eslint-disable-next-line no-var
  var _mongoClientPromise: Promise<MongoClient>;
}

if (process.env.NODE_ENV === 'development') {
  if (!global._mongoClientPromise) {
    client = new MongoClient(uri, options);
    global._mongoClientPromise = client.connect();
  }
  clientPromise = global._mongoClientPromise;
} else {
  client = new MongoClient(uri, options);
  clientPromise = client.connect();
}

export default clientPromise;
```

---

### **5.3. `azure.ts`**

Handles interactions with the Azure OpenAI API.

**File:** `apps/backend/utils/azure.ts`

```typescript
import axios from 'axios';

export async function getAzureResponse(messages: any[]): Promise<string> {
  const apiUrl = process.env.AZURE_API_URL || '';
  const apiKey = process.env.API_KEY || '';

  if (!apiUrl || !apiKey) {
    throw new Error('Azure API URL and API Key must be set.');
  }

  try {
    const response = await axios.post(
      apiUrl,
      {
        messages,
        max_tokens: parseInt(process.env.REPLY_TOKENS || '800', 10),
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'api-key': apiKey,
        },
      }
    );

    return response.data.choices[0].message.content;
  } catch (error: any) {
    console.error('Error calling Azure OpenAI API:', error.response?.data || error.message);
    throw new Error('Failed to get response from Azure OpenAI API.');
  }
}

export async function analyzeFileContent(content: string): Promise<string> {
  // Implement the logic to analyze file content using Azure OpenAI API
  // For example, you can use the summarization or analysis capabilities
  // This is a placeholder function
  return 'Analysis result of the file content.';
}
```

---

### **5.4. `helpers.ts`**

Contains helper functions such as file validation.

**File:** `apps/backend/utils/helpers.ts`

```typescript
import { encode } from 'gpt-3-encoder';

export const MAX_FILE_SIZE_MB = parseFloat(process.env.MAX_FILE_SIZE_MB || '5.0');
export const ALLOWED_EXTENSIONS = (process.env.ALLOWED_EXTENSIONS || 'txt,md,json')
  .split(',')
  .map((ext) => ext.trim().toLowerCase());

export function allowedFile(filename: string): boolean {
  const extension = filename.split('.').pop()?.toLowerCase();
  return ALLOWED_EXTENSIONS.includes(extension || '');
}

export function fileSizeUnderLimit(sizeInBytes: number): boolean {
  const sizeInMB = sizeInBytes / (1024 * 1024);
  return sizeInMB <= MAX_FILE_SIZE_MB;
}

export function countTokens(text: string): number {
  const tokens = encode(text);
  return tokens.length;
}
```

---

### **5.5. `pusher.ts`**

Initializes the Pusher instance.

**File:** `apps/backend/utils/pusher.ts`

```typescript
import Pusher from 'pusher';

export const PusherInstance = new Pusher({
  appId: process.env.PUSHER_APP_ID || '',
  key: process.env.PUSHER_KEY || '',
  secret: process.env.PUSHER_SECRET || '',
  cluster: process.env.PUSHER_CLUSTER || '',
  useTLS: true,
});
```

---

## **6. `middleware/` Directory with Middleware Functions**

### **6.1. `cors.ts`**

Handles CORS configuration.

**File:** `apps/backend/middleware/cors.ts`

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import NextCors from 'nextjs-cors';

export async function cors(req: NextApiRequest, res: NextApiResponse) {
  await NextCors(req, res, {
    origin: (process.env.ALLOWED_ORIGINS || '*').split(','),
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    optionsSuccessStatus: 200,
  });
}
```

---

## **7. `types/` Directory for TypeScript Interfaces** (Optional)

**File:** `apps/backend/types/index.d.ts`

```typescript
export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  conversation_id: string;
  user_id: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
  title?: string;
}

export interface FewShotExample {
  user_id: string;
  user_prompt: string;
  assistant_response: string;
  created_at: Date;
}

export interface User {
  id: string;
  email: string;
}
```

---

## **8. Environment Variables Setup**

The backend application requires several environment variables. These should be defined in a `.env.local` file in the `apps/backend/` directory and set in your deployment environment (e.g., Vercel).

**File:** `apps/backend/.env.local` (Do not commit this file to version control)

```ini
# Azure OpenAI API
AZURE_API_URL=your_azure_api_url
API_KEY=your_azure_api_key

# MongoDB Atlas
MONGODB_URI=your_mongodb_uri

# JWT Secret
JWT_SECRET=your_jwt_secret

# Pusher Credentials
PUSHER_APP_ID=your_pusher_app_id
PUSHER_KEY=your_pusher_key
PUSHER_SECRET=your_pusher_secret
PUSHER_CLUSTER=your_pusher_cluster

# Configuration Settings
MAX_TOKENS=128000
REPLY_TOKENS=800
CHUNK_SIZE_TOKENS=1000
MAX_FILE_SIZE_MB=5.0
ALLOWED_EXTENSIONS=txt,md,json

# CORS Allowed Origins
ALLOWED_ORIGINS=https://your-frontend-app.vercel.app
```

**Note:** Replace the placeholder values with your actual credentials and settings. Do not commit this file to your repository. Ensure these variables are also set in your deployment environment (e.g., Vercel's project settings).

---

Certainly! I'll provide any remaining files needed for your application to function correctly, ensuring that all placeholder functions and incomplete logic are fully implemented. The files include:

1. **Frontend `pages` Directory**
   - `index.tsx`: The main page that imports and uses the components.

2. **Frontend `utils` Directory**
   - `fetchWithAuth.ts`: Utility function for making authenticated API requests.
   - `config.ts`: Configuration constants.
   - `auth.ts`: Functions for handling authentication (e.g., storing and retrieving JWT tokens).

3. **Frontend Global Styles**
   - `globals.css`: Global CSS styles for your application.

4. **Frontend `App` Component**
   - `_app.tsx`: Custom App component for Next.js to include global styles and provide context.

5. **Frontend `Document` Component**
   - `_document.tsx`: Custom Document component for Next.js to modify the HTML document structure if needed.

6. **Additional Backend Files**
   - Ensure all placeholder functions are fully implemented in the backend code provided earlier.

---

I'll provide each file in detail, ensuring completeness and correctness.

---

## **Frontend Files**

### **1. `pages/index.tsx`**

**File:** `apps/frontend/src/pages/index.tsx`

```tsx
import React from 'react';
import Chat from '../components/Chat';
import FewShotForm from '../components/FewShotForm';
import FileUploadForm from '../components/FileUploadForm';
import SearchForm from '../components/SearchForm';

const HomePage: React.FC = () => {
  return (
    <div className="home-page">
      {/* You can include a header or navigation here if needed */}
      <SearchForm />
      <div className="main-content">
        <Chat />
        <aside className="side-panel">
          <FewShotForm />
          <FileUploadForm />
        </aside>
      </div>
    </div>
  );
};

export default HomePage;
```

**Notes:**

- The `HomePage` component imports and uses the `Chat`, `FewShotForm`, `FileUploadForm`, and `SearchForm` components.
- You can adjust the layout as needed, possibly by adding a header or adjusting the structure.

---

### **2. `utils/fetchWithAuth.ts`**

**File:** `apps/frontend/src/utils/fetchWithAuth.ts`

```tsx
import { API_BASE_URL } from './config';

const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
  const token = localStorage.getItem('jwt_token');

  const headers = {
    ...(options.headers || {}),
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  const response = await fetch(`${API_BASE_URL}${url}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
  }

  return response;
};

export default fetchWithAuth;
```

**Notes:**

- This utility function handles API requests, including the JWT token for authentication.
- It throws an error if the response is not OK, including any error message from the server.

---

### **3. `utils/config.ts`**

**File:** `apps/frontend/src/utils/config.ts`

```tsx
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';
```

**Notes:**

- `API_BASE_URL` is the base URL for your backend API.
- Ensure that `NEXT_PUBLIC_API_BASE_URL` is set in your environment variables.

---

### **4. `utils/auth.ts`**

**File:** `apps/frontend/src/utils/auth.ts`

```tsx
import jwt_decode from 'jwt-decode';

interface User {
  id: string;
  email: string;
}

export function saveToken(token: string) {
  localStorage.setItem('jwt_token', token);
}

export function getToken(): string | null {
  return localStorage.getItem('jwt_token');
}

export function removeToken() {
  localStorage.removeItem('jwt_token');
}

export function getUser(): User | null {
  const token = getToken();
  if (token) {
    try {
      const user = jwt_decode<User>(token);
      return user;
    } catch (error) {
      console.error('Invalid token:', error);
      removeToken();
      return null;
    }
  }
  return null;
}
```

**Notes:**

- This file includes functions to save, retrieve, and remove the JWT token from `localStorage`.
- It also includes a function to decode the token and retrieve the user information.
- Install `jwt-decode` if you haven't already:

  ```bash
  npm install jwt-decode
  ```

---

### **5. `styles/globals.css`**

**File:** `apps/frontend/src/styles/globals.css`

```css
/* Reset styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* Global styles */
body {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  background-color: [[f5f5f5]];
  color: #333333;
}

/* Utility classes */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.btn {
  display: inline-block;
  padding: 10px 20px;
  font-size: 16px;
  background-color: [[007bff]];
  color: [[ffffff]];
  text-align: center;
  text-decoration: none;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.btn:hover {
  background-color: [[0056b3]];
}

/* Other global styles as needed */
```

**Notes:**

- This file includes global CSS styles applied throughout your application.
- You can customize or extend this file as needed.

---

### **6. `_app.tsx`**

**File:** `apps/frontend/src/pages/_app.tsx`

```tsx
import React from 'react';
import type { AppProps } from 'next/app';
import '../styles/globals.css';

const MyApp = ({ Component, pageProps }: AppProps) => {
  return <Component {...pageProps} />;
};

export default MyApp;
```

**Notes:**

- The custom App component includes the global CSS styles.
- It renders the page components with any provided props.

---

### **7. `_document.tsx`**

**File:** `apps/frontend/src/pages/_document.tsx`

```tsx
import Document, { Html, Head, Main, NextScript, DocumentContext } from 'next/document';

class MyDocument extends Document {
  static async getInitialProps(ctx: DocumentContext) {
    const initialProps = await Document.getInitialProps(ctx);
    return { ...initialProps };
  }

  render() {
    return (
      <Html lang="en">
        <Head>
          {/* Include any additional head elements here */}
          <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.4/css/all.css" />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
```

**Notes:**

- The custom Document component can be used to modify the HTML document structure.
- Here, we include a link to Font Awesome CSS for icon usage.
- Ensure that you have the appropriate license or use a different method for including icons.

---

### **8. Adjusting Placeholder Functions**

In the previously provided components, there were placeholder functions like `listConversations` in `Chat.tsx` that needed full implementation.

#### **Update `Chat.tsx` Placeholder Functions**

**File:** `apps/frontend/src/components/Chat.tsx`

```tsx
// ... existing imports and code ...

const Chat: React.FC = () => {
  // ... existing state and useEffect hooks ...

  // Update the listConversations function to toggle the sidebar
  const listConversations = () => {
    toggleSidebar();
  };

  // Update the startNewConversation function to close the sidebar
  const startNewConversation = async () => {
    try {
      const response = await fetchWithAuth('/api/start_conversation', { method: 'POST' });
      const data = await response.json();
      setConversationId(data.conversation_id);
      setMessages([]);
      notyf.success('Started a new conversation.');
      setIsSidebarOpen(false); // Close the sidebar if open
    } catch (error: any) {
      notyf.error(error.message || 'Failed to start a new conversation.');
    }
  };

  // ... rest of the component code ...
};

export default Chat;
```

**Notes:**

- The `listConversations` function now toggles the sidebar visibility.
- The `startNewConversation` function closes the sidebar if it's open.

---

### **9. `utils/pusher.ts`**

Ensure Pusher is properly initialized and used in the frontend.

**File:** `apps/frontend/src/utils/pusher.ts`

```tsx
import Pusher from 'pusher-js';

export const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY || '', {
  cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER || '',
});

export default pusher;
```

**Notes:**

- This utility file initializes Pusher for use across your frontend application.
- Ensure that `NEXT_PUBLIC_PUSHER_KEY` and `NEXT_PUBLIC_PUSHER_CLUSTER` are set in your environment variables.

---

## **Backend Files**

Review the backend code to ensure all placeholder functions and incomplete logic are fully implemented.

### **1. Complete the `azure.ts` Functions**

In `azure.ts`, the `analyzeFileContent` function was a placeholder. We'll provide a complete implementation.

**File:** `apps/backend/utils/azure.ts`

```typescript
import axios from 'axios';

export async function getAzureResponse(messages: any[]): Promise<string> {
  const apiUrl = process.env.AZURE_API_URL || '';
  const apiKey = process.env.API_KEY || '';

  if (!apiUrl || !apiKey) {
    throw new Error('Azure API URL and API Key must be set.');
  }

  try {
    const response = await axios.post(
      apiUrl,
      {
        messages,
        max_tokens: parseInt(process.env.REPLY_TOKENS || '800', 10),
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'api-key': apiKey,
        },
      }
    );

    return response.data.choices[0].message.content;
  } catch (error: any) {
    console.error('Error calling Azure OpenAI API:', error.response?.data || error.message);
    throw new Error('Failed to get response from Azure OpenAI API.');
  }
}

export async function analyzeFileContent(content: string): Promise<string> {
  const apiUrl = process.env.AZURE_API_URL || '';
  const apiKey = process.env.API_KEY || '';

  if (!apiUrl || !apiKey) {
    throw new Error('Azure API URL and API Key must be set.');
  }

  try {
    const response = await axios.post(
      apiUrl,
      {
        prompt: `Analyze the following content:\n\n${content}\n\nProvide a detailed analysis.`,
        max_tokens: parseInt(process.env.REPLY_TOKENS || '800', 10),
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'api-key': apiKey,
        },
      }
    );

    return response.data.choices[0].text;
  } catch (error: any) {
    console.error('Error analyzing file content:', error.response?.data || error.message);
    throw new Error('Failed to analyze file content with Azure OpenAI API.');
  }
}
```

**Notes:**

- The `analyzeFileContent` function now sends a prompt to Azure OpenAI API to analyze the file content.
- Adjust the prompt as needed to suit your application's requirements.

---

### **2. Complete `upload_file.ts`**

Ensure that `upload_file.ts` fully handles file uploads and content analysis.

**File:** `apps/backend/pages/api/upload_file.ts`

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import formidable, { IncomingForm } from 'formidable';
import { promises as fs } from 'fs';
import { authenticate } from '@/utils/auth';
import { allowedFile, fileSizeUnderLimit, MAX_FILE_SIZE_MB } from '@/utils/helpers';
import { analyzeFileContent } from '@/utils/azure';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const user = authenticate(req, res);
  if (!user) return;

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const form = new IncomingForm();

  form.parse(req, async (err, fields, files) => {
    if (err) {
      console.error('Formidable error:', err);
      return res.status(500).json({ message: 'Error parsing the uploaded file.' });
    }

    const file = files.file as formidable.File;

    if (!file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }

    if (!allowedFile(file.originalFilename || '')) {
      return res.status(400).json({ message: 'File type not allowed.' });
    }

    if (!fileSizeUnderLimit(file.size)) {
      return res
        .status(400)
        .json({ message: `File size exceeds the limit of ${MAX_FILE_SIZE_MB} MB.` });
    }

    try {
      // Read file content
      const content = await fs.readFile(file.filepath, 'utf-8');

      // Analyze file content using Azure OpenAI API
      const analysis = await analyzeFileContent(content);

      res.status(200).json({ analysis });
    } catch (error: any) {
      console.error('Error processing file:', error);
      res.status(500).json({ message: 'An error occurred.', error: error.message });
    }
  });
}
```