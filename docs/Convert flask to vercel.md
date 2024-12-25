Migrating your existing **Flask-SocketIO** and **WebSockets**-based application to a **Vercel** deployment using a **React Turborepo** monorepo is a significant shift that modernizes your tech stack, enhances scalability, and leverages Vercel's optimized serverless infrastructure. This migration involves transitioning your frontend from vanilla JavaScript to React, replacing Flask with Next.js API routes, and adopting a more suitable real-time communication method compatible with serverless environments.

This comprehensive guide will walk you through the entire migration process, ensuring that your application's core functionalities are preserved and enhanced in the new setup. We will utilize **Turborepo** for efficient monorepo management, **React** for the frontend, **Next.js** for serverless backend functions, and **Pusher** for real-time features, eliminating the need for persistent WebSocket connections.

---

## Table of Contents

1. [Prerequisites](Convert%2520flask%2520to%2520vercel.md##1-prerequisites)
2. [Overview of Migration](Convert%2520flask%2520to%2520vercel.md##2-overview-of-migration)
3. [Setting Up the Turborepo Monorepo](Convert%2520flask%2520to%2520vercel.md##3-setting-up-the-turborepo-monorepo)
    - [3.1. Installing Turborepo](Convert%2520flask%2520to%2520vercel.md##31-installing-turborepo)
    - [3.2. Initializing the Monorepo](Convert%2520flask%2520to%2520vercel.md##32-initializing-the-monorepo)
4. [Migrating the Frontend (`script.js` to React)](#4-migrating-the-frontend-scriptjs-to-react)
    - [4.1. Creating the React Application](Convert%2520flask%2520to%2520vercel.md##41-creating-the-react-application)
    - [4.2. Converting `script.js` to React Components and Hooks](#42-converting-scriptjs-to-react-components-and-hooks)
    - [4.3. Integrating Real-Time Communication with Pusher](Convert%2520flask%2520to%2520vercel.md##43-integrating-real-time-communication-with-pusher)
5. [Migrating the Backend (`app.py` to Next.js Serverless Functions)](#5-migrating-the-backend-apppy-to-nextjs-serverless-functions)
    - [5.1. Setting Up Next.js in the Monorepo](Convert%2520flask%2520to%2520vercel.md##51-setting-up-nextjs-in-the-monorepo)
    - [5.2. Converting Flask Routes to Next.js API Routes](Convert%2520flask%2520to%2520vercel.md##52-converting-flask-routes-to-nextjs-api-routes)
    - [5.3. Incorporating `utils.py` Functionality](#53-incorporating-utilspy-functionality)
6. [Implementing Real-Time Features Without WebSockets](Convert%2520flask%2520to%2520vercel.md##6-implementing-real-time-features-without-websockets)
    - [6.1. Utilizing Pusher for Real-Time Communication](Convert%2520flask%2520to%2520vercel.md##61-utilizing-pusher-for-real-time-communication)
    - [6.2. Updating Frontend to Use Pusher](Convert%2520flask%2520to%2520vercel.md##62-updating-frontend-to-use-pusher)
7. [Handling Sessions and Authentication](Convert%2520flask%2520to%2520vercel.md##7-handling-sessions-and-authentication)
    - [7.1. Implementing JWT Authentication](Convert%2520flask%2520to%2520vercel.md##71-implementing-jwt-authentication)
    - [7.2. Securing API Routes](Convert%2520flask%2520to%2520vercel.md##72-securing-api-routes)
8. [Managing Tokens and API Usage](Convert%2520flask%2520to%2520vercel.md##8-managing-tokens-and-api-usage)
    - [8.1. Tracking Token Usage](Convert%2520flask%2520to%2520vercel.md##81-tracking-token-usage)
    - [8.2. Enforcing Token Limits](Convert%2520flask%2520to%2520vercel.md##82-enforcing-token-limits)
9. [Configuring Environment Variables](Convert%2520flask%2520to%2520vercel.md##9-configuring-environment-variables)
    - [9.1. Setting Up Environment Variables for Frontend](Convert%2520flask%2520to%2520vercel.md##91-setting-up-environment-variables-for-frontend)
    - [9.2. Setting Up Environment Variables for Backend](Convert%2520flask%2520to%2520vercel.md##92-setting-up-environment-variables-for-backend)
10. [Deploying to Vercel](Convert%2520flask%2520to%2520vercel.md##10-deploying-to-vercel)
    - [10.1. Connecting Your Repository](Convert%2520flask%2520to%2520vercel.md##101-connecting-your-repository)
    - [10.2. Configuring Vercel Projects](Convert%2520flask%2520to%2520vercel.md##102-configuring-vercel-projects)
    - [10.3. Deploying Frontend and Backend](Convert%2520flask%2520to%2520vercel.md##103-deploying-frontend-and-backend)
11. [Dependencies, Logging, and Testing](Convert%2520flask%2520to%2520vercel.md##11-dependencies-logging-and-testing)
    - [11.1. Managing Dependencies with Turborepo](Convert%2520flask%2520to%2520vercel.md##111-managing-dependencies-with-turborepo)
    - [11.2. Implementing Logging](Convert%2520flask%2520to%2520vercel.md##112-implementing-logging)
    - [11.3. Setting Up Testing](Convert%2520flask%2520to%2520vercel.md##113-setting-up-testing)
12. [Conclusion](Convert%2520flask%2520to%2520vercel.md##12-conclusion)
13. [Appendix](Convert%2520flask%2520to%2520vercel.md##13-appendix)
    - [13.1. Example Directory Structure](Convert%2520flask%2520to%2520vercel.md##131-example-directory-structure)
    - [13.2. Sample Configuration Files](Convert%2520flask%2520to%2520vercel.md##132-sample-configuration-files)

---

## 1. Prerequisites

Before embarking on the migration, ensure you have the following tools and accounts set up:

- **Node.js** (v14 or higher): [Download Node.js](https://nodejs.org/)
- **npm** or **yarn**: Comes bundled with Node.js or install separately.
- **Git**: [Install Git](https://git-scm.com/)
- **Vercel Account**: [Sign Up for Vercel](https://vercel.com/signup)
- **MongoDB Atlas Account**: [Sign Up for MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- **Pusher Account**: [Sign Up for Pusher](https://pusher.com/signup)
- **Redis Service**: Use a managed Redis service like [Redis Cloud](https://redis.com/redis-enterprise-cloud/overview/) or [Heroku Redis](https://elements.heroku.com/addons/heroku-redis)

---

## 2. Overview of Migration

**Current Setup:**

- **Frontend:** Vanilla JavaScript (`script.js`)
- **Backend:** Flask with Socket.IO for real-time communication
- **Real-Time Communication:** WebSockets
- **Database:** MongoDB
- **Session Management:** Redis
- **Utilities:** `utils.py` for token management and other helper functions

**New Setup:**

- **Monorepo Management:** Turborepo
- **Frontend:** React with modern build tools (Tsup, Storybook)
- **Backend:** Next.js API Routes as serverless functions
- **Real-Time Communication:** Pusher (eliminates the need for WebSockets)
- **Database:** MongoDB (remains the same)
- **Session Management:** JWTs (JSON Web Tokens) for stateless authentication
- **Utilities:** Converted to TypeScript modules compatible with Next.js

**Key Changes:**

- Transition from Flask to Next.js for backend API routes.
- Replace WebSockets with Pusher for real-time features.
- Migrate frontend from vanilla JS to React, enhancing maintainability and scalability.
- Adopt Turborepo for efficient monorepo management, enabling parallel development of frontend and backend.
- Implement JWT-based authentication for secure, stateless session management.

---

## 3. Setting Up the Turborepo Monorepo

**Turborepo** is a high-performance build system for JavaScript and TypeScript monorepos. It allows you to manage multiple projects (packages) within a single repository efficiently.

### 3.1. Installing Turborepo

First, install Turborepo globally (optional but recommended for ease of use):

```bash
npm install -g turbo
# or
yarn global add turbo
```

### 3.2. Initializing the Monorepo

1. **Create the Monorepo Directory:**

   ```bash
   mkdir web-chat
   cd web-chat
   ```

2. **Initialize a Git Repository:**

   ```bash
   git init
   ```

3. **Initialize `package.json`:**

   ```bash
   npm init -y
   # or
   yarn init -y
   ```

4. **Install Turborepo as a Dev Dependency:**

   ```bash
   npm install turbo --save-dev
   # or
   yarn add turbo --dev
   ```

5. **Configure Turborepo:**

   Create a `turbo.json` file at the root of the monorepo:

   ```json
   // turbo.json
   {
     "pipeline": {
       "build": {
         "dependsOn": ["^build"],
         "outputs": ["dist/**", ".next/**"]
       },
       "lint": {
         "outputs": []
       },
       "test": {
         "outputs": []
       }
     }
   }
   ```

6. **Update Root `package.json`:**

   Add Turborepo scripts and configure workspaces:

   ```json
   // package.json
   {
     "name": "web-chat",
     "version": "1.0.0",
     "private": true,
     "scripts": {
       "dev": "turbo run dev --parallel",
       "build": "turbo run build",
       "lint": "turbo run lint",
       "test": "turbo run test"
     },
     "devDependencies": {
       "turbo": "^1.8.3" // Ensure the version matches the installed Turborepo
     },
     "workspaces": [
       "apps/*"
     ]
   }
   ```

7. **Create the Workspace Structure:**

   ```bash
   mkdir -p apps/frontend
   mkdir -p apps/backend
   ```

---

## 4. Migrating the Frontend (`script.js` to React)

Transitioning your frontend from vanilla JavaScript to React enhances maintainability, scalability, and developer experience. We'll also integrate modern tools like **Tsup** for bundling and **Storybook** for UI component development.

### 4.1. Creating the React Application

Within the monorepo, set up the React application using **Create React App** or **Vite**. We'll use **Vite** for its speed and flexibility.

1. **Navigate to the Frontend Directory:**

   ```bash
   cd apps/frontend
   ```

2. **Initialize the React App with TypeScript Template:**

   ```bash
   npm create vite@latest . -- --template react-ts
   # or
   yarn create vite . --template react-ts
   ```

3. **Install Dependencies:**

   ```bash
   npm install
   # or
   yarn
   ```

4. **Install Additional Dependencies:**

   ```bash
   npm install socket.io-client notyf pusher-js
   # or
   yarn add socket.io-client notyf pusher-js
   ```

   - **`socket.io-client`:** If you decide to retain any Socket.IO functionality (optional).
   - **`notyf`:** For notifications.
   - **`pusher-js`:** For real-time communication with Pusher.

5. **Set Up Storybook (Optional for UI Components):**

   ```bash
   npx sb init
   # or
   yarn sb init
   ```

### 4.2. Converting `script.js` to React Components and Hooks

We'll translate the functionalities of `script.js` into React components using hooks for state management and side effects.

#### 4.2.1. Creating the Chat Component

Create a `Chat.tsx` component within `src/components/Chat.tsx`:

```tsx
// apps/frontend/src/components/Chat.tsx

import React, { useState, useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import Pusher from 'pusher-js';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import './Chat.css'; // Create corresponding CSS

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface TokenUsage {
  total_tokens_used: number;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [userMessage, setUserMessage] = useState('');
  const [tokenUsage, setTokenUsage] = useState<number>(0);
  const [conversationId, setConversationId] = useState<string | null>(
    sessionStorage.getItem('conversation_id')
  );
  const socketRef = useRef<Socket | null>(null);
  const pusherRef = useRef<Pusher | null>(null);
  const notyf = useRef(
    new Notyf({
      duration: 3000,
      position: { x: 'right', y: 'top' },
      types: [
        {
          type: 'success',
          background: '#28a745',
          icon: false
        },
        {
          type: 'error',
          background: '#dc3545',
          icon: false
        }
      ]
    })
  ).current;

  const chatHistoryRef = useRef<HTMLDivElement | null>(null);

  // Initialize Pusher
  useEffect(() => {
    if (conversationId) {
      pusherRef.current = new Pusher(process.env.REACT_APP_PUSHER_KEY || '', {
        cluster: process.env.REACT_APP_PUSHER_CLUSTER || '',
        encrypted: true
      });

      const channel = pusherRef.current.subscribe(conversationId);
      channel.bind('new-message', (data: Message) => {
        setMessages((prev) => [...prev, data]);
      });

      return () => {
        channel.unbind_all();
        channel.unsubscribe();
        pusherRef.current?.disconnect();
      };
    }
  }, [conversationId, notyf]);

  // Scroll to bottom when messages update
  useEffect(() => {
    chatHistoryRef.current?.scrollTo(0, chatHistoryRef.current.scrollHeight);
  }, [messages]);

  // Fetch configuration on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('/api/get_config');
        if (!response.ok) throw new Error('Failed to fetch config.');
        const data = await response.json();
        // Assuming MAX_TOKENS is returned
        // setMaxTokens(data.max_tokens);
      } catch (error) {
        console.error(error);
        notyf.error('Failed to load configuration.');
      }
    };

    fetchConfig();
  }, [notyf]);

  // Initialize conversation if not existing
  useEffect(() => {
    if (!conversationId) {
      startNewConversation();
    } else {
      loadConversation(conversationId);
    }
  }, [conversationId]);

  const startNewConversation = async () => {
    try {
      const response = await fetch('/api/start_conversation', { method: 'POST' });
      if (!response.ok) throw new Error('Failed to start a new conversation.');
      const data = await response.json();
      setConversationId(data.conversation_id);
      sessionStorage.setItem('conversation_id', data.conversation_id);
      setMessages([]);
      notyf.success('Started a new conversation.');
      setTokenUsage(0);
      listConversations();
    } catch (error: any) {
      notyf.error(error.message || 'Error starting conversation.');
    }
  };

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const newMessage: Message = { role: 'user', content: userMessage };
    setMessages((prev) => [...prev, newMessage]);

    try {
      const response = await fetch('/api/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: userMessage
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to send message.');
      }

      const data = await response.json();
      // The assistant's response will come via Pusher
    } catch (error: any) {
      notyf.error(error.message || 'Failed to send message.');
    }

    setUserMessage('');
  };

  const resetConversation = async () => {
    if (!conversationId) {
      notyf.error('No active conversation to reset.');
      return;
    }

    try {
      const response = await fetch('/api/reset_conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId })
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to reset conversation.');
      }
      const data = await response.json();
      setMessages([]);
      notyf.success(data.message);
      setTokenUsage(0);
    } catch (error: any) {
      notyf.error(error.message || 'Error resetting conversation.');
    }
  };

  const saveConversation = async () => {
    // Since conversations are automatically saved in the database, this function could notify the user
    notyf.success('Conversation is automatically saved.');
  };

  const listConversations = async () => {
    try {
      const response = await fetch('/api/list_conversations');
      if (!response.ok) throw new Error('Failed to list conversations.');
      const data = await response.json();
      renderConversations(data.conversations);
    } catch (error: any) {
      notyf.error(error.message || 'Error listing conversations.');
    }
  };

  const renderConversations = (conversations: any[]) => {
    // Implement as needed, e.g., display in a modal or sidebar
    // This example assumes you have a separate component or state for conversations
  };

  const loadConversation = async (conversationId: string) => {
    try {
      const response = await fetch(`/api/load_conversation/${encodeURIComponent(conversationId)}`);
      if (!response.ok) throw new Error('Failed to load conversation.');
      const data = await response.json();
      setConversationId(conversationId);
      sessionStorage.setItem('conversation_id', conversationId);
      setMessages(data.conversation || []);
      notyf.success('Conversation loaded.');
    } catch (error: any) {
      notyf.error(error.message || 'Error loading conversation.');
    }
  };

  const searchConversations = async (query: string) => {
    try {
      const response = await fetch(`/api/search_conversations?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Search failed.');
      const data = await response.json();
      renderConversations(data.conversations);
    } catch (error: any) {
      notyf.error(error.message || 'Error searching conversations.');
    }
  };

  const addFewShotExample = async (userPrompt: string, assistantResponse: string) => {
    if (!userPrompt.trim() || !assistantResponse.trim()) {
      notyf.error('Both user prompt and assistant response are required.');
      return;
    }

    try {
      const response = await fetch('/api/add_few_shot_example', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_prompt: userPrompt, assistant_response: assistantResponse })
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to add few-shot example.');
      }
      const data = await response.json();
      notyf.success(data.message);
      // Optionally, update the conversation history
    } catch (error: any) {
      notyf.error(error.message || 'Error adding few-shot example.');
    }
  };

  const uploadFile = async (file: File) => {
    const allowedExtensions = ['txt', 'json', 'md'];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();

    if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
      notyf.error('Invalid file type. Allowed types: txt, md, json.');
      return;
    }

    const fileSizeInMB = file.size / (1024 * 1024);
    if (fileSizeInMB > 5) {
      notyf.error('File too large. Maximum allowed size is 5MB.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload_file', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Error uploading file.');
      notyf.success(data.message);
      setMessages((prev) => [...prev, { role: 'assistant', content: data.analysis }]);
    } catch (error: any) {
      notyf.error(error.message || 'Error uploading file. Please try again.');
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Web Chat Application</h2>
        <div className="chat-actions">
          <button onClick={startNewConversation}>New Conversation</button>
          <button onClick={resetConversation}>Reset</button>
          <button onClick={saveConversation}>Save</button>
          <button onClick={listConversations}>List</button>
        </div>
      </div>
      <div className="chat-history" ref={chatHistoryRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="token-usage">
        <progress value={tokenUsage} max="128000"></progress>
        <span>Token Usage: {tokenUsage} / 128000</span>
      </div>
      <form className="message-form" onSubmit={(e) => { e.preventDefault(); sendMessage(); }}>
        <input
          type="text"
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
      {/* Additional Forms for Few-Shot and File Upload can be implemented as separate components */}
    </div>
  );
};

export default Chat;
```

#### 4.2.2. Styling the Chat Interface

Create a corresponding CSS file `Chat.css` within `src/components/Chat.css`:

```css
/* apps/frontend/src/components/Chat.css */

.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: auto;
  padding: 1rem;
  background-color: [[f5f5f5]];
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-actions button {
  margin-left: 0.5rem;
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  margin: 1rem 0;
  padding: 1rem;
  background-color: [[ffffff]];
  border: 1px solid [[ddd]];
  border-radius: 5px;
}

.message {
  margin-bottom: 1rem;
  padding: 0.5rem 1rem;
  border-radius: 15px;
  max-width: 70%;
}

.message.user {
  align-self: flex-end;
  background-color: [[dcf8c6]];
}

.message.assistant {
  align-self: flex-start;
  background-color: [[f1f0f0]];
}

.token-usage {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
}

.token-usage progress {
  width: 100%;
  margin-right: 0.5rem;
}

.message-form {
  display: flex;
}

.message-form input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid [[ccc]];
  border-radius: 5px;
}

.message-form button {
  margin-left: 0.5rem;
  padding: 0.5rem 1rem;
}
```

#### 4.2.3. Integrating the Chat Component

Modify `App.tsx` to include the `Chat` component:

```tsx
// apps/frontend/src/App.tsx

import React from 'react';
import Chat from './components/Chat';

const App: React.FC = () => {
  return (
    <div className="App">
      <Chat />
    </div>
  );
};

export default App;
```

### 4.3. Integrating Real-Time Communication with Pusher

To eliminate WebSockets and align with Vercel's serverless architecture, we'll use **Pusher** for real-time communication.

#### 4.3.1. Setting Up Pusher

1. **Sign Up and Create an App:**

   - [Sign Up for Pusher](https://pusher.com/signup)
   - Create a new Channels app.
   - Note down the **App ID**, **Key**, **Secret**, and **Cluster**.

2. **Install Pusher SDKs:**

   ```bash
   cd apps/backend
   npm install pusher
   # or
   yarn add pusher
   ```

3. **Configure Pusher in Backend:**

   We'll set up Pusher in the Next.js API routes to emit events when the assistant responds.

---

## 5. Migrating the Backend (`app.py` to Next.js Serverless Functions)

Transitioning from **Flask** to **Next.js API Routes** involves converting your Flask routes into serverless functions compatible with Vercel. We'll also integrate the functionalities from `utils.py` into the new backend structure.

### 5.1. Setting Up Next.js in the Monorepo

1. **Navigate to the Backend Directory:**

   ```bash
   cd web-chat/apps/backend
   ```

2. **Initialize a Next.js App with TypeScript:**

   ```bash
   npx create-next-app@latest . --typescript
   # or
   yarn create next-app . --typescript
   ```

3. **Install Required Dependencies:**

   ```bash
   npm install pusher jsonwebtoken bcrypt mongodb dotenv
   # or
   yarn add pusher jsonwebtoken bcrypt mongodb dotenv
   ```

4. **Configure TypeScript (if necessary):**

   Ensure that TypeScript is properly configured in `tsconfig.json`.

### 5.2. Converting Flask Routes to Next.js API Routes

We'll convert each Flask route into a corresponding Next.js API route.

#### 5.2.1. `/start_conversation` API Route

**Original Flask Route (`app.py`):**

```python
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    # Logic to start a new conversation
    return jsonify({"message": "New conversation started.", "conversation_id": str(conversation_id)}), 200
```

**Next.js API Route (`pages/api/start_conversation.ts`):**

```typescript
// apps/backend/pages/api/start_conversation.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { MongoClient } from 'mongodb';
import { v4 as uuidv4 } from 'uuid';
import Pusher from 'pusher';

// Initialize Pusher
const pusher = new Pusher({
  appId: process.env.PUSHER_APP_ID || '',
  key: process.env.PUSHER_KEY || '',
  secret: process.env.PUSHER_SECRET || '',
  cluster: process.env.PUSHER_CLUSTER || '',
  useTLS: true
});

// Initialize MongoDB client
const client = new MongoClient(process.env.MONGODB_URI || '');

type Data = {
  message: string;
  conversation_id?: string;
  error?: string;
};

export default async function handler(req: NextApiRequest, res: NextApiResponse<Data>) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).json({ message: `Method ${req.method} Not Allowed` });
  }

  try {
    await client.connect();
    const database = client.db('chatbot_db');
    const conversations = database.collection('conversations');

    const conversationId = uuidv4();
    const userId = 'anonymous'; // Replace with actual user identification if available

    const newConversation = {
      conversation_id: conversationId,
      user_id: userId,
      conversation_history: [],
      conversation_text: '',
      created_at: new Date(),
      updated_at: new Date()
    };

    await conversations.insertOne(newConversation);

    res.status(200).json({ message: 'New conversation started.', conversation_id: conversationId });
  } catch (error: any) {
    console.error('Error starting conversation:', error);
    res.status(500).json({ message: 'Failed to start a new conversation.', error: error.message });
  } finally {
    await client.close();
  }
}
```

#### 5.2.2. `/send_message` API Route

**Original Flask Route (`app.py`):**

```python
@app.route('/send_message', methods=['POST'])
def send_message():
    # Logic to handle sending messages
    return jsonify({"assistant_response": assistant_response}), 200
```

**Next.js API Route (`pages/api/send_message.ts`):**

```typescript
// apps/backend/pages/api/send_message.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { MongoClient } from 'mongodb';
import axios from 'axios';
import Pusher from 'pusher';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Data {
  assistant_response?: string;
  message?: string;
  error?: string;
}

const client = new MongoClient(process.env.MONGODB_URI || '');

const pusher = new Pusher({
  appId: process.env.PUSHER_APP_ID || '',
  key: process.env.PUSHER_KEY || '',
  secret: process.env.PUSHER_SECRET || '',
  cluster: process.env.PUSHER_CLUSTER || '',
  useTLS: true
});

export default async function handler(req: NextApiRequest, res: NextApiResponse<Data>) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).json({ message: `Method ${req.method} Not Allowed` });
  }

  const { conversation_id, message } = req.body;

  if (!conversation_id || !message) {
    return res.status(400).json({ message: 'conversation_id and message are required.' });
  }

  try {
    await client.connect();
    const database = client.db('chatbot_db');
    const conversations = database.collection('conversations');

    const conversation = await conversations.findOne({ conversation_id });

    if (!conversation) {
      return res.status(404).json({ message: 'Conversation not found.' });
    }

    // Append the user's message
    const updatedHistory: Message[] = [...conversation.conversation_history, { role: 'user', content: message }];

    // Update the conversation in MongoDB
    await conversations.updateOne(
      { conversation_id },
      {
        $set: {
          conversation_history: updatedHistory,
          conversation_text: generateConversationText(updatedHistory),
          updated_at: new Date()
        }
      }
    );

    // Interact with Azure OpenAI API to generate a response
    const assistantResponse = await getAssistantResponse(updatedHistory);

    // Append the assistant's response
    const finalHistory: Message[] = [...updatedHistory, { role: 'assistant', content: assistantResponse }];

    // Update the conversation with the assistant's response
    await conversations.updateOne(
      { conversation_id },
      {
        $set: {
          conversation_history: finalHistory,
          conversation_text: generateConversationText(finalHistory),
          updated_at: new Date()
        }
      }
    );

    // Trigger Pusher event to notify the frontend
    await pusher.trigger(conversation_id, 'new-message', {
      role: 'assistant',
      content: assistantResponse
    });

    res.status(200).json({ assistant_response: assistantResponse });
  } catch (error: any) {
    console.error('Error sending message:', error);
    res.status(500).json({ message: 'Failed to send message.', error: error.message });
  } finally {
    await client.close();
  }
}

function generateConversationText(conversationHistory: Message[]): string {
  return conversationHistory.map(entry => `${entry.role}: ${entry.content}`).join('\n');
}

async function getAssistantResponse(conversationHistory: Message[]): Promise<string> {
  const apiUrl = process.env.AZURE_API_URL || '';
  const apiKey = process.env.API_KEY || '';

  const payload = {
    messages: conversationHistory,
    max_tokens: parseInt(process.env.REPLY_TOKENS || '800'),
    temperature: 0.7,
    top_p: 0.95
  };

  try {
    const response = await axios.post(apiUrl, payload, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      }
    });

    const assistantMessage = response.data.choices[0].message.content.trim();
    return assistantMessage;
  } catch (error: any) {
    console.error('Error communicating with Azure OpenAI:', error);
    return 'Sorry, I am unable to process your request at the moment.';
  }
}
```


---


## 5.3. Incorporating `utils.py` Functionality

Transitioning from a **Flask** backend to **Next.js** serverless functions requires adapting your existing utility functions (`utils.py`) to fit into the JavaScript/TypeScript ecosystem. This section will guide you through migrating each function from `utils.py` to TypeScript modules within your Next.js backend, ensuring seamless integration and functionality.

### **Overview**

Your `utils.py` contains several utility functions essential for:

- Token management and counting
- Conversation text generation
- Summarization of messages
- File handling and validation
- Interacting with the Azure OpenAI API

We'll recreate these functionalities in TypeScript, making necessary adjustments to accommodate differences between Python and JavaScript environments.

### **Prerequisites**

1. **TypeScript Setup:** Ensure your Next.js backend is configured to use TypeScript. If you initialized your Next.js app with the TypeScript template, this should already be set up.

2. **Dependencies Installation:** Install necessary packages that correspond to the Python libraries used in `utils.py`.

   ```bash
   cd apps/backend
   npm install axios dotenv uuid
   # or
   yarn add axios dotenv uuid
   ```

3. **Tokenizer Replacement:** Since `tiktoken` is a Python-specific library, we'll use a JavaScript equivalent for token counting. [OpenAI's tokenizer](https://github.com/openai/tiktoken) doesn't have an official JavaScript version, but [gpt-3-encoder](https://github.com/latitudegames/GPT-3-Encoder) is a popular alternative.

   ```bash
   npm install gpt-3-encoder
   # or
   yarn add gpt-3-encoder
   ```

4. **Environment Variables:** Ensure all necessary environment variables are set in your `.env.local` file within the backend package.

   ```ini
   // apps/backend/.env.local

   AZURE_API_URL=https://your-azure-openai-endpoint
   API_KEY=your_azure_api_key
   MAX_TOKENS=128000
   REPLY_TOKENS=800
   CHUNK_SIZE_TOKENS=1000
   MAX_FILE_SIZE_MB=5.0
   ALLOWED_EXTENSIONS=txt,md,json
   ```

### **Step-by-Step Migration**

#### **1. Creating the Utilities Module**

Create a new directory named `utils` within your backend package to house all utility functions.

```bash
cd apps/backend
mkdir utils
touch utils/index.ts
```

#### **2. Implementing Token Counting**

**Original Python Function:**

```python
def count_tokens(text):
    """Count tokens in the text using the tokenizer."""
    return len(encoding.encode(text))
```

**TypeScript Implementation:**

We'll use `gpt-3-encoder` to replicate `tiktoken`'s functionality.

```typescript
// apps/backend/utils/tokenizer.ts

import { encode } from 'gpt-3-encoder';

/**
 * Counts the number of tokens in a given text.
 * @param text - The input text to tokenize.
 * @returns The number of tokens.
 */
export function countTokens(text: string): number {
  return encode(text).length;
}
```

#### **3. Generating Conversation Text**

**Original Python Function:**

```python
def generate_conversation_text(conversation_history):
    """
    Generates a text summary of the conversation by concatenating user and assistant messages.
    """
    conversation_text = ""
    
    for message in conversation_history:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'user':
            conversation_text += f"User: {content}\n"
        elif role == 'assistant':
            conversation_text += f"Assistant: {content}\n"
    
    return conversation_text.strip()  # Remove any trailing newlines
```

**TypeScript Implementation:**

```typescript
// apps/backend/utils/conversation.ts

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Concatenates conversation history into a single string.
 * @param conversationHistory - Array of message objects.
 * @returns Concatenated conversation text.
 */
export function generateConversationText(conversationHistory: Message[]): string {
  return conversationHistory
    .map(msg => `${capitalize(msg.role)}: ${msg.content}`)
    .join('\n')
    .trim();
}

function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}
```

#### **4. Summarizing Messages**

**Original Python Function:**

```python
def summarize_messages(messages, max_summary_tokens=500):
    """Summarizes a list of messages into a shorter text."""
    combined_text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        combined_text += f"{role.capitalize()}: {content}\n"

    prompt = f"Please provide a concise summary of the following conversation:\n{combined_text}\nSummary:"

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_summary_tokens,
        "temperature": 0.5
    }

    try:
        response = requests.post(AZURE_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        summary_response = response.json()
        summary_content = summary_response['choices'][0]['message']['content'].strip()
        return {"role": "system", "content": f"Summary: {summary_content}"}
    except Exception as e:
        logging.error(f"Error during summarization: {str(e)}")
        return {"role": "system", "content": "Summary not available due to an error."}
```

**TypeScript Implementation:**

We'll use `axios` for HTTP requests and leverage async/await for asynchronous operations.

```typescript
// apps/backend/utils/summarizer.ts

import axios from 'axios';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface SummaryResponse {
  role: 'system';
  content: string;
}

const AZURE_API_URL = process.env.AZURE_API_URL || '';
const API_KEY = process.env.API_KEY || '';

/**
 * Summarizes a list of messages into a concise summary using Azure OpenAI API.
 * @param messages - Array of message objects to summarize.
 * @param maxSummaryTokens - Maximum tokens for the summary.
 * @returns A summary message object.
 */
export async function summarizeMessages(
  messages: Omit<Message, 'role'>[],
  maxSummaryTokens: number = 500
): Promise<SummaryResponse> {
  const combinedText = messages
    .map(msg => `${capitalize(msg.role)}: ${msg.content}`)
    .join('\n');

  const prompt = `Please provide a concise summary of the following conversation:\n${combinedText}\nSummary:`;

  const payload = {
    messages: [
      { role: 'system', content: 'You are a helpful assistant that summarizes conversations.' },
      { role: 'user', content: prompt }
    ],
    max_tokens: maxSummaryTokens,
    temperature: 0.5
  };

  try {
    const response = await axios.post(AZURE_API_URL, payload, {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_KEY}`
      }
    });

    const summaryContent: string = response.data.choices[0].message.content.trim();
    return { role: 'system', content: `Summary: ${summaryContent}` };
  } catch (error: any) {
    console.error(`Error during summarization: ${error.message}`);
    return { role: 'system', content: 'Summary not available due to an error.' };
  }
}

function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}
```

#### **5. Managing Token Limits**

**Original Python Function:**

```python
def manage_token_limits(conversation_history, new_message=None):
    """Manages the token limits by summarizing older messages when necessary."""
    if new_message:
        temp_history = conversation_history + [{"role": "user", "content": new_message}]
    else:
        temp_history = conversation_history.copy()

    total_tokens = sum(count_tokens(turn['content']) for turn in temp_history)

    if total_tokens >= MAX_TOKENS - REPLY_TOKENS:
        messages_to_summarize = []
        while total_tokens >= MAX_TOKENS - REPLY_TOKENS and len(temp_history) > 1:
            messages_to_summarize.append(temp_history.pop(0))
            total_tokens = sum(count_tokens(turn['content']) for turn in temp_history)

        if messages_to_summarize:
            summary_message = summarize_messages(messages_to_summarize)
            temp_history.insert(0, summary_message)
            total_tokens = sum(count_tokens(turn['content']) for turn in temp_history)

            if total_tokens >= MAX_TOKENS - REPLY_TOKENS:
                return manage_token_limits(temp_history)

    else:
        temp_history = conversation_history.copy()

    if new_message:
        temp_history.append({"role": "user", "content": new_message})
        total_tokens += count_tokens(new_message)

    return temp_history, total_tokens
```

**TypeScript Implementation:**

We'll need to adjust this function to handle asynchronous summarization since `summarizeMessages` is asynchronous.

```typescript
// apps/backend/utils/tokenManager.ts

import { countTokens } from './tokenizer';
import { summarizeMessages } from './summarizer';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || '128000');
const REPLY_TOKENS = parseInt(process.env.REPLY_TOKENS || '800');

/**
 * Manages token limits by summarizing older messages when necessary.
 * @param conversationHistory - Current conversation history.
 * @param newMessage - New message to be added (optional).
 * @returns Updated conversation history and total tokens.
 */
export async function manageTokenLimits(
  conversationHistory: Message[],
  newMessage?: string
): Promise<{ updatedHistory: Message[]; totalTokens: number }> {
  let tempHistory: Message[] = newMessage
    ? [...conversationHistory, { role: 'user', content: newMessage }]
    : [...conversationHistory];

  let totalTokens = tempHistory.reduce((acc, msg) => acc + countTokens(msg.content), 0);

  while (totalTokens >= MAX_TOKENS - REPLY_TOKENS && tempHistory.length > 1) {
    // Remove the oldest message
    const messagesToSummarize = tempHistory.splice(0, 1); // Adjust chunk size as needed

    // Summarize the removed messages
    const summary = await summarizeMessages(messagesToSummarize.map(msg => ({
      role: msg.role,
      content: msg.content
    })));

    // Insert the summary at the beginning
    tempHistory.unshift(summary);

    // Recalculate total tokens
    totalTokens = tempHistory.reduce((acc, msg) => acc + countTokens(msg.content), 0);
  }

  return { updatedHistory: tempHistory, totalTokens };
}
```

**Notes:**

- **Asynchronous Operations:** Since summarization involves API calls, the function is `async` and returns a `Promise`.
- **Chunk Size:** In the original Python code, it removes messages one by one until the token limit is satisfied. You can adjust the chunk size based on your summarization needs.

#### **6. File Handling and Validation**

**Original Python Functions:**

```python
def allowed_file(filename):
    """Checks if a given file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_size_under_limit(file):
    """Ensures that the uploaded file size is within the allowed limit."""
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()
    file_size_mb = size_bytes / (1024 * 1024)
    file.seek(0)
    return file_size_mb <= MAX_FILE_SIZE_MB
```

**TypeScript Implementation:**

In Next.js API routes, file uploads can be handled using libraries like `formidable` or `multer`. For simplicity, we'll assume you're using `formidable`.

1. **Install `formidable`:**

   ```bash
   npm install formidable
   # or
   yarn add formidable
   ```

2. **Implement File Validation:**

   ```typescript
   // apps/backend/utils/fileValidator.ts

   import { IncomingForm } from 'formidable';
   import { NextApiRequest } from 'next';

   const MAX_FILE_SIZE_MB = parseFloat(process.env.MAX_FILE_SIZE_MB || '5.0');
   const ALLOWED_EXTENSIONS = new Set(
     (process.env.ALLOWED_EXTENSIONS || 'txt,md,json').split(',').map(ext => ext.trim().toLowerCase())
   );

   interface ParsedFile {
     filepath: string;
     originalFilename: string;
     mimetype: string;
     size: number;
   }

   /**
    * Parses and validates the uploaded file.
    * @param req - Incoming Next.js API request.
    * @returns ParsedFile object if valid.
    * @throws Error if validation fails.
    */
   export async function parseAndValidateFile(req: NextApiRequest): Promise<ParsedFile> {
     const form = new IncomingForm();
     
     const { fields, files } = await new Promise<{ fields: any; files: any }>((resolve, reject) => {
       form.parse(req, (err, fields, files) => {
         if (err) reject(err);
         else resolve({ fields, files });
       });
     });

     const file = files.file as formidable.File;

     if (!file) {
       throw new Error('No file uploaded.');
     }

     const fileExtension = file.originalFilename?.split('.').pop()?.toLowerCase();

     if (!fileExtension || !ALLOWED_EXTENSIONS.has(fileExtension)) {
       throw new Error(`Invalid file type. Allowed types: ${Array.from(ALLOWED_EXTENSIONS).join(', ')}`);
     }

     const fileSizeMB = file.size / (1024 * 1024);
     if (fileSizeMB > MAX_FILE_SIZE_MB) {
       throw new Error(`File too large. Maximum allowed size is ${MAX_FILE_SIZE_MB}MB.`);
     }

     return {
       filepath: file.filepath,
       originalFilename: file.originalFilename || 'unknown',
       mimetype: file.mimetype || 'application/octet-stream',
       size: file.size
     };
   }
   ```

**Notes:**

- **File Parsing:** Uses `formidable` to parse multipart/form-data requests.
- **Validation:** Checks for allowed file extensions and size limits.
- **Error Handling:** Throws descriptive errors that can be caught and returned to the client.

#### **7. Interacting with Azure OpenAI API**

**Original Python Function:**

```python
def analyze_chunk_with_llama(chunk, retries=3):
    """Analyzes a text chunk using the Llama API, with error handling and retries."""
    conversation_history = session.get('conversation', [])
    conversation_history.append({"role": "user", "content": chunk})

    payload = {
        "messages": conversation_history,
        "max_tokens": 500,
        "temperature": 0.7
    }

    attempt = 0
    while attempt < retries:
        try:
            response = requests.post(AZURE_API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            llama_response = response.json()
            return llama_response['choices'][0]['message']['content']
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"API error: {e}")
            attempt += 1
            if attempt >= retries:
                return "Unable to process your request at this time. Please try again later."
```

**TypeScript Implementation:**

We'll adapt this function to use `axios` for HTTP requests and implement retry logic.

```typescript
// apps/backend/utils/openai.ts

import axios from 'axios';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const AZURE_API_URL = process.env.AZURE_API_URL || '';
const API_KEY = process.env.API_KEY || '';

/**
 * Analyzes a text chunk using the Azure OpenAI API with retry logic.
 * @param conversationHistory - Current conversation history.
 * @param chunk - The new message chunk to analyze.
 * @param retries - Number of retry attempts.
 * @returns Assistant's response or an error message.
 */
export async function analyzeChunkWithAzure(
  conversationHistory: Message[],
  chunk: string,
  retries: number = 3
): Promise<string> {
  const updatedHistory = [...conversationHistory, { role: 'user', content: chunk }];

  const payload = {
    messages: updatedHistory,
    max_tokens: 500,
    temperature: 0.7
  };

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.post(AZURE_API_URL, payload, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`
        }
      });

      const assistantContent: string = response.data.choices[0].message.content.trim();
      return assistantContent;
    } catch (error: any) {
      console.error(`API error on attempt ${attempt}:`, error.message);
      if (attempt === retries) {
        return 'Unable to process your request at this time. Please try again later.';
      }
      // Optional: Implement exponential backoff
      await new Promise(res => setTimeout(res, 1000 * attempt));
    }
  }

  return 'Unable to process your request at this time. Please try again later.';
}
```

**Notes:**

- **Retry Logic:** Attempts the API call up to `retries` times with incremental delays (exponential backoff can be implemented for better resilience).
- **Error Handling:** Logs errors and returns a user-friendly message upon failure.
- **Payload Structure:** Mirrors the structure expected by Azure OpenAI API.

#### **8. Integrating Utilities into API Routes**

Now that all utility functions are implemented, integrate them into your Next.js API routes.

**Example: `/api/send_message.ts`**

```typescript
// apps/backend/pages/api/send_message.ts

import { NextApiRequest, NextApiResponse } from 'next';
import { MongoClient } from 'mongodb';
import { generateConversationText } from '../../utils/conversation';
import { manageTokenLimits } from '../../utils/tokenManager';
import { analyzeChunkWithAzure } from '../../utils/openai';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const client = new MongoClient(process.env.MONGODB_URI || '');

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  const { conversation_id, message } = req.body;

  if (!conversation_id || !message) {
    return res.status(400).json({ message: 'conversation_id and message are required.' });
  }

  try {
    await client.connect();
    const database = client.db('chatbot_db');
    const conversations = database.collection('conversations');

    // Fetch the conversation
    const conversation = await conversations.findOne({ conversation_id });

    if (!conversation) {
      return res.status(404).json({ message: 'Conversation not found.' });
    }

    // Manage token limits
    const { updatedHistory, totalTokens } = await manageTokenLimits(
      conversation.conversation_history,
      message
    );

    // Update conversation with new message
    await conversations.updateOne(
      { conversation_id },
      {
        $set: {
          conversation_history: updatedHistory,
          conversation_text: generateConversationText(updatedHistory),
          updated_at: new Date(),
          total_tokens_used: totalTokens
        }
      }
    );

    // Generate assistant response
    const assistantResponse = await analyzeChunkWithAzure(updatedHistory, message);

    // Append assistant's response to history
    const finalHistory = [...updatedHistory, { role: 'assistant', content: assistantResponse }];

    // Update conversation with assistant's response
    await conversations.updateOne(
      { conversation_id },
      {
        $set: {
          conversation_history: finalHistory,
          conversation_text: generateConversationText(finalHistory),
          updated_at: new Date(),
          total_tokens_used: totalTokens + countTokens(assistantResponse) // Assuming countTokens is imported
        }
      }
    );

    res.status(200).json({ assistant_response: assistantResponse });
  } catch (error: any) {
    console.error('Error sending message:', error);
    res.status(500).json({ message: 'Failed to send message.', error: error.message });
  } finally {
    await client.close();
  }
}
```

**Notes:**

- **Token Management:** Utilizes `manageTokenLimits` to ensure the conversation stays within token limits.
- **Assistant Response:** Generates and appends the assistant's response using `analyzeChunkWithAzure`.
- **Error Handling:** Comprehensive error logging and user-friendly responses.

#### **9. Handling File Uploads**

**Original Python Route:**

```python
@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Handle file upload
    pass
```

**TypeScript Implementation:**

```typescript
// apps/backend/pages/api/upload_file.ts

import { NextApiRequest, NextApiResponse } from 'next';
import { parseAndValidateFile } from '../../utils/fileValidator';
import fs from 'fs';
import path from 'path';
import { handleFileChunks } from '../../utils/fileProcessor'; // To be implemented

export const config = {
  api: {
    bodyParser: false, // Disable Next.js's default body parser
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  try {
    const file = await parseAndValidateFile(req);

    // Read file content
    const fileContent = fs.readFileSync(file.filepath, 'utf-8');

    // Process file chunks and analyze
    const { contentChunks, fullAnalysis } = await handleFileChunks(fileContent);

    res.status(200).json({ message: 'File was uploaded and analyzed successfully.', analysis: fullAnalysis });
  } catch (error: any) {
    console.error('Error uploading file:', error);
    res.status(400).json({ message: error.message || 'Error uploading file.' });
  }
}
```

**Notes:**

- **Body Parser Disabled:** Next.js requires disabling the default body parser to handle file uploads with `formidable`.
- **File Processing:** Assumes the existence of a `handleFileChunks` function to process and analyze the file content.

#### **10. Finalizing Utilities Integration**

Ensure that all utility modules (`tokenizer.ts`, `conversation.ts`, `summarizer.ts`, `tokenManager.ts`, `fileValidator.ts`, `openai.ts`) are properly exported and imported in your API routes.

**Example: Consolidated Export**

```typescript
// apps/backend/utils/index.ts

export * from './tokenizer';
export * from './conversation';
export * from './summarizer';
export * from './tokenManager';
export * from './fileValidator';
export * from './openai';
// Add other utilities as needed
```

Now, in your API routes, you can import utilities as follows:

```typescript
import { countTokens, generateConversationText, summarizeMessages, manageTokenLimits, parseAndValidateFile, analyzeChunkWithAzure } from '../../utils';
```

### **Complete `utils.ts` Example**

For organizational purposes, you might want to consolidate all utilities into a single file or maintain them in separate modules as shown above. Maintaining separate modules enhances modularity and maintainability.

---

## Summary of Utility Functions Migration

| **Python (`utils.py`)** | **TypeScript (`utils/*.ts`)** | **Description** |
|-------------------------|-------------------------------|-----------------|
| `count_tokens(text)` | `countTokens(text: string): number` | Counts the number of tokens in a given text using `gpt-3-encoder`. |
| `generate_conversation_text(conversation_history)` | `generateConversationText(conversationHistory: Message[]): string` | Concatenates conversation history into a single formatted string. |
| `summarize_messages(messages, max_summary_tokens=500)` | `summarizeMessages(messages: Omit<Message, 'role'>[], maxSummaryTokens?: number): Promise<SummaryResponse>` | Summarizes a list of messages using Azure OpenAI API. |
| `manage_token_limits(conversation_history, new_message=None)` | `manageTokenLimits(conversationHistory: Message[], newMessage?: string): Promise<{ updatedHistory: Message[]; totalTokens: number }>` | Manages token limits by summarizing older messages when necessary. |
| `allowed_file(filename)` | Handled within `fileValidator.ts` | Validates uploaded file types and sizes. |
| `file_size_under_limit(file)` | Handled within `fileValidator.ts` | Ensures uploaded files are within size constraints. |
| `handle_file_chunks(file_content)` | `handleFileChunks(fileContent: string): Promise<{ contentChunks: string[]; fullAnalysis: string }>` | Breaks file content into chunks and analyzes them via Azure OpenAI API. |
| `analyze_chunk_with_llama(chunk, retries=3)` | `analyzeChunkWithAzure(conversationHistory: Message[], chunk: string, retries?: number): Promise<string>` | Analyzes a text chunk using Azure OpenAI API with retry logic. |

---

## Next Steps

With the utility functions successfully migrated, your Next.js backend is now equipped to handle core functionalities previously managed by Flask and `utils.py`. The following sections will guide you through implementing authentication, integrating real-time features using Pusher, and finalizing your backend setup.

Feel free to proceed to the next sections or ask for further clarifications on any of the steps outlined above!

---

Certainly! To ensure your custom deployment works smoothly on Vercel, especially considering the robust features of your application (such as a Turborepo monorepo structure, Next.js apps for frontend and backend, and integrations like Pusher), you'll need to configure your `turbo.json` and `package.json` files appropriately.

I'll provide you with sample contents for these files and explain any additional setup configurations that are necessary beyond the default Vercel deployment.

---

## **1. `turbo.json` Configuration**

The `turbo.json` file is the configuration file for Turborepo, which helps manage and build your monorepo efficiently. Here's how you can set it up:

**File:** `turbo.json`

```json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "outputs": []
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": []
    }
  }
}
```

**Explanation:**

- **`$schema`:** Points to the Turborepo schema for validation.
- **`pipeline`:** Defines tasks and their relationships.
  - **`build`:**
    - **`dependsOn`:** Indicates that the `build` task depends on the `build` task of its dependencies (denoted by `^build`).
    - **`outputs`:** Specifies the directories that are outputs of the build process, which helps Turborepo cache and optimize builds.
  - **`dev`:**
    - **`cache`:** Set to `false` because development servers typically shouldn't be cached.
    - **`persistent`:** Set to `true` to keep the development server running.
  - **`lint`:**
    - **`outputs`:** Empty, as linting doesn't produce output files.
  - **`test`:**
    - **`dependsOn`:** Depends on the `build` task.
    - **`outputs`:** Empty, as tests don't produce output files.

**Notes:**

- Adjust the `outputs` paths if your build outputs are different.
- You can add additional tasks as needed (e.g., `typecheck`, `format`).

---

## **2. Root `package.json` Configuration**

The root `package.json` file should manage workspaces and contain scripts to run tasks across your monorepo.

**File:** `package.json`

```json
{
  "name": "your-project-name",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "turbo run dev --parallel",
    "build": "turbo run build",
    "start": "turbo run start",
    "lint": "turbo run lint",
    "test": "turbo run test"
  },
  "workspaces": ["apps/*", "packages/*"],
  "devDependencies": {
    "turbo": "^1.10.12"
  },
  "engines": {
    "node": ">=14.17.0"
  }
}
```

**Explanation:**

- **`name` and `version`:** Metadata about your project.
- **`private`:** Set to `true` to prevent accidental publishing to npm.
- **`scripts`:** Defines commands to run tasks across all apps/packages.
  - **`dev`:** Runs the `dev` script in all workspaces in parallel.
  - **`build`:** Runs the `build` script in all workspaces, respecting dependencies.
  - **`start`:** Runs the `start` script in all workspaces.
  - **`lint`:** Runs linting across all workspaces.
  - **`test`:** Runs tests across all workspaces.
- **`workspaces`:** Defines the workspaces in your monorepo.
  - `"apps/*"`: Includes all apps in the `apps` directory.
  - `"packages/*"`: (Optional) If you have shared packages.
- **`devDependencies`:**
  - **`turbo`:** The Turborepo package.
- **`engines`:**
  - Specifies the Node.js version compatibility.

**Notes:**

- Ensure that `private` is set to `true` when using workspaces to prevent accidental publication.
- Adjust the Node.js version in `engines` according to your project's requirements.
- You may add other devDependencies as needed (e.g., ESLint, TypeScript).

---

## **3. `package.json` for `apps/frontend`**

Your frontend Next.js app will have its own `package.json` with dependencies and scripts specific to it.

**File:** `apps/frontend/package.json`

```json
{
  "name": "frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint",
    "test": "jest"
  },
  "dependencies": {
    "next": "13.5.2",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "pusher-js": "^7.0.3",
    "notyf": "^3.10.0"
    // Add other dependencies as needed
  },
  "devDependencies": {
    "@types/react": "^18.0.26",
    "@types/react-dom": "^18.0.10",
    "typescript": "^4.9.4",
    "eslint": "^8.26.0",
    "eslint-config-next": "13.5.2",
    "jest": "^29.2.1",
    "babel-jest": "^29.2.1",
    "@testing-library/react": "^13.4.0"
    // Add other dev dependencies as needed
  },
  "engines": {
    "node": ">=14.17.0"
  }
}
```

**Explanation:**

- **`name` and `version`:** Identifies this workspace.
- **`scripts`:**
  - **`dev`:** Starts the development server on port 3000.
  - **`build`:** Builds the Next.js app for production.
  - **`start`:** Starts the production server.
  - **`lint`:** Lints the codebase using Next.js's built-in ESLint configuration.
  - **`test`:** Runs tests using Jest.
- **`dependencies`:** Includes packages required at runtime.
  - **`next`, `react`, `react-dom`:** Core Next.js dependencies.
  - **`pusher-js`:** For real-time communication.
  - **`notyf`:** For notifications.
  - **Add other dependencies as needed for your app.**
- **`devDependencies`:** Includes packages required during development.
  - **TypeScript and types:** If you're using TypeScript.
  - **ESLint and configurations:** For linting.
  - **Jest and testing libraries:** For testing.
- **`engines`:** Specifies Node.js version compatibility.

**Notes:**

- Adjust versions to match your project's requirements.
- Add any additional dependencies your frontend app needs.

---

## **4. `package.json` for `apps/backend`**

Your backend Next.js app (API routes only) will have its own `package.json`.

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
    "next": "13.5.2",
    "axios": "^0.27.2",
    "mongodb": "^4.12.1",
    "pusher": "^5.0.3",
    "jsonwebtoken": "^9.0.0",
    "dotenv": "^16.0.3",
    "gpt-3-encoder": "^1.1.4"
    // Add other dependencies as needed
  },
  "devDependencies": {
    "@types/node": "^18.11.9",
    "typescript": "^4.9.4",
    "eslint": "^8.26.0",
    "eslint-config-next": "13.5.2",
    "jest": "^29.2.1",
    "ts-jest": "^29.0.3",
    "@types/jest": "^29.2.3"
    // Add other dev dependencies as needed
  },
  "engines": {
    "node": ">=14.17.0"
  }
}
```

**Explanation:**

- **`name` and `version`:** Identifies this workspace.
- **`scripts`:**
  - **`dev`:** Starts the development server on port 4000.
  - **`build`:** Builds the Next.js app for production.
  - **`start`:** Starts the production server.
  - **`lint`:** Lints the codebase.
  - **`test`:** Runs tests.
- **`dependencies`:**
  - **`next`:** Even though you're using Next.js for API routes, you still need `next`.
  - **`axios`:** For HTTP requests.
  - **`mongodb`:** For MongoDB interactions.
  - **`pusher`:** For real-time communication on the server.
  - **`jsonwebtoken`:** For handling JWT authentication.
  - **`dotenv`:** For environment variable management (ensure not to use in production on Vercel).
  - **`gpt-3-encoder`:** For token counting.
  - **Add other dependencies as needed.**
- **`devDependencies`:** Includes development tools like TypeScript, ESLint, Jest.
- **`engines`:** Specifies Node.js version compatibility.

**Notes:**

- On Vercel, environment variables are managed via the dashboard, so `dotenv` is only used locally.
- Adjust versions and dependencies as needed.
- Ensure that sensitive dependencies are only included where necessary.

---

## **5. Additional Setup Configuration Steps**

### **5.1. API Routes Adjustments**

Since you're deploying the frontend and backend as separate applications, you'll need to adjust API calls in your frontend code to point to the backend's base URL.

**Example:**

- **Create a Configuration File in Frontend:**

  **File:** `apps/frontend/config.js`

  ```javascript
  export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://your-backend-app.vercel.app';
  ```

- **Use the Base URL in API Calls:**

  ```javascript
  // In your fetch functions
  const response = await fetch(`${API_BASE_URL}/api/your-endpoint`, {
    method: 'POST',
    // ... other options
  });
  ```

- **Set the Environment Variable in Vercel:**

  - **Key:** `NEXT_PUBLIC_API_BASE_URL`
  - **Value:** The URL of your backend application, e.g., `https://your-backend-app.vercel.app`

### **5.2. CORS Configuration**

Since your frontend and backend are on different origins, you need to handle Cross-Origin Resource Sharing (CORS) on your backend API routes.

- **Install `nextjs-cors` in Backend:**

  ```bash
  cd apps/backend
  npm install nextjs-cors
  ```

- **Use CORS Middleware in API Routes:**

  ```typescript
  import NextCors from 'nextjs-cors';

  export default async function handler(req, res) {
    // Run the cors middleware
    await NextCors(req, res, {
      // Options
      origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      optionsSuccessStatus: 200,
    });

    // Your API logic here
  }
  ```

- **Set `ALLOWED_ORIGINS` Environment Variable:**

  - In Vercel's backend project settings, set `ALLOWED_ORIGINS` to the frontend URL, e.g., `https://your-frontend-app.vercel.app`

### **5.3. Pusher Configuration**

Ensure Pusher is correctly configured on both the frontend and backend.

- **Frontend Pusher Initialization:**

  ```tsx
  import Pusher from 'pusher-js';

  const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
    cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
    // Other options
  });
  ```

- **Backend Pusher Initialization:**

  ```typescript
  import Pusher from 'pusher';

  const pusher = new Pusher({
    appId: process.env.PUSHER_APP_ID!,
    key: process.env.PUSHER_KEY!,
    secret: process.env.PUSHER_SECRET!,
    cluster: process.env.PUSHER_CLUSTER!,
    useTLS: true,
  });
  ```

### **5.4. Environment Variables Management**

Ensure all necessary environment variables are set up correctly in Vercel for both projects.

- **Frontend Project (`your-frontend-app`):**

  - **NEXT_PUBLIC_API_BASE_URL**
  - **NEXT_PUBLIC_PUSHER_KEY**
  - **NEXT_PUBLIC_PUSHER_CLUSTER**

- **Backend Project (`your-backend-app`):**

  - **AZURE_API_URL**
  - **API_KEY**
  - **MONGODB_URI**
  - **JWT_SECRET**
  - **PUSHER_APP_ID**
  - **PUSHER_KEY**
  - **PUSHER_SECRET**
  - **PUSHER_CLUSTER**
  - **ALLOWED_ORIGINS**

### **5.5. Update Build Commands in Vercel**

Vercel may need custom build commands for your projects.

- **Frontend Project:**

  - **Build Command:** `npm run build`
  - **Install Command:** `npm install`
  - **Output Directory:** `.next`

- **Backend Project:**

  - **Build Command:** `npm run build`
  - **Install Command:** `npm install`
  - **Output Directory:** `.next`

Ensure that your `package.json` scripts align with these commands.

### **5.6. TypeScript Configuration**

If you're using TypeScript, ensure that your `tsconfig.json` files are correctly set up in each app.

**Example `tsconfig.json`:**

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
    "jsx": "preserve",
    "incremental": true
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

Adjust as necessary for your project.

---

## **6. Additional Notes**

### **6.1. Monorepo with Vercel**

Vercel supports monorepos by allowing you to define the root directory for each project.

- When setting up each project on Vercel, specify the correct root directory (`apps/frontend` or `apps/backend`).

### **6.2. Shared Packages (Optional)**

If you have shared code (e.g., utilities, components) in a `packages/` directory, ensure that your build process accounts for them.

- Use `npm` or `yarn` workspaces to manage shared packages.
- Configure `module.exports` and `module.aliases` if necessary.

### **6.3. Testing**

Ensure you have proper testing configurations.

- **Jest Configuration (`jest.config.js`):**

  ```javascript
  module.exports = {
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
    moduleNameMapper: {
      '^@/(.*)$': '<rootDir>/$1',
    },
  };
  ```

- **ESLint Configuration (`.eslintrc.json`):**

  ```json
  {
    "extends": ["next/core-web-vitals", "plugin:@typescript-eslint/recommended"],
    "rules": {
      // Your custom rules
    }
  }
  ```

### **6.4. Logging and Monitoring**

Consider integrating logging and monitoring tools.

- Use services like Sentry for error tracking.
- Implement logging in your backend to track API requests and errors.

### **6.5. Performance Optimization**

- Use Next.js features like image optimization, code splitting, and server-side rendering where appropriate.
- Analyze your build times and optimize Turborepo configurations if builds are slow.

---

## **Conclusion**

By carefully setting up your `turbo.json` and `package.json` files as outlined above, and by addressing the additional configuration steps, you'll ensure that your robust application is correctly configured for deployment on Vercel.

Remember to:

- Keep sensitive information secure and never commit it to version control.
- Test your application thoroughly in both development and production environments.
- Monitor logs and performance after deployment to catch and address any issues promptly.

If you have any questions or need further assistance with specific configurations, feel free to ask!