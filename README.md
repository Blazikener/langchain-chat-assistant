# langchain-chat-assistant
This is an interactive conversational interface that enables users to chat with their own data (PDFs). Built using Python, Panel, and LangChain, the bot keeps track of conversation history, detects user mood, extracts discussed topics, and enhances responses using Retrieval-Augmented Generation (RAG) over documents like lecture notes.
## Features
-> Upload and interact with PDF documents

-> Memory aware conversational history

-> Mood detection using sentiment analysis

-> Topic tracking for context retention

-> Retrieval-Augmented Generation (RAG) using LangChain

-> Saves user context for continued sessions

## Installation:

Clone the repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Set up your environment:

Create a .env file in the root directory

Add your OpenAI key:

OPENAI_API_KEY=your_openai_key_here

## Workflow:

1. Login with email: Initializes mood/topic tracking

2. Optional: Upload PDF: Load a document to build a searchable vector DB

3. Ask questions: The bot answers using context-aware retrieval

4.Review tabs:

-> Conversation: Chat with the bot

-> Database: See how the bot queried your document

-> Chat History: Review all previous questions/answers

-> Configure: Load new files or reset history

Example Prompt Flow:

User: “I'm feeling overwhelmed with exams.”

System detects: mood = sad, topic = exams

Response is enhanced with mood + topic history.

System returns a short, contextual answer with “thanks for asking!” at the end.

## Screenshot:
Chatting in with the bot
![image](https://github.com/user-attachments/assets/91d6ca5b-70af-4c70-95b7-b9146fd065a1)

Database
![image](https://github.com/user-attachments/assets/ceb2e458-74d4-4ebe-b450-8112841c815b)

Chat History
![image](https://github.com/user-attachments/assets/f75ee629-d9f7-45da-b907-0eb9e3f5247c)

Configure
![image](https://github.com/user-attachments/assets/1d384ed0-52ce-49ba-9c69-e2d9ccd94fa0)
