# Deep Learning Research Assistant Agent 🤖

An autonomous RAG (Retrieval-Augmented Generation) agent built with LangGraph, Streamlit, and Groq. This assistant answers complex questions about neural network architectures, optimization, and training strategies using a local knowledge base and autonomous web search fallback.

## 🚀 Features

* **Autonomous RAG**: Searches a local ChromaDB vector store containing deep learning research topics.
* **Web Search Fallback**: Automatically searches the web via DuckDuckGo if the answer is not found in the local knowledge base.
* **Agentic Logic**: Uses LangGraph to manage state, memory, and autonomous routing.
* **Streamlit UI**: A clean, interactive chat interface with session persistence and source tracking.

## 🛠️ Setup Instructions

### 1. Prerequisites

* **Python 3.12** (Recommended)
* **Groq API Key**: Required for the LLM (Llama 3.3).
* **Hugging Face Token**: Required for the sentence-transformer embeddings.

### 2. Environment Setup

Open PowerShell and run the following commands to set up your isolated environment:

```powershell
# Create a Python 3.12 virtual environment
python -m venv ven

# Activate the virtual environment
.\ven\Scripts\activateDeep Learning Research Assistant Agent 🤖
An autonomous RAG (Retrieval-Augmented Generation) agent built with LangGraph, Streamlit, and Groq. This assistant answers complex questions about neural network architectures, optimization, and training strategies using a local knowledge base and autonomous web search fallback.

🚀 Features
Autonomous RAG: Searches a local ChromaDB vector store containing deep learning research topics.

Web Search Fallback: Automatically searches the web via DuckDuckGo if the answer is not found in the local knowledge base.

Agentic Logic: Uses LangGraph to manage state, memory, and autonomous routing.

Streamlit UI: A clean, interactive chat interface with session persistence and source tracking.

🛠️ Setup Instructions
1. Prerequisites
Python 3.12 (Recommended)

A Groq API Key (for the LLM)

A Hugging Face Token (for embeddings)

2. Environment Setup
Open PowerShell and run the following commands to set up your isolated environment:

PowerShell
# Create a Python 3.12 virtual environment


python -m venv ven

# Activate the virtual environment


.\ven\Scripts\activate
3. Installation


Install the required dependencies:

PowerShell

pip install -r requirements.txt


4. Configuration (.env setup)

5. 
You must create a file named .env in the root directory. This file is ignored by Git to keep your keys secret. Add your keys in the following format:

Plaintext
GROQ_API_KEY="your_groq_api_key_here"
HF_TOKEN="your_huggingface_token_here"
Note: You can get your Groq key from the Groq Console and your Hugging Face token from your Hugging Face Settings.

🏃 Running the Application
Once your .env is ready and the environment is activated, run:

PowerShell
streamlit run capstone_streamlit.py
📂 Project Structure
agent.py: The backend logic, including LangGraph nodes and RAG setup.

capstone_streamlit.py: The frontend interface.

requirements.txt: List of Python dependencies.

.gitignore: Configured to ignore .env and ven/ folders.

📝 License
This project was developed as a Capstone Project for deep learning research and agentic AI.
