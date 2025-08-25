# 🤖 USP Chatbot (With RAG - Retrieval Augmented Generation)

This is an advanced GPT-powered chatbot designed for USP Polymers LLP. It uses Retrieval-Augmented Generation (RAG) to answer domain-specific queries by fetching relevant context from uploaded PDF documents. The project uses Flask, OpenAI API (GPT-4o or GPT-4), and vector embeddings for accurate, document-aware responses.

---

## 📁 Project Structure

```
Chatbot-RAG/
├── app.py                  # Main Flask backend
├── rag_utils.py            # RAG logic: embedding, context retrieval
├── static/
│   └── style.css           # Optional frontend styling
├── templates/
│   └── chat.html           # Chatbot frontend interface
├── pdfs/                   # Folder to store user-uploaded PDFs
├── embeddings/             # Saved vector embeddings (e.g., FAISS index)
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🚀 Features

- ✅ **GPT-4o / GPT-4 API-based generation**
- ✅ **PDF Upload + Parsing + Embedding**
- ✅ **Document-based QA (contextual retrieval)**
- ✅ **Text, Image, and Voice input support**
- ✅ **TTS voice output (optional)**
- ✅ **Streamed responses with bullet formatting**
- ✅ **Short-term conversation memory**
- 🔒 **Environment-based API key management**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sWAGish/USP-Chatbot-with-RAG.git
cd USP-Chatbot-with-RAG
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file and add your OpenAI API Key:

```env
OPENAI_API_KEY=your-key-here
```

Ensure `.env` is added to `.gitignore`.

### 5. Run the App

```bash
python app.py
```

Visit `http://localhost:5000` to interact with the chatbot.

---

## 📦 Required API Keys

- **OpenAI API Key** (GPT-4 / GPT-4o)
- Optional: Google Speech APIs for voice

---

## 🔍 RAG Workflow Summary

1. Upload PDF file(s)
2. Split into chunks (using `PyMuPDF`, `langchain.text_splitter`)
3. Generate embeddings using OpenAI
4. Store in FAISS (or other) vector database
5. During user query:
   - Retrieve relevant chunks via semantic similarity
   - Inject into prompt context window
   - Generate response using GPT

---

## 🛡️ Security Notes

- NEVER commit `.env` or any API key to GitHub
- Delete `__pycache__/` and `.pyc` before pushing
- Use `git-filter-repo` if a secret is leaked in git history

---

## 🧠 Future Upgrades

- Multi-document support
- Supabase or Pinecone for persistent vector storage
- LLM model switching (e.g., Anthropic Claude, Gemini)
- Web page crawler integration
- Role-based query routing (Sales vs Technical)

---
