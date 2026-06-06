# 📄 RAG-Powered PDF Question Answering Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, FAISS, and Llama 3.3 70B, deployed on AWS EC2 with Docker.

---

## 🚀 Live Demo

- **Frontend:** https://rag-based-pdf-chatboat.streamlit.app/
- **Backend API:** `http://13.50.231.216:8000`
- **API Docs:** `http://13.50.231.216:8000/docs`

---

## 🏗️ Architecture

```
┌─────────────────────┐        ┌──────────────────────────────────────┐
│   Streamlit Cloud   │        │           AWS EC2 (Ubuntu)           │
│                     │        │                                      │
│   frontend.py       │──────▶ │   FastAPI (main.py)                  │
│                     │  HTTP  │        │                             │
│  - File Upload      │        │        ▼                             │
│  - Chat History     │        │   RAG Pipeline (rag.py)              │
│  - Session ID       │        │        │                             │
└─────────────────────┘        │        ├── PyPDFLoader               │
                               │        ├── RecursiveTextSplitter     │
                               │        ├── HuggingFace Embeddings    │
                               │        ├── FAISS Vector Store        │
                               │        ├── MMR Retriever             │
                               │        └── Llama 3.3 70B (Groq)      |
                               │                                      │
                               │   Docker Container                   │
                               │   AWS SSM (API Key Storage)          │
                               └──────────────────────────────────────┘
```

---

## ✨ Features

- Upload any PDF (up to 20MB) and ask questions in natural language
- **MMR (Maximal Marginal Relevance)** retrieval for diverse, relevant chunks
- **Session-based** architecture — multiple users can use it simultaneously
- **Chat history** — see all previous Q&A in the same session
- Rate limiting on all API endpoints (slowapi)
- Secrets managed via **AWS SSM Parameter Store**
- Dockerized backend deployed on **AWS EC2**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Cloud | AWS EC2 (Ubuntu) |
| Secrets | AWS SSM Parameter Store |
| Evaluation | RAGAS |

---

## 📁 Project Structure

```
RAG-Powered-PDF-Question-Answering-Chatbot/
├── rag.py              # RAG pipeline — PDF loading, chunking, retrieval, LLM chain
├── main.py             # FastAPI backend — upload and ask endpoints
├── frontend.py         # Streamlit frontend — UI and chat history
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies              
├── .gitignore
├── .dockerignore
├── testing.ipynb       # Development and experimentation notebook
└── eval/
    ├── evaluate.py         # RAGAS evaluation script (offline, not in production)
    └── ragas_results.csv   # Evaluation results
```

---

## ⚙️ RAG Pipeline

```
PDF Upload
    │
    ▼
PyPDFLoader  ──▶  RecursiveCharacterTextSplitter  ──▶  HuggingFace Embeddings
                  chunk_size=1500                       all-MiniLM-L6-v2
                  chunk_overlap=300
                        │
                        ▼
                  FAISS Vector Store
                        │
                        ▼
                  MMR Retriever
                  k=6, fetch_k=15, lambda_mult=0.7
                        │
                        ▼
                  Llama 3.3 70B (Groq)
                        │
                        ▼
                    Answer
```

---

## 🏃 Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/pratikbugade01/RAG-Powered-PDF-Question-Answering-Chatbot.git
cd RAG-Powered-PDF-Question-Answering-Chatbot
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

**5. Run the backend:**
```bash
uvicorn main:app --reload --port 8000
```

**6. Run the frontend (new terminal):**
```bash
streamlit run frontend.py
```

---

## 🐳 Run with Docker

```bash
docker pull pratikbugade/pdf_chatboat:latest
docker run -p 8000:8000 -e GROQ_API_KEY="your_key_here" pratikbugade/pdf_chatboat:latest
```

---

## ☁️ AWS Deployment

Backend is deployed on AWS EC2 using Docker. API key is stored securely in **AWS SSM Parameter Store**.

```bash
# fetch key from SSM and run container
export GROQ_API_KEY=$(aws ssm get-parameter \
    --name "/pdf-chatbot/GROQ_API_KEY" \
    --with-decryption \
    --query "Parameter.Value" \
    --output text)

docker run -d -p 80:8000 -e GROQ_API_KEY="$GROQ_API_KEY" pratikbugade/pdf_chatboat:latest
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/upload-pdf` | Upload and process a PDF |
| POST | `/ask` | Ask a question about the uploaded PDF |

**Upload PDF:**
```bash
curl -X POST http://13.50.231.216:8000/upload-pdf \
  -F "file=@your_document.pdf" \
  -F "session_id=your_session_id"
```

**Ask a question:**
```bash
curl -X POST http://13.50.231.216:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your_session_id", "question": "What is this document about?"}'
```

---

## 📊 Evaluation (RAGAS)

RAG pipeline evaluated using **RAGAS** on a LangChain research paper across 3 questions:

| Question | Faithfulness |
|---|---|
| How does LangChain handle data before retrieval? | 0.80 |
| What is the role of LangSmith in LangChain? | 1.00 |
| What are the security concerns with LangChain? | 1.00 |
| **Average** | **0.93** |

> Faithfulness measures whether the RAG answer is grounded in the retrieved context (no hallucinations).
> Evaluated using `llama-3.1-8b-instant` via Groq as judge LLM and `all-MiniLM-L6-v2` embeddings.
> Full results in `eval/ragas_results.csv`.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |

---

## 📦 Requirements

```
langchain
langchain-community
langchain-huggingface
langchain-groq
langchain-core
faiss-cpu
pypdf
cryptography>=3.1
fastapi
uvicorn
streamlit
python-dotenv
slowapi
sentence-transformers
```

---

## 🙋 Author

**Pratik Bugade**
- GitHub: [pratikbugade01](https://github.com/pratikbugade01)
- Docker Hub: [pratikbugade/pdf_chatboat](https://hub.docker.com/r/pratikbugade/pdf_chatboat)

---

## 📄 License

MIT License