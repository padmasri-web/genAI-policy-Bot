# Autonomous MirAI Student Policy Advisor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C.svg)](https://www.langchain.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com)

A production-grade Retrieval-Augmented Generation (RAG) system built to automate inquiries regarding university attendance, grading, medical leave procedures, and club operations for **Mirai School of Technology (MSOT)**. 

The system operates under strict zero-hallucination guardrails: all responses are strictly confined to the facts provided in the official `Mirai_SoT_Policy_Handbook_2026.pdf`.

---

## 🏗️ System Architecture

```
                  +-----------------------------------+
                  | Mirai_SoT_Policy_Handbook_2026.pdf |
                  +-----------------+-----------------+
                                    |
                            [PyPDFLoader]
                                    |
                   [RecursiveCharacterTextSplitter]
                    (chunk_size=1000, overlap=200)
                                    |
                    [GoogleGenerativeAIEmbeddings]
                                    |
                           [(Local ChromaDB)]
                                    |
 [Student Query] ---> [MultiQueryRetriever] ---> [Retrieved Policy Chunks]
                            |                                |
                            +--------------+-----------------+
                                           |
                                  [LCEL RAG Pipeline]
                              (Strict Guardrail Prompt)
                                           |
                             [ChatGoogleGenerativeAI]
                             (gemini-3.8-flash, T=0.0)
                                           |
                       +-------------------+-------------------+
                       |                                       |
              [FastAPI Backend]                       [Streamlit UI]
           (POST /chat, /ingest)                   (Interactive Chat)
```

### Key Technical Components

1. **Data Pipeline (Ingestion & Vectorization)**:
   - **Document Loader**: Ingests `Mirai_SoT_Policy_Handbook_2026.pdf` using `PyPDFLoader`.
   - **Semantic Chunking**: Splits policy text with `RecursiveCharacterTextSplitter` configured at `chunk_size=1000` and `chunk_overlap=200` to preserve semantic continuity across clauses.
   - **Vector Store**: Embeds chunks using `GoogleGenerativeAIEmbeddings` and stores vectors in a persistent local `ChromaDB` instance (`./chroma_db`).

2. **Retrieval Architecture (LangChain & LCEL)**:
   - **Advanced Retrieval**: Overcomes the vocabulary gap between casual student language and formal handbook terminology using `MultiQueryRetriever` to rewrite and expand user queries from multiple angles.
   - **Generator Configuration**: Utilizes `ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)` for deterministic, strictly factual responses, with model fallback resilience.
   - **Guardrails**: System prompt explicitly blocks base training data reliance. If information (such as specific monetary fines) is missing from the retrieved context, the system strictly declines to hallucinate numbers and states institutional protocol.

3. **Application Layer**:
   - **Backend (`backend.py`)**: Asynchronous FastAPI service exposing `POST /ingest` (multipart PDF) and `POST /chat` (JSON query).
   - **Frontend (`frontend.py`)**: Student-friendly Streamlit web application with error handling, quick test query triggers, document ingestion, and context evidence expanders.

---

## 🚀 Standard Operating Procedures (SOP)

### 1. Prerequisites & Environment Setup

Clone this repository and ensure Python (3.10+) is available:

```bash
cd RAGofStdPolicy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your-google-gemini-api-key"
GOOGLE_API_KEY="your-google-gemini-api-key"
GEMINI_MODEL="gemini-3.8-flash"
CHROMA_PERSIST_DIRECTORY="./chroma_db"
DEFAULT_PDF_PATH="Mirai_SoT_Policy_Handbook_2026.pdf"
```

---

### 2. Running the FastAPI Backend

Start the backend server using `uvicorn`:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

#### Available REST Endpoints:
- `GET /health`: Returns vector store connection status and chunk count.
- `POST /ingest`: Accepts a PDF file via `multipart/form-data`, chunks it, and updates ChromaDB.
  ```bash
  curl -X POST "http://localhost:8000/ingest" \
    -F "file=@Mirai_SoT_Policy_Handbook_2026.pdf"
  ```
- `POST /chat`: Accepts JSON query and returns factual response with sources.
  ```bash
  curl -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{"question": "I have 72% attendance. How many attendance marks will I get?"}'
  ```

---

### 3. Running the Streamlit Frontend

In a separate terminal (with the virtual environment activated), run:

```bash
streamlit run frontend.py --server.port 8501
```

Open your browser at `http://localhost:8501`.

**Features included:**
- 💬 Real-time chat with message history.
- 🎯 One-click quick test buttons for all 4 certification audit queries.
- 📄 Document upload & vectorization interface in sidebar.
- 📚 Expandable source citations and exact handbook page numbers.
- 🛡️ Resilient error handling for backend timeouts and disconnections.

---

### 4. Running the Automated Evaluation Suite

To run the LLM-as-a-judge benchmarking script and regenerate `rag_eval_scores.csv`:

```bash
python evaluate.py
```

---

## 📊 Certification Audit & Benchmarking Deliverable

The system has been audited against the 4 mandatory test queries. All queries scored **5/5 (Outstanding)** on the LLM-as-a-judge rubric:

| ID | Test Category | Query | Expected Outcome | System Outcome | Score |
|---|---|---|---|---|---|
| **1** | **Precision Verification** | *"I have 72% attendance. How many attendance marks will I get?"* | State student will receive 4 marks | Correctly identifies 60%-74.99% tier grants 4 marks, noting 75% debarment policy and 15% medical buffer. | **5/5** |
| **2** | **Multi-Hop Reasoning** | *"I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"* | Instruct to contact Yashaswini Ma'am within exactly 7 days of illness/treatment | Synthesizes Ratnam Campus Manager (Yashaswini Ma'am) and strict 7-day post-illness deadline. | **5/5** |
| **3** | **Process Verification** | *"We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"* | State 40% batch support required; submit to Faculty Coordinator first, not Management directly | Confirms Management must NOT be approached directly; 40% batch support and Faculty Coordinator review required. | **5/5** |
| **4** | **Negative Constraint Testing** | *"How much is the fine for smoking a cigarette on campus?"* | State tobacco is prohibited and leads to Disciplinary Committee; must not fabricate fine | States tobacco is prohibited with Disciplinary Committee actions; confirms no monetary fine is specified. | **5/5** |

Full benchmark logs and detailed reasoning are persisted in [`rag_eval_scores.csv`](rag_eval_scores.csv).

---

## 📁 Repository Structure

```
.
├── Mirai_SoT_Policy_Handbook_2026.pdf   # Official student policy handbook
├── backend.py                           # FastAPI REST server (/chat, /ingest)
├── frontend.py                          # Streamlit student chat interface
├── rag_pipeline.py                      # LangChain LCEL RAG pipeline & ChromaDB manager
├── evaluate.py                          # LLM-as-a-Judge benchmarking pipeline
├── rag_eval_scores.csv                  # Certification audit benchmark logs
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # System documentation & operating procedures
```
