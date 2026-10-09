import os
import requests
import streamlit as st
from typing import Dict, Any

# Configure Page
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Academic Interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        margin-top: 0.4rem;
        margin-bottom: 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-online {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    
    .status-offline {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
    }
    
    .source-card {
        background: #1e293b;
        border-left: 3px solid #6366f1;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    .quick-query-btn {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_backend_status() -> Dict[str, Any]:
    """Verify backend health with short timeout."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json()}
        return {"online": False, "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"online": False, "error": str(e)}


def query_rag_backend(question: str) -> Dict[str, Any]:
    """Execute question against FastAPI backend with comprehensive error handling."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question},
            timeout=45
        )
        if resp.status_code == 200:
            return {"success": True, "data": resp.json()}
        else:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            return {"success": False, "error": f"Backend Error (Status {resp.status_code}): {err_detail}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The server took too long to formulate a response. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Unable to reach the backend at {BACKEND_URL}. Ensure the FastAPI server is running (`uvicorn backend:app --port 8000`)."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected communication error: {str(e)}"}


def upload_document_to_backend(uploaded_file) -> Dict[str, Any]:
    """Send uploaded PDF to backend /ingest endpoint."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
        if resp.status_code == 200:
            return {"success": True, "data": resp.json()}
        else:
            try:
                err = resp.json().get("detail", resp.text)
            except Exception:
                err = resp.text
            return {"success": False, "error": f"Ingestion failed ({resp.status_code}): {err}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to upload document: {str(e)}"}


# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Advisor Controls")

    # Backend Status
    backend_status = check_backend_status()
    if backend_status["online"]:
        records = backend_status["data"].get("collection_records", 0)
        st.markdown(f"""
        <div class="status-badge status-online">
            ● Backend Connected ({records} Chunks)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-badge status-offline">
            ○ Backend Disconnected
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Target: `{BACKEND_URL}`")
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Document Ingestion Panel
    st.subheader("📄 Handbook Ingestion")
    st.caption("Upload an updated or supplementary policy document to refresh the vector store:")
    uploaded_pdf = st.file_uploader("Upload Policy PDF", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("⚡ Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Processing pages, generating embeddings, and updating ChromaDB..."):
                ingest_res = upload_document_to_backend(uploaded_pdf)
                if ingest_res["success"]:
                    st.success(f"Ingested {ingest_res['data']['chunks_created']} chunks from {ingest_res['data']['filename']}!")
                    st.rerun()
                else:
                    st.error(ingest_res["error"])

    st.markdown("---")

    # Certification Audit Test Queries
    st.subheader("🎯 Certification Audit Queries")
    st.caption("Click any standard audit test question to test accuracy:")

    test_queries = [
        ("Attendance Marks", "I have 72% attendance. How many attendance marks will I get?"),
        ("Ratnam Medical Leave", "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"),
        ("Cybersecurity Society", "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"),
        ("Smoking Fine Check", "How much is the fine for smoking a cigarette on campus?")
    ]

    selected_test_query = None
    for label, query_text in test_queries:
        if st.button(f"📌 {label}", use_container_width=True):
            selected_test_query = query_text

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main View Header
st.markdown("""
<div class="main-header">
    <h1>Autonomous MirAI Student Policy Advisor</h1>
    <p>Official Institutional Policy Assistant • MultiQuery Vector Retrieval • Zero-Hallucination Guardrails</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your autonomous MirAI Student Policy Advisor. You can ask me any question regarding attendance policies, grading schemes, medical & duty leave procedures, club formations, or student conduct rules. How can I assist you today?"
        }
    ]

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Source Citations & Context Evidence", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <b>Page {src.get('page', 'N/A')}</b> ({src.get('source', 'Handbook')})<br/>
                        <em>"{src.get('snippet', '')}..."</em>
                    </div>
                    """, unsafe_allow_html=True)

# Handle Query (from text input or quick audit buttons)
prompt_input = st.chat_input("Ask a policy question (e.g., 'How is medical leave regularized?')...")
query_to_send = selected_test_query if selected_test_query else prompt_input

if query_to_send:
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": query_to_send})
    with st.chat_message("user"):
        st.markdown(query_to_send)

    # Generate Response from Backend
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing policy handbook and consulting MultiQueryRetriever..."):
            backend_res = query_rag_backend(query_to_send)

            if backend_res["success"]:
                data = backend_res["data"]
                answer = data.get("answer", "No answer received.")
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Source Citations & Context Evidence", expanded=False):
                        for src in sources:
                            st.markdown(f"""
                            <div class="source-card">
                                <b>Page {src.get('page', 'N/A')}</b> ({src.get('source', 'Handbook')})<br/>
                                <em>"{src.get('snippet', '')}..."</em>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error(backend_res["error"])
                st.info("💡 Tip: Verify that the FastAPI backend server is running on `http://localhost:8000`.")
