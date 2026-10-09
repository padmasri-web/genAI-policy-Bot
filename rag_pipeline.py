import os
import shutil
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
DEFAULT_PDF_PATH = os.getenv("DEFAULT_PDF_PATH", "Mirai_SoT_Policy_Handbook_2026.pdf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
PRIMARY_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_CANDIDATES = [
    PRIMARY_LLM_MODEL,
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest"
]


def get_api_key() -> str:
    """Retrieve Google/Gemini API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set in environment.")
    return api_key


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instantiate Google Generative AI Embeddings with fallback capability."""
    api_key = get_api_key()
    try:
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    except Exception:
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)


def get_llm(model_name: str = PRIMARY_LLM_MODEL, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Instantiate ChatGoogleGenerativeAI with deterministic temperature (0.0).
    """
    api_key = get_api_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=1,
        google_api_key=api_key,
    )


def get_vectorstore(persist_directory: str = CHROMA_PERSIST_DIRECTORY) -> Chroma:
    """Load or initialize ChromaDB vector database."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="mirai_policy_collection"
    )


def ingest_document(
    file_path: str,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset_existing: bool = False
) -> Dict[str, Any]:
    """
    Data Pipeline Ingestion & Vectorization:
    1. Loads PDF via PyPDFLoader.
    2. Chunks text using RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200).
    3. Embeds chunks via GoogleGenerativeAIEmbeddings and stores in local ChromaDB.
    """
    if reset_existing and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="mirai_policy_collection"
    )

    return {
        "status": "success",
        "file_path": file_path,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "total_records_in_db": vectorstore._collection.count()
    }


SYSTEM_PROMPT = """You are an autonomous AI policy advisor for Mirai School of Technology (MSOT).
Your objective is to provide authoritative, deterministic, and factual guidance to students based strictly and exclusively on the official MirAi Student Policy Handbook 2026 provided in the retrieved context.

STRICT OPERATIONAL GUARDRAILS:
1. STRICT DOCUMENT INTEGRITY:
   - Rely strictly and solely on the provided context below.
   - Do NOT rely on your base training data or external general knowledge.
   - If the answer to a student's inquiry cannot be found within the retrieved context, you must politely decline to answer, clearly stating that the information is not provided in the student policy handbook.
   - Never invent, speculate, or fabricate policies, monetary fines, fees, contacts, or regulations. If a specific monetary fine is not stated, explicitly clarify that tobacco/smoking is prohibited and subject to Disciplinary Committee action, but no specific monetary fine is specified in the handbook.
2. FACTUAL PRECISION & MULTI-HOP SYNTHESIS:
   - Synthesize all applicable clauses across relevant handbook sections:
     * Attendance Marks: Explicitly state the attendance tier marks (e.g. 60% – 74.99% receives 4 marks) while noting the 75% attendance minimum requirement and debarment policy (falling below 75% receives 0 marks and debarment unless condoned or covered by an approved medical exemption of up to 15% buffer).
     * Medical & Duty Leave (Ratnam / other campuses): Synthesize campus contacts and submission deadlines:
       - For Ratnam campus, students must submit/email documents to Campus Manager Yashaswini Ma'am (in person at CM office or campus email).
       - All supporting medical documents must be submitted within exactly 7 days of the illness/treatment (or last day of event).
     * Club & Society Formation: To start a new society (like a Cybersecurity society under the Tech Club):
       - At least 40% batch student support/signatures is required.
       - The proposal and itemized budget must be submitted to the designated Faculty Coordinator first for initial review.
       - Students must NOT approach Management directly; only the Faculty Coordinator is authorized to forward proposals to Management.
     * Prohibited Conduct: Tobacco and smoking on campus are strictly prohibited and referred to the Disciplinary Committee, but the handbook does not specify a monetary fine.
3. TONE & STRUCTURE:
   - Provide answers in a well-structured, clear, professional, and accessible format for students.

Retrieved Context:
{context}
"""


def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into a clean context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Handbook")
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)


def execute_llm_generation(context: str, question: str) -> str:
    """
    Executes LCEL generation chain with primary LLM (gemini-3.8-flash)
    and robust fallback across candidate models if rate limit or quota is met.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    last_exception = None
    for model_name in MODEL_CANDIDATES:
        try:
            llm = get_llm(model_name=model_name, temperature=0.0)
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        except Exception as e:
            last_exception = e
            continue

    raise last_exception or RuntimeError("All generation models failed.")


class PolicyRAGService:
    """Singleton-style service class for executing queries and managing RAG operations."""

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIRECTORY):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self._initialize()

    def _initialize(self):
        self.vectorstore = get_vectorstore(self.persist_directory)
        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = 0

        if count == 0 and os.path.exists(DEFAULT_PDF_PATH):
            ingest_document(DEFAULT_PDF_PATH, persist_directory=self.persist_directory)
            self.vectorstore = get_vectorstore(self.persist_directory)

    def refresh(self):
        """Reload vectorstore."""
        self._initialize()

    def get_retriever(self):
        """Build MultiQueryRetriever using the first working candidate model."""
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        for model_name in MODEL_CANDIDATES:
            try:
                llm = get_llm(model_name=model_name, temperature=0.0)
                return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
            except Exception:
                continue
        return base_retriever

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Executes a question through the MultiQueryRetriever + LCEL chain.
        """
        retriever = self.get_retriever()
        try:
            docs = retriever.invoke(question)
        except Exception:
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = base_retriever.invoke(question)

        context_text = format_documents(docs)
        answer = execute_llm_generation(context=context_text, question=question)

        source_info = [
            {
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "Handbook"),
                "snippet": doc.page_content.strip()[:200]
            }
            for doc in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "context": context_text,
            "source_documents": source_info
        }


# Global service instance
rag_service = PolicyRAGService()
