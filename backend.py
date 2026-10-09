import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import rag_service, ingest_document, CHROMA_PERSIST_DIRECTORY

app = FastAPI(
    title="MirAI Student Policy Advisor API",
    description="Production-grade RAG backend powered by LangChain, MultiQueryRetriever, ChromaDB, and Google Gemini.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="Student inquiry regarding Mirai School of Technology policies.", min_length=2)


class SourceDocument(BaseModel):
    page: int
    source: str
    snippet: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    status: str = "success"


class IngestResponse(BaseModel):
    message: str
    filename: str
    pages_loaded: int
    chunks_created: int
    total_records_in_db: int
    status: str = "success"


@app.get("/", tags=["Health"])
async def root():
    """Service status and greeting."""
    return {
        "service": "Autonomous MirAI Student Policy Advisor",
        "status": "online",
        "docs_url": "/docs",
        "persist_directory": CHROMA_PERSIST_DIRECTORY
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint providing vector store statistics."""
    try:
        count = rag_service.vectorstore._collection.count()
        return {
            "status": "healthy",
            "collection_records": count,
            "vector_store": "ChromaDB",
            "model_ready": True
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file via multipart/form-data, processes the text into semantic chunks,
    and updates the local ChromaDB vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    # Save uploaded file temporarily for PyPDFLoader
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest document into vector store
        ingest_result = ingest_document(
            file_path=temp_file_path,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            chunk_size=1000,
            chunk_overlap=200,
            reset_existing=False
        )

        # Refresh RAG service to load updated vector store
        rag_service.refresh()

        return IngestResponse(
            message=f"Document '{file.filename}' successfully ingested and vectorized.",
            filename=file.filename,
            pages_loaded=ingest_result["pages_loaded"],
            chunks_created=ingest_result["chunks_created"],
            total_records_in_db=ingest_result["total_records_in_db"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )
    finally:
        # Clean up temporary upload directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Accepts a JSON payload containing the student's question,
    executes the MultiQueryRetriever and LCEL RAG chain,
    and returns the factual answer with source citations.
    """
    try:
        result = rag_service.ask(request.question)
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceDocument(
                    page=doc.get("page", 0),
                    source=os.path.basename(doc.get("source", "Handbook")),
                    snippet=doc.get("snippet", "")
                )
                for doc in result.get("source_documents", [])
            ],
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
