"""
FastAPI Backend for RAG Chat Application

Endpoints:
- POST /chat - Send message and get RAG response
- POST /upload - Upload document and process with LandingAI ADE
- GET /history - Get chat history
- GET /stats - Get collection statistics
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chat API", version="1.0.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

# In-memory chat history (in production, use a database)
chat_history: List[dict] = []

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    timestamp: str


class UploadResponse(BaseModel):
    filename: str
    chunks_loaded: int
    message: str


class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    db_path: str
    chat_messages: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chat API",
        "version": "1.0.0",
        "endpoints": ["/chat", "/upload", "/history", "/stats"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get RAG response.
    
    Args:
        request: ChatRequest with message
        
    Returns:
        ChatResponse with answer and sources
    """
    try:
        # Query RAG system
        result = rag_system.query(request.message)
        
        # Create response
        response = ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            timestamp=datetime.now().isoformat()
        )
        
        # Save to history
        chat_history.append({
            "role": "user",
            "content": request.message,
            "timestamp": response.timestamp
        })
        chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "timestamp": response.timestamp
        })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and process it with LandingAI ADE.
    
    Args:
        file: Uploaded file (PDF, image, etc.)
        
    Returns:
        UploadResponse with processing results
    """
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📄 Saved file: {file_path}")
        
        # Check if Vision Agent API key is set (used by LandingAI ADE)
        ade_api_key = os.getenv("VISION_AGENT_API_KEY")
        if not ade_api_key or ade_api_key.startswith("your-"):
            raise HTTPException(
                status_code=400,
                detail="Vision Agent API key not configured. Please set VISION_AGENT_API_KEY in .env file"
            )
        
        # Import LandingAI ADE
        try:
            from landingai_ade import LandingAIADE
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="landingai-ade package not installed. Run: pip install landingai-ade"
            )
        
        # Parse document with LandingAI ADE
        # Note: API key is read from VISION_AGENT_API_KEY environment variable
        print("⚡ Calling LandingAI ADE API...")
        try:
            client = LandingAIADE()
            # Pass the Path object directly
            parse_result = client.parse(
                document=file_path,  # Path object
                model="dpt-2-latest"
            )
            
            # Convert parse result to dict if needed
            if hasattr(parse_result, 'model_dump'):
                parse_result = parse_result.model_dump()
            elif hasattr(parse_result, 'dict'):
                parse_result = parse_result.dict()
                
        except Exception as e:
            # Clean up file
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"LandingAI ADE parsing failed: {str(e)}"
            )
        
        # Extract chunks from parse result
        chunks = parse_result.get("chunks", [])
        
        # Load chunks into RAG system
        chunks_loaded = rag_system.load_documents_from_ade_result(chunks)
        
        # Clean up uploaded file
        file_path.unlink()
        
        return UploadResponse(
            filename=file.filename,
            chunks_loaded=chunks_loaded,
            message=f"Successfully processed {chunks_loaded} chunks from {file.filename}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if it exists
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """
    Get chat history.
    
    Returns:
        List of chat messages
    """
    return {"history": chat_history}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get collection statistics.
    
    Returns:
        StatsResponse with collection info
    """
    stats = rag_system.get_collection_stats()
    stats["chat_messages"] = len(chat_history)
    return StatsResponse(**stats)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
