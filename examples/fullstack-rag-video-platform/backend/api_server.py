"""
FastAPI server for document management and analytics.
Provides REST API endpoints for the frontend.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config
from rag_engine import RAGEngine
from memory_manager import MemoryManager

logger = logging.getLogger("api-server")
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Video Platform API",
    description="API for document management and conversation analytics",
    version="1.0.0",
)

# Configuration
config = Config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine: Optional[RAGEngine] = None
memory_manager: Optional[MemoryManager] = None


# Request/Response Models
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""

    success: bool
    document_id: str
    filename: str
    message: str


class DocumentInfo(BaseModel):
    """Document information."""

    document_id: str
    filename: str
    upload_date: str
    size_bytes: int


class ConversationMessage(BaseModel):
    """Conversation message."""

    role: str
    content: str
    timestamp: str


class ConversationHistory(BaseModel):
    """Conversation history."""

    user_id: str
    messages: List[ConversationMessage]
    total_count: int


class SystemStats(BaseModel):
    """System statistics."""

    rag_status: str
    document_count: int
    total_conversations: int
    active_sessions: int
    uptime_seconds: float


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_engine, memory_manager

    logger.info("Starting API server...")

    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_db_url=config.vector_db_url,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    await rag_engine.initialize()
    logger.info("✓ RAG Engine initialized")

    # Initialize memory manager
    memory_manager = MemoryManager(
        db_path=config.memory_db_path,
        window_size=config.memory_window,
    )
    await memory_manager.initialize()
    logger.info("✓ Memory Manager initialized")

    logger.info("🚀 API server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")
    if memory_manager:
        await memory_manager.close()


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Document Management Endpoints
@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the RAG system.

    Args:
        file: The file to upload

    Returns:
        Upload confirmation with document ID
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        allowed_types = config.allowed_file_types.split(",")
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed types: {allowed_types}",
            )

        # Create upload directory
        upload_dir = Path(config.storage_path) / "documents"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_dir / f"{datetime.utcnow().timestamp()}_{file.filename}"
        content = await file.read()

        if len(content) > config.upload_max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {config.upload_max_size} bytes",
            )

        with open(file_path, "wb") as f:
            f.write(content)

        # Add to RAG system
        doc_count = await rag_engine.add_documents(
            file_paths=[str(file_path)],
            metadata={
                "filename": file.filename,
                "upload_date": datetime.utcnow().isoformat(),
                "size_bytes": len(content),
            },
        )

        logger.info(f"Uploaded document: {file.filename} ({len(content)} bytes)")

        return DocumentUploadResponse(
            success=True,
            document_id=str(file_path.stem),
            filename=file.filename,
            message=f"Successfully uploaded and indexed {file.filename}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all uploaded documents.

    Returns:
        List of document information
    """
    try:
        upload_dir = Path(config.storage_path) / "documents"
        if not upload_dir.exists():
            return []

        documents = []
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                documents.append(
                    DocumentInfo(
                        document_id=file_path.stem,
                        filename=file_path.name,
                        upload_date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        size_bytes=stat.st_size,
                    )
                )

        return documents

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the RAG system.

    Args:
        document_id: Document ID to delete

    Returns:
        Deletion confirmation
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        # Delete from RAG engine
        success = await rag_engine.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"success": True, "message": f"Document {document_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Management Endpoints
@app.get("/api/conversations/{user_id}", response_model=ConversationHistory)
async def get_conversation_history(user_id: str, limit: int = 50):
    """
    Get conversation history for a user.

    Args:
        user_id: User identifier
        limit: Maximum number of messages to return

    Returns:
        Conversation history
    """
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        messages = await memory_manager.get_recent_history(user_id, limit=limit)

        conversation_messages = [
            ConversationMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
            )
            for msg in messages
        ]

        return ConversationHistory(
            user_id=user_id,
            messages=conversation_messages,
            total_count=len(conversation_messages),
        )

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{user_id}")
async def delete_conversation(user_id: str):
    """
    Delete conversation history for a user.

    Args:
        user_id: User identifier

    Returns:
        Deletion confirmation
    """
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        success = await memory_manager.delete_conversation(user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"success": True, "message": f"Conversation for {user_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.get("/api/analytics/stats", response_model=SystemStats)
async def get_system_stats():
    """
    Get system statistics.

    Returns:
        System statistics
    """
    try:
        rag_stats = await rag_engine.get_stats() if rag_engine else {}

        return SystemStats(
            rag_status=rag_stats.get("status", "unknown"),
            document_count=rag_stats.get("document_count", 0),
            total_conversations=0,  # Implement if needed
            active_sessions=0,  # Implement if needed
            uptime_seconds=0.0,  # Implement if needed
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    """
    Get current configuration (non-sensitive values only).

    Returns:
        Configuration dictionary
    """
    return config.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info",
    )
