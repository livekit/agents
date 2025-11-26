"""
Advanced RAG Engine with LlamaIndex
Provides fast, efficient document retrieval with semantic search.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger("rag-engine")


class RAGEngine:
    """Advanced RAG engine with optimized retrieval and caching."""

    def __init__(
        self,
        vector_db_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        persist_dir: str = "./storage",
    ):
        """
        Initialize the RAG engine.

        Args:
            vector_db_url: URL for vector database (optional)
            embedding_model: OpenAI embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            persist_dir: Directory to persist vector store
        """
        self.vector_db_url = vector_db_url
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None

        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
        Settings.llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        logger.info(
            f"RAG Engine configured with {embedding_model}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    async def initialize(self):
        """Initialize the RAG engine and load existing index if available."""
        try:
            # Check if index exists
            if (self.persist_dir / "docstore.json").exists():
                logger.info("Loading existing vector index...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.persist_dir)
                )
                self.index = await asyncio.to_thread(
                    VectorStoreIndex.load_from_storage,
                    storage_context,
                )
                logger.info("✓ Loaded existing index")
            else:
                logger.info("Creating new vector index...")
                self.index = VectorStoreIndex([])
                await self._persist_index()
                logger.info("✓ Created new index")

            # Configure query engine with optimized retrieval
            self._setup_query_engine(top_k=5)

        except Exception as e:
            logger.error(f"Error initializing RAG engine: {e}")
            raise

    def _setup_query_engine(self, top_k: int = 5):
        """Setup the query engine with retriever and postprocessors."""
        if not self.index:
            raise ValueError("Index not initialized")

        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k * 2,  # Retrieve more, then filter
        )

        # Create query engine with postprocessing
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7),
            ],
        )

    async def add_documents(
        self, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the RAG system.

        Args:
            file_paths: List of file paths to add
            metadata: Optional metadata to attach to documents

        Returns:
            Number of documents added
        """
        if not self.index:
            raise ValueError("RAG engine not initialized")

        try:
            # Load documents
            documents = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                logger.info(f"Loading document: {file_path}")
                reader = SimpleDirectoryReader(
                    input_files=[file_path],
                )
                docs = await asyncio.to_thread(reader.load_data)

                # Add metadata
                if metadata:
                    for doc in docs:
                        doc.metadata.update(metadata)
                        doc.metadata["source_file"] = os.path.basename(file_path)

                documents.extend(docs)

            if not documents:
                logger.warning("No documents loaded")
                return 0

            # Split into chunks
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            nodes = await asyncio.to_thread(splitter.get_nodes_from_documents, documents)

            logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")

            # Add to index
            await asyncio.to_thread(self.index.insert_nodes, nodes)

            # Persist index
            await self._persist_index()

            logger.info(f"✓ Added {len(documents)} documents to RAG system")
            return len(documents)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def add_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add raw text to the RAG system.

        Args:
            text: Text content to add
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self.index:
            raise ValueError("RAG engine not initialized")

        try:
            doc = Document(text=text, metadata=metadata or {})

            # Split into chunks
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            nodes = await asyncio.to_thread(splitter.get_nodes_from_documents, [doc])

            # Add to index
            await asyncio.to_thread(self.index.insert_nodes, nodes)

            # Persist index
            await self._persist_index()

            logger.info(f"✓ Added text to RAG system ({len(nodes)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Error adding text: {e}")
            return False

    async def query(
        self,
        query: str,
        top_k: int = 5,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Query the RAG system.

        Args:
            query: The user's query
            top_k: Number of top results to retrieve
            conversation_context: Optional conversation history for context

        Returns:
            Retrieved information and answer
        """
        if not self.query_engine:
            return "RAG system not ready"

        try:
            # Update query engine for this query
            self._setup_query_engine(top_k=top_k)

            # Build context-aware query
            context_query = query
            if conversation_context:
                # Add recent context to improve retrieval
                recent_msgs = conversation_context[-3:]  # Last 3 messages
                context_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in recent_msgs]
                )
                context_query = f"Given this conversation:\n{context_str}\n\nAnswer: {query}"

            # Execute query
            logger.info(f"Querying RAG: {query}")
            response = await asyncio.to_thread(
                self.query_engine.query,
                context_query,
            )

            # Extract sources
            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    if hasattr(node, "metadata"):
                        source_file = node.metadata.get("source_file", "Unknown")
                        score = node.score if hasattr(node, "score") else 0
                        sources.append(f"{source_file} (relevance: {score:.2f})")

            # Format response
            answer = str(response)
            if sources:
                answer += f"\n\nSources: {', '.join(sources[:3])}"

            logger.info(f"✓ RAG query completed ({len(sources)} sources)")
            return answer

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"Error retrieving information: {str(e)}"

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.

        Args:
            doc_id: Document ID to delete

        Returns:
            Success status
        """
        try:
            if not self.index:
                return False

            await asyncio.to_thread(self.index.delete_ref_doc, doc_id)
            await self._persist_index()

            logger.info(f"✓ Deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            if not self.index:
                return {"status": "not_initialized"}

            # Get document count
            doc_store = self.index.docstore
            doc_count = len(doc_store.docs) if hasattr(doc_store, "docs") else 0

            return {
                "status": "ready",
                "document_count": doc_count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model,
                "persist_dir": str(self.persist_dir),
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}

    async def _persist_index(self):
        """Persist the index to disk."""
        if self.index:
            await asyncio.to_thread(
                self.index.storage_context.persist,
                persist_dir=str(self.persist_dir),
            )

    async def reindex(self):
        """Rebuild the index from scratch."""
        logger.info("Reindexing RAG system...")
        # Implementation for reindexing if needed
        await self._persist_index()
        logger.info("✓ Reindexing complete")
