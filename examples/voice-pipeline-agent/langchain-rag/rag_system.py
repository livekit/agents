import logging
import os
import pickle
import sqlite3
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever

# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_nomic import NomicEmbeddings
from rich.logging import RichHandler

# Configure logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simplify log format for Rich
    datefmt="[%X]",
    handlers=[
        RichHandler(show_time=True, show_level=True, show_path=False),
        logging.FileHandler("rag_system.log")  # Log to file
    ]
)

load_dotenv()

class RAGSystem:
    def __init__(self, collection_name, data_source_path, persist_directory='rag_vector_store', create_if_not_exists=False):
        """
        Each Instance Creates a collection which will represent a Separate RAG System
        For Example,
            1. Gym Trainer(with gym_data collection)
            2. Recipe Specialist(with recipe_data collection)
        
        :param create_if_not_exists: Flag to specify whether to create collection if it doesn't exist.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.data_source_path = data_source_path
        self.create_if_not_exists = create_if_not_exists  # New flag
        self.keyword_search_retriever = None
        self.vector_store_retriever = None
        self.embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", dimensionality=768)
        self.vector_store_collection = None
        self.bm25_cache_path = Path(persist_directory) / f"{collection_name}_bm25_docs.pkl"

        # Check for Nomic API key
        nomic_api_key = os.getenv("NOMIC_API_KEY")
        if not nomic_api_key:
            logging.error("NOMIC_API_KEY is missing. Please provide it in your environment variables(.env file)")
            raise ValueError("Missing NOMIC_API_KEY. Please provide the NOMIC_API_KEY in your environment(.env file)")

        # Try to load the collection if it exists
        if self.collection_exists_in_db():
            try:
                self.vector_store_collection = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory,
                    create_collection_if_not_exists=False
                )
                logging.info(f"Existing {self.collection_name} collection loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load existing {self.collection_name} collection: {e}")
                self.vector_store_collection = None
        
        # If collection doesn't exist and create_if_not_exists flag is True, create the collection
        if not self.collection_exists_in_db() and self.create_if_not_exists:
            logging.info(f"Collection {self.collection_name} does not exist. Creating new collection.")
            self.process()

        # Try to load existing BM25 data
        self.load_bm25_retriever()


    @classmethod
    def list_available_collections(cls, persist_directory='rag_vector_store'):
        """
        Class method that lists all available collections with complete details from the SQLite database.
        :param persist_directory: Directory where the database is stored (default is 'rag_vector_store').
        """
        db_path = Path(persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            logging.warning("No database found.")
            return []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Retrieve all collections with their details (name, dimension, metadata, etc.)
            cursor.execute("SELECT name, id, dimension FROM collections")
            collections = cursor.fetchall()
            logging.info(f"Available collections with details: {collections}")
            
            # Prepare a list of collections with complete details
            collection_details = []
            for collection in collections:
                collection_info = {
                    'name': collection[0],
                    'collection_id' : collection[1],
                    'dimension': collection[2],
                }
                collection_details.append(collection_info)
            
            return collection_details
            
        except Exception as e:
            logging.error(f"Error retrieving collections: {e}")
            return []
        finally:
            conn.close()



    def collection_exists_in_db(self):
        """
        Check if collection exists in the SQLite database and has valid dimension
        """
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if collection exists and has valid dimension
            cursor.execute("SELECT name, dimension FROM collections WHERE name = ?", (self.collection_name,))
            result = cursor.fetchone()
            conn.close()
            
            # Return True only if collection exists and dimension is not None/NULL
            return result is not None and result[1] is not None
            
        except Exception as e:
            logging.error(f"Error checking collection existence: {e}")
            return False

    def loader(self, path):
        """
        Loads the document(s) from the provided directory.
        :param path: directory path
        :return: List of documents
        """
        path = Path(path)
        loader = DirectoryLoader(path)
        document = loader.load()  # This will handle both files and directories
        return document


    def chunk_text(self, document, chunk_size=1000, chunk_overlap=200):
        """
        Splits the document into semantically meaningful chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logging.info("Chunking...")
        chunks = text_splitter.split_documents(document)
        logging.info(f"Chunking Complete. Total number of chunks: {len(chunks)}")
        return chunks

    def create_keyword_search_retriever(self, document_chunks):
        """
        Creates Keyword Search Retriever using BM25Retriever and caches the documents
        """
        logging.info("Initiating Keyword Search Retriever Creation.")
        if document_chunks:
            self.keyword_search_retriever = BM25Retriever.from_documents(
                documents=document_chunks
            )
            self.keyword_search_retriever.k = 2
            
            # Cache the document chunks for future use
            try:
                with open(self.bm25_cache_path, 'wb') as f:
                    pickle.dump(document_chunks, f)
                logging.info("BM25 documents cached successfully.")
            except Exception as e:
                logging.error(f"Failed to cache BM25 documents: {e}")
                
        logging.info("Keyword Search Retriever Created Successfully.")

    def load_bm25_retriever(self):
        """
        Attempts to load BM25Retriever from cached documents
        """
        if self.bm25_cache_path.exists():
            try:
                with open(self.bm25_cache_path, 'rb') as f:
                    cached_docs = pickle.load(f)
                self.keyword_search_retriever = BM25Retriever.from_documents(
                    documents=cached_docs
                )
                self.keyword_search_retriever.k = 2
                logging.info("BM25 retriever loaded from cache successfully.")
                return True
            except Exception as e:
                logging.error(f"Failed to load BM25 retriever from cache: {e}")
                return False
        return False

    def create_vector_store_collection(self, document_chunks=None):
        """
        Creates or loads a Chroma vector store collection using embeddings.
        """
        logging.info("Initiating Vector Store Creation.")
        self.vector_store_collection = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        if document_chunks:
            logging.info("Adding Documents(chunks) to vector store")
            self.vector_store_collection.add_documents(
                ids=[str(uuid4()) for _ in range(len(document_chunks))],
                documents=document_chunks
            )
        logging.info("Vector Store Created Successfully.")

    def get_vector_store_collection(self):
        if self.vector_store_collection:
            return self.vector_store_collection
        else:
            raise ValueError("VectorStore not Initialized Yet")

    def create_vector_store_retriever(self):
        """
        Searches the Chroma vector store collection for relevant chunks using the query.
        """
        logging.info("Creating VectorStore Retriever")
        if not self.vector_store_collection:
            raise ValueError("Vector store collection not initialized. Call 'create_vector_store_collection' first.")
        self.vector_store_retriever = self.vector_store_collection.as_retriever(search_kwargs={'k': 2})
        logging.info("VectorStore Retriever Created Successfully.")

    def search_vector_store_retriever(self, query):
        """
        Searches the Chroma vector store collection for relevant chunks using the query.
        """
        relevant_docs = self.vector_store_retriever.invoke(query)
        return relevant_docs

    def search_keyword_retriver(self, query):
        logging.info("Searching by Keyword...")
        return self.keyword_search_retriever.invoke(query)

    def search(self, query):
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_store_retriever, self.keyword_search_retriever], weights=[1, 0]
        )
        relevant_docs = ensemble_retriever.invoke(query)
        return relevant_docs

    def process(self, query=None):
        """
        A single function that performs all the steps: loading, chunking (if necessary), creating/loading vector store collections, and searching.
        """
        need_chunks = False
        
        # Check if we need to load and chunk documents
        if self.vector_store_retriever is None and not self.collection_exists_in_db():
            need_chunks = True
        if self.keyword_search_retriever is None and not self.load_bm25_retriever():
            need_chunks = True
        
        # Only load and chunk documents if necessary
        chunks = None
        if need_chunks:
            document = self.loader(self.data_source_path)
            chunks = self.chunk_text(document)

        # Initialize vector store retriever if needed
        if self.vector_store_retriever is None:
            if not self.collection_exists_in_db():
                if chunks is None:
                    document = self.loader(self.data_source_path)
                    chunks = self.chunk_text(document)
                self.create_vector_store_collection(chunks)
            self.create_vector_store_retriever()

        # Initialize keyword search retriever if needed
        if self.keyword_search_retriever is None:
            if not self.load_bm25_retriever():
                if chunks is None:
                    document = self.loader(self.data_source_path)
                    chunks = self.chunk_text(document)
                self.create_keyword_search_retriever(chunks)

        # If query is provided, perform search
        if query is not None:
            return self.search(query)

    def __str__(self):
        return f"RAG for {self.collection_name} Initialized"


if __name__ == '__main__':
    data_expert = RAGSystem(collection_name='data-expert', data_source_path='data',create_if_not_exists=True)
    if data_expert.collection_exists_in_db():
        print(f"Collection named '{data_expert.collection_name}' exists")
    else:
        print(f"Collection named '{data_expert.collection_name}' does not exists")


    all_collections = RAGSystem.list_available_collections()
    print(all_collections)
