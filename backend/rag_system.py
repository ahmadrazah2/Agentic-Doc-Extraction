"""
RAG System with Hybrid Search and Groq LLM

This module handles:
- Vector database initialization (ChromaDB)
- Document loading and processing
- Hybrid search (semantic + keyword)
- LLM integration (Groq)
- Custom prompts for table parsing
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import chromadb

from hybrid_search import HybridSearcher, HybridRetriever
from markdown_loader import load_markdown_chunks

# Load environment variables
load_dotenv()


class RAGSystem:
    """
    Complete RAG system with hybrid search and Groq LLM.
    """
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG system.
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_model: HuggingFace embedding model name
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        print("🔄 Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Initialize LangChain vector store
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.chroma_db_path)
        )
        
        # Hybrid searcher will be initialized lazily when needed
        self._hybrid_searcher = None
        self._hybrid_retriever = None
        
        # Initialize Groq LLM
        print("🔄 Initializing Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Create custom prompt for table parsing
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
The context may contain HTML tables with <table>, <tr>, <td> tags. 
Extract values from these tables carefully, looking at the row with "MGRRN(ours)" or "MGRRN" 
and the column headers to find the requested values.

Context: {context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        print("✅ RAG system initialized!")
    
    @property
    def hybrid_searcher(self):
        """Lazy initialization of hybrid searcher."""
        if self._hybrid_searcher is None:
            # Check if collection has documents
            if self.collection.count() == 0:
                raise ValueError(
                    "Cannot initialize hybrid search: collection is empty. "
                    "Please load documents first using load_documents_from_markdown() or load_documents_from_ade_result()"
                )
            
            print("🔄 Initializing hybrid search...")
            self._hybrid_searcher = HybridSearcher(
                vector_db=self.vector_db,
                collection=self.collection,
                embedding_model=self.embedding_model
            )
        return self._hybrid_searcher
    
    @property
    def hybrid_retriever(self):
        """Lazy initialization of hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                searcher=self.hybrid_searcher,  # This will trigger lazy init
                k=15,
                alpha=0.3
            )
        return self._hybrid_retriever
    
    @property
    def qa_chain(self):
        """Lazy initialization of QA chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.hybrid_retriever,  # This will trigger lazy init
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def load_documents_from_markdown(self, file_path: str) -> int:
        """
        Load documents from markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Number of documents loaded
        """
        print(f"📄 Loading documents from {file_path}...")
        documents = load_markdown_chunks(file_path)
        
        added = 0
        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue
            
            # Generate embedding
            embedding = self.embedding_model.embed_query(text)
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[doc.metadata['chunk_id']],
                documents=[text],
                metadatas=doc.metadata,
                embeddings=[embedding]
            )
            added += 1
        
        print(f"✅ Loaded {added} documents")
        return added
    
    def load_documents_from_ade_result(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Load documents from LandingAI ADE parse result.
        
        Args:
            chunks: List of chunks from ADE parse result
            
        Returns:
            Number of documents loaded
        """
        print(f"📄 Loading {len(chunks)} chunks from ADE result...")
        
        added = 0
        for chunk in chunks:
            chunk_id = chunk.get("id")
            text = chunk.get("markdown", chunk.get("text", "")).strip()
            
            if not text:
                continue
            
            # Generate embedding
            embedding = self.embedding_model.embed_query(text)
            
            # Prepare metadata
            grounding = chunk.get("grounding", {})
            metadata = {
                "chunk_id": chunk_id,
                "page": int(grounding.get("page", 0)),
                "source": "ade_upload",
                "chunk_type": chunk.get("type", "text")
            }
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[chunk_id],
                documents=[text],
                metadatas=metadata,
                embeddings=[embedding]
            )
            added += 1
        
        print(f"✅ Loaded {added} documents")
        return added
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        response = self.qa_chain.invoke(question)
        
        return {
            "answer": response['result'],
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in response.get('source_documents', [])[:3]
            ]
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "db_path": str(self.chroma_db_path)
        }


if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    
    # Test query
    result = rag.query("What is MGRRN?")
    print(f"\n💡 Answer: {result['answer']}")
