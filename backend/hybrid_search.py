"""
Hybrid Search Module for RAG Pipeline

This module combines semantic (vector) search with keyword (BM25) search
using Reciprocal Rank Fusion (RRF) for better retrieval results.

Usage:
    from hybrid_search import HybridSearcher, HybridRetriever
    
    # Initialize
    searcher = HybridSearcher(vector_db, collection, embedding_model)
    
    # Search
    results = searcher.search("your query", k=5, alpha=0.5)
"""

import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


class HybridSearcher:
    """
    Combines semantic (vector) and keyword (BM25) search for better retrieval.
    
    Attributes:
        vector_db: LangChain Chroma vector store
        collection: ChromaDB collection
        embedding_model: HuggingFace embedding model
        bm25: BM25Okapi index
        all_documents: List of all documents
    """
    
    def __init__(self, vector_db, collection, embedding_model):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_db: LangChain Chroma vector store
            collection: ChromaDB collection
            embedding_model: HuggingFace embedding model
        """
        self.vector_db = vector_db
        self.collection = collection
        self.embedding_model = embedding_model
        
        # Build BM25 index
        print("🔄 Building BM25 index...")
        self._build_bm25_index()
        print(f"✅ BM25 index built with {len(self.all_documents)} documents")
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        # Fetch all documents
        all_data = self.collection.get(include=["documents", "metadatas"])
        
        # Create Document objects
        self.all_documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_data['documents'], all_data['metadatas'])
        ]
        
        # Tokenize for BM25
        self.tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in self.all_documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of Document objects
        """
        return self.vector_db.similarity_search(query, k=k)
    
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform keyword search using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of Document objects with BM25 scores
        """
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        
        # Return documents with scores
        results = []
        for idx in top_indices:
            doc = self.all_documents[idx]
            # Add BM25 score to metadata
            doc.metadata['bm25_score'] = float(bm25_scores[idx])
            results.append(doc)
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        alpha: float = 0.5,
        rrf_k: int = 60
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Uses Reciprocal Rank Fusion (RRF) to combine rankings.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (0-1)
                   1.0 = pure semantic
                   0.5 = balanced (recommended)
                   0.0 = pure keyword
            rrf_k: RRF constant (default: 60)
            
        Returns:
            List of Document objects ranked by hybrid score
        """
        # Get results from both methods (fetch more for better fusion)
        fetch_k = k * 2
        
        semantic_docs = self.semantic_search(query, k=fetch_k)
        keyword_docs = self.keyword_search(query, k=fetch_k)
        
        # Reciprocal Rank Fusion
        doc_scores = {}
        
        # Score semantic results
        for rank, doc in enumerate(semantic_docs, 1):
            doc_id = id(doc.page_content)  # Use object id as unique identifier
            rrf_score = alpha * (1.0 / (rank + rrf_k))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'score': rrf_score,
                    'doc': doc,
                    'semantic_rank': rank,
                    'keyword_rank': None
                }
            else:
                doc_scores[doc_id]['score'] += rrf_score
                doc_scores[doc_id]['semantic_rank'] = rank
        
        # Score keyword results
        for rank, doc in enumerate(keyword_docs, 1):
            doc_id = id(doc.page_content)
            rrf_score = (1 - alpha) * (1.0 / (rank + rrf_k))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'score': rrf_score,
                    'doc': doc,
                    'semantic_rank': None,
                    'keyword_rank': rank
                }
            else:
                doc_scores[doc_id]['score'] += rrf_score
                doc_scores[doc_id]['keyword_rank'] = rank
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Add hybrid score to metadata and return top k
        results = []
        for item in sorted_results[:k]:
            doc = item['doc']
            doc.metadata['hybrid_score'] = item['score']
            doc.metadata['semantic_rank'] = item['semantic_rank']
            doc.metadata['keyword_rank'] = item['keyword_rank']
            results.append(doc)
        
        return results
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        alpha: float = 0.5,
        method: str = "hybrid"
    ) -> List[Document]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            k: Number of results
            alpha: Semantic weight (for hybrid only)
            method: "semantic", "keyword", or "hybrid"
            
        Returns:
            List of Document objects
        """
        if method == "semantic":
            return self.semantic_search(query, k)
        elif method == "keyword":
            return self.keyword_search(query, k)
        elif method == "hybrid":
            return self.hybrid_search(query, k, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_methods(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """
        Compare all three search methods side-by-side.
        
        Args:
            query: Search query
            k: Number of results per method
            
        Returns:
            Dictionary with results from each method
        """
        return {
            "semantic": self.semantic_search(query, k),
            "keyword": self.keyword_search(query, k),
            "hybrid": self.hybrid_search(query, k, alpha=0.5)
        }


class HybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever using hybrid search.
    
    Can be used directly with RetrievalQA and other LangChain chains.
    """
    
    # Pydantic v2 field definitions - use Any to avoid strict type checking
    searcher: Any  # Object with hybrid_search method
    k: int = 5
    alpha: float = 0.5
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context: Any) -> None:
        """Validate searcher has required method after initialization."""
        if not hasattr(self.searcher, 'hybrid_search'):
            raise ValueError("searcher must have a 'hybrid_search' method")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            run_manager: Callback manager (optional)
            
        Returns:
            List of relevant Document objects
        """
        return self.searcher.hybrid_search(query, k=self.k, alpha=self.alpha)


def print_search_results(results: List[Document], title: str = "Search Results"):
    """
    Pretty print search results.
    
    Args:
        results: List of Document objects
        title: Title for the results section
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    for i, doc in enumerate(results, 1):
        print(f"[{i}] Page {doc.metadata.get('page', 'N/A')}")
        
        # Show scores if available
        if 'hybrid_score' in doc.metadata:
            print(f"    Hybrid Score: {doc.metadata['hybrid_score']:.4f}")
            print(f"    Semantic Rank: {doc.metadata.get('semantic_rank', 'N/A')}")
            print(f"    Keyword Rank: {doc.metadata.get('keyword_rank', 'N/A')}")
        elif 'bm25_score' in doc.metadata:
            print(f"    BM25 Score: {doc.metadata['bm25_score']:.3f}")
        
        # Show content preview
        preview = doc.page_content[:200].replace('\n', ' ')
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    print("Hybrid Search Module")
    print("=" * 50)
    print("\nThis module provides hybrid search functionality")
    print("combining semantic and keyword search.")
    print("\nImport this in your notebook to use hybrid search!")
