from typing import List, Tuple
from embedding_engines.base_engine import EmbeddingEngine
from retrieval_engines.base_engine import RetrievalEngine
from structures.document import Document

import faiss

from structures.query_context import QueryContext

class VectorDBEngine(RetrievalEngine):
    """Vector database retrieval engine using FAISS"""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.index = None
        self.documents = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in vector database"""
        self.documents = documents
        embeddings = self.embedding_engine.embed_documents(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query: QueryContext, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve documents using vector similarity"""
        if self.index is None:
            return []
        
        query_embedding = query.embeddings.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results