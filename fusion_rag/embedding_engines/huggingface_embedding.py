from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding_engines.base_engine import EmbeddingEngine
from structures.document import Document


class HuggingFaceEmbedding(EmbeddingEngine):
    """HuggingFace sentence transformers embedding"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        texts = [doc.content for doc in documents]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]