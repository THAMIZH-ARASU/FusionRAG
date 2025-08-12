from typing import List

import numpy as np
from fusion_rag.embedding_engines.base_engine import EmbeddingEngine
from fusion_rag.structures.document import Document


class HyDEEmbedding(EmbeddingEngine):
    """Hypothetical Document Embedding (HyDE) implementation"""
    
    def __init__(self, base_embedding: EmbeddingEngine, llm_client=None):
        self.base_embedding = base_embedding
        self.llm_client = llm_client
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document for the query"""
        if not self.llm_client:
            return query  # Fallback to original query
        
        prompt = f"""Given the question: "{query}"
        
        Write a detailed paragraph that would likely contain the answer to this question. 
        Focus on being informative and comprehensive."""
        
        try:
            response = self.llm_client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=200
            )
            return response.choices[0].text.strip()
        except:
            return query
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        return self.base_embedding.embed_documents(documents)
    
    def embed_query(self, query: str) -> np.ndarray:
        hypothetical_doc = self.generate_hypothetical_document(query)
        return self.base_embedding.embed_query(hypothetical_doc)