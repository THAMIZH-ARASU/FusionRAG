from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import asyncio
from core.interfaces import BaseEmbedding, QueryContext
from core.logger import RAGLogger
from core.exceptions import EmbeddingException

class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence transformer-based embedding service."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = RAGLogger("embedding_service")
        self.model_name = model_name
        
        with self.logger.log_operation("model_loading", model_name=model_name):
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                raise EmbeddingException(f"Failed to load embedding model: {e}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        try:
            # Run in thread pool since sentence-transformers is sync
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.model.encode, text)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingException(f"Failed to embed text: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        with self.logger.log_operation("batch_embedding", batch_size=len(texts)):
            try:
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(None, self.model.encode, texts)
                return embeddings.tolist()
            except Exception as e:
                raise EmbeddingException(f"Failed to embed batch: {e}")

class HyDEGenerator:
    """Generates hypothetical documents for improved retrieval."""
    
    def __init__(self, llm_client, embedding_service: BaseEmbedding):
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.logger = RAGLogger("hyde_generator")
    
    async def generate_hyde_embedding(self, query_context: QueryContext) -> List[float]:
        """Generate HyDE embedding for the query."""
        with self.logger.log_operation("hyde_generation", query=query_context.original_query):
            # Generate hypothetical document
            hyde_document = await self._generate_hypothetical_document(query_context)
            
            # Embed the hypothetical document
            embedding = await self.embedding_service.embed_text(hyde_document)
            
            self.logger.logger.info(
                "hyde_generated",
                query=query_context.original_query,
                hyde_document=hyde_document[:200] + "..." if len(hyde_document) > 200 else hyde_document
            )
            
            return embedding
    
    async def _generate_hypothetical_document(self, query_context: QueryContext) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Given the following query, write a short, informative document that would directly answer it.
Be specific and factual. Write as if you are providing the ideal document that would contain the answer.

Query: {query_context.original_query}
Intent: {query_context.intent}
Entities: {', '.join(query_context.entities)}

Hypothetical Document:"""
        
        try:
            response = await self.llm_client.generate(prompt, max_tokens=300, temperature=0.3)
            return response.strip()
        except Exception as e:
            self.logger.logger.warning("hyde_generation_failed", error=str(e))
            # Fallback to simple document generation
            return f"This document explains {query_context.original_query} and provides information about {', '.join(query_context.entities)}."
