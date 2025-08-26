from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import numpy as np
from core.interfaces import BaseRetriever, RetrievalResult, QueryContext, Document, RetrievalType
from core.logging import RAGLogger

class BM25Retriever(BaseRetriever):
    """BM25-based keyword retrieval."""
    
    def __init__(self, documents: List[Document]):
        self.logger = RAGLogger("bm25_retriever")
        self.documents = documents
        
        with self.logger.log_operation("bm25_index_building", num_documents=len(documents)):
            # Tokenize documents
            tokenized_docs = [doc.content.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
    
    async def retrieve(self, query_context: QueryContext, top_k: int) -> List[RetrievalResult]:
        """Retrieve documents using BM25."""
        with self.logger.log_operation("bm25_retrieval", query=query_context.transformed_query, top_k=top_k):
            query_tokens = query_context.transformed_query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    results.append(RetrievalResult(
                        document=self.documents[idx],
                        score=float(scores[idx]),
                        retrieval_type=RetrievalType.BM25,
                        explanation=f"BM25 lexical match score: {scores[idx]:.4f}"
                    ))
            
            self.logger.logger.info(
                "bm25_retrieval_completed",
                num_results=len(results),
                top_score=results[0].score if results else 0
            )
            
            return results