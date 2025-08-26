from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.interfaces import BaseReranker, RetrievalResult
from core.logging import RAGLogger

class ReciprocalRankFusionReranker(BaseReranker):
    """Reranks results using Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        self.k = k
        self.logger = RAGLogger("rrf_reranker")
    
    async def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using RRF."""
        with self.logger.log_operation("rrf_reranking", num_results=len(results)):
            if not results:
                return results
            
            # Group results by retrieval type
            grouped_results = {}
            for result in results:
                ret_type = result.retrieval_type.value
                if ret_type not in grouped_results:
                    grouped_results[ret_type] = []
                grouped_results[ret_type].append(result)
            
            # Sort each group by score
            for ret_type in grouped_results:
                grouped_results[ret_type].sort(key=lambda x: x.score, reverse=True)
            
            # Calculate RRF scores
            rrf_scores = {}
            for ret_type, type_results in grouped_results.items():
                for rank, result in enumerate(type_results):
                    doc_id = result.document.id
                    if doc_id not in rrf_scores:
                        rrf_scores[doc_id] = {
                            'score': 0,
                            'result': result,
                            'ranks': {}
                        }
                    
                    # RRF formula: 1 / (k + rank)
                    rrf_scores[doc_id]['score'] += 1.0 / (self.k + rank + 1)
                    rrf_scores[doc_id]['ranks'][ret_type] = rank + 1
            
            # Sort by RRF score
            sorted_results = sorted(
                rrf_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Update scores and create final results
            final_results = []
            for item in sorted_results:
                result = item['result']
                result.score = item['score']
                result.explanation = f"RRF score: {item['score']:.4f} (ranks: {item['ranks']})"
                final_results.append(result)
            
            self.logger.logger.info(
                "rrf_reranking_completed",
                original_count=len(results),
                final_count=len(final_results),
                top_score=final_results[0].score if final_results else 0
            )
            
            return final_results

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranking for higher quality."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.logger = RAGLogger("cross_encoder_reranker")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            self.logger.logger.warning("CrossEncoder not available, falling back to TF-IDF similarity")
            self.model = None
            self.tfidf = TfidfVectorizer()
    
    async def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank using cross-encoder or TF-IDF similarity."""
        with self.logger.log_operation("cross_encoder_reranking", num_results=len(results)):
            if not results:
                return results
            
            if self.model:
                return await self._rerank_with_cross_encoder(query, results)
            else:
                return await self._rerank_with_tfidf(query, results)
    
    async def _rerank_with_cross_encoder(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank using cross-encoder model."""
        query_doc_pairs = [(query, result.document.content) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(query_doc_pairs)
        
        # Update results with new scores
        for i, result in enumerate(results):
            result.score = float(scores[i])
            result.explanation = f"Cross-encoder relevance: {scores[i]:.4f}"
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    async def _rerank_with_tfidf(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fallback reranking using TF-IDF similarity."""
        documents = [result.document.content for result in results]
        all_texts = [query] + documents
        
        # Fit TF-IDF and compute similarity
        tfidf_matrix = self.tfidf.fit_transform(all_texts)
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        # Update scores
        for i, result in enumerate(results):
            result.score = float(similarities[i])
            result.explanation = f"TF-IDF similarity: {similarities[i]:.4f}"
        
        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        return results