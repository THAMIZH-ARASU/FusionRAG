from typing import Dict, List, Tuple

from fusion_rag.retrieval_engines.base_engine import RetrievalEngine
from fusion_rag.structures.document import Document
from fusion_rag.structures.query_context import QueryContext


class HybridRetrievalEngine:
    """Combines multiple retrieval methods"""
    
    def __init__(self, engines: Dict[str, RetrievalEngine], weights: Dict[str, float] = None):
        self.engines = engines
        self.weights = weights or {name: 1.0 for name in engines.keys()}
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in all engines"""
        for engine in self.engines.values():
            engine.index_documents(documents)
    
    def retrieve(self, query: QueryContext, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve and combine results from all engines"""
        all_results = {}
        
        # Get results from each engine
        for name, engine in self.engines.items():
            results = engine.retrieve(query, k * 2)  # Get more results for better fusion
            weight = self.weights.get(name, 1.0)
            
            for doc, score in results:
                doc_id = doc.id
                if doc_id not in all_results:
                    all_results[doc_id] = {'doc': doc, 'scores': {}}
                all_results[doc_id]['scores'][name] = score * weight
        
        # Combine scores using RRF (Reciprocal Rank Fusion)
        final_results = []
        for doc_data in all_results.values():
            combined_score = self._calculate_rrf_score(doc_data['scores'])
            final_results.append((doc_data['doc'], combined_score))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    def _calculate_rrf_score(self, scores: Dict[str, float]) -> float:
        """Calculate Reciprocal Rank Fusion score"""
        rrf_score = 0.0
        for engine_name, score in scores.items():
            # Convert score to rank (higher score = lower rank)
            rank = 1 / (1 + score)  # Simple ranking conversion
            rrf_score += 1 / (60 + rank)  # RRF formula with k=60
        return rrf_score