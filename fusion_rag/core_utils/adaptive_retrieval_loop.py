from typing import Any, Dict, Tuple

from fusion_rag.core_utils.context_manager import ContextManager
from fusion_rag.retrieval_engines.hybrid_engine import HybridRetrievalEngine
from fusion_rag.structures.query_context import QueryContext


class AdaptiveRetrievalLoop:
    """Implements adaptive retrieval with feedback"""
    
    def __init__(self, retrieval_engine: HybridRetrievalEngine, 
                 context_manager: ContextManager, 
                 max_iterations: int = 3):
        self.retrieval_engine = retrieval_engine
        self.context_manager = context_manager
        self.max_iterations = max_iterations
    
    def retrieve_with_feedback(self, query: QueryContext, 
                             initial_k: int = 10) -> Tuple[str, Dict[str, Any]]:
        """Perform adaptive retrieval with feedback loop"""
        context_sufficient = False
        iteration = 0
        all_retrieved_docs = []
        feedback_logs = []
        
        current_k = initial_k
        
        while not context_sufficient and iteration < self.max_iterations:
            # Retrieve documents
            retrieved_docs = self.retrieval_engine.retrieve(query, current_k)
            all_retrieved_docs.extend(retrieved_docs)
            
            # Generate context
            context = self.context_manager.enrich_context(all_retrieved_docs, query)
            
            # Evaluate context sufficiency (simple heuristic)
            context_sufficient = self._evaluate_context_sufficiency(context, query)
            
            feedback_logs.append({
                'iteration': iteration,
                'retrieved_count': len(retrieved_docs),
                'context_length': len(context),
                'sufficient': context_sufficient
            })
            
            if not context_sufficient:
                # Adjust retrieval parameters for next iteration
                current_k = min(current_k + 5, 50)
                iteration += 1
        
        final_context = self.context_manager.enrich_context(all_retrieved_docs, query)
        
        metadata = {
            'iterations': iteration + 1,
            'total_retrieved': len(all_retrieved_docs),
            'feedback_logs': feedback_logs,
            'context_sufficient': context_sufficient
        }
        
        return final_context, metadata
    
    def _evaluate_context_sufficiency(self, context: str, query: QueryContext) -> bool:
        """Evaluate if the context is sufficient (simple heuristic)"""
        # Simple heuristic: check if query terms appear in context
        query_terms = set(query.transformed_query.lower().split())
        context_terms = set(context.lower().split())
        
        coverage = len(query_terms.intersection(context_terms)) / len(query_terms)
        min_context_length = 500  # Minimum context length
        
        return coverage >= 0.7 and len(context) >= min_context_length