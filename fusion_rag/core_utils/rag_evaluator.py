from typing import Dict, List

from fusion_rag.structures.document import Document


class RAGEvaluator:
    """Evaluation metrics for RAG pipeline"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, retrieved_docs: List[Document], 
                          relevant_docs: List[str]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        retrieved_ids = set(doc.id for doc in retrieved_docs)
        relevant_ids = set(relevant_docs)
        
        if not relevant_ids:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = len(retrieved_ids.intersection(relevant_ids))
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(relevant_ids)
        }
    
    def evaluate_context_relevance(self, context: str, query: str) -> float:
        """Evaluate context relevance to query"""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(context_terms))
        return overlap / len(query_terms)