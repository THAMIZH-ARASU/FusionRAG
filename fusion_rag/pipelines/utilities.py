import time
from typing import Any, Dict, List

from pipelines.rag_pipeline import RAGPipeline


class RAGPipelineUtils:
    """Utility functions for RAG pipeline management"""
    
    @staticmethod
    def estimate_memory_usage(num_documents: int, avg_doc_length: int, 
                            embedding_dim: int = 384) -> Dict[str, float]:
        """Estimate memory usage for the pipeline"""
        
        # Document storage (MB)
        doc_storage = (num_documents * avg_doc_length * 2) / (1024 * 1024)  # 2 bytes per char
        
        # Embeddings storage (MB)
        embedding_storage = (num_documents * embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float
        
        # FAISS index overhead (MB)
        faiss_overhead = embedding_storage * 0.1  # Roughly 10% overhead
        
        total = doc_storage + embedding_storage + faiss_overhead
        
        return {
            'document_storage_mb': doc_storage,
            'embedding_storage_mb': embedding_storage,
            'faiss_overhead_mb': faiss_overhead,
            'total_estimated_mb': total
        }
    
    @staticmethod
    def benchmark_retrieval_speed(pipeline: RAGPipeline, test_queries: List[str]) -> Dict[str, float]:
        """Benchmark retrieval speed"""
        if not pipeline.indexed:
            raise ValueError("Pipeline must be indexed first")
        
        times = []
        for query in test_queries:
            start_time = time.time()
            pipeline.query(query, use_adaptive_retrieval=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_query_time': sum(times) / len(times),
            'min_query_time': min(times),
            'max_query_time': max(times),
            'total_queries': len(test_queries)
        }
    
    @staticmethod
    def validate_pipeline_config(config: Dict[str, Any]) -> List[str]:
        """Validate pipeline configuration"""
        warnings = []
        
        # Check chunk size vs context length
        if config['chunk_size'] > config['max_context_length'] / 2:
            warnings.append("Chunk size might be too large relative to context length")
        
        # Check overlap
        if config['chunk_overlap'] >= config['chunk_size']:
            warnings.append("Chunk overlap should be smaller than chunk size")
        
        # Check retrieval weights
        weights = config.get('retrieval_weights', {})
        if abs(sum(weights.values()) - 1.0) > 0.01:
            warnings.append("Retrieval weights should sum to 1.0")
        
        return warnings