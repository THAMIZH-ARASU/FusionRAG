from fusion_rag.pipelines.rag_pipeline import RAGPipeline


class RAGPipelineFactory:
    """Factory for creating RAG pipelines with different configurations"""
    
    @staticmethod
    def create_basic_pipeline(**kwargs) -> RAGPipeline:
        """Create a basic RAG pipeline"""
        config = {
            'embedding_engine': 'huggingface',
            'embedding_model': 'all-MiniLM-L6-v2',
            'retrieval_engines': ['bm25', 'vector_db'],
            'retrieval_weights': {'bm25': 0.3, 'vector_db': 0.7},
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunking_strategy': 'recursive',
            'max_context_length': 4000,
            'max_adaptive_iterations': 3,
            'use_hyde': False
        }
        config.update(kwargs)
        return RAGPipeline(config)
    
    @staticmethod
    def create_advanced_pipeline(**kwargs) -> RAGPipeline:
        """Create an advanced RAG pipeline with all features"""
        config = {
            'embedding_engine': 'huggingface',
            'embedding_model': 'all-MiniLM-L6-v2',
            'retrieval_engines': ['bm25', 'vector_db', 'knowledge_graph'],
            'retrieval_weights': {'bm25': 0.2, 'vector_db': 0.6, 'knowledge_graph': 0.2},
            'chunk_size': 800,
            'chunk_overlap': 150,
            'chunking_strategy': 'semantic',
            'max_context_length': 6000,
            'max_adaptive_iterations': 5,
            'use_hyde': True
        }
        config.update(kwargs)
        return RAGPipeline(config)
    
    @staticmethod
    def create_lightweight_pipeline(**kwargs) -> RAGPipeline:
        """Create a lightweight pipeline for resource-constrained environments"""
        config = {
            'embedding_engine': 'huggingface',
            'embedding_model': 'all-MiniLM-L6-v2',
            'retrieval_engines': ['bm25'],
            'retrieval_weights': {'bm25': 1.0},
            'chunk_size': 500,
            'chunk_overlap': 100,
            'chunking_strategy': 'simple',
            'max_context_length': 2000,
            'max_adaptive_iterations': 1,
            'use_hyde': False
        }
        config.update(kwargs)
        return RAGPipeline(config)
