from core_utils.llm_integration import LLMIntegration
from pipelines.rag_factory import RAGPipelineFactory
from pipelines.rag_pipeline import RAGPipeline


def example_usage():
    """Example of how to use the RAG pipeline"""
    
    # Create a pipeline
    pipeline = RAGPipelineFactory.create_basic_pipeline()
    
    # Load documents
    file_paths = [
        'documents/doc1.pdf',
        'documents/doc2.docx',
        'documents/doc3.txt'
    ]
    pipeline.load_documents(file_paths)
    
    # Index documents
    pipeline.index_documents()
    
    # Query the pipeline
    query = "What is artificial intelligence?"
    result = pipeline.query(query, use_adaptive_retrieval=True)
    
    # Get the context for LLM
    context = result['context']
    
    # Initialize LLM integration
    llm = LLMIntegration('openai', api_key='your-api-key')
    
    # Generate final response
    final_answer = llm.generate_response(query, context)
    
    # Get explainable logs
    logs = pipeline.get_explainable_logs(result)
    
    return {
        'query': query,
        'context': context,
        'answer': final_answer,
        'logs': logs,
        'metadata': result['retrieval_metadata']
    }

def optimize_for_small_context_llm(pipeline: RAGPipeline, max_context_tokens: int = 1000):
    """Optimize pipeline for LLMs with small context windows"""
    
    # Adjust context manager
    pipeline.context_manager.max_context_length = max_context_tokens
    
    # Use more aggressive chunking
    pipeline.text_chunker.chunk_size = min(pipeline.text_chunker.chunk_size, 300)
    pipeline.text_chunker.overlap = min(pipeline.text_chunker.overlap, 50)
    
    # Reduce retrieval count
    pipeline.config['max_adaptive_iterations'] = 1
    
    return pipeline


if __name__ == "__main__":
    # Example configuration for different use cases
    
    print("RAG Pipeline Implementation")
    print("==========================")
    
    # Create different pipeline configurations
    configs = {
        'basic': RAGPipelineFactory.create_basic_pipeline(),
        'advanced': RAGPipelineFactory.create_advanced_pipeline(),
        'lightweight': RAGPipelineFactory.create_lightweight_pipeline()
    }
    
    # Example usage
    for name, pipeline in configs.items():
        print(f"\n{name.upper()} Pipeline Configuration:")
        print(f"- Embedding Model: {pipeline.config['embedding_model']}")
        print(f"- Retrieval Engines: {', '.join(pipeline.config['retrieval_engines'])}")
        print(f"- Chunk Size: {pipeline.config['chunk_size']}")
        print(f"- Max Context Length: {pipeline.config['max_context_length']}")
        print(f"- Uses HyDE: {pipeline.config['use_hyde']}")
    
    print("\nTo use the pipeline:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create pipeline: pipeline = RAGPipelineFactory.create_basic_pipeline()")
    print("3. Load documents: pipeline.load_documents(['path/to/docs'])")
    print("4. Index documents: pipeline.index_documents()")
    print("5. Query: result = pipeline.query('your question')")
    print("6. Get context: context = result['context']")
    print("7. Use with your LLM of choice!")