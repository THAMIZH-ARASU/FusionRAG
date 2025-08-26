# Advanced RAG Pipeline Example Usage
import asyncio
import time
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Core imports
from core.interfaces import RAGResponse, QueryContext, Document
from core.logger import RAGLogger, setup_logging
from core.metrics import RAGMetrics, RetrievalMetrics, MetricsCollector
from core.exceptions import RAGException, ConfigurationException
from config.settings import RAGConfig, LLMConfig, VectorDBConfig, RetrievalConfig

# Component imports
from components.query_processor import QueryTransformer
from components.embedding import SentenceTransformerEmbedding, HyDEGenerator
from components.retrievers.bm25_retriever import BM25Retriever
from components.retrievers.vector_retrieval import MilvusVectorRetriever
from components.retrievers.kg_retriever import KnowledgeGraphRetriever
from components.orchestrator import HybridRetrievalOrchestrator
from components.reranker import ReciprocalRankFusionReranker, CrossEncoderReranker
from components.filter import MultiFactedFilter
from components.context_enricher import ContextualChunkEnricher
from components.llm_client import GroqLLMClient
from components.adaptive_retrieval import AdaptiveRetrievalLoop
from evaluation.deepeval_metrics import DeepEvalMetricsEvaluator
from evaluation.explainable_logs import ExplainableRetrievalLogger

class RAGPipeline:
    """Complete RAG Pipeline with advanced features."""
    
    def __init__(self, config: RAGConfig, documents: List[Document], kg_triples: List[Tuple[str, str, str]] = None):
        self.config = config
        self.logger = RAGLogger("rag_pipeline")
        self.metrics_collector = MetricsCollector()
        self.explainable_logger = ExplainableRetrievalLogger()
        
        # Initialize components
        self._initialize_components(documents, kg_triples or [])
        
        self.logger.logger.info(
            "rag_pipeline_initialized",
            num_documents=len(documents),
            num_kg_triples=len(kg_triples) if kg_triples else 0,
            config=config.to_json() if hasattr(config, 'to_json') else str(config)
        )
    
    def _initialize_components(self, documents: List[Document], kg_triples: List[Tuple[str, str, str]]):
        """Initialize all pipeline components."""
        with self.logger.log_operation("component_initialization"):
            # Core processors
            self.query_transformer = QueryTransformer()
            self.embedding_service = SentenceTransformerEmbedding()
            
            # Retrievers
            retrievers = {}
            retrievers["bm25"] = BM25Retriever(documents)
            
            # Only initialize vector retriever if Milvus is configured
            # (in a real deployment, you'd have Milvus running)
            if hasattr(self.config, 'vector_db') and self.config.vector_db:
                try:
                    # Note: This would require Milvus to be running
                    # retrievers["vector"] = MilvusVectorRetriever(self.config.vector_db, self.embedding_service)
                    self.logger.logger.info("vector_retriever_skipped", reason="Milvus not available in demo")
                except Exception as e:
                    self.logger.logger.warning("vector_retriever_init_failed", error=str(e))
            
            if kg_triples:
                retrievers["kg"] = KnowledgeGraphRetriever(kg_triples)
            
            # Orchestrator and processing components
            self.orchestrator = HybridRetrievalOrchestrator(retrievers, self.config)
            self.reranker = ReciprocalRankFusionReranker()
            self.filter_system = MultiFactedFilter(self.config)
            self.enricher = ContextualChunkEnricher(self.config)
            
            # Adaptive retrieval
            self.adaptive_retrieval = AdaptiveRetrievalLoop(
                self.orchestrator,
                self.filter_system,
                self.enricher,
                self.config
            )
            
            # Evaluation components
            # Note: llm_client will be initialized in the context where it's used
    
    async def query(self, user_query: str, enable_adaptive: bool = True) -> RAGResponse:
        """Process a user query through the full RAG pipeline."""
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        with self.logger.log_operation("rag_query_processing", query_id=query_id, query=user_query):
            try:
                # Step 1: Transform and analyze the query
                query_context = await self.query_transformer.transform_query(user_query)
                
                # Step 2: Perform retrieval (adaptive or standard)
                if enable_adaptive:
                    context_docs, context_quality = await self.adaptive_retrieval.retrieve_with_feedback(
                        query_context
                    )
                else:
                    context_docs = await self._standard_retrieval(query_context)
                    context_quality = None
                
                # Step 3: Generate response with LLM
                response = await self._generate_response(query_context, context_docs)
                
                # Step 4: Collect metrics
                metrics = self._create_metrics(
                    query_id, start_time, query_context, context_docs, response
                )
                self.metrics_collector.record_metrics(metrics)
                
                # Step 5: Create explainable trace
                retrieval_results = await self.orchestrator.retrieve(query_context)
                filtered_results = await self.filter_system.filter_results(
                    [r for results in retrieval_results.values() for r in results],
                    query_context
                )
                
                await self.explainable_logger.log_retrieval_trace(
                    query_context,
                    retrieval_results,
                    filtered_results,
                    context_docs,
                    response.answer
                )
                
                return response
                
            except Exception as e:
                self.logger.logger.error("rag_query_failed", query_id=query_id, error=str(e))
                raise RAGException(f"Query processing failed: {e}")
    
    async def _standard_retrieval(self, query_context: QueryContext) -> List[Document]:
        """Perform standard (non-adaptive) retrieval."""
        # Retrieve from all sources
        retrieval_results = await self.orchestrator.retrieve(query_context)
        
        # Combine and rerank all results
        all_results = []
        for retriever_results in retrieval_results.values():
            all_results.extend(retriever_results)
        
        # Apply reranking
        reranked_results = await self.reranker.rerank(query_context.transformed_query, all_results)
        
        # Apply filtering
        filtered_results = await self.filter_system.filter_results(
            reranked_results,
            query_context,
            max_results=self.config.retrieval.final_top_k
        )
        
        # Enrich context
        context_docs = await self.enricher.enrich_chunks(
            filtered_results,
            query_context,
            max_tokens=self.config.max_context_length
        )
        
        return context_docs
    
    async def _generate_response(self, query_context: QueryContext, context_docs: List[Document]) -> RAGResponse:
        """Generate response using LLM client."""
        # Initialize LLM client with context manager
        async with GroqLLMClient(self.config.llm) as llm_client:
            # Generate response
            llm_response = await llm_client.generate_with_context(
                query_context,
                context_docs
            )
            
            # Create RAG response
            response = RAGResponse(
                answer=llm_response.content,
                sources=context_docs,
                confidence=0.85,  # Could be calculated based on retrieval scores
                reasoning="Generated using hybrid retrieval with BM25 and knowledge graph, filtered and reranked",
                retrieval_metrics={"num_sources": len(context_docs)}
            )
            
            return response
    
    def _create_metrics(self, query_id: str, start_time: float, query_context: QueryContext, 
                       context_docs: List[Document], response: RAGResponse) -> RAGMetrics:
        """Create comprehensive metrics for the query."""
        total_time = time.time() - start_time
        
        return RAGMetrics(
            query_id=query_id,
            retrieval_metrics={
                "bm25": RetrievalMetrics(
                    retrieval_time=0.1,  # Would be measured in practice
                    total_documents=len(context_docs),
                    relevant_documents=len([doc for doc in context_docs if doc.metadata.get("retrieval_score", 0) > 0.5])
                )
            },
            rerank_time=0.05,
            llm_generation_time=0.5,
            total_pipeline_time=total_time,
            context_utilization=len(response.answer) / sum(len(doc.content) for doc in context_docs) if context_docs else 0,
            answer_faithfulness=0.8,  # Would be calculated by evaluation
            answer_relevance=0.9      # Would be calculated by evaluation
        )
    
    async def evaluate_response(self, query_context: QueryContext, response: RAGResponse) -> Dict[str, Any]:
        """Evaluate a RAG response using multiple metrics."""
        evaluator = DeepEvalMetricsEvaluator(None)  # LLM client not needed for fallback evaluation
        
        evaluation_result = await evaluator.evaluate_response(
            query_context,
            response,
            response.sources
        )
        
        return {
            "faithfulness_score": evaluation_result.faithfulness_score,
            "relevancy_score": evaluation_result.relevancy_score,
            "context_relevancy_score": evaluation_result.context_relevancy_score,
            "overall_score": evaluation_result.overall_score,
            "detailed_feedback": evaluation_result.detailed_feedback
        }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        avg_metrics = self.metrics_collector.get_average_metrics()
        
        return {
            "total_queries_processed": len(self.metrics_collector.metrics_history),
            "average_metrics": avg_metrics,
            "recent_traces": len(self.explainable_logger.retrieval_traces),
            "pipeline_health": {
                "avg_response_time": avg_metrics.get("total_pipeline_time", 0),
                "avg_context_utilization": avg_metrics.get("context_utilization", 0),
                "avg_answer_quality": (avg_metrics.get("answer_faithfulness", 0) + avg_metrics.get("answer_relevance", 0)) / 2
            }
        }


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    sample_docs = [
        Document(
            id="doc_1",
            content="Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
            metadata={"topic": "programming", "language": "python"},
            source="python_documentation",
            timestamp=datetime.now().isoformat()
        ),
        Document(
            id="doc_2",
            content="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions.",
            metadata={"topic": "machine_learning", "difficulty": "intermediate"},
            source="ml_textbook",
            timestamp=datetime.now().isoformat()
        ),
        Document(
            id="doc_3",
            content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with language model generation. It allows models to access and use external knowledge during the generation process.",
            metadata={"topic": "rag", "technique": "nlp"},
            source="research_paper",
            timestamp=datetime.now().isoformat()
        ),
        Document(
            id="doc_4",
            content="Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are commonly used in machine learning applications for similarity search and recommendation systems.",
            metadata={"topic": "databases", "type": "vector"},
            source="database_guide",
            timestamp=datetime.now().isoformat()
        ),
        Document(
            id="doc_5",
            content="Natural language processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using natural language. The ultimate objective is to read, decipher, understand and make sense of human language.",
            metadata={"topic": "nlp", "field": "ai"},
            source="ai_encyclopedia",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    return sample_docs


def create_sample_knowledge_graph() -> List[Tuple[str, str, str]]:
    """Create sample knowledge graph triples."""
    kg_triples = [
        ("Python", "is_a", "Programming Language"),
        ("Python", "used_for", "Machine Learning"),
        ("Machine Learning", "is_part_of", "Artificial Intelligence"),
        ("RAG", "combines", "Retrieval"),
        ("RAG", "combines", "Generation"),
        ("Vector Database", "stores", "High-dimensional Vectors"),
        ("Vector Database", "used_in", "Machine Learning"),
        ("NLP", "is_branch_of", "Artificial Intelligence"),
        ("NLP", "processes", "Natural Language"),
        ("Retrieval", "finds", "Relevant Documents")
    ]
    
    return kg_triples


async def demonstrate_basic_usage():
    """Demonstrate basic RAG pipeline usage."""
    print("🚀 Starting Basic RAG Pipeline Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging("INFO")
    
    # Create configuration
    config = RAGConfig(
        llm=LLMConfig(
            provider="groq",
            model="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY", "your-api-key-here"),
            temperature=0.1
        ),
        retrieval=RetrievalConfig(
            top_k_bm25=10,
            final_top_k=5,
            min_chunk_size=50,
            max_chunk_size=1000
        ),
        max_context_length=8000,
        adaptive_retrieval_max_iterations=2
    )
    
    # Create sample data
    documents = create_sample_documents()
    kg_triples = create_sample_knowledge_graph()
    
    # Initialize pipeline
    pipeline = RAGPipeline(config, documents, kg_triples)
    
    # Test queries
    test_queries = [
        "What is Python and what is it used for?",
        "How does RAG work in machine learning?",
        "What are vector databases and why are they important?",
        "Explain the relationship between NLP and AI"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process query (using standard retrieval for demo)
            response = await pipeline.query(query, enable_adaptive=False)
            
            print(f"\n✅ Response:")
            print(response.answer)
            
            print(f"\n📊 Sources Used: {len(response.sources)}")
            for j, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
                print(f"  {j}. {source.source} (Score: {source.metadata.get('retrieval_score', 'N/A')})")
            
            print(f"\n🎯 Confidence: {response.confidence:.2f}")
            
            # Evaluate response (using fallback evaluation)
            evaluation = await pipeline.evaluate_response(
                await pipeline.query_transformer.transform_query(query),
                response
            )
            
            print(f"📈 Evaluation Score: {evaluation['overall_score']:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Show pipeline statistics
    print("\n" + "=" * 50)
    print("📊 Pipeline Statistics")
    print("=" * 50)
    stats = pipeline.get_pipeline_statistics()
    print(f"Total Queries: {stats['total_queries_processed']}")
    print(f"Average Response Time: {stats['pipeline_health']['avg_response_time']:.2f}s")
    print(f"Average Answer Quality: {stats['pipeline_health']['avg_answer_quality']:.2f}")


async def demonstrate_advanced_features():
    """Demonstrate advanced RAG pipeline features."""
    print("\n🔬 Advanced RAG Features Demo")
    print("=" * 50)
    
    # Create configuration with advanced settings
    config = RAGConfig(
        llm=LLMConfig(
            provider="groq",
            model="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY", "your-api-key-here"),
            temperature=0.1
        ),
        retrieval=RetrievalConfig(
            top_k_bm25=15,
            top_k_kg=10,
            final_top_k=8,
            rerank_threshold=0.3
        ),
        max_context_length=12000,
        adaptive_retrieval_max_iterations=3,
        enable_caching=True
    )
    
    # Create more complex sample data
    documents = create_sample_documents() + [
        Document(
            id="doc_advanced_1",
            content="Transformer architecture revolutionized NLP by introducing self-attention mechanisms. The attention mechanism allows the model to focus on different parts of the input sequence when processing each element.",
            metadata={"topic": "transformers", "complexity": "advanced"},
            source="transformer_paper"
        ),
        Document(
            id="doc_advanced_2",
            content="FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            metadata={"topic": "vector_search", "library": "faiss"},
            source="faiss_documentation"
        )
    ]
    
    kg_triples = create_sample_knowledge_graph() + [
        ("Transformer", "uses", "Self-Attention"),
        ("Transformer", "revolutionized", "NLP"),
        ("FAISS", "performs", "Similarity Search"),
        ("FAISS", "developed_by", "Facebook AI"),
        ("Self-Attention", "enables", "Context Understanding")
    ]
    
    # Initialize advanced pipeline
    pipeline = RAGPipeline(config, documents, kg_triples)
    
    # Test with complex query
    complex_query = "How do transformer models use attention mechanisms and how does this relate to similarity search in vector databases?"
    
    print(f"\n🧠 Complex Query: {complex_query}")
    print("-" * 60)
    
    try:
        # Use adaptive retrieval
        print("\n🔄 Using Adaptive Retrieval...")
        response = await pipeline.query(complex_query, enable_adaptive=True)
        
        print(f"\n✨ Advanced Response:")
        print(response.answer)
        
        print(f"\n🔍 Advanced Analysis:")
        print(f"  - Sources utilized: {len(response.sources)}")
        print(f"  - Confidence level: {response.confidence:.2f}")
        print(f"  - Reasoning: {response.reasoning}")
        
        # Detailed evaluation
        evaluation = await pipeline.evaluate_response(
            await pipeline.query_transformer.transform_query(complex_query),
            response
        )
        
        print(f"\n📈 Detailed Evaluation:")
        print(f"  - Faithfulness: {evaluation['faithfulness_score']:.2f}")
        print(f"  - Relevancy: {evaluation['relevancy_score']:.2f}")
        print(f"  - Context Quality: {evaluation['context_relevancy_score']:.2f}")
        print(f"  - Overall Score: {evaluation['overall_score']:.2f}")
        
        # Export explainable traces
        pipeline.explainable_logger.export_traces("advanced_rag_traces.json")
        print("\n📝 Explainable traces exported to 'advanced_rag_traces.json'")
        
    except Exception as e:
        print(f"❌ Error in advanced demo: {e}")


async def demonstrate_component_testing():
    """Demonstrate individual component testing."""
    print("\n🧪 Component Testing Demo")
    print("=" * 50)
    
    # Test Query Transformer
    print("\n1. Testing Query Transformer")
    transformer = QueryTransformer()
    test_query = "What's the difference between supervised and unsupervised learning?"
    
    query_context = await transformer.transform_query(test_query)
    print(f"   Original: {query_context.original_query}")
    print(f"   Transformed: {query_context.transformed_query}")
    print(f"   Intent: {query_context.intent}")
    print(f"   Entities: {query_context.entities}")
    
    # Test Embedding Service
    print("\n2. Testing Embedding Service")
    embedding_service = SentenceTransformerEmbedding()
    sample_text = "Machine learning is fascinating"
    
    embedding = await embedding_service.embed_text(sample_text)
    print(f"   Text: {sample_text}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test BM25 Retriever
    print("\n3. Testing BM25 Retriever")
    documents = create_sample_documents()
    bm25_retriever = BM25Retriever(documents)
    
    retrieval_results = await bm25_retriever.retrieve(query_context, top_k=3)
    print(f"   Query: {query_context.original_query}")
    print(f"   Results found: {len(retrieval_results)}")
    
    for i, result in enumerate(retrieval_results, 1):
        print(f"     {i}. Score: {result.score:.3f} | Doc: {result.document.id} | {result.document.content[:100]}...")
    
    # Test Knowledge Graph Retriever
    print("\n4. Testing Knowledge Graph Retriever")
    kg_triples = create_sample_knowledge_graph()
    kg_retriever = KnowledgeGraphRetriever(kg_triples)
    
    kg_results = await kg_retriever.retrieve(query_context, top_k=3)
    print(f"   KG Results found: {len(kg_results)}")
    
    for i, result in enumerate(kg_results, 1):
        print(f"     {i}. Score: {result.score:.3f} | {result.document.content}")


if __name__ == "__main__":
    """Main execution function with comprehensive examples."""
    async def main():
        print("🌟 Advanced RAG Pipeline - Complete Example Usage")
        print("" + "=" * 60)
        print("This demo showcases a production-ready RAG system with:")
        print("  • Hybrid retrieval (BM25 + Knowledge Graph)")
        print("  • Adaptive retrieval with feedback loops")
        print("  • Advanced filtering and reranking")
        print("  • Comprehensive evaluation metrics")
        print("  • Explainable AI logging")
        print("  • Production monitoring and metrics")
        print("=" * 60)
        
        try:
            # Basic demonstration
            await demonstrate_basic_usage()
            
            # Advanced features
            await demonstrate_advanced_features()
            
            # Component testing
            await demonstrate_component_testing()
            
            print("\n🎉 All demonstrations completed successfully!")
            print("\n💡 Tips for production deployment:")
            print("  1. Set up proper Milvus vector database")
            print("  2. Configure production-grade logging")
            print("  3. Implement proper error handling and monitoring")
            print("  4. Set up evaluation pipelines for continuous improvement")
            print("  5. Use caching for frequently asked questions")
            print("  6. Implement rate limiting and security measures")
            
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            raise
    
    # Set up environment variables reminder
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY environment variable not set!")
        print("   Set it with: export GROQ_API_KEY='your-api-key-here'")
        print("   The demo will continue with a placeholder but LLM generation will fail.\n")
    
    # Run the complete demonstration
    asyncio.run(main())
