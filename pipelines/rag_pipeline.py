from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import time
from typing import Any, Dict, List

from core_utils.adaptive_retrieval_loop import AdaptiveRetrievalLoop
from core_utils.context_manager import ContextManager
from core_utils.query_transformer import QueryTransformer
from core_utils.text_chunker import TextChunker
from embedding_engines.base_engine import EmbeddingEngine
from embedding_engines.huggingface_embedding import HuggingFaceEmbedding
from embedding_engines.hyde_embedding import HyDEEmbedding
from loaders.document_loader_factory import DocumentLoaderFactory
from retrieval_engines.base_engine import RetrievalEngine
from retrieval_engines.bm25_engine import BM25Engine
from retrieval_engines.hybrid_engine import HybridRetrievalEngine
from retrieval_engines.knowledge_graph_engine import KnowledgeGraphEngine
from retrieval_engines.vectordb_engine import VectorDBEngine
from structures.query_context import QueryContext
from utils.logger import logger
import faiss

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_transformer = QueryTransformer()
        
        # Initialize embedding engine
        self.embedding_engine = self._create_embedding_engine()
        
        # Initialize retrieval engines
        self.retrieval_engines = self._create_retrieval_engines()
        self.hybrid_engine = HybridRetrievalEngine(
            self.retrieval_engines, 
            config.get('retrieval_weights', {})
        )
        
        # Initialize other components
        self.text_chunker = TextChunker(
            chunk_size=config.get('chunk_size', 1000),
            overlap=config.get('chunk_overlap', 200),
            strategy=config.get('chunking_strategy', 'recursive')
        )
        
        self.context_manager = ContextManager(
            max_context_length=config.get('max_context_length', 4000)
        )
        
        self.adaptive_retrieval = AdaptiveRetrievalLoop(
            self.hybrid_engine,
            self.context_manager,
            max_iterations=config.get('max_adaptive_iterations', 3)
        )
        
        # Document storage
        self.documents = []
        self.indexed = False
    
    def _create_embedding_engine(self) -> EmbeddingEngine:
        """Create embedding engine based on config"""
        engine_type = self.config.get('embedding_engine', 'huggingface')
        
        if engine_type == 'huggingface':
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            base_engine = HuggingFaceEmbedding(model_name)
            
            # Check if HyDE is enabled
            if self.config.get('use_hyde', False):
                llm_client = self.config.get('llm_client')
                return HyDEEmbedding(base_engine, llm_client)
            return base_engine
        else:
            raise ValueError(f"Unsupported embedding engine: {engine_type}")
    
    def _create_retrieval_engines(self) -> Dict[str, RetrievalEngine]:
        """Create retrieval engines based on config"""
        engines = {}
        
        enabled_engines = self.config.get('retrieval_engines', ['bm25', 'vector_db'])
        
        if 'bm25' in enabled_engines:
            engines['bm25'] = BM25Engine()
        
        if 'vector_db' in enabled_engines:
            engines['vector_db'] = VectorDBEngine(self.embedding_engine)
        
        if 'knowledge_graph' in enabled_engines:
            engines['knowledge_graph'] = KnowledgeGraphEngine()
        
        return engines
    
    def load_documents(self, file_paths: List[str]) -> None:
        """Load documents from file paths"""
        all_documents = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {}
            
            for file_path in file_paths:
                try:
                    loader = DocumentLoaderFactory.get_loader(file_path)
                    future = executor.submit(loader.load, file_path)
                    future_to_path[future] = file_path
                except Exception as e:
                    logger.error(f"Error creating loader for {file_path}: {e}")
            
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        # Chunk documents
        self.documents = self.text_chunker.chunk_documents(all_documents)
        logger.info(f"Total chunks created: {len(self.documents)}")
        
        self.indexed = False
    
    def index_documents(self) -> None:
        """Index all loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info("Starting document indexing...")
        start_time = time.time()
        
        self.hybrid_engine.index_documents(self.documents)
        
        self.indexed = True
        end_time = time.time()
        logger.info(f"Indexing completed in {end_time - start_time:.2f} seconds")
    
    def query(self, user_query: str, use_adaptive_retrieval: bool = True) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        if not self.indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")
        
        start_time = time.time()
        
        # Step 1: Query transformation
        transformed_query = self.query_transformer.transform(user_query)
        
        # Step 2: Generate query embeddings
        query_embeddings = self.embedding_engine.embed_query(transformed_query)
        
        query_context = QueryContext(
            original_query=user_query,
            transformed_query=transformed_query,
            embeddings=query_embeddings,
            metadata={'timestamp': time.time()}
        )
        
        # Step 3: Retrieval
        if use_adaptive_retrieval:
            context, retrieval_metadata = self.adaptive_retrieval.retrieve_with_feedback(query_context)
        else:
            # Standard retrieval
            retrieved_docs = self.hybrid_engine.retrieve(query_context, k=10)
            context = self.context_manager.enrich_context(retrieved_docs, query_context)
            retrieval_metadata = {
                'retrieved_count': len(retrieved_docs),
                'context_length': len(context)
            }
        
        # Step 4: Prepare response
        end_time = time.time()
        
        response = {
            'context': context,
            'query_context': {
                'original_query': user_query,
                'transformed_query': transformed_query,
                'processing_time': end_time - start_time
            },
            'retrieval_metadata': retrieval_metadata,
            'pipeline_config': self.config
        }
        
        return response
    
    def get_explainable_logs(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainable logs for the retrieval process"""
        return {
            'query_transformation': {
                'original': query_result['query_context']['original_query'],
                'transformed': query_result['query_context']['transformed_query'],
                'transformation_method': 'synonym_expansion'
            },
            'retrieval_process': query_result['retrieval_metadata'],
            'context_stats': {
                'context_length': len(query_result['context']),
                'estimated_tokens': len(query_result['context'].split()) * 1.3  # Rough estimate
            },
            'pipeline_settings': self.config
        }
    
    def save_index(self, filepath: str) -> None:
        """Save the indexed pipeline to disk"""
        if not self.indexed:
            raise ValueError("No index to save. Call index_documents() first.")
        
        save_data = {
            'documents': self.documents,
            'config': self.config,
            'indexed': self.indexed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save FAISS indices separately
        if 'vector_db' in self.retrieval_engines:
            vector_engine = self.retrieval_engines['vector_db']
            if vector_engine.index is not None:
                faiss.write_index(vector_engine.index, f"{filepath}.faiss")
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load a saved pipeline from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data['documents']
        self.config = save_data['config']
        self.indexed = save_data['indexed']
        
        # Rebuild engines
        self.retrieval_engines = self._create_retrieval_engines()
        self.hybrid_engine = HybridRetrievalEngine(
            self.retrieval_engines,
            self.config.get('retrieval_weights', {})
        )
        
        # Load FAISS index if exists
        faiss_path = f"{filepath}.faiss"
        if os.path.exists(faiss_path) and 'vector_db' in self.retrieval_engines:
            vector_engine = self.retrieval_engines['vector_db']
            vector_engine.documents = self.documents
            vector_engine.index = faiss.read_index(faiss_path)
        
        logger.info(f"Pipeline loaded from {filepath}")