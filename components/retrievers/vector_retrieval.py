from pymilvus import Collection, connections
from typing import List, Optional
from core.interfaces import BaseRetriever, RetrievalResult, QueryContext, Document, RetrievalType
from core.logging import RAGLogger
from core.exceptions import RetrievalException
from config.settings import VectorDBConfig

class MilvusVectorRetriever(BaseRetriever):
    """Milvus-based vector similarity retrieval."""
    
    def __init__(self, config: VectorDBConfig, embedding_service):
        self.config = config
        self.embedding_service = embedding_service
        self.logger = RAGLogger("vector_retriever")
        self.collection: Optional[Collection] = None
        
        self._connect()
    
    def _connect(self):
        """Connect to Milvus."""
        with self.logger.log_operation("milvus_connection", host=self.config.host, port=self.config.port):
            try:
                connections.connect("default", host=self.config.host, port=self.config.port)
                self.collection = Collection(name=self.config.collection_name)
                self.collection.load()
            except Exception as e:
                raise RetrievalException(f"Failed to connect to Milvus: {e}")
    
    async def retrieve(self, query_context: QueryContext, top_k: int) -> List[RetrievalResult]:
        """Retrieve documents using vector similarity."""
        with self.logger.log_operation("vector_retrieval", top_k=top_k):
            try:
                # Use HyDE embedding if available, otherwise embed the query
                if query_context.hyde_embedding:
                    query_embedding = query_context.hyde_embedding
                else:
                    query_embedding = await self.embedding_service.embed_text(query_context.transformed_query)
                
                # Search in Milvus
                search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": 10}}
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=["content", "metadata", "source", "timestamp"]
                )
                
                retrieval_results = []
                for hits in results:
                    for hit in hits:
                        doc = Document(
                            id=str(hit.id),
                            content=hit.entity.get("content", ""),
                            metadata=hit.entity.get("metadata", {}),
                            source=hit.entity.get("source"),
                            timestamp=hit.entity.get("timestamp")
                        )
                        
                        retrieval_results.append(RetrievalResult(
                            document=doc,
                            score=1.0 - hit.distance,  # Convert distance to similarity
                            retrieval_type=RetrievalType.VECTOR,
                            explanation=f"Vector similarity score: {1.0 - hit.distance:.4f}"
                        ))
                
                self.logger.logger.info(
                    "vector_retrieval_completed",
                    num_results=len(retrieval_results),
                    top_score=retrieval_results[0].score if retrieval_results else 0
                )
                
                return retrieval_results
                
            except Exception as e:
                raise RetrievalException(f"Vector retrieval failed: {e}")