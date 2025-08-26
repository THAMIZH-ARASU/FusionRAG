import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time
from core.interfaces import BaseRetriever, RetrievalResult, QueryContext, RetrievalType
from core.logging import RAGLogger
from core.metrics import RetrievalMetrics

class HybridRetrievalOrchestrator:
    """Orchestrates multiple retrieval strategies in parallel."""
    
    def __init__(self, retrievers: Dict[str, BaseRetriever], config):
        self.retrievers = retrievers
        self.config = config
        self.logger = RAGLogger("hybrid_orchestrator")
        self.executor = ThreadPoolExecutor(max_workers=len(retrievers))
    
    async def retrieve(self, query_context: QueryContext) -> Dict[str, List[RetrievalResult]]:
        """Run all retrievers in parallel and collect results."""
        with self.logger.log_operation("hybrid_retrieval", query=query_context.original_query):
            retrieval_tasks = {}
            
            # Create tasks for each retriever
            if "bm25" in self.retrievers:
                retrieval_tasks["bm25"] = self._run_retriever(
                    "bm25", query_context, self.config.retrieval.top_k_bm25
                )
            
            if "vector" in self.retrievers:
                retrieval_tasks["vector"] = self._run_retriever(
                    "vector", query_context, self.config.retrieval.top_k_vector
                )
            
            if "kg" in self.retrievers:
                retrieval_tasks["kg"] = self._run_retriever(
                    "kg", query_context, self.config.retrieval.top_k_kg
                )
            
            # Wait for all retrievers to complete
            results = await asyncio.gather(*retrieval_tasks.values(), return_exceptions=True)
            
            # Organize results by retriever type
            retrieval_results = {}
            for i, (retriever_name, task) in enumerate(retrieval_tasks.items()):
                if isinstance(results[i], Exception):
                    self.logger.logger.error(
                        "retriever_failed",
                        retriever=retriever_name,
                        error=str(results[i])
                    )
                    retrieval_results[retriever_name] = []
                else:
                    retrieval_results[retriever_name] = results[i]
            
            self._log_retrieval_stats(retrieval_results)
            return retrieval_results
    
    async def _run_retriever(self, name: str, query_context: QueryContext, top_k: int) -> List[RetrievalResult]:
        """Run a single retriever with error handling."""
        start_time = time.time()
        try:
            results = await self.retrievers[name].retrieve(query_context, top_k)
            duration = time.time() - start_time
            
            self.logger.logger.info(
                "retriever_completed",
                retriever=name,
                duration=duration,
                num_results=len(results)
            )
            return results
        except Exception as e:
            self.logger.logger.error(
                "retriever_error",
                retriever=name,
                error=str(e),
                duration=time.time() - start_time
            )
            raise
    
    def _log_retrieval_stats(self, results: Dict[str, List[RetrievalResult]]):
        """Log statistics for all retrieval results."""
        total_results = sum(len(r) for r in results.values())
        stats = {
            "total_results": total_results,
            "retriever_stats": {
                name: {
                    "count": len(res),
                    "avg_score": sum(r.score for r in res) / len(res) if res else 0,
                    "max_score": max(r.score for r in res) if res else 0
                }
                for name, res in results.items()
            }
        }
        
        self.logger.logger.info("retrieval_stats", **stats)