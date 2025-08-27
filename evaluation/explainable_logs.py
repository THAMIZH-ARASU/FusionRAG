from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from core.logger import RAGLogger
from core.interfaces import QueryContext, RetrievalResult, Document

class ExplainableRetrievalLogger:
    """Creates human-readable logs for retrieval transparency."""
    
    def __init__(self):
        self.logger = RAGLogger("explainable_logger")
        self.retrieval_traces: List[Dict[str, Any]] = []
    
    async def log_retrieval_trace(
        self,
        query_context: QueryContext,
        retrieval_results: Dict[str, List[RetrievalResult]],
        filtered_results: List[RetrievalResult],
        final_context: List[Document],
        response: str,
        evaluation_metrics: Optional[Dict[str, Any]] = None
    ):
        """Create comprehensive retrieval trace for transparency."""
        trace_id = f"trace_{datetime.now().isoformat()}_{hash(query_context.original_query) % 10000}"
        
        trace = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "query_analysis": {
                "original_query": query_context.original_query,
                "transformed_query": query_context.transformed_query,
                "detected_intent": query_context.intent,
                "extracted_entities": query_context.entities,
                "query_metadata": query_context.metadata
            },
            "retrieval_phase": await self._format_retrieval_results(retrieval_results),
            "filtering_phase": await self._format_filtering_results(filtered_results),
            "context_construction": await self._format_context_construction(final_context),
            "response_generation": {
                "generated_response": response,
                "response_length": len(response),
                "contains_citations": "[Document" in response
            }
        }
        
        if evaluation_metrics:
            trace["evaluation_metrics"] = evaluation_metrics
        
        # Store trace
        self.retrieval_traces.append(trace)
        
        # Log summary
        self.logger.logger.info(
            "retrieval_trace_created",
            trace_id=trace_id,
            query=query_context.original_query,
            total_retrieved=sum(len(results) for results in retrieval_results.values()),
            final_context_docs=len(final_context)
        )
        
        return trace_id
    
    async def _format_retrieval_results(self, retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, Any]:
        """Format retrieval results for human readability."""
        formatted_results = {}
        
        for retriever_name, results in retrieval_results.items():
            formatted_results[retriever_name] = {
                "total_results": len(results),
                "score_distribution": {
                    "max_score": max(r.score for r in results) if results else 0,
                    "min_score": min(r.score for r in results) if results else 0,
                    "avg_score": sum(r.score for r in results) / len(results) if results else 0
                },
                "top_results": [
                    {
                        "document_id": result.document.id,
                        "score": result.score,
                        "explanation": result.explanation,
                        "content_preview": result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                        "source": result.document.source,
                        "metadata": result.document.metadata
                    }
                    for result in results[:3]  # Top 3 results
                ]
            }
        
        return formatted_results
    
    async def _format_filtering_results(self, filtered_results: List[RetrievalResult]) -> Dict[str, Any]:
        """Format filtering results for transparency."""
        return {
            "total_after_filtering": len(filtered_results),
            "filtering_summary": {
                "deduplication_applied": True,
                "quality_filtering_applied": True,
                "relevance_threshold_applied": True,
                "diversity_filtering_applied": True
            },
            "final_results": [
                {
                    "rank": i + 1,
                    "document_id": result.document.id,
                    "final_score": result.score,
                    "retrieval_type": result.retrieval_type.value,
                    "explanation": result.explanation,
                    "source": result.document.source
                }
                for i, result in enumerate(filtered_results)
            ]
        }
    
    async def _format_context_construction(self, final_context: List[Document]) -> Dict[str, Any]:
        """Format context construction details."""
        return {
            "total_context_documents": len(final_context),
            "total_context_length": sum(len(doc.content) for doc in final_context),
            "context_documents": [
                {
                    "position": i + 1,
                    "document_id": doc.id,
                    "content_length": len(doc.content),
                    "source": doc.source,
                    "enrichment_applied": doc.metadata.get("enriched", False),
                    "quality_score": doc.metadata.get("quality_score", "N/A")
                }
                for i, doc in enumerate(final_context)
            ]
        }
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a human-readable summary of a retrieval trace."""
        trace = next((t for t in self.retrieval_traces if t["trace_id"] == trace_id), None)
        if not trace:
            return None
        
        return {
            "trace_id": trace_id,
            "query": trace["query_analysis"]["original_query"],
            "intent": trace["query_analysis"]["detected_intent"],
            "entities": trace["query_analysis"]["extracted_entities"],
            "retrieval_summary": {
                retriever: f"{data['total_results']} results (avg score: {data['score_distribution']['avg_score']:.3f})"
                for retriever, data in trace["retrieval_phase"].items()
            },
            "final_context_count": trace["context_construction"]["total_context_documents"],
            "response_length": trace["response_generation"]["response_length"],
            "evaluation_score": trace.get("evaluation_metrics", {}).get("overall_score", "N/A")
        }
    
    def export_traces(self, filename: str):
        """Export all traces to a JSON file for analysis."""
        with open(filename, 'w') as f:
            json.dump(self.retrieval_traces, f, indent=2)
        
        self.logger.logger.info(
            "traces_exported",
            filename=filename,
            trace_count=len(self.retrieval_traces)
        )