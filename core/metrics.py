from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field
from core.logger import RAGLogger

@dataclass
class RetrievalMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    retrieval_time: float = 0.0
    total_documents: int = 0
    relevant_documents: int = 0

@dataclass
class RAGMetrics:
    query_id: str
    retrieval_metrics: Dict[str, RetrievalMetrics] = field(default_factory=dict)
    rerank_time: float = 0.0
    llm_generation_time: float = 0.0
    total_pipeline_time: float = 0.0
    context_utilization: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "retrieval_metrics": {
                k: {
                    "precision": v.precision,
                    "recall": v.recall,
                    "f1_score": v.f1_score,
                    "retrieval_time": v.retrieval_time,
                    "total_documents": v.total_documents,
                    "relevant_documents": v.relevant_documents
                } for k, v in self.retrieval_metrics.items()
            },
            "rerank_time": self.rerank_time,
            "llm_generation_time": self.llm_generation_time,
            "total_pipeline_time": self.total_pipeline_time,
            "context_utilization": self.context_utilization,
            "answer_faithfulness": self.answer_faithfulness,
            "answer_relevance": self.answer_relevance
        }

class MetricsCollector:
    def __init__(self):
        self.metrics_history: List[RAGMetrics] = []
        self.logger = RAGLogger("metrics_collector")
    
    def record_metrics(self, metrics: RAGMetrics):
        """Record metrics for analysis."""
        self.metrics_history.append(metrics)
        self.logger.logger.info(
            "metrics_recorded",
            metrics=metrics.to_dict()
        )
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Calculate average metrics over recent queries."""
        recent_metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not recent_metrics:
            return {}
        
        total_metrics = {
            "total_pipeline_time": sum(m.total_pipeline_time for m in recent_metrics),
            "llm_generation_time": sum(m.llm_generation_time for m in recent_metrics),
            "rerank_time": sum(m.rerank_time for m in recent_metrics),
            "context_utilization": sum(m.context_utilization for m in recent_metrics),
            "answer_faithfulness": sum(m.answer_faithfulness for m in recent_metrics),
            "answer_relevance": sum(m.answer_relevance for m in recent_metrics)
        }
        
        count = len(recent_metrics)
        return {k: v / count for k, v in total_metrics.items()}