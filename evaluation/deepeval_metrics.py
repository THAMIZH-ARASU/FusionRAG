from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass
try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from core.interfaces import Document, QueryContext, RAGResponse
from core.logging import RAGLogger
from core.metrics import RAGMetrics

@dataclass
class EvaluationResult:
    faithfulness_score: float
    relevancy_score: float
    context_relevancy_score: float
    overall_score: float
    detailed_feedback: Dict[str, Any]

class DeepEvalMetricsEvaluator:
    """DeepEval-based evaluation system for RAG responses."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = RAGLogger("deepeval_evaluator")
        
        if not DEEPEVAL_AVAILABLE:
            self.logger.logger.warning("DeepEval not available. Install with: pip install deepeval")
            self.metrics = None
        else:
            self.metrics = {
                "faithfulness": FaithfulnessMetric(),
                "answer_relevancy": AnswerRelevancyMetric(),
                "context_relevancy": ContextRelevancyMetric()
            }
    
    async def evaluate_response(
        self,
        query_context: QueryContext,
        response: RAGResponse,
        context_documents: List[Document]
    ) -> EvaluationResult:
        """Evaluate RAG response using DeepEval metrics."""
        with self.logger.log_operation("deepeval_evaluation", query=query_context.original_query):
            if not DEEPEVAL_AVAILABLE:
                return await self._fallback_evaluation(query_context, response, context_documents)
            
            # Prepare context for evaluation
            context = [doc.content for doc in context_documents]
            
            # Create test case
            test_case = LLMTestCase(
                input=query_context.original_query,
                actual_output=response.answer,
                retrieval_context=context
            )
            
            # Run evaluations
            results = {}
            detailed_feedback = {}
            
            try:
                # Faithfulness evaluation
                faithfulness_metric = self.metrics["faithfulness"]
                faithfulness_score = await self._run_metric(faithfulness_metric, test_case)
                results["faithfulness"] = faithfulness_score
                detailed_feedback["faithfulness"] = getattr(faithfulness_metric, 'reason', 'No detailed feedback')
                
                # Answer relevancy evaluation
                relevancy_metric = self.metrics["answer_relevancy"]
                relevancy_score = await self._run_metric(relevancy_metric, test_case)
                results["relevancy"] = relevancy_score
                detailed_feedback["answer_relevancy"] = getattr(relevancy_metric, 'reason', 'No detailed feedback')
                
                # Context relevancy evaluation
                context_relevancy_metric = self.metrics["context_relevancy"]
                context_relevancy_score = await self._run_metric(context_relevancy_metric, test_case)
                results["context_relevancy"] = context_relevancy_score
                detailed_feedback["context_relevancy"] = getattr(context_relevancy_metric, 'reason', 'No detailed feedback')
                
            except Exception as e:
                self.logger.logger.error("deepeval_evaluation_failed", error=str(e))
                return await self._fallback_evaluation(query_context, response, context_documents)
            
            # Calculate overall score
            overall_score = (
                results.get("faithfulness", 0) * 0.4 +
                results.get("relevancy", 0) * 0.4 +
                results.get("context_relevancy", 0) * 0.2
            )
            
            evaluation_result = EvaluationResult(
                faithfulness_score=results.get("faithfulness", 0),
                relevancy_score=results.get("relevancy", 0),
                context_relevancy_score=results.get("context_relevancy", 0),
                overall_score=overall_score,
                detailed_feedback=detailed_feedback
            )
            
            self.logger.logger.info(
                "deepeval_completed",
                overall_score=overall_score,
                faithfulness=results.get("faithfulness", 0),
                relevancy=results.get("relevancy", 0)
            )
            
            return evaluation_result
    
    async def _run_metric(self, metric, test_case) -> float:
        """Run a single DeepEval metric."""
        loop = asyncio.get_event_loop()
        # Run in thread pool since DeepEval might be synchronous
        score = await loop.run_in_executor(None, lambda: metric.measure(test_case))
        return float(score)
    
    async def _fallback_evaluation(
        self,
        query_context: QueryContext,
        response: RAGResponse,
        context_documents: List[Document]
    ) -> EvaluationResult:
        """Fallback evaluation when DeepEval is not available."""
        # Simple heuristic-based evaluation
        faithfulness_score = await self._calculate_faithfulness_heuristic(response, context_documents)
        relevancy_score = await self._calculate_relevancy_heuristic(query_context, response)
        context_relevancy_score = await self._calculate_context_relevancy_heuristic(query_context, context_documents)
        
        overall_score = (faithfulness_score * 0.4 + relevancy_score * 0.4 + context_relevancy_score * 0.2)
        
        return EvaluationResult(
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
            context_relevancy_score=context_relevancy_score,
            overall_score=overall_score,
            detailed_feedback={
                "note": "Fallback heuristic evaluation used (DeepEval not available)"
            }
        )
    
    async def _calculate_faithfulness_heuristic(self, response: RAGResponse, context_documents: List[Document]) -> float:
        """Calculate faithfulness using simple heuristics."""
        # Check if response contains citations
        citation_bonus = 0.2 if "[Document" in response.answer else 0
        
        # Check if response claims no information when context is empty
        if not context_documents and "not enough information" in response.answer.lower():
            return 0.9 + citation_bonus
        
        # Simple word overlap between response and context
        response_words = set(response.answer.lower().split())
        context_words = set()
        for doc in context_documents:
            context_words.update(doc.content.lower().split())
        
        if not response_words or not context_words:
            return 0.5
        
        overlap_ratio = len(response_words.intersection(context_words)) / len(response_words)
        return min(overlap_ratio + citation_bonus, 1.0)
    
    async def _calculate_relevancy_heuristic(self, query_context: QueryContext, response: RAGResponse) -> float:
        """Calculate answer relevancy using simple heuristics."""
        query_words = set(query_context.original_query.lower().split())
        response_words = set(response.answer.lower().split())
        
        if not query_words or not response_words:
            return 0.5
        
        # Word overlap
        overlap_ratio = len(query_words.intersection(response_words)) / len(query_words)
        
        # Intent matching bonus
        intent_bonus = 0
        if query_context.intent == "definition" and any(word in response.answer.lower() for word in ["is", "are", "means"]):
            intent_bonus = 0.2
        elif query_context.intent == "how_to" and any(word in response.answer.lower() for word in ["step", "first", "process"]):
            intent_bonus = 0.2
        
        return min(overlap_ratio + intent_bonus, 1.0)
    
    async def _calculate_context_relevancy_heuristic(self, query_context: QueryContext, context_documents: List[Document]) -> float:
        """Calculate context relevancy using simple heuristics."""
        if not context_documents:
            return 0
        
        query_words = set(query_context.transformed_query.lower().split())
        total_relevancy = 0
        
        for doc in context_documents:
            doc_words = set(doc.content.lower().split())
            doc_relevancy = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
            total_relevancy += doc_relevancy
        
        return total_relevancy / len(context_documents)