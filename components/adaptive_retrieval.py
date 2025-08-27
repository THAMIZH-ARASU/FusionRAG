from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from core.interfaces import QueryContext, RetrievalResult, Document
from core.logger import RAGLogger
from components.orchestrator import HybridRetrievalOrchestrator
from components.filter import MultiFactedFilter
from components.context_enricher import ContextualChunkEnricher

@dataclass
class ContextQualityMetrics:
    coverage_score: float  # How well the context covers the query
    diversity_score: float  # Diversity of sources and perspectives
    coherence_score: float  # Internal consistency
    answerability_score: float  # Likelihood that context can answer query
    overall_score: float

class AdaptiveRetrievalLoop:
    """Implements adaptive retrieval with feedback loops."""
    
    def __init__(
        self,
        orchestrator: HybridRetrievalOrchestrator,
        filter_system: MultiFactedFilter,
        enricher: ContextualChunkEnricher,
        config
    ):
        self.orchestrator = orchestrator
        self.filter_system = filter_system
        self.enricher = enricher
        self.config = config
        self.logger = RAGLogger("adaptive_retrieval")
    
    async def retrieve_with_feedback(
        self,
        query_context: QueryContext,
        quality_threshold: float = 0.7
    ) -> Tuple[List[Document], ContextQualityMetrics]:
        """Perform retrieval with adaptive feedback loop."""
        with self.logger.log_operation("adaptive_retrieval", query=query_context.original_query):
            iteration = 0
            best_context = None
            best_metrics = None
            
            while iteration < self.config.adaptive_retrieval_max_iterations:
                iteration += 1
                
                self.logger.logger.info(
                    "adaptive_iteration_start",
                    iteration=iteration,
                    query=query_context.original_query
                )
                
                # Perform retrieval
                retrieval_results = await self.orchestrator.retrieve(query_context)
                
                # Combine all results
                all_results = []
                for retriever_results in retrieval_results.values():
                    all_results.extend(retriever_results)
                
                # Filter and enrich
                filtered_results = await self.filter_system.filter_results(
                    all_results,
                    query_context,
                    max_results=self.config.retrieval.final_top_k
                )
                
                context_docs = await self.enricher.enrich_chunks(
                    filtered_results,
                    query_context,
                    max_tokens=self.config.max_context_length
                )
                
                # Evaluate context quality
                quality_metrics = await self._evaluate_context_quality(
                    context_docs,
                    query_context
                )
                
                self.logger.logger.info(
                    "context_quality_evaluation",
                    iteration=iteration,
                    overall_score=quality_metrics.overall_score,
                    coverage=quality_metrics.coverage_score,
                    diversity=quality_metrics.diversity_score
                )
                
                # Check if quality is sufficient
                if quality_metrics.overall_score >= quality_threshold:
                    self.logger.logger.info(
                        "quality_threshold_met",
                        iteration=iteration,
                        score=quality_metrics.overall_score
                    )
                    return context_docs, quality_metrics
                
                # Store best result so far
                if best_metrics is None or quality_metrics.overall_score > best_metrics.overall_score:
                    best_context = context_docs
                    best_metrics = quality_metrics
                
                # Apply feedback to improve next iteration
                if iteration < self.config.adaptive_retrieval_max_iterations:
                    query_context = await self._apply_feedback(
                        query_context,
                        quality_metrics,
                        context_docs
                    )
            
            # Return best result if threshold not met
            self.logger.logger.info(
                "max_iterations_reached",
                best_score=best_metrics.overall_score if best_metrics else 0
            )
            return best_context or [], best_metrics or ContextQualityMetrics(0, 0, 0, 0, 0)
    
    async def _evaluate_context_quality(
        self,
        context_docs: List[Document],
        query_context: QueryContext
    ) -> ContextQualityMetrics:
        """Evaluate the quality of retrieved context."""
        if not context_docs:
            return ContextQualityMetrics(0, 0, 0, 0, 0)
        
        # Coverage score: how well does context cover query terms
        coverage_score = await self._calculate_coverage_score(context_docs, query_context)
        
        # Diversity score: variety in sources and content
        diversity_score = await self._calculate_diversity_score(context_docs)
        
        # Coherence score: internal consistency of information
        coherence_score = await self._calculate_coherence_score(context_docs)
        
        # Answerability score: likelihood context can answer query
        answerability_score = await self._calculate_answerability_score(context_docs, query_context)
        
        # Overall score (weighted average)
        overall_score = (
            coverage_score * 0.3 +
            diversity_score * 0.2 +
            coherence_score * 0.2 +
            answerability_score * 0.3
        )
        
        return ContextQualityMetrics(
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            coherence_score=coherence_score,
            answerability_score=answerability_score,
            overall_score=overall_score
        )
    
    async def _calculate_coverage_score(
        self,
        context_docs: List[Document],
        query_context: QueryContext
    ) -> float:
        """Calculate how well context covers the query."""
        query_terms = set(query_context.transformed_query.lower().split())
        entity_terms = set(entity.lower() for entity in query_context.entities)
        all_query_terms = query_terms.union(entity_terms)
        
        if not all_query_terms:
            return 1.0
        
        covered_terms = set()
        for doc in context_docs:
            doc_terms = set(doc.content.lower().split())
            covered_terms.update(all_query_terms.intersection(doc_terms))
        
        coverage_ratio = len(covered_terms) / len(all_query_terms)
        return min(coverage_ratio, 1.0)
    
    async def _calculate_diversity_score(self, context_docs: List[Document]) -> float:
        """Calculate diversity of sources and content."""
        if len(context_docs) <= 1:
            return 0.5
        
        # Source diversity
        sources = set()
        for doc in context_docs:
            if doc.source:
                sources.add(doc.source)
        
        source_diversity = len(sources) / len(context_docs)
        
        # Content diversity (simple measure based on unique words)
        all_words = set()
        doc_word_sets = []
        
        for doc in context_docs:
            doc_words = set(doc.content.lower().split())
            doc_word_sets.append(doc_words)
            all_words.update(doc_words)
        
        # Calculate pairwise overlap
        total_comparisons = 0
        total_overlap = 0
        
        for i in range(len(doc_word_sets)):
            for j in range(i + 1, len(doc_word_sets)):
                intersection = doc_word_sets[i].intersection(doc_word_sets[j])
                union = doc_word_sets[i].union(doc_word_sets[j])
                overlap = len(intersection) / len(union) if union else 0
                total_overlap += overlap
                total_comparisons += 1
        
        content_diversity = 1 - (total_overlap / total_comparisons) if total_comparisons > 0 else 0.5
        
        # Combine source and content diversity
        return (source_diversity + content_diversity) / 2
    
    async def _calculate_coherence_score(self, context_docs: List[Document]) -> float:
        """Calculate internal coherence of the context."""
        if len(context_docs) <= 1:
            return 1.0
        
        # Simple coherence based on common terms between documents
        doc_term_sets = [set(doc.content.lower().split()) for doc in context_docs]
        
        coherence_scores = []
        for i in range(len(doc_term_sets)):
            for j in range(i + 1, len(doc_term_sets)):
                intersection = doc_term_sets[i].intersection(doc_term_sets[j])
                union = doc_term_sets[i].union(doc_term_sets[j])
                coherence = len(intersection) / len(union) if union else 0
                coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    async def _calculate_answerability_score(
        self,
        context_docs: List[Document],
        query_context: QueryContext
    ) -> float:
        """Calculate likelihood that context can answer the query."""
        # Intent-based scoring
        intent_keywords = {
            "definition": ["is", "are", "means", "definition", "defined"],
            "how_to": ["how", "steps", "process", "method", "way"],
            "explanation": ["because", "reason", "why", "cause", "due to"],
            "comparison": ["versus", "compared", "difference", "similar", "unlike"],
            "location_lookup": ["located", "in", "at", "where", "place"],
            "temporal_lookup": ["when", "date", "time", "year", "period"]
        }
        
        intent = query_context.intent
        expected_keywords = intent_keywords.get(intent, [])
        
        score = 0.5  # Base score
        
        for doc in context_docs:
            doc_content_lower = doc.content.lower()
            
            # Boost score if document contains intent-specific keywords
            for keyword in expected_keywords:
                if keyword in doc_content_lower:
                    score += 0.1
            
            # Boost score for question words being answered
            if intent == "definition" and any(word in doc_content_lower for word in ["is", "are", "means"]):
                score += 0.2
            elif intent == "how_to" and any(word in doc_content_lower for word in ["step", "first", "then", "finally"]):
                score += 0.2
        
        return min(score, 1.0)
    
    async def _apply_feedback(
        self,
        query_context: QueryContext,
        quality_metrics: ContextQualityMetrics,
        context_docs: List[Document]
    ) -> QueryContext:
        """Apply feedback to improve query for next iteration."""
        feedback_applied = []
        
        # If coverage is low, expand query with related terms
        if quality_metrics.coverage_score < 0.5:
            # Extract important terms from existing context
            important_terms = self._extract_important_terms(context_docs)
            if important_terms:
                expanded_query = f"{query_context.transformed_query} {' '.join(important_terms[:3])}"
                query_context.transformed_query = expanded_query
                feedback_applied.append("expanded_query")
        
        # If diversity is low, modify retrieval strategy
        if quality_metrics.diversity_score < 0.4:
            # This would typically involve adjusting retrieval parameters
            # For now, we'll add diversity-promoting terms
            query_context.metadata["boost_diversity"] = True
            feedback_applied.append("boost_diversity")
        
        # If answerability is low, refine intent detection
        if quality_metrics.answerability_score < 0.5:
            # Re-analyze intent with more context
            refined_intent = await self._refine_intent(query_context, context_docs)
            if refined_intent != query_context.intent:
                query_context.intent = refined_intent
                feedback_applied.append("refined_intent")
        
        self.logger.logger.info(
            "feedback_applied",
            original_query=query_context.original_query,
            feedback_types=feedback_applied,
            new_transformed_query=query_context.transformed_query
        )
        
        return query_context
    
    def _extract_important_terms(self, context_docs: List[Document]) -> List[str]:
        """Extract important terms from context for query expansion."""
        # Simple TF-IDF-like approach
        term_counts = {}
        doc_count = len(context_docs)
        
        for doc in context_docs:
            doc_terms = set(doc.content.lower().split())
            for term in doc_terms:
                if len(term) > 3 and term.isalpha():  # Filter short and non-alphabetic terms
                    term_counts[term] = term_counts.get(term, 0) + 1
        
        # Sort by frequency but avoid very common terms
        important_terms = [
            term for term, count in term_counts.items()
            if 1 < count < doc_count * 0.8  # Appear in multiple docs but not all
        ]
        
        return sorted(important_terms, key=lambda x: term_counts[x], reverse=True)[:5]
    
    async def _refine_intent(self, query_context: QueryContext, context_docs: List[Document]) -> str:
        """Refine intent based on available context."""
        # Analyze context to better understand what kind of answer is possible
        combined_content = " ".join(doc.content.lower() for doc in context_docs)
        
        # Update intent based on context content
        if "definition" in combined_content or "means" in combined_content:
            return "definition"
        elif any(word in combined_content for word in ["step", "process", "method"]):
            return "how_to"
        elif any(word in combined_content for word in ["because", "reason", "cause"]):
            return "explanation"
        elif any(word in combined_content for word in ["compare", "versus", "difference"]):
            return "comparison"
        
        return query_context.intent  # Keep original if no clear refinement