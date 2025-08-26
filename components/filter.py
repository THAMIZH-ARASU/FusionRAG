from typing import List, Set, Dict, Any
import re
from datetime import datetime, timedelta
from core.interfaces import RetrievalResult, Document
from core.logging import RAGLogger

class MultiFactedFilter:
    """Advanced filtering system for retrieval results."""
    
    def __init__(self, config):
        self.config = config
        self.logger = RAGLogger("result_filter")
    
    async def filter_results(
        self,
        results: List[RetrievalResult],
        query_context,
        max_results: int = None
    ) -> List[RetrievalResult]:
        """Apply multiple filtering strategies."""
        with self.logger.log_operation("multi_filter", input_count=len(results)):
            if not results:
                return results
            
            # Step 1: Deduplicate
            results = await self._deduplicate(results)
            
            # Step 2: Quality filtering
            results = await self._quality_filter(results)
            
            # Step 3: Relevance threshold
            results = await self._relevance_filter(results)
            
            # Step 4: Diversity filtering
            results = await self._diversity_filter(results)
            
            # Step 5: Final count limit
            if max_results:
                results = results[:max_results]
            
            self.logger.logger.info(
                "filtering_completed",
                final_count=len(results),
                avg_score=sum(r.score for r in results) / len(results) if results else 0
            )
            
            return results
    
    async def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate documents based on content similarity."""
        if not results:
            return results
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Create content hash for deduplication
            content_hash = self._create_content_hash(result.document.content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
            else:
                # If duplicate, keep the one with higher score
                for i, existing in enumerate(unique_results):
                    if self._create_content_hash(existing.document.content) == content_hash:
                        if result.score > existing.score:
                            unique_results[i] = result
                        break
        
        return unique_results
    
    def _create_content_hash(self, content: str) -> str:
        """Create a hash for content similarity checking."""
        # Normalize content for comparison
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        # Use first 200 characters for similarity check
        return normalized[:200]
    
    async def _quality_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Filter out low-quality results."""
        filtered_results = []
        
        for result in results:
            quality_score = self._calculate_quality_score(result.document)
            
            if quality_score > 0.3:  # Quality threshold
                # Update metadata with quality score
                result.document.metadata["quality_score"] = quality_score
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_quality_score(self, document: Document) -> float:
        """Calculate quality score for a document."""
        content = document.content
        score = 1.0
        
        # Length penalty for very short or very long content
        length = len(content)
        if length < self.config.retrieval.min_chunk_size:
            score *= 0.5
        elif length > self.config.retrieval.max_chunk_size * 3:
            score *= 0.7
        
        # Penalize documents with too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', content)) / len(content)
        if special_char_ratio > 0.3:
            score *= 0.6
        
        # Boost documents with proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) >= 2:
            score *= 1.1
        
        # Check for metadata quality indicators
        if document.source and document.source != "unknown":
            score *= 1.1
        
        if document.timestamp:
            try:
                doc_date = datetime.fromisoformat(document.timestamp.replace('Z', '+00:00'))
                days_old = (datetime.now().replace(tzinfo=doc_date.tzinfo) - doc_date).days
                if days_old < 365:  # Boost recent documents
                    score *= 1.2
            except:
                pass
        
        return min(score, 1.0)
    
    async def _relevance_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Filter by relevance threshold."""
        threshold = self.config.retrieval.rerank_threshold
        return [r for r in results if r.score >= threshold]
    
    async def _diversity_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Ensure diversity in results."""
        if len(results) <= 3:
            return results
        
        diverse_results = [results[0]]  # Always keep top result
        
        for result in results[1:]:
            # Check if this result is too similar to existing ones
            is_diverse = True
            for existing in diverse_results:
                similarity = self._calculate_content_similarity(
                    result.document.content,
                    existing.document.content
                )
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
                
                # Limit diverse results to maintain performance
                if len(diverse_results) >= min(len(results), 10):
                    break
        
        return diverse_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0