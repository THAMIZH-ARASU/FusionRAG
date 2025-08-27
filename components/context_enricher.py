import re
from typing import List, Dict, Any
from core.interfaces import RetrievalResult, Document
from core.logger import RAGLogger

class ContextualChunkEnricher:
    """Enriches chunks with contextual information for better LLM processing."""
    
    def __init__(self, config):
        self.config = config
        self.logger = RAGLogger("context_enricher")
    
    async def enrich_chunks(
        self,
        results: List[RetrievalResult],
        query_context,
        max_tokens: int = 16000
    ) -> List[Document]:
        """Enrich and prepare chunks for LLM context."""
        with self.logger.log_operation("context_enrichment", num_chunks=len(results)):
            enriched_docs = []
            total_tokens = 0
            
            for i, result in enumerate(results):
                # Enrich the document
                enriched_doc = await self._enrich_single_document(
                    result.document,
                    result,
                    query_context,
                    position=i + 1
                )
                
                # Estimate tokens (rough approximation: 4 chars per token)
                estimated_tokens = len(enriched_doc.content) // 4
                
                if total_tokens + estimated_tokens <= max_tokens:
                    enriched_docs.append(enriched_doc)
                    total_tokens += estimated_tokens
                else:
                    self.logger.logger.info(
                        "context_window_limit_reached",
                        included_docs=len(enriched_docs),
                        excluded_docs=len(results) - len(enriched_docs)
                    )
                    break
            
            return enriched_docs
    
    async def _enrich_single_document(
        self,
        document: Document,
        result: RetrievalResult,
        query_context,
        position: int
    ) -> Document:
        """Enrich a single document with contextual information."""
        # Create contextual header
        header_parts = [f"[Document {position}]"]
        
        # Add source information
        if document.source:
            header_parts.append(f"Source: {document.source}")
        
        # Add timestamp if available
        if document.timestamp:
            header_parts.append(f"Date: {document.timestamp}")
        
        # Add retrieval information
        header_parts.append(f"Retrieved via: {result.retrieval_type.value}")
        header_parts.append(f"Relevance score: {result.score:.3f}")
        
        # Add entity context if available
        entities_in_content = self._extract_entities_from_content(
            document.content,
            query_context.entities
        )
        if entities_in_content:
            header_parts.append(f"Key entities: {', '.join(entities_in_content)}")
        
        # Create header
        header = "\n".join(header_parts)
        
        # Apply semantic chunking if content is too long
        processed_content = await self._apply_semantic_chunking(document.content)
        
        # Apply contextual compression if needed
        if len(processed_content) > self.config.retrieval.max_chunk_size:
            processed_content = await self._apply_contextual_compression(
                processed_content,
                query_context
            )
        
        # Combine header and content
        final_content = f"{header}\n{'-' * 50}\n{processed_content}\n"
        
        # Create enriched document
        enriched_doc = Document(
            id=document.id,
            content=final_content,
            metadata={
                **document.metadata,
                "enriched": True,
                "position": position,
                "retrieval_score": result.score,
                "retrieval_type": result.retrieval_type.value
            },
            source=document.source,
            timestamp=document.timestamp
        )
        
        return enriched_doc
    
    def _extract_entities_from_content(self, content: str, query_entities: List[str]) -> List[str]:
        """Extract entities that appear in the content."""
        content_lower = content.lower()
        found_entities = []
        
        for entity in query_entities:
            if entity.lower() in content_lower:
                found_entities.append(entity)
        
        return found_entities
    
    async def _apply_semantic_chunking(self, content: str) -> str:
        """Apply semantic chunking to preserve meaning."""
        if len(content) <= self.config.retrieval.max_chunk_size:
            return content
        
        # Split by sentences while preserving context
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.retrieval.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Return the most relevant chunk (first one for now)
        return chunks[0] if chunks else content[:self.config.retrieval.max_chunk_size]
    
    async def _apply_contextual_compression(self, content: str, query_context) -> str:
        """Apply contextual compression to retain key information."""
        # Simple extraction of key sentences based on query terms
        sentences = re.split(r'(?<=[.!?])\s+', content)
        query_terms = set(query_context.transformed_query.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms.intersection(sentence_terms))
            scored_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        compressed_content = ""
        for sentence, score in scored_sentences:
            if len(compressed_content) + len(sentence) <= self.config.retrieval.max_chunk_size:
                compressed_content += sentence + " "
            else:
                break
        
        return compressed_content.strip() or content[:self.config.retrieval.max_chunk_size]