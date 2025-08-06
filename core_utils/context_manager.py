import re
from typing import List, Tuple
import tiktoken

from structures.document import Document
from structures.query_context import QueryContext


class ContextManager:
    """Manages context window and enrichment"""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def enrich_context(self, retrieved_docs: List[Tuple[Document, float]], 
                      query: QueryContext) -> str:
        """Enrich and format context for LLM"""
        context_parts = []
        current_length = 0
        
        # Sort by relevance score
        sorted_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
        
        for doc, score in sorted_docs:
            # Apply contextual enrichment techniques
            enriched_content = self._apply_contextual_enrichment(doc, query)
            
            # Check if adding this document would exceed context limit
            doc_length = len(self.tokenizer.encode(enriched_content))
            if current_length + doc_length > self.max_context_length:
                # Try to fit a compressed version
                compressed_content = self._compress_content(enriched_content, 
                                                          self.max_context_length - current_length)
                if compressed_content:
                    context_parts.append(compressed_content)
                break
            
            context_parts.append(enriched_content)
            current_length += doc_length
        
        return "\n\n---\n\n".join(context_parts)
    
    def _apply_contextual_enrichment(self, doc: Document, query: QueryContext) -> str:
        """Apply contextual enrichment techniques"""
        content = doc.content
        
        # Add contextual headers
        if doc.metadata.get('file_type'):
            content = f"[Source: {doc.source}, Type: {doc.metadata['file_type']}]\n{content}"
        
        # Highlight relevant segments (simple implementation)
        query_terms = query.transformed_query.lower().split()
        for term in query_terms:
            content = re.sub(f"({re.escape(term)})", r"**\1**", content, flags=re.IGNORECASE)
        
        return content
    
    def _compress_content(self, content: str, max_length: int) -> str:
        """Compress content to fit within token limit"""
        tokens = self.tokenizer.encode(content)
        if len(tokens) <= max_length:
            return content
        
        # Simple truncation - in practice, you'd use more sophisticated compression
        truncated_tokens = tokens[:max_length]
        return self.tokenizer.decode(truncated_tokens)
