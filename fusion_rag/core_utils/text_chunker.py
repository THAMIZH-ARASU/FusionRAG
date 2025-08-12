import re
from typing import List
import tiktoken

from fusion_rag.structures.document import Document


class TextChunker:
    """Handles text chunking with various strategies"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, strategy: str = "recursive"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents based on the selected strategy"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self._chunk_text(doc.content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk,
                    source=doc.source,
                    page_number=doc.page_number,
                    chunk_index=i,
                    metadata={**doc.metadata, "original_doc_id": doc.id}
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text using the selected strategy"""
        if self.strategy == "recursive":
            return self._recursive_chunk(text)
        elif self.strategy == "semantic":
            return self._semantic_chunk(text)
        else:
            return self._simple_chunk(text)
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple character-based chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Find the last sentence boundary within the chunk
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - self.overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """Recursive chunking that tries to maintain semantic boundaries"""
        separators = ['\n\n', '\n', '.', '!', '?', ';', ' ']
        return self._split_text_recursive(text, separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using different separators"""
        if not separators:
            return [text]
        
        separator = separators[0]
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(split) > self.chunk_size:
                    # Split is too large, try next separator
                    sub_chunks = self._split_text_recursive(split, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """Semantic chunking based on sentence embeddings"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        # For now, fall back to simple chunking
        # In a full implementation, you'd use sentence embeddings to group similar sentences
        return self._simple_chunk(text)