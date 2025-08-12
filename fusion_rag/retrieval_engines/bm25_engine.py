import re
from typing import List, Tuple

import numpy as np

from fusion_rag.retrieval_engines.base_engine import RetrievalEngine
from fusion_rag.structures.document import Document
from fusion_rag.structures.query_context import QueryContext


class BM25Engine(RetrievalEngine):
    """BM25 keyword-based retrieval engine"""
    
    def __init__(self):
        self.documents = []
        self.doc_frequencies = {}
        self.avg_doc_length = 0
        self.k1 = 1.5
        self.b = 0.75
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for BM25 retrieval"""
        self.documents = documents
        
        # Calculate document frequencies and average document length
        doc_lengths = []
        term_doc_freq = {}
        
        for doc in documents:
            terms = self._tokenize(doc.content)
            doc_lengths.append(len(terms))
            unique_terms = set(terms)
            
            for term in unique_terms:
                if term not in term_doc_freq:
                    term_doc_freq[term] = 0
                term_doc_freq[term] += 1
        
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        self.doc_frequencies = term_doc_freq
    
    def retrieve(self, query: QueryContext, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve documents using BM25"""
        query_terms = self._tokenize(query.transformed_query)
        scores = []
        
        for doc in self.documents:
            score = self._calculate_bm25_score(doc, query_terms)
            scores.append((doc, score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\w+', text.lower())
    
    def _calculate_bm25_score(self, document: Document, query_terms: List[str]) -> float:
        """Calculate BM25 score for a document"""
        doc_terms = self._tokenize(document.content)
        doc_length = len(doc_terms)
        score = 0.0
        
        term_frequencies = {}
        for term in doc_terms:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1
        
        for query_term in query_terms:
            if query_term in term_frequencies:
                tf = term_frequencies[query_term]
                df = self.doc_frequencies.get(query_term, 0)
                
                if df > 0:
                    idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5))
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))
        
        return score