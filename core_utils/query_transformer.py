from typing import Dict, List


class QueryTransformer:
    """Handles query transformation and enhancement"""
    
    def __init__(self):
        self.synonyms = self._load_synonyms()
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary (could be from file or API)"""
        return {
            'artificial intelligence': ['AI', 'machine learning', 'ML'],
            'natural language processing': ['NLP', 'text processing'],
            'computer': ['machine', 'device', 'system'],
        }
    
    def transform(self, query: str) -> str:
        """Transform and enhance the input query"""
        # Basic cleaning
        query = query.lower().strip()
        
        # Expand with synonyms
        expanded_terms = []
        words = query.split()
        
        for word in words:
            expanded_terms.append(word)
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return ' '.join(unique_terms)