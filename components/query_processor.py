import re
import spacy
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from core.interfaces import QueryContext
from core.logger import RAGLogger
from core.exceptions import RAGException

class QueryTransformer:
    """Handles query transformation, normalization, and intent detection."""
    
    def __init__(self):
        self.logger = RAGLogger("query_transformer")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def transform_query(self, original_query: str) -> QueryContext:
        """Transform and enrich the original query."""
        with self.logger.log_operation("query_transformation", query=original_query):
            # Normalize the query
            normalized = self._normalize_query(original_query)
            
            # Extract entities
            entities = await self._extract_entities(normalized)
            
            # Detect intent
            intent = await self._detect_intent(normalized)
            
            # Apply query expansion
            expanded = await self._expand_query(normalized, entities)
            
            return QueryContext(
                original_query=original_query,
                transformed_query=expanded,
                intent=intent,
                entities=entities,
                metadata={"normalized_query": normalized}
            )
    
    def _normalize_query(self, query: str) -> str:
        """Basic query normalization."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations
        abbreviations = {
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "how's": "how is",
            "why's": "why is",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not"
        }
        
        for abbrev, expanded in abbreviations.items():
            query = re.sub(r'\b' + re.escape(abbrev) + r'\b', expanded, query, flags=re.IGNORECASE)
        
        return query
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.append(ent.text)
        
        # Also extract noun phrases as potential entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to reasonable phrases
                entities.append(chunk.text)
        
        return list(set(entities))
    
    async def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        # Simple rule-based intent detection
        if any(word in query_lower for word in ["what is", "what are", "define", "definition"]):
            return "definition"
        elif any(word in query_lower for word in ["how to", "how do", "how can", "steps"]):
            return "how_to"
        elif any(word in query_lower for word in ["why", "reason", "cause", "because"]):
            return "explanation"
        elif any(word in query_lower for word in ["who is", "who are", "who was", "who were"]):
            return "person_lookup"
        elif any(word in query_lower for word in ["where", "location", "place"]):
            return "location_lookup"
        elif any(word in query_lower for word in ["when", "time", "date", "year"]):
            return "temporal_lookup"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summarization"
        else:
            return "general_query"
    
    async def _expand_query(self, query: str, entities: List[str]) -> str:
        """Expand query with synonyms and related terms."""
        # Simple expansion - in production, use word embeddings or knowledge graphs
        synonyms = {
            "company": ["corporation", "business", "organization", "firm"],
            "person": ["individual", "people", "human"],
            "location": ["place", "area", "region", "site"],
            "create": ["make", "build", "develop", "generate"],
            "understand": ["comprehend", "grasp", "know", "learn"]
        }
        
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            if word in synonyms:
                expanded_terms.extend(synonyms[word][:2])  # Add top 2 synonyms
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query