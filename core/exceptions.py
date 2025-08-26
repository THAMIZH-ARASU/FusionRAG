class RAGException(Exception):
    """Base exception for RAG pipeline."""
    pass

class RetrievalException(RAGException):
    """Exception raised during retrieval operations."""
    pass

class EmbeddingException(RAGException):
    """Exception raised during embedding operations."""
    pass

class LLMException(RAGException):
    """Exception raised during LLM operations."""
    pass

class ConfigurationException(RAGException):
    """Exception raised for configuration errors."""
    pass