from enum import Enum


class RetrievalMethod(Enum):
    """Available retrieval methods"""
    KEYWORD_BM25 = "bm25"
    VECTOR_DB = "vector_db"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HYBRID = "hybrid"