from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class RetrievalType(Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "kg"

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class RetrievalResult:
    document: Document
    score: float
    retrieval_type: RetrievalType
    explanation: Optional[str] = None

@dataclass
class QueryContext:
    original_query: str
    transformed_query: str
    intent: str
    entities: List[str]
    hyde_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    confidence: float
    reasoning: str
    retrieval_metrics: Dict[str, Any]

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query_context: QueryContext, top_k: int) -> List[RetrievalResult]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        pass

class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass