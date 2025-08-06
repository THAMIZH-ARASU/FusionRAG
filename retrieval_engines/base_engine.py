from abc import ABC, abstractmethod
from typing import List, Tuple

from structures.document import Document
from structures.query_context import QueryContext


class RetrievalEngine(ABC):
    """Abstract base class for retrieval engines"""
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def retrieve(self, query: QueryContext, k: int = 10) -> List[Tuple[Document, float]]:
        pass