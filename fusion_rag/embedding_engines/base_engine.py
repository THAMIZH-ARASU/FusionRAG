from abc import ABC, abstractmethod
from typing import List

import numpy as np

from structures.document import Document


class EmbeddingEngine(ABC):
    """Abstract base class for embedding engines"""
    
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass