from abc import ABC, abstractmethod
from typing import List

from fusion_rag.structures.document import Document


class DocumentLoader(ABC):
    """Abstract base class for document loaders"""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        pass