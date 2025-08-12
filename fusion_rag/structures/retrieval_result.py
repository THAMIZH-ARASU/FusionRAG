from dataclasses import dataclass, field
from typing import Any, Dict, List

from fusion_rag.structures.document import Document

@dataclass
class RetrievalResult:
    """Result from retrieval process"""
    documents: List[Document]
    scores: List[float]
    retrieval_method: str
    context_window: str
    metadata: Dict[str, Any] = field(default_factory=dict)