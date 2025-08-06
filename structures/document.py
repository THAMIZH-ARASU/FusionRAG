from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """Represents a document in the RAG pipeline"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None