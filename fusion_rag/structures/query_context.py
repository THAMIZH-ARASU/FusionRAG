from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class QueryContext:
    """Context for query processing"""
    original_query: str
    transformed_query: str
    embeddings: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)