from pathlib import Path
from fusion_rag.loaders.base_loader import DocumentLoader
from fusion_rag.loaders.docx_loader import DOCXLoader
from fusion_rag.loaders.pdf_loader import PDFLoader
from fusion_rag.loaders.txt_loader import TXTLoader


class DocumentLoaderFactory:
    """Factory for creating document loaders"""
    
    _loaders = {
        '.pdf': PDFLoader,
        '.docx': DOCXLoader,
        '.txt': TXTLoader,
        '.md': TXTLoader,  # Markdown files treated as text
    }
    
    @classmethod
    def get_loader(cls, file_path: str) -> DocumentLoader:
        extension = Path(file_path).suffix.lower()
        loader_class = cls._loaders.get(extension)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {extension}")
        return loader_class()