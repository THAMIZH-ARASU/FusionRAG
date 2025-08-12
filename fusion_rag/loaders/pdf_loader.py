from pathlib import Path
from typing import List
from fusion_rag.loaders.base_loader import DocumentLoader
from fusion_rag.structures.document import Document
from fusion_rag.utils.logger import logger

import PyPDF2

class PDFLoader(DocumentLoader):
    """Loader for PDF files"""
    
    def load(self, file_path: str) -> List[Document]:
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            id=f"{Path(file_path).stem}_page_{page_num}",
                            content=text,
                            source=file_path,
                            page_number=page_num,
                            metadata={"file_type": "pdf", "total_pages": len(pdf_reader.pages)}
                        )
                        documents.append(doc)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
        return documents