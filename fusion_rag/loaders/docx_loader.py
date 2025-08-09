from pathlib import Path
from typing import List
from loaders.base_loader import DocumentLoader
from structures.document import Document
from utils.logger import logger


class DOCXLoader(DocumentLoader):
    """Loader for DOCX files"""
    
    def load(self, file_path: str) -> List[Document]:
        documents = []
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            
            content = '\n'.join(full_text)
            if content.strip():
                document = Document(
                    id=f"{Path(file_path).stem}",
                    content=content,
                    source=file_path,
                    metadata={"file_type": "docx"}
                )
                documents.append(document)
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
        return documents