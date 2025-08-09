from pathlib import Path
from typing import List
from loaders.base_loader import DocumentLoader
from structures.document import Document
from utils.logger import logger

class TXTLoader(DocumentLoader):
    """Loader for TXT files"""
    
    def load(self, file_path: str) -> List[Document]:
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():
                    document = Document(
                        id=f"{Path(file_path).stem}",
                        content=content,
                        source=file_path,
                        metadata={"file_type": "txt"}
                    )
                    documents.append(document)
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
        return documents