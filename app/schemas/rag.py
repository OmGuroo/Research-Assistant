from pydantic import BaseModel
from typing import List

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str

class DocumentListModel(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentListModel]
