import uuid
import os
import tempfile
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx

from app.config import settings
from app.rag.vector_store import get_collection

class DocumentIngestionError(Exception):
    pass

async def extract_document_pages(file: UploadFile, temp_path: str) -> list[dict]:
    """Extracts text from the document, preserving page numbers when possible."""
    pages = []
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext == ".pdf":
        try:
            reader = PdfReader(temp_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({"text": text, "page": i + 1})
        except Exception as e:
            raise DocumentIngestionError(f"Failed to parse PDF: {str(e)}")
            
    elif ext == ".docx":
        try:
            doc = docx.Document(temp_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            pages.append({"text": full_text, "page": 1})
        except Exception as e:
            raise DocumentIngestionError(f"Failed to parse DOCX: {str(e)}")
            
    elif ext in [".txt", ".md"]:
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                pages.append({"text": f.read(), "page": 1})
        except Exception as e:
            raise DocumentIngestionError(f"Failed to parse text file: {str(e)}")
    else:
        raise DocumentIngestionError(f"Unsupported file format: {ext}")
        
    if not pages:
        raise DocumentIngestionError("Document appears to be empty or unreadable.")
        
    return pages

async def ingest_document(file: UploadFile) -> dict:
    """Ingests a document, chunks it, and stores it in ChromaDB."""
    document_id = str(uuid.uuid4())
    filename = file.filename
    source_type = os.path.splitext(filename)[1].lower()
    
    # Save uploaded file temporarily to process it
    fd, temp_path = tempfile.mkstemp(suffix=source_type)
    try:
        with os.fdopen(fd, 'wb') as f:
            content = await file.read()
            f.write(content)
            
        pages = await extract_document_pages(file, temp_path)
        
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        collection = get_collection()
        chunks_created = 0
        
        for page_data in pages:
            chunks = splitter.split_text(page_data["text"])
            
            if not chunks:
                continue
                
            ids = []
            metadatas = []
            documents = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{document_id}_p{page_data['page']}_c{i}"
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "source_type": source_type,
                    "page": page_data["page"],
                    "chunk_index": i
                })
                
            # Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            chunks_created += len(chunks)
            
        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": chunks_created,
            "status": "indexed"
        }
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
