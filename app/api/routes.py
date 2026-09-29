import uuid
import time
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from app.config import settings
from app.logger import logger
from app.schemas.llm import LLMRequest, LLMResponse
from app.schemas.research import ResearchRequest, ResearchResponse, SourcePartitions, ResearchMetadata
from app.schemas.rag import DocumentResponse, DocumentListResponse, DocumentListModel
from app.services.llm_service import generate_response, LLMError
from app.graph.research_graph import run_research_workflow
from app.rag.document_ingestion import ingest_document, DocumentIngestionError
from app.rag.vector_store import get_collection
from app.memory.store import get_session, save_session, delete_session
from app.memory.models import Message, ResearchHistoryEntry
from app.memory.contextualizer import contextualize_query

router = APIRouter()

def create_error(code: str, message: str, request_id: str = None) -> dict:
    return {"error": {"code": code, "message": message}, "request_id": request_id}

@router.post("/api/v1/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.txt', '.docx', '.md')):
        raise HTTPException(status_code=400, detail=create_error("UNSUPPORTED_FILE_TYPE", "Only PDF, TXT, DOCX, and MD files are supported."))
        
    try:
        logger.info(f"Uploading document: {file.filename}")
        result = await ingest_document(file)
        return DocumentResponse(**result)
    except DocumentIngestionError as e:
        logger.error(f"Document ingestion error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=create_error("INGESTION_ERROR", str(e)))
    except Exception as e:
        logger.error(f"Failed to ingest document {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=create_error("INTERNAL_ERROR", "Failed to ingest document."))

@router.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents():
    try:
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        
        doc_map = {}
        for meta in results.get("metadatas", []):
            if not meta:
                continue
            doc_id = meta.get("document_id")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "Unknown"),
                    "source_type": meta.get("source_type", "Unknown"),
                    "chunk_count": 0
                }
            doc_map[doc_id]["chunk_count"] += 1
            
        docs = [DocumentListModel(**data) for data in doc_map.values()]
        return DocumentListResponse(documents=docs)
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=create_error("INTERNAL_ERROR", "Failed to list documents."))

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env
    }

@router.post("/api/v1/test-llm", response_model=LLMResponse)
async def test_llm(request: LLMRequest):
    try:
        answer, _, _ = await generate_response(request.prompt)
        return LLMResponse(answer=answer)
    except LLMError as le:
        raise HTTPException(status_code=le.status_code, detail=create_error("LLM_ERROR", le.message))
    except Exception as e:
        logger.error(f"Test LLM error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error("INTERNAL_ERROR", "Internal server error"))

@router.post("/api/v1/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    request_id = request.request_id or uuid.uuid4().hex
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail=create_error("INVALID_REQUEST", "Query cannot be empty.", request_id))
        
    start_time = time.time()
    logger.info(f"[{request_id}] Received research request: '{request.query[:50]}...'")
    
    # 1. Get or create session
    session = get_session(request.session_id)
    
    # 2. Contextualize query
    standalone_query, is_follow_up, ctx_meta = await contextualize_query(request.query, session, request_id)
    
    try:
        # 3. Run research with the standalone contextualized query
        state_result = await run_research_workflow(standalone_query, request_id)
        
        sources = SourcePartitions(
            web=state_result.get("search_results", []),
            documents=state_result.get("document_results", [])
        )
        
        metadata_dict = state_result.get("metadata", {})
        metadata_dict["request_id"] = request_id
        metadata_dict["contextualizer_latency"] = ctx_meta.get("latency", 0.0)
        metadata_dict["llm_calls"] = metadata_dict.get("llm_calls", 0) + (1 if is_follow_up or ctx_meta.get("latency", 0) > 0 else 0)
        metadata_dict["total_input_tokens"] = metadata_dict.get("total_input_tokens", 0) + ctx_meta.get("in", 0)
        metadata_dict["total_output_tokens"] = metadata_dict.get("total_output_tokens", 0) + ctx_meta.get("out", 0)
        
        metadata_dict["total_latency"] = round(time.time() - start_time, 2)
        metadata = ResearchMetadata(**metadata_dict)
        
        # 4. Save to session
        answer = state_result.get("final_answer", "")
        session.messages.append(Message(role="user", content=request.query))
        session.messages.append(Message(role="assistant", content=answer))
        
        session.research_history.append(ResearchHistoryEntry(
            original_query=request.query,
            standalone_query=standalone_query,
            answer=answer,
            summary=state_result.get("findings", ""),
            sources=sources.model_dump(),
            research_iterations=state_result.get("research_iterations", 0)
        ))
        
        save_session(session)
        
        logger.info(f"[{request_id}] Research completed in {metadata.total_latency}s. Iterations: {state_result.get('research_iterations', 0)}")
        
        return ResearchResponse(
            request_id=request_id,
            session_id=session.session_id,
            query=standalone_query,
            plan=state_result.get("plan", []),
            findings=state_result.get("findings", ""),
            review=state_result.get("review", ""),
            answer=answer,
            sources=sources,
            research_iterations=state_result.get("research_iterations", 0),
            metadata=metadata
        )
    except LLMError as le:
        logger.error(f"[{request_id}] LLMError: {le.message}")
        raise HTTPException(status_code=le.status_code, detail=create_error("LLM_ERROR", le.message, request_id))
    except Exception as e:
        logger.error(f"[{request_id}] Internal server error during research: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error("INTERNAL_ERROR", "An unexpected error occurred during research.", request_id))

@router.get("/api/v1/sessions/{session_id}")
async def get_session_history(session_id: str):
    session = get_session(session_id)
    # If the session was just created because it didn't exist, it will have 0 messages.
    # But we want to return 404 if it truly doesn't exist. We can check if it has history.
    if not session.messages and not session.research_history:
        # Clean up the empty session we just created implicitly
        delete_session(session_id)
        raise HTTPException(status_code=404, detail=create_error("NOT_FOUND", "Session not found"))
    return session

@router.delete("/api/v1/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    if delete_session(session_id):
        return {"status": "success", "message": "Session deleted"}
    raise HTTPException(status_code=404, detail=create_error("NOT_FOUND", "Session not found"))
