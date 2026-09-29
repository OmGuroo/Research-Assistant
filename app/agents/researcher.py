import time
from app.services.llm_service import generate_structured_response
from app.services.search_service import search_web
from app.rag.retriever import retrieve_documents
from app.schemas.research import ResearchFindings, ResearchState, SearchResult
from app.logger import logger

RESEARCHER_PROMPT = """
You are an AI Researcher. Your job is to synthesize findings based on the original query, previous findings, reviewer feedback, and the provided search results (Web and Documents).
Use ONLY the information provided in the search results. Do not invent facts, citations, URLs, or titles.
Clearly distinguish between information retrieved from Web Sources and Document Sources.

Original Query: {query}

Reviewer Feedback (if any):
{review}

Web Sources:
{web_context}

Document Sources:
{doc_context}
"""

async def run_researcher(state: ResearchState) -> dict:
    """Performs web and document searches for each task and synthesizes findings."""
    request_id = state.get("request_id", "unknown")
    start_time = time.time()
    logger.info(f"[{request_id}] Researcher starting.")
    
    web_results: list[SearchResult] = list(state.get("search_results", []))
    doc_results: list[SearchResult] = list(state.get("document_results", []))
    
    seen_web_urls = {res.url for res in web_results}
    seen_doc_ids = {f"{res.title}_{res.content[:20]}" for res in doc_results}
    
    web_idx = len(web_results) + 1
    doc_idx = len(doc_results) + 1
    
    iterations = state.get("research_iterations", 0)
    metadata = state.get("metadata", {})
    
    # Decide which queries to run
    if iterations == 0:
        search_queries = state.get("plan", [])
    else:
        search_queries = state.get("targeted_queries", [])
        if not search_queries:
            search_queries = state.get("plan", [])
            
    new_web_count = 0
    new_doc_count = 0
            
    # 1. Perform web search and document retrieval for each task
    for task in search_queries:
        w_res = search_web(query=task, start_idx=web_idx)
        metadata["tavily_searches"] += 1
        for res in w_res:
            if res.url not in seen_web_urls:
                seen_web_urls.add(res.url)
                res.id = f"S{web_idx}"
                web_results.append(res)
                web_idx += 1
                new_web_count += 1
                
        d_res = retrieve_documents(query=task, k=3, start_idx=doc_idx)
        metadata["document_retrievals"] += 1
        for res in d_res:
            unique_id = f"{res.title}_{res.content[:20]}"
            if unique_id not in seen_doc_ids:
                seen_doc_ids.add(unique_id)
                res.id = f"D{doc_idx}"
                doc_results.append(res)
                doc_idx += 1
                new_doc_count += 1
                
    web_context = "".join(f"[{res.id}] Title: {res.title}\nURL: {res.url}\nContent: {res.content}\n\n" for res in web_results)
    doc_context = "".join(f"[{res.id}] Document: {res.title}\nPage: {res.url}\nContent: {res.content}\n\n" for res in doc_results)
        
    if not web_context:
        web_context = "No relevant web search results were found."
    if not doc_context:
        doc_context = "No relevant document chunks were found."

    prompt = RESEARCHER_PROMPT.format(
        query=state["query"], 
        review=state.get("review", "No feedback yet."),
        web_context=web_context,
        doc_context=doc_context
    )
    result, in_tok, out_tok = await generate_structured_response(prompt, ResearchFindings)
    
    latency = round(time.time() - start_time, 2)
    logger.info(f"[{request_id}] Researcher finished in {latency}s. New sources: {new_web_count} web, {new_doc_count} doc.")
    
    metadata["research_latency"] += latency
    metadata["llm_calls"] += 1
    metadata["total_input_tokens"] += in_tok
    metadata["total_output_tokens"] += out_tok
    
    return {
        "findings": result.findings,
        "search_results": web_results,
        "document_results": doc_results,
        "research_iterations": iterations + 1,
        "metadata": metadata
    }
