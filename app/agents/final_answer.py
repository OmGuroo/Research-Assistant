import time
from app.services.llm_service import generate_structured_response
from app.schemas.research import FinalAnswer, ResearchState
from app.logger import logger

FINAL_ANSWER_PROMPT = """
You are an expert synthesizer. Your job is to write the final polished response to the user's query.
Synthesize the original query, the research findings, and the reviewer's feedback into a coherent, high-quality final answer.
IMPORTANT: When making claims based on the research, you MUST use inline source references such as [S1], [S2], etc., corresponding to the provided sources.
Do NOT create fake citations. Use ONLY the source IDs provided in the retrieved sources context.

Original Query: {query}

Retrieved Sources Context:
{sources_context}

Research Findings:
{findings}

Reviewer Feedback:
{review}
"""

async def run_final_answer(state: ResearchState) -> dict:
    """Generates the final synthesized answer with source attribution."""
    request_id = state.get("request_id", "unknown")
    start_time = time.time()
    logger.info(f"[{request_id}] Final Answer generation starting.")
    
    sources_context = "WEB SOURCES:\n"
    for res in state.get("search_results", []):
        sources_context += f"[{res.id}] {res.title} ({res.url})\n"
        
    sources_context += "\nDOCUMENT SOURCES:\n"
    for res in state.get("document_results", []):
        sources_context += f"[{res.id}] {res.title} ({res.url})\n"
        
    prompt = FINAL_ANSWER_PROMPT.format(
        query=state["query"], 
        sources_context=sources_context,
        findings=state.get("findings", ""), 
        review=state.get("review", "")
    )
    result, in_tok, out_tok = await generate_structured_response(prompt, FinalAnswer)
    
    latency = round(time.time() - start_time, 2)
    logger.info(f"[{request_id}] Final Answer finished in {latency}s.")
    
    metadata = state.get("metadata", {})
    metadata["final_answer_latency"] += latency
    metadata["llm_calls"] += 1
    metadata["total_input_tokens"] += in_tok
    metadata["total_output_tokens"] += out_tok
    
    return {
        "final_answer": result.answer,
        "metadata": metadata
    }
