import time
from app.services.llm_service import generate_structured_response
from app.schemas.research import ReviewFeedback, ResearchState
from app.logger import logger

REVIEWER_PROMPT = """
You are a Research Reviewer. Your job is to review the synthesized findings and determine if they adequately answer the user's original query.
Identify any missing areas, contradictions, or weak reasoning.

User Query: {query}
Synthesized Findings: {findings}
"""

async def run_reviewer(state: ResearchState) -> dict:
    """Evaluates the research findings and provides feedback."""
    request_id = state.get("request_id", "unknown")
    start_time = time.time()
    logger.info(f"[{request_id}] Reviewer starting.")
    
    prompt = REVIEWER_PROMPT.format(
        query=state["query"], 
        findings=state.get("findings", "")
    )
    result, in_tok, out_tok = await generate_structured_response(prompt, ReviewFeedback)
    
    # Format the review text clearly for the researcher or final answer
    review_text = result.feedback
    if result.missing_areas:
        review_text += f"\n\nMissing areas to research further:\n- " + "\n- ".join(result.missing_areas)
        
    latency = round(time.time() - start_time, 2)
    logger.info(f"[{request_id}] Reviewer finished in {latency}s. Sufficient: {result.is_sufficient}")
    
    metadata = state.get("metadata", {})
    metadata["reviewer_latency"] += latency
    metadata["llm_calls"] += 1
    metadata["total_input_tokens"] += in_tok
    metadata["total_output_tokens"] += out_tok
        
    return {
        "review": review_text,
        "is_sufficient": result.is_sufficient,
        "targeted_queries": result.targeted_queries,
        "metadata": metadata
    }
