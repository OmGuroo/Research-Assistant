import time
from app.services.llm_service import generate_structured_response
from app.schemas.research import ResearchPlan, ResearchState
from app.logger import logger

PLANNER_PROMPT = """
You are an AI Research Planner. Your goal is to break down the user's complex research query into 3-5 distinct, focused search tasks.
Each task should be a specific string query that can be executed against a web search engine or document retriever.

User Query: {query}
"""

async def run_planner(state: ResearchState) -> dict:
    """Generates a research plan based on the original query."""
    request_id = state.get("request_id", "unknown")
    logger.info(f"[{request_id}] Planner starting for query: '{state['query'][:50]}...'")
    
    start_time = time.time()
    prompt = PLANNER_PROMPT.format(query=state["query"])
    result, in_tok, out_tok = await generate_structured_response(prompt, ResearchPlan)
    
    latency = round(time.time() - start_time, 2)
    logger.info(f"[{request_id}] Planner finished in {latency}s. Found {len(result.tasks)} tasks.")
    
    metadata = state.get("metadata", {})
    metadata["planner_latency"] += latency
    metadata["llm_calls"] += 1
    metadata["total_input_tokens"] += in_tok
    metadata["total_output_tokens"] += out_tok
    
    return {
        "plan": result.tasks,
        "metadata": metadata
    }
