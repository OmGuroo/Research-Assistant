from pydantic import BaseModel, Field
from app.services.llm_service import generate_structured_response
from app.memory.models import Session
from app.logger import logger
import time

class ContextualizedQuery(BaseModel):
    is_follow_up: bool = Field(description="True if the query relies on previous conversation context")
    standalone_query: str = Field(description="The rewritten standalone query that incorporates necessary context, or the original query if no context is needed")

CONTEXTUALIZER_PROMPT = """
You are a context resolution assistant. Your job is to determine if a user's query is a follow-up to previous research, and if so, rewrite it into a self-contained, standalone query that can be executed independently.

PREVIOUS RESEARCH SUMMARY:
{summary}

RECENT CONVERSATION HISTORY:
{history}

NEW USER QUERY:
{query}

Determine if the new user query relies on the previous context.
If it does, rewrite it to be completely standalone. Preserve the exact intent of the user's question, but include the subject matter explicitly.
If it is completely unrelated, is_follow_up should be false and standalone_query should be exactly the new user query.
"""

async def contextualize_query(query: str, session: Session, request_id: str) -> tuple[str, bool, dict]:
    """
    Returns (standalone_query, is_follow_up, metadata_dict).
    """
    if not session.research_history:
        return query, False, {"in": 0, "out": 0, "latency": 0.0}
        
    latest_research = session.research_history[-1]
    summary = latest_research.summary
    
    # Format history
    history_str = ""
    for msg in session.messages[-4:]:  # last 2 turns
        history_str += f"{msg.role.upper()}: {msg.content}\n"
        
    prompt = CONTEXTUALIZER_PROMPT.format(summary=summary, history=history_str, query=query)
    
    start_time = time.time()
    try:
        result, in_tok, out_tok = await generate_structured_response(prompt, ContextualizedQuery)
        latency = round(time.time() - start_time, 2)
        logger.info(f"[{request_id}] Contextualizer finished in {latency}s. Follow-up: {result.is_follow_up}. Rewritten: '{result.standalone_query}'")
        return result.standalone_query, result.is_follow_up, {"in": in_tok, "out": out_tok, "latency": latency}
    except Exception as e:
        logger.warning(f"[{request_id}] Contextualizer failed: {e}. Falling back to original query.")
        return query, False, {"in": 0, "out": 0, "latency": round(time.time() - start_time, 2)}
