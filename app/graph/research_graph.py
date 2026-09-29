from langgraph.graph import StateGraph, START, END
from app.config import settings
from app.schemas.research import ResearchState
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.reviewer import run_reviewer
from app.agents.final_answer import run_final_answer

def should_continue(state: ResearchState) -> str:
    """Determine whether to continue research or provide final answer."""
    iterations = state.get("research_iterations", 0)
    is_sufficient = state.get("is_sufficient", False)
    
    if iterations >= settings.max_research_iterations:
        return "final_answer"
        
    if is_sufficient:
        return "final_answer"
        
    return "researcher"

def compile_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", run_planner)
    workflow.add_node("researcher", run_researcher)
    workflow.add_node("reviewer", run_reviewer)
    workflow.add_node("final_answer", run_final_answer)
    
    # Define adaptive flow
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "reviewer")
    
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "final_answer": "final_answer",
            "researcher": "researcher"
        }
    )
    
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()

graph = compile_research_graph()

async def run_research_workflow(query: str, request_id: str) -> dict:
    """Executes the complete research workflow using LangGraph."""
    initial_state = {
        "request_id": request_id,
        "query": query,
        "research_iterations": 0,
        "search_results": [],
        "document_results": [],
        "metadata": {
            "request_id": request_id,
            "planner_latency": 0.0,
            "research_latency": 0.0,
            "reviewer_latency": 0.0,
            "final_answer_latency": 0.0,
            "total_latency": 0.0,
            "llm_calls": 0,
            "tavily_searches": 0,
            "document_retrievals": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }
    }
    final_state = await graph.ainvoke(initial_state)
    return final_state
