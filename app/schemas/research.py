from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict

# API Endpoint Schemas
class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    content: str
    score: Optional[float] = None
    domain: Optional[str] = None
    source_type: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1500)
    request_id: Optional[str] = None
    session_id: Optional[str] = None

class ResearchMetadata(BaseModel):
    request_id: str
    contextualizer_latency: float = 0.0
    planner_latency: float = 0.0
    research_latency: float = 0.0
    reviewer_latency: float = 0.0
    final_answer_latency: float = 0.0
    total_latency: float = 0.0
    llm_calls: int = 0
    tavily_searches: int = 0
    document_retrievals: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class SourcePartitions(BaseModel):
    web: List[SearchResult] = []
    documents: List[SearchResult] = []

class ResearchResponse(BaseModel):
    request_id: str
    session_id: str
    query: str
    plan: List[str]
    findings: str
    review: str
    answer: str
    sources: SourcePartitions
    research_iterations: int = 0
    metadata: ResearchMetadata

# LangGraph State Schema
class ResearchState(TypedDict):
    request_id: str
    query: str
    plan: List[str]
    search_results: List[SearchResult]
    document_results: List[SearchResult]
    findings: str
    review: str
    is_sufficient: bool
    targeted_queries: List[str]
    research_iterations: int
    final_answer: str
    metadata: Dict[str, Any]

# LLM Structured Output Schemas
class ResearchPlan(BaseModel):
    tasks: List[str] = Field(description="A list of focused research tasks to answer the query")

class ResearchFindings(BaseModel):
    findings: str = Field(description="Synthesized research findings covering the research tasks")

class ReviewFeedback(BaseModel):
    missing_areas: List[str] = Field(description="Areas that the research findings missed")
    targeted_queries: List[str] = Field(default=[], description="Targeted search queries to find the missing information")
    feedback: str = Field(description="Detailed feedback on contradictions, weak reasoning, or relevance")
    is_sufficient: bool = Field(description="True if the findings adequately answer the original query")

class FinalAnswer(BaseModel):
    answer: str = Field(description="The final polished response to the user's query")
