from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time

class Message(BaseModel):
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)

class ResearchHistoryEntry(BaseModel):
    original_query: str
    standalone_query: str
    answer: str
    summary: str
    sources: Dict[str, Any]
    research_iterations: int
    timestamp: float = Field(default_factory=time.time)

class Session(BaseModel):
    session_id: str
    messages: List[Message] = []
    research_history: List[ResearchHistoryEntry] = []
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
