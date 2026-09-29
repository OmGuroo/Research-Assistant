import time
import uuid
from typing import Dict, Optional
from threading import Lock
from app.memory.models import Session

# In-memory dict for sessions. Not suitable for multi-instance production.
_sessions: Dict[str, Session] = {}
_lock = Lock()

MAX_SESSIONS = 1000
MAX_CONVERSATION_TURNS = 6

def get_session(session_id: Optional[str] = None) -> Session:
    """Gets an existing session or creates a new one."""
    with _lock:
        if not session_id:
            session_id = uuid.uuid4().hex
            
        if session_id not in _sessions:
            # Enforce max sessions bound
            if len(_sessions) >= MAX_SESSIONS:
                # Evict oldest session
                oldest = min(_sessions.values(), key=lambda s: s.updated_at)
                del _sessions[oldest.session_id]
                
            _sessions[session_id] = Session(session_id=session_id)
            
        return _sessions[session_id]

def save_session(session: Session):
    """Saves a session back to the store and enforces boundaries."""
    with _lock:
        session.updated_at = time.time()
        
        # Enforce max conversation turns by truncating older messages
        # Each "turn" is typically a user+assistant message (so 2 messages per turn)
        max_messages = MAX_CONVERSATION_TURNS * 2
        if len(session.messages) > max_messages:
            session.messages = session.messages[-max_messages:]
            
        # Also limit research history size just to be safe
        if len(session.research_history) > MAX_CONVERSATION_TURNS:
            session.research_history = session.research_history[-MAX_CONVERSATION_TURNS:]
            
        _sessions[session.session_id] = session

def delete_session(session_id: str) -> bool:
    """Deletes a session. Returns True if deleted, False if not found."""
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False
