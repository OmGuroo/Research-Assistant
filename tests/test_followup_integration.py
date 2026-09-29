import pytest
from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

@pytest.mark.integration
def test_followup_integration():
    """
    Optional integration test that hits real LLM and Search APIs.
    Requires Groq/Tavily keys.
    """
    print("Testing Independent Query...")
    q1 = {"query": "What are the main challenges of production RAG systems?"}
    r1 = client.post("/api/v1/research", json=q1)
    assert r1.status_code == 200
    
    data1 = r1.json()
    session_id = data1["session_id"]
    print(f"Session ID created: {session_id}")
    
    print("Testing Follow-Up Query...")
    # Sleep to avoid rate limiting
    time.sleep(5)
    
    q2 = {"query": "Which of these is hardest to solve?", "session_id": session_id}
    r2 = client.post("/api/v1/research", json=q2)
    assert r2.status_code == 200
    
    data2 = r2.json()
    print("Follow up standalone query generated:")
    print(data2["query"])
    
    assert "rag" in data2["query"].lower() or "production" in data2["query"].lower()
