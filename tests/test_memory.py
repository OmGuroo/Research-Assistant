import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_research():
    with patch("app.api.routes.run_research_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "query": "test query",
            "plan": ["task 1"],
            "findings": "test findings",
            "review": "looks good",
            "final_answer": "This is the final answer.",
            "search_results": [],
            "document_results": [],
            "research_iterations": 1,
            "metadata": {}
        }
        yield mock_run

@pytest.fixture
def mock_contextualizer():
    with patch("app.api.routes.contextualize_query", new_callable=AsyncMock) as mock_ctx:
        # returns standalone_query, is_follow_up, metadata
        mock_ctx.return_value = ("test query", False, {"latency": 0.1, "in": 10, "out": 10})
        yield mock_ctx

def test_new_session(mock_research, mock_contextualizer):
    response = client.post("/api/v1/research", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] is not None
    session_id = data["session_id"]
    
    # Retrieve session
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    session_data = get_resp.json()
    assert len(session_data["messages"]) == 2
    assert len(session_data["research_history"]) == 1

def test_existing_session(mock_research, mock_contextualizer):
    # First request
    response1 = client.post("/api/v1/research", json={"query": "first query"})
    session_id = response1.json()["session_id"]
    
    # Follow-up query simulation
    mock_contextualizer.return_value = ("rewritten query", True, {"latency": 0.1, "in": 10, "out": 10})
    response2 = client.post("/api/v1/research", json={"query": "second query", "session_id": session_id})
    assert response2.status_code == 200
    assert response2.json()["session_id"] == session_id
    
    # Check history grew
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    session_data = get_resp.json()
    assert len(session_data["messages"]) == 4
    assert len(session_data["research_history"]) == 2
    assert session_data["research_history"][1]["standalone_query"] == "rewritten query"

def test_session_deletion(mock_research, mock_contextualizer):
    response = client.post("/api/v1/research", json={"query": "delete me"})
    session_id = response.json()["session_id"]
    
    del_resp = client.delete(f"/api/v1/sessions/{session_id}")
    assert del_resp.status_code == 200
    
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 404
