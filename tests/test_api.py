import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_research_empty_query():
    # Send spaces to bypass Pydantic min_length but trigger router .strip() check
    response = client.post("/api/v1/research", json={"query": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "error" in response.json()["detail"]
    assert response.json()["detail"]["error"]["code"] == "INVALID_REQUEST"
