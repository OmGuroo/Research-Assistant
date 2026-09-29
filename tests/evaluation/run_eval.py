import asyncio
import json
import os
import re
from fastapi.testclient import TestClient
from app.main import app

def run_evaluation():
    print("Starting automated evaluation...")
    
    questions_path = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(questions_path, "r") as f:
        questions = json.load(f)
        
    client = TestClient(app)
    
    total = len(questions)
    passed = 0
    citation_errors = 0
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{total}] Testing query: {q['query']} (Complexity: {q['complexity']})")
        
        response = client.post("/api/v1/research", json={"query": q["query"]})
        
        if response.status_code != 200:
            print(f"  ❌ FAILED: HTTP {response.status_code}")
            print(f"     Details: {response.json()}")
            continue
            
        data = response.json()
        metadata = data.get("metadata", {})
        
        print(f"  ✅ SUCCESS: HTTP 200")
        print(f"     Latency: {metadata.get('total_latency')}s")
        print(f"     Tokens: {metadata.get('total_input_tokens')} in, {metadata.get('total_output_tokens')} out")
        print(f"     Iterations: {data.get('research_iterations')}")
        
        # Verify citations
        answer = data.get("answer", "")
        # Extract all citation tags like [S1], [D2]
        citations_in_answer = set(re.findall(r'\[([SD]\d+)\]', answer))
        
        # Gather available source IDs
        available_ids = set()
        for src in data.get("sources", {}).get("web", []):
            available_ids.add(src["id"])
        for src in data.get("sources", {}).get("documents", []):
            available_ids.add(src["id"])
            
        # Check if any citation is missing from sources
        invalid_citations = citations_in_answer - available_ids
        if invalid_citations:
            print(f"  ⚠️ CITATION ERROR: Found citations in answer that are not in sources: {invalid_citations}")
            citation_errors += 1
        else:
            print(f"  ✅ Citations validated: All {len(citations_in_answer)} citations match retrieved sources.")
            
        passed += 1

    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total queries: {total}")
    print(f"Successful requests: {passed}")
    print(f"Citation errors: {citation_errors}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
