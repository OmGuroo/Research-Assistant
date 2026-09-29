import urllib.request
import json
import re
import time

API_URL = "http://127.0.0.1:8000/api/v1/research"

def test_query(test_name, query, expected_min_iterations):
    print(f"\n{'='*50}\nTEST: {test_name}\nQUERY: {query}\n{'='*50}")
    
    req = urllib.request.Request(
        API_URL, 
        data=json.dumps({"query": query}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            iters = data.get("research_iterations", 0)
            
            print(f"Research Iterations: {iters}")
            print(f"Time Taken: {time.time() - start_time:.2f}s")
            
            web_sources = data.get('sources', {}).get('web', [])
            doc_sources = data.get('sources', {}).get('documents', [])
            
            print(f"Web Sources Retrieved: {len(web_sources)}")
            print(f"Document Sources Retrieved: {len(doc_sources)}")
            
            if iters >= expected_min_iterations:
                print("ITERATION CHECK: PASSED")
            else:
                print("ITERATION CHECK: FAILED (Expected more iterations)")
                
            # Citation Validation
            answer = data.get("answer", "")
            citations = re.findall(r'\[([SD]\d+)\]', answer)
            
            valid_ids = {s['id'] for s in web_sources} | {s['id'] for s in doc_sources}
            
            invalid_citations = [c for c in citations if c not in valid_ids]
            
            if invalid_citations:
                print(f"CITATION CHECK: FAILED (Invalid citations found: {invalid_citations})")
            else:
                print(f"CITATION CHECK: PASSED (Found {len(citations)} citations, all valid)")
                
    except Exception as e:
        print(f"ERROR: Request failed: {e}")

    print("Sleeping for 15 seconds to avoid 429 rate limits...")
    time.sleep(15)

if __name__ == "__main__":
    print("Starting Adaptive Workflow Tests...")
    
    # Test Case 1: Simple query, should be sufficient in 1 iteration
    test_query(
        "Case 1 (Simple)", 
        "What is 2+2? Answer in one sentence.", 
        expected_min_iterations=1
    )
    
    # Test Case 2: Highly complex/obscure query designed to trigger reviewer feedback and a second iteration
    test_query(
        "Case 2 (Complex/Insufficient)", 
        "Give me a deeply technical analysis of the latency overhead when implementing Orion-RAG over traditional Naive RAG in enterprise ecosystems. Include specific connection path latency metrics if possible.", 
        expected_min_iterations=2
    )
    
    # Test Case 3: Impossible query (should hit max limit of 2)
    test_query(
        "Case 3 (Max Limit)", 
        "What is the exact secret underlying architecture of the unreleased hypothetical 'GPT-10' model that hasn't been invented yet? Be highly specific with technical specs.", 
        expected_min_iterations=2
    )
