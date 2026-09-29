# Multi-Agent Research Assistant

A production-quality Multi-Agent Research Assistant using LangGraph, LangChain, Google Gemini API, FAISS, ChromaDB, and FastAPI.

## Overview
This system uses a multi-agent architecture (Planner, Researcher, Reviewer) to answer complex research questions. It plans the research, gathers/retrieves relevant information via RAG, reviews findings, and produces a high-quality final answer.

## Project Structure
- `app/api`: FastAPI routes and endpoints.
- `app/agents`: Definitions of individual AI agents (Planner, Researcher, Reviewer).
- `app/graph`: LangGraph workflow definitions combining the agents.
- `app/rag`: Retrieval-Augmented Generation logic (FAISS, ChromaDB).
- `app/services`: Business logic and external service integrations.
- `app/schemas`: Pydantic models for data validation and API schemas.
- `app/utils`: Helper functions and shared utilities.

### Requirements
- Python 3.9+
- FastAPI
- LangGraph
- LangChain
- Groq or Google Gemini API Key
- Tavily API Key

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd "Research Assistant"
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory. You can copy the template:
   ```bash
   cp .env.example .env
   ```

### Optional: Session ID
The API supports conversation memory. You can supply an optional `session_id` to persist context across follow-up queries. If omitted, the API will generate and return a new `session_id`.

```bash
curl -X POST "http://localhost:8000/api/v1/research" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Which of these challenges is hardest to solve?",
           "session_id": "YOUR_SESSION_ID"
         }'
```

The memory is process-local and temporary. It will be replaced by persistent storage such as Redis/PostgreSQL during production deployment.
Max memory retention defaults to the last 6 conversational turns to prevent unbounded context growth.

### Get Session History
```bash
curl "http://localhost:8000/api/v1/sessions/YOUR_SESSION_ID"
```

### Delete Session
```bash
curl -X DELETE "http://localhost:8000/api/v1/sessions/YOUR_SESSION_ID"
```
   - `GEMINI_API_KEY="your_google_gemini_api_key_here"`
   - `GEMINI_MODEL_NAME="gemini-3.5-flash"`

## Research Workflow

The core application runs a multi-agent workflow using LangGraph. This workflow is **adaptive**, meaning it loops to refine research based on critique:

1. **Planner**: Converts the user's research query into 3-5 focused search tasks.
2. **Researcher**: Executes real-time web searches via the Tavily API, and semantic document retrieval via ChromaDB. 
3. **Reviewer**: Evaluates the findings against the retrieved sources. If the research is insufficient (e.g., missing evidence, contradictions), the Reviewer provides specific feedback and the **Researcher** executes another targeted search loop to gather more data. (Max limit: 2 iterations).
4. **Final Answer**: Once the Reviewer deems the research sufficient (or the iteration limit is reached), it synthesizes the final output, providing inline citations backed by real source metadata.
   - `[S1]`: Web Source Citation
   - `[D1]`: Document Source Citation

### Document Ingestion (RAG)

The system supports uploading documents to be indexed and semantically retrieved later.
Supported formats: `.txt`, `.md`, `.pdf`, `.docx`.

**Upload a Document:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

**List Indexed Documents:**
```bash
curl http://127.0.0.1:8000/api/v1/documents
```
*Note: Embeddings run locally using the `sentence-transformers/all-MiniLM-L6-v2` model. Vectors are persisted locally in `data/chroma`.*

### Test the Hybrid Research Endpoint

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the major challenges of deploying RAG systems in production according to the web, and what database does Project Orion use according to the uploaded documents?"}'
```

The response includes the final synthesized `answer`, the `plan`, the raw `findings`, the `review` feedback, and partitioned `sources` objects (web and documents). 

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Test the health endpoint:**
   ```bash
   curl http://127.0.0.1:8000/health
   ```

6. **Test the LLM endpoint:**
   Once the server is running, you can test the Gemini integration:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/test-llm \
   -H "Content-Type: application/json" \
   -d '{"prompt": "Explain what RAG is in two sentences."}'
   ```
   *Expected Response:*
   ```json
   {
       "answer": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by grounding their responses in external, up-to-date knowledge bases. It works by first retrieving relevant information from a database based on a user's query, and then feeding that context to the model to generate a more accurate and context-aware answer."
   }
   ```
