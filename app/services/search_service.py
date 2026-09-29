import logging
import time
from typing import List
from tavily import TavilyClient
from app.config import settings
from app.schemas.research import SearchResult
from app.logger import logger

class SearchServiceError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

def get_tavily_client() -> TavilyClient:
    if not settings.tavily_api_key or settings.tavily_api_key == "your_tavily_api_key_here":
        raise SearchServiceError("Tavily API key is missing or not configured.")
    return TavilyClient(api_key=settings.tavily_api_key)

def _categorize_domain(domain: str) -> str:
    if not domain:
        return "unknown"
    domain = domain.lower()
    if domain.endswith(".edu") or "arxiv.org" in domain or "researchgate" in domain:
        return "academic"
    if domain.endswith(".gov"):
        return "government"
    if "github.com" in domain or "stackoverflow.com" in domain or "reddit.com" in domain:
        return "community"
    if "docs." in domain or "documentation" in domain:
        return "documentation"
    return "general_web"

def search_web(query: str, source_id_prefix: str = "S", start_idx: int = 1, max_retries: int = 2) -> List[SearchResult]:
    """
    Performs a web search using Tavily and returns a list of SearchResult objects.
    Includes simple retry logic for transient errors.
    """
    for attempt in range(max_retries):
        try:
            client = get_tavily_client()
            response = client.search(
                query=query,
                search_depth=settings.tavily_search_depth,
                max_results=settings.tavily_max_results,
                include_answer=settings.tavily_include_answer,
                include_raw_content=settings.tavily_include_raw_content,
            )
            
            results = []
            for idx, item in enumerate(response.get("results", [])):
                url = item.get("url", "")
                domain = url.split("/")[2] if "://" in url else None
                source_type = _categorize_domain(domain)
                
                results.append(SearchResult(
                    id=f"{source_id_prefix}{start_idx + idx}",
                    title=item.get("title", ""),
                    url=url,
                    content=item.get("content", ""),
                    score=item.get("score"),
                    domain=domain,
                    source_type=source_type
                ))
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Tavily search transient error, retrying {attempt + 1}/{max_retries}: {e}")
                time.sleep(1.0)
                continue
            logger.error(f"Tavily search failed for query '{query}': {e}")
            return [] # Return empty list on failure to allow graceful degradation
