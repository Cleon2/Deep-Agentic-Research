from tavily import AsyncTavilyClient
from typing import List, Dict, Any
from deep_research.config import TAVILY_API_KEY

tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


async def tavily_search(
    query: str, 
    max_results: int = 5, 
    search_depth: str = "basic",
    include_raw_content: bool = False
) -> List[Dict[str, Any]]:
    """
    Search using Tavily API
    
    Args:
        query: Search query
        max_results: Number of results to return
        search_depth: "basic" (snippets only) or "advanced" (includes full content)
        include_raw_content: Whether to include raw HTML content
    
    Returns:
        List of search results with title, url, content, score
    """
    try:
        response = await tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=include_raw_content
        )
        
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", ""),
                "raw_content": result.get("raw_content", "") if include_raw_content else ""
            })
        
        return results
    
    except Exception as e:
        print(f"Search error for query '{query}': {e}")
        return []


async def extract_content(url: str) -> str:
    """
    Extract full page content for a specific URL
    Used when snippets aren't sufficient
    """
    try:
        response = await tavily_client.extract(urls=[url])
        if response and "results" in response and len(response["results"]) > 0:
            return response["results"][0].get("raw_content", "")
        return ""
    except Exception as e:
        print(f"Content extraction error for {url}: {e}")
        return ""