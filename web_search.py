"""
Web Search Module for Pocket AI
================================
Free web search and scraping using DuckDuckGo + Trafilatura.
No API keys required.

Usage:
    from web_search import search, scrape, search_and_scrape
    
    # Quick search
    results = search("Python tutorials")
    
    # Scrape a URL
    content = scrape("https://example.com")
    
    # Search and get full content
    results = search_and_scrape("latest AI news", num_results=3)
"""

import time
import warnings
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress the package rename warning
warnings.filterwarnings("ignore", message=".*has been renamed.*")

# --- Dependencies ---
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("[web_search] duckduckgo-search not installed. Run: pip install duckduckgo-search")

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("[web_search] trafilatura not installed. Run: pip install trafilatura")


def search(query: str, max_results: int = 5, region: str = "wt-wt", retries: int = 3) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        region: Region code (default: "wt-wt" for worldwide)
        retries: Number of retry attempts (default: 3)
    
    Returns:
        List of dicts with keys: title, href, body
    """
    if not HAS_DDGS:
        return [{"error": "duckduckgo-search not installed"}]
    
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, region=region))
            
            if results:
                return results
            
            # Empty results, wait and retry
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                return [{"error": f"Search failed: {str(e)}"}]
    
    return []


def scrape(url: str, include_links: bool = False, timeout: int = 10) -> Optional[str]:
    """
    Extract clean text content from a URL.
    
    Args:
        url: The URL to scrape
        include_links: Whether to include hyperlinks in output
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content, or None if extraction failed
    """
    if not HAS_TRAFILATURA:
        return None
    
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        
        text = trafilatura.extract(
            downloaded,
            include_links=include_links,
            include_tables=True,
            include_images=False,
            no_fallback=False
        )
        return text
    except Exception as e:
        return None


def search_and_scrape(
    query: str,
    num_results: int = 3,
    max_content_length: int = 4000,
    parallel: bool = True
) -> list[dict]:
    """
    Search the web and scrape content from top results.
    
    Args:
        query: Search query string
        num_results: Number of results to fetch and scrape (default: 3)
        max_content_length: Max characters of content per result (default: 4000)
        parallel: Whether to scrape URLs in parallel (default: True)
    
    Returns:
        List of dicts with keys: title, url, snippet, content
    """
    # Step 1: Search
    search_results = search(query, max_results=num_results)
    
    if not search_results or "error" in search_results[0]:
        return search_results
    
    # Step 2: Scrape each result
    enriched_results = []
    
    if parallel and len(search_results) > 1:
        # Parallel scraping for speed
        with ThreadPoolExecutor(max_workers=min(5, len(search_results))) as executor:
            future_to_result = {
                executor.submit(scrape, r['href']): r 
                for r in search_results
            }
            
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                content = future.result()
                
                enriched_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "content": (content[:max_content_length] + "...") if content and len(content) > max_content_length else content
                })
    else:
        # Sequential scraping
        for result in search_results:
            content = scrape(result['href'])
            enriched_results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "content": (content[:max_content_length] + "...") if content and len(content) > max_content_length else content
            })
    
    return enriched_results


def format_for_llm(results: list[dict], include_urls: bool = True) -> str:
    """
    Format search results into a string suitable for LLM context.
    
    Args:
        results: List of search results from search_and_scrape()
        include_urls: Whether to include source URLs
    
    Returns:
        Formatted string with all search results
    """
    if not results:
        return "No search results found."
    
    if "error" in results[0]:
        return f"Search error: {results[0]['error']}"
    
    formatted = []
    for i, r in enumerate(results, 1):
        parts = [f"[{i}] {r['title']}"]
        
        if include_urls:
            parts.append(f"    Source: {r['url']}")
        
        if r.get('content'):
            parts.append(f"    Content: {r['content']}")
        elif r.get('snippet'):
            parts.append(f"    Summary: {r['snippet']}")
        
        formatted.append("\n".join(parts))
    
    return "\n\n".join(formatted)


# --- Quick Test ---
if __name__ == "__main__":
    print("Testing web search module...\n")
    
    # Test 1: Basic search
    print("=" * 50)
    print("Test 1: Basic Search")
    print("=" * 50)
    results = search("Python programming language", max_results=3)
    for r in results:
        print(f"- {r.get('title', 'N/A')}")
        print(f"  {r.get('href', 'N/A')[:60]}...")
    
    print("\n")
    
    # Test 2: Search and scrape
    print("=" * 50)
    print("Test 2: Search and Scrape")
    print("=" * 50)
    results = search_and_scrape("what is artificial intelligence", num_results=2)
    print(format_for_llm(results))
