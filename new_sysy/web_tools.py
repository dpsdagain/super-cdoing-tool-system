from tool_registry import register_tool
"""
web_tools.py — Web Search and URL Fetching Tools.
"""
import os
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description='The search query to look up on the internet.')

class WebFetchInput(BaseModel):
    url: str = Field(description='The URL of the page to fetch and read.')


@register_tool(name="read_url", description="Fetch and distill web documentation or articles into Markdown.", input_schema=WebFetchInput, is_read_only=True)
def read_url(url: str, prompt: Optional[str] = None) -> str:
    """Fetch and distill content from a URL. Best for reading documentation websites."""
    from web_utils import WebFetcher
    results = WebFetcher.fetch_markdown(url, prompt)
    if 'error' in results:
        return f"Error fetching URL: {results['error']}"
    return f"--- Content from {url} ---\n{results['content']}\n\n[Total Length: {results['length']} chars]"

@register_tool(name="web_search", description="Perform a Google search to find information outside the codebase.", input_schema=WebSearchInput, is_read_only=True)
def web_search(query: str) -> str:
    """Perform a web search using a robust strategy with specialized headers."""
    import requests
    import re
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://duckduckgo.com/', 'DNT': '1', 'Connection': 'keep-alive'}
        url = f'https://html.duckduckgo.com/html/?q={query}'
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result'):
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            if title_tag and title_tag.get('href'):
                title = title_tag.get_text().strip()
                link = title_tag.get('href')
                if 'uddg=' in link:
                    match = re.search('uddg=([^&]+)', link)
                    if match:
                        from urllib.parse import unquote
                        link = unquote(match.group(1))
                snippet = snippet_tag.get_text().strip() if snippet_tag else 'No snippet.'
                results.append(f'Title: {title}\nURL: {link}\nSnippet: {snippet}')
            if len(results) >= 5:
                break
        if not results:
            return f"No results found for '{query}'. The search engine layout may have changed."
        return 'Search Results:\n\n' + '\n\n---\n\n'.join(results)
    except Exception as e:
        return f'Error performing web search: {str(e)}'

def is_safe_url(url: str) -> bool:
    """Check if a URL is safe (not loopback/private)."""
    try:
        from urllib.parse import urlparse
        import ipaddress
        import socket
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'): return False
        if not parsed.hostname: return False
        ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_loopback or ip_obj.is_private:
            return False
        return True
    except Exception:
        return False

@register_tool(name="web_fetch", description="Fetch a URL and extract its main content as Markdown. Ideal for reading documentation.", input_schema=WebFetchInput, is_read_only=True)
def web_fetch(url: str) -> str:
    """Fetch webpage content with improved extraction and noise reduction."""
    if not is_safe_url(url):
        return 'Error: Access to local or private network addresses is restricted.'
    import requests
    import re
    try:
        from bs4 import BeautifulSoup
        has_bs4 = True
    except ImportError:
        has_bs4 = False
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        if has_bs4:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
                element.decompose()
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id=re.compile('content|main|body', re.I))
            if not main_content:
                main_content = soup.find('div', class_=re.compile('content|main|body', re.I))
            text = (main_content or soup).get_text(separator='\n')
        else:
            text = resp.text
            text = re.sub('<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub('<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub('<.*?>', ' ', text)
        return text
    except Exception as e:
        return f'Error fetching webpage: {str(e)}'