"""
browsing_tool.py

A robust tool for fetching and extracting clean text content from web pages.
Used by the WebResearchAgent to "read" articles deeply.
"""
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrowsingTool:
    """
    Tool for browsing specific URLs and extracting their main content.
    Handles HTML parsing, cleaning, and metadata extraction.
    """
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5,si;q=0.3"
        }
        self.timeout = 10
        
    def scrape_url(self, url: str) -> Dict[str, str]:
        """
        Fetch and scrape a single URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing:
            - title: Page title
            - content: Cleaned text content
            - text_content: Raw text content (for fallback)
            - error: Error message if any
        """
        try:
            logger.info(f"Scraping URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Detect encoding if not present
            if response.encoding is None:
                response.encoding = response.apparent_encoding
                
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script.decompose()
                
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract main content using heuristics
            content = self._extract_main_content(soup)
            
            # Fallback if main extraction fails
            if not content:
                content = soup.get_text(separator="\n", strip=True)
            
            # Clean up content
            cleaned_content = self._clean_text(content)
            
            return {
                "url": url,
                "title": title.strip() if title else "No Title",
                "content": cleaned_content,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "title": "Error",
                "content": "",
                "status": "error",
                "error": str(e)
            }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Attempt to extract the main article content using common structural patterns.
        """
        # 1. Try generic article content tags using 'article' tag
        article = soup.find('article')
        if article:
            return article.get_text(separator="\n\n")
            
        # 2. Try common class names / IDs for main content
        common_ids = ['content', 'main', 'main-content', 'article-body', 'post-content']
        for cid in common_ids:
            element = soup.find(id=re.compile(cid, re.I))
            if element:
                return element.get_text(separator="\n\n")
                
        common_classes = ['content', 'main', 'post-content', 'entry-content', 'article-text']
        for cls in common_classes:
            element = soup.find(class_=re.compile(cls, re.I))
            if element:
                return element.get_text(separator="\n\n")
        
        # 3. Fallback: Find the div/section with the most paragraph text
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 50:  # Only consider substantial paragraphs
                paragraphs.append(text)
                
        return "\n\n".join(paragraphs)
        
    def _clean_text(self, text: str) -> str:
        """Clean extra whitespace and newlines."""
        if not text:
            return ""
            
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

# Singleton instance
_browsing_tool = None

def get_browsing_tool() -> BrowsingTool:
    global _browsing_tool
    if _browsing_tool is None:
        _browsing_tool = BrowsingTool()
    return _browsing_tool
