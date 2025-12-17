"""
text_normalize.py - Text Preprocessing and Normalization

This module provides functions to clean and normalize text before
embedding and indexing. Removes headers, footers, ads, and other noise.
"""
import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Normalize unicode characters and remove invisible characters.
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    return text


def preprocess_for_indexing(text: str) -> str:
    """
    Preprocess text before indexing to Pinecone.
    
    This function removes:
    - Headers and navigation text
    - Footers and copyright notices
    - Advertisements
    - Social media buttons text
    - Repeated whitespace
    - URLs
    - Email addresses
    
    Args:
        text: Raw text from scraping
        
    Returns:
        Cleaned text ready for embedding
    """
    if not text:
        return ""
    
    print("[preprocess] Processing text of length:", len(text))
    
    # Normalize unicode first
    text = unicodedata.normalize('NFKC', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove common header/footer patterns (English)
    header_patterns = [
        r'Home\s*[|>]\s*News',
        r'Share\s*(on)?\s*(Facebook|Twitter|WhatsApp)',
        r'Follow\s*us\s*on',
        r'Subscribe\s*to\s*our',
        r'All\s*rights\s*reserved',
        r'Copyright\s*©?\s*\d{4}',
        r'Read\s*more\s*:',
        r'Related\s*(Articles?|News|Stories)',
        r'Tags?\s*:',
        r'Category\s*:',
        r'Posted\s*(on|by)',
        r'Published\s*(on|by)',
        r'Last\s*updated',
        r'Advertisement',
        r'Sponsored',
        r'Breaking\s*News\s*:?',
    ]
    
    # Remove common Sinhala header/footer patterns
    sinhala_patterns = [
        r'මුල්\s*පිටුව',           # Home page
        r'පුවත්\s*ගෙදර',          # News home
        r'බෙදාගන්න',              # Share
        r'අදහස්\s*දක්වන්න',       # Comment
        r'වැඩිදුර\s*කියවන්න',     # Read more
        r'අදාළ\s*පුවත්',          # Related news
        r'ප්‍රකාශන\s*හිමිකම',      # Copyright
        r'සියලු\s*හිමිකම්',        # All rights
        r'නවතම\s*පුවත්',          # Latest news
        r'දැන්වීම',                # Advertisement
        r'අනුග්‍රහය',              # Sponsored
    ]
    
    # Apply all patterns
    for pattern in header_patterns + sinhala_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove lines that are too short (likely navigation)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Keep lines that have substantial content
        if len(line) > 20:
            cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    print("[preprocess] Cleaned text length:", len(text))
    
    return text


def clean_scraped_article(title: str, content: str) -> dict:
    """
    Clean a scraped news article.
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        Dictionary with cleaned title and content
    """
    # Clean title
    clean_title = normalize_text(title)
    clean_title = re.sub(r'\s+', ' ', clean_title)
    clean_title = clean_title[:200]  # Limit title length
    
    # Clean content
    clean_content = preprocess_for_indexing(content)
    
    return {
        "title": clean_title,
        "content": clean_content
    }


def is_valid_for_indexing(text: str, min_length: int = 50) -> bool:
    """
    Check if text is valid for indexing.
    
    Args:
        text: Text to check
        min_length: Minimum required length
        
    Returns:
        True if text is valid for indexing
    """
    if not text:
        return False
    
    if len(text) < min_length:
        print("[preprocess] Text too short:", len(text))
        return False
    
    # Check if text has enough Sinhala characters
    sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', text))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0:
        sinhala_ratio = sinhala_chars / total_chars
        if sinhala_ratio < 0.3:  # At least 30% Sinhala
            print("[preprocess] Not enough Sinhala content:", f"{sinhala_ratio:.1%}")
            return False
    
    return True
