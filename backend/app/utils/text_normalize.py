"""
Text normalization utility.
"""
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize unicode characters and remove invisible characters.
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    return text
