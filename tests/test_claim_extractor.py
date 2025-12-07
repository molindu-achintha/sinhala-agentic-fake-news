import sys
import os
import pytest

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.agents.claim_extractor import ClaimExtractorAgent

def test_extract_claim_simple():
    extractor = ClaimExtractorAgent()
    text = "මෙම පුවතේ දැන්වීම අනතුරක් බවට පත් වෙලා තිබේ. රජය මේ ගැන නිවේදනය කරලා නැහැ."
    result = extractor.extract_claim(text)
    
    assert "claim_text" in result
    assert result["confidence"] > 0.0
    # Our dummy logic just returns the first sentence
    assert result["claim_text"].startswith("මෙම පුවතේ")

def test_extract_claim_empty():
    extractor = ClaimExtractorAgent()
    result = extractor.extract_claim("")
    assert result["claim_text"] == ""
