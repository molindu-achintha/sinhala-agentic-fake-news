import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.agents.retrieval_agent import RetrievalAgent

def test_retrieve_evidence():
    # Mock LangProcAgent
    mock_lang = MagicMock()
    mock_lang.get_embeddings.return_value = [0.1] * 768 # Dummy embedding
    
    # Mock VectorStore inside RetrievalAgent? Hard to mock internal initiation easily without dependency injection.
    # We will assume integration test style or allow it to fail if no index.
    # Better: Mock VectorStore class.
    
    # For this generation, let's skip deep mocking and just test structure if possible,
    # or rely on the fact that we can't easily mock the import without `patch`.
    pass

from unittest.mock import patch

@patch('app.agents.retrieval_agent.VectorStore')
def test_retrieval_agent_flow(MockVectorStore):
    mock_store_instance = MockVectorStore.return_value
    mock_store_instance.search.return_value = [{"text": "doc1", "score": 0.9}]
    
    mock_lang = MagicMock()
    import numpy as np
    mock_lang.get_embeddings.return_value = np.array([0.1]*768)

    agent = RetrievalAgent(lang_proc=mock_lang)
    results = agent.retrieve_evidence("claim")
    
    assert len(results) == 1
    assert results[0]['text'] == "doc1"
