"""
Database models definition using SQLModel.
"""
from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime

class RequestLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    claim_text: str
    verdict: str
    confidence: float
    explanation: str

class Source(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    url: str
    credibility_score: float
