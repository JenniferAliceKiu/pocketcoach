### schemas.py

from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    sentiment: dict   # {"label": ..., "score": ...}
    llm_response: str
