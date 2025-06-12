from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    sentiment: Dict[str, Any]
    llm_response: str
