###chat_manager.py

import uuid
import threading
from typing import Dict
from langchain.memory import ConversationBufferMemory

_lock = threading.Lock()
session_memories: Dict[str, ConversationBufferMemory] = {}

def create_memory_for_session(system_prompt: str = None) -> (str, ConversationBufferMemory):
    session_id = str(uuid.uuid4())
    memory = ConversationBufferMemory(memory_key="history", return_messages=False)
    # Optionally: not storing system_prompt inside memory; handled in chain
    with _lock:
        session_memories[session_id] = memory
    return session_id, memory

def get_memory_for_session(session_id: str) -> ConversationBufferMemory:
    with _lock:
        memory = session_memories.get(session_id)
    if memory is None:
        raise KeyError(f"Session {session_id} not found")
    return memory

def delete_session(session_id: str):
    with _lock:
        session_memories.pop(session_id, None)
