import os
import json
import uuid
import threading
import logging
from datetime import datetime
from typing import Tuple, List, Dict
from pathlib import Path

from langchain.memory import ConversationBufferMemory

# Store sessions for long term ### Change to Database in the future
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# Global lock to protect file operations in multithreaded context
_lock = threading.Lock()

def _session_file_path(session_id: str) -> str:
    """
    Returns the full file path for a given session_id.
    """
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def get_or_create_session(session_id: str):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        with open(session_file, "w") as f:
            json.dump({"messages": []}, f, indent=2)
    return session_id, not session_file.exists()

def append_to_history(session_id: str, role: str, content: str):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        logging.error(f"Session file {session_file} does not exist when trying to append.")
        raise KeyError("Session file does not exist")
    with open(session_file, "r") as f:
        data = json.load(f)
    data.setdefault("messages", []).append({"role": role, "content": content})
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Appended message to {session_file}")

def get_history_for_session(session_id: str) -> List[Dict[str, str]]:
    """
    Return the structured history list for the session: a list of {"role":..., "content":..., "timestamp":...}.
    Raises KeyError if session not found.
    """
    path = _session_file_path(session_id)
    with _lock:
        if not os.path.isfile(path):
            raise KeyError(f"Session {session_id} not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Return a copy of messages list
    messages = data.get("messages", [])
    # We can omit timestamp if not needed by client; here we return full entries.
    return list(messages)

def get_memory_for_session(session_id: str) -> ConversationBufferMemory:
    """
    Reconstruct a new ConversationBufferMemory for this session by replaying messages from the JSON file.
    We iterate through the messages list and for each user->assistant pair, we call save_context.
    Raises KeyError if session not found.
    """
    path = _session_file_path(session_id)
    with _lock:
        if not os.path.isfile(path):
            raise KeyError(f"Session {session_id} not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    messages = data.get("messages", [])
    memory = ConversationBufferMemory(memory_key="history", return_messages=False)
    # Iterate messages in order, pairing user->assistant
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.get("role") == "user":
            user_text = msg.get("content", "")
            # Check next for assistant
            if i + 1 < n and messages[i+1].get("role") == "assistant":
                assistant_text = messages[i+1].get("content", "")
                memory.save_context({"user_text": user_text}, {"response": assistant_text})
                i += 2
            else:
                # No assistant reply yet; skip or stop
                i += 1
        else:
            # If assistant message without preceding user, skip
            i += 1
    return memory

def delete_session(session_id: str) -> None:
    """
    Delete the session file. Raises KeyError if session not found.
    """
    path = _session_file_path(session_id)
    with _lock:
        if not os.path.isfile(path):
            raise KeyError(f"Session {session_id} not found")
        os.remove(path)
