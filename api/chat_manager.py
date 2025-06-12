import os
import json
import uuid
import threading
from datetime import datetime
from typing import Tuple, List, Dict

from langchain.memory import ConversationBufferMemory

# Store sessions for long term ### Change to Database in the future
SESSIONS_DIR = os.path.join(os.getcwd(), "sessions")

# Ensure the directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Global lock to protect file operations in multithreaded context
_lock = threading.Lock()

def _session_file_path(session_id: str) -> str:
    """
    Returns the full file path for a given session_id.
    """
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def get_or_create_session(session_id: str = None) -> Tuple[str, bool]:
    """
    If session_id is given and a corresponding file exists, returns (session_id, False).
    Otherwise, creates a new session file with a new UUID and returns (new_session_id, True).
    """
    with _lock:
        if session_id:
            path = _session_file_path(session_id)
            if os.path.isfile(path):
                # existing session
                return session_id, False
        # create new session
        new_id = str(uuid.uuid4())
        path = _session_file_path(new_id)
        # Initial content: created_at and empty messages list
        data = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "messages": []
        }
        # Write to temp and rename preventing corupted/half files
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return new_id, True

def append_to_history(session_id: str, role: str, content: str) -> None:
    """
    Append a message to the session's JSON file.
    Raises KeyError if session not found.
    """
    path = _session_file_path(session_id)
    with _lock:
        if not os.path.isfile(path):
            raise KeyError(f"Session {session_id} not found")
        # Load existing data
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Append new message with timestamp
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        data.setdefault("messages", []).append(entry)
        # Write back (atomic), preventing partial writing to path
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

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
