import os
import json
import threading
import logging
from typing import List, Dict
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from pocketcoach.llm_logic.llm_logic import pick_random_question
from pocketcoach.params import *
from google.cloud import bigquery
from datetime import datetime
import os

# Store sessions for long term ### Change to Database in the future
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
USER_SESSION_FILE = Path("sessions/user_sessions.json")

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

def append_to_history(session_id: str, role: str, content: str, sentiment=None):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        logging.error(f"Session file {session_file} does not exist when trying to append.")
        raise KeyError("Session file does not exist")
    with open(session_file, "r") as f:
        data = json.load(f)
    message = {"role": role, "content": content}
    if sentiment is not None:
        message["sentiment"] = sentiment
    data.setdefault("messages", []).append(message)
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


def get_system_prompt_with_question(username: str = None):
    question = pick_random_question()
    base_prompt = SYSTEM_PROMPT
    if username:
        base_prompt += f" The user's name is {username}."
    return (
        base_prompt + f" Start the conversation by asking the user: \"{question}\""
    )

def get_user_sessions():
    if USER_SESSION_FILE.exists():
        with open(USER_SESSION_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_sessions(sessions):
    with open(USER_SESSION_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def log_to_bigquery(user_uuid, sentiment, user_message, assistant_message, sentiment_value, user_name,  timestamp=None):
    print("initializing client")
    client = bigquery.Client(project="lewagon-bootcamp-457509")
    table_id = "lewagon-bootcamp-457509.pocketcoachbq.user_sentiment"
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    print("ready to write to BigQuery")
    row = {
        "user_uuid": user_uuid,
        "user_name": user_name,
        "timestamp": timestamp,
        "sentiment": sentiment,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "sentiment_value": sentiment_value,
    }
    errors = client.insert_rows_json(table_id, [row])
    print("apparently wrote to bq")
    if errors:
        print("BigQuery insert errors:", errors)
