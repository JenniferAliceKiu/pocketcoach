import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tensorflow import keras
import tempfile
import librosa
import numpy as np
import json
import uuid
from fastapi import Body
from pocketcoach.params import *
import os
import shutil
from datetime import datetime
from pocketcoach.whisper_function import transcribe_audio


from api.schemas import ChatRequest, ChatResponse, LoginRequest
from api.chat_manager import (
    get_or_create_session,
    get_memory_for_session,
    append_to_history,
    get_history_for_session,
    delete_session,
)
from pocketcoach.llm_logic.llm_logic import init_models, pick_random_question, build_and_run_chain
from transformers import pipeline #

from pocketcoach.main import classify

import json
from pathlib import Path

USER_SESSION_FILE = Path("sessions/user_sessions.json")

def get_user_sessions():
    if USER_SESSION_FILE.exists():
        with open(USER_SESSION_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_sessions(sessions):
    with open(USER_SESSION_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

app = FastAPI()

# CORS settings (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw_data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


whisper_pipe = None

@app.post("/login")
async def login(req: LoginRequest):
    username = req.username
    sessions = get_user_sessions()
    if username in sessions:
        session_id = sessions[username]
        logging.info(f"Existing user: {username}, session_id: {session_id}")
    else:
        session_id = str(uuid.uuid4())
        sessions[username] = session_id
        save_user_sessions(sessions)
        logging.info(f"New user: {username}, session_id: {session_id}")

    # --- Always ensure session file exists and has initial question ---
    from api.chat_manager import append_to_history, get_or_create_session
    session_file = Path("sessions") / f"{session_id}.json"
    if not session_file.exists():
        await run_in_threadpool(get_or_create_session, session_id)
        logging.info(f"Session file created for {session_id}")
        question = pick_random_question()
        await run_in_threadpool(append_to_history, session_id, "assistant", question)
        logging.info(f"Initial question written for {session_id}")

    return {"session_id": session_id}

@app.on_event("startup")
def on_startup():
    logging.info("Initializing models/pipelines...")
    init_models()
    logging.info("Models initialized.")

@app.get("/")
def root():
    return {"greeting": "Hello"}

@app.get("/first_question")
async def first_question():
    question = pick_random_question()
    return {"greeting": question}


async def process_user_message(user_text: str, session_id: str = None) -> dict:
    """
    Process a user message (text), manage session, memory, LLM call, and persisting history.
    Returns a dict with keys: session_id, sentiment, llm_response.
    """
    # 1. Get or create session
    try:
        sid, is_new = await run_in_threadpool(get_or_create_session, session_id)
        session_id_used = sid
    except Exception:
        logging.exception("Error in get_or_create_session; creating new session")
        sid, _ = await run_in_threadpool(get_or_create_session, None)
        session_id_used = sid

    # 2. Reconstruct memory
    try:
        memory = await run_in_threadpool(get_memory_for_session, session_id_used)
        print("Reconstructin memory")
        memory = await run_in_threadpool(get_memory_for_session, session_id)
    except KeyError:
        logging.exception(f"Session {session_id_used} not found; creating fresh session")
        sid, _ = await run_in_threadpool(get_or_create_session, None)
        session_id_used = sid
        memory = await run_in_threadpool(get_memory_for_session, session_id_used)
    except Exception:
        logging.exception("Unexpected error in get_memory_for_session")
        raise HTTPException(status_code=500, detail="Internal server error loading session history.")

    # 3. Call LLM logic in threadpool
    try:
        if is_new:
            system_prompt = get_system_prompt_with_question()
        else:
            system_prompt = SYSTEM_PROMPT
        result = await run_in_threadpool(build_and_run_chain, user_text, memory, system_prompt)
    except Exception:
        logging.exception("Error in build_and_run_chain")
        raise HTTPException(status_code=500, detail="Internal model error, please try again later.")

    llm_response = result.get("llm_response", "")
    sentiment = result.get("sentiment")

    # 4. Persist messages
    try:
        await run_in_threadpool(append_to_history, session_id_used, "user", user_text)
        await run_in_threadpool(append_to_history, session_id_used, "assistant", llm_response)
    except KeyError:
        logging.exception(f"Session {session_id_used} disappeared when appending history")
    except Exception:
        logging.exception("Error appending to history")


    return {"session_id": session_id_used, "sentiment": sentiment, "llm_response": llm_response}

from api.schemas import ChatRequest, ChatResponse

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_text = req.message.strip()
    # Allow empty message only if this is a new session (no history)
    if not user_text:
        # Check if session exists and has history
        try:
            history = await run_in_threadpool(get_history_for_session, req.session_id)
            if history:  # If history is not empty, reject
                raise HTTPException(status_code=400, detail="Empty message is not allowed.")
        except KeyError:
            pass  # No history, allow
    out = await process_user_message(user_text, req.session_id)
    return ChatResponse(**out)

@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    Return the structured chat history for a given session_id, as read from the JSON file.
    """
    try:
        history = await run_in_threadpool(get_history_for_session, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception:
        logging.exception("Error retrieving history")
        raise HTTPException(status_code=500, detail="Internal server error retrieving history")
    return {"session_id": session_id, "history": history}

@app.post("/chat/{session_id}/reset")
async def reset_chat(session_id: str):
    """
    Delete the session file. Subsequent POST /chat without session_id creates a new session.
    """
    try:
        await run_in_threadpool(delete_session, session_id)
        logging.info(f"Session {session_id} reset by user request.")
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception:
        logging.exception("Error deleting session")
        raise HTTPException(status_code=500, detail="Internal server error deleting session")
    return {"detail": "Session reset. Start a new chat by POST /chat with no session_id."}


def get_system_prompt_with_question(username: str = None):
    question = pick_random_question()
    base_prompt = SYSTEM_PROMPT
    if username:
        base_prompt += f" The user's name is {username}."
    return (
        base_prompt + f" Start the conversation by asking the user: \"{question}\""
    )

#added frm other API file (Jen, from Cursor)
@app.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Endpoint to handle audio file uploads and transcription.
    """

    try:
        breakpoint()
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"audio_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio
        transcription, saved_file = transcribe_audio(file_path, "online")

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "status": "success",
            "transcription": transcription,
            "saved_file": saved_file
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/classify")
async def map(req: ChatRequest):
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message is not allowed.")

    print(f"USER TEXT: {user_text}")
    emotion_classificaiton = classify(user_text)

    return {user_text: emotion_classificaiton}

@app.post("/upload-audio/")
async def upload_audio(audio_file: UploadFile = File(...)):
    # Validate WAV file
    if not audio_file.filename.endswith('.wav'):
        raise HTTPException(400, "Only WAV files allowed")

    # Save file
    file_path = f"raw_data/{audio_file.filename}"
    with open(file_path, "wb") as f:
        content = await audio_file.read()
        f.write(content)

    return {"message": "File uploaded successfully", "filename": audio_file.filename}


@app.post("/transcribe-audio/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    if not audio_file.filename.endswith('.wav'):
        raise HTTPException(400, "Only WAV files allowed")
    audio_bytes = await audio_file.read()
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        transcription = transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)
    return {"transcription": transcription}
