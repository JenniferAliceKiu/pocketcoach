import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import uuid
from pocketcoach.params import *
import os
import shutil
from datetime import datetime
from pocketcoach.whisper_function import transcribe_audio
import soundfile as sf
import io
from pathlib import Path

from api.schemas import ChatRequest, ChatResponse, LoginRequest
from api.chat_manager import (
    get_or_create_session,
    get_memory_for_session,
    append_to_history,
    log_to_bigquery,
    get_history_for_session,
    delete_session,
    get_system_prompt_with_question,
    get_user_sessions,
    save_user_sessions,
)
from pocketcoach.llm_logic.llm_logic import init_models, pick_random_question, build_and_run_chain
from pocketcoach.main import classify


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
        await run_in_threadpool(append_to_history, session_id_used, "user", user_text, sentiment)
        await run_in_threadpool(append_to_history, session_id_used, "assistant", llm_response)

        sessions = get_user_sessions()
        username = None
        for user, sid in sessions.items():
            if sid == session_id_used:
                username = user
                break

        log_to_bigquery(
            user_uuid=session_id_used,
            sentiment=sentiment.get("label") if sentiment else None,
            user_message=user_text,
            assistant_message=llm_response,
            sentiment_value=sentiment.get("score", 0.0) if sentiment else 0.0,
            user_name=username,
        )

    except KeyError:
        logging.exception(f"Session {session_id_used} disappeared when appending history")
    except Exception:
        logging.exception("Error appending to history")


    return {"session_id": session_id_used, "llm_response": llm_response}

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

@app.post("/classify")
async def map(req: ChatRequest):
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message is not allowed.")

    print(f"USER TEXT: {user_text}")
    emotion_classificaiton = classify(user_text)

    return {user_text: emotion_classificaiton}

@app.post("/transcribe-audio/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    if not audio_file.filename.endswith('.wav'):
        raise HTTPException(400, "Only WAV files allowed")
    audio_bytes = await audio_file.read()

    tmp_path = None
    try:
        # Try to read the audio bytes as a WAV file
        audio_buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_buffer)

        # Use the resampled data
        data_2 = data[:, :1].reshape(-1)

        # Save as a proper WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data_2, samplerate)
            tmp_path = tmp.name
        # Transcribe
        transcription = transcribe_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode or transcribe audio: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {"transcription": transcription}
