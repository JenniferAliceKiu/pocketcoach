import logging
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from pocketcoach.api.schemas import ChatRequest, ChatResponse
from pocketcoach.api.chat_manager import (
    get_or_create_session,
    get_memory_for_session,
    append_to_history,
    get_history_for_session,
    delete_session,
)
from pocketcoach.llm_logic.llm_logic import init_models, pick_random_question, build_and_run_chain

app = FastAPI()

# CORS settings (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# System prompt; could be read from env/config
SYSTEM_PROMPT = "You are a helpful therapist assistant. Be empathetic and concise."

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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message is not allowed.")

    # 1. Get or create session (file-based)
    try:
        session_id, is_new = await run_in_threadpool(get_or_create_session, req.session_id)
        if is_new:
            logging.info(f"Created new session: {session_id}")
        else:
            logging.info(f"Using existing session: {session_id}")
    except Exception:
        logging.exception("Error in get_or_create_session; creating new session")
        session_id, _ = await run_in_threadpool(get_or_create_session, None)

    # 2. Reconstruct memory from JSON file
    try:
        memory = await run_in_threadpool(get_memory_for_session, session_id)
    except KeyError:
        # Unexpected: session file missing despite just created/validated; create fresh
        logging.exception(f"Session {session_id} not found when reconstructing memory; creating fresh session")
        session_id, _ = await run_in_threadpool(get_or_create_session, None)
        memory = await run_in_threadpool(get_memory_for_session, session_id)
    except Exception:
        logging.exception("Unexpected error in get_memory_for_session")
        raise HTTPException(status_code=500, detail="Internal server error loading session history.")

    # 3. Invoke LLM logic (blocking) in threadpool
    try:
        result = await run_in_threadpool(
            build_and_run_chain,
            user_text,
            memory,
            SYSTEM_PROMPT
        )
    except Exception:
        logging.exception("Error in build_and_run_chain")
        raise HTTPException(status_code=500, detail="Internal model error, please try again later.")

    # 4. Persist messages to file: user then assistant
    try:
        await run_in_threadpool(append_to_history, session_id, "user", user_text)
        await run_in_threadpool(append_to_history, session_id, "assistant", result["llm_response"])
    except KeyError:
        logging.exception(f"Session {session_id} disappeared when appending history")
    except Exception:
        logging.exception("Error appending to history")

    # 5. Return response including session_id
    return ChatResponse(
        session_id=session_id,
        sentiment=result["sentiment"],
        llm_response=result["llm_response"],
    )

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
