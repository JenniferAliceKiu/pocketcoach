###fast.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from pocketcoach.api.schemas import ChatRequest, ChatResponse
from pocketcoach.api.chat_manager import create_memory_for_session, get_memory_for_session, delete_session
from fastapi.middleware.cors import CORSMiddleware
from pocketcoach.llm_logic.llm_logic import main, build_and_run_chain, pick_random_question
import logging

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return_call = {
        'greeting': 'Hello'
        }

    return return_call


@app.get("/first_question")
def first_question():
    question = pick_random_question()
    return_call = {
        'greeting': f'{question}'
        }

    return return_call


SYSTEM_PROMPT = "You are a helpful therapist assistant. Be empathetic and concise."

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    try:
        if session_id is None:
            session_id, memory = create_memory_for_session(system_prompt=SYSTEM_PROMPT)
        else:
            memory = get_memory_for_session(session_id)
    except KeyError:
        session_id, memory = create_memory_for_session(system_prompt=SYSTEM_PROMPT)

    user_text = req.message
    try:
        result = build_and_run_chain(user_text, memory)
    except Exception as e:
        logging.exception("Error in build_and_run_chain")
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    return ChatResponse(
        session_id=session_id,
        sentiment=result["sentiment"],
        llm_response=result["llm_response"],
    )

"""
@app.post("/chat/{session_id}/reset")
async def reset_chat(session_id: str):
    delete_session(session_id)
    return {"detail": "Session reset. Start a new chat by POST /chat with no session_id."}
"""
