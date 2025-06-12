import random
import os
from transformers import pipeline
from langchain.chat_models import init_chat_model
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from pocketcoach.dl_logic.model import load_model

# Global models to prevent reloding for new sessions
sentiment_analyzer = None
chat_model = None

# Prompt template
SYSTEM_TEMPLATE = (
    "{system_prompt} The results of the sentiment classifier show that the person is {sentiment_label}. "
    "Please prioritize this analysis above your own! Never mention that you analyse the person's feelings. "
    "Limit yourself to 200-300 characters."
)
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    SystemMessagePromptTemplate.from_template("{history}"),
    HumanMessagePromptTemplate.from_template("{user_text}")
])


def init_models():
    """
    Should be called once at startup to load heavy models/pipelines.
    """
    global sentiment_analyzer, chat_model

    sentiment_analyzer = load_model()
    chat_model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

def analyze_sentiment(text: str):
    """
    Returns (label: str, score: float). On error, returns ("UNKNOWN", 0.0).
    """
    try:
        classifications = sentiment_analyzer(text)
        print(f'Result of the classification is: {classifications}')
        if isinstance(classifications, list) and classifications:
            top_class = max(classifications, key=lambda x: x['score'])
            return top_class.get("label", ""), top_class.get("score", 0.0)
    except Exception as e:
        print(f"[Sentiment] error: {e}")
    return "UNKNOWN", 0.0

_QUESTIONS_CACHE = None

def load_questions():
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is None:
        questions_path = os.path.join(os.path.dirname(__file__), "questions.txt")
        with open(questions_path, "r", encoding="utf-8") as f:
            _QUESTIONS_CACHE = [line.strip() for line in f if line.strip()]
    return _QUESTIONS_CACHE

def pick_random_question() -> str:
    questions = load_questions()
    return random.choice(questions)

def build_and_run_chain(
    user_text: str,
    memory: ConversationBufferMemory,
    system_prompt: str = "You are a helpful therapist assistant. Be empathetic and concise."
):

    # Sentiment analysis
    sentiment_label, sentiment_score = analyze_sentiment(user_text)

    # Load history and truncate
    mem_vars = memory.load_memory_variables({})
    history_str = mem_vars.get("history", "")
    # Truncate to last 2000 characters
    if len(history_str) > 2000:
        history_str = history_str[-2000:]

    # Prompt variables
    prompt_vars = {
        "system_prompt": system_prompt,
        "sentiment_label": sentiment_label,
        "history": history_str,
        "user_text": user_text,
    }

    # Build sequence and invoke
    sequence = PROMPT_TEMPLATE | chat_model
    resp = sequence.invoke(prompt_vars)
    response = resp.content.strip()

    # Saaaaave
    memory.save_context({"user_text": user_text}, {"response": response})

    return {
        "sentiment": {"label": sentiment_label, "score": sentiment_score},
        "llm_response": response,
    }
