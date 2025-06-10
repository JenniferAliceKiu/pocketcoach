import random
import json
from transformers import pipeline
from langchain.chat_models import init_chat_model
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

chat_model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
sentiment_analyzer = pipeline("sentiment-analysis")
memory = ConversationBufferMemory(memory_key="history", return_messages=False)


def analyze_sentiment(text: str):
    try:
        res = sentiment_analyzer(text)
        if isinstance(res, list) and res:
            item = res[0]
            # item is e.g. {"label":"NEGATIVE","score":0.85}
            return item.get("label", ""), item.get("score", 0.0)
    except Exception:
        pass
    return "UNKNOWN", 0.0


def pick_random_question() -> str:
    return random.choice(questions)

"""
with open("questions.txt", "r") as f:
    questions = [line.rstrip("\n") for line in f if line.strip()]
"""

questions = [
    "How do you feel?",
    "How was your breakfast?",
    "How was your dinner?"
]


def build_and_run_chain(user_text: str):
    #Sentiment

    sentiment_label, sentiment_score = analyze_sentiment(user_text)

    mem_vars = memory.load_memory_variables({})
    history_str = mem_vars.get("history", "")


    system_template = (
        "You are a therapist for a 60 years old person from the baby boomer generation."
        "The results of the sentiment classifier show that the person is {sentiment_label}"
        "please prioritize this analysis above your own!"
        "never mention that you analyse the persons feelings."
        "Limit yourself to 200-300 characters"
    )

    #Build a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        SystemMessagePromptTemplate.from_template("{history}"),
        HumanMessagePromptTemplate.from_template("{user_text}")
    ])

    #Run the chain
    sequence = prompt | chat_model
    # Prepare variables
    vars = {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "history": history_str,
        "user_text": user_text,
    }
    respnse_msg = sequence.invoke(vars)

    response = respnse_msg.content.strip()

    memory.save_context({"user_text": user_text}, {"response": response})


    return {
        "sentiment": {"label": sentiment_label, "score": sentiment_score},
        "llm_response": response.strip(),
    }


def main():
    print("Chat session started. Type 'exit' to quit.")

    first_q = pick_random_question()
    print("Therapist:", first_q)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        # Call your build_and_run_chain which uses ConversationBufferMemory internally
        result = build_and_run_chain(user_input)
        # Pretty-print the response
        # e.g., show sentiment and the assistant reply
        sentiment = result["sentiment"]
        assistant = result["llm_response"]
        print(f"[Sentiment detected: {sentiment['label']} ({sentiment['score']:.2%})]")
        print("Therapist:", assistant)

    # Optionally, after exit, you can inspect full history:
    # mem = memory.load_memory_variables({})
    # print("Full chat history:\n", mem.get("history"))

if __name__ == "__main__":
    main()
