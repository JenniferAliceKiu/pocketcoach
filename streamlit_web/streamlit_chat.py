import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"
INITIAL_QUESTIONS = [
    "Hi there! I'm here to listen. How are you feeling today?",
    "Hello! What would you like to talk about today?",
    "Good day! What's on your mind right now?",
    "Hi! How can I support you today?"
]

st.set_page_config(page_title="Therapist Chat", page_icon="💬")

# --- 1. Initialize session_id from query params if available ---
if "session_id" not in st.session_state:
    params = st.experimental_get_query_params()
    sid = None
    if "session_id" in params and params["session_id"]:
        sid = params["session_id"][0]
    st.session_state.session_id = sid

# --- 2. Initialize message history ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    # Show a first UX prompt locally
    first_q = random.choice(INITIAL_QUESTIONS)
    st.session_state.messages.append({"role": "assistant", "content": first_q})
    st.session_state.initialized = True

st.title("Therapist Chat")

# Display the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input area
if user_input := st.chat_input("Your message..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2. Prepare payload
    payload = {"message": user_input}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    # 3. Send to backend
    with st.spinner("Therapist is thinking..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # 4. Update session_id (and persist in URL if new)
            new_sid = data.get("session_id", None)
            if new_sid and new_sid != st.session_state.session_id:
                st.session_state.session_id = new_sid
                st.experimental_set_query_params(session_id=new_sid)
            # 5. Extract assistant reply
            reply = data.get("llm_response", "")
            sentiment = data.get("sentiment")
        except Exception as e:
            st.error(f"Error: {e}")
            reply = "Sorry, I couldn't reach the server."
            sentiment = None

    # 6. Optionally display sentiment
    if sentiment:
        label = sentiment.get("label", "")
        score = sentiment.get("score", 0.0)
        st.chat_message("assistant").write(f"*Sentiment: {label} ({score:.1%})*")

    # 7. Display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
