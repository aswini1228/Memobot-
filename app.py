
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Page settings
st.set_page_config(
    page_title="MemoBot AI",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 MemoBot")
st.subheader("Your AI That Remembers")
st.caption("Powered by Groq + LangChain Conversation Memory")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("Model: GPT-OSS 20B")

    temperature = st.slider(
        "Response Creativity",
        0.0, 1.0, 0.7
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = {}
        st.rerun()

# Initialize session data
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# Check API key
if "GROQ_API_KEY" not in st.secrets:
    st.error("Groq API key missing. Add it in Streamlit Secrets.")
    st.stop()

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=temperature,
    api_key=st.secrets["GROQ_API_KEY"]
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are MemoBot, a friendly AI assistant. "
        "Use previous conversation history to answer questions. "
        "Remember the user's details within this chat session."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Create chain
chain = prompt | llm

# Memory function
def get_session_history(session_id):
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.chat_history[session_id]

# Connect chain with memory
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

session_id = "memobot_user"

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask MemoBot anything...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("MemoBot is thinking..."):
            try:
                response = chatbot.invoke(
                    {"question": user_input},
                    config={
                        "configurable": {
                            "session_id": session_id
                        }
                    }
                )

                answer = response.content
                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error("Unable to generate response.")
                st.caption(str(e))

st.divider()
st.caption("MemoBot AI | Built with Streamlit and LangChain")
