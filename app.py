
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Page Configuration
st.set_page_config(
    page_title="MemoBot AI",
    page_icon="🤖",
    layout="wide"
)

# Custom UI
st.title("🤖 MemoBot AI")
st.subheader("Your AI That Remembers")
st.caption("Powered by Local LLM and LangChain Memory")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("Model: Llama 3.2")
    temperature = st.slider(
        "Response Creativity",
        0.0, 1.0, 0.7
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history.clear()
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# Local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=temperature
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are MemoBot, a friendly AI assistant. "
               "Use previous conversation to answer questions."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Conversation chain
chain = prompt | llm

# Memory function
def get_session_history(session_id):
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = (
            InMemoryChatMessageHistory()
        )
    return st.session_state.chat_history[session_id]

# LangChain memory
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

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
                            "session_id": "memobot_user"
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
                st.error(
                    "Unable to connect to Ollama. "
                    "Please check your local model."
                )
                st.caption(str(e))

# Footer
st.divider()
st.caption("MemoBot AI | Built with Streamlit + LangChain")
                  
