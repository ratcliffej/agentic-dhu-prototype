# import os
# from pathlib import Path
# import streamlit as st
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms.openai import OpenAI

# st.set_page_config(page_title="DHU 111 Call Handler Assistant", page_icon=":books:")

# # Configure your LLM once (outside cache)

# Settings.llm = OpenAI(
#     model="gpt-4.1-mini-2025-04-14",
#     temperature=0.5,
#     system_prompt=(
#         "You are an expert on managing healthcare call centre queries. "
#         "Assume that all questions are related to handling an inbound clinical call. "
#         "Use bullet points where appropriate. "
#         "Keep answers clear, concise and factual; avoid hallucinations; include citations."
#     ),
# )

# def _dir_signature(base: Path) -> tuple:
#     """Return a signature that changes when any file under `base` changes."""
#     return tuple(sorted(
#         (str(p), p.stat().st_mtime_ns)
#         for p in base.rglob("*") if p.is_file()
#     ))

# @st.cache_resource(show_spinner=False)
# def get_index(input_dir: str, signature: tuple):
#     """Build once; re-build only if `signature` changes."""
#     base = Path(input_dir).resolve()
#     reader = SimpleDirectoryReader(input_dir=str(base), recursive=True)

#     # Nice relative file list for display
#     relative_files = []
#     for p in reader.input_files:
#         p = Path(p).resolve()
#         try:
#             rel = p.relative_to(base)
#         except ValueError:
#             rel = Path(os.path.relpath(p, start=base))
#         relative_files.append(str(rel))

#     docs = reader.load_data()
#     index = VectorStoreIndex.from_documents(docs)
#     return index, relative_files

# INPUT_DIR = "./data"
# sig = _dir_signature(Path(INPUT_DIR).resolve())
# index, relative_files = get_index(INPUT_DIR, sig)

# st.markdown("**Sources:**\n" + "\n".join(f"- {rf}" for rf in relative_files))

# # Build the chat engine each run (cheap), or cache it too if you like
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True) # type: ignore

# # --- chat UI (unchanged) ---
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about DHU policies and procedures."}
#     ]

# if prompt := st.chat_input("Your question"):
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = chat_engine.chat(prompt) # type: ignore
#             st.write(response.response)
#             st.session_state.messages.append({"role": "assistant", "content": response.response})

# # Optional: a manual “Rebuild library” button
# if st.button("Rebuild library"):
#     get_index.clear()   # clears the cache so next run rebuilds
#     st.rerun()





import base64, streamlit as st

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader

from pathlib import Path


st.set_page_config(page_title="DHU 111 Call Handler Assistant", page_icon=":books:")
openai.api_key = st.secrets.openai_key
filenames_loaded = ""

# with st.sidebar:
#     page = st.selectbox("Navigate", ["Home","Analytics","Admin"])


logo_path = Path(__file__).resolve().parents[1] / "img" / "logo.jpg"
data = base64.b64encode(logo_path.read_bytes()).decode()
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{data}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.write("""
         # DHU 111 Call Handler Assistant
         """)


if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question DHU policies and procedures."}
    ]

st.cache_resource(show_spinner=False)

def load_data():
    Settings.llm = OpenAI(
        model="gpt-4.1-mini-2025-04-14",
        temperature=0.5,
        system_prompt=(
            "You are an expert on managing healthcare call centre queries. "
            "Assume that all questions are related to handling an inbound clinical call. Use bullet points where appropriate."
            "Keep answers clear, concise and factual; avoid hallucinations; include citations."
        ),
    )
    with st.spinner(text="Loading and indexing the document library – hang tight! This should take 1-2 minutes."):
        input_dir = "./data"
        base = Path("./data").resolve()
        reader = SimpleDirectoryReader(input_dir=str(base), recursive=True)

        # Get relative paths
        relative_files = []
        for p in reader.input_files:
            p = Path(p).resolve()
            try:
                rel = p.relative_to(base)
            except ValueError:
                # Fallback in case of symlinks or odd mounts
                rel = Path(os.path.relpath(p, start=base))
            relative_files.append(str(rel))

        st.markdown("**Sources:**\n" + "\n".join(f"- {rf}" for rf in relative_files))
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) 
    