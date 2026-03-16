import streamlit as st
import pickle
import os
from dotenv import load_dotenv

import sys
import types

# --- PICKLE COMPATIBILITY STUB ---
# The vector stores were created with a transformers version that included 
# an internal module called 'transformers.core_model_loading'. 
# Since this module is missing in the current version, we stub it here 
# to allow pickle.load() to succeed.
if "transformers.core_model_loading" not in sys.modules:
    stub_module = types.ModuleType("transformers.core_model_loading")
    sys.modules["transformers.core_model_loading"] = stub_module
    # Add dummy classes found during inspection to satisfy unpickler
    stub_module.PyTorchModelHubMixin = type("PyTorchModelHubMixin", (), {})
# ---------------------------------

# Try importing with error handling
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    from streamlit_extras.add_vertical_space import add_vertical_space
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Dependencies may be incompatible. Check requirements.txt")
    st.info("Required: pydantic, langchain-google-genai, streamlit_extras")
    st.stop()

# Import subject registry
from subjects_config import SUBJECTS

load_dotenv()

# API key handling
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    pass # Already set via .env
else:
    st.error("⚠️ GOOGLE_API_KEY not found!")
    st.info("Please add your API key in Streamlit Cloud Secrets or local .env file.")
    st.stop()

st.set_page_config(
    page_title="Roma AI Study Assistant 🎓",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    body {
        color: #FAFAFA;
        background-color:#020203;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        text-align: right;
    }
    .stChatMessage.assistant {
        text-align: left;
    }
    /* Style for the subject badge */
    .subject-badge {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar contents
with st.sidebar:
    st.title("Bsta AI Assistant ✌️")
    st.markdown("### 📚 Choose a Course")
    
    # Subject Selection
    subject_names = list(SUBJECTS.keys())
    selected_subject_name = st.radio(
        "Select a course unit to study:",
        subject_names,
        index=0
    )
    
    selected_cfg = SUBJECTS[selected_subject_name]
    
    add_vertical_space()
    st.divider()
    st.markdown(f"**About {selected_subject_name}:**")
    st.caption(selected_cfg["description"])
    st.divider()
    st.write('Made by Khris Calvin')

# --- Session State Management for Subject Switching ---
if "active_subject" not in st.session_state:
    st.session_state.active_subject = selected_subject_name
    st.session_state.messages = []

# Detect subject change and reset history
if st.session_state.active_subject != selected_subject_name:
    st.session_state.active_subject = selected_subject_name
    st.session_state.messages = []
    st.rerun()

# --- Resource Loading Functions (Cached) ---

@st.cache_resource
def load_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

@st.cache_resource
def load_vector_store(path, subject_name):
    """Load the pre-created vector store from pickle file"""
    if not os.path.exists(path):
        return None, f"❌ Vector store for **{subject_name}** not found at `{path}`."
    
    try:
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        return vector_store, None
    except Exception as e:
        return None, f"❌ Error loading vector store for {subject_name}: {str(e)}"

@st.cache_resource
def setup_qa_chain(_vector_store, subject_prompt):
    """Set up the QA chain with a specific system prompt"""
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template(subject_prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    return create_retrieval_chain(retriever, document_chain)

# --- MAIN APP LOGIC ---

st.title("Bsta AI")
st.markdown(f"<div class='subject-badge'>{selected_cfg['icon']} {selected_subject_name}</div>", unsafe_allow_html=True)

# Load resources for the current subject
try:
    embeddings = load_embeddings()
    vector_store, error = load_vector_store(selected_cfg["pkl"], selected_subject_name)
    
    if error:
        st.error(error)
        st.warning(f"Please ensure `{selected_cfg['pkl']}` is present in the project directory.")
        st.stop()
        
    qa_chain = setup_qa_chain(vector_store, selected_cfg["prompt"])
    
    # One-time success message per subject switch
    if not st.session_state.messages:
        st.success(f"Hi there ✌️ I am Bsta AI designed to help you answer questions about **{selected_subject_name}**. Ask any question below.")

except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(f"Ask a question about {selected_subject_name}"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"input": prompt})
                full_response = response["answer"]
            
            if not full_response or full_response.strip() == "":
                full_response = "I couldn't find relevant information in the course notes. Please try rephrasing your question."
            
            message_placeholder.markdown(full_response)
            
            # Show sources in expander if available
            if "context" in response and response["context"]:
                with st.expander("📚 View source documents"):
                    for i, doc in enumerate(response["context"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()
                        
        except Exception as e:
            full_response = f"❌ An error occurred: {str(e)}\n\nPlease try again later."
            st.error(f"Error type: {type(e).__name__}")
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
