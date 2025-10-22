import streamlit as st
import pickle
import os
from dotenv import load_dotenv

# Try importing with error handling
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Dependencies may be incompatible. Check requirements.txt")
    st.info("Required: pydantic==1.10.13, langchain-google-genai==0.0.11")
    st.stop()

load_dotenv()

# IMPORTANT: API key is stored in Streamlit Cloud Secrets
# Go to: App Settings > Secrets > Add GOOGLE_API_KEY
# This code will NOT expose your API key in the public repo
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets!")
    st.info("Please add your API key in Streamlit Cloud: Settings > Secrets")
    st.stop()

st.set_page_config(
    page_title="IBT Chatbot",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="auto",
)

# Sidebar contents
with st.sidebar:
    st.title("IBT assistant😊")
    st.markdown('''
        ## About
        This app was designed your fellow students to ease your revision process 
    ''')
    st.write('Made by romy')

# Custom CSS for dark theme
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
    .stTextArea > div > div > textarea {
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IBT Chatbot")

VECTOR_STORE_PATH = "ITB_notes_2025.pkl"

@st.cache_resource
def load_embeddings():
    """
    Load the EXACT same embedding model used during vector store creation.
    Model: sentence-transformers/all-MiniLM-L6-v2
    Device: CPU (as specified in main.py)
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    return embeddings

@st.cache_resource
def load_vector_store(path):
    """Load the pre-created vector store from pickle file"""
    if not os.path.exists(path):
        st.error(f"❌ Vector store file not found: {path}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Available files: {os.listdir('.')}")
        st.stop()
    
    file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
    
    try:
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        st.success(f"✅ Vector store loaded successfully! ({file_size:.2f} MB) - Everything is set, make a prompt below!")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.stop()

@st.cache_resource
def setup_qa_chain(_vector_store):
    """Set up the QA chain using modern LangChain approach"""
    llm = ChatGoogleGenerativeAI(
        temperature=0, 
        model="gemini-1.5-flash"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer:""")
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retriever
    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Load resources
try:
    embeddings = load_embeddings()
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    qa_chain = setup_qa_chain(vector_store)
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about System Analysis and Design:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Use the retrieval chain
            response = qa_chain.invoke({"input": prompt})
            full_response = response["answer"]
            
            if not full_response or full_response.strip() == "":
                full_response = "I couldn't find relevant information in the document. Please try rephrasing your question."
            
        except Exception as e:
            full_response = f"❌ An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
            st.error(f"Error type: {type(e).__name__}")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})