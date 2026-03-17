import streamlit as st
import os
import sys
import shutil
from pathlib import Path

# Add the parent directory to sys.path so we can import multimodal_rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multimodal_rag import ingest_documents, run_reflective_query, setup_query_engine, DATA_DIR, CHROMA_PATH

st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="wide")

st.title("⚙️ Admin Panel - Upload SOPs")
st.markdown("Upload Standard Operating Procedures to process and store them in the database.")

# --- Session State ---
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom CSS ---
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #28a745;
        color: white;
        border-color: #28a745;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838;
        color: white;
        border-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_engine():
    if os.path.exists(CHROMA_PATH):
        with st.spinner("Loading Multimodal Query Engine..."):
            engine = setup_query_engine()
            st.session_state.query_engine = engine
            st.success("Query Engine ready!")

st.header("Upload SOP Documents")
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents", type="primary"):
    if uploaded_files:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Enforce exclusive upload: Clear data directory before saving new files
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                try:
                    os.remove(os.path.join(DATA_DIR, f))
                except Exception:
                    pass
        
        # Save uploaded files
        for uf in uploaded_files:
            file_path = os.path.join(DATA_DIR, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
                
        with st.spinner("Extracting text and indexing... (This can take a few minutes)"):
            try:
                # Ingest and get the index
                ingest_documents(DATA_DIR)
                initialize_engine()
                
                st.success("Indexing complete! Query Engine ready. Users can now query these documents.")
                st.session_state.messages = [] # Reset chat when new documents are uploaded
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
    else:
        st.warning("Please upload files first.")
        
st.markdown("---")
st.header("Database Management")
st.warning("Warning: Clearing the database will remove all uploaded documents from the AI's knowledge.")
if st.button("Clear Database"):
    # Explicitly clear engine to release file locks on Windows
    st.session_state.query_engine = None
    st.session_state.messages = []
    
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            st.error(f"Error clearing database: {e}. Try closing any other apps using these files.")
    
    # Also clean up extracted images from data directory
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            try:
                os.remove(os.path.join(DATA_DIR, file))
            except Exception:
                pass

    st.success("Database and PDF files cleared!")
    st.rerun()

# Display currently available files
if os.path.exists(DATA_DIR):
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if all_files:
        st.subheader("Currently Indexed Files:")
        for f in all_files:
            st.write(f"- {f}")
    else:
        st.info("No files currently indexed.")
else:
    st.info("No files currently indexed.")
