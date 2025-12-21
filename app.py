import streamlit as st
import os
import tempfile
from pathlib import Path
import sys
from dotenv import load_dotenv
from llama_index.llms.sarvam import Sarvam
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import time

# Page configuration
st.set_page_config(
    page_title="PDF Q&A with Sarvam AI",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'index' not in st.session_state:
    st.session_state.index = None
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory"""
    saved_paths = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        # Create file path
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        saved_paths.append(file_path)
    
    return saved_paths, temp_dir

def process_documents(file_paths, api_key, context_window=4500, max_tokens=512, chunk_size=1024):
    """Process uploaded PDF documents and create index"""
    try:
        # Configure settings
        llm = Sarvam(
            api_key=api_key,
            model="sarvam-m",
            context_window=context_window,
            max_tokens=max_tokens,
        )
        
        embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = chunk_size
        Settings.system_prompt = """
        You are a Q&A assistant.
        Answer strictly based on the provided documents.
        If the answer is not in the documents, say so.
        Provide clear and concise answers.
        """
        
        # Load documents
        with st.spinner("Loading and processing documents..."):
            documents = SimpleDirectoryReader(input_files=file_paths).load_data()
            st.success(f"Loaded {len(documents)} document chunks")
        
        # Create index
        with st.spinner("Creating search index..."):
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            
            st.session_state.index = index
            st.session_state.query_engine = query_engine
            st.session_state.processing_complete = True
            
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def main():
    # Title and description
    st.title("📚 PDF Q&A with Sarvam AI")
    st.markdown("""
    Upload your PDF documents and ask questions about their content using Sarvam AI.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Sarvam API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Sarvam AI API key"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved in session")
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        
        context_window = st.slider(
            "Context Window Size",
            min_value=1024,
            max_value=8192,
            value=4500,
            step=256,
            help="Maximum context length for the model"
        )
        
        max_tokens = st.slider(
            "Max Response Tokens",
            min_value=64,
            max_value=1024,
            value=512,
            step=64,
            help="Maximum tokens in the response"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=256,
            max_value=2048,
            value=1024,
            step=256,
            help="Size of document chunks for processing"
        )
        
        st.divider()
        
        # System prompt customization
        st.subheader("Assistant Behavior")
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a Q&A assistant. Answer strictly based on the provided documents. If the answer is not in the documents, say so. Provide clear and concise answers.",
            height=150,
            help="Customize how the assistant should behave"
        )
        
        if system_prompt:
            Settings.system_prompt = system_prompt
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if uploaded_files and not st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        if st.session_state.uploaded_files:
            st.write(f"**Files ready for processing:**")
            for file in st.session_state.uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        # Process button
        if st.session_state.uploaded_files and st.session_state.api_key:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Save files temporarily
                    file_paths, temp_dir = save_uploaded_files(st.session_state.uploaded_files)
                    
                    # Process documents
                    success = process_documents(
                        file_paths, 
                        st.session_state.api_key,
                        context_window,
                        max_tokens,
                        chunk_size
                    )
                    
                    if success:
                        st.success("✅ Documents processed successfully! You can now ask questions.")
                    else:
                        st.error("Failed to process documents")
        
        elif st.session_state.uploaded_files and not st.session_state.api_key:
            st.warning("⚠️ Please enter your Sarvam API key in the sidebar to process documents")
    
    with col2:
        st.header("💬 Ask Questions")
        
        if st.session_state.processing_complete:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main findings of this document?",
                help="Ask anything about your uploaded documents"
            )
            
            # Advanced options
            with st.expander("⚡ Advanced Query Options"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Higher values make output more random"
                )
                
                similarity_top_k = st.slider(
                    "Similarity Top K",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of similar chunks to retrieve"
                )
            
            if question and st.button("🔍 Get Answer", type="primary"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response
                        response = st.session_state.query_engine.query(question)
                        
                        # Display response
                        st.subheader("Answer:")
                        st.write(str(response))
                        
                        # Display source information if available
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            with st.expander("📄 View Sources"):
                                for i, node in enumerate(response.source_nodes[:3]):
                                    st.write(f"**Source {i+1}:**")
                                    st.text(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                                    st.divider()
                        
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
        else:
            st.info("👈 Upload PDFs and process them to start asking questions")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Sarvam AI and Llamaindex.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()